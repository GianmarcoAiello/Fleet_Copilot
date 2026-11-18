"""Fleet Copilot web UI — stdlib HTTP server, no external deps.

Sections: Chat (query -> streaming tool trace + answer card), Audit log,
Pending actions (approve/reject). Serves on http://localhost:8080.
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from src.db.loader import Repository  # noqa: E402
from src.graph.nodes import build_workflow  # noqa: E402

_repo = None
_wf = None  # built lazily on first API call (DuckDB is single-process)


def _get_repo(read_only: bool = True):
    """Lazy repository handle. DuckDB allows one process per file: while a CLI
    session is open, the web UI reports a clear error instead of crashing."""
    global _repo
    if _repo is None:
        path = ROOT / "data" / "fleet.duckdb"
        _repo = (Repository._open_readonly(path) if read_only
                 else Repository(path))
    return _repo


def _get_wf():
    global _wf
    if _wf is None:
        _wf = build_workflow(_get_repo(read_only=False), checkpointer=MemorySaver())
    return _wf


_lock = threading.Lock()  # one graph run at a time


# ── API ─────────────────────────────────────────────────────────────────────

def api_query(body: dict) -> dict:
    tenant = body.get("tenant", "acme-001")
    query = body.get("query", "")
    with _lock:
        state = _get_wf().invoke(
            {"query": query, "tenant_id": tenant},
            config={"configurable": {"thread_id": f"web-{body.get('n', 0)}",
                                     "tenant_id": tenant}},
        )
    # one entry per analyst turn: tool calls it issued + its draft text
    trace, tools = [], []
    from langchain_core.messages import AIMessage
    for m in state.get("messages", []):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for c in m.tool_calls:
                trace.append({"step": "call", "tool": c["name"], "args": c["args"]})
        elif isinstance(m, AIMessage):
            txt = m.content if isinstance(m.content, str) else " ".join(
                b.get("text", "") for b in m.content if isinstance(b, dict))
            if txt.strip():
                trace.append({"step": "thought", "text": txt[:600]})
    for t in state.get("tool_outputs", []):
        trace.append({"step": "result", "tool": t["tool"],
                      "result": str(t["result"])[:400]})
    return {
        "response": state.get("response") or "(no response produced)",
        "trace": trace,
        "tools_called": sorted({t["tool"] for t in state.get("tool_outputs", [])}),
        "intent": state["plan"].intent if state.get("plan") else "?",
        "interrupted": "__interrupt__" in state,
    }


def api_audit(body: dict) -> dict:
    rows = _get_repo().audit_tail(body.get("tenant", "acme-001"), limit=body.get("limit", 50))
    return {"rows": rows}


def api_pending(body: dict) -> dict:
    tenant = body.get("tenant", "acme-001")
    rows = _get_repo().con.execute(
        "SELECT action_id, tool, device_id, args_json, created_at FROM actions "
        "WHERE company_id=? AND status='proposed' ORDER BY created_at DESC",
        (tenant,)).fetchall()
    return {"rows": [{"action_id": r[0], "tool": r[1], "device_id": r[2],
                      "args": json.loads(r[3]), "created_at": str(r[4])} for r in rows]}


def api_resolve(body: dict) -> dict:
    ok = _get_repo(read_only=False).resolve_action(body.get("tenant", "acme-001"), body["action_id"],
                              approved=bool(body.get("approved")),
                              approver=body.get("approver", "web-user"))
    return {"ok": ok}


# ── HTTP layer ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, ctype: str, payload: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, obj, code: int = 200):
        self._send(code, "application/json", json.dumps(obj, default=str).encode())

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", INDEX.encode())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or "{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)
        routes = {"/api/query": api_query, "/api/audit": api_audit,
                  "/api/pending": api_pending, "/api/resolve": api_resolve}
        fn = routes.get(self.path)
        if not fn:
            return self._json({"error": "not found"}, 404)
        try:
            self._json(fn(body))
        except Exception as exc:  # noqa: BLE001 — surface errors to the UI
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def log_message(self, *a):  # quiet
        pass


INDEX = """<!doctype html><html><head><meta charset="utf-8">
<title>Fleet Copilot</title>
<style>
 body{font-family:-apple-system,system-ui,sans-serif;margin:0;background:#0f1420;color:#e2e8f0}
 nav{display:flex;gap:0;background:#1a2234;border-bottom:1px solid #2a3550}
 nav button{flex:1;padding:14px;background:none;border:none;color:#8fa2c0;font-size:14px;cursor:pointer}
 nav button.active{color:#fff;border-bottom:2px solid #5b8cff}
 main{max-width:860px;margin:0 auto;padding:20px}
 .card{background:#1a2234;border:1px solid #2a3550;border-radius:12px;padding:16px;margin:12px 0}
 .row{display:flex;gap:8px;margin-top:8px}
 input[type=text]{flex:1;padding:10px;border-radius:8px;border:1px solid #2a3550;background:#0f1420;color:#fff}
 select,input[type=text]{font-size:14px}
 button.primary{background:#5b8cff;color:#fff;border:none;border-radius:8px;padding:10px 18px;cursor:pointer}
 .tool{font-family:ui-monospace,monospace;font-size:12px;padding:6px 10px;border-radius:6px;
       background:#101827;margin:4px 0;border-left:3px solid #5b8cff}
 .tool.result{border-left-color:#3ddc97;color:#9fb8d8}
 .thought{font-size:13px;color:#b9c8e0;margin:4px 0}
 .answer{line-height:1.55;white-space:pre-wrap}
 .evidence{color:#7ee0b0;font-size:12px}
 .badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;background:#2a3550;margin-right:6px}
 .log-row{font-family:ui-monospace,monospace;font-size:12px;padding:8px;border-bottom:1px solid #232d45}
 .pending{display:flex;justify-content:space-between;align-items:center;gap:8px}
 .pending button{padding:8px 14px;border-radius:8px;border:none;cursor:pointer}
 .approve{background:#3ddc97;color:#06281a}.reject{background:#ff6b7a;color:#3a0b10}
 h3{margin:8px 0 4px;color:#8fa2c0;font-size:13px;text-transform:uppercase;letter-spacing:1px}
</style></head><body>
<nav id="tabs">
 <button data-tab="chat" class="active">Copilot</button>
 <button data-tab="audit">Audit log</button>
 <button data-tab="pending">Pending actions <span id="pending-count"></span></button>
</nav>
<main id="chat">
  <div class="card">
    <div class="row">
      <select id="tenant"><option>acme-001</option><option>globex-002</option><option>initech-003</option></select>
      <input type="text" id="q" placeholder="Ask about your fleet…" onkeydown="if(event.key==='Enter')ask()">
      <button class="primary" onclick="ask()">Send</button>
    </div>
  </div>
  <div id="chat-history"></div>
</main>
<main id="audit" style="display:none"><h3>Audit log (latest 50)</h3><div class="card" id="audit-rows"></div></main>
<main id="pending" style="display:none"><h3>Actions awaiting decision</h3><div id="pending-rows"></div></main>
<script>
const $=id=>document.getElementById(id);
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('nav button').forEach(x=>x.classList.remove('active'));b.classList.add('active');
  for(const t of['chat','audit','pending'])$(''+t).style.display=t===b.dataset.tab?'':'none';
  if(b.dataset.tab==='audit')loadAudit(); if(b.dataset.tab==='pending')loadPending();
});
async function post(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});return r.json();}
function toolRow(t){
  if(t.step==='call')return `<div class="tool">→ ${t.tool}(${JSON.stringify(t.args).slice(0,120)})</div>`;
  if(t.step==='result')return `<div class="tool result">⎋ ${t.tool}: ${t.result.slice(0,160)}</div>`;
  return `<div class="thought">${t.text.slice(0,200)}…</div>`;
}
async function ask(){
  const q=$('q').value.trim(); if(!q)return;
  const hist=$('chat-history');
  hist.insertAdjacentHTML('afterbegin',`<div class="card"><b>${q}</b><div id="running" class="thought">working…</div></div>`);
  $('q').value='';
  const res=await post('/api/query',{query:q,tenant:$('tenant').value,n:Date.now()});
  hist.querySelector('#running')?.remove();
  const tools=res.trace?.map(toolRow).join('')||'';
  hist.insertAdjacentHTML('afterbegin',`<div class="card">
    <span class="badge">${res.intent||'?'}</span><span class="badge">${(res.tools_called||[]).join(', ')||'no tools'}</span>
    ${res.error?`<div class="answer" style="color:#ff6b7a">${res.error}</div>`:`
    <div class="answer">${res.response.replace(/\\n/g,'<br>')}</div>
    <h3>Tool trace</h3>${tools}`}
  </div>`);
}
async function loadAudit(){
  const res=await post('/api/audit',{tenant:$('tenant').value,limit:50});
  $('audit-rows').innerHTML=res.rows?.map(r=>`<div class="log-row"><span class="badge">${r.event}</span>${r.tool||''} <span style="color:#7a8db0">${r.detail||''}</span> <span style="color:#55648a">${(r.ts||'').slice(0,19)}</span></div>`).join('')||'empty';
}
async function loadPending(){
  const res=await post('/api/pending',{tenant:$('tenant').value});
  $('pending-count').textContent=res.rows?.length?`(${res.rows.length})`:'';
  $('pending-rows').innerHTML=res.rows?.map(r=>`<div class="card pending">
    <div><b>${r.tool}</b> <span class="badge">${r.device_id}</span>
      <div style="font-family:monospace;font-size:12px;color:#9fb8d8">${JSON.stringify(r.args).slice(0,100)}</div>
      <div style="font-size:11px;color:#55648a">${r.action_id} · ${r.created_at.slice(0,19)}</div></div>
    <div><button class="approve" onclick="resolve('${r.action_id}',true)">Approve</button>
         <button class="reject" onclick="resolve('${r.action_id}',false)">Reject</button></div>
  </div>`).join('')||'<div class="card">No pending actions 🎉</div>';
}
async function resolve(aid,approved){
  await post('/api/resolve',{action_id:aid,approved,tenant:$('tenant').value,approver:'web-user'});
  loadPending();
}
</script></body></html>"""


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Fleet Copilot UI → http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
