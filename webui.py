"""Fleet Copilot web UI — stdlib HTTP server, Rayda-inspired design system.

Sections: Copilot (query -> live agent-pipeline + tool trace + answer card),
Audit log, Pending actions (approve/reject). Single shared DuckDB handle
(DuckDB is single-process: close the CLI before using the UI).
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from src.db.loader import Repository  # noqa: E402
from src.graph.nodes import build_workflow  # noqa: E402

_repo = None
_repo_lock = threading.Lock()
_run_lock = threading.Lock()   # one graph run at a time


def _get_repo() -> Repository:
    """Single shared read-WRITE handle. DuckDB is single-process per file:
    if a CLI session holds the lock, APIs surface a clear error (the web UI
    shows it inline) instead of mixing read-only and read-write handles."""
    global _repo
    with _repo_lock:
        if _repo is None:
            _repo = Repository(ROOT / "data" / "fleet.duckdb")
        return _repo


def _get_wf():
    return build_workflow(_get_repo(), checkpointer=MemorySaver())


NODES = [("planner", "Planner"), ("analyst", "Analyst"), ("verifier", "Verifier"),
         ("hitl", "HITL gate"), ("formatter", "Formatter")]


def stream_query(body: dict):
    """Generator yielding SSE events while the graph runs."""
    tenant = body.get("tenant", "acme-001")
    query = body.get("query", "")

    def ev(etype: str, **kw) -> str:
        return "data: " + json.dumps({"type": etype, **kw}, default=str) + "\n\n"

    yield ev("start", tenant=tenant, nodes=[n for n, _ in NODES])
    with _run_lock:
        wf = _get_wf()
        thread = {"configurable": {"thread_id": f"web-{body.get('n', 0)}",
                                   "tenant_id": tenant}}
        final = {}
        try:
            for chunk in wf.stream(
                    {"query": query, "tenant_id": tenant},
                    config=thread, stream_mode="updates"):
                for node, delta in chunk.items():
                    if node == "__interrupt__":
                        yield ev("node", node="hitl", status="active",
                                 detail="awaiting human approval")
                        continue
                    yield ev("node", node=node, status="active")
                    if node == "execute_tools" and delta:
                        for msg in (delta.get("messages") or []):
                            if isinstance(msg, ToolMessage):
                                yield ev("tool_result", result=str(msg.content)[:280])
                    if node == "analyst" and delta:
                        for msg in (delta.get("messages") or []):
                            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                                for c in msg.tool_calls:
                                    yield ev("tool_call", tool=c["name"],
                                             args=json.dumps(c["args"], default=str)[:220])
                    final.update(delta or {})
                    yield ev("node", node=node, status="done")
            # HITL pause: graph returns with __interrupt__ in state, resume here
            state = wf.get_state(thread)
            if state.next:  # interrupted mid-graph
                payload = {}
                for task in state.tasks:
                    interrupts = getattr(task, "interrupts", ()) or ()
                    if interrupts:
                        payload = interrupts[0].value or {}
                        break
                yield ev("interrupt", payload=json.loads(json.dumps(payload, default=str)))
                return
            yield ev("done", response=final.get("response") or "(no response produced)",
                     intent=final.get("plan", {}).intent if final.get("plan") else "?",
                     tools=sorted({t["tool"] for t in final.get("tool_outputs", [])}))
        except Exception as exc:  # noqa: BLE001
            yield ev("error", message=f"{type(exc).__name__}: {exc}")


# ── JSON APIs ───────────────────────────────────────────────────────────────

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
    ok = _get_repo().resolve_action(body.get("tenant", "acme-001"), body["action_id"],
                                    approved=bool(body.get("approved")),
                                    approver=body.get("approver", "web-user"))
    return {"ok": ok}


# ── HTTP layer ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, ctype: str, payload: bytes, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
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
        if self.path == "/api/query_stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for chunk in stream_query(body):
                    self.wfile.write(chunk.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        routes = {"/api/audit": api_audit, "/api/pending": api_pending,
                  "/api/resolve": api_resolve}
        fn = routes.get(self.path)
        if not fn:
            return self._json({"error": "not found"}, 404)
        try:
            self._json(fn(body))
        except Exception as exc:  # noqa: BLE001
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def log_message(self, *a, **k):  # quiet
        pass


INDEX = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fleet Copilot</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
 --bg:#021027; --surface:#0B1328; --surface2:#101828; --border:#1D2939;
 --border2:#24344d; --text:#EAECF0; --muted:#98A2B3; --faint:#667085;
 --blue:#0C66FF; --blue-soft:#D1E0FF; --green:#17B26A; --green-bg:#083227;
 --red:#F97066; --red-bg:#3a1114; --amber:#FDB022;
}
*{box-sizing:border-box}
body{font-family:Inter,-apple-system,sans-serif;margin:0;background:
 radial-gradient(1200px 600px at 80% -10%, #0a1f4d 0%, transparent 60%), var(--bg);
 color:var(--text);min-height:100vh}
header{display:flex;align-items:center;gap:14px;padding:12px 24px;border-bottom:1px solid var(--border);
 background:rgba(2,16,39,.7);backdrop-filter:blur(8px);position:sticky;top:0;z-index:5;flex-wrap:wrap}
.logo{display:flex;align-items:center;gap:10px;font-weight:700;letter-spacing:-.02em}
.logo .dot{width:26px;height:26px;border-radius:8px;background:linear-gradient(135deg,#0C66FF,#175CD3);
 display:grid;place-items:center;font-size:13px;color:#fff}
.logo small{color:var(--muted);font-weight:500}
.tenant-wrap{margin-left:auto;display:flex;align-items:center;gap:8px}
.tenant-wrap label{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--faint)}
header select{background:var(--surface2);color:var(--text);
 border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:13px}
main{max-width:860px;margin:0 auto;padding:20px 20px 60px}
.tabs{display:flex;gap:4px;background:var(--surface);border:1px solid var(--border);
 border-radius:10px;padding:4px;margin-bottom:16px}
.tabs button{background:none;border:none;color:#b6c2d6;font-family:inherit;font-size:13px;
 font-weight:500;padding:10px 16px;border-radius:8px;cursor:pointer;transition:.15s;min-height:40px}
.tabs button.active{background:var(--surface2);color:#fff;font-weight:600;
 box-shadow:inset 0 -2px 0 var(--blue)}
.tabs button:hover{color:var(--text)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:18px;margin:12px 0}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
input[type=text]{flex:1;min-width:0;padding:12px 14px;border-radius:10px;border:1px solid var(--border);
 background:var(--surface2);color:var(--text);font-size:14px;font-family:inherit;outline:none;transition:border .15s}
input[type=text]:focus{border-color:var(--blue)}
input[type=text]::placeholder{color:#8290a8}
button.primary{background:var(--blue);color:#fff;border:none;border-radius:10px;padding:12px 20px;
 font-family:inherit;font-size:14px;font-weight:600;cursor:pointer;transition:.15s}
button.primary:hover{background:#175CD3}
.chip{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;font-size:11px;
 font-weight:600;background:var(--surface2);border:1px solid var(--border);color:var(--muted)}
.chip.green{background:var(--green-bg);color:var(--green);border-color:#14532d}
.chip.blue{background:rgba(12,102,255,.12);color:var(--blue-soft);border-color:#194185}
.chip.red{background:var(--red-bg);color:var(--red);border-color:#5f2120}
/* pipeline */
.pipe{display:flex;align-items:center;gap:0;flex-wrap:wrap;margin:14px 0 6px}
.pnode{display:flex;align-items:center;gap:8px}
.pstep{display:flex;align-items:center;gap:8px;padding:8px 14px;border-radius:10px;
 background:var(--surface2);border:1px solid var(--border);font-size:12.5px;font-weight:600;
 color:var(--faint);transition:all .25s}
.pstep .bullet{width:8px;height:8px;border-radius:50%;background:var(--border2);transition:.25s}
.pstep.active{border-color:var(--blue);color:var(--blue-soft);
 box-shadow:0 0 0 3px rgba(12,102,255,.15);animation:pulse 1.2s infinite}
.pstep.active .bullet{background:var(--blue);animation:pulse 1.2s infinite}
.pstep.done{border-color:#14532d;color:var(--green)}
.pstep.done .bullet{background:var(--green)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.55}}
.parrow{color:var(--border2);margin:0 6px;font-size:14px}
/* tool trace */
.trace{margin-top:12px;border-top:1px solid var(--border);padding-top:10px}
.trow{display:flex;gap:10px;align-items:baseline;font-family:ui-monospace,'SF Mono',monospace;
 font-size:12px;padding:6px 10px;border-radius:8px;margin:3px 0;background:var(--surface2)}
.trow .k{color:var(--faint);min-width:70px}
.trow.call .k{color:var(--blue-soft)}
.trow.result{color:#9fb8d8}
.trow.result .k{color:var(--green)}
.trow.thought{font-family:Inter,sans-serif;color:var(--muted)}
.trow.thought .k{color:var(--amber)}
.answer{line-height:1.6;font-size:14.5px;white-space:pre-wrap;margin:0}
.muted{color:var(--muted);font-size:13px}
.section-label{font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;
 color:var(--faint);margin:18px 0 6px}
.log-row{font-family:ui-monospace,monospace;font-size:12px;padding:9px 12px;
 border-bottom:1px solid var(--border);display:flex;gap:10px;align-items:baseline}
.log-row:last-child{border-bottom:none}
.log-row .ts{color:var(--faint);font-size:11px;min-width:130px}
.pending-card{display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap}
.pending-card .meta{font-family:ui-monospace,monospace;font-size:11.5px;color:var(--muted);margin-top:4px}
.pending-card button{padding:9px 16px;border-radius:9px;border:none;font-family:inherit;
 font-size:13px;font-weight:600;cursor:pointer;transition:.15s}
.btn-approve{background:var(--green);color:#04180f}
.btn-approve:hover{filter:brightness(1.1)}
.btn-reject{background:transparent;color:var(--red);border:1px solid #5f2120}
.btn-reject:hover{background:var(--red-bg)}
.empty{text-align:center;color:var(--faint);padding:30px;font-size:14px}
.spin{display:inline-block;width:12px;height:12px;border:2px solid var(--border2);
 border-top-color:var(--blue);border-radius:50%;animation:rot .8s linear infinite;vertical-align:-2px}
@keyframes rot{to{transform:rotate(360deg)}}
.hitl-banner{border:1px solid var(--amber);background:rgba(253,176,34,.08);color:var(--amber);
 border-radius:10px;padding:10px 14px;font-size:13px;margin-top:10px}
.suggest{cursor:pointer}
.suggest:hover{border-color:var(--blue);color:var(--blue-soft)}
.btn-approve,.btn-reject{min-height:40px}
@media (max-width:640px){
 .tagline{display:none}
 .tenant-wrap{margin-left:0;width:100%;justify-content:space-between}
 .pending-card{flex-direction:column;align-items:stretch}
 .pending-card>div:last-child{display:flex;gap:8px}
 .pending-card button{flex:1;min-height:44px}
 .pstep{padding:7px 10px;font-size:11.5px}
 .pnode:first-child .pstep,.pnode .pstep{white-space:nowrap}
}
</style></head><body>
<header>
 <div class="logo"><div class="dot">◈</div>Fleet Copilot <small class="tagline">telemetry · insights · actions</small></div>
 <div class="tenant-wrap"><label>Tenant</label>
  <select id="tenant"><option>acme-001</option><option>globex-002</option><option>initech-003</option></select>
 </div>
</header>
<main>
 <div class="tabs">
  <button data-tab="chat" class="active">Copilot</button>
  <button data-tab="pending">Pending actions</button>
  <button data-tab="audit">Audit log</button>
 </div>
 <div id="tab-chat">
  <div class="card">
   <div class="row">
    <input type="text" id="q" placeholder="Ask about your fleet — health, compliance, trends, actions…"
           onkeydown="if(event.key==='Enter')ask()">
    <button class="primary" onclick="ask()">Send</button>
   </div>
   <div id="pipeline"></div>
  </div>
  <div id="history">
   <div class="card empty" id="empty-state">
    <div style="font-size:15px;font-weight:600;margin-bottom:6px">Ask your fleet anything</div>
    <div class="muted" style="margin-bottom:12px">Grounded answers with cited evidence. Try:</div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center">
     <span class="chip suggest">Which devices are low on disk?</span>
     <span class="chip suggest">Any batteries approaching end of life?</span>
     <span class="chip suggest">Show compliance drift this month</span>
     <span class="chip suggest">Flag M4XVHUV1MEPZ for replacement</span>
    </div>
   </div>
  </div>
 </div>
 <div id="tab-pending" style="display:none"><div id="pending-rows"></div></div>
 <div id="tab-audit" style="display:none"><div class="card" id="audit-rows"></div></div>
</main>
<script>
const $=id=>document.getElementById(id);
const NODES=[["planner","Planner"],["analyst","Analyst"],["execute_tools","Tools"],["verifier","Verifier"],["hitl","HITL gate"],["formatter","Formatter"]];
document.querySelectorAll('.suggest').forEach(c=>c.onclick=()=>{$('q').value=c.textContent;ask();});
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('active'));b.classList.add('active');
 for(const t of['chat','pending','audit'])$('tab-'+t).style.display=(t===b.dataset.tab)?'':'none';
 if(b.dataset.tab==='audit')loadAudit(); if(b.dataset.tab==='pending')loadPending();
});
async function post(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});return r.json();}
function pipelineHTML(){
 return `<div class="pipe">${NODES.map(([id,label],i)=>
  `<span class="pnode"><span class="pstep" id="pn-${id}"><span class="bullet"></span>${label}</span>${i<NODES.length-1?'<span class="parrow">→</span>':''}</span>`).join('')}
  </div><div id="live-trace"></div>`;
}
function setNode(id,status){
 const el=$('pn-'+id); if(!el)return;
 el.classList.remove('active','done'); if(status)el.classList.add(status);
}
async function ask(){
 const q=$('q').value.trim(); if(!q)return;
 $('empty-state')?.remove();
 const card=document.createElement('div');card.className='card';
 card.innerHTML=`<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
   <b style="font-size:14px">${q}</b><span class="chip blue" id="run-badge"><span class="spin"></span> running</span></div>
   ${pipelineHTML()}<div id="answer"></div>`;
 $('history').prepend(card);
 $('q').value=''; NODES.forEach(([id])=>setNode(id,null));
 const res=await fetch('/api/query_stream',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({query:q,tenant:$('tenant').value,n:Date.now()})});
 const reader=res.body.getReader(); const dec=new TextDecoder(); let buf='';
 let answer=null, tools=new Set(), hitlShown=false;
 while(true){
   const {value,done}=await reader.read(); if(done)break;
   buf+=dec.decode(value,{stream:true});
   let idx;
   while((idx=buf.indexOf('\n\n'))>=0){
     const line=buf.slice(0,idx); buf=buf.slice(idx+2);
     if(!line.startsWith('data: '))continue;
     const e=JSON.parse(line.slice(6));
     if(e.type==='node'){setNode(e.node,e.status); if(e.node==='analyst'&&e.status==='active')setNode('execute_tools',null);}
     else if(e.type==='tool_call'){
       setNode('execute_tools','active'); tools.add(e.tool);
       $('live-trace').insertAdjacentHTML('beforeend',
        `<div class="trow call"><span class="k">→ ${e.tool}</span><span>${e.args}</span></div>`);
     }
     else if(e.type==='tool_result'){
       $('live-trace').insertAdjacentHTML('beforeend',
        `<div class="trow result"><span class="k">⎋ result</span><span>${e.result.slice(0,180)}</span></div>`);
     }
     else if(e.type==='interrupt'&&!hitlShown){
       hitlShown=true; setNode('hitl','active');
       $('run-badge').outerHTML=`<span class="chip" style="border-color:#8a6116;color:var(--amber)">⏸ awaiting approval</span>`;
       const acts=(e.payload&&e.payload.proposed_actions)||[];
       const list=acts.map(a=>`<div class="trow call"><span class="k">▸ ${a.tool}</span><span>${a.device_id} · ${a.rationale||''}</span></div>`).join('');
       $('answer').innerHTML=`<div class="hitl-banner">⏸ <b>HITL gate</b> — ${acts.length} action(s) proposed and awaiting human approval.
         ${list}
         Open the <b>Pending actions</b> tab to approve or reject.</div>`;
     }
     else if(e.type==='done'){
       ['verifier','formatter'].forEach(n=>setNode(n,'done'));
       $('run-badge').outerHTML=`<span class="chip green">${e.intent}</span>`;
       $('answer').innerHTML=`<div class="section-label">Answer</div><p class="answer">${e.response.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/\n/g,'<br>')}</p>`;
       if(tools.size)$('answer').innerHTML=`<div class="section-label">Tools used</div><div style="margin:4px 0 10px">${[...tools].map(t=>`<span class="chip">${t}</span>`).join(' ')}</div>`+$('answer').innerHTML;
     }
     else if(e.type==='error'){
       $('run-badge').outerHTML=`<span class="chip red">error</span>`;
       $('answer').innerHTML=`<div class="answer" style="color:var(--red)">${e.message}</div>`;
     }
   }
 }
}
async function loadAudit(){
 try{
  const res=await post('/api/audit',{tenant:$('tenant').value,limit:50});
  if(res.error){$('audit-rows').innerHTML=`<div class="empty">${res.error}</div>`;return;}
  $('audit-rows').innerHTML=res.rows?.map(r=>`<div class="log-row">
   <span class="ts">${(r.ts||'').slice(0,19)}</span><span class="chip ${r.event==='refusal'?'red':r.event==='approved'?'green':'blue'}">${r.event}</span>
   <span>${r.tool||''}</span><span class="muted">${r.detail||''}</span></div>`).join('')||'<div class="empty">No audit events yet</div>';
 }catch(err){$('audit-rows').innerHTML=`<div class="empty">DuckDB busy — close the CLI session first.</div>`;}
}
async function loadPending(){
 const box=$('pending-rows');
 try{
  const res=await post('/api/pending',{tenant:$('tenant').value});
  if(res.error){box.innerHTML=`<div class="card empty">${res.error}</div>`;return;}
  box.innerHTML=res.rows?.map(r=>`<div class="card pending-card">
    <div><div style="display:flex;gap:8px;align-items:center"><b>${r.tool}</b><span class="chip">${r.device_id}</span></div>
      <div class="meta">${JSON.stringify(r.args).slice(0,120)}</div>
      <div class="meta">${r.action_id} · ${r.created_at.slice(0,19)}</div></div>
    <div style="display:flex;gap:8px">
      <button class="btn-approve" onclick="resolve('${r.action_id}',true)">Approve</button>
      <button class="btn-reject" onclick="resolve('${r.action_id}',false)">Reject</button></div>
  </div>`).join('')||'<div class="card empty">No pending actions 🎉</div>';
 }catch(err){box.innerHTML=`<div class="card empty">DuckDB busy — close the CLI session first.</div>`;}
}
async function resolve(aid,approved){
 const btns=document.querySelectorAll(`.pending-card button[onclick*="${aid}"]`);
 btns.forEach(b=>b.disabled=true);
 try{
  const res=await post('/api/resolve',{action_id:aid,approved,tenant:$('tenant').value,approver:'web-user'});
  if(!res.ok)alert('Could not resolve: not found, already resolved, or wrong tenant.');
 }catch(err){alert('DuckDB busy — close the CLI session first.');}
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
