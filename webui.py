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


_wf_cache = None
_parked = {}  # action_id -> thread_id (HITL proposals waiting for a decision)


def _get_wf():
    """Build the compiled graph ONCE — the MemorySaver holds parked HITL
    threads; rebuilding it would orphan them (KeyError on resume)."""
    global _wf_cache
    if _wf_cache is None:
        _wf_cache = build_workflow(_get_repo(), checkpointer=MemorySaver())
    return _wf_cache


NODES = [("planner", "Planner"), ("analyst", "Analyst"), ("verifier", "Verifier"),
         ("hitl", "HITL gate"), ("formatter", "Formatter")]


def stream_query(body: dict):
    """Generator yielding SSE events while the graph runs."""
    tenant = body.get("tenant", "acme-001")
    query = body.get("query", "")
    thread_id = f"web-{body.get('n', 0)}"

    def ev(etype: str, **kw) -> str:
        return "data: " + json.dumps({"type": etype, **kw}, default=str) + "\n\n"

    yield ev("start", tenant=tenant, thread_id=thread_id)
    with _run_lock:
        wf = _get_wf()
        thread = {"configurable": {"thread_id": thread_id, "tenant_id": tenant}}
        final = {}
        try:
            updates = wf.stream({"query": query, "tenant_id": tenant},
                                config=thread, stream_mode="updates")
            interrupted = False
            for chunk in updates:
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
                    if node == "verifier" and delta:
                        # surface the grounding gate's verdict — the reviewer's
                        # core proof that grounding actually ran
                        yield ev("verifier",
                                 verified=[{"claim": c.claim,
                                            "evidence": [e.model_dump() for e in c.evidence]}
                                           for c in (delta.get("verified_claims") or [])],
                                 unsupported=list(delta.get("unsupported") or []))
                    final.update(delta or {})
                    yield ev("node", node=node, status="done")
            state = wf.get_state(thread)
            if state.next:  # interrupted mid-graph: hand the proposal to the client
                payload = {}
                for task in state.tasks:
                    interrupts = getattr(task, "interrupts", ()) or ()
                    if interrupts:
                        payload = interrupts[0].value or {}
                        break
                for p in payload.get("proposed_actions", []):
                    if str(p.get("action_id", "")).startswith("act_"):
                        _parked[p["action_id"]] = thread_id
                yield ev("interrupt", thread_id=thread_id,
                         payload=json.loads(json.dumps(payload, default=str)))
                return
            yield ev("done", response=final.get("response") or "(no response produced)",
                     intent=final.get("plan", {}).intent if final.get("plan") else "?",
                     tools=sorted({t["tool"] for t in final.get("tool_outputs", [])}))
        except Exception as exc:  # noqa: BLE001
            import traceback
            yield ev("error", message=f"{type(exc).__name__}: {exc}",
                     trace=traceback.format_exc()[-1200:])


def stream_resume(body: dict):
    """Resolve the proposed action(s) and resume a parked graph thread via SSE."""
    tenant = body.get("tenant", "acme-001")
    thread_id = body.get("thread_id", "")
    approved = bool(body.get("approved"))
    approver = body.get("approver", "web-user")

    def ev(etype: str, **kw) -> str:
        return "data: " + json.dumps({"type": etype, **kw}, default=str) + "\n\n"

    yield ev("start", tenant=tenant, thread_id=thread_id)
    with _run_lock:
        repo = _get_repo()
        wf = _get_wf()
        thread = {"configurable": {"thread_id": thread_id, "tenant_id": tenant}}
        final = {}
        try:
            state = wf.get_state(thread)
            payload = {}
            for task in state.tasks:
                interrupts = getattr(task, "interrupts", ()) or ()
                if interrupts:
                    payload = interrupts[0].value or {}
                    break
            # resolve every real DB action tied to this proposal
            resolved = []
            for p in payload.get("proposed_actions", []):
                aid = p.get("action_id", "")
                if aid.startswith("act_"):
                    ok = repo.resolve_action(tenant, aid, approved, approver)
                    _parked.pop(aid, None)
                    resolved.append({"action_id": aid, "resolved": ok})
            yield ev("resolved", actions=resolved, approved=approved)
            for chunk in wf.stream(Command(resume={"approved": approved, "approver": approver}),
                                   config=thread, stream_mode="updates"):
                for node, delta in chunk.items():
                    if node in ("__interrupt__",):
                        continue
                    yield ev("node", node=node, status="active")
                    final.update(delta or {})
                    yield ev("node", node=node, status="done")
            yield ev("done", response=final.get("response") or "(no response produced)",
                     intent=final.get("plan", {}).intent if final.get("plan") else "?",
                     tools=sorted({t["tool"] for t in final.get("tool_outputs", [])}),
                     decision="APPROVED" if approved else "REJECTED")
        except Exception as exc:  # noqa: BLE001
            import traceback
            yield ev("error", message=f"{type(exc).__name__}: {exc}",
                     trace=traceback.format_exc()[-1200:])


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
    out = []
    for r in rows:
        # the proposal's audit entry is the only place evidence is stored
        ev = _get_repo().con.execute(
            "SELECT evidence_json FROM audit_log WHERE tenant_id=? AND event='proposal' "
            "AND detail=? ORDER BY ts DESC LIMIT 1",
            (tenant, f"action_id={r[0]}")).fetchone()
        out.append({"action_id": r[0], "tool": r[1], "device_id": r[2],
                    "args": json.loads(r[3]), "created_at": str(r[4]),
                    "evidence": json.loads(ev[0]) if ev else []})
    return {"rows": out}


def api_resolve(body: dict) -> dict:
    tenant = body.get("tenant", "acme-001")
    aid = body["action_id"]
    approved = bool(body.get("approved"))
    ok = _get_repo().resolve_action(tenant, aid, approved=approved,
                                    approver=body.get("approver", "web-user"))
    # if this action parks a graph thread (approved from the Pending tab),
    # resume it server-side so the checkpoint completes and state stays consistent
    thread_id = _parked.pop(aid, None)
    resumed = False
    if ok and thread_id:
        try:
            with _run_lock:
                wf = _get_wf()
                thread = {"configurable": {"thread_id": thread_id, "tenant_id": tenant}}
                for _ in wf.stream(Command(resume={"approved": approved,
                                                   "approver": body.get("approver", "web-user")}),
                                   config=thread, stream_mode="updates"):
                    pass
            resumed = True
        except Exception:  # noqa: BLE001 — resolve already succeeded; resume is best-effort
            resumed = False
    return {"ok": ok, "resumed": resumed}


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
        elif self.path.startswith("/static/"):
            # self-hosted assets (fonts): keeps the strictly-local promise
            base = (ROOT / "webui_static").resolve()
            fp = (base / self.path[len("/static/"):]).resolve()
            if not fp.is_file() or base not in fp.parents:
                return self._json({"error": "not found"}, 404)
            ctype = "font/woff2" if fp.suffix == ".woff2" else "application/octet-stream"
            self._send(200, ctype, fp.read_bytes(), {"Cache-Control": "max-age=86400"})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or "{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad json"}, 400)
        if self.path == "/api/query_stream":
            gen = stream_query(body)
        elif self.path == "/api/resume_stream":
            gen = stream_resume(body)
        else:
            gen = None
        if gen is not None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for chunk in gen:
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
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Crect x='4' y='4' width='16' height='16' rx='4' fill='%230C66FF'/%3E%3Crect x='9' y='9' width='6' height='6' rx='1.5' fill='%23fff'/%3E%3C/svg%3E">
<style>
/* self-hosted IBM Plex: no CDN, works offline, distinctive technical voice */
@font-face{font-family:'IBM Plex Sans';font-style:normal;font-weight:400;font-display:swap;
 src:url(/static/fonts/ibm-plex-sans-latin-400-normal.woff2) format('woff2')}
@font-face{font-family:'IBM Plex Sans';font-style:normal;font-weight:500;font-display:swap;
 src:url(/static/fonts/ibm-plex-sans-latin-500-normal.woff2) format('woff2')}
@font-face{font-family:'IBM Plex Sans';font-style:normal;font-weight:600;font-display:swap;
 src:url(/static/fonts/ibm-plex-sans-latin-600-normal.woff2) format('woff2')}
@font-face{font-family:'IBM Plex Sans';font-style:normal;font-weight:700;font-display:swap;
 src:url(/static/fonts/ibm-plex-sans-latin-700-normal.woff2) format('woff2')}
@font-face{font-family:'IBM Plex Mono';font-style:normal;font-weight:400;font-display:swap;
 src:url(/static/fonts/ibm-plex-mono-latin-400-normal.woff2) format('woff2')}
@font-face{font-family:'IBM Plex Mono';font-style:normal;font-weight:500;font-display:swap;
 src:url(/static/fonts/ibm-plex-mono-latin-500-normal.woff2) format('woff2')}
:root{
 --bg:#021027; --surface:#0B1328; --surface2:#101828; --border:#1D2939;
 --border2:#24344d; --text:#EAECF0; --muted:#98A2B3; --faint:#667085;
 --blue:#0C66FF; --blue-soft:#D1E0FF; --green:#17B26A; --green-bg:#083227;
 --red:#F97066; --red-bg:#3a1114; --amber:#FDB022;
}
*{box-sizing:border-box}
body{font-family:'IBM Plex Sans',-apple-system,sans-serif;margin:0;background:
 radial-gradient(1200px 600px at 80% -10%, #0a1f4d 0%, transparent 60%), var(--bg);
 color:var(--text);min-height:100vh}
header{display:flex;align-items:center;gap:14px;padding:12px 24px;border-bottom:1px solid var(--border);
 background:rgba(2,16,39,.7);backdrop-filter:blur(8px);position:sticky;top:0;z-index:5;flex-wrap:wrap}
.logo{display:flex;align-items:center;gap:10px;font-weight:700;letter-spacing:-.02em}
.logo .dot{width:26px;height:26px;border-radius:8px;background:linear-gradient(135deg,#0C66FF,#175CD3);
 display:grid;place-items:center;font-size:13px;color:#fff}
.logo small{color:var(--muted);font-weight:500}
header select, #tenant{background:var(--surface2);color:var(--blue-soft);
 border:1px solid #194185;border-radius:10px;padding:12px 10px;font-size:13px;
 font-weight:600;outline:none;transition:border .15s}
#tenant{flex:0 0 auto;min-width:130px}
#tenant:focus{border-color:var(--blue)}
#tenant option{color:var(--text);background:var(--surface2)}
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
.trow{display:flex;gap:10px;align-items:baseline;font-family:'IBM Plex Mono',ui-monospace,monospace;
 font-size:12px;padding:6px 10px;border-radius:8px;margin:3px 0;background:var(--surface2)}
.trow .k{color:var(--faint);min-width:70px}
.trow.call .k{color:var(--blue-soft)}
.trow.result{color:#9fb8d8}
.trow.result .k{color:var(--green)}
.trow.thought{font-family:'IBM Plex Sans',sans-serif;color:var(--muted)}
.trow.thought .k{color:var(--amber)}
.answer{line-height:1.6;font-size:14.5px;white-space:pre-wrap;margin:0}
.muted{color:var(--muted);font-size:13px}
.section-label{font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;
 color:var(--faint);margin:18px 0 6px}
.log-row{font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:12px;padding:9px 12px;
 border-bottom:1px solid var(--border);display:flex;gap:10px;align-items:baseline}
.log-row:last-child{border-bottom:none}
.log-row .ts{color:var(--faint);font-size:11px;min-width:130px}
.pending-card{display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap}
.pending-card .meta{font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:11.5px;color:var(--muted);margin-top:4px}
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
/* grounding verdict panel */
details.grounding{margin:12px 0 4px;border:1px solid var(--border);border-radius:10px;
 background:var(--surface2);font-size:13px;overflow:hidden}
details.grounding summary{cursor:pointer;padding:9px 12px;display:flex;align-items:center;
 gap:8px;color:var(--muted);list-style:none}
details.grounding summary::-webkit-details-marker{display:none}
details.grounding .vlist{max-height:260px;overflow:auto;border-top:1px solid var(--border)}
.vclaim{padding:8px 12px;border-bottom:1px solid var(--border);line-height:1.45}
.vclaim:last-child{border-bottom:none}
.vev{font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:11px;color:var(--faint);margin-top:3px}
.vclaim.unsupported>div:first-child{color:var(--red)}
/* a11y + states */
@media (prefers-reduced-motion:reduce){
 .pstep.active,.pstep.active .bullet{animation:none}
 .spin{animation-duration:2s}
 *{transition-duration:.01ms !important}
}
button.chip{font-family:inherit}
button.chip.suggest{font-size:11px}
.btn-armed{background:var(--amber) !important;color:#2a1c00 !important}
.btn-cancel{background:transparent;color:var(--muted);border:1px solid var(--border2);
 border-radius:8px;padding:4px 10px;font-family:inherit;font-size:11px;font-weight:600;
 cursor:pointer;min-height:0}
.btn-cancel:hover{color:var(--red);border-color:#5f2120}
.inline-error{color:var(--red);font-size:13px;padding:8px 12px;border:1px solid #5f2120;
 border-radius:8px;background:var(--red-bg);margin-top:8px}
.run-retry{background:transparent;color:var(--blue-soft);border:1px solid #194185;border-radius:8px;
 padding:6px 12px;font-family:inherit;font-size:12px;font-weight:600;cursor:pointer;margin-top:8px}
/* HITL banner */
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
 <div class="logo"><svg width="26" height="26" viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="4" width="16" height="16" rx="4" fill="#0C66FF"/><rect x="9" y="9" width="6" height="6" rx="1.5" fill="#fff"/></svg>Fleet Copilot <small class="tagline">telemetry · insights · actions</small></div>
</header>
<main>
 <div class="tabs" role="tablist" aria-label="Sections">
  <button data-tab="chat" class="active" role="tab" aria-selected="true" id="tabbtn-chat" aria-controls="tab-chat">Copilot</button>
  <button data-tab="pending" role="tab" aria-selected="false" id="tabbtn-pending" aria-controls="tab-pending">Pending actions</button>
  <button data-tab="audit" role="tab" aria-selected="false" id="tabbtn-audit" aria-controls="tab-audit">Audit log</button>
 </div>
 <div id="tab-chat" role="tabpanel" aria-labelledby="tabbtn-chat">
  <div class="card">
   <div class="row">
    <select id="tenant" aria-label="Active tenant" title="Tenant this query runs against"><option>acme-001</option><option>globex-002</option><option>initech-003</option></select>
    <input type="text" id="q" placeholder="Ask about your fleet — health, compliance, trends, actions…"
           onkeydown="if(event.key==='Enter')ask()" aria-label="Ask the fleet copilot">
    <button class="primary" onclick="ask()">Send</button>
   </div>
   <div id="pipeline"></div>
  </div>
  <div id="history">
   <div class="card empty" id="empty-state">
    <div style="font-size:15px;font-weight:600;margin-bottom:6px">Ask your fleet anything</div>
    <div class="muted" style="margin-bottom:12px">Grounded answers with cited evidence. Try:</div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center">
     <button class="chip suggest">Which devices are low on disk?</button>
     <button class="chip suggest">Any batteries approaching end of life?</button>
     <button class="chip suggest">Show compliance drift this month</button>
     <button class="chip suggest">Flag M4XVHUV1MEPZ for replacement</button>
    </div>
   </div>
  </div>
 </div>
 <div id="tab-pending" role="tabpanel" aria-labelledby="tabbtn-pending" style="display:none"><div id="pending-rows" aria-live="polite"></div></div>
 <div id="tab-audit" role="tabpanel" aria-labelledby="tabbtn-audit" style="display:none"><div class="card" id="audit-rows"></div></div>
</main>
<script>
const $=id=>document.getElementById(id);
const NODES=[["planner","Planner"],["analyst","Analyst"],["execute_tools","Tools"],["verifier","Verifier"],["hitl","HITL gate"],["formatter","Formatter"]];
document.querySelectorAll('.suggest').forEach(c=>c.onclick=()=>{$('q').value=c.textContent;ask();});
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('.tabs button').forEach(x=>{x.classList.remove('active');x.setAttribute('aria-selected','false');});
 b.classList.add('active');b.setAttribute('aria-selected','true');
 for(const t of['chat','pending','audit'])$('tab-'+t).style.display=(t===b.dataset.tab)?'':'none';
 if(b.dataset.tab==='audit')loadAudit(); if(b.dataset.tab==='pending')loadPending();
});
const _runs={};  // uid -> per-card run context (no cross-card globals)
async function post(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});return r.json();}

/* per-card context: all DOM refs are scoped, never by global id */
function makeCtx(card){
  const uid='c'+Date.now()+Math.floor(Math.random()*1e4);
  card.setAttribute('data-uid',uid); _runs[uid]={};
  const q=s=>card.querySelector(s);
  const ctx={uid, card,
    badge:q('.run-badge'), answer:q('.answer-slot'), trace:q('.live-trace'),
    thread:null, ac:null, toolTs:0, toolName:'',
    setNode(node,status){
      const el=q(`.pstep[data-node="${node}"]`); if(!el)return;
      el.classList.remove('active','done'); if(status)el.classList.add(status);
    },
    badgeHTML(html){this.badge.outerHTML=html; this.badge=q('.run-badge')||q('.chip');
      q('.btn-cancel')?.remove();},
  };
  _runs[uid].ctx=ctx; return ctx;
}
function ctxOf(el){
  const card=el.closest('.card'); if(!card)return null;
  return _runs[card.dataset.uid]?.ctx||null;
}
function armed(btn,fn){
  /* two-stage confirm: first click arms (3s), second click acts */
  if(btn.dataset.armed){fn();return;}
  btn.dataset.armed='1'; btn.dataset.orig=btn.textContent;
  btn.textContent='Confirm — irreversible'; btn.classList.add('btn-armed');
  setTimeout(()=>{if(btn.isConnected){delete btn.dataset.armed;
    btn.textContent=btn.dataset.orig; btn.classList.remove('btn-armed');}},3000);
}
function pipelineHTML(){
 return `<div class="pipe">${NODES.map(([id,label],i)=>
  `<span class="pnode"><span class="pstep" data-node="${id}"><span class="bullet"></span>${label}</span>${i<NODES.length-1?'<span class="parrow">→</span>':''}</span>`).join('')}
  </div><div class="live-trace"></div>`;
}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');}

async function askQuestion(q,tenant){
 $('empty-state')?.remove();
 const card=document.createElement('div');card.className='card';
 card.innerHTML=`<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
   <b style="font-size:14px">${esc(q)}</b><span style="display:flex;gap:6px;align-items:center">
     <span class="chip" title="tenant of this run">${esc(tenant)}</span>
     <span class="chip blue run-badge" aria-live="polite"><span class="spin"></span> running</span>
     <button class="btn-cancel" onclick="cancelRun(this)">Cancel</button></span></div>
   ${pipelineHTML()}<div class="answer-slot" aria-live="polite"></div>`;
 $('history').prepend(card);
 const ctx=makeCtx(card); ctx.query=q; ctx.tenant=tenant; ctx.ac=new AbortController();
 NODES.forEach(([id])=>ctx.setNode(id,null));
 await consumeStream('/api/query_stream',{query:q,tenant:tenant,n:Date.now()},ctx,{});
}
async function ask(){
 const q=$('q').value.trim(); if(!q)return;
 $('q').value='';
 await askQuestion(q,$('tenant').value);
}
function cancelRun(btn){
 const ctx=ctxOf(btn); if(!ctx)return;
 btn.disabled=true; btn.textContent='Cancelled';
 ctx.ac?.abort();
}
async function resumeThread(approved,el){
 const ctx=ctxOf(el); if(!ctx||!ctx.thread)return;
 ctx.setNode('hitl','done');
 // tenant of the RUN, not the current selector — switching tenants mid-decision
 // must never redirect the resume to another company
 await consumeStream('/api/resume_stream',
   {thread_id:ctx.thread,tenant:ctx.tenant||$('tenant').value,approved,approver:'web-user'},
   ctx,{resume:true,approved});
}

async function consumeStream(url,body,ctx,st){
 const res=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify(body),signal:ctx.ac?.signal});
 const reader=res.body.getReader(); const dec=new TextDecoder(); let buf='';
 let tools=new Set(), hitlShown=false;
 const toolsHTML=()=>tools.size?`<div class="section-label">Tools used</div><div style="margin:4px 0 10px">${[...tools].map(t=>`<span class="chip">${t}</span>`).join(' ')}</div>`:'';
 const handleEvent=(e)=>{
   if(e.type==='node'){
     ctx.setNode(e.node,e.status);
     if(e.node==='analyst'&&e.status==='active')ctx.setNode('execute_tools',null);
   }
   else if(e.type==='tool_call'){
     ctx.setNode('execute_tools','active'); tools.add(e.tool);
     ctx.toolTs=Date.now(); ctx.toolName=e.tool;
     ctx.trace.insertAdjacentHTML('beforeend',
      `<div class="trow call"><span class="k">→ ${e.tool}</span><span>${esc(e.args)}</span></div>`);
   }
   else if(e.type==='tool_result'){
     const elapsed=ctx.toolTs?` <span style="color:var(--faint)">· ${((Date.now()-ctx.toolTs)/1000).toFixed(1)}s</span>`:'';
     const cut=e.result.length>180?'…':'';
     ctx.trace.insertAdjacentHTML('beforeend',
      `<div class="trow result"><span class="k">result${elapsed}</span><span>${esc(e.result.slice(0,180))}${cut}</span></div>`);
   }
   else if(e.type==='verifier'){
     const nv=(e.verified||[]).length, nu=(e.unsupported||[]).length;
     const evLine=c=>(c.evidence||[]).map(x=>`${x.device_id}@${String(x.collected_at).slice(0,19)} ${x.metric}=${x.value}`).join(' · ');
     const vrows=(e.verified||[]).map(c=>
       `<div class="vclaim"><div>${esc(c.claim)}</div>${evLine(c)?`<div class="vev">${esc(evLine(c))}</div>`:''}</div>`).join('');
     const urows=(e.unsupported||[]).map(u=>
       `<div class="vclaim unsupported"><div>not grounded — excluded from the answer</div><div>${esc(u)}</div></div>`).join('');
     ctx.trace.insertAdjacentHTML('beforeend',
      `<details class="grounding"${nu?' open':''}>
        <summary><span class="chip ${nu?'red':'green'}">grounding check</span>
         <span><b>${nv}</b> claim(s) verified · <b>${nu}</b> unsupported</span></summary>
        <div class="vlist">${vrows}${urows}</div></details>`);
   }
   else if(e.type==='interrupt'&&!hitlShown){
     hitlShown=true; ctx.setNode('hitl','active');
     ctx.thread=e.thread_id;
     ctx.badgeHTML(`<span class="chip run-badge awaiting" style="border-color:#8a6116;color:var(--amber)" aria-live="polite"><svg width="11" height="11" viewBox="0 0 12 12" style="vertical-align:-1px" aria-hidden="true"><rect x="2" y="1.5" width="3" height="9" rx="1" fill="currentColor"/><rect x="7" y="1.5" width="3" height="9" rx="1" fill="currentColor"/></svg> awaiting approval</span>`);
     const acts=(e.payload&&e.payload.proposed_actions)||[];
     const ACT_DESC={
       notify_employee:a=>`send this message to the employee of the device: "${a.args?.message||''}"`,
       create_upgrade_order:a=>`create a ${a.args?.component||'?'} upgrade order (spec: ${a.args?.spec||'?'})`,
       open_remediation_ticket:a=>`open a remediation ticket for check "${a.args?.check_id||'?'}" — note: ${a.args?.note||''}`,
       flag_device_for_replacement:a=>`flag the device for replacement — reason: ${a.args?.reason||''}`
     };
     const list=acts.map(a=>{
       const ev=(a.evidence||[]).map(x=>`${x.device_id}@${String(x.collected_at).slice(0,19)} ${x.metric}=${x.value}`).join(' · ');
       const desc=(ACT_DESC[a.tool]||(()=>a.rationale||''))(a);
       return `<div class="trow call" data-aid="${a.action_id}"><span class="k">${a.tool}</span><span>${esc(a.device_id+' — '+desc)}</span></div>`+
        (ev?`<div class="trow result"><span class="k">evidence</span><span>${esc(ev)}</span></div>`
           :`<div class="trow"><span class="k">evidence</span><span>none attached</span></div>`)+
        `<div class="trow"><span class="k">on approve</span><span>action is recorded as APPROVED in the audit log</span></div>`+
        `<div class="trow"><span class="k">on reject</span><span>action is recorded as REFUSED in the audit log</span></div>`;
     }).join('');
     ctx.answer.insertAdjacentHTML('afterbegin',`<div class="hitl-banner"><b>HITL gate</b> — ${acts.length} action(s) proposed.
       ${list}
       <div class="row" style="margin-top:10px">
        <button class="primary btn-approve" onclick="armed(this,()=>resumeThread(true,this))">Approve &amp; continue</button>
        <button class="btn-reject" onclick="armed(this,()=>resumeThread(false,this))" style="min-height:40px">Reject</button>
       </div></div>`);
   }
   else if(e.type==='resolved'){
     ctx.setNode('hitl','done');
     ctx.badgeHTML(`<span class="chip ${e.approved?'green':'red'}" aria-live="polite">${e.approved?'APPROVED':'REJECTED'} · resuming</span>`);
   }
   else if(e.type==='done'){
     ['verifier','formatter'].forEach(n=>ctx.setNode(n,'done'));
     const decision=e.decision||(st.resume?(e.decision||''):'');
     ctx.badgeHTML(`<span class="chip ${decision?(decision==='APPROVED'?'green':'red'):'green'}" aria-live="polite">${decision||e.intent||'done'}</span>`);
     ctx.answer.innerHTML=toolsHTML()+`<div class="section-label">Answer</div><p class="answer">${esc(e.response||'(no response produced)').replace(/\n/g,'<br>')}</p>`;
     loadPending();
   }
   else if(e.type==='error'){
     ctx.badgeHTML(`<span class="chip red" aria-live="polite">error</span>`);
     ctx.answer.insertAdjacentHTML('beforeend',
      `<div class="inline-error">Run failed — ${esc(e.message.split('\n')[0].slice(0,180))}
        <button class="run-retry" onclick="askQuestion(ctxOf(this).query,$('tenant').value)">Retry this question</button></div>`);
   }
 };
 try{
   while(true){
     const {value,done}=await reader.read(); if(done)break;
     buf+=dec.decode(value,{stream:true});
     let idx;
     while((idx=buf.indexOf('\n\n'))>=0){
       const line=buf.slice(0,idx); buf=buf.slice(idx+2);
       if(!line.startsWith('data: '))continue;
       try{handleEvent(JSON.parse(line.slice(6)));}
       catch(err){console.error('handler:',err);}
     }
   }
 }catch(err){
   if(err.name==='AbortError'){
     ctx.badgeHTML(`<span class="chip" aria-live="polite">cancelled</span>`);
     ctx.answer.insertAdjacentHTML('beforeend',
      `<div class="inline-error">Run cancelled by you. <button class="run-retry" onclick="askQuestion(ctxOf(this).query,$('tenant').value)">Run it again</button></div>`);
   }else{
     ctx.badgeHTML(`<span class="chip red" aria-live="polite">error</span>`);
     ctx.answer.insertAdjacentHTML('beforeend',
      `<div class="inline-error">Stream failed — ${esc(err.message)}
        <button class="run-retry" onclick="askQuestion(ctxOf(this).query,$('tenant').value)">Retry this question</button></div>`);
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
  box.innerHTML=res.rows?.map(r=>{
   const evRows=(r.evidence&&r.evidence.length?r.evidence:[]).map(x=>
     `<div class="trow result"><span class="k">⎋ ev</span><span>${esc(`${x.device_id||''}@${String(x.collected_at||'').slice(0,19)} ${x.metric||''}=${x.value||''}`)}</span></div>`).join('')
    ||`<div class="trow"><span class="k">evidence</span><span>none attached</span></div>`;
   return `<div class="card pending-card">
    <div style="min-width:0;flex:1">
     <div style="display:flex;gap:8px;align-items:center"><b>${r.tool}</b><span class="chip">${r.device_id}</span></div>
     <div class="meta">${esc(JSON.stringify(r.args))}</div>
     ${evRows}
     <div class="meta">on approve → recorded as APPROVED in the audit log · on reject → REFUSED</div>
     <div class="meta">${r.action_id} · ${r.created_at.slice(0,19)}</div>
    </div>
    <div style="display:flex;gap:8px">
      <button class="btn-approve" onclick="armed(this,()=>resolve('${r.action_id}',true,this))">Approve</button>
      <button class="btn-reject" onclick="armed(this,()=>resolve('${r.action_id}',false,this))">Reject</button></div>
   </div>`;}).join('')||'<div class="card empty">No pending actions</div>';
 }catch(err){box.innerHTML=`<div class="card empty">DuckDB busy — close the CLI session first.</div>`;}
}
async function resolve(aid,approved,btn){
 const card=btn?.closest('.pending-card');
 card?.querySelectorAll('button').forEach(b=>{if(b!==btn)b.disabled=true;});
 try{
  const res=await post('/api/resolve',{action_id:aid,approved,tenant:$('tenant').value,approver:'web-user'});
  if(!res.ok){
   card?.querySelectorAll('button').forEach(b=>{b.disabled=false;delete b.dataset.armed;
     if(b.dataset.orig)b.textContent=b.dataset.orig;b.classList.remove('btn-armed');});
   card?.insertAdjacentHTML('beforeend',
    `<div class="inline-error">Could not resolve ${esc(aid)}: not found, already resolved, or wrong tenant.</div>`);
  }
  else markResolvedInChat(aid,approved,res.resumed);
 }catch(err){
  card?.insertAdjacentHTML('beforeend',
   `<div class="inline-error">DuckDB busy — close the CLI session first, then retry.</div>`);
 }
 loadPending();
}
function markResolvedInChat(aid,approved,resumed){
 const row=document.querySelector(`.trow[data-aid="${aid}"]`);
 if(!row)return;
 const banner=row.closest('.hitl-banner'); if(!banner)return;
 banner.innerHTML=`<span class="chip ${approved?'green':'red'}">${approved?'APPROVED':'REJECTED'} via Pending tab</span>
   <span class="muted" style="margin-left:8px">${aid}${resumed?' — graph resumed server-side':''}</span>`;
 const badge=row.closest('.card')?.querySelector('.run-badge.awaiting');
 if(badge)badge.outerHTML=`<span class="chip ${approved?'green':'red'}">${approved?'APPROVED':'REJECTED'}</span>`;
}
</script></body></html>"""


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Fleet Copilot UI → http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
