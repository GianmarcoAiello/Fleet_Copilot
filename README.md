# Fleet Copilot

Agentic Fleet Copilot over device telemetry — take-home assessment for
**Senior Agentic AI Engineer** (Rayda).

A LangGraph agent answers fleet health / compliance / trend questions and
proposes remediation actions for three synthetic companies, with every answer
grounded in telemetry (device_id + collected_at citations) and every
state-changing action gated by explicit human approval.

```
planner ──► analyst ◄──► execute_tools ──► verifier ──► HITL gate ──► formatter
             ▲  │  (ReAct loop)             │  ▲          (interrupt)
             └──┘        │                  │  └─ re_analyze (re-ground loop)
                         └──────────────────┴─────── respond ──► END
```

- **Telemetry**: `device-telemetry-dataset.ndjson` — 750 snapshots, 25 devices,
  3 companies (`acme-001`, `globex-002`, `initech-003`), 30 days (2026-05-14 → 06-12).
- **Storage**: DuckDB, rebuilt from the NDJSON (`data/fleet.duckdb`).
- **Strictly local**: identifiers never leave the project except toward the LLM you configure.

---

## Try it with your own key

The agent is BYOK: it talks to any provider langchain supports, or to any
OpenAI-compatible endpoint. Three env vars, one file.

```bash
cp .env.example .env      # then edit
python3.12 -m venv .venv && source .venv/bin/activate
uv pip install -e .       # or: pip install -e .
python webui.py 8080      # → http://127.0.0.1:8080
```

| variable | meaning | example |
|---|---|---|
| `FLEET_LLM_MODEL` | provider-qualified model id | `google_genai:gemini-2.0-flash` |
| `LLM_API_KEY` | your key (routed to the right constructor param automatically) | `sk-…` |
| `LLM_BASE_URL` | optional OpenAI-compatible endpoint | `https://openrouter.ai/api/v1` |

Routing rules (implemented in `src/graph/nodes.py::get_llm`, unit-verified for all cases):

- **`LLM_BASE_URL` set** → *any* model id is served through an OpenAI-compatible
  endpoint (Nous Portal, OpenRouter, vLLM, Ollama): `openai/gpt-6-luna`,
  `my-finetuned-32b`, …
- **slash-form id without base URL** (`openai/gpt-6-luna`) → defaults to the
  Nous Portal endpoint.
- **native provider** (`google_genai:…`, `anthropic:…`, `openai:…`) →
  `LLM_API_KEY` is mapped to `google_api_key` / `anthropic_api_key` / `api_key`.
  If `LLM_API_KEY` is unset, the provider's standard env var
  (`GOOGLE_API_KEY`, …) still works.

Examples:

```bash
# Nous Portal (slash id, base URL implied)
FLEET_LLM_MODEL=openai/gpt-6-luna
LLM_API_KEY=nous_sk_...

# OpenRouter
FLEET_LLM_MODEL=openai/anthropic/claude-sonnet-4.5
LLM_API_KEY=sk-or-...
LLM_BASE_URL=https://openrouter.ai/api/v1

# native Anthropic
FLEET_LLM_MODEL=anthropic:claude-sonnet-4-5
LLM_API_KEY=sk-ant-...

# local vLLM
FLEET_LLM_MODEL=qwen3-32b
LLM_API_KEY=none
LLM_BASE_URL=http://localhost:8000/v1
```

> Note: `temperature=0` everywhere. Small nondeterminism can still surface from
> the provider side; the guardrails below do not depend on it.

---

## Run

| entry point | what it does |
|---|---|
| `python webui.py 8080` | web UI: live agent pipeline + tool trace, HITL approve/reject, audit log, pending actions |
| `python cli.py --tenant acme-001` | interactive CLI; every action proposal pauses with y/N (+ audit inspection) |
| `python cli.py --auto-approve --query "…"` | scripted demo run |
| `python run_once.py` | one QA question end-to-end (no HITL pause) |
| `python run_hitl.py "<query>" <tenant> approve\|reject` | scripted HITL demo (interrupt → resume → audit tail) |
| `python actions_cli.py list\|approve <id>\|reject <id>` | manage pending actions from the audit trail |

The first run rebuilds `data/fleet.duckdb` from the NDJSON. DuckDB is
single-writer: while the web UI is open, CLI sessions are refused with a
clear message (and vice versa).

## Evaluation

```bash
python evals/run_evals.py            # deterministic: tool-level guardrails, tenant scoping, audit trail (no LLM)
python evals/run_evals.py --live     # full graph through the real LLM
python evals/run_qa100.py            # live 100-question suite (see below)
```

`evals/run_qa100.py` runs the agent against `fleet_copilot_100_qa.json`
(100 generated QA pairs: 45 grounded QA, 25 trend insights, 20 action
proposals, 10 guardrail refusals) and scores three layers:

1. **deterministic** — expected-device coverage in the answer, evidence
   citations, cross-tenant leaks, HITL interrupt presence, audit outcome;
2. **LLM judge** — agent answer vs the reference answer (factual_match 0..1,
   hallucination flag);
3. **dataset validation** — expected facts are recomputed from the NDJSON, so
   a wrong expectation is never charged to the agent (currently 0 mismatches).

First 10 live cases (gpt-6-luna via Nous Portal): **8/10 pass**, 0
hallucinations, 0 cross-tenant leaks; the 1 real miss (QA-010) exposed a tool
bug — `query_compliance` returned raw 30-day history capped at 25 rows,
hiding 2 of 4 devices failing `os_up_to_date` — since fixed with a
`scope=latest` dedup. Results stream to `evals/qa100_results.json`
(resume-safe).

---

## Architecture

```
src/
  graph/nodes.py      planner, analyst (ReAct), verifier, HITL gate, formatter
  graph/state.py      typed state: plan, messages, tool_outputs, claims, proposals
  tools/read_tools.py 4 read tools + 4 action tools (tenant injected, never LLM-settable)
  db/loader.py        NDJSON → DuckDB schema; queries; actions/audit (append-only)
webui.py              stdlib HTTP server + SSE streaming (no web framework)
cli.py / run_once.py / run_hitl.py / actions_cli.py
evals/                deterministic suite + 100-QA live runner
```

**Explicit loop, not a prompt wrapper.** The planner emits a typed
`ExecutionPlan` (intent: `qa | trend_insight | action_request | mixed`, steps,
action requests). The analyst runs a bounded ReAct loop (max 6 iterations) with
tool calling. The verifier is a *separate* LLM pass that re-extracts verifiable
claims from the draft and checks each against the evidence pool; unsupported
claims are labelled, and if any remain the graph routes the analyst back for
one more grounding pass (`re_analyze`) before responding. Action requests are
never executed inline: they stop the graph at the HITL gate via
`interrupt()` and resume only with a human `Command(resume=…)`.

### Tool catalog

| tool | type | notes |
|---|---|---|
| `query_fleet_snapshot` | read | latest snapshot per device; filters: os_version_lt, min_avail_disk_gb, check, software |
| `analyze_time_series_trends` | read | pre-built 30-day aggregates: disk, battery, ram, compliance drift, software |
| `query_device_history` | read | full time-series for one device (tenant-checked) |
| `query_compliance` | read | `scope=latest` (dedup per device+check) or `scope=all` (history); cap declared with `total_matching`/`truncated` |
| `create_upgrade_order` | action | component/spec; propose-only |
| `open_remediation_ticket` | action | check_id/note; propose-only |
| `flag_device_for_replacement` | action | reason; propose-only |
| `notify_employee` | action | message for the device's employee; propose-only |

Tenant id is injected via `RunnableConfig` (`InjectedToolArg`) — it is never a
parameter the model can set.

### Grounding strategy

- Every fact must trace to a tool result; answers cite `device_id` +
  `collected_at` (+ metric/value).
- **Evidence pool** = device ids *confirmed by tool results only* — never from
  tool-call args (a failed lookup proves the device was NOT found, but the id
  appears in the error text) and never from error/refusal replies. This closed
  a real hole where a cross-tenant device the tools never found could get an
  action approved.
- The verifier scores every claim against that pool; unsupported ones are
  surfaced in the answer under "NOT stated (insufficient evidence)" instead of
  being silently dropped or invented.
- Read tools never fabricate: empty results come back as empty, and the
  analyst prompt requires saying so.

### Guardrails (Action Quality)

1. **Tenant isolation** — every query/action is scoped by injected tenant;
   cross-tenant devices are refused and audited; the QA suite checks no
   foreign device id ever appears in an answer.
2. **Propose-only action tools** — the LLM can create a *proposal*, never
   execute. Execution requires a human decision that re-enters the graph as a
   structured `Command(resume=...)`.
3. **Evidence gate** — an action tool without telemetry evidence refuses
   (`insufficient evidence`), and the HITL gate refuses *before* asking the
   human if the target device was never confirmed by a tool result.
4. **Audit trail** — append-only `audit_log`: proposal → approved/refused,
   with args + evidence JSON and approver id; visible in the web UI and
   `actions_cli.py`.

---

## Design decisions (and why)

- **LangGraph with an explicit planner/verifier/HITL topology** instead of a
  single ReAct chain: the assessment asks for planning + tool calling +
  guardrails as first-class, inspectable steps; each is a node you can test in
  isolation.
- **DuckDB single file, rebuilt from NDJSON**: deterministic, zero-setup,
  trivially resettable; the web UI and CLIs coordinate on the file lock
  instead of a server process.
- **LLM-judge in evals, but never as the only gate**: deterministic checks
  (coverage, citations, leaks, interrupts, audit) carry the pass/fail signal;
  the judge adds semantic comparison against reference answers that are
  themselves validated against the telemetry.
- **Declared truncation over silent truncation**: results that can be large
  (`query_compliance` history) report `total_matching` + `truncated` so the
  agent knows it sees a subset. The QA-010 failure mode (a silent 25-row cap
  deciding which devices the agent could ever see) is exactly what this
  prevents.
- **Generic BYOK env (`FLEET_LLM_MODEL` / `LLM_API_KEY` / `LLM_BASE_URL`)**
  with a small routing table in one function (`get_llm`), unit-verified:
  OpenAI-compatible endpoints, native providers, and legacy Nous vars all
  resolve to the right client.

## Deliberately left out (prototype scope)

- **No web auth / multi-user**: the UI binds to 127.0.0.1 and trusts the
  tenant selector; real deployment needs SSO + per-user tenant binding.
- **In-memory HITL checkpointing** (`MemorySaver`): parked threads don't
  survive a restart; a durable checkpointer (SQLite/Postgres) is a drop-in.
- **Single-process DuckDB**: no server mode, no concurrent writers; fine for
  750 snapshots, wrong for a real fleet ingest pipeline.
- **Hard caps** (25–40 rows) instead of pagination/streaming for very large
  fleets; caps are declared, not solved.
- **Evals run sequentially, judge = same model as the agent**: a stronger
  judge model and parallel case execution are the next rigor step; live runs
  keep small provider-side nondeterminism despite temperature 0.
- **No CI wiring, no Docker, no i18n** (agent answers in the user's language;
  eval checks understand EN + IT refusal phrasing).
- **Insights are prompt-driven, not statistical**: trend tools return
  aggregates, the analyst narrates; dedicated anomaly detection (e.g. z-score
  on disk burn rate) is out of scope for the prototype.
