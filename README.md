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
- **Storage**: DuckDB at `data/fleet.duckdb`, auto-built from the NDJSON on
  first open (the Repository bootstraps a missing DB — a fresh clone just works).
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
pip install -e . && pytest tests -q     # deterministic graph tests, stub LLM, no API key
python evals/run_evals.py               # tool-level guardrails on a hermetic temp DB (no LLM)
python evals/run_evals.py --live        # full graph through the real LLM
python evals/run_qa100.py               # live 100-question suite (see below)
```

Both eval runners are **hermetic**: the DB is rebuilt from the NDJSON into a
temp file — nothing under `data/` or git is touched, no API key needed for the
deterministic suite. `tests/test_graph.py` drives the real LangGraph with a
stub LLM and asserts the properties that matter: linear message growth (no
duplicated `tool_call_id` — the reducer bug class), the verifier rejecting
wrong values / invented timestamps / ungrounded aggregates, invented action
evidence refused with exactly one audit row, tenant isolation with `config`
as the single source of truth, and the `get_llm` env routing table.

`evals/run_qa100.py` runs the agent against `evals/fleet_copilot_100_qa.json`
(100 generated QA pairs: 45 grounded QA, 25 trend insights, 20 action
proposals, 10 guardrail refusals) and scores three layers:

1. **deterministic** — expected-device coverage in the answer, evidence
   citations, cross-tenant leaks, HITL interrupt presence, audit outcome;
2. **LLM judge** — agent answer vs the reference answer (factual_match 0..1;
   the hallucination flag counts only for direct contradictions, omissions
   lower factual_match only);
3. **dataset validation** — expected facts are recomputed from the NDJSON
   (device counts, OS breakdowns; currently 0 mismatches). It does not yet
   recompute disk/battery aggregates — the trend unit caveat below applies.

**QA-100 result (gpt-6-luna, temperature 0, commit `0c64439`): 64/100 overall.**
This is the last full run. It **predates** the QA-failure fixes listed below,
which were not re-measured (inference cost).

| category | overall pass | deterministic layer only |
|---|---|---|
| grounded_qa | 30/45 | 44/45 |
| trend_insight | 15/25 | 22/25 |
| action_proposal | 9/20 | 9/20 |
| guardrail_refusal | 10/10 | 10/10 |
| **total** | **64/100** | **85/100** |

*overall* requires both layers to pass; *deterministic layer only* ignores the
LLM judge (device coverage, citations, tenant leaks, HITL interrupt, proposed
action set, audit outcome). Mean judge `factual_match`: 0.71 over 90 judged
cases. The deterministic layer is the stable one: an earlier run (commit
`d44a492`, 70/100 overall) scored 86/100 on it; judge-only failures moved from
16 to 21.

**Changes after this run (not re-measured):** per-snapshot disk and network
fields exposed in device history (the run's judge contradictions included answers
claiming disk data was missing), new `gaps` and `versions` trends and dated
disk/battery/software trends, one deterministic retry when an action request
produced no proposal, prompt rules for choosing the action. They target the
failure classes above (tool gaps behind judge contradictions; 4 action requests
with no proposal; 5 with a subset of the reference set). The expected effect is a
hypothesis: only the mechanics are covered by the stub-LLM tests, there is no
live measurement.

Honest caveats:

- **Units**: disk aggregates are **GiB (2^30 bytes)** across tools, snapshots
  and the QA reference answers. An earlier run mixed GB-decimal (tool) with
  GiB (references) and the judge mis-read the difference as contradictions;
  the unit fix is one of the changes behind the trend score moving between
  the two stored runs.
- **Run history, not variance**: an earlier run (commit `d44a492`) scored
  70/100 overall, the committed one 64/100. The code changed in between (units,
  verifier hardening, executor, disk fields), so the 22 cases that flipped mix
  code effects and judge sensitivity. Pure run-to-run noise of a fixed agent
  was **not measured**. What stayed stable is the deterministic layer
  (86 → 85 of 100).
- **action_proposal** (9/20, same on the deterministic layer): 11 failures — 4
  requests ended with no proposal at all (no HITL interrupt); 5 proposed a
  subset of the reference set (typically the ticket without the notification);
  2 chose different actions. One of these (QA-081) expects a replacement flag
  for a low-disk desktop, which the sufficiency rule (battery grounds only)
  refuses by design: a reference/policy inconsistency, not an agent error. The
  "no proposal" cases are real agent misses, and are what the retry and prompt
  changes above target.
- **judge = agent model**: factual_match carries some model bias; hallucination
  detection proper is deterministic (the verifier's grounding check), the judge
  flag is a secondary signal.
- The HITL decision is one per gate invocation (all proposed actions share it);
  the **Pending actions** tab allows approving/rejecting each action separately.
- The planner/analyst prompts were iterated against failures of this QA set, so
  the reported scores are a development measure, not a held-out one.
- Raw results per case: `evals/qa100_results.json` (the runner is resume-safe).

---

## Architecture

```
src/
  graph/nodes.py      planner, analyst (ReAct), verifier, HITL gate, formatter
  graph/state.py      typed state: plan, messages, tool_outputs, claims, proposals
  tools/read_tools.py 4 read tools + 4 action tools (tenant injected, never LLM-settable)
  db/loader.py        NDJSON → DuckDB schema; queries; actions/audit (append-only by convention)
webui.py              stdlib HTTP server + SSE streaming (no web framework)
cli.py / run_once.py / run_hitl.py / actions_cli.py
evals/                deterministic suite + 100-QA live runner
```

> **Note on the web UI**: it is deliberately basic — a minimal surface whose
> only job is to make the agent's behavior observable (live pipeline, tool
> trace, grounding verdict, HITL decisions, audit trail). It is a
> demonstration instrument, not a product shell: polish was intentionally
> spent on the evidence surfaces, not on the chrome.

**Explicit loop, not a prompt wrapper.** The planner emits a typed
`ExecutionPlan` (intent: `qa | trend_insight | action_request | mixed`, steps —
it never invents actions). The analyst runs a bounded ReAct loop (max 6 turns,
4 tool calls per turn; at budget exhaustion a final answer is forced without
tools bound). The verifier is a *separate* LLM pass that re-extracts verifiable
claims and checks them against a deterministic **fact index** built from the
tool-result payloads: every evidence ref must resolve to a real
`(device_id, collected_at)` row with a matching value (1% numeric tolerance),
unknown metrics and ungrounded numbers are rejected. Unsupported claims are
labelled, and if any remain the graph routes the analyst back for **one
informed re-ground pass** (`re_analyze`) — the retry message lists the rejected
claims. Action proposals stop the graph at the HITL gate via `interrupt()`
and resume only with a human `Command(resume=…)`; after approval an executor
node flips the action `approved → executed` through a simulated connector.

### Tool catalog

| tool | type | notes |
|---|---|---|
| `query_fleet_snapshot` | read | latest snapshot per device **with disk/battery/compliance aggregates joined** (disk in bytes + GiB + free %); filters combine with AND: min_avail_disk_gib (latest value, GiB), os_version_lt (numeric compare, macOS), software, check |
| `analyze_time_series_trends` | read | pre-built 30-day aggregates: disk, battery, ram, compliance pass→fail transitions (with first fail date), software |
| `query_device_history` | read | full time-series for one device (tenant-checked) |
| `query_compliance` | read | `scope=latest` dedups the **latest result per device+check across all statuses first, then filters**; `scope=all` keeps history; cap declared with `total_matching`/`truncated` |
| `create_upgrade_order` | action | component/spec; evidence re-read from the DB; sufficiency rule: real RAM (>90%) or disk (<20 GB) pressure, or degraded battery |
| `open_remediation_ticket` | action | check_id/note; rule: the cited check must be failing on that device right now |
| `flag_device_for_replacement` | action | rule: battery condition must be 'Service Recommended' |
| `notify_employee` | action | rule: the device must have an assigned employee |

**Evidence is never trusted from the model**: action tools re-read every
evidence ref from the tenant-scoped DB (metric + value, 1% numeric tolerance),
store the DB value in the audit trail, and refuse on mismatch, unknown metric,
or failed sufficiency rule. Proposals have a single path: the analyst proposes
through action tools after reading telemetry; the planner cannot invent
actions.

Tenant id is injected via `RunnableConfig` (`InjectedToolArg`) — it is never a
parameter the model can set.

### Grounding strategy

- Every fact must trace to a tool result; answers cite `device_id` +
  `collected_at` (+ metric/value). Disk aggregates are GiB everywhere.
- **Evidence pool** = device ids *confirmed by tool results only* — never from
  tool-call args (a failed lookup proves the device was NOT found, but the id
  appears in the error text) and never from error/refusal replies. This closed
  a real hole where a cross-tenant device the tools never found could get an
  action approved.
- **Fact index + value check** — the verifier does not just check device
  membership: it builds a `(device_id, collected_at) → fields` index from the
  tool-result payloads and validates every evidence ref against it (value
  within 1%, timestamp exact, unknown metrics rejected, bare numbers must
  match an actual payload value or a count).
- Unsupported claims are surfaced in the answer under "NOT stated (insufficient"
  "evidence)" instead of being silently dropped or invented, and trigger one
  informed re-ground pass that names them.
- Read tools never fabricate: empty results come back as empty, and the
  analyst prompt requires saying so.

### Guardrails (Action Quality)

1. **Tenant isolation** — every query/action is scoped by injected tenant;
   cross-tenant devices are refused and audited; the QA suite checks no
   foreign device id ever appears in an answer.
2. **Propose-only action tools + single proposal path** — the LLM can create a
   *proposal*, never execute. Proposals come only from action tools called by
   the analyst on DB-verified evidence; execution requires a human decision
   that re-enters the graph as a structured `Command(resume=…)`. After
   approval a dedicated executor node flips the action `approved → executed`
   through a **simulated** connector (documented prototype scope) and writes
   the `execution` audit event.
3. **Evidence gate (deterministic)** — an action tool refuses on: missing
   evidence, evidence value mismatching the DB (1% numeric tolerance), unknown
   metric, cross-tenant device, or a failed per-action sufficiency rule. The
   HITL gate is pure (nothing writes before the interrupt, so a resume can
   never duplicate an audit row) and never asks the human to approve an action
   on a device no tool result ever confirmed.
4. **Audit trail** — `audit_log` is append-only by convention (no DB constraint): proposal → approved/refused,
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
  agent knows it sees a subset. 
- **Generic BYOK env (`FLEET_LLM_MODEL` / `LLM_API_KEY` / `LLM_BASE_URL`)**
  with a small routing table in one function (`get_llm`), unit-verified:
  OpenAI-compatible endpoints, native providers, and legacy Nous vars all
  resolve to the right client.

## Deliberately left out (prototype scope)

- **State persistence**: `data/fleet.duckdb` is rebuilt from the NDJSON on first
  run (and whenever an expected table is missing, e.g. after a schema change),
  which resets `actions` and `audit_log`. In production, keep telemetry and state
  in separate stores, or persist the DB volume and migrate additively.
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
- **Insights are prompt-driven, not statistical**: trend tools return
  aggregates, the analyst narrates; dedicated anomaly detection (e.g. z-score
  on disk burn rate) is out of scope for the prototype.
