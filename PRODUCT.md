# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary audience: **assessment reviewers** (senior engineers at Rayda grading
the "Senior Agentic AI Engineer" take-home). They read the UI to judge agent
design, grounding, action quality/guardrails, evaluation rigor and engineering
craft. Their success: understanding the architecture, the evidence trail and
the guardrail behavior quickly, without reading source code first.

Secondary (implicit, not designed-for): an IT administrator persona — the role
the agent serves (fleet health, compliance, trends, remediation actions across
three synthetic tenants: acme-001, globex-002, initech-003).

## Product Purpose

Demonstrate an agentic Fleet Copilot over device telemetry: grounded QA,
trend insights, and propose-only remediation actions gated by explicit human
approval (HITL), with a complete audit trail. Success = a reviewer can verify
every claim's evidence and every guardrail directly from the UI.

## Positioning

Not a chat wrapper: an explicit LangGraph pipeline (planner → ReAct analyst →
grounding verifier → HITL gate → formatter) where grounding, guardrails and
auditability are inspectable first-class steps. The UI makes the evidence
trail — tool calls, cited telemetry (device_id + collected_at), HITL banner
with action effects, audit log — the main character.

## Operating Context

- Take-home assessment context; evaluated against the published rubric
  (Agent Design 25%, Grounding & Correctness 20%, Action Quality & Guardrails
  20%, Evaluation Rigor 20%, Engineering & Communication 15%).
- Runs strictly locally: stdlib HTTP server (`webui.py <port>`, default 8080),
  DuckDB single-file (`data/fleet.duckdb`, rebuilt from the NDJSON dataset).
- BYOK LLM: provider configured via env (`FLEET_LLM_MODEL` / `LLM_API_KEY` /
  `LLM_BASE_URL`), see README "Try it with your own key".
- Desktop-only usage: the IT admin works from a console; mobile is not a
  design target (it should degrade without breaking, nothing more).

## Capabilities and Constraints

- Copilot tab: query → live pipeline visualization (planner → analyst →
  verifier → HITL → formatter), streaming tool trace, cited answer.
- HITL gate: proposed actions pause the graph; the banner must state what the
  action does, on which device, with which cited evidence, and the consequence
  of approve/reject.
- Pending actions tab: approve/reject parked proposals (resolved server-side).
- Audit log tab: append-only trail (proposal → approved/refused, args,
  evidence, approver).
- Tenant selector: acme-001 / globex-002 / initech-003; cross-tenant data
  must never appear (guardrail demo surface).
- Constraints: stdlib HTTP + SSE only (no web framework, no build step);
  DuckDB is single-writer (web UI or CLI, not both); no web auth (binds to
  127.0.0.1, tenant is a demo selector, not a security boundary).
- Evaluation assets: `evals/run_evals.py` (deterministic), `evals/run_qa100.py`
  (live 100-question suite); results are part of the credibility story.

## Brand Commitments

- Name: **Fleet Copilot** (◈ logo mark).
- Binding visual world: dark navy theme inspired by Rayda's design system —
  near-black/navy background, Inter for UI text, monospace for tool traces,
  blue primary (#0C66FF family), green/amber/red semantic chips. This is a
  deliberate constraint, not incidental; refinements stay inside this world.
- Voice: precise, technical, evidence-first; no marketing fluff in the UI.

## Evidence on Hand

- `device-telemetry-dataset.ndjson`: 750 snapshots, 25 devices, 3 tenants,
  30 days (2026-05-14 → 06-12) — the only data the UI may ever show.
- `evals/qa100_results.json`: first 10 live eval results (8/10 pass) — real
  numbers, never embellished.
- README.md documents architecture, grounding strategy, guardrails,
  deliberately-left-out scope.
- No testimonials, no customer logos, no deployment claims: none exist and
  none may be invented.

## Product Principles

1. **Evidence is the interface** — every claim, action and decision shows its
   telemetry proof inline; a reviewer never has to trust prose.
2. **Guardrails visible, not hidden** — refusals, evidence gates, HITL pauses
   and audit entries are first-class UI moments, exactly where they happen.
3. **Inspectability over decoration** — reviewer legibility outranks visual
   flourish; polish lives in precision, not ornament.
4. **Honest scope** — prototype limits are stated in the product, not papered
   over.
