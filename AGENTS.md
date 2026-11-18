# Fleet_Copilot

Take-home assessment "Senior Agentic AI Engineer" (Rayda) — Agentic Fleet Copilot su telemetria dispositivi. Obiettivo: agent LangGraph che risponde su fleet health/compliance, rileva insight/trend, propone azioni con guardrails.

## Contesto e vincoli (da assignment_docs/Take-Home Assessment — Senior Agentic AI Engineer (1).pdf)

- Time budget ~5 giorni. Stack: Python, LangGraph preferito, loop esplicito (planning + tool calling), NON un prompt wrapper.
- Valutazione: Agent Design 25%, Grounding & Correctness 20%, Action Quality & Guardrails 20%, Evaluation Rigor 20%, Engineering & Communication 15%.
- Deliverable: repo Git + README (setup, run, eval, architettura, tool catalog, grounding strategy, guardrails, trade-offs) + evaluation suite runnable, esecuzione ragionevolmente deterministica.

## Dati

- `device-telemetry-dataset.ndjson`: 750 snapshot, 3 companies, 25 devices, 30 giorni (2026-05).
- Ogni record: device_id, company_id, employee_id, collected_at, agent_version, os, device_identity, memory, disk_volumes, battery, compliance_results, installed_software, network.
- Strettamente locale: nessun identificatore reale fuori dal progetto.

## Capabilities richieste

1. Grounded QA (solo dai dati, risposte con evidenza citata e tracciabile).
2. Insight & trend (batterie EOL, RAM/disk costretti, compliance drift).
3. Action proposals via tool (create_upgrade_order, open_remediation_ticket, flag_device_for_replacement, notify_employee).
4. Guardrails: tenant isolation, no cross-company access, human approval esplicita, audit log, rifiuto senza evidenza sufficiente.

## Convenzioni

- Python 3.12+, venv locale (.venv), dipendenze in requirements.txt / pyproject.
- Evaluations in `evals/`, agent core in `agent/`, tools in `tools/`.
- Ogni risposta grounded cita device_id + collected_at come evidenza.
