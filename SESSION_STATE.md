# Fleet_Copilot — stato sessione (2026-11-20)

## Architettura implementata e verificata
- **Graph** (`src/graph/nodes.py`): Planner → Analyst (ReAct, max 6 iter) → Verifier
  (matching deterministico evidenza/claim, regex device_id 12 char) → HITL gate
  (`interrupt()`/`Command(resume)`) → Formatter. LLM BYOK via `init_chat_model`
  (`FLEET_LLM_MODEL=google_genai:gemini-flash-lite-latest`).
- **Repository** (`src/db/loader.py`): DuckDB tenant-scoped, tabelle telemetry +
  actions + audit_log append-only. `software_events` con join employee_id.
- **Tools** (`src/tools/read_tools.py`): 4 read + 4 action tools con schemi Pydantic
  tipizzati (component/spec, check_id/note, reason, message). Tenant via InjectedToolArg.
- **Web UI** (`webui.py`, stdlib, :8080): design Rayda, SSE pipeline live
  (Planner→Analyst→Tools→Verifier→HITL→Formatter), ctx per-card (no ID globali),
  Approve/Reject in chat e in Pending (resume server-side del thread parcheggiato).
- **Evals** (`evals/run_evals.py`): --mock 7 deterministici + --live 7 casi. 7/7.
- **CLI** (`cli.py` interattiva HITL, `actions_cli.py` pending manager).

## Gotchas risolti (non reintrodurli)
- DuckDB è single-process per file: chiudere CLI prima di web UI/evals.
- `_get_wf()` deve cachare il grafo: ricompilare orfanizza i thread parcheggiati.
- LangGraph: `task.interrupts` (non `task.interrupt`); Gemini rifiuta turni che
  finiscono con tool/model → chiudere con HumanMessage.
- Gemini quota free: 3-6 LLM call per domanda → si esaurisce in fretta.
- device_id sono 12 caratteri (regex `[A-Z0-9]{10,14}`).
- Chiave Gemini formato nuovo "AQ." funziona solo via Bearer/`?key=` (non x-goog).

## TODO rimasti
1. README finale (architettura, tool catalog, grounding strategy, guardrails, trade-offs).
2. Documentare executor post-approval simulato (approved/rejected è stato terminale).
3. Possibile micro web-UI per insight explanation — il prompt Analyst già la impone.

## Accesso remoto
- Telegram: profilo `fleet_copilot` (bot dedicato, token nel suo .env), cwd progetto.
- Web UI: `webui.py 8080` + Tailscale/SSH tunnel per remoto.
