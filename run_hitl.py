#!/usr/bin/env python3
"""HITL demo: action request -> interrupt() -> resume with approve/reject.
Uses thread checkpointing (MemorySaver) so the same thread_id resumes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).parent / ".env")

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from src.db.loader import Repository  # noqa: E402
from src.graph.nodes import build_workflow  # noqa: E402


def main() -> None:
    repo = Repository(Path(__file__).parent / "data" / "fleet.duckdb")
    checkpointer = MemorySaver()
    wf = build_workflow(repo, checkpointer=checkpointer)
    query = sys.argv[1] if len(sys.argv) > 1 else (
        "Battery on device 8NM23J95R5I6 is failing. "
        "Create an upgrade order for a new battery and flag the device for replacement."
    )
    tenant = sys.argv[2] if len(sys.argv) > 2 else "acme-001"
    thread = {"configurable": {"thread_id": "hitl-demo-1", "tenant_id": tenant}}
    state = wf.invoke({"query": query, "tenant_id": tenant}, config=thread)
    if "__interrupt__" in state:
        info = state["__interrupt__"][0]
        payload = info.value if hasattr(info, "value") else info
        print("=== INTERRUPTED: HITL gate ===")
        print(json.dumps(payload, indent=1, default=str)[:1500])
        decision = sys.argv[3] if len(sys.argv) > 3 else "approve"
        approved = decision != "reject"
        result = wf.invoke(Command(resume={
            "approved": approved,
            "approver": "gianmarco-test",
        }), config=thread)
        print(f"=== RESUMED ({decision}) ===")
        print(result.get("response", result))
        print("=== AUDIT TAIL ===")
        for e in repo.audit_tail(tenant, limit=5):
            print(f"- {e['event']} {e['tool']} {e['detail']}")
    else:
        print("=== NO INTERRUPT (no actions proposed) ===")
        print(state.get("response", state))


if __name__ == "__main__":
    main()
