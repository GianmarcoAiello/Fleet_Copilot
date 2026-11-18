#!/usr/bin/env python3
"""Interactive Fleet Copilot CLI with HITL approval loop.

Usage:
  python cli.py [--tenant acme-001] [--auto-approve] [--query "..."]

--auto-approve: HITL decisions are approved automatically (for demos/evals).
Without it, every action proposal pauses and asks y/n on the terminal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from src.db.loader import Repository  # noqa: E402
from src.graph.nodes import build_workflow  # noqa: E402

COMPANIES = {
    "1": "acme-001",
    "2": "globex-002",
    "3": "initech-003",
}


def pick_tenant(default: str) -> str:
    print("Companies: [1] acme-001  [2] globex-002  [3] initech-003")
    choice = input(f"Tenant [{default}]: ").strip()
    return COMPANIES.get(choice, default if choice == "" else choice)


def handle_interrupt(wf, state: dict, thread: dict, auto: bool, repo: Repository) -> dict:
    info = state["__interrupt__"][0]
    payload = info.value if hasattr(info, "value") else info
    print("\n" + "═" * 60)
    print("HITL GATE — actions awaiting approval:")
    for a in payload.get("proposed_actions", []):
        print(f"  • {a['tool']} on {a['device_id']}")
        print(f"    rationale: {a.get('rationale', '')}")
        for e in a.get("evidence", []):
            print(f"    evidence: {e.get('device_id')}@{e.get('collected_at')} "
                  f"{e.get('metric')}={e.get('value')}")
    if auto:
        approved, approver = True, "auto-approver"
        print("  → auto-approved (--auto-approve)")
    else:
        ans = input("Approve all? [y/N/i(nspect audit)]: ").strip().lower()
        if ans == "i":
            for e in repo.audit_tail(thread["configurable"]["tenant_id"], limit=8):
                print(f"    audit: {e['event']} {e['tool']} {e['detail']}")
            ans = input("Approve all? [y/N]: ").strip().lower()
        approved, approver = ans.startswith("y"), "terminal-user"
    return wf.invoke(Command(resume={"approved": approved, "approver": approver}),
                     config=thread)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fleet Copilot interactive CLI")
    ap.add_argument("--tenant", default="acme-001")
    ap.add_argument("--auto-approve", action="store_true")
    ap.add_argument("--query", default=None, help="single query, then exit")
    args = ap.parse_args()

    repo = Repository(ROOT / "data" / "fleet.duckdb")
    wf = build_workflow(repo, checkpointer=MemorySaver())
    tenant = args.tenant
    print("Fleet Copilot — agentic fleet assistant (Ctrl-D to exit)\n")

    queries = [args.query] if args.query else None
    n = 0
    while True:
        if queries:
            if n >= len(queries):
                break
            q = queries[n]; n += 1
        else:
            try:
                q = input("\n❯ ").strip()
            except EOFError:
                break
            if not q:
                continue
        tenant = pick_tenant(tenant) if not args.query and not queries else tenant
        thread = {"configurable": {"thread_id": f"cli-{n}", "tenant_id": tenant}}
        try:
            state = wf.invoke({"query": q, "tenant_id": tenant}, config=thread)
            while "__interrupt__" in state:
                state = handle_interrupt(wf, state, thread, args.auto_approve, repo)
            print("\n" + "─" * 60)
            print(state.get("response", "(no response)"))
        except KeyboardInterrupt:
            print("\n(interrupted)")
        except Exception as exc:  # noqa: BLE001
            print(f"\nerror: {exc}")
    repo.close()


if __name__ == "__main__":
    main()
