#!/usr/bin/env python3
"""Pending-action manager: list / approve / reject actions awaiting HITL decision.

CLI:  python actions_cli.py list|approve <action_id>|reject <action_id> [--tenant T]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.db.loader import Repository  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "approve", "reject"])
    ap.add_argument("action_id", nargs="?")
    ap.add_argument("--tenant", default="acme-001")
    ap.add_argument("--approver", default="terminal-user")
    args = ap.parse_args()
    repo = Repository(ROOT / "data" / "fleet.duckdb")
    if args.cmd == "list":
        rows = repo.con.execute(
            "SELECT action_id, tool, device_id, args_json, created_at FROM actions "
            "WHERE company_id=? AND status='proposed' ORDER BY created_at DESC",
            (args.tenant,)).fetchall()
        if not rows:
            print(f"no pending actions for {args.tenant}")
        for aid, tool, dev, args_json, created in rows:
            print(f"  {aid}  {tool}  device={dev}  args={args_json}  at={created}")
    else:
        ok = repo.resolve_action(args.tenant, args.action_id,
                                 approved=(args.cmd == "approve"), approver=args.approver)
        print(f"{args.cmd}: {'OK' if ok else 'FAILED (not found, already resolved, or wrong tenant)'}")
    repo.close()


if __name__ == "__main__":
    main()
