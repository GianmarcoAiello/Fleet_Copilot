#!/usr/bin/env python3
"""First E2E run: QA question through the full graph (no actions -> no HITL pause)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).parent / ".env")

from src.db.loader import Repository  # noqa: E402
from src.graph.nodes import build_workflow  # noqa: E402


def main() -> None:
    repo = Repository(Path(__file__).parent / "data" / "fleet.duckdb")
    wf = build_workflow(repo)
    query = sys.argv[1] if len(sys.argv) > 1 else \
        "Which devices at my company are low on disk space and how bad is it?"
    tenant = sys.argv[2] if len(sys.argv) > 2 else "acme-001"
    result = wf.invoke(
        {"query": query, "tenant_id": tenant},
        config={"configurable": {"tenant_id": tenant}},
    )
    print("=== PLAN ===")
    print(result["plan"].model_dump_json(indent=1))
    print("=== TOOL CALLS ===")
    for t in result.get("tool_outputs", []):
        print(f"- {t['tool']}({json.dumps(t['args'])[:80]}) -> {str(t['result'])[:100]}")
    print("=== RESPONSE ===")
    print(result["response"])


if __name__ == "__main__":
    main()
