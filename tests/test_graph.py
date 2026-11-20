"""Deterministic graph tests with a stub LLM — no API key, no network.

Covers the findings of the external review: linear message growth (reducer
bug), value/timestamp-grounded verifier, invented evidence refused at tool
level with a single audit row, tenant isolation, and get_llm env routing.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

import src.graph.nodes as nodes  # noqa: E402
from src.db.loader import Repository, load_ndjson  # noqa: E402
from src.tools.read_tools import set_repo  # noqa: E402

NDJSON = ROOT / "device-telemetry-dataset.ndjson"


# ── stubs ───────────────────────────────────────────────────────────────────

class FakeStructured:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, *_a, **_k):
        return self.payload


class FakeLLM:
    """Stub chat model: returns scripted AIMessages and records call sizes."""

    verifier_claims: list = []
    structured_payload: object = None

    def __init__(self, script):
        self.script = list(script)  # list of (tool_calls|None, content)
        self.calls = []             # len(call_msgs) per invocation
        self.calls_seen = []

    def invoke(self, messages, *_a, **_k):
        self.calls.append(len(messages))
        self.calls_seen.append([m.id for m in messages if getattr(m, "id", None)])
        script = self.script.pop(0) if self.script else (None, "done")
        tool_calls, content = script
        msg = AIMessage(content=content or "")
        if tool_calls:
            msg.tool_calls = tool_calls
        return msg

    def bind_tools(self, *_a, **_k):
        return self

    def with_structured_output(self, schema, method=None):
        from src.graph.state import DraftClaim
        if "RootModel" in str(schema):
            claims = [DraftClaim(**c) if isinstance(c, dict) else c
                      for c in self.verifier_claims]
            return FakeStructured(type("R", (), {"root": claims}))
        return FakeStructured(self.structured_payload)


@pytest.fixture
def repo():
    tmp = Path(tempfile.mkdtemp()) / "t.duckdb"
    load_ndjson(tmp, NDJSON)
    r = Repository(tmp)
    set_repo(r)
    nodes._REPO_REF = r
    yield r


# ── 1. reducer: linear message growth, no duplicated tool_call ids ─────────

def test_messages_grow_linearly_and_ids_unique(repo, monkeypatch):
    dev = "MT7PJB7N5LRE"
    row = repo.latest_snapshot_row("acme-001", dev)
    val = row["disk_available_bytes"]
    fake = FakeLLM(script=[
        ([{"name": "query_fleet_snapshot", "args": {}, "id": "call_1"}], ""),
        (None, f"{dev} has {val} bytes free."),
    ])
    fake.verifier_claims = [{"claim": f"{dev} has {val} bytes free", "evidence": [
        {"device_id": dev, "collected_at": str(row["collected_at"])[:19],
         "metric": "disk_available_bytes", "value": str(val)}]}]
    fake.structured_payload = nodes.ExecutionPlan(intent="qa", steps=["x"])
    monkeypatch.setattr(nodes, "get_llm", lambda temperature=0.0: fake)

    wf = nodes.build_workflow(repo, checkpointer=None)
    state = wf.invoke({"query": "low disk?", "tenant_id": "acme-001"},
                      config={"configurable": {"thread_id": "t1", "tenant_id": "acme-001"}})

    # analyst saw exactly [2, 4]: System+Human, then +AI(tool)+Tool — linear
    assert fake.calls == [2, 5], f"non-linear growth: {fake.calls}"  # 2=Sys+Human; 5=+AI+Tool+closer
    # every tool_call id appears once in the final message history
    ids = [m.tool_call_id for m in state["messages"] if getattr(m, "tool_call_id", None)]
    assert len(ids) == len(set(ids)) == 1
    assert state.get("executed_action_ids", []) == []


# ── 2. verifier: value, timestamp and ungrounded aggregates are rejected ────

def _verifier_state(monkeypatch, tool_outputs, claims):
    class C:  # minimal DraftClaim stand-in
        def __init__(self, claim, evidence):
            self.claim, self.evidence = claim, evidence
    from src.graph.state import EvidenceRef
    parsed = [C(c["claim"], [EvidenceRef(**e) for e in c.get("evidence", [])])
              for c in claims]
    fake = FakeLLM([])
    fake.verifier_claims = [{"claim": p.claim,
                             "evidence": [e.model_dump() for e in p.evidence]}
                            for p in parsed]
    fake.structured_payload = nodes.ExecutionPlan(intent="qa", steps=[])
    monkeypatch.setattr(nodes, "get_llm", lambda temperature=0.0: fake)
    return {"messages": [AIMessage(content="draft")], "tool_outputs": tool_outputs,
            "tenant_id": "acme-001"}


def test_verifier_rejects_wrong_value_and_invented_timestamp(repo, monkeypatch):
    outs = [{"tool": "query_fleet_snapshot", "args": {}, "result": json.dumps({
        "count": 1, "rows": [{"device_id": "MT7PJB7N5LRE",
                              "collected_at": "2026-06-12T08:39:00Z",
                              "disk_available_bytes": 10995116278}]})}]
    st = _verifier_state(monkeypatch, outs, [
        {"claim": "device has 99 GB free", "evidence": [
            {"device_id": "MT7PJB7N5LRE", "collected_at": "2026-06-12T08:39:00Z",
             "metric": "disk_available_bytes", "value": "99"}]},
        {"claim": "device reported in 1999", "evidence": [
            {"device_id": "MT7PJB7N5LRE", "collected_at": "1999-01-01T00:00:00Z",
             "metric": "disk_available_bytes", "value": "10995116278"}]},
        {"claim": "average disk usage 97%", "evidence": []},
        {"claim": "device has 2 GB free", "evidence": [
            {"device_id": "MT7PJB7N5LRE", "collected_at": "2026-06-12T08:39:00Z",
             "metric": "free_gb", "value": "2"}]},
    ])
    out = nodes.verifier(st)
    assert len(out["unsupported"]) == 4, out
    assert out["reground_attempts"] == 1
    assert "NOT supported" in out["reground_feedback"]


def test_verifier_accepts_matching_value(repo, monkeypatch):
    outs = [{"tool": "query_fleet_snapshot", "args": {}, "result": json.dumps({
        "count": 1, "rows": [{"device_id": "MT7PJB7N5LRE",
                              "collected_at": "2026-06-12T08:39:00Z",
                              "disk_available_bytes": 10995116278}]})}]
    st = _verifier_state(monkeypatch, outs, [
        {"claim": "device has 10995116278 bytes free", "evidence": [
            {"device_id": "MT7PJB7N5LRE", "collected_at": "2026-06-12 08:39:00",
             "metric": "disk_available_bytes", "value": "10995116278"}]}])
    out = nodes.verifier(st)
    assert len(out["verified_claims"]) == 1 and not out["unsupported"]


# ── 3. invented evidence refused, audit written exactly once ────────────────

def test_action_tool_refuses_invented_value_and_audits_once(repo):
    from src.tools.read_tools import flag_device_for_replacement
    cfg = {"configurable": {"tenant_id": "acme-001"}}
    before = len(repo.audit_tail("acme-001", limit=100))
    r = flag_device_for_replacement.invoke(
        {"device_id": "MT7PJB7N5LRE", "reason": "x",
         "evidence": [{"metric": "disk_available_bytes", "value": "99"}]},
        config=cfg)
    assert json.loads(r)["refused"]
    after = repo.audit_tail("acme-001", limit=100)
    assert len(after) == before + 1  # exactly ONE refusal row
    assert "mismatch" in after[0]["detail"]


def test_resolve_action_idempotent_single_audit(repo):
    from src.tools.read_tools import flag_device_for_replacement
    cfg = {"configurable": {"tenant_id": "acme-001"}}
    # pick ANY acme device that meets the sufficiency rule dynamically
    dev = next(d for d, r in [(x["device_id"], x) for x in repo.latest_snapshots("acme-001")]
               if r.get("battery_condition") == "Service Recommended")
    r = json.loads(flag_device_for_replacement.invoke(
        {"device_id": dev, "reason": "x",
         "evidence": [{"metric": "battery_condition", "value": "Service Recommended"}]},
        config=cfg))
    assert r["proposed"]
    aid = r["action_id"]
    tail0 = [e["event"] for e in repo.audit_tail("acme-001", limit=100)]
    assert repo.resolve_action("acme-001", aid, True, "tester")
    assert not repo.resolve_action("acme-001", aid, True, "tester")  # second: no-op
    tail1 = [e["event"] for e in repo.audit_tail("acme-001", limit=100)]
    assert tail1.count("approved") - tail0.count("approved") == 1
    # executor flips approved -> executed with an `execution` audit event
    assert repo.execute_action("acme-001", aid)
    assert "execution" in [e["event"] for e in repo.audit_tail("acme-001", limit=100)]


def test_graph_executor_runs_on_approval(repo, monkeypatch):
    """Full graph path: analyst calls the action tool -> interrupt -> resume
    approve -> executor flips the action to `executed` + `execution` audit."""
    from langgraph.types import Command
    dev, row = next((x["device_id"], x) for x in repo.latest_snapshots("acme-001")
                    if x.get("battery_condition") == "Service Recommended")
    fake = FakeLLM(script=[
        ([{"name": "flag_device_for_replacement",
           "args": {"device_id": dev, "reason": "battery degraded",
                    "evidence": [{"metric": "battery_condition",
                                  "value": "Service Recommended"}]},
           "id": "call_1"}], ""),
        (None, f"Proposed replacement flag for {dev}."),
    ])
    fake.verifier_claims = [{"claim": f"{dev} flagged, awaiting approval", "evidence": []}]
    fake.structured_payload = nodes.ExecutionPlan(intent="action_request", steps=["x"])
    monkeypatch.setattr(nodes, "get_llm", lambda temperature=0.0: fake)

    wf = nodes.build_workflow(repo, checkpointer=MemorySaver())
    thread = {"configurable": {"thread_id": "t-exec", "tenant_id": "acme-001"}}
    state = wf.invoke({"query": f"Flag {dev} for replacement", "tenant_id": "acme-001"},
                      config=thread)
    assert "__interrupt__" in state, "expected HITL interrupt"
    state = wf.invoke(Command(resume={"approved": True, "approver": "tester"}), config=thread)
    assert state.get("approved") is True
    assert state.get("executed_action_ids"), "executor did not run"
    events = [e["event"] for e in repo.audit_tail("acme-001", limit=20)]
    assert events.count("execution") == 1
    status = repo.con.execute("SELECT status FROM actions WHERE action_id=?",
                              (state["executed_action_ids"][0],)).fetchone()[0]
    assert status == "executed"


def test_repo_bootstraps_from_empty_path(tmp_path):
    """Clean-clone convenience: a missing/empty DB file is rebuilt from the NDJSON."""
    p = tmp_path / "fresh.duckdb"
    r = Repository(p)
    n = r.con.execute("SELECT count(*) FROM snapshots").fetchone()[0]
    assert n == 750
    assert r.latest_snapshot_row("acme-001", "MT7PJB7N5LRE") is not None


# ── tenant isolation + config as single source of truth ─────────────────────

def test_cross_tenant_read_and_action_refused(repo):
    from src.tools.read_tools import query_device_history, notify_employee
    cfg = {"configurable": {"tenant_id": "acme-001"}}
    r = query_device_history.invoke({"device_id": "CUXW0BQSV1B7"}, config=cfg)  # globex device
    assert "not found" in r
    r2 = notify_employee.invoke({"device_id": "CUXW0BQSV1B7", "message": "hi",
                                 "evidence": [{"metric": "compliance_fails", "value": 1}]},
                                config=cfg)
    assert json.loads(r2)["refused"]


def test_state_and_config_tenant_disagree_config_wins(repo, monkeypatch):
    # the gate reads the tenant from RunnableConfig, never from state
    monkeypatch.setattr(nodes, "interrupt",
                        lambda payload: (_ for _ in ()).throw(RuntimeError("no interrupt expected")))
    state = {"tool_outputs": [], "tenant_id": "globex-002"}
    out = nodes.hitl_gate(state, {"configurable": {"tenant_id": "acme-001"}})
    assert out["approved"] is False and out["approval"] is None


# ── get_llm env routing ──────────────────────────────────────────────────────

def test_get_llm_routing(monkeypatch):
    import src.graph.nodes as n
    cases = [
        ({"FLEET_LLM_MODEL": "openai/gpt-6-luna", "LLM_API_KEY": "k",
          "LLM_BASE_URL": "http://x/v1"}, "ChatOpenAI", "http://x/v1"),
        ({"FLEET_LLM_MODEL": "openai/gpt-6-luna", "LLM_API_KEY": "k"},
         "ChatOpenAI", n.NOUS_DEFAULT_BASE_URL),
        ({"FLEET_LLM_MODEL": "anthropic:claude-sonnet-4-5", "LLM_API_KEY": "k"},
         "ChatAnthropic", None),
        ({"FLEET_LLM_MODEL": "google_genai:gemini-2.0-flash", "LLM_API_KEY": "k"},
         "ChatGoogleGenerativeAI", None),
        ({"FLEET_LLM_MODEL": "openai/gpt-6-luna", "NOUS_API_KEY": "k",
          "NOUS_BASE_URL": "http://y/v1"}, "ChatOpenAI", "http://y/v1"),
    ]
    for env, cls, base in cases:
        for k in ("FLEET_LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL",
                  "NOUS_API_KEY", "NOUS_BASE_URL", "GOOGLE_API_KEY"):
            os.environ.pop(k, None)
        os.environ.update(env)
        llm = n.get_llm()
        assert type(llm).__name__ == cls, env
        url = getattr(getattr(llm, "root_client", None), "base_url", None)
        assert (str(url).rstrip("/") if url else None) == (base.rstrip("/") if base else None), env
