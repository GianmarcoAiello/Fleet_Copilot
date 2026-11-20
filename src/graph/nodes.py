"""Nodes: Planner, Analyst (ReAct), Grounding Verifier, HITL gate, Formatter."""
from __future__ import annotations

import json
import os
import re

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from src.graph.state import AgentState, DraftClaim, EvidenceRef, ExecutionPlan, ProposedAction
from src.tools.read_tools import ALL_TOOLS, READ_TOOLS, set_repo

MAX_ITERATIONS = 6
TOOL_CALLS_PER_TURN = 4  # per-turn cap: bounds total calls to ~6×4 even if the model batches greedily

PLANNER_SYSTEM = """You are the planner of a Fleet Copilot for IT administrators.
Classify the request and emit a short execution plan.
Rules:
- intent: 'qa' (current-state question), 'trend_insight' (patterns over the 30-day window),
  'action_request' (user asks to create orders/tickets/flags/notifications), or 'mixed'.
- Steps must be concrete tool tasks ('query disk trends', 'filter devices with <20GB free').
- You NEVER create action proposals: actions are proposed by the analyst through
  action tools AFTER reading telemetry, and require human approval.
"""

ANALYST_SYSTEM = """You are the analyst of a Fleet Copilot. You answer ONLY from tool outputs.
Rules:
- Every fact in your answer must come from a tool call result in this conversation.
- Cite device_id and collected_at for each fact you state.
- Every insight you report MUST have three parts: the finding, the supporting
  evidence (device_id + collected_at + metric values), and a brief explanation of
  why the pattern matters (e.g. 'capacity down 15% in 30 days → battery near EOL').
- If tools return nothing relevant, say so — do not extrapolate.
- For action tools, ALWAYS attach telemetry evidence (device_id, collected_at,
  metric, value) — the metric MUST be a field of the latest snapshot row (e.g.
  battery_condition, disk_available_bytes, battery_cycle_count, compliance_fails)
  and values are re-checked against the DB: mismatched or invented evidence is refused.
- Actions also have sufficiency rules checked on the DB: a remediation ticket
  needs the cited check currently failing, a replacement flag needs battery
  'Service Recommended', an upgrade needs real RAM/disk pressure. If the rule
  fails the tool refuses — do not propose actions without reading telemetry first.
- Device snapshots include employee_id: use it with notify_employee when the user
  asks to contact or warn a device's user (e.g. unauthorized software found).
- If the user references a device outside your organization, the tools will refuse: report that.
Tool economy (hard budget):
- You have at most 6 turns and 4 tool calls per turn. Batch everything you need
  into as few calls as possible.
- For fleet-wide or filtered questions use query_fleet_snapshot (it supports
  min_avail_disk_gb, os_version_lt, check, software filters) — ONE call answers
  "which devices…" questions. Do NOT call query_device_history per device;
  history is only for drilling into a device the snapshot flagged.
- For 30-day patterns use analyze_time_series_trends — it is pre-computed.
- Stop calling tools as soon as you can answer; answer, don't explore.
"""

VERIFIER_SYSTEM = """You extract verifiable claims from the analyst's draft answer.
Return one claim per sentence containing a device_id or numeric metric, with the
evidence refs (device_id, collected_at, metric, value) that support it. Claims with
no supporting evidence must be listed as unsupported."""


NOUS_DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"

# LLM_API_KEY → the constructor kwarg each langchain provider expects
_PROVIDER_KEY_PARAM = {
    "anthropic": "anthropic_api_key",
    "google_genai": "google_api_key",
    "openai": "api_key",
}


def get_llm(temperature: float = 0.0):
    """BYOK: bring your own provider via three env vars (see .env.example):
      FLEET_LLM_MODEL  model id, provider-qualified: 'google_genai:gemini-2.0-flash',
                       'anthropic:claude-sonnet-4-5', 'openai/gpt-6-luna' (slash form
                       = OpenAI-compatible endpoint), or a bare id when LLM_BASE_URL
                       identifies the endpoint.
      LLM_API_KEY      your key for that provider.
      LLM_BASE_URL     optional OpenAI-compatible endpoint (Nous Portal, OpenRouter,
                       vLLM, Ollama…). Any model id is then served through it.
    Legacy NOUS_API_KEY / NOUS_BASE_URL are still honoured."""
    model = os.environ.get("FLEET_LLM_MODEL", "google_genai:gemini-flash-lite-latest")
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("NOUS_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("NOUS_BASE_URL")
    kwargs: dict = {}
    if base_url:
        # OpenAI-compatible endpoint: serve ANY model id through ChatOpenAI.
        # The full id (including a 'openai/' prefix) is the model name the
        # endpoint expects — Nous Portal ids are slash-style, do not strip it.
        kwargs["api_key"] = api_key
        kwargs["base_url"] = base_url
        model = "openai:" + model  # langchain wants 'provider:model'
    elif model.startswith("openai/") and api_key:
        # slash id without explicit base_url: a Nous Portal id keeps working
        kwargs["api_key"] = api_key
        kwargs["base_url"] = NOUS_DEFAULT_BASE_URL
        model = "openai:" + model
    elif api_key and ":" in model:
        # native provider: map the generic key to the constructor kwarg it expects
        provider = model.split(":", 1)[0]
        if param := _PROVIDER_KEY_PARAM.get(provider):
            kwargs[param] = api_key
    return init_chat_model(model, temperature=temperature, **kwargs)


# ── 1. Planner ──────────────────────────────────────────────────────────────

def planner(state: AgentState) -> dict:
    llm = get_llm().with_structured_output(ExecutionPlan, method="function_calling")
    plan = llm.invoke([SystemMessage(content=PLANNER_SYSTEM),
                       HumanMessage(content=state["query"])])
    return {"plan": plan, "iteration": 0, "tool_outputs": [], "messages": [
        HumanMessage(content=f"[request] {state['query']}\n[plan] {plan.model_dump_json()}")]}


# ── 2. Analyst (explicit ReAct loop) ───────────────────────────────────────

def analyst(state: AgentState) -> dict:
    # the state reducer is operator.add: return ONLY the delta, never the
    # history — re-emitting it duplicates every message every turn (and
    # duplicate tool_call ids are rejected by OpenAI/Anthropic APIs)
    history = state["messages"]
    call_msgs = [SystemMessage(content=ANALYST_SYSTEM), *history]
    # Gemini rejects requests ending on a tool/model turn; close the turn as user.
    if history and not isinstance(history[-1], HumanMessage):
        call_msgs.append(HumanMessage(
            content="Continue: use the tool results above, or call more tools, "
                    "then write your grounded answer with citations."))
    feedback = state.get("reground_feedback")
    if feedback:
        call_msgs.append(HumanMessage(content=feedback))
    iteration = state.get("iteration", 0)
    if iteration >= MAX_ITERATIONS:
        # budget exhausted: final forced answer WITHOUT tools bound — the model
        # literally cannot emit new tool calls, so the loop must terminate
        call_msgs.append(HumanMessage(
            content="TOOL BUDGET EXHAUSTED: do not call any tool. Write your final "
                    "grounded answer NOW from the evidence already collected, with "
                    "citations; if something was not retrieved, say so explicitly."))
        out = get_llm().invoke(call_msgs)
    else:
        out = get_llm().bind_tools(ALL_TOOLS).invoke(call_msgs)
    return {"messages": [out], "iteration": iteration + 1}


def analyst_tools(state: AgentState) -> dict:
    """Execute the analyst's pending tool calls, capture structured outputs."""
    from langchain_core.messages import ToolMessage
    last = state["messages"][-1]
    tool_map = {t.name: t for t in ALL_TOOLS}
    outputs, msgs = [], []
    calls = last.tool_calls
    if len(calls) > TOOL_CALLS_PER_TURN:
        # keep the first N, decline the rest with a proper tool result so the
        # API contract stays intact and the model learns the budget
        for call in calls[TOOL_CALLS_PER_TURN:]:
            msgs.append(ToolMessage(
                content="skipped: per-turn tool budget reached — fold this need "
                        "into fewer calls or use a fleet-wide tool",
                tool_call_id=call["id"]))
        calls = calls[:TOOL_CALLS_PER_TURN]
    for call in calls:
        tool = tool_map[call["name"]]
        try:
            result = tool.invoke({"args": call["args"], "id": call["id"],
                                  "type": "tool_call",
                                  **{"config": {"configurable": {"tenant_id": state["tenant_id"]}}}})
            outputs.append({"tool": call["name"], "args": call["args"],
                            "result": result.content})
            msgs.append(ToolMessage(content=result.content, tool_call_id=call["id"]))
        except Exception as exc:  # noqa: BLE001 — surface the failure to the model
            msgs.append(ToolMessage(content=f"tool error: {exc}", tool_call_id=call["id"]))
            outputs.append({"tool": call["name"], "args": call["args"], "result": f"error: {exc}"})
    return {"messages": msgs, "tool_outputs": outputs}


def analyst_router(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        # pending calls must always be executed (the APIs require their results);
        # the turn budget is enforced in analyst(), which stops binding tools
        return "execute"
    return "verify"


# ── 3. Grounding Verifier ───────────────────────────────────────────────────

DEVICE_ID_RE = re.compile(
    r"\b(?=[A-Z0-9]*[A-Z])[A-Z0-9]{10,14}\b")  # must contain a letter: keeps out byte counts


def _evidence_pool(tool_outputs: list[dict]) -> set[str]:
    """Device ids CONFIRMED by tool RESULTS. Deliberately excludes tool-call
    args and error/refusal replies: "device 'X' not found in your organization"
    mentions the id too, but a failed lookup proves the device was NOT found —
    counting it enabled approving actions for cross-tenant devices."""
    pool = set()
    for out in tool_outputs:
        text = str(out["result"])
        if text.startswith("error:") or '"error"' in text or '"refused"' in text:
            continue
        pool.update(DEVICE_ID_RE.findall(text))
    return pool


def _draft_text(message) -> str:
    """Gemini returns content as a list of blocks; normalise to plain text."""
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(
        b.get("text", "") for b in content if isinstance(b, dict) and "text" in b)


MAX_REGROUND = 1  # informed retry passes: the README promises "one more grounding pass"


def _norm_ts(ts) -> str:
    return str(ts).replace("T", " ").replace("Z", "").strip()[:19]


def _fact_index(tool_outputs: list[dict]) -> dict:
    """(device_id, collected_at) -> {field: value}, built from tool RESULT payloads.
    The deterministic fact store: evidence refs are checked against this, not
    against the model's own words."""
    idx: dict = {}

    def walk(obj):
        if isinstance(obj, dict):
            if obj.get("device_id") and obj.get("collected_at"):
                key = (str(obj["device_id"]), _norm_ts(obj["collected_at"]))
                bucket = idx.setdefault(key, {})
                for k, v in obj.items():
                    bucket.setdefault(k, v)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    aggs: dict = {}
    for out in tool_outputs:
        text = str(out["result"])
        if text.startswith("error:") or '"error"' in text or '"refused"' in text:
            continue
        try:
            data = json.loads(text)
        except Exception:
            continue
        walk(data)
        if isinstance(data, dict):
            for k in ("count", "total_matching"):
                if k in data:
                    aggs[k] = data[k]
    if aggs:
        idx[("__agg__", "")] = aggs
    return idx


def _claim_problems(c, idx: dict, pool: set[str]) -> list[str]:
    """Deterministic checks for one claim: device membership, then every
    EvidenceRef against the fact index (value + timestamp)."""
    from src.tools.read_tools import _values_match
    problems: list[str] = []
    ids = {e.device_id for e in c.evidence} | set(DEVICE_ID_RE.findall(c.claim))
    if ids and not ids.issubset(pool):
        problems.append(f"device(s) {sorted(ids - pool)} never confirmed by any tool result")
    for e in c.evidence:
        key = (e.device_id, _norm_ts(e.collected_at))
        bucket = idx.get(key)
        if bucket is None:
            # an invented timestamp must be rejected: the analyst cites
            # collected_at from tool results, which normalize identically
            problems.append(f"no snapshot fact found for {e.device_id}@{e.collected_at}")
            continue
        if e.metric and e.metric in bucket and e.value not in (None, ""):
            if not _values_match(e.value, bucket[e.metric]):
                problems.append(f"{e.device_id}@{e.collected_at} {e.metric}: claimed "
                                f"{e.value!r}, tool results say {bucket[e.metric]!r}")
    # numbers asserted with no evidence refs: acceptable only if some tool
    # aggregate (count/total) or fact value actually contains them
    if not c.evidence:
        nums = re.findall(r"\d+(?:\.\d+)?", c.claim)
        if nums:
            haystack = " ".join(str(v) for bucket in idx.values() for v in bucket.values())
            if not any(n in haystack for n in nums):
                problems.append(f"numeric claim without any evidence ref: {c.claim[:120]}")
    return problems


def verifier(state: AgentState) -> dict:
    draft = _draft_text(state["messages"][-1])
    claims = []
    try:
        llm = get_llm().with_structured_output(
            __import__("pydantic").RootModel[list[DraftClaim]], method="function_calling")
        claims = llm.invoke([SystemMessage(content=VERIFIER_SYSTEM),
                             HumanMessage(content=f"Draft:\n{draft}")]).root
    except Exception:  # noqa: BLE001 — claim extraction must never erase an answer
        claims = []
    if not claims and draft.strip():
        # fall back: surface the whole draft as a single unverified-but-sourced answer
        claims = [DraftClaim(claim=draft.strip(), evidence=[])]
    pool = _evidence_pool(state.get("tool_outputs", []))
    idx = _fact_index(state.get("tool_outputs", []))
    verified, unsupported = [], []
    for c in claims:
        if _claim_problems(c, idx, pool):
            unsupported.append(c.claim)
        else:
            verified.append(c)
    attempts = state.get("reground_attempts", 0)
    feedback = ""
    if unsupported and attempts < MAX_REGROUND:
        feedback = ("These claims are NOT supported by the tool results:\n"
                    + "\n".join(f"- {u}" for u in unsupported)
                    + "\nEither fetch the missing evidence with a tool call, or remove "
                      "these claims from your answer.")
    return {"verified_claims": verified, "unsupported": unsupported,
            "reground_feedback": feedback,
            "reground_attempts": attempts + (1 if unsupported else 0),
            "response": None}  # clear stale response from a previous loop


def verifier_router(state: AgentState) -> str:
    if _proposed_action_ids(state.get("tool_outputs", [])):
        return "hitl"
    if (state.get("unsupported") and state.get("iteration", 0) < MAX_ITERATIONS
            and state.get("reground_attempts", 0) <= MAX_REGROUND):
        return "re_analyze"
    return "respond"


# ── 4. HITL gate (single path) + executor ───────────────────────────────────

ACTION_TOOL_NAMES = {"create_upgrade_order", "open_remediation_ticket",
                     "flag_device_for_replacement", "notify_employee"}


def _proposed_action_ids(tool_outputs: list[dict]) -> list[str]:
    """Action ids created by the action tools DURING this run — the single
    proposal path. Planner-invented proposals do not exist anymore."""
    aids: list[str] = []
    for out in tool_outputs or []:
        if out.get("tool") not in ACTION_TOOL_NAMES:
            continue
        try:
            r = json.loads(str(out["result"]))
        except Exception:
            continue
        if r.get("proposed") and r.get("action_id"):
            if r["action_id"] not in aids:
                aids.append(r["action_id"])
    return aids


def _tenant_of(state: AgentState, config=None) -> str:
    # config is the single source of truth for the tenant (injected at invoke)
    t = (config or {}).get("configurable", {}).get("tenant_id")
    return t or state.get("tenant_id", "")


def hitl_gate(state: AgentState, config=None) -> dict:
    """Pure gate: nothing writes before the interrupt, so a resume can never
    duplicate an audit row. Proposals are read from the DB (source of truth)."""
    repo = _REPO_REF
    tenant = _tenant_of(state, config)
    aids = _proposed_action_ids(state.get("tool_outputs", []))
    proposals = repo.get_actions(tenant, aids) if (repo is not None and aids) else []
    if not proposals:
        # no action proposed (or all refused at tool level): no human ask
        return {"approval": None, "approved": False, "approved_action_ids": []}
    pending = [p for p in proposals if p["status"] == "proposed"]
    if pending:
        decision = interrupt({
            "proposed_actions": proposals,
            "message": "Approve these actions? Resume with Command(resume={'approved': bool, 'approver': str})",
        })
        approved = bool(decision.get("approved"))
        approver = str(decision.get("approver") or "human")
        # resolve_action is idempotent (status='proposed' guard): resolving here
        # and from the web UI never double-audits
        for p in pending:
            repo.resolve_action(tenant, p["action_id"], approved, approver)
    else:
        # the decision was already taken elsewhere (web UI Pending tab):
        # reflect it so the executor still runs on approval
        approved = all(p["status"] == "approved" for p in proposals)
        decision = {"approved": approved, "approver": "pre-resolved"}
    return {"approval": decision, "approved": approved,
            "approved_action_ids": [p["action_id"] for p in proposals
                                    if approved and p["status"] in ("approved", "executed")]}


def hitl_router(state: AgentState) -> str:
    return "execute" if state.get("approved") else "respond"


def executor(state: AgentState, config=None) -> dict:
    """approved -> executed, via a SIMULATED connector (documented prototype
    scope: no real ticketing/email). Writes the `execution` audit event."""
    repo = _REPO_REF
    tenant = _tenant_of(state, config)
    executed = []
    if repo is not None:
        for aid in state.get("approved_action_ids", []):
            if repo.execute_action(tenant, aid):
                executed.append(aid)
    return {"executed_action_ids": executed}


# ── 5. Formatter ────────────────────────────────────────────────────────────

def formatter(state: AgentState) -> dict:
    parts = []
    if state.get("verified_claims"):
        lines = []
        for c in state["verified_claims"]:
            ev = "; ".join(f"{e.device_id}@{e.collected_at} {e.metric}={e.value}"
                           for e in c.evidence) or "(derived from tool outputs)"
            lines.append(f"- {c.claim}\n  evidence: {ev}")
        parts.append("\n".join(lines))
    if state.get("unsupported"):
        parts.append("NOT stated (insufficient evidence): " + "; ".join(state["unsupported"]))
    if state.get("approval") is not None:
        parts.append(f"HITL decision: {'APPROVED' if state.get('approved') else 'REJECTED'} "
                     f"by {state.get('approval', {}).get('approver', 'unknown')}")
    if state.get("executed_action_ids"):
        parts.append(f"Executed via simulated connector: {', '.join(state['executed_action_ids'])}")
    parts.append(f"(plan intent: {state['plan'].intent}; tool calls: "
                 f"{len(state.get('tool_outputs', []))})")
    return {"response": "\n\n".join(parts)}


def build_workflow(repo, checkpointer=None):
    """Compile the StateGraph. `repo` must be set before invocation."""
    global _REPO_REF
    _REPO_REF = repo
    from langgraph.graph import END, StateGraph
    set_repo(repo)
    g = StateGraph(AgentState)
    g.add_node("planner", planner)
    g.add_node("analyst", analyst)
    g.add_node("execute_tools", analyst_tools)
    g.add_node("verifier", verifier)
    g.add_node("hitl", hitl_gate)
    g.add_node("executor", executor)
    g.add_node("formatter", formatter)
    g.set_entry_point("planner")
    g.add_edge("planner", "analyst")
    g.add_conditional_edges("analyst", analyst_router,
                            {"execute": "execute_tools", "verify": "verifier"})
    g.add_edge("execute_tools", "analyst")
    g.add_conditional_edges("verifier", verifier_router,
                            {"hitl": "hitl", "respond": "formatter",
                             "re_analyze": "analyst"})
    g.add_conditional_edges("hitl", hitl_router,
                            {"execute": "executor", "respond": "formatter"})
    g.add_edge("executor", "formatter")
    g.add_edge("formatter", END)
    return g.compile(checkpointer=checkpointer)
