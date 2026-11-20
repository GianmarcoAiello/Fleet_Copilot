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

PLANNER_SYSTEM = """You are the planner of a Fleet Copilot for IT administrators.
Classify the request and emit a short execution plan.
Rules:
- intent: 'qa' (current-state question), 'trend_insight' (patterns over the 30-day window),
  'action_request' (user asks to create orders/tickets/flags/notifications), or 'mixed'.
- Only include action_requests when the user EXPLICITLY asks to perform an action.
- Steps must be concrete tool tasks ('query disk trends', 'filter devices with <20GB free').
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
  metric, value) — an action without evidence will be refused.
- Device snapshots include employee_id: use it with notify_employee when the user
  asks to contact or warn a device's user (e.g. unauthorized software found).
- If the user references a device outside your organization, the tools will refuse: report that.
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
    llm = get_llm().bind_tools(ALL_TOOLS)
    messages: list[AnyMessage] = state["messages"]
    if len(messages) == 1:
        messages = messages + [SystemMessage(content=ANALYST_SYSTEM)]
    # Gemini rejects requests ending on a tool/model turn; close the turn as user.
    if messages and not isinstance(messages[-1], HumanMessage):
        messages = messages + [HumanMessage(
            content="Continue: use the tool results above, or call more tools, "
                    "then write your grounded answer with citations.")]
    iteration = state.get("iteration", 0)
    out = llm.invoke(messages)
    return {"messages": messages + [out], "iteration": iteration + 1}


def analyst_tools(state: AgentState) -> dict:
    """Execute the analyst's pending tool calls, capture structured outputs."""
    from langchain_core.messages import ToolMessage
    last = state["messages"][-1]
    tool_map = {t.name: t for t in ALL_TOOLS}
    outputs, msgs = [], []
    for call in last.tool_calls:
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
    if getattr(last, "tool_calls", None) and state.get("iteration", 0) < MAX_ITERATIONS:
        return "execute"
    if last.content and any(isinstance(getattr(last, "tool_calls", None), list) for _ in [0]) is False:
        pass
    return "verify" if not getattr(last, "tool_calls", None) else "execute"


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
    verified, unsupported = [], []
    for c in claims:
        ids = {e.device_id for e in c.evidence} | set(
            DEVICE_ID_RE.findall(c.claim))
        if not ids or ids.issubset(pool):
            verified.append(c)
        else:
            unsupported.append(c.claim)
    return {"verified_claims": verified, "unsupported": unsupported,
            "response": None}  # clear stale response from a previous loop


def verifier_router(state: AgentState) -> str:
    plan: ExecutionPlan = state["plan"]
    wants_actions = bool(plan.action_requests)
    if wants_actions:
        return "hitl"
    if state.get("unsupported") and state.get("iteration", 0) < MAX_ITERATIONS:
        # give the analyst one more pass to ground or drop the unsupported claims
        return "re_analyze"
    return "respond"


# ── 4. HITL gate ────────────────────────────────────────────────────────────

def hitl_gate(state: AgentState) -> dict:
    plan: ExecutionPlan = state["plan"]
    pool = _evidence_pool(state.get("tool_outputs", []))
    proposals = []
    for act in plan.action_requests:
        ok = all(e.device_id in pool for e in act.evidence)
        # a device never confirmed by any tool result (not found / cross-tenant)
        # must be refused here, BEFORE asking a human to approve an action on it
        if act.device_id not in pool:
            if _REPO_REF is not None:
                _REPO_REF.audit(state["tenant_id"], event="refusal", tool=act.tool,
                                args=act.args, evidence=[e.model_dump() for e in act.evidence],
                                detail=f"device {act.device_id} not confirmed by any tool result")
            continue
        # if the planner sent no structured evidence, harvest device/timestamp
        # facts for this device from the captured tool outputs so the proposal
        # is self-contained and auditable
        # keep evidence typed as EvidenceRef; dump to dicts only at the repo/interrupt boundary
        evidence = list(act.evidence)
        if not evidence:
            for out in state.get("tool_outputs", []):
                text = str(out["result"])
                if text.startswith("error:") or '"error"' in text or '"refused"' in text:
                    continue  # never fabricate evidence from a failed lookup
                if act.device_id in text:
                    ts = re.search(rf"{act.device_id}\|(\d{{4}}-\d{{2}}-\d{{2}}T[^\"']*)", text)
                    if ts:
                        evidence = [EvidenceRef(device_id=act.device_id,
                                     collected_at=ts.group(1)[:19],
                                     metric="snapshot",
                                     value=f"referenced in {out['tool']} output")]
                        break
        # link to the real DB action created by the tool call (if any), so the
        # UI can resolve it by id without a second lookup
        real = None
        if _REPO_REF is not None:
            row = _REPO_REF.con.execute(
                "SELECT action_id FROM actions WHERE company_id=? AND status='proposed' "
                "AND tool=? AND device_id=? ORDER BY created_at DESC LIMIT 1",
                (state["tenant_id"], act.tool, act.device_id)).fetchone()
            real = row[0] if row else None
        proposals.append(act.model_copy(update={
            "action_id": real or f"unproposed_{len(proposals)}",
            "evidence": evidence,
            "rationale": act.rationale or ("evidence-grounded" if ok else "INSUFFICIENT EVIDENCE")}))
    if not proposals:
        # every action was refused pre-gate: never bother the human with an empty ask
        return {"approval": None, "approved": False}
    decision = interrupt({
        "proposed_actions": [p.model_dump() for p in proposals],
        "evidence_pool": sorted(_evidence_pool(state.get("tool_outputs", []))),
        "message": "Approve these actions? Resume with Command(resume={'approved': bool, 'approver': str})",
    })
    approved = bool(decision.get("approved"))
    approver = str(decision.get("approver") or "human")
    # proposals that never reached the action tools (no DB row yet) are created
    # and resolved here, so approval always leaves a complete audit trail
    if _REPO_REF is not None:
        for p in proposals:
            if p.action_id.startswith("unproposed_"):
                if approved and p.device_id in _evidence_pool(state.get("tool_outputs", [])):
                    real = _REPO_REF.propose_action(
                        state["tenant_id"], p.device_id, p.tool, p.args,
                        [e.model_dump() for e in p.evidence])
                    _REPO_REF.resolve_action(state["tenant_id"], real, True, approver)
                    p.action_id = real
                else:
                    _REPO_REF.audit(state["tenant_id"], event="refusal", tool=p.tool,
                                    args=p.args, evidence=[e.model_dump() for e in p.evidence],
                                    detail="unproposed action rejected or ungrounded")
    return {"approval": decision, "approved": approved}


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
    parts.append(f"(plan intent: {state['plan'].intent}; tool calls: "
                 f"{len(state.get('tool_outputs', []))})")
    return {"response": "\n\n".join(parts)}


_REPO_REF = None  # set by build_workflow; used by hitl_gate for action-id linkage


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
    g.add_node("formatter", formatter)
    g.set_entry_point("planner")
    g.add_edge("planner", "analyst")
    g.add_conditional_edges("analyst", analyst_router,
                            {"execute": "execute_tools", "verify": "verifier"})
    g.add_edge("execute_tools", "analyst")
    g.add_conditional_edges("verifier", verifier_router,
                            {"hitl": "hitl", "respond": "formatter",
                             "re_analyze": "analyst"})
    g.add_edge("hitl", "formatter")
    g.add_edge("formatter", END)
    return g.compile(checkpointer=checkpointer)
