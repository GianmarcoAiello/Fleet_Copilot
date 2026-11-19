"""Nodes: Planner, Analyst (ReAct), Grounding Verifier, HITL gate, Formatter."""
from __future__ import annotations

import json
import os

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
- If tools return nothing relevant, say so — do not extrapolate.
- For action tools, always attach telemetry evidence (device_id, collected_at, metric, value).
- If the user references a device outside your organization, the tools will refuse: report that.
"""

VERIFIER_SYSTEM = """You extract verifiable claims from the analyst's draft answer.
Return one claim per sentence containing a device_id or numeric metric, with the
evidence refs (device_id, collected_at, metric, value) that support it. Claims with
no supporting evidence must be listed as unsupported."""


def get_llm(temperature: float = 0.0):
    """BYOK via langchain init_chat_model. Provider/model/keys come from env:
    FLEET_LLM_MODEL (e.g. 'google:gemini-flash-lite-latest') + the provider's
    standard key var (GOOGLE_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)."""
    model = os.environ.get("FLEET_LLM_MODEL", "google_genai:gemini-flash-lite-latest")
    return init_chat_model(model, temperature=temperature)


# ── 1. Planner ──────────────────────────────────────────────────────────────

def planner(state: AgentState) -> dict:
    llm = get_llm().with_structured_output(ExecutionPlan)
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
                            "result": result.content[:4000]})
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

def _evidence_pool(tool_outputs: list[dict]) -> set[str]:
    pool = set()
    for out in tool_outputs:
        pool.add(str(out["args"].get("device_id", "")))
        text = str(out["result"])
        import re
        pool.update(re.findall(r"\b[A-Z0-9]{10,14}\b", text))  # device/serial-shaped ids
        pool.update(re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", text))
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
        llm = get_llm().with_structured_output(__import__("pydantic").RootModel[list[DraftClaim]])
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
            __import__("re").findall(r"\b[A-Z0-9]{10,14}\b", c.claim))
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
    proposals = []
    for act in plan.action_requests:
        ok = all(e.device_id in _evidence_pool(state.get("tool_outputs", []))
                 for e in act.evidence)
        # link to the real DB action created by the tool call (if any), so the
        # UI can resolve it by id without a second lookup
        real = None
        if _REPO_REF is not None:
            row = _REPO_REF.con.execute(
                "SELECT action_id FROM actions WHERE company_id=? AND status='proposed' "
                "AND tool=? AND device_id=? ORDER BY created_at DESC LIMIT 1",
                (state["tenant_id"], act.tool, act.device_id)).fetchone()
            real = row[0] if row else None
        evidence = act.evidence or []
        if not evidence and ok is False:
            evidence = []
        proposals.append(act.model_copy(update={
            "action_id": real or f"unproposed_{len(proposals)}",
            "rationale": act.rationale or ("evidence-grounded" if ok else "INSUFFICIENT EVIDENCE")}))
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
