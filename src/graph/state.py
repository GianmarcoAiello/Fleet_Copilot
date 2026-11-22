"""Graph state + typed contracts for the Fleet Copilot workflow."""
from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class EvidenceRef(BaseModel):
    """A citation pointing at a concrete telemetry fact."""
    device_id: str
    collected_at: str
    metric: str
    value: str
    source_tool_call_id: Optional[str] = None


class ProposedAction(BaseModel):
    """A state-changing action awaiting human approval."""
    tool: Literal["create_upgrade_order", "open_remediation_ticket",
                  "flag_device_for_replacement", "notify_employee"]
    device_id: str
    args: dict = Field(default_factory=dict)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    rationale: str = ""
    action_id: Optional[str] = None  # filled by the HITL gate when proposed


class ExecutionPlan(BaseModel):
    intent: Literal["qa", "trend_insight", "action_request", "mixed"]
    steps: list[str] = Field(description="Concrete ordered steps for the analyst")


class DraftClaim(BaseModel):
    claim: str
    evidence: list[EvidenceRef] = []


class AgentState(TypedDict, total=False):
    # inputs
    query: str
    tenant_id: str                    # injected at invocation, NEVER model-visible
    approver: str
    # control
    plan: ExecutionPlan
    # analyst loop
    messages: Annotated[list, operator.add]   # analyst LLM + tool messages
    tool_outputs: Annotated[list[dict], operator.add]  # structured results for verifier
    iteration: int
    # verification
    draft_claims: list[DraftClaim]
    verified_claims: list[DraftClaim]
    unsupported: list[str]
    unverified: list[str]
    reground_feedback: str
    reground_attempts: int
    action_nudges: int               # deterministic action-ask retries (bounded)
    # HITL
    pending_actions: list[ProposedAction]
    approved_action_ids: list[str]
    executed_action_ids: list[str]
    approved: Optional[bool]
    approval: Optional[dict]          # Command(resume=...) payload
    # output
    response: str
    audit_refs: list[str]
    refused: bool
    refusal_reason: str
