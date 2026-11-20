"""LLM-facing tools. Tenant_id is injected from RunnableConfig (InjectedToolArg) —
it is NEVER a parameter the model can set. Read tools wrap Repository queries;
action tools only propose (execution requires HITL approval)."""
from __future__ import annotations

import json
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field

from src.db.loader import Repository

_repo: Repository | None = None


def set_repo(repo: Repository) -> None:
    global _repo
    _repo = repo


def _tenant(config: Annotated[RunnableConfig, InjectedToolArg]) -> str:
    tenant = (config.get("configurable") or {}).get("tenant_id")
    if not tenant:
        raise ValueError("tenant_id missing from RunnableConfig — refusing to run")
    return tenant


def _slim(rows: list[dict], max_rows: int = 25) -> list[dict]:
    return rows[:max_rows]


# ── read tools ──────────────────────────────────────────────────────────────

@tool
def query_fleet_snapshot(
    config: Annotated[RunnableConfig, InjectedToolArg],
    filters: Optional[str] = Field(
        default=None,
        description="Optional free-text description of the filter intent (e.g. 'macOS 15+'). "
                    "Predefined filters: os_version_lt=X, min_avail_disk_gb=X, check=ID, "
                    "software=NAME",
    ),
    min_avail_disk_gb: Optional[float] = None,
    os_version_lt: Optional[str] = None,
    software: Optional[str] = None,
) -> str:
    """Latest snapshot per device for the caller's company. Use for current-state questions."""
    t = _tenant(config)
    assert _repo is not None
    if min_avail_disk_gb is not None:
        rows = _repo.latest_snapshots(
            t,
            "s.device_id IN (SELECT device_id FROM disk_volumes WHERE company_id = ? "
            "GROUP BY device_id HAVING min(available_bytes)/1e9 < ?)",
            (t, min_avail_disk_gb),
        )
    elif software:
        rows = _repo.latest_snapshots(
            t,
            "s.device_id IN (SELECT device_id FROM installed_software WHERE company_id=? AND name=?)",
            (t, software),
        )
    elif os_version_lt:
        rows = _repo.latest_snapshots(t, "s.os_version < ?", (os_version_lt,))
    else:
        rows = _repo.latest_snapshots(t)
    return json.dumps({"count": len(rows), "rows": _slim(rows)}, default=str)


@tool
def analyze_time_series_trends(
    config: Annotated[RunnableConfig, InjectedToolArg],
    trend: str = Field(description="One of: disk | battery | ram | compliance | software"),
) -> str:
    """Pre-built 30-day trend analyses for the caller's company. Returns per-device aggregates."""
    t = _tenant(config)
    assert _repo is not None
    fn = {"disk": _repo.disk_trend, "battery": _repo.battery_report,
          "ram": _repo.ram_pressure}.get(trend)
    if fn:
        return json.dumps({"trend": trend, "rows": _slim(fn(t))}, default=str)
    if trend == "compliance":
        fails = _repo.compliance(t, status="fail")
        drift: dict[str, list] = {}
        for r in fails:
            drift.setdefault(r["device_id"], []).append(
                {"check": r["check_id"], "sev": r["severity"], "at": str(r["collected_at"])})
        return json.dumps({"trend": "compliance", "devices_with_fails": drift}, default=str)
    if trend == "software":
        return json.dumps({"trend": "software", "rows": _repo.software_events(t)}, default=str)
    return json.dumps({"error": f"unknown trend '{trend}', pick disk|battery|ram|compliance|software"})


@tool
def query_device_history(
    config: Annotated[RunnableConfig, InjectedToolArg],
    device_id: str,
    limit: int = Field(default=30, description="Number of most recent snapshots"),
) -> str:
    """Full time-series history for ONE device (must belong to the caller's company)."""
    t = _tenant(config)
    assert _repo is not None
    if not _repo.device_exists(t, device_id):
        return json.dumps({"error": f"device '{device_id}' not found in your organization"})
    return json.dumps({"rows": _repo.device_history(t, device_id, limit)}, default=str)


@tool
def query_compliance(
    config: Annotated[RunnableConfig, InjectedToolArg],
    status: Optional[str] = Field(default=None, description="pass | fail"),
    severity: Optional[str] = Field(default=None, description="low | medium | high"),
) -> str:
    """Compliance results across all devices of the caller's company."""
    t = _tenant(config)
    assert _repo is not None
    return json.dumps({"rows": _slim(_repo.compliance(t, status, severity), 40)}, default=str)


# ── action tools (propose-only; HITL gate handles approval) ────────────────

class _UpgradeArgs(BaseModel):
    device_id: str
    component: str = Field(description="Component to upgrade: ram | disk | battery")
    spec: str = Field(description="Target spec, e.g. '32 GB', '1 TB SSD'")
    evidence: list[dict] = Field(description="Evidence refs: device_id, collected_at, metric, value")
    rationale: str = ""


class _TicketArgs(BaseModel):
    device_id: str
    check_id: str = Field(description="Failed compliance check: disk_encryption | screen_lock | os_up_to_date")
    note: str = Field(description="Remediation note for the IT admin")
    evidence: list[dict] = Field(description="Evidence refs: device_id, collected_at, metric, value")
    rationale: str = ""


class _ReplaceArgs(BaseModel):
    device_id: str
    reason: str = Field(description="Why the device should be replaced (end-of-life hardware)")
    evidence: list[dict] = Field(description="Evidence refs: device_id, collected_at, metric, value")
    rationale: str = ""


class _NotifyArgs(BaseModel):
    device_id: str
    message: str = Field(description="Message for the employee using the device")
    evidence: list[dict] = Field(description="Evidence refs: device_id, collected_at, metric, value")
    rationale: str = ""


def _make_action_tool(name: str, description: str, schema: type[BaseModel]):
    @tool(name, description=description, args_schema=schema)
    def _act(config: Annotated[RunnableConfig, InjectedToolArg],
             device_id: str, evidence: list[dict], rationale: str = "", **kwargs) -> str:
        t = _tenant(config)
        assert _repo is not None
        args = {k: v for k, v in kwargs.items()}
        if not _repo.device_exists(t, device_id):
            _repo.audit(t, event="refusal", tool=name, args=args, evidence=evidence,
                        detail=f"unknown or cross-tenant device {device_id}")
            return json.dumps({"refused": True,
                               "reason": f"device '{device_id}' not found in your organization"})
        if not evidence:
            _repo.audit(t, event="refusal", tool=name, args=args, evidence=[],
                        detail="no evidence supplied")
            return json.dumps({"refused": True,
                               "reason": "insufficient evidence: cite telemetry facts first"})
        action_id = _repo.propose_action(t, device_id, name, args, evidence)
        return json.dumps({"proposed": True, "action_id": action_id,
                           "note": "awaiting explicit human approval"})
    return _act


create_upgrade_order = _make_action_tool(
    "create_upgrade_order",
    "Propose a hardware/component upgrade order for a device (e.g. RAM, disk, battery).",
    _UpgradeArgs)
open_remediation_ticket = _make_action_tool(
    "open_remediation_ticket",
    "Propose opening a remediation ticket for a failed compliance check on a device.",
    _TicketArgs)
flag_device_for_replacement = _make_action_tool(
    "flag_device_for_replacement",
    "Propose flagging a device for replacement (end-of-life hardware).",
    _ReplaceArgs)
notify_employee = _make_action_tool(
    "notify_employee",
    "Propose notifying the employee assigned to a device (e.g. about software policy violations).",
    _NotifyArgs)


READ_TOOLS = [query_fleet_snapshot, analyze_time_series_trends, query_device_history, query_compliance]
ACTION_TOOLS = [create_upgrade_order, open_remediation_ticket, flag_device_for_replacement, notify_employee]
ALL_TOOLS = READ_TOOLS + ACTION_TOOLS
