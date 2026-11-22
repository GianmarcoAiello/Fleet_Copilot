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

def _ver_tuple(v: str | None):
    """'14.5' -> (14, 5): OS versions compare numerically, never as strings."""
    import re
    nums = re.findall(r"\d+", (v or "").split()[0]) if v else []
    return tuple(int(x) for x in nums)


@tool
def query_fleet_snapshot(
    config: Annotated[RunnableConfig, InjectedToolArg],
    min_avail_disk_gib: Optional[float] = Field(
        default=None, description="Latest snapshot: devices with less free disk than this (GiB, 2^30 bytes)"),
    os_version_lt: Optional[str] = Field(
        default=None, description="macOS version filter, numeric compare (applies to macOS devices only)"),
    software: Optional[str] = Field(default=None, description="Devices with this software installed"),
    check: Optional[str] = Field(
        default=None, description="Devices currently failing this compliance check "
                                  "(disk_encryption | screen_lock | os_up_to_date)"),
) -> str:
    """Latest snapshot per device for the caller's company, with disk, battery and
    compliance aggregates. Filters combine with AND. Use for current-state questions."""
    t = _tenant(config)
    assert _repo is not None
    sql_filters, sql_params = [], []
    if software:
        sql_filters.append(
            "s.device_id IN (SELECT device_id FROM installed_software "
            "WHERE company_id = ? AND name = ?)")
        sql_params += [t, software]
    rows = _repo.latest_snapshots(t, " AND ".join(sql_filters), tuple(sql_params))
    if min_avail_disk_gib is not None:
        rows = [r for r in rows if r.get("disk_available_gib") is not None
                and r["disk_available_gib"] < min_avail_disk_gib]
    if os_version_lt:
        rows = [r for r in rows if r.get("os_platform") == "darwin"
                and _ver_tuple(r.get("os_version")) < _ver_tuple(os_version_lt)]
    if check:
        rows = [r for r in rows
                if check in [c.strip() for c in (r.get("failed_checks") or "").split(",")]]
    return json.dumps({"count": len(rows), "rows": _slim(rows)}, default=str)


@tool
def analyze_time_series_trends(
    config: Annotated[RunnableConfig, InjectedToolArg],
    trend: str = Field(description="One of: disk | battery | ram | compliance | "
                                  "software | gaps | versions"),
) -> str:
    """Pre-built 30-day trend analyses for the caller's company: disk (per-device
    first/last/min free space with dates), battery (cycles + daily cycle rate),
    ram, compliance (pass→fail drift), software (installs with first/last seen,
    version), gaps (snapshots missing network/battery telemetry, with dates),
    versions (OS/agent version changes over the window)."""
    t = _tenant(config)
    assert _repo is not None
    fn = {"disk": _repo.disk_trend, "battery": _repo.battery_report,
          "ram": _repo.ram_pressure, "gaps": _repo.telemetry_gaps,
          "versions": _repo.version_changes}.get(trend)
    if fn:
        return json.dumps({"trend": trend, "rows": _slim(fn(t))}, default=str)
    if trend == "compliance":
        # pre-computed pass -> fail transitions with first fail date — the model
        # cannot be asked to diff 25k chars of raw fail rows
        return json.dumps({"trend": "compliance", **_repo.compliance_transitions(t)},
                          default=str)
    if trend == "software":
        return json.dumps({"trend": trend, "rows": _repo.software_events(t)}, default=str)
    return json.dumps({"error": f"unknown trend '{trend}', pick disk|battery|ram|"
                                f"compliance|software|gaps|versions"})


@tool
def query_device_history(
    config: Annotated[RunnableConfig, InjectedToolArg],
    device_id: str,
    limit: int = Field(default=30, description="Number of most recent snapshots"),
) -> str:
    """Full time-series history for ONE device (must belong to the caller's
    company). Each row carries RAM, battery, disk total/free and network_info."""
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
    scope: str = Field(default="latest",
                       description="latest = one row per device+check (most recent collected_at, "
                                   "use for current-state questions); all = full 30-day history"),
) -> str:
    """Compliance results across all devices of the caller's company."""
    t = _tenant(config)
    assert _repo is not None
    if scope == "all":
        rows = _repo.compliance(t, status, severity)
        total = len(rows)
        slim = _slim(rows, 40)
    else:
        # dedup on ALL statuses first: the latest result per device+check decides,
        # THEN filter — a remediated device must not resurface as failing
        rows = _repo.compliance(t, severity=severity)
        latest: dict[tuple, dict] = {}
        for r in rows:  # ordered device_id, collected_at DESC
            key = (r["device_id"], r["check_id"])
            if key not in latest:
                latest[key] = r
        rows = list(latest.values())
        if status:
            rows = [r for r in rows if r["status"] == status]
        total = len(rows)
        slim = _slim(rows, 40)
    out: dict = {"rows": slim, "total_matching": total}
    if total > len(slim):
        out["truncated"] = True
        out["note"] = f"showing {len(slim)} of {total} rows"
    return json.dumps(out, default=str)


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


def _values_match(provided, db_value) -> bool:
    """Model-provided evidence value vs DB value: exact/normalised for strings,
    1% tolerance for numbers. A mismatch means the evidence is invented."""
    if provided is None:
        return True
    if isinstance(db_value, (int, float)) and not isinstance(db_value, bool):
        try:
            p = float(str(provided).replace(",", "").rstrip("%"))
        except ValueError:
            return False
        if db_value == 0:
            return p == 0
        return abs(p - float(db_value)) <= abs(float(db_value)) * 0.01
    return str(provided).strip().lower() == str(db_value).strip().lower()


_METRIC_ALIASES = {  # analyst-facing name -> latest-snapshot row field
    "cycle_count": "battery_cycle_count",
    "battery_cycles": "battery_cycle_count",
    "condition": "battery_condition",
    "free_disk_bytes": "disk_available_bytes",
    "disk_free_bytes": "disk_available_bytes",
    "available_bytes": "disk_available_bytes",
}


def _sufficiency(name: str, row: dict, args: dict) -> tuple[bool, str]:
    """Per-action minimum-evidence rule, checked against the DB — this is the
    real 'refuse without sufficient evidence' guardrail."""
    if name == "open_remediation_ticket":
        failed = [c.strip() for c in (row.get("failed_checks") or "").split(",") if c.strip()]
        if args.get("check_id") not in failed:
            return False, (f"check '{args.get('check_id')}' is not currently failing on "
                           f"{row['device_id']} (failed now: {failed or 'none'})")
    elif name == "flag_device_for_replacement":
        if row.get("battery_condition") != "Service Recommended":
            return False, (f"battery condition is '{row.get('battery_condition')}', "
                           "not 'Service Recommended' — no replacement grounds")
    elif name == "create_upgrade_order":
        comp = (args.get("component") or "").lower()
        if comp == "ram":
            total, used = row.get("ram_total_bytes"), row.get("ram_used_bytes")
            if not total or used / total < 0.9:
                return False, f"RAM pressure is below the 90% upgrade threshold on {row['device_id']}"
        elif comp == "disk":
            avail = row.get("disk_available_gib")
            if avail is None or avail >= 20:
                return False, f"free disk is at/above 20 GiB on {row['device_id']} — no upgrade grounds"
        elif comp == "battery":
            if row.get("battery_condition") == "Normal":
                return False, f"battery condition is Normal on {row['device_id']}"
    elif name == "notify_employee":
        if not row.get("employee_id"):
            return False, f"device {row['device_id']} has no assigned employee to notify"
    return True, ""


def _make_action_tool(name: str, description: str, schema: type[BaseModel]):
    @tool(name, description=description, args_schema=schema)
    def _act(config: Annotated[RunnableConfig, InjectedToolArg],
             device_id: str, evidence: list[dict], rationale: str = "", **kwargs) -> str:
        t = _tenant(config)
        assert _repo is not None
        args = {k: v for k, v in kwargs.items()}

        def refuse(reason: str) -> str:
            _repo.audit(t, event="refusal", tool=name, args=args,
                        evidence=evidence if isinstance(evidence, list) else [],
                        detail=reason)
            return json.dumps({"refused": True, "reason": reason})

        if not _repo.device_exists(t, device_id):
            return refuse(f"unknown or cross-tenant device {device_id}")
        if not evidence:
            return refuse("insufficient evidence: cite telemetry facts first")

        # the model's evidence values are NEVER trusted: re-read every ref from
        # the DB (tenant-scoped) and store the DB value in the audit trail
        row = _repo.latest_snapshot_row(t, device_id)
        if row is None:
            return refuse(f"no snapshot found for {device_id}")
        verified = []
        for ref in evidence:
            if not isinstance(ref, dict) or not ref.get("metric"):
                return refuse(f"malformed evidence ref: {ref!r} — need at least a metric name")
            metric = _METRIC_ALIASES.get(ref["metric"], ref["metric"])
            if metric not in row or row[metric] is None:
                return refuse(f"evidence metric '{ref['metric']}' not found on the "
                              f"latest snapshot of {device_id}")
            if not _values_match(ref.get("value"), row[metric]):
                return refuse(f"evidence value mismatch for '{ref['metric']}': you said "
                              f"{ref.get('value')!r}, the DB says {row[metric]!r}")
            verified.append({"device_id": device_id,
                             "collected_at": str(row["collected_at"])[:19],
                             "metric": metric, "value": row[metric]})
        ok, why = _sufficiency(name, row, args)
        if not ok:
            return refuse(f"insufficient grounds: {why}")
        action_id = _repo.propose_action(t, device_id, name, args, verified)
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
