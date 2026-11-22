"""NDJSON telemetry -> DuckDB repository with tenant-scoped views and audit log.

One DuckDB file holds everything: normalized telemetry tables, derived trend
views, the action registry, and the append-only audit log. Tenant isolation
happens at this boundary: every read/write helper takes tenant_id and appends
`company_id = ?` — the LLM never composes the WHERE clause.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import duckdb

# ── schema ──────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id      VARCHAR,        -- deterministic: device_id|collected_at
    device_id        VARCHAR,
    company_id       VARCHAR,
    employee_id      VARCHAR,
    collected_at     TIMESTAMP,
    agent_version    VARCHAR,
    os_platform      VARCHAR,
    os_product       VARCHAR,
    os_version       VARCHAR,
    os_arch          VARCHAR,
    model_name       VARCHAR,
    serial_number    VARCHAR,
    processor        VARCHAR,
    ram_total_bytes  BIGINT,
    ram_used_bytes   BIGINT
);
CREATE TABLE IF NOT EXISTS disk_volumes (
    snapshot_id    VARCHAR,
    device_id      VARCHAR,
    company_id     VARCHAR,
    volume_name    VARCHAR,
    file_system    VARCHAR,
    mount_point    VARCHAR,
    encrypted      BOOLEAN,
    total_bytes    BIGINT,   -- dataset field: size_bytes
    available_bytes BIGINT
);
CREATE TABLE IF NOT EXISTS battery (
    snapshot_id        VARCHAR,
    device_id          VARCHAR,
    company_id         VARCHAR,
    battery_present    BOOLEAN,
    percentage         INTEGER,
    condition          VARCHAR,
    cycle_count        INTEGER,
    full_charge_capacity INTEGER
);
CREATE TABLE IF NOT EXISTS compliance_results (
    snapshot_id  VARCHAR,
    device_id    VARCHAR,
    company_id   VARCHAR,
    check_id     VARCHAR,
    status       VARCHAR,
    severity     VARCHAR
);
CREATE TABLE IF NOT EXISTS installed_software (
    snapshot_id  VARCHAR,
    device_id    VARCHAR,
    company_id   VARCHAR,
    name         VARCHAR,
    version      VARCHAR
);
CREATE TABLE IF NOT EXISTS network_interfaces (
    snapshot_id  VARCHAR,
    device_id    VARCHAR,
    company_id   VARCHAR,
    interface_name VARCHAR,
    family       VARCHAR,
    address      VARCHAR,
    mac          VARCHAR,
    internal     BOOLEAN
);
CREATE TABLE IF NOT EXISTS actions (
    action_id   VARCHAR PRIMARY KEY,
    company_id  VARCHAR,
    device_id   VARCHAR,
    tool        VARCHAR,
    args_json   VARCHAR,
    status      VARCHAR,   -- proposed | approved | rejected | executed
    created_at  TIMESTAMP
);
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id     VARCHAR,
    ts           TIMESTAMP,
    tenant_id    VARCHAR,
    approver     VARCHAR,
    event        VARCHAR,  -- proposal | approval | rejection | refusal | execution
    tool         VARCHAR,
    args_json    VARCHAR,
    evidence_json VARCHAR,
    detail       VARCHAR
);
"""


def _flat(r: dict) -> tuple[list, list, list, list, list]:
    sid = f"{r['device_id']}|{r['collected_at']}"
    ident = r.get("device_identity") or {}
    mem = r.get("memory") or {}
    os_ = r.get("os") or {}
    snaps = [(
        sid, r["device_id"], r["company_id"], r.get("employee_id"), r["collected_at"],
        r.get("agent_version"), os_.get("platform"), os_.get("product_name"),
        os_.get("product_version"), os_.get("architecture"), ident.get("model_name"),
        ident.get("serial_number"), ident.get("processor"),
        mem.get("total_memory_bytes"), mem.get("used_memory_bytes"),
    )]
    disks, bats, comps, softs, nets = [], [], [], [], []
    for v in r.get("disk_volumes") or []:
        disks.append((sid, r["device_id"], r["company_id"], v.get("volume_name"),
                      v.get("file_system"), v.get("mount_point"), v.get("encrypted"),
                      v.get("size_bytes"), v.get("available_bytes")))
    b = r.get("battery") or {}
    bats.append((sid, r["device_id"], r["company_id"], b.get("battery_present"),
                 b.get("percentage"), b.get("condition"), b.get("cycle_count"),
                 b.get("full_charge_capacity")))
    for c in r.get("compliance_results") or []:
        comps.append((sid, r["device_id"], r["company_id"], c.get("check_id"),
                      c.get("status"), c.get("severity")))
    for s in r.get("installed_software") or []:
        softs.append((sid, r["device_id"], r["company_id"], s.get("name"), s.get("version")))
    for n in r.get("network") or []:
        nets.append((sid, r["device_id"], r["company_id"], n.get("interface_name"),
                     n.get("family"), n.get("address"), n.get("mac"), n.get("internal")))
    return snaps, disks, bats, comps, softs, nets


def _insert_many(con, table: str, rows: list[tuple], ncols: int) -> None:
    """Fast multi-row INSERT: duckdb executemany pays per-row overhead
    (~1-3s per table on this dataset); chunked VALUES is ~50x faster."""
    if not rows:
        return
    ph = "(" + ",".join(["?"] * ncols) + ")"
    for i in range(0, len(rows), 500):
        part = rows[i:i + 500]
        sql = f"INSERT INTO {table} SELECT * FROM (VALUES " + ",".join([ph] * len(part)) + ")"
        con.execute(sql, [v for r in part for v in r])


def load_ndjson(db_path: str | os.PathLike, ndjson_path: str | os.PathLike) -> int:
    """Load NDJSON into a fresh DuckDB file. Returns number of snapshots."""
    Path(db_path).unlink(missing_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA)
    snaps, disks, bats, comps, softs, nets = [], [], [], [], [], []
    with open(ndjson_path) as f:
        for line in f:
            s, d, b, c, w, n = _flat(json.loads(line))
            snaps += s; disks += d; bats += b; comps += c; softs += w; nets += n
    _insert_many(con, "snapshots", snaps, 15)
    _insert_many(con, "disk_volumes", disks, 9)
    _insert_many(con, "battery", bats, 8)
    _insert_many(con, "compliance_results", comps, 6)
    _insert_many(con, "installed_software", softs, 5)
    _insert_many(con, "network_interfaces", nets, 8)
    n = con.execute("SELECT count(*) FROM snapshots").fetchone()[0]
    con.close()
    return n


class Repository:
    """Tenant-scoped access to telemetry + actions + audit log.

    Tenant isolation contract: every method requires tenant_id and injects it
    as a parameter, never as string interpolation. The LLM-facing tools call
    these methods with the tenant_id taken from RunnableConfig — it is never a
    model-controlled argument.
    """

    def __init__(self, db_path: str | os.PathLike, bootstrap: bool = True):
        self.db_path = str(db_path)
        self.con = duckdb.connect(self.db_path, read_only=False)
        if bootstrap and self._needs_bootstrap():
            # clean-clone convenience: rebuild from the NDJSON so every entry
            # point works without a manual load step
            self.con.close()
            load_ndjson(self.db_path, Path(__file__).resolve().parent.parent.parent
                        / "device-telemetry-dataset.ndjson")
            self.con = duckdb.connect(self.db_path, read_only=False)

    EXPECTED_TABLES = ("snapshots", "disk_volumes", "battery", "compliance_results",
                       "installed_software", "network_interfaces", "actions", "audit_log")

    def _needs_bootstrap(self) -> bool:
        try:
            rows = self.con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'").fetchall()
            present = {r[0] for r in rows}
            # an older DB file (pre-migration schema) is rebuilt from source:
            # everything in it is derived data
            return not set(self.EXPECTED_TABLES).issubset(present)
        except duckdb.Error:
            return True

    @classmethod
    def _open_readonly(cls, db_path: str | os.PathLike) -> "Repository":
        """Read-only handle that coexists with an external writer's lock."""
        obj = cls.__new__(cls)
        obj.con = duckdb.connect(str(db_path), read_only=True)
        return obj

    # ── reads (tenant-scoped) ────────────────────────────────────────────
    def latest_snapshots(self, tenant_id: str, filters: str = "", params: tuple = ()) -> list[dict]:
        # one call, full picture: latest snapshot per device with its disk,
        # battery and compliance aggregates joined — the LLM can answer
        # "which devices…" questions without per-device follow-up calls
        rows = self.con.execute(
            f"""
            SELECT s.*,
                   dv.total_bytes     AS disk_total_bytes,
                   dv.available_bytes AS disk_available_bytes,
                   dv.total_bytes / POW(1024, 3)     AS disk_total_gib,
                   dv.available_bytes / POW(1024, 3) AS disk_available_gib,
                   CASE WHEN dv.total_bytes > 0
                        THEN dv.available_bytes::DOUBLE / dv.total_bytes END
                                      AS disk_free_pct,
                   b.percentage       AS battery_percentage,
                   b.condition        AS battery_condition,
                   b.cycle_count      AS battery_cycle_count,
                   b.full_charge_capacity AS battery_full_charge_capacity,
                   cf.fails           AS compliance_fails,
                   cf.failed_checks   AS failed_checks,
                   ni.network_info    AS network_info
            FROM snapshots s
            LEFT JOIN (SELECT snapshot_id, SUM(total_bytes)     AS total_bytes,
                              SUM(available_bytes)              AS available_bytes
                       FROM disk_volumes GROUP BY snapshot_id) dv USING (snapshot_id)
            LEFT JOIN battery b USING (snapshot_id)
            LEFT JOIN (SELECT snapshot_id, COUNT(*) AS fails,
                              string_agg(check_id, ',') AS failed_checks
                       FROM compliance_results
                       WHERE status = 'fail' GROUP BY snapshot_id) cf USING (snapshot_id)
            LEFT JOIN (SELECT snapshot_id,
                              string_agg(interface_name || ' (' || family || ' ' ||
                                         address || ')', '; ') AS network_info
                       FROM network_interfaces GROUP BY snapshot_id) ni USING (snapshot_id)
            WHERE s.company_id = ?
              AND s.collected_at = (SELECT max(collected_at) FROM snapshots w
                                    WHERE w.device_id = s.device_id)
              {"AND " + filters if filters else ""}
            ORDER BY s.device_id
            """,
            (tenant_id, *params),
        ).fetchall()
        cols = [d[0] for d in self.con.description]
        return [dict(zip(cols, r)) for r in rows]

    def latest_snapshot_row(self, tenant_id: str, device_id: str) -> Optional[dict]:
        """The single latest joined row for one device (evidence re-reads)."""
        rows = self.latest_snapshots(tenant_id, "s.device_id = ?", (device_id,))
        return rows[0] if rows else None

    def device_history(self, tenant_id: str, device_id: str, limit: int = 30) -> list[dict]:
        # disk aggregates are joined per snapshot: without them the model sees a
        # history with no disk fields and wrongly concludes the data "is missing"
        rows = self.con.execute(
            """
            SELECT s.*, b.percentage, b.condition AS battery_condition,
                   b.cycle_count, b.full_charge_capacity,
                   dv.disk_total_bytes, dv.disk_available_bytes,
                   dv.disk_available_bytes / POW(1024, 3) AS disk_available_gib,
                   CASE WHEN dv.disk_total_bytes > 0
                        THEN dv.disk_available_bytes::DOUBLE / dv.disk_total_bytes END
                                        AS disk_free_pct,
                   (ni.snapshot_id IS NOT NULL) AS has_network_info,
                   ni.network_info
            FROM snapshots s
            LEFT JOIN battery b USING (snapshot_id)
            LEFT JOIN (SELECT snapshot_id, SUM(total_bytes) AS disk_total_bytes,
                              SUM(available_bytes) AS disk_available_bytes
                       FROM disk_volumes GROUP BY snapshot_id) dv USING (snapshot_id)
            LEFT JOIN (SELECT snapshot_id,
                              string_agg(interface_name || ' (' || family || ' ' ||
                                         address || ')', '; ') AS network_info
                       FROM network_interfaces GROUP BY snapshot_id) ni USING (snapshot_id)
            WHERE s.company_id = ? AND s.device_id = ?
            ORDER BY s.collected_at DESC LIMIT ?
            """,
            (tenant_id, device_id, limit),
        ).fetchall()
        cols = [d[0] for d in self.con.description]
        return [dict(zip(cols, r)) for r in rows]

    def compliance(self, tenant_id: str, status: str | None = None,
                   severity: str | None = None) -> list[dict]:
        clauses, params = ["c.company_id = ?"], [tenant_id]
        if status:
            clauses.append("c.status = ?"); params.append(status)
        if severity:
            clauses.append("c.severity = ?"); params.append(severity)
        rows = self.con.execute(
            f"""
            SELECT c.check_id, c.status, c.severity, c.device_id, s.model_name,
                   s.os_version, s.collected_at
            FROM compliance_results c JOIN snapshots s USING (snapshot_id)
            WHERE {' AND '.join(clauses)}
            ORDER BY c.device_id, s.collected_at DESC
            """,
            params,
        ).fetchall()
        cols = [d[0] for d in self.con.description]
        return [dict(zip(cols, r)) for r in rows]

    def disk_trend(self, tenant_id: str) -> list[dict]:
        # per-snapshot SUM first: a device with multiple volumes must report the
        # whole-disk first→last delta, not a first/last across mixed volume rows.
        # Units: GiB (2^30 bytes) — consistent with the snapshot fields and the
        # QA reference answers.
        rows = self.con.execute(
            """
            WITH per_snap AS (
                SELECT d.device_id, s.collected_at,
                       SUM(d.available_bytes)/POW(1024,3) AS avail_gib
                FROM disk_volumes d JOIN snapshots s USING (snapshot_id)
                WHERE d.company_id = ?
                GROUP BY d.snapshot_id, d.device_id, s.collected_at
            )
            SELECT device_id,
                   min(avail_gib) AS min_avail_gib,
                   first(avail_gib ORDER BY collected_at) AS avail_gib_start,
                   last(avail_gib ORDER BY collected_at)  AS avail_gib_end,
                   min(collected_at) AS first_seen,
                   max(collected_at) AS last_seen,
                   (first(avail_gib ORDER BY collected_at)
                    - last(avail_gib ORDER BY collected_at)) AS burned_gib,
                   count(*) AS n
            FROM per_snap GROUP BY device_id ORDER BY min_avail_gib ASC
            """,
            (tenant_id,),
        ).fetchall()
        cols = [d[0] for d in self.con.description]
        return [dict(zip(cols, r)) for r in rows]

    def battery_report(self, tenant_id: str) -> list[dict]:
        rows = self.con.execute(
            """
            SELECT b.device_id,
                   max(b.cycle_count) AS max_cycles,
                   min(b.cycle_count) AS cycles_start,
                   max(b.cycle_count) AS cycles_end,
                   max_by(b.condition, s.collected_at) AS latest_condition,
                   arg_min(b.full_charge_capacity, s.collected_at) AS capacity_start,
                   arg_max(b.full_charge_capacity, s.collected_at) AS capacity_end,
                   min(s.collected_at) AS first_seen,
                   max(s.collected_at) AS last_seen,
                   (max(b.cycle_count) - min(b.cycle_count))::DOUBLE
                     / GREATEST(date_diff('day', min(s.collected_at), max(s.collected_at)), 1)
                     AS cycles_per_day
            FROM battery b JOIN snapshots s USING (snapshot_id)
            WHERE b.company_id = ? AND b.battery_present
            GROUP BY b.device_id ORDER BY max_cycles DESC
            """,
            (tenant_id,),
        ).fetchall()
        cols = [d[0] for d in self.con.description]
        return [dict(zip(cols, r)) for r in rows]

    def ram_pressure(self, tenant_id: str) -> list[dict]:
        rows = self.con.execute(
            """
            SELECT device_id, avg(ram_used_bytes::DOUBLE / ram_total_bytes) AS avg_pressure,
                   max(ram_used_bytes::DOUBLE / ram_total_bytes) AS max_pressure
            FROM snapshots WHERE company_id = ? AND ram_total_bytes > 0
            GROUP BY device_id ORDER BY avg_pressure DESC
            """,
            (tenant_id,),
        ).fetchall()
        cols = [d[0] for d in self.con.description]
        return [dict(zip(cols, r)) for r in rows]

    def software_events(self, tenant_id: str, names: list[str] | None = None) -> list[dict]:
        """Software inventory per device with first-seen/last-seen timestamps,
        version and assignee — last_seen shows whether the software was still
        present at the final snapshot; the join supplies employee_id so
        notify_employee can target users."""
        base = ("SELECT i.name, i.version, i.device_id, s.employee_id, "
                "min(s.collected_at) AS first_seen, max(s.collected_at) AS last_seen "
                "FROM installed_software i JOIN snapshots s USING (snapshot_id) "
                "WHERE i.company_id = ?")
        group = " GROUP BY i.name, i.version, i.device_id, s.employee_id "
        if names:
            marks = ",".join("?" * len(names))
            rows = self.con.execute(
                f"{base} AND i.name IN ({marks}){group}ORDER BY first_seen DESC",
                (tenant_id, *names),
            ).fetchall()
        else:
            rows = self.con.execute(
                f"{base}{group}ORDER BY first_seen DESC",
                (tenant_id,),
            ).fetchall()
        cols = [d[0] for d in self.con.description]
        return [dict(zip(cols, r)) for r in rows]

    def telemetry_gaps(self, tenant_id: str) -> list[dict]:
        """Per-device snapshots with MISSING subsystem telemetry: network rows
        absent, or battery fields all null (battery_present IS NULL — a desktop
        with battery_present=False is 'no battery', not a gap). Feeds the
        'gaps' trend: incomplete-telemetry and missing-data-pattern questions."""
        rows = self.con.execute(
            """
            SELECT s.device_id,
                   count(*) AS n_snapshots,
                   list(strftime(s.collected_at, '%Y-%m-%d %H:%M'))
                       FILTER (WHERE n.snapshot_id IS NULL) AS network_missing_at,
                   list(strftime(s.collected_at, '%Y-%m-%d %H:%M'))
                       FILTER (WHERE b.battery_present IS NULL) AS battery_missing_at
            FROM snapshots s
            LEFT JOIN (SELECT DISTINCT snapshot_id FROM network_interfaces) n USING (snapshot_id)
            LEFT JOIN battery b USING (snapshot_id)
            WHERE s.company_id = ?
            GROUP BY s.device_id
            HAVING len(network_missing_at) > 0 OR len(battery_missing_at) > 0
            ORDER BY s.device_id
            """,
            (tenant_id,),
        ).fetchall()
        cols = [d[0] for d in self.con.description]
        return [dict(zip(cols, r)) for r in rows]

    def version_changes(self, tenant_id: str) -> list[dict]:
        """OS/agent version at the window edges per device: answers 'did anything
        update OS or agent in the last 30 days' without per-device history."""
        rows = self.con.execute(
            """
            SELECT device_id,
                   first(os_version ORDER BY collected_at)    AS os_version_start,
                   last(os_version ORDER BY collected_at)     AS os_version_end,
                   first(agent_version ORDER BY collected_at) AS agent_version_start,
                   last(agent_version ORDER BY collected_at)  AS agent_version_end
            FROM snapshots WHERE company_id = ?
            GROUP BY device_id ORDER BY device_id
            """,
            (tenant_id,),
        ).fetchall()
        cols = [d[0] for d in self.con.description]
        out = [dict(zip(cols, r)) for r in rows]
        for r in out:  # flag in Python: the model must not diff version strings
            r["os_changed"] = r["os_version_start"] != r["os_version_end"]
            r["agent_changed"] = r["agent_version_start"] != r["agent_version_end"]
        return out

    def device_exists(self, tenant_id: str, device_id: str) -> bool:
        return self.con.execute(
            "SELECT 1 FROM snapshots WHERE company_id = ? AND device_id = ? LIMIT 1",
            (tenant_id, device_id),
        ).fetchone() is not None

    def device_tenant(self, device_id: str) -> str | None:
        row = self.con.execute(
            "SELECT company_id FROM snapshots WHERE device_id = ? LIMIT 1", (device_id,)
        ).fetchone()
        return row[0] if row else None

    # ── actions + audit (append-only) ────────────────────────────────────
    def propose_action(self, tenant_id: str, device_id: str, tool: str,
                       args: dict, evidence: list[dict]) -> str:
        aid = f"act_{uuid.uuid4().hex[:12]}"
        self.con.execute(
            "INSERT INTO actions VALUES (?,?,?,?,?,?,now())",
            (aid, tenant_id, device_id, tool, json.dumps(args, sort_keys=True), "proposed"),
        )
        self.audit(tenant_id, event="proposal", tool=tool, args=args,
                   evidence=evidence, detail=f"action_id={aid}")
        return aid

    def resolve_action(self, tenant_id: str, action_id: str, approved: bool,
                       approver: str) -> bool:
        status = "approved" if approved else "rejected"
        row = self.con.execute(
            "SELECT company_id, tool, args_json FROM actions WHERE action_id = ? AND status='proposed'",
            (action_id,),
        ).fetchone()
        if not row or row[0] != tenant_id:
            return False
        self.con.execute("UPDATE actions SET status=? WHERE action_id=?", (status, action_id))
        self.audit(tenant_id, approver=approver, event=status, tool=row[1],
                   args=json.loads(row[2]), evidence=[],
                   detail=f"action_id={action_id}")
        return True

    def get_actions(self, tenant_id: str, action_ids: list[str]) -> list[dict]:
        """Full payloads for the HITL gate — any status, so a decision already
        taken elsewhere (web UI resolve) is reflected instead of lost. The DB
        is the source of truth, not model output."""
        out = []
        for aid in action_ids:
            row = self.con.execute(
                "SELECT action_id, tool, device_id, args_json, status FROM actions "
                "WHERE action_id=? AND company_id=?",
                (aid, tenant_id)).fetchone()
            if not row:
                continue
            ev = self.con.execute(
                "SELECT evidence_json FROM audit_log WHERE tenant_id=? AND event='proposal' "
                "AND detail=? ORDER BY ts DESC LIMIT 1",
                (tenant_id, f"action_id={aid}")).fetchone()
            out.append({"action_id": row[0], "tool": row[1], "device_id": row[2],
                        "args": json.loads(row[3]), "status": row[4],
                        "evidence": json.loads(ev[0]) if ev else []})
        return out

    def execute_action(self, tenant_id: str, action_id: str,
                       connector: str = "simulated") -> bool:
        """approved -> executed, with a `execution` audit event. The connector is
        a documented simulation (no real ticketing/email in the prototype)."""
        row = self.con.execute(
            "SELECT company_id, tool, args_json FROM actions WHERE action_id=? AND status='approved'",
            (action_id,)).fetchone()
        if not row or row[0] != tenant_id:
            return False
        self.con.execute("UPDATE actions SET status='executed' WHERE action_id=?", (action_id,))
        self.audit(tenant_id, approver=connector, event="execution", tool=row[1],
                   args=json.loads(row[2]), evidence=[],
                   detail=f"action_id={action_id} executed via simulated {connector} connector")
        return True

    def compliance_transitions(self, tenant_id: str) -> dict:
        """pass -> fail transitions with the first fail date per (device, check):
        pre-computed drift, so the model does not have to diff 25k chars of rows."""
        rows = self.con.execute(
            """
            SELECT c.device_id, c.check_id, c.status, s.collected_at, c.severity
            FROM compliance_results c JOIN snapshots s USING (snapshot_id)
            WHERE c.company_id = ?
            ORDER BY c.device_id, c.check_id, s.collected_at
            """,
            (tenant_id,)).fetchall()
        last: dict[tuple, str] = {}
        out: dict[str, list] = {}
        for dev, check, status, ts, sev in rows:
            key = (dev, check)
            if status == "fail" and last.get(key) == "pass":
                out.setdefault(dev, []).append(
                    {"check": check, "severity": sev, "first_fail_at": str(ts)})
            last[key] = status
        return {"transitions": out}

    def audit(self, tenant_id: str, event: str, tool: str, args: dict,
              evidence: list[dict], detail: str = "", approver: str = "") -> str:
        aid = f"aud_{uuid.uuid4().hex[:12]}"
        now_row = self.con.execute("SELECT now()").fetchone()
        self.con.execute(
            "INSERT INTO audit_log VALUES (?,?,?,?,?,?,?,?,?)",
            (aid, now_row[0] if now_row else None, tenant_id, approver, event, tool,
             json.dumps(args, sort_keys=True), json.dumps(evidence, default=str), detail),
        )
        return aid

    def audit_tail(self, tenant_id: str, limit: int = 20) -> list[dict]:
        rows = self.con.execute(
            "SELECT * FROM audit_log WHERE tenant_id = ? ORDER BY ts DESC LIMIT ?",
            (tenant_id, limit),
        ).fetchall()
        cols = [d[0] for d in self.con.description]
        return [dict(zip(cols, r)) for r in rows]

    def close(self):
        self.con.close()


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    src = Path(__file__).resolve().parent.parent.parent / "device-telemetry-dataset.ndjson"
    db = data_dir / "fleet.duckdb"
    n = load_ndjson(db, src)
    print(f"loaded {n} snapshots -> {db}")
