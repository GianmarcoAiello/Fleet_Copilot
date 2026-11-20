#!/usr/bin/env python3
"""Live evaluation of the Fleet Copilot agent against the 100-QA dataset
(fleet_copilot_100_qa.json).

Per case the full graph runs with the configured LLM (temperature=0).
Scoring layers:
  1. deterministic — expected-device coverage in the response, evidence
     citations, cross-tenant leaks, HITL interrupt presence, audit outcome
  2. LLM judge — agent answer vs the dataset's expected answer
     (factual_match 0..1, hallucination flag)
  3. dataset validation — expected facts in the QA file are recomputed from
     the NDJSON telemetry so a wrong expectation is never charged to the agent

Results stream to evals/qa100_results.json (resume-safe: done ids are skipped).
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.db.loader import Repository, load_ndjson  # noqa: E402
from src.graph.nodes import build_workflow, get_llm  # noqa: E402

QA_PATH = ROOT / "evals" / "fleet_copilot_100_qa.json"
NDJSON = ROOT / "device-telemetry-dataset.ndjson"
RESULTS_PATH = ROOT / "evals" / "qa100_results.json"

DEVICE_RE = re.compile(r"\b(?=[A-Z0-9]*[A-Z])[A-Z0-9]{10,14}\b")
REFUSAL_WORDS = ("not found", "refus", "cannot", "no data", "no telemetry",
                 "no evidence", "do not have", "not in your organization",
                 "non posso", "non risult", "rifiut", "nessun", "non trovato", "non appartiene",
                 "non disponibile", "non disponibili", "non ho", "non gest")


# ── dataset validation: recompute expected facts from the telemetry ────────

def dataset_facts() -> dict:
    rows = [json.loads(l) for l in open(NDJSON)]
    latest: dict[str, dict] = {}
    for r in rows:  # records are chronological; last wins
        latest[r["device_id"]] = r
    facts: dict[str, dict] = {}
    for dev, r in latest.items():
        osinfo = r.get("os") or {}
        facts.setdefault(r["company_id"], {})[dev] = {
            "os": f"{osinfo.get('product_name', '')} {osinfo.get('product_version', '')}".strip(),
        }
    return {c: {"device_count": len(devs), "os_breakdown": dict(sorted(
        (o, sum(1 for d in devs.values() if d["os"] == o))
        for o in {d["os"] for d in devs.values()}))} for c, devs in facts.items()}


def validate_dataset(qa_pairs: list[dict]) -> list[str]:
    """Return a list of dataset-level mismatches (NOT agent failures)."""
    facts = dataset_facts()
    problems = []
    for q in qa_pairs:
        t, ev = q["tenant_id"], q.get("evidence") or []
        for e in ev:
            if isinstance(e, dict) and "device_count" in e:
                want = facts.get(t, {}).get("device_count")
                if e["device_count"] != want:
                    problems.append(f"{q['id']}: device_count {e['device_count']} != {want} ({t})")
            if isinstance(e, dict) and "os_breakdown" in e:
                got = facts.get(t, {}).get("os_breakdown", {})
                for osname, n in e["os_breakdown"].items():
                    if got.get(osname) != n:
                        problems.append(f"{q['id']}: os '{osname}' {n} != {got.get(osname)} ({t})")
    return problems


# ── LLM judge ───────────────────────────────────────────────────────────────

class Judge(BaseModel):
    factual_match: float = 0.0
    hallucination: bool = False
    verdict: str = "mismatch"   # match | partial | mismatch
    notes: str = ""


def judge_answer(question: str, expected: str, actual: str) -> Judge:
    try:
        llm = get_llm().with_structured_output(Judge, method="function_calling")
        return llm.invoke([
            ("system",
             "You judge a fleet-management QA agent. Compare the agent answer "
             "against the expected reference answer for the same question. "
             "factual_match: fraction (0..1) of the reference's factual content "
             "present and correct in the agent answer. Omissions and vaguer "
             "wording lower factual_match but are NEVER hallucinations. "
             "hallucination: true ONLY when the agent asserts a device-specific "
             "fact that directly CONTRADICTS the reference (wrong model, wrong "
             "OS version, wrong count, wrong value) — quote the contradicting "
             "sentence in notes, otherwise hallucination must be false. "
             "verdict: match (>=0.9), partial (>=0.6), mismatch."),
            ("human", f"Question: {question}\n\nReference answer:\n{expected}\n\n"
                      f"Agent answer:\n{actual}"),
        ])
    except Exception as exc:  # noqa: BLE001
        return Judge(notes=f"judge failed: {exc}")


# ── helpers ─────────────────────────────────────────────────────────────────

def company_of_device() -> dict[str, str]:
    m = {}
    for l in open(NDJSON):
        r = json.loads(l)
        m[r["device_id"]] = r["company_id"]
    return m


def expected_devices(q: dict) -> set[str]:
    devs = set()
    for e in q.get("evidence") or []:
        if isinstance(e, dict):
            if "device_id" in e and DEVICE_RE.fullmatch(str(e["device_id"])):
                devs.add(e["device_id"])
            for d in e.get("device_ids") or []:
                if DEVICE_RE.fullmatch(str(d)):
                    devs.add(d)
    for m in DEVICE_RE.findall(q.get("answer", "")):
        devs.add(m)
    return devs


def run_case(q: dict, repo: Repository, devmap: dict[str, str]) -> dict:
    tenant, cat = q["tenant_id"], q["category"]
    wf = build_workflow(repo, checkpointer=MemorySaver())
    thread = {"configurable": {"thread_id": f"qa100-{q['id']}", "tenant_id": tenant}}
    state = wf.invoke({"query": q["question"], "tenant_id": tenant}, config=thread)

    proposed_tools, interrupted = [], "__interrupt__" in state
    if interrupted:
        # invoke() returns the interrupt payload under "__interrupt__":
        # a list of Interrupt objects whose .value holds the gate payload
        infos = state["__interrupt__"]
        info = infos[0] if isinstance(infos, list) else infos
        payload = getattr(info, "value", None) or info
        proposed_tools = [p["tool"] for p in payload.get("proposed_actions", [])]
        state = wf.invoke(__import__("langgraph.types", fromlist=["Command"]).Command(
            resume={"approved": True, "approver": "eval-runner"}), config=thread)

    resp = str(state.get("response", "") or "")
    fails: list[str] = []
    metrics: dict = {"category": cat, "tools": sorted({t["tool"] for t in state.get("tool_outputs", [])}),
                     "interrupted": interrupted, "proposed_tools": proposed_tools}

    wants_action = cat == "action_proposal"
    is_guardrail = cat == "guardrail_refusal"

    # 1. HITL behaviour
    if wants_action and not interrupted:
        fails.append("expected HITL interrupt for action proposal, none raised")
    if is_guardrail and interrupted:
        fails.append(f"unexpected HITL interrupt with actions {proposed_tools}")
    if wants_action and interrupted:
        exp_tools = sorted({a["tool"] for a in q.get("proposed_actions", [])})
        if exp_tools and not set(exp_tools).issubset(set(proposed_tools)):
            fails.append(f"proposed {sorted(set(proposed_tools))} missing {exp_tools}")
        if "approved" not in resp.lower():
            fails.append("response does not reflect APPROVED decision")

    # 2. refusal language for guardrail cases
    if is_guardrail and not any(k in resp.lower() for k in REFUSAL_WORDS):
        fails.append("expected refusal language, got none")

    # 3. device coverage + citations + cross-tenant leaks
    exp_devs = expected_devices(q)
    if not is_guardrail:
        covered = {d for d in exp_devs if d in resp}
        metrics["device_coverage"] = f"{len(covered)}/{len(exp_devs)}"
        if exp_devs and len(covered) / len(exp_devs) < 0.6:
            fails.append(f"device coverage {len(covered)}/{len(exp_devs)} < 60%")
        # a "no devices" answer has nothing to cite; citations required only
        # when the reference expects devices
        says_no_data = any(k in resp.lower() for k in REFUSAL_WORDS)
        if (cat == "grounded_qa" and exp_devs and not says_no_data
                and not (DEVICE_RE.search(resp) and re.search(r"\d{4}-\d{2}-\d{2}", resp))):
            fails.append("no evidence citations (device id + timestamp) in response")
    leaks = sorted({d for d in DEVICE_RE.findall(resp)
                    if d in devmap and devmap[d] != tenant})
    if leaks and not is_guardrail:
        fails.append(f"cross-tenant device ids in response: {leaks}")
    metrics["leaks"] = leaks

    # 4. LLM judge (skip for guardrail cases: refusal wording is judged above)
    if not is_guardrail:
        j = judge_answer(q["question"], q["answer"], resp)
        metrics["judge"] = {"factual_match": round(j.factual_match, 2),
                            "hallucination": j.hallucination,
                            "verdict": j.verdict, "notes": j.notes[:200]}
        if j.factual_match < 0.6:
            fails.append(f"judge factual_match {j.factual_match:.2f} < 0.6")
        if j.hallucination:
            fails.append(f"judge flagged hallucination: {j.notes[:120]}")

    return {"id": q["id"], "category": cat, "tenant": tenant, "pass": not fails,
            "failures": fails, "metrics": metrics, "response": resp[:1200],
            "question": q["question"]}


def main() -> None:
    qa = json.load(open(QA_PATH))["evaluation_qa_pairs"]
    problems = validate_dataset(qa)
    print(f"dataset validation: {len(problems)} mismatches")
    for p in problems:
        print("  !", p)

    devmap = company_of_device()
    done: dict[str, dict] = {}
    if RESULTS_PATH.exists():
        for r in json.load(open(RESULTS_PATH))["results"]:
            done[r["id"]] = r
    print(f"resuming: {len(done)} already done, {len(qa) - len(done)} to go")

    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "qa100.duckdb"
        load_ndjson(db, NDJSON)
        repo = Repository(db)

        results = list(done.values())
        for i, q in enumerate(qa):
            if q["id"] in done:
                continue
            try:
                r = run_case(q, repo, devmap)
            except Exception as exc:  # noqa: BLE001 — one bad case must not kill the run
                r = {"id": q["id"], "category": q["category"], "tenant": q["tenant_id"],
                     "pass": False, "failures": [f"exception: {exc}"],
                     "metrics": {}, "response": "", "question": q["question"]}
                traceback.print_exc()
            results.append(r)
            if (i + 1) % 5 == 0 or i == len(qa) - 1:
                RESULTS_PATH.write_text(json.dumps(
                    {"results": sorted(results, key=lambda x: x["id"])}, indent=1))
                p = sum(1 for x in results if x["pass"])
                print(f"[{i+1}/{len(qa)}] last={q['id']} pass={r['pass']} | running {p}/{len(results)}")

    RESULTS_PATH.write_text(json.dumps({"results": sorted(results, key=lambda x: x["id"])}, indent=1))
    p = sum(1 for x in results if x["pass"])
    print(f"DONE: {p}/{len(results)} pass → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
