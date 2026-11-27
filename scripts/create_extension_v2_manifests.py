"""Generate hash-verified batch and annotation manifests for extension-v2 (B21-B30)."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.review_generation import file_hash, protocol_signature, read_json

BENCHMARK_REL = "benchmarks/extension-v2.json"
AUDIT_FILES = [
    PROJECT_ROOT / "semantic-audits" / "B21-B25.json",
    PROJECT_ROOT / "semantic-audits" / "B26-B30.json",
]


def derive_verdict(assessment, answerable):
    decision = assessment["decision"]
    missing = [rc["claim"] for rc in assessment["required_claims"] if rc["status"] != "supported"]
    is_complete = len(missing) == 0

    if decision == "abstain":
        if not assessment["abstention_supported"]:
            return "unnecessary_abstention"
        if not answerable and not is_complete:
            return "partially_complete_abstention"
        return "appropriate_abstention"

    if not assessment["response_schema_and_quote_occurrence_valid"]:
        return "invalid_output"
    if not assessment["claims_grounded"]:
        return "unsupported_extrapolation" if is_complete else "unsupported_answer"
    if not is_complete:
        return "incomplete"
    if not assessment["quote_length_instruction_valid"]:
        return "pass_with_length_flag"
    return "pass"


def build_manifests():
    benchmark_path = PROJECT_ROOT / BENCHMARK_REL
    benchmark_hash = file_hash(benchmark_path)
    cases = {c["id"]: c for c in read_json(benchmark_path)}

    audits = []
    for path in AUDIT_FILES:
        audits.extend(read_json(path))

    by_branch = {"k3": [], "k5": []}
    for entry in audits:
        trace_path = PROJECT_ROOT / entry["trace_path"]
        actual_trace_hash = file_hash(trace_path)
        if entry["trace_sha256"].lower() != actual_trace_hash:
            raise ValueError(f"Trace hash mismatch for {entry['trace_path']}; re-review required.")
        if entry["benchmark_sha256"].lower() != benchmark_hash:
            raise ValueError(f"Benchmark hash mismatch for {entry['case_id']}; re-review required.")

        trace = read_json(trace_path)
        branch = f"k{trace['k']}"
        by_branch[branch].append((entry, trace, actual_trace_hash))

    batch_dir = PROJECT_ROOT / "runs" / "generation" / "review_batches"
    annot_dir = PROJECT_ROOT / "runs" / "generation" / "annotations"
    batch_dir.mkdir(parents=True, exist_ok=True)
    annot_dir.mkdir(parents=True, exist_ok=True)

    for branch in ("k3", "k5"):
        k_num = 3 if branch == "k3" else 5
        items = sorted(by_branch[branch], key=lambda x: x[0]["case_id"])
        case_ids = [entry["case_id"] for entry, _, _ in items]
        signatures = {protocol_signature(trace, expected_passages=k_num) for _, trace, _ in items}
        if len(signatures) != 1:
            raise ValueError(f"Mixed protocol signatures in extension-v2 {branch}.")
        sig = next(iter(signatures))

        traces = []
        annotations = []
        for entry, trace, trace_sha in items:
            cid = entry["case_id"]
            a = entry["assessment"]
            missing = [rc["claim"] for rc in a["required_claims"] if rc["status"] != "supported"]
            traces.append({
                "case_id": cid,
                "trace_path": entry["trace_path"],
                "trace_sha256": trace_sha,
            })
            annotations.append({
                "case_id": cid,
                "trace_path": entry["trace_path"],
                "trace_sha256": trace_sha,
                "prompt_version": trace["prompt_version"],
                "benchmark_sha256": benchmark_hash,
                "historical_benchmark_hash": entry.get("historical_benchmark_hash"),
                "provenance": (
                    "Manual passage-level semantic audit (semantic-audits/B21-B25.json, "
                    "B26-B30.json) bound to exact trace and current benchmark hashes; "
                    "historical benchmark hash unknown."
                ),
                "assessment": {
                    "claims_grounded": a["claims_grounded"],
                    "completeness": {
                        "is_complete": len(missing) == 0,
                        "missing_claims": missing,
                    },
                    "abstention_supported": a["abstention_supported"],
                    "protocol_valid": a["protocol_valid"],
                    "response_schema_and_quote_occurrence_valid": a["response_schema_and_quote_occurrence_valid"],
                    "quote_length_instruction_valid": a["quote_length_instruction_valid"],
                    "verdict": derive_verdict(a, cases[cid]["answerable"]),
                    "notes": a["notes"],
                },
            })

        batch_id = f"evidence-first-v1-extension-v2-{branch}"
        batch_path = batch_dir / f"{batch_id}.json"
        batch_path.write_text(
            json.dumps({
                "batch_id": batch_id,
                "benchmark_path": BENCHMARK_REL,
                "benchmark_sha256": benchmark_hash,
                "top_k": k_num,
                "case_ids": case_ids,
                "traces": traces,
                "protocol_signature": sig,
            }, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        annot_path = annot_dir / f"{batch_id}.json"
        annot_path.write_text(
            json.dumps({
                "annotation_version": "1",
                "annotations": annotations,
            }, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Saved {batch_path} and {annot_path}")


if __name__ == "__main__":
    build_manifests()
