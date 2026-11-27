"""Blinded semantic grading and scoring for the evidence-first v2 comparison.

prepare: emit a grading sheet with arm labels replaced by opaque IDs and a sealed
         arm mapping whose hash is recorded before any grading happens.
score:   validate annotation hashes, join the sealed mapping, and report metrics
         plus the advance gate.
"""

import argparse
import hashlib
import json
import secrets
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGETED = {"B13", "B17", "B21", "B22", "B27"}
EXPECTED_ANSWER = {"B01", "B05", "B11", "B13", "B17", "B21", "B22", "B27", "B28"}
EXPECTED_ABSTAIN = {"B26", "B29", "B30"}
CASE_ORDER = [
    "B01", "B05", "B11", "B13", "B17", "B21",
    "B22", "B26", "B27", "B28", "B29", "B30",
]
ARMS = ("v1", "v2")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    return sha256_bytes(Path(path).read_bytes())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_run(run_dir):
    manifest = read_json(run_dir / "manifest.json")
    attempts = {}
    for case_id in CASE_ORDER:
        for arm in ARMS:
            arm_dir = run_dir / case_id / arm
            attempt = read_json(arm_dir / "attempt.json")
            trace_path = arm_dir / "trace.json"
            trace = read_json(trace_path) if trace_path.exists() else None
            attempts[(case_id, arm)] = {
                "attempt": attempt,
                "trace": trace,
                "trace_sha256": file_hash(trace_path) if trace_path.exists() else None,
                "request_sha256": file_hash(arm_dir / "request.raw.json"),
                "response_sha256": (
                    file_hash(arm_dir / "response.raw.json")
                    if (arm_dir / "response.raw.json").exists() else None
                ),
            }
    return manifest, attempts


def prepare(run_dir):
    manifest, attempts = load_run(run_dir)
    benchmark_cases = {}
    frozen = read_json(run_dir / "verification.json")["frozen_cases"]
    for entry in frozen:
        benchmark_cases[entry["case_id"]] = {
            "path": entry["benchmark_path"],
            "sha256": entry["benchmark_sha256"],
            "case": entry["benchmark_case"],
        }

    seed = secrets.token_hex(16)
    mapping = {"seed": seed, "created_at": datetime.now(timezone.utc).isoformat(), "map": {}}
    sheet = {"run_id": run_dir.name, "cases": [], "grading_rules": GRADING_RULES}

    for case_id in CASE_ORDER:
        ordered = sorted(ARMS, key=lambda arm: sha256_bytes(f"{seed}:{case_id}:{arm}".encode()))
        variants = []
        for index, arm in enumerate(ordered):
            anonymous_id = f"{case_id}-{chr(65 + index)}"
            mapping["map"][anonymous_id] = {"case_id": case_id, "arm": arm}
            record = attempts[(case_id, arm)]
            reply = record["attempt"].get("reply")
            variants.append({
                "anonymous_id": anonymous_id,
                "trace_sha256": record["trace_sha256"],
                "decision": record["attempt"]["decision"],
                "answer": record["attempt"]["answer"],
                "abstention_reason": record["attempt"]["abstention_reason"],
                "evidence": (reply or {}).get("evidence"),
            })
        case = benchmark_cases[case_id]
        sheet["cases"].append({
            "case_id": case_id,
            "question": case["case"]["question"],
            "passages": frozen[[e["case_id"] for e in frozen].index(case_id)]["passages"],
            "rubric": {
                "answerable": case["case"]["answerable"],
                "reference_answer": case["case"]["reference_answer"],
                "required_claims": case["case"]["required_claims"],
                "forbidden_claims": case["case"]["forbidden_claims"],
                "absence_reason": case["case"]["absence_reason"],
                "benchmark_path": case["path"],
                "benchmark_sha256": case["sha256"],
            },
            "expected_with_frozen_passages": (
                "abstain" if case_id in EXPECTED_ABSTAIN else "answer"
            ),
            "variants": variants,
        })

    blind_dir = run_dir / "blind"
    blind_dir.mkdir(exist_ok=True)
    (blind_dir / "sheet.json").write_text(
        json.dumps(sheet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    sealed_bytes = (
        json.dumps(mapping, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    (blind_dir / "sealed_map.json").write_bytes(sealed_bytes)
    sealed_hash = sha256_bytes(sealed_bytes)
    (blind_dir / "sealed_map.sha256").write_text(sealed_hash + "\n", encoding="utf-8")
    print(f"Sheet:  {blind_dir / 'sheet.json'}")
    print(f"Sealed: {blind_dir / 'sealed_map.json'} (not read during grading)")
    print(f"Sealed hash: {sealed_hash}")


GRADING_RULES = [
    "Grade each variant using only the question, the supplied passages, and the rubric.",
    "Do not infer or look up which arm produced a variant.",
    "substantive_answer: decision == 'answer' with nonempty answer text.",
    "claims_grounded: every material claim in the answer is supported by the supplied passages.",
    "is_complete: the answer covers every required claim that the supplied passages can support.",
    "missing_claims: required claims not covered by the answer.",
    "unsupported_claims: material answer claims not supported by the supplied passages.",
    "abstention_justified: for an abstention, whether the supplied passages genuinely lack "
    "what the question requires.",
    "abstention_complete: whether the abstention reason satisfies the rubric's required "
    "claims about the missing evidence or premise correction.",
    "missing_rubric_claims: rubric claims the supplied passages cannot support, recorded "
    "separately from abstention justification.",
    "Quote-length flags are recorded by the structural auditor and are not semantic judgments.",
]


def score(run_dir, annotations_path):
    manifest, attempts = load_run(run_dir)
    blind_dir = run_dir / "blind"
    sealed_hash = (blind_dir / "sealed_map.sha256").read_text(encoding="utf-8").strip()
    sealed_bytes = (blind_dir / "sealed_map.json").read_bytes()
    if sha256_bytes(sealed_bytes) != sealed_hash:
        raise SystemExit("Sealed arm mapping changed; refusing to score.")
    mapping = json.loads(sealed_bytes)

    annotations = read_json(annotations_path)
    frozen = {e["case_id"]: e for e in read_json(run_dir / "verification.json")["frozen_cases"]}
    by_id = {a["anonymous_id"]: a for a in annotations["annotations"]}
    if len(by_id) != len(annotations["annotations"]):
        raise SystemExit("Duplicate annotations.")
    if set(by_id) != set(mapping["map"]):
        raise SystemExit("Annotations do not cover exactly the sealed anonymous IDs.")

    rows = []
    for anonymous_id, entry in by_id.items():
        if anonymous_id not in mapping["map"]:
            raise SystemExit(f"Unknown anonymous ID: {anonymous_id}")
        case_id = mapping["map"][anonymous_id]["case_id"]
        arm = mapping["map"][anonymous_id]["arm"]
        record = attempts[(case_id, arm)]
        if entry["trace_sha256"] != record["trace_sha256"]:
            raise SystemExit(f"{anonymous_id}: trace hash mismatch.")
        if entry["benchmark_sha256"] != frozen[case_id]["benchmark_sha256"]:
            raise SystemExit(f"{anonymous_id}: benchmark hash mismatch.")
        rows.append({**entry, "case_id": case_id, "arm": arm})

    if len(rows) != len(CASE_ORDER) * 2:
        raise SystemExit(f"Expected 24 annotations, got {len(rows)}.")

    def substantive(row):
        return (
            attempts[(row["case_id"], row["arm"])]["attempt"]["decision"] == "answer"
            and bool((attempts[(row["case_id"], row["arm"])]["attempt"]["answer"] or "").strip())
        )

    arms = {}
    for arm in ARMS:
        arm_rows = {r["case_id"]: r for r in rows if r["arm"] == arm}
        cases = {}
        for case_id, row in arm_rows.items():
            record = attempts[(case_id, arm)]
            attempt = record["attempt"]
            quotes = attempt["quote_checks"]
            overlength_excerpts = sum(bool(q.get("violates_under_30_words")) for q in quotes)
            structural_invalid = bool(attempt["audit_errors"])
            is_substantive = substantive(row)
            if is_substantive:
                selective_error = bool(
                    not row["claims_grounded"]
                    or not row["is_complete"]
                    or structural_invalid
                    or overlength_excerpts
                )
            else:
                selective_error = None
            clean_answer = bool(
                is_substantive
                and row["claims_grounded"]
                and row["is_complete"]
                and not structural_invalid
                and overlength_excerpts == 0
            )
            false_abstention = bool(
                case_id in EXPECTED_ANSWER
                and not is_substantive
                and not row.get("abstention_justified")
            )
            cases[case_id] = {
                "case_id": case_id,
                "arm": arm,
                "status": attempt["status"],
                "decision": attempt["decision"],
                "answer": attempt["answer"],
                "abstention_reason": attempt["abstention_reason"],
                "substantive_answer": is_substantive,
                "claims_grounded": row["claims_grounded"],
                "is_complete": row["is_complete"],
                "missing_claims": row["missing_claims"],
                "unsupported_claims": row["unsupported_claims"],
                "abstention_justified": row["abstention_justified"],
                "abstention_complete": row["abstention_complete"],
                "missing_rubric_claims": row["missing_rubric_claims"],
                "structural_invalid": structural_invalid,
                "audit_errors": attempt["audit_errors"],
                "overlength_excerpts": overlength_excerpts,
                "quote_word_counts": [q.get("word_count") for q in quotes],
                "selective_error": selective_error,
                "clean_answer": clean_answer,
                "false_abstention": false_abstention,
                "input_tokens": attempt["input_tokens"],
                "output_tokens": attempt["output_tokens"],
                "elapsed_seconds": attempt["elapsed_seconds"],
                "notes": row.get("notes", ""),
            }

        substantial = [c for c in cases.values() if c["substantive_answer"]]
        selective_errors = sum(c["selective_error"] for c in substantial)
        summary = {
            "attempts": len(cases),
            "transport_failures": sum(c["status"] == "transport_failure" for c in cases.values()),
            "invalid_outputs": sum(c["structural_invalid"] for c in cases.values()),
            "answers": sum(c["decision"] == "answer" for c in cases.values()),
            "abstentions": sum(c["decision"] == "abstain" for c in cases.values()),
            "substantive_answers": len(substantial),
            "coverage": f"{len(substantial)}/{len(cases)}",
            "clean_answers": sum(c["clean_answer"] for c in cases.values()),
            "complete_responses": sum(c["is_complete"] for c in cases.values()),
            "completeness": f"{sum(c['is_complete'] for c in cases.values())}/{len(cases)}",
            "false_abstentions": sum(c["false_abstention"] for c in cases.values()),
            "unsupported_answers": sum(
                c["substantive_answer"] and not c["claims_grounded"] for c in cases.values()
            ),
            "overlength_cases": sum(c["overlength_excerpts"] > 0 for c in cases.values()),
            "overlength_excerpts": sum(c["overlength_excerpts"] for c in cases.values()),
            "selective_error": (
                f"{selective_errors}/{len(substantial)}"
                if substantial else "undefined (no substantive answers)"
            ),
            "selective_error_count": selective_errors,
            "input_tokens_known": sum(c["input_tokens"] or 0 for c in cases.values()),
            "input_tokens_missing": sum(c["input_tokens"] is None for c in cases.values()),
            "output_tokens_known": sum(c["output_tokens"] or 0 for c in cases.values()),
            "output_tokens_missing": sum(c["output_tokens"] is None for c in cases.values()),
        }
        arms[arm] = {"summary": summary, "cases": cases}

    transitions = {}
    for case_id in CASE_ORDER:
        v1 = arms["v1"]["cases"][case_id]
        v2 = arms["v2"]["cases"][case_id]
        transitions[case_id] = {
            "v1": {
                "decision": v1["decision"], "clean_answer": v1["clean_answer"],
                "false_abstention": v1["false_abstention"],
                "abstention_justified": v1["abstention_justified"],
            },
            "v2": {
                "decision": v2["decision"], "clean_answer": v2["clean_answer"],
                "false_abstention": v2["false_abstention"],
                "abstention_justified": v2["abstention_justified"],
            },
        }

    v1_cases = arms["v1"]["cases"]
    v2_cases = arms["v2"]["cases"]
    control_false_targets = [
        c for c in TARGETED if v1_cases[c]["false_abstention"]
    ]
    recovered_targets = [
        c for c in control_false_targets if v2_cases[c]["clean_answer"]
    ]

    gate = {
        "control_reproduced_targeted_false_abstentions": control_false_targets,
        "recovery_criterion_verdict": (
            "inconclusive"
            if len(control_false_targets) < 2
            else ("pass" if len(recovered_targets) >= 2 else "fail")
        ),
        "recovered_targets_to_clean_answer": recovered_targets,
        "clean_answers_increase": (
            v2_cases and sum(c["clean_answer"] for c in v2_cases.values())
            > sum(c["clean_answer"] for c in v1_cases.values())
        ),
        "false_abstentions_decrease": (
            sum(c["false_abstention"] for c in v2_cases.values())
            < sum(c["false_abstention"] for c in v1_cases.values())
        ),
        "B01_clean_v2": v2_cases["B01"]["clean_answer"],
        "B11_clean_v2": v2_cases["B11"]["clean_answer"],
        "B26_grounded_abstention_v2": bool(
            v2_cases["B26"]["decision"] == "abstain"
            and v2_cases["B26"]["abstention_justified"]
        ),
        "B29_grounded_abstention_v2": bool(
            v2_cases["B29"]["decision"] == "abstain"
            and v2_cases["B29"]["abstention_justified"]
        ),
        "B30_grounded_abstention_v2": bool(
            v2_cases["B30"]["decision"] == "abstain"
            and v2_cases["B30"]["abstention_justified"]
        ),
        "B29_rejects_stratify_ratio_v2": bool(
            v2_cases["B29"]["decision"] == "abstain"
            and "stratify_ratio" in (v2_cases["B29"]["abstention_reason"] or "").lower()
        ),
        "control_complete_abstentions_preserved": [
            c for c in CASE_ORDER
            if v1_cases[c]["decision"] == "abstain" and v1_cases[c]["abstention_complete"]
        ],
        "overlength_rule": (
            "not_applicable" if arms["v1"]["summary"]["overlength_cases"] == 0
            else (
                "pass" if arms["v2"]["summary"]["overlength_cases"]
                < arms["v1"]["summary"]["overlength_cases"] else "fail"
            )
        ),
        "overlength_rule_detail": {
            "v1_overlength_cases": arms["v1"]["summary"]["overlength_cases"],
            "v2_overlength_cases": arms["v2"]["summary"]["overlength_cases"],
        },
        "invalid_outputs_do_not_increase": (
            arms["v2"]["summary"]["invalid_outputs"]
            <= arms["v1"]["summary"]["invalid_outputs"]
        ),
        "selective_errors_do_not_increase": None,
    }
    gate["selective_errors_do_not_increase"] = (
        arms["v2"]["summary"]["selective_error_count"]
        <= arms["v1"]["summary"]["selective_error_count"]
    )

    gate["advance"] = all([
        gate["recovery_criterion_verdict"] == "pass",
        gate["clean_answers_increase"],
        gate["false_abstentions_decrease"],
        gate["B01_clean_v2"],
        gate["B11_clean_v2"],
        gate["B26_grounded_abstention_v2"],
        gate["B29_grounded_abstention_v2"],
        gate["B30_grounded_abstention_v2"],
        gate["B29_rejects_stratify_ratio_v2"],
        all(
            v2_cases[c]["abstention_complete"]
            for c in gate["control_complete_abstentions_preserved"]
        ),
        gate["overlength_rule"] != "fail",
        gate["invalid_outputs_do_not_increase"],
        gate["selective_errors_do_not_increase"],
    ])

    report = {
        "run_id": run_dir.name,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "annotations_path": str(annotations_path),
        "annotations_sha256": file_hash(annotations_path),
        "sealed_map_sha256": sealed_hash,
        "model": manifest["model"],
        "model_digest": manifest["model_digest"],
        "ollama_version": manifest["ollama_version"],
        "settings": manifest["settings"],
        "prompts": manifest["prompts"],
        "arms": arms,
        "transitions": transitions,
        "gate": gate,
    }
    out = run_dir / "score.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"arms": {a: arms[a]["summary"] for a in ARMS}, "gate": gate},
                     indent=2, ensure_ascii=False))
    print(f"Saved score: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "score"])
    parser.add_argument("run_dir")
    parser.add_argument("--annotations")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    if args.command == "prepare":
        prepare(run_dir)
    else:
        if not args.annotations:
            raise SystemExit("--annotations is required for score.")
        score(run_dir, Path(args.annotations).resolve())


if __name__ == "__main__":
    main()
