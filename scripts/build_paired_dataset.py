"""Build a leakage-safe, provenance-checked question-level paired dataset (k=3 vs k=5)."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_policy_features import (
    DEFAULT_CHUNKS_REL,
    extract_runtime_features,
    load_retrieval_index,
)
from scripts.review_generation import audit_batch, file_hash, project_path, read_json

DEFAULT_SUITES = [
    {
        "suite": "pilot",
        "k3_batch": "runs/generation/review_batches/evidence-first-v1-pilot.json",
        "k3_annotations": "runs/generation/annotations/evidence-first-v1-pilot.json",
        "k5_batch": "runs/generation/review_batches/evidence-first-v1-top5-pilot.json",
        "k5_annotations": "runs/generation/annotations/evidence-first-v1-top5-pilot.json",
    },
    {
        "suite": "extension-v1",
        "k3_batch": "runs/generation/review_batches/evidence-first-v1-extension-k3.json",
        "k3_annotations": "runs/generation/annotations/evidence-first-v1-extension-k3.json",
        "k5_batch": "runs/generation/review_batches/evidence-first-v1-extension-k5.json",
        "k5_annotations": "runs/generation/annotations/evidence-first-v1-extension-k5.json",
    },
    {
        "suite": "extension-v2",
        "k3_batch": "runs/generation/review_batches/evidence-first-v1-extension-v2-k3.json",
        "k3_annotations": "runs/generation/annotations/evidence-first-v1-extension-v2-k3.json",
        "k5_batch": "runs/generation/review_batches/evidence-first-v1-extension-v2-k5.json",
        "k5_annotations": "runs/generation/annotations/evidence-first-v1-extension-v2-k5.json",
    },
]

K3_FEATURE_KEYS = (
    "query_word_count",
    "query_char_count",
    "query_oov_ratio",
    "unique_sources_in_passages",
    "retrieval_top1_score",
    "retrieval_top2_score",
    "retrieval_top3_score",
    "margin_top1_top2",
    "margin_top1_top3",
    "score_ratio_top3_top1",
    "response_parse_valid",
    "model_decision",
    "quote_count",
    "has_quotes",
    "quote_total_words",
    "quote_max_words",
    "quote_fidelity_ratio",
    "answer_word_count",
    "abstention_word_count",
)


def _user_payload(trace):
    users = [m for m in trace["request"]["messages"] if m["role"] == "user"]
    return json.loads(users[0]["content"])


def _normalize_schema(value, allowed_ids):
    if isinstance(value, dict):
        return {k: _normalize_schema(v, allowed_ids) for k, v in value.items()}
    if isinstance(value, list):
        if value and all(isinstance(v, str) for v in value) and set(value) == allowed_ids:
            return ["<request-specific-chunk-ids>"]
        return [_normalize_schema(v, allowed_ids) for v in value]
    return value


def _load_verified_retrieval_meta(root, trace):
    run_id = trace.get("retrieval_run_id")
    expected_sha = trace.get("retrieval_results_sha256")
    if not run_id or not expected_sha:
        return None, "unknown"
    path = root / "runs" / "retrieval" / run_id / "results.json"
    if not path.exists():
        return None, "unknown"
    if file_hash(path) != expected_sha:
        return None, "mismatch"
    return read_json(path), "loaded"


def check_pair_compatibility(
    case,
    k3_trace,
    k5_trace,
    chunks_index,
    chunks_corpus_id,
    root=None,
    chunks_sha256=None,
    current_benchmark_sha256=None,
):
    p3 = _user_payload(k3_trace)
    p5 = _user_payload(k5_trace)

    q_check = (
        "matched"
        if p3.get("question") == p5.get("question") == case.get("question")
        else "mismatch"
    )

    corpora = [k3_trace.get("corpus_id"), k5_trace.get("corpus_id"), case.get("corpus_id"), chunks_corpus_id]
    if any(c is None for c in corpora):
        corpus_check = "unknown"
    elif len(set(corpora)) == 1:
        corpus_check = "matched"
    else:
        corpus_check = "mismatch"

    passages3 = p3.get("passages", [])
    passages5 = p5.get("passages", [])
    chunks_match = (
        "matched"
        if all(
            p["chunk_id"] in chunks_index and chunks_index[p["chunk_id"]]["text"] == p["text"]
            for p in (*passages3, *passages5)
        )
        else "mismatch"
    )

    prefix_check = (
        "matched"
        if len(passages3) >= 3
        and len(passages5) >= 3
        and [(p["chunk_id"], p["text"]) for p in passages3[:3]]
        == [(p["chunk_id"], p["text"]) for p in passages5[:3]]
        else "mismatch"
    )

    sys3 = [m for m in k3_trace["request"]["messages"] if m["role"] == "system"]
    sys5 = [m for m in k5_trace["request"]["messages"] if m["role"] == "system"]
    prompt_check = (
        "matched"
        if k3_trace.get("prompt_version") == k5_trace.get("prompt_version") and sys3 == sys5
        else "mismatch"
    )

    s3 = _normalize_schema(k3_trace["request"].get("format"), {p["chunk_id"] for p in passages3})
    s5 = _normalize_schema(k5_trace["request"].get("format"), {p["chunk_id"] for p in passages5})
    schema_check = "matched" if s3 == s5 else "mismatch"

    req3, req5 = k3_trace["request"], k5_trace["request"]
    settings_fields = ("model", "options", "think", "stream")
    if any(f not in req3 or f not in req5 for f in settings_fields):
        settings_check = "unknown"
    elif all(req3[f] == req5[f] for f in settings_fields):
        settings_check = "matched"
    else:
        settings_check = "mismatch"

    if root is not None:
        m3, st3 = _load_verified_retrieval_meta(root, k3_trace)
        m5, st5 = _load_verified_retrieval_meta(root, k5_trace)
        if st3 == "mismatch" or st5 == "mismatch":
            retrieval_meta_check = "mismatch"
        elif m3 is None or m5 is None:
            retrieval_meta_check = "unknown"
        else:
            same_retriever = (
                m3.get("retriever") == m5.get("retriever")
                and m3.get("settings") == m5.get("settings")
                and m3.get("corpus_id") == m5.get("corpus_id")
                and m3.get("chunks_sha256") == m5.get("chunks_sha256")
                and (chunks_sha256 is None or m3.get("chunks_sha256") == chunks_sha256)
            )
            retrieval_meta_check = "matched" if same_retriever else "mismatch"
    else:
        m3 = m5 = None
        retrieval_meta_check = "unknown"

    hb3 = k3_trace.get("benchmark_sha256") or (m3.get("benchmark_sha256") if m3 else None)
    hb5 = k5_trace.get("benchmark_sha256") or (m5.get("benchmark_sha256") if m5 else None)
    if hb3 is None or hb5 is None:
        hist_bench = "unknown"
    elif hb3 == hb5 and (current_benchmark_sha256 is None or hb3 == current_benchmark_sha256):
        hist_bench = "matched"
    else:
        hist_bench = "mismatch"

    hist_model = (
        "unknown"
        if k3_trace.get("model_digest") is None or k5_trace.get("model_digest") is None
        else ("matched" if k3_trace["model_digest"] == k5_trace["model_digest"] else "mismatch")
    )

    return {
        "question": q_check,
        "corpus": corpus_check,
        "passages_match_current_chunks": chunks_match,
        "top3_prefix": prefix_check,
        "prompt": prompt_check,
        "schema": schema_check,
        "settings": settings_check,
        "historical_retrieval_metadata": retrieval_meta_check,
        "historical_benchmark_hash": hist_bench,
        "historical_model_digest": hist_model,
    }


def _outcome_summary(case_result, runtime_features):
    completeness = case_result.get("completeness")
    quotes = case_result["quotation_fidelity"]["quotes"]
    quote_len_violations = sum(q["violates_under_30_words"] for q in quotes)
    answer = case_result.get("answer")
    decision = case_result.get("decision")
    parse_valid = runtime_features["response_parse_valid"]
    substantive = bool(parse_valid and decision == "answer" and isinstance(answer, str) and answer.strip())
    return {
        "decision": decision,
        "response_parse_valid": parse_valid,
        "substantive_answer": substantive,
        "automatic_errors": list(case_result["automatic_errors"]),
        "quote_status": case_result["quotation_fidelity"]["status"],
        "quotes_violating_under_30_words": quote_len_violations,
        "protocol_valid": len(case_result["automatic_errors"]) == 0 and quote_len_violations == 0,
        "claims_grounded": case_result["claim_grounding"],
        "is_complete": completeness.get("is_complete") if isinstance(completeness, dict) else None,
        "missing_claims": completeness.get("missing_claims", []) if isinstance(completeness, dict) else None,
        "abstention_supported": case_result["abstention_supported"],
        "manual_verdict": case_result["manual_verdict"],
        "verdict": case_result["verdict"],
    }


def build_suite_pairs(root, suite_cfg, chunks_path=None):
    resolved_chunks = Path(chunks_path) if chunks_path else (PROJECT_ROOT / DEFAULT_CHUNKS_REL)
    index = load_retrieval_index(str(resolved_chunks.resolve()))

    k3_batch = read_json(project_path(root, suite_cfg["k3_batch"]))
    k3_annots = read_json(project_path(root, suite_cfg["k3_annotations"]))
    k5_batch = read_json(project_path(root, suite_cfg["k5_batch"]))
    k5_annots = read_json(project_path(root, suite_cfg["k5_annotations"]))

    k3_report = audit_batch(root, k3_batch, k3_annots, expected_passages=3)
    k5_report = audit_batch(root, k5_batch, k5_annots, expected_passages=5)

    bench_cases = {
        c["id"]: c for c in read_json(project_path(root, k3_batch["benchmark_path"]))
    }
    k3_by_id = {c["case_id"]: c for c in k3_report["cases"]}
    k5_by_id = {c["case_id"]: c for c in k5_report["cases"]}

    all_case_ids = sorted(set(k3_by_id) | set(k5_by_id))
    rows = []
    for cid in all_case_ids:
        c3 = k3_by_id.get(cid)
        c5 = k5_by_id.get(cid)
        case = bench_cases[cid]

        exclusion_reasons = []
        if c3 is None:
            exclusion_reasons.append("missing_k3_trace")
        if c5 is None:
            exclusion_reasons.append("missing_k5_trace")
        if exclusion_reasons:
            rows.append({
                "case_id": cid,
                "suite": suite_cfg["suite"],
                "family_id": case.get("family_id"),
                "category": case.get("category"),
                "answerable": case.get("answerable"),
                "provenance": {
                    "eligible": False,
                    "cohort": "excluded",
                    "exclusion_reasons": exclusion_reasons,
                },
            })
            continue

        if not c3["batch_trace_hash_matches"]:
            exclusion_reasons.append("k3_trace_hash_mismatch")
        if not c5["batch_trace_hash_matches"]:
            exclusion_reasons.append("k5_trace_hash_mismatch")
        if c3["annotation_status"] != "matched":
            exclusion_reasons.append(f"k3_annotation_{c3['annotation_status']}")
        if c5["annotation_status"] != "matched":
            exclusion_reasons.append(f"k5_annotation_{c5['annotation_status']}")
        if k3_batch["benchmark_sha256"] != k5_batch["benchmark_sha256"]:
            exclusion_reasons.append("batch_benchmark_sha256_mismatch")

        k3_trace = read_json(project_path(root, c3["trace_path"]))
        k5_trace = read_json(project_path(root, c5["trace_path"]))

        compat = check_pair_compatibility(
            case,
            k3_trace,
            k5_trace,
            index["chunks_index"],
            index["corpus_id"],
            root=root,
            chunks_sha256=index["chunks_sha256"],
            current_benchmark_sha256=k3_batch["benchmark_sha256"],
        )
        mismatches = [k for k, v in compat.items() if v == "mismatch"]
        unknowns = [k for k, v in compat.items() if v == "unknown"]
        for m in mismatches:
            exclusion_reasons.append(f"compatibility_mismatch:{m}")

        eligible = len(exclusion_reasons) == 0
        if not eligible:
            cohort = "excluded"
        elif unknowns:
            cohort = "exploratory_unknown_provenance"
        else:
            cohort = "verified"

        f3 = extract_runtime_features(k3_trace, chunks_path=resolved_chunks)
        f5 = extract_runtime_features(k5_trace, chunks_path=resolved_chunks)

        rows.append({
            "case_id": cid,
            "suite": suite_cfg["suite"],
            "family_id": case.get("family_id"),
            "category": case.get("category"),
            "answerable": case.get("answerable"),
            "provenance": {
                "benchmark_path": k3_batch["benchmark_path"],
                "benchmark_sha256": k3_batch["benchmark_sha256"],
                "chunks_sha256": index["chunks_sha256"],
                "k3_trace_path": c3["trace_path"],
                "k3_trace_sha256": c3["trace_sha256"],
                "k5_trace_path": c5["trace_path"],
                "k5_trace_sha256": c5["trace_sha256"],
                "k3_annotation_status": c3["annotation_status"],
                "k5_annotation_status": c5["annotation_status"],
                "compatibility": compat,
                "mismatch_checks": mismatches,
                "unknown_checks": unknowns,
                "eligible": eligible,
                "cohort": cohort,
                "exclusion_reasons": exclusion_reasons,
            },
            "k3_features": {k: f3[k] for k in K3_FEATURE_KEYS},
            "k3_outcome": _outcome_summary(c3, f3),
            "k5_outcome": _outcome_summary(c5, f5),
            "costs": {
                "k3_input_tokens": f3["input_tokens"],
                "k3_output_tokens": f3["output_tokens"],
                "k5_input_tokens": f5["input_tokens"],
                "k5_output_tokens": f5["output_tokens"],
            },
        })
    return rows


def build_paired_dataset(root=PROJECT_ROOT, suites=None, chunks_path=None):
    suites = suites or DEFAULT_SUITES
    rows = []
    for cfg in suites:
        rows.extend(build_suite_pairs(root, cfg, chunks_path=chunks_path))

    eligible_rows = [r for r in rows if r["provenance"]["eligible"]]
    excluded_rows = [r for r in rows if not r["provenance"]["eligible"]]
    cohorts = {}
    for r in rows:
        c = r["provenance"]["cohort"]
        cohorts[c] = cohorts.get(c, 0) + 1

    return {
        "dataset_version": "1",
        "total_questions": len(rows),
        "eligible_questions": len(eligible_rows),
        "excluded_questions": len(excluded_rows),
        "cohorts": cohorts,
        "exclusions": [
            {"case_id": r["case_id"], "suite": r["suite"], "reasons": r["provenance"]["exclusion_reasons"]}
            for r in excluded_rows
        ],
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="runs/policy_data/paired_questions_v1.json",
        help="Output JSON path relative to project root",
    )
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS_REL, help="Chunks JSON artifact")
    args = parser.parse_args()

    dataset = build_paired_dataset(PROJECT_ROOT, chunks_path=project_path(PROJECT_ROOT, args.chunks))
    out_path = project_path(PROJECT_ROOT, args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"Built paired dataset: {dataset['total_questions']} questions "
        f"({dataset['eligible_questions']} eligible, {dataset['excluded_questions']} excluded)"
    )
    print("Cohorts:", json.dumps(dataset["cohorts"]))
    if dataset["exclusions"]:
        print("Exclusions:", json.dumps(dataset["exclusions"], indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
