"""Extract prediction-time features and attach provenance-verified audit labels."""

import argparse
import csv
import json
import sys
from functools import lru_cache
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.review_generation import audit_batch, file_hash, project_path, read_json

DEFAULT_CHUNKS_REL = "corpus/chunks_v2/20261105T054527423001Z/chunks.json"


def _optional_int(value):
    return value if type(value) is int else None


@lru_cache(maxsize=4)
def load_retrieval_index(chunks_path_str):
    chunks_path = Path(chunks_path_str)
    artifact = read_json(chunks_path)
    chunks = artifact["chunks"]
    chunks_index = {c["chunk_id"]: c for c in chunks}
    texts = [c["text"] for c in chunks]

    word_vec = TfidfVectorizer(ngram_range=(1, 1), token_pattern=r"(?u)\b\w+\b")
    word_vec.fit(texts)
    analyzer = word_vec.build_analyzer()
    vocab_set = set(word_vec.vocabulary_.keys())

    retriever = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", norm="l2")),
        ("char_wb", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), norm="l2")),
    ])
    retrieval_matrix = retriever.fit_transform(texts)

    return {
        "chunks_sha256": file_hash(chunks_path),
        "corpus_id": artifact.get("corpus_id"),
        "chunks_index": chunks_index,
        "analyzer": analyzer,
        "vocab_set": vocab_set,
        "retriever": retriever,
        "retrieval_matrix": retrieval_matrix,
    }


def compute_retrieval_margins(question, index):
    q_vec = index["retriever"].transform([question])
    scores = (index["retrieval_matrix"] @ q_vec.T).toarray().ravel()
    ranked = sorted((float(s) for s in scores if s > 0), reverse=True)

    s1 = ranked[0] if len(ranked) > 0 else 0.0
    s2 = ranked[1] if len(ranked) > 1 else 0.0
    s3 = ranked[2] if len(ranked) > 2 else 0.0

    return {
        "retrieval_top1_score": round(s1, 4),
        "retrieval_top2_score": round(s2, 4),
        "retrieval_top3_score": round(s3, 4),
        "margin_top1_top2": round(s1 - s2, 4),
        "margin_top1_top3": round(s1 - s3, 4),
        "score_ratio_top3_top1": round(s3 / (s1 + 1e-6), 4),
    }


def extract_runtime_features(trace, chunks_path=None):
    resolved_chunks = Path(chunks_path) if chunks_path else (PROJECT_ROOT / DEFAULT_CHUNKS_REL)
    index = load_retrieval_index(str(resolved_chunks.resolve()))

    if index["corpus_id"] and trace.get("corpus_id") != index["corpus_id"]:
        raise ValueError(
            f"Trace corpus_id {trace.get('corpus_id')} does not match chunks corpus_id {index['corpus_id']}."
        )

    req_messages = trace["request"]["messages"]
    user_payload = json.loads(
        next(m["content"] for m in req_messages if m["role"] == "user")
    )
    question = user_payload["question"]
    passages = user_payload["passages"]
    p_map = {p["chunk_id"]: p["text"] for p in passages}

    chunks_index = index["chunks_index"]
    passages_match_chunks = all(
        p["chunk_id"] in chunks_index and chunks_index[p["chunk_id"]]["text"] == p["text"]
        for p in passages
    )

    tokens = index["analyzer"](question)
    oov_count = sum(t not in index["vocab_set"] for t in tokens)
    oov_ratio = oov_count / max(len(tokens), 1)

    retrieval_stats = compute_retrieval_margins(question, index)
    unique_sources = len({
        chunks_index[p["chunk_id"]].get("source_id", "")
        for p in passages
        if p["chunk_id"] in chunks_index
    })

    raw_content = trace.get("response", {}).get("message", {}).get("content", "")
    try:
        response = json.loads(raw_content)
        parse_valid = isinstance(response, dict)
    except (json.JSONDecodeError, TypeError):
        response = None
        parse_valid = False

    if parse_valid:
        decision = response.get("decision")
        evidence = response.get("evidence") if isinstance(response.get("evidence"), list) else []
        answer = response.get("answer") if isinstance(response.get("answer"), str) else ""
        abstention_reason = (
            response.get("abstention_reason")
            if isinstance(response.get("abstention_reason"), str)
            else ""
        )
        quote_lengths = [
            len(ev["quote"].split())
            for ev in evidence
            if isinstance(ev, dict) and isinstance(ev.get("quote"), str)
        ]
        valid_quotes = sum(
            isinstance(ev, dict)
            and isinstance(ev.get("quote"), str)
            and bool(ev["quote"].strip())
            and isinstance(ev.get("chunk_id"), str)
            and ev["chunk_id"] in p_map
            and " ".join(ev["quote"].split()) in " ".join(p_map[ev["chunk_id"]].split())
            for ev in evidence
        )
        has_quotes = len(evidence) > 0
        quote_fidelity = round(valid_quotes / len(evidence), 4) if has_quotes else None
        model_decision = 1 if decision == "answer" else (0 if decision == "abstain" else None)
        answer_wc = len(answer.split())
        abstention_wc = len(abstention_reason.split())
    else:
        evidence = []
        quote_lengths = []
        has_quotes = False
        quote_fidelity = None
        model_decision = None
        answer_wc = None
        abstention_wc = None

    resp_meta = trace.get("response", {})
    return {
        "case_id": trace["case_id"],
        "k": trace.get("k", len(passages)),
        "chunks_sha256": index["chunks_sha256"],
        "passages_match_chunks_artifact": passages_match_chunks,
        "query_word_count": len(tokens),
        "query_char_count": len(question),
        "query_oov_ratio": round(oov_ratio, 4),
        "unique_sources_in_passages": unique_sources,
        **retrieval_stats,
        "response_parse_valid": parse_valid,
        "model_decision": model_decision,
        "quote_count": len(evidence),
        "has_quotes": has_quotes,
        "quote_total_words": sum(quote_lengths),
        "quote_max_words": max(quote_lengths) if quote_lengths else 0,
        "quote_fidelity_ratio": quote_fidelity,
        "answer_word_count": answer_wc,
        "abstention_word_count": abstention_wc,
        "input_tokens": _optional_int(resp_meta.get("prompt_eval_count")),
        "output_tokens": _optional_int(resp_meta.get("eval_count")),
    }


def extract_batch_features(
    root,
    batch,
    annotations,
    chunks_path=None,
    expected_passages=None,
    allow_unreviewed=False,
):
    report = audit_batch(root, batch, annotations, expected_passages=expected_passages)
    resolved_chunks = Path(chunks_path) if chunks_path else (PROJECT_ROOT / DEFAULT_CHUNKS_REL)

    rows = []
    for case_result in report["cases"]:
        if not case_result["batch_trace_hash_matches"]:
            raise ValueError(
                f"Trace hash mismatch for {case_result['case_id']} ({case_result['trace_path']})."
            )
        if case_result["annotation_status"] == "stale":
            raise ValueError(
                f"Stale annotation for {case_result['case_id']} ({case_result['trace_path']})."
            )
        if not allow_unreviewed and case_result["annotation_status"] != "matched":
            raise ValueError(
                f"Unreviewed trace for {case_result['case_id']} ({case_result['trace_path']})."
            )

        trace_path = project_path(root, case_result["trace_path"])
        trace = read_json(trace_path)
        features = extract_runtime_features(trace, chunks_path=resolved_chunks)

        completeness = case_result.get("completeness")
        quotes = case_result["quotation_fidelity"]["quotes"]
        features.update({
            "trace_path": case_result["trace_path"],
            "trace_sha256": case_result["trace_sha256"],
            "benchmark_sha256": batch["benchmark_sha256"],
            "annotation_status": case_result["annotation_status"],
            "automatic_error_count": len(case_result["automatic_errors"]),
            "quote_status": case_result["quotation_fidelity"]["status"],
            "quotes_violating_under_30_words": sum(
                q["violates_under_30_words"] for q in quotes
            ),
            "label_verdict": case_result["verdict"],
            "label_manual_verdict": case_result["manual_verdict"],
            "label_claims_grounded": case_result["claim_grounding"],
            "label_is_complete": (
                completeness.get("is_complete") if isinstance(completeness, dict) else None
            ),
            "label_abstention_supported": case_result["abstention_supported"],
        })
        rows.append(features)
    return rows, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, help="Batch manifest JSON")
    parser.add_argument("--annotations", required=True, help="Annotation manifest JSON")
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS_REL, help="Chunks JSON artifact")
    parser.add_argument("--top-k", type=int, default=None, help="Optional passage count check")
    parser.add_argument("--output", help="Optional output JSON path")
    args = parser.parse_args()

    batch_path = project_path(PROJECT_ROOT, args.batch)
    annot_path = project_path(PROJECT_ROOT, args.annotations)
    chunks_path = project_path(PROJECT_ROOT, args.chunks)

    batch = read_json(batch_path)
    annotations = read_json(annot_path)
    rows, _ = extract_batch_features(
        PROJECT_ROOT,
        batch,
        annotations,
        chunks_path=chunks_path,
        expected_passages=args.top_k,
    )

    print(f"\n--- Extracted {len(rows)} Feature Rows for Batch: {batch['batch_id']} ---")
    print(f"{'Case':<6} {'k':<3} {'Words':<6} {'OOV':<7} {'Top1':<7} {'Margin':<8} {'Quotes':<7} {'Fidelity':<9} {'Dec':<4} {'Verdict'}")
    print("-" * 80)
    for r in rows:
        fid_str = f"{r['quote_fidelity_ratio']:<9.2f}" if r["quote_fidelity_ratio"] is not None else f"{'N/A':<9}"
        dec_str = f"{r['model_decision']:<4}" if r["model_decision"] is not None else f"{'N/A':<4}"
        print(
            f"{r['case_id']:<6} {r['k']:<3} {r['query_word_count']:<6} {r['query_oov_ratio']:<7.2f} "
            f"{r['retrieval_top1_score']:<7.4f} {r['margin_top1_top2']:<8.4f} {r['quote_count']:<7} "
            f"{fid_str} {dec_str} {r['label_verdict']}"
        )

    out_file = project_path(PROJECT_ROOT, args.output or f"runs/policy_data/{batch['batch_id']}.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved feature dataset to: {out_file}")

    csv_file = out_file.with_suffix(".csv")
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV dataset to: {csv_file}")


if __name__ == "__main__":
    main()
