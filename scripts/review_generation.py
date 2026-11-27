"""Audit explicit trace batches; attach only hash-matched manual annotations."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = "runs/generation/review_batches/evidence-first-v1-pilot.json"
DEFAULT_ANNOTATIONS = "runs/generation/annotations/evidence-first-v1-pilot.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Review inputs must be inside the project directory.")
    return path


def request_passages(trace, expected_count=None):
    users = [m for m in trace["request"]["messages"] if m["role"] == "user"]
    if len(users) != 1:
        raise ValueError("Expected one user message containing question and passages.")
    payload = json.loads(users[0]["content"])
    passages = {}
    for passage in payload["passages"]:
        chunk_id, text = passage["chunk_id"], passage["text"]
        if not isinstance(chunk_id, str) or not isinstance(text, str):
            raise ValueError("Supplied passage IDs and text must be strings.")
        if chunk_id in passages:
            raise ValueError("Duplicate supplied passage ID.")
        passages[chunk_id] = text
    if expected_count is not None and len(passages) != expected_count:
        raise ValueError(
            f"Expected {expected_count} supplied passages, got {len(passages)}."
        )
    return passages


def verify_quote(chunk_id, quote, passages, exclusive_word_limit=30):
    """Check occurrence in this request, separately from semantic support."""
    is_text = isinstance(quote, str)
    nonempty = is_text and bool(quote.strip())
    in_request = isinstance(chunk_id, str) and chunk_id in passages
    word_count = len(quote.split()) if is_text else None
    text = passages[chunk_id] if in_request else ""
    return {
        "in_supplied_passages": in_request,
        "nonempty": nonempty,
        "word_count": word_count,
        "exact_match": bool(nonempty and in_request and quote in text),
        "normalized_match": bool(
            nonempty and in_request
            and " ".join(quote.split()) in " ".join(text.split())
        ),
        "violates_under_30_words": (
            word_count is not None and word_count >= exclusive_word_limit
        ),
    }


def protocol_signature(trace, expected_passages=None):
    request = trace["request"]
    allowed = set(request_passages(trace, expected_count=expected_passages))

    def normalize_schema(value):
        if isinstance(value, dict):
            return {k: normalize_schema(v) for k, v in value.items()}
        if isinstance(value, list):
            if (value and all(isinstance(v, str) for v in value)
                    and set(value) == allowed):
                return ["<request-specific-chunk-ids>"]
            return [normalize_schema(v) for v in value]
        return value

    controls = {
        "prompt_version": trace["prompt_version"],
        "retrieval_run_id": trace.get("retrieval_run_id"),
        "retrieval_results_sha256": trace.get("retrieval_results_sha256"),
        "corpus_id": trace["corpus_id"],
        "model": request["model"],
        "options": request.get("options"),
        "think": request.get("think"),
        "stream": request.get("stream"),
        "system_messages": [
            m for m in request["messages"] if m["role"] == "system"
        ],
        "schema": normalize_schema(request.get("format")),
    }
    if "k" in trace:
        controls["k"] = trace["k"]
    return hashlib.sha256(
        json.dumps(controls, sort_keys=True).encode("utf-8")
    ).hexdigest()


def audit_trace(trace, passages, expected_passages=None):
    errors = []
    quote_checks = []
    if expected_passages is not None and len(passages) != expected_passages:
        errors.append(f"Expected {expected_passages} supplied passages, got {len(passages)}.")
    try:
        reply = json.loads(trace["response"]["message"]["content"])
    except json.JSONDecodeError:
        return None, [], ["Response is not valid JSON."]
    required = {"evidence", "decision", "answer", "abstention_reason"}
    if not isinstance(reply, dict) or set(reply) != required:
        return reply, [], ["Response must have exactly the four protocol fields."]
    if trace["response"].get("done_reason") != "stop":
        errors.append("Generation did not finish normally.")
    decision = reply["decision"]
    if decision not in ("answer", "abstain"):
        errors.append("Invalid decision.")
    strings_valid = all(
        isinstance(reply[field], str) for field in ("answer", "abstention_reason")
    )
    if not strings_valid:
        errors.append("Answer and abstention reason must be strings.")
    evidence = reply["evidence"]
    if not isinstance(evidence, list):
        errors.append("Evidence must be an array.")
        evidence = []
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"chunk_id", "quote"}:
            errors.append("Evidence entries need exactly chunk_id and quote.")
            continue
        check = verify_quote(item["chunk_id"], item["quote"], passages)
        quote_checks.append({**item, **check})
        if not check["normalized_match"]:
            errors.append("Evidence is empty, unretrieved, or absent from its passage.")
    if strings_valid and decision == "answer":
        if not reply["answer"].strip() or not evidence or reply["abstention_reason"]:
            errors.append("Answer decision requires text, evidence, and no abstention reason.")
    elif strings_valid and decision == "abstain":
        if reply["answer"] or not reply["abstention_reason"].strip():
            errors.append("Abstention requires an empty answer and a reason.")
    # Length is a separate instruction violation, not a truth judgment.
    return reply, quote_checks, errors


def audit_batch(root, batch, annotations, expected_passages=None):
    batch_top_k = batch.get("top_k")
    if batch_top_k is not None and (type(batch_top_k) is not int or batch_top_k < 1):
        raise ValueError("Batch top_k must be a positive integer.")
    if expected_passages is not None:
        if type(expected_passages) is not int or expected_passages < 1:
            raise ValueError("Expected passage count must be a positive integer.")
        if batch_top_k is not None and expected_passages != batch_top_k:
            raise ValueError("--top-k conflicts with the selected batch's pinned top_k.")
    else:
        expected_passages = batch_top_k
    benchmark_path = project_path(root, batch["benchmark_path"])
    benchmark_hash = file_hash(benchmark_path)
    if benchmark_hash != batch["benchmark_sha256"]:
        raise ValueError("Benchmark changed; explicitly define and review a new batch.")
    cases = {c["id"]: c for c in read_json(benchmark_path)}
    entries = batch["traces"]
    case_ids = [entry["case_id"] for entry in entries]
    if not entries or len(case_ids) != len(set(case_ids)):
        raise ValueError("Batch must contain one explicit trace per case.")
    if set(case_ids) != set(batch["case_ids"]) or not set(case_ids) <= set(cases):
        raise ValueError("Batch case list does not match its traces or benchmark.")
    loaded = []
    for entry in entries:
        path = project_path(root, entry["trace_path"])
        trace = read_json(path)
        if trace["case_id"] != entry["case_id"]:
            raise ValueError("Trace case ID does not match its batch entry.")
        loaded.append((entry, path, trace, protocol_signature(trace, expected_passages=expected_passages)))
    signatures = {item[3] for item in loaded}
    if len(signatures) != 1:
        raise ValueError("Batch mixes protocols, prompts, schemas, or retrieval/settings.")
    if signatures != {batch["protocol_signature"]}:
        raise ValueError("Batch protocol/settings differ from its pinned signature.")
    annotation_entries = annotations.get("annotations", [])
    by_path = {a["trace_path"]: a for a in annotation_entries}
    if len(by_path) != len(annotation_entries):
        raise ValueError("Duplicate manual annotations for a trace.")
    results = []
    for entry, path, trace, _ in loaded:
        actual_hash = file_hash(path)
        annotation = by_path.get(entry["trace_path"])
        status = "unreviewed"
        bound = (
            annotation is not None
            and annotation.get("trace_sha256") == actual_hash == entry["trace_sha256"]
            and annotation.get("case_id") == trace["case_id"]
            and annotation.get("prompt_version") == trace["prompt_version"]
            and annotation.get("benchmark_sha256") == benchmark_hash
        )
        if bound:
            status = "matched"
        elif annotation is not None:
            status = "stale"
        passages = request_passages(trace, expected_count=expected_passages)
        reply, quote_checks, errors = audit_trace(trace, passages, expected_passages=expected_passages)
        assessment = annotation["assessment"] if bound else {}
        verdict = assessment.get("verdict", "unreviewed")
        if errors:
            verdict = "invalid_output"
        if not quote_checks:
            quote_status = "NO_QUOTES"
        elif all(q["exact_match"] for q in quote_checks):
            quote_status = "VERBATIM"
        elif all(q["normalized_match"] for q in quote_checks):
            quote_status = "WHITESPACE_NORMALIZED_MATCH"
        else:
            quote_status = "MISMATCH"
        results.append({
            "case_id": trace["case_id"],
            "trace_path": entry["trace_path"],
            "trace_sha256": actual_hash,
            "batch_trace_hash_matches": actual_hash == entry["trace_sha256"],
            "prompt_version": trace["prompt_version"],
            "answerable": cases[trace["case_id"]]["answerable"],
            "decision": reply.get("decision") if isinstance(reply, dict) else None,
            "answer": reply.get("answer") if isinstance(reply, dict) else None,
            "abstention_reason": reply.get("abstention_reason") if isinstance(reply, dict) else None,
            "automatic_errors": errors,
            "quotation_fidelity": {"status": quote_status, "quotes": quote_checks},
            "annotation_status": status,
            "annotation_provenance": annotation.get("provenance") if bound else None,
            "claim_grounding": assessment.get("claims_grounded"),
            "completeness": assessment.get("completeness"),
            "abstention_supported": assessment.get("abstention_supported"),
            "manual_verdict": assessment.get("verdict"),
            "verdict": verdict,
            "notes": assessment.get("notes", ""),
        })
    quotes = [q for result in results for q in result["quotation_fidelity"]["quotes"]]
    summary = {
        "total_cases": len(results),
        "answers": sum(r["decision"] == "answer" for r in results),
        "abstentions": sum(r["decision"] == "abstain" for r in results),
        "invalid_outputs": sum(bool(r["automatic_errors"]) for r in results),
        "quotes_checked": len(quotes),
        "quotes_exact_matches": sum(q["exact_match"] for q in quotes),
        "quotes_normalized_matches": sum(q["normalized_match"] for q in quotes),
        "quotes_violating_under_30_words": sum(q["violates_under_30_words"] for q in quotes),
        "stale_annotations": sum(r["annotation_status"] == "stale" for r in results),
        "batch_trace_hash_mismatches": sum(not r["batch_trace_hash_matches"] for r in results),
        "verdicts": dict(Counter(r["verdict"] for r in results)),
    }
    return {
        "batch_id": batch["batch_id"], "expected_passages": expected_passages,
        "summary": summary, "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default=DEFAULT_BATCH)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional passage count; defaults to the selected batch's pinned top_k",
    )
    args = parser.parse_args()
    batch_path = project_path(PROJECT_ROOT, args.batch)
    annotation_path = project_path(PROJECT_ROOT, args.annotations)
    try:
        report = audit_batch(
            PROJECT_ROOT,
            read_json(batch_path),
            read_json(annotation_path),
            expected_passages=args.top_k,
        )
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(1, f"Audit rejected: {error}\n")
    report["batch_sha256"] = file_hash(batch_path)
    report["annotations_sha256"] = file_hash(annotation_path)
    report["reviewer_version"] = "2"
    review_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = PROJECT_ROOT / "runs" / "generation" / "reviews" / review_id
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "review.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for result in report["cases"]:
        print(f"[{result['case_id']}] {result['decision']} | Quotes: "
              f"{result['quotation_fidelity']['status']} | "
              f"Annotation: {result['annotation_status']} | Verdict: {result['verdict']}")
    print(f"Saved review: {output}")
    print(json.dumps(report["summary"], indent=2))
    if (report["summary"]["invalid_outputs"]
            or report["summary"]["batch_trace_hash_mismatches"]
            or report["summary"]["stale_annotations"]):
        parser.exit(1, "Audit found invalid outputs, changed traces, or stale annotations.\n")


if __name__ == "__main__":
    main()
