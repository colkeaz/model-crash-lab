"""Evaluate fixed recovery baselines on eligible question-level pairs."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.review_generation import project_path, read_json

CONTENT_ERROR_VERDICTS = {"incomplete", "unsupported_answer", "unsupported_extrapolation"}


def _rate(count, denom):
    return round(count / denom, 4) if denom else None


def _is_content_error(outcome):
    if not outcome["substantive_answer"]:
        return False
    if outcome["claims_grounded"] is False:
        return True
    if outcome["is_complete"] is False:
        return True
    if outcome.get("manual_verdict") in CONTENT_ERROR_VERDICTS:
        return True
    if outcome.get("verdict") in CONTENT_ERROR_VERDICTS:
        return True
    return False


def _is_protocol_violation(outcome):
    return bool(outcome["automatic_errors"]) or outcome["quotes_violating_under_30_words"] > 0


def _aggregate_tokens(values):
    known = [v for v in values if type(v) is int]
    missing = len(values) - len(known)
    subtotal = sum(known)
    return {
        "known_subtotal": subtotal,
        "missing_count": missing,
        "total": subtotal if missing == 0 else None,
    }


def _score_outcomes(name, outcomes, input_tokens_list, output_tokens_list, cost_assumption):
    n = len(outcomes)
    substantive = [o for o in outcomes if o["substantive_answer"]]
    n_sub = len(substantive)
    abstentions = sum(1 for o in outcomes if o["response_parse_valid"] and o["decision"] == "abstain")
    unparseable = sum(1 for o in outcomes if not o["response_parse_valid"])
    invalid_outputs = sum(1 for o in outcomes if bool(o["automatic_errors"]))

    content_errors = sum(1 for o in substantive if _is_content_error(o))
    protocol_violations_sub = sum(1 for o in substantive if _is_protocol_violation(o))
    protocol_violations_all = sum(1 for o in outcomes if _is_protocol_violation(o))
    quote_length_violations = sum(1 for o in outcomes if o["quotes_violating_under_30_words"] > 0)
    combined_selective_errors = sum(
        1 for o in substantive if _is_content_error(o) or _is_protocol_violation(o)
    )

    complete_answers = sum(1 for o in substantive if o["is_complete"] is True)
    complete_abstentions = sum(
        1 for o in outcomes
        if o["response_parse_valid"] and o["decision"] == "abstain" and o["is_complete"] is True
    )
    complete_total = sum(1 for o in outcomes if o["is_complete"] is True)
    completeness_unknown = sum(1 for o in outcomes if o["is_complete"] is None)

    return {
        "baseline": name,
        "eligible_questions": n,
        "cost_assumption": cost_assumption,
        "substantive_answers": {"count": n_sub, "denominator": n, "rate": _rate(n_sub, n)},
        "abstentions": {"count": abstentions, "denominator": n, "rate": _rate(abstentions, n)},
        "unparseable_invalid_attempts": {
            "count": unparseable,
            "denominator": n,
            "rate": _rate(unparseable, n),
        },
        "invalid_outputs": {
            "count": invalid_outputs,
            "denominator": n,
            "rate": _rate(invalid_outputs, n),
        },
        "content_errors": {
            "count": content_errors,
            "denominator": n_sub,
            "rate": _rate(content_errors, n_sub),
        },
        "protocol_violations_in_answers": {
            "count": protocol_violations_sub,
            "denominator": n_sub,
            "rate": _rate(protocol_violations_sub, n_sub),
        },
        "protocol_violations_all": {
            "count": protocol_violations_all,
            "denominator": n,
            "rate": _rate(protocol_violations_all, n),
        },
        "quote_length_violations": {
            "count": quote_length_violations,
            "denominator": n,
            "rate": _rate(quote_length_violations, n),
        },
        "combined_selective_error": {
            "count": combined_selective_errors,
            "denominator": n_sub,
            "rate": _rate(combined_selective_errors, n_sub),
        },
        "completeness": {
            "complete_total": complete_total,
            "denominator": n,
            "rate": _rate(complete_total, n),
            "complete_answers": complete_answers,
            "complete_abstentions": complete_abstentions,
            "completeness_unknown": completeness_unknown,
        },
        "tokens": {
            "input": _aggregate_tokens(input_tokens_list),
            "output": _aggregate_tokens(output_tokens_list),
        },
    }


def evaluate_baselines(dataset):
    rows = dataset["rows"]
    eligible = [r for r in rows if r["provenance"]["eligible"]]
    excluded = [r for r in rows if not r["provenance"]["eligible"]]

    k3_outcomes = [r["k3_outcome"] for r in eligible]
    k5_outcomes = [r["k5_outcome"] for r in eligible]
    abstain_outcomes = [
        {
            "decision": "abstain",
            "response_parse_valid": True,
            "substantive_answer": False,
            "automatic_errors": [],
            "quote_status": "NO_QUOTES",
            "quotes_violating_under_30_words": 0,
            "protocol_valid": True,
            "claims_grounded": None,
            "is_complete": False,
            "missing_claims": None,
            "abstention_supported": not bool(r["answerable"]),
            "manual_verdict": "immediate_abstain",
            "verdict": "immediate_abstain",
        }
        for r in eligible
    ]

    k3_in = [r["costs"]["k3_input_tokens"] for r in eligible]
    k3_out = [r["costs"]["k3_output_tokens"] for r in eligible]
    k5_in = [r["costs"]["k5_input_tokens"] for r in eligible]
    k5_out = [r["costs"]["k5_output_tokens"] for r in eligible]

    baselines = [
        _score_outcomes(
            "k3_response",
            k3_outcomes,
            k3_in,
            k3_out,
            "Single k=3 generation per eligible question.",
        ),
        _score_outcomes(
            "k3→k5 recovery",
            k5_outcomes,
            k3_in + k5_in,
            k3_out + k5_out,
            "Two generations per eligible question (k=3 followed by k=5 expansion); selected response is k=5.",
        ),
        _score_outcomes(
            "direct_k5",
            k5_outcomes,
            k5_in,
            k5_out,
            "Single direct k=5 generation per eligible question (no prior k=3 call).",
        ),
        _score_outcomes(
            "immediate_abstain",
            abstain_outcomes,
            [0] * len(eligible),
            [0] * len(eligible),
            "Assumes zero generation tokens; explicitly excludes retrieval and infrastructure costs.",
        ),
    ]

    cohorts = {}
    for r in eligible:
        c = r["provenance"]["cohort"]
        cohorts[c] = cohorts.get(c, 0) + 1

    return {
        "evaluation_version": "1",
        "total_questions": len(rows),
        "eligible_questions": len(eligible),
        "excluded_questions": len(excluded),
        "eligible_cohorts": cohorts,
        "exclusions": [
            {"case_id": r["case_id"], "suite": r["suite"], "reasons": r["provenance"]["exclusion_reasons"]}
            for r in excluded
        ],
        "baselines": baselines,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="runs/policy_data/paired_questions_v1.json",
        help="Paired question dataset JSON",
    )
    parser.add_argument(
        "--output",
        default="runs/policy_data/fixed_baselines_report_v1.json",
        help="Output report JSON path",
    )
    args = parser.parse_args()

    dataset = read_json(project_path(PROJECT_ROOT, args.dataset))
    report = evaluate_baselines(dataset)
    out_path = project_path(PROJECT_ROOT, args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"Fixed Baseline Evaluation ({report['eligible_questions']} eligible / "
        f"{report['total_questions']} total; cohorts={report['eligible_cohorts']})"
    )
    print("-" * 100)
    for b in report["baselines"]:
        sel = b["combined_selective_error"]
        sel_str = f"{sel['count']}/{sel['denominator']} ({sel['rate']:.2%})" if sel["rate"] is not None else "0/0 (undef)"
        cov = b["substantive_answers"]
        comp = b["completeness"]
        tok_in = b["tokens"]["input"]
        tok_out = b["tokens"]["output"]
        print(
            f"{b['baseline']:<18} | Cov: {cov['count']:>2}/{cov['denominator']} ({cov['rate']:.1%}) | "
            f"SelErr: {sel_str:<14} | Comp: {comp['complete_total']:>2}/{comp['denominator']} "
            f"(unk={comp['completeness_unknown']}) | InTok: {tok_in['total']} | OutTok: {tok_out['total']}"
        )
    print(f"\nSaved baseline report: {out_path}")


if __name__ == "__main__":
    main()
