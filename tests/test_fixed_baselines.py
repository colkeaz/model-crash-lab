"""Tests for fixed baseline evaluation accounting and edge cases."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_fixed_baselines as efb


def _make_row(cid, eligible=True, k3_outcome=None, k5_outcome=None, costs=None, answerable=True):
    default_abstain = {
        "decision": "abstain",
        "response_parse_valid": True,
        "substantive_answer": False,
        "automatic_errors": [],
        "quote_status": "NO_QUOTES",
        "quotes_violating_under_30_words": 0,
        "protocol_valid": True,
        "claims_grounded": None,
        "is_complete": False,
        "missing_claims": [],
        "abstention_supported": False,
        "manual_verdict": "unnecessary_abstention",
        "verdict": "unnecessary_abstention",
    }
    return {
        "case_id": cid,
        "suite": "test",
        "answerable": answerable,
        "provenance": {
            "eligible": eligible,
            "cohort": "exploratory_unknown_provenance" if eligible else "excluded",
            "exclusion_reasons": [] if eligible else ["k3_annotation_stale"],
        },
        "k3_outcome": k3_outcome or dict(default_abstain),
        "k5_outcome": k5_outcome or dict(default_abstain),
        "costs": costs or {
            "k3_input_tokens": 100,
            "k3_output_tokens": 20,
            "k5_input_tokens": 200,
            "k5_output_tokens": 40,
        },
    }


class TestFixedBaselines(unittest.TestCase):
    def test_shared_denominator_invalid_answer_malformed_and_missing_tokens(self):
        # Row 1: k3 is substantive answer with quote mismatch (invalid_output), k5 is unparseable JSON
        r1 = _make_row(
            "T1",
            k3_outcome={
                "decision": "answer",
                "response_parse_valid": True,
                "substantive_answer": True,
                "automatic_errors": ["Evidence is empty, unretrieved, or absent from its passage."],
                "quote_status": "MISMATCH",
                "quotes_violating_under_30_words": 1,
                "protocol_valid": False,
                "claims_grounded": True,
                "is_complete": True,
                "missing_claims": [],
                "abstention_supported": None,
                "manual_verdict": "protocol_mismatch",
                "verdict": "invalid_output",
            },
            k5_outcome={
                "decision": None,
                "response_parse_valid": False,
                "substantive_answer": False,
                "automatic_errors": ["Response is not valid JSON."],
                "quote_status": "NO_QUOTES",
                "quotes_violating_under_30_words": 0,
                "protocol_valid": False,
                "claims_grounded": None,
                "is_complete": None,
                "missing_claims": None,
                "abstention_supported": None,
                "manual_verdict": None,
                "verdict": "invalid_output",
            },
            costs={
                "k3_input_tokens": 100,
                "k3_output_tokens": None,  # missing token count!
                "k5_input_tokens": 200,
                "k5_output_tokens": 50,
            },
        )
        # Row 2: excluded row must not enter any baseline denominator
        r2 = _make_row("T2", eligible=False)

        report = efb.evaluate_baselines({"rows": [r1, r2]})
        self.assertEqual(report["eligible_questions"], 1)
        self.assertEqual(report["excluded_questions"], 1)

        by_name = {b["baseline"]: b for b in report["baselines"]}
        # All baselines share eligible_questions == 1
        for b in by_name.values():
            self.assertEqual(b["eligible_questions"], 1)

        # k3_response: T1 is substantive answer with protocol violation -> selective error 1/1
        k3 = by_name["k3_response"]
        self.assertEqual(k3["substantive_answers"]["count"], 1)
        self.assertEqual(k3["combined_selective_error"]["count"], 1)
        self.assertEqual(k3["combined_selective_error"]["rate"], 1.0)
        self.assertEqual(k3["content_errors"]["count"], 0)
        self.assertEqual(k3["protocol_violations_in_answers"]["count"], 1)
        self.assertEqual(k3["tokens"]["output"]["known_subtotal"], 0)
        self.assertEqual(k3["tokens"]["output"]["missing_count"], 1)
        self.assertIsNone(k3["tokens"]["output"]["total"])

        # direct_k5: T1 is unparseable -> outside substantive-answer denominator, selective error undefined
        k5 = by_name["direct_k5"]
        self.assertEqual(k5["substantive_answers"]["count"], 0)
        self.assertEqual(k5["unparseable_invalid_attempts"]["count"], 1)
        self.assertIsNone(k5["combined_selective_error"]["rate"])
        self.assertEqual(k5["tokens"]["input"]["total"], 200)

        # k3->k5 recovery counts both generations (input 100+200=300; output missing 1, known subtotal 50)
        rec = by_name["k3→k5 recovery"]
        self.assertEqual(rec["tokens"]["input"]["total"], 300)
        self.assertEqual(rec["tokens"]["output"]["known_subtotal"], 50)
        self.assertEqual(rec["tokens"]["output"]["missing_count"], 1)
        self.assertIsNone(rec["tokens"]["output"]["total"])

        # immediate_abstain: zero generation tokens, 0 answers, undefined selective error
        imm = by_name["immediate_abstain"]
        self.assertEqual(imm["substantive_answers"]["count"], 0)
        self.assertIsNone(imm["combined_selective_error"]["rate"])
        self.assertEqual(imm["tokens"]["input"]["total"], 0)
        self.assertEqual(imm["tokens"]["output"]["total"], 0)


if __name__ == "__main__":
    unittest.main()
