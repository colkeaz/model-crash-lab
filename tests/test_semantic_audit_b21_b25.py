"""Verification test for model-crash-lab/semantic-audits/B21-B25.json."""

import hashlib
import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_FILE = PROJECT_ROOT / "semantic-audits" / "B21-B25.json"
BENCHMARK_FILE = PROJECT_ROOT / "benchmarks" / "extension-v2.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


class TestSemanticAuditIntegrity(unittest.TestCase):
    def test_semantic_audit_integrity(self):
        self.assertTrue(AUDIT_FILE.exists(), f"Audit file not found: {AUDIT_FILE}")
        self.assertTrue(BENCHMARK_FILE.exists(), f"Benchmark file not found: {BENCHMARK_FILE}")

        benchmark_hash = sha256_file(BENCHMARK_FILE)
        benchmark_data = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
        benchmark_cases = {case["id"]: case for case in benchmark_data}

        audit_entries = json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
        self.assertIsInstance(audit_entries, list), "Audit file root must be a list"
        self.assertEqual(len(audit_entries), 10, f"Expected 10 paired entries (5 cases x 2 depths), got {len(audit_entries)}")

        observed_pairs = set()

        for idx, entry in enumerate(audit_entries):
            case_id = entry.get("case_id")
            trace_path_str = entry.get("trace_path")
            self.assertIn(case_id, benchmark_cases, f"Entry {idx}: Unknown case_id {case_id}")
            self.assertTrue(trace_path_str, f"Entry {idx}: Missing trace_path")

            trace_path = PROJECT_ROOT / trace_path_str
            self.assertTrue(trace_path.exists(), f"Entry {idx}: Trace file not found: {trace_path}")

            expected_trace_hash = sha256_file(trace_path)
            actual_trace_hash = entry.get("trace_sha256", "").upper()
            self.assertEqual(
                actual_trace_hash,
                expected_trace_hash,
                f"Entry {idx} ({case_id}): trace_sha256 mismatch. Expected {expected_trace_hash}, got {actual_trace_hash}",
            )

            actual_benchmark_hash = entry.get("benchmark_sha256", "").upper()
            self.assertEqual(
                actual_benchmark_hash,
                benchmark_hash,
                f"Entry {idx} ({case_id}): benchmark_sha256 mismatch. Expected {benchmark_hash}, got {actual_benchmark_hash}",
            )

            trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
            k = trace_data.get("k")
            self.assertIn(k, (3, 5), f"Entry {idx}: unexpected k value {k}")
            pair_key = (case_id, k)
            self.assertNotIn(pair_key, observed_pairs, f"Duplicate audit entry for {pair_key}")
            observed_pairs.add(pair_key)

            assessment = entry.get("assessment")
            self.assertIsInstance(assessment, dict), f"Entry {idx}: assessment must be a dict"
            decision = assessment.get("decision")
            self.assertIn(decision, ("answer", "abstain"), f"Entry {idx}: invalid decision {decision}")
            saved_response = json.loads(trace_data["response"]["message"]["content"])
            self.assertEqual(
                decision,
                saved_response.get("decision"),
                f"Entry {idx} ({case_id}, k={k}): recorded decision does not match saved response",
            )

            # Check required claims match benchmark
            benchmark_req_claims = benchmark_cases[case_id]["required_claims"]
            audited_claims = [rc["claim"] for rc in assessment.get("required_claims", [])]
            self.assertEqual(
                audited_claims,
                benchmark_req_claims,
                f"Entry {idx} ({case_id}): required_claims do not match benchmark.\n"
                f"Audited: {audited_claims}\nBenchmark: {benchmark_req_claims}",
            )

            # Check status of each claim
            for rc in assessment.get("required_claims", []):
                self.assertIn(
                    rc["status"],
                    ("supported", "missing", "unsupported"),
                    f"Entry {idx}: invalid claim status {rc['status']}",
                )
                self.assertIsInstance(rc.get("note"), str)
                self.assertGreater(len(rc["note"].strip()), 0, f"Entry {idx}: claim note must be non-empty string")

            # Check historical benchmark hash remains null
            self.assertIsNone(
                entry.get("historical_benchmark_hash"),
                f"Entry {idx}: historical_benchmark_hash must remain null",
            )

            # Logical consistency checks
            claims_grounded = assessment.get("claims_grounded")
            abstention_supported = assessment.get("abstention_supported")
            protocol_valid = assessment.get("protocol_valid")
            response_schema_valid = assessment.get("response_schema_and_quote_occurrence_valid")
            quote_len_valid = assessment.get("quote_length_instruction_valid")

            self.assertIsInstance(protocol_valid, bool), f"Entry {idx}: protocol_valid must be a boolean"
            self.assertIsInstance(response_schema_valid, bool), f"Entry {idx}: response_schema_and_quote_occurrence_valid must be bool"
            self.assertIsInstance(quote_len_valid, bool), f"Entry {idx}: quote_length_instruction_valid must be bool"

            if case_id == "B24":
                self.assertTrue(response_schema_valid, f"Entry {idx}: B24 should have response_schema_and_quote_occurrence_valid == True")
                self.assertFalse(quote_len_valid, f"Entry {idx}: B24 should have quote_length_instruction_valid == False (30 words)")
                self.assertFalse(protocol_valid, f"Entry {idx}: B24 protocol_valid should be False under full prompt-instruction compliance")
            else:
                self.assertTrue(response_schema_valid, f"Entry {idx}: {case_id} response_schema_and_quote_occurrence_valid should be True")
                self.assertTrue(quote_len_valid, f"Entry {idx}: {case_id} quote_length_instruction_valid should be True")
                self.assertTrue(protocol_valid, f"Entry {idx}: {case_id} protocol_valid should be True")

            if decision == "abstain":
                self.assertIsNone(claims_grounded, f"Entry {idx}: claims_grounded must be null when decision is abstain")
                self.assertIsInstance(
                    abstention_supported,
                    bool,
                    f"Entry {idx}: abstention_supported must be boolean when decision is abstain",
                )
            elif decision == "answer":
                self.assertIsInstance(
                    claims_grounded,
                    bool,
                    f"Entry {idx}: claims_grounded must be boolean when decision is answer",
                )
                self.assertIsNone(
                    abstention_supported,
                    f"Entry {idx}: abstention_supported must be null when decision is answer",
                )

            self.assertIsInstance(assessment.get("notes"), str)
            self.assertGreater(len(assessment["notes"].strip()), 0, f"Entry {idx}: notes must be non-empty string")

        expected_pairs = {(f"B{i}", k) for i in range(21, 26) for k in (3, 5)}
        self.assertEqual(observed_pairs, expected_pairs)


if __name__ == "__main__":
    unittest.main()

