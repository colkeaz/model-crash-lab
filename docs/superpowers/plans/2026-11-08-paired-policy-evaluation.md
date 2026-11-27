# Paired Policy Evaluation Implementation Plan

**Goal:** Produce trace-bound development reviews, a leakage-safe paired dataset, and descriptive fixed-baseline metrics.

**Architecture:** Audit original traces independently, bind manually reviewed outcomes to hashes, then construct one record per compatible reviewed question and compute baseline metrics from those records. Unknown historical provenance stays explicit.

**Tech Stack:** Python, JSON, pytest.

**Spec:** `docs/paired-policy-evaluation-design.md`.

## Global constraints
- Preserve original traces and failed attempts. Never treat mechanical checks as semantic review.
- Exclude unreviewed and hash-mismatched questions from all scored baselines; report reasons and exploratory unknown provenance.
- k=3 features must not include k=5 outputs or labels; do not train a policy.
- Use identical eligible questions for all baselines; no tuning or generalization claims.

## Review focus
- Historical hashes missing: report unknown rather than matched.
- Paired requests disagree or cannot be compared: expose failed or unknown compatibility checks.
- Invalid substantive answers: include in selective-error denominator.
- Malformed output: count as invalid attempt outside the substantive-answer denominator.
- Missing token counts: preserve known subtotal and missing count; total remains unknown.

### Task 1: Audit and bindings

**Files:** `runs/generation/paired/20261107T164628568864Z/`, `runs/generation/annotations/`, `runs/generation/review_batches/`, `scripts/review_generation.py`, `tests/`.

- [ ] Inspect B21–B30 benchmark, both requests, responses, citations and retrieval evidence; manually record per-claim completeness, support, abstention, and protocol findings bound to exact trace and current benchmark hashes, with historical provenance marked unknown.
- [ ] Test hash mismatch, missing annotations and missing historical retrieval metadata; ensure they cannot silently acquire verified labels.
- [ ] Generate explicit review manifests and run review verification; report invalid responses without erasing any traces or treating mechanical verification as semantic correctness.

### Task 2: Exploratory paired dataset

**Files:** `scripts/extract_policy_features.py`, a focused paired dataset script, and tests in `tests/`.

- [ ] Add tests for question, corpus/chunk, prompt/schema/settings and prefix compatibility, including missing metadata and mismatches.
- [ ] Build one row per question from trace-bound assessments, isolate k=3 decision-time features from branch outcomes and costs, and report excluded and unknown-provenance records.
- [ ] Verify stale/missing bindings are excluded and k=5 fields cannot enter k=3 features; run focused tests.

### Task 3: Fixed baselines

**Files:** a focused baseline evaluation script, tests in `tests/`, and an exploratory report under `runs/`.

- [ ] Test shared eligible denominators, invalid substantive answers, malformed responses, zero answers, missing tokens, and immediate-abstain zero generation cost.
- [ ] Implement k3 response, k3→k5 recovery, direct k5 and immediate abstain; report completeness, content errors, protocol violations, combined selective error, coverage and token known subtotals/missing counts.
- [ ] Run the baseline comparison on reviewed eligible pairs, record exclusions and limitations, and run relevant regression tests.
