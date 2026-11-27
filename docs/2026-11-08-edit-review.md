# Review of November 8 edits

Read-only application review; no inference was run and no existing trace, benchmark or annotation was changed. Applied the Python review skill. A delegated read-only review covered feature extraction and manifest creation.

## Findings

1. **High: feature labels bypass provenance validation.** `scripts/extract_policy_features.py:124` indexes annotations by trace path alone, then exports their assessments without checking trace SHA256, benchmark SHA256, case identity or protocol signature. Modified traces can inherit stale labels. Reuse the batch audit and reject stale/mismatched inputs before exporting reviewed labels. Preserve the audit's effective invalid-output verdict separately from manual semantic judgments.

2. **High: protocol signature change breaks historical reviews.** `scripts/review_generation.py:91` adds k to the signed controls without a signature-version migration. The unchanged pilot batch is rejected; 10 of 22 existing review tests error at the pinned-signature check. Removing only the new k field from the signature calculation reproduces the historical pilot signature. Version the signature representation and preserve verification of old batches; do not disable signature checks.

3. **High: malformed model outputs can still disappear.** `scripts/run_paired_evaluation.py:179` and `:200` parse before saving the response. Invalid JSON aborts the run before that attempt's trace is written. Persist each request/response first, then report parse/validation failure and continue according to an explicit failure policy. The runner still records benchmark path rather than immutable active benchmark bytes/hash, and lacks a pinned chunks hash/model digest.

4. **Medium: historical retrieval features use current corpus state.** `scripts/extract_policy_features.py:20` loads hardcoded current chunks and recomputes margins, without validating the historical trace's corpus/chunks/settings. Bind features to frozen artifact hashes and validate supplied passage IDs/text. A corpus ID alone does not identify chunk boundaries.

5. **Medium: feature definitions misrepresent evidence and OOV.** At `scripts/extract_policy_features.py:75`, whitespace query tokens are compared with a word-token analyzer vocabulary; punctuation creates false OOVs. Use the fitted vectorizer's analyzer. At `:92`, an empty evidence array gets fidelity 1.0. Represent absence explicitly, separately from successful quote checks.

6. **Medium: manifest regeneration can rebind old case-level judgments.** `scripts/create_extension_manifests.py` combines fixed case-ID assessments with hashes from an audit file and overwrites manifests. If that audit is regenerated after a trace change, old judgments can appear newly hash-matched. Store reviewed trace hashes with the judgments and require explicit re-review for changed traces.

## Verified improvements and results

Both newly created extension-v1 branch batches audit successfully against their annotations with no stale annotations or trace-hash mismatches. k3 reports one invalid output and two quote-length flags; k5 reports one invalid output and five length flags. The revised B13/B14/B17/B18/B20 annotations reflect the earlier audit. These are retrospective review bindings, not proof that benchmark bytes were frozen before inference.

The new extension-v2 run `20261107T164628568864Z` has request questions matching the current B21–B30 file. Spot checks show B22 k5 omits the required float distinction; B29 k5 describes stratify rather than rejecting stratify_ratio. B28 k3 fails quotation validation. Decision flips alone remain insufficient recovery labels. Full semantic annotation of this new run is still outstanding.

Several extension-v2 families rename previously exercised topics: B22 overlaps split configuration, B28 overlaps preprocessing leakage, and B30 overlaps unavailable results. Assign family grouping before any train/test split; unique family strings do not establish independent concepts.

## Validation performed

- Ran `python -m unittest discover -s tests -p test_review_generation.py`: 22 tests, 10 errors caused by the historical signature mismatch.
- Ran the import-safe batch audit in memory for both extension-v1 branches and the pilot; extension branches succeeded, pilot failed signature verification.
- Checked all twenty new extension-v2 responses with the existing structural/quotation auditor. No model calls were made.

Fix provenance-safe feature export and signature compatibility before using the exported rows for policy training. Keep existing raw artifacts unchanged.
