# Independent review of paired extension run

Original traces and paired_report.json were preserved. Mechanical checks and trace hashes are recorded in independent_audit.json. Semantic judgments below are reviewer assessments, not automatic entailment tests.

## Verified observations

The reported decisions and input token totals are correct: k3 answers 3/10, k5 answers 5/10; 11,165 versus 16,821 input tokens (+5,656, approximately 50.7%). Output tokens are 1,580 versus 2,419 (+839, approximately 53.1%). Every top-3 request is the exact prefix of its top-5 partner. Recorded model tag, options and system prompt agree across all 20 requests. A model tag alone does not establish immutable model weights.

Current extension file SHA256 is e4c23556120ebeab76b5ce87c9dd5e26392cd387767391035472bbe6dcb66fba, different from the reported 57d84ede4883ea3bc57cdb991baeee941a1781be506cf8e3655c88ee269acc53. All statuses are approved, and questions match requests. The historical benchmark bytes cannot be established because the report and traces do not record that hash.

| Case | Corrected interpretation |
| --- | --- |
| B11 | k3 answers the required claim; k5 abstains despite retaining supporting evidence. Observed context-expansion regression; one pair does not establish distraction as the mechanism. |
| B12 | Both answers satisfy the required claim; all emitted quotes match exactly. |
| B13 | The memory parameter is already rank 1 in k3. This is evidence-use recovery, not retrieval recovery. k5 answers only memory and omits the required accepted types. One quote violates the length instruction. |
| B14 | k5 obtains named_steps evidence from the newly added memory passage, not an Attributes chunk. It omits the required read-only description. Partial answer, not a complete rubric pass. |
| B15 | The full mutation/reset explanation already appears at rank 2 in k3. k5 gives the explanation, but one quote fails whitespace-normalized matching and one exceeds the length limit. Not a clean validated recovery. |
| B16 | Both answers cover the core shared-RNG mechanism. k3 has a quote mismatch; k5 has two overlong quotes. Neither is a protocol-clean stable pass. |
| B17 | The same-length requirement is rank 1 in both branches. Both abstentions are evidence-use failures, not retrieval misses. |
| B18 | The training-only preprocessing rule is rank 1 in both branches and supports the normalization-statistics inference. Both abstentions are evidence-use/generalization failures, not retrieval misses. |
| B19 | Both appropriately decline to invent the undocumented method's parameters. |
| B20 | Both refuse to name an optimal seed, but neither includes the required estimator-RNG tuning caution. k5 also has an overlong quote. Safe decision, incomplete rubric coverage. |

k3 emits 9 quotes: 1 normalized mismatch and 2 quotes violating under-30 words. k5 emits 17 quotes: 1 normalized mismatch and 5 length violations. Empty evidence is not counted as a verbatim-quote success. The mismatches include altered punctuation spacing, not necessarily invented substantive content; they still fail the declared quotation contract.

## Next steps

Keep decision correctness, required-claim completeness, grounding, quote validity and cost as separate labels. Complete a trace-bound annotation pass before assigning recovery utility. Do not train a policy from these ten correlated development cases alone or label every answer flip as a successful recovery.

For future runs, snapshot and hash the active benchmark bytes, chunks, prompt/schema and model digest before inference; save each raw response before parsing so malformed outputs remain reviewable. The current runner parses model JSON before saving and may lose invalid attempts. Run order is always k3 then k5, so observed timing differences are not a controlled latency comparison. Preserve this run and use additional family-separated cases and repeats to assess whether context effects recur.
