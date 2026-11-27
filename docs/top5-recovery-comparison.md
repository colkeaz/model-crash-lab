# Evidence-first generation: top-three versus top-five retrieval

## Recorded inputs

- Top-three retrieval run: `20261107T043727597033Z`.
- Top-five retrieval run: `20261107T085502793343Z`.
- Model tag: `qwen3:4b`; prompt version: `evidence-first-v1`.
- Both: temperature 0, context 4096, output cap 512 tokens.
- Every question is unchanged; the first three passages in each top-five request
  exactly match its earlier top-three request. System instructions are unchanged.
- All ten attempts are retained, including B05's client-validation failure.
- Batch and annotation manifests pin explicit trace paths/hashes. No latest-per-case
  discovery or automatic replacement of failed attempts was used.

## Observations on this development pilot

| Measure | Top three | Top five |
| --- | ---: | ---: |
| Cases | 10 | 10 |
| Answers / abstentions | 7 / 3 | 9 / 1 |
| Complete grounded passes under the existing strict rubric | 4 | 4 |
| Quote occurrence matches | 24 / 24 | 25 / 26 |
| Invalid recorded outputs | 0 | 1 |
| Under-30-word instruction violations | 4 | 3 |
| Appropriately abstained unanswerable cases | 2 / 2 | 1 / 2 |
| Input tokens | 11,251 | 19,661 |
| Generated tokens | 3,169 | 3,251 |
| Observed elapsed seconds, summed | 97.30 | 93.70 |

Elapsed times are observations from individual requests, not a controlled speed
benchmark. Model loading, caching, execution order, and hardware state can affect
them; the shorter observed wall time is not evidence that five passages cost less.
The input-token load rose about 75%; the configured output cap was unchanged.

## Case reviews

- B01, B02, B03, B08: complete and supported, as before.
- B04: still incomplete; it does not specify splitting before fitting
  preprocessing and fitting the pipeline only on training data.
- B05: invalid quote attribution. The model quotes the sentence about
  cross-validation/hyper-parameter tuning from `How to avoid data leakage`, but
  cites the `Data leakage during pre-processing` chunk. The original saved
  response is retained unchanged. The audit reports the mismatch and continues
  to the remaining cases rather than discarding or retrying B05.
- B06: the core scaling question is now covered by relevant retrieved evidence.
  The model nevertheless adds the unqualified claim that using a fitted Pipeline
  prevents leakage and includes an irrelevant feature-selection quote. Under the
  existing rule against material unsupported extrapolation this is not a clean
  grounded pass. Completeness and grounding are recorded separately.
- B07: still recommends an integer seed without function-specific parameter
  documentation in the retrieved evidence. Correctness/completeness of the core
  recommendation does not establish its grounding.
- B09: regresses from abstention to the unsupported answer `0.5`, copying an
  accuracy from a synthetic documentation experiment. A genuine quoted number
  is not evidence about the user's unpublished dataset.
- B10: retains an appropriate abstention. Missing retrieved parameter evidence
  is not treated as proof that a parameter does not exist.

These are explicitly authored semantic assessments, not automatic findings of
the quote-checking program. They use the same completeness and grounding rules
as the earlier experiment.

## Run the complete top-five audit

```powershell
python scripts/review_generation.py --batch runs/generation/review_batches/evidence-first-v1-top5-pilot.json --annotations runs/generation/annotations/evidence-first-v1-top5-pilot.json
```

The command writes the complete report, then returns exit status 1 to flag B05's
invalid output. That status is expected and does not mean the audit omitted later
cases. The normal no-argument command still audits the original top-three batch.

## Implication

Increasing retrieval depth is not a uniformly beneficial recovery action on
this pilot. It brings in B06's missing material but can also encourage copying
irrelevant example results, as in B09. Keep top-three evidence-first as the
reference baseline. The paired attempts are useful recovery-action observations;
ten development cases are insufficient to train or validate a general recovery
policy. Preserve these results before collecting more independent questions.
