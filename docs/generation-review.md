# Generation reviewer v2

The reviewer has two separate jobs:

1. Automatically audit recorded response structure, finish status, quotation
   provenance, and quote length.
2. Attach existing manual judgments only to the exact response that was reviewed.

It does not automatically decide whether an answer is factually correct, whether
a quote entails a claim, or whether all required claims are covered.

## Run the recorded pilot audit

From `model-crash-lab`:

```powershell
python scripts/review_generation.py
python -m unittest discover -s tests -p test_review_generation.py -v
```

The default batch is:

`runs/generation/review_batches/evidence-first-v1-pilot.json`

Its manual annotations are:

`runs/generation/annotations/evidence-first-v1-pilot.json`

Every invocation writes a separate `runs/generation/reviews/<timestamp>/review.json`.
The previous `latest_evidence_first_review.json` remains a legacy record; the
repaired reviewer does not overwrite it or any original generation trace.

## What is pinned

The batch lists one explicit trace per case, with each trace's SHA-256 hash. It
also pins the benchmark hash and a protocol signature covering the prompt
version, system instructions, model tag, generation options, thinking/streaming
settings, response schema, retrieval run/hash, and corpus ID. The schema's
request-specific chunk ID enumeration is normalized so different questions can
belong to the same protocol batch.

Model tags are recorded by the existing generation traces; these historical
traces do not prove a particular model-weight digest. Future generation capture
can record a digest independently. This repair does not invent missing metadata.

Annotations bind to trace path, trace hash, case ID, prompt version, and benchmark
hash. If response bytes change, an old annotation becomes `stale` and its manual
verdict is not applied. A new trace path starts `unreviewed`, even for a familiar
case ID. Mixed protocols/settings and changed benchmarks are rejected. The CLI
returns a nonzero exit status for stale annotations, changed batch traces, or
invalid recorded outputs, after saving the report where possible.

The report includes the batch and annotation file hashes for reproducibility.
Do not silently update a pinned hash to make a changed response pass an old
review; review the new trace and create a new batch/annotation record explicitly.

## Quote checks

Quotes are checked only against passages embedded in that exact model request,
not against a global union of every corpus version.

- Empty or whitespace-only quotes fail.
- Unretrieved IDs fail, even if their text exists somewhere else in the corpus.
- `exact_match` and `normalized_match` are recorded separately. The latter
  collapses whitespace; it is not called a verbatim match.
- Under 30 words means `len(quote.split()) < 30`. A 30-word quote violates the
  prompt just as a 31-word quote does.
- No quotes produces `NO_QUOTES`, not `VERBATIM`. An answer with no evidence is
  an invalid output; an abstention may have an empty evidence list.
- A real quote can still be irrelevant. Semantic judgments remain manual.

Quote-length violations are reported independently from answer truth/completeness.
The historical generation outputs remain unchanged, rather than being repaired
or discarded to improve an experimental score.

## Corrected pilot annotations

- B05 evidence-first: `incomplete`. The earlier transform-only-training error
  belongs to the baseline output. The evidence-first response still omits
  cross-validation of the complete pipeline and fold-specific selection fitting.
- B07: complete in terms of the requested integer-seed recommendation, but not
  grounded in the supplied evidence for the function-specific API claim.
- B06, B09, B10: answer-claim grounding and answer completeness are not applicable
  (`null`); abstention appropriateness is recorded separately.
- B04: missing split-first and explicit training-only pipeline fitting claims.

The manual judgments were migrated from the user's original entries and corrected
with the assistant. Their provenance is explicit; the program does not claim to
have inferred these semantic labels automatically.

Expected current summary: 10 cases, 7 answers, 3 abstentions, 24 exact quote
matches, 4 quote-length violations, 4 passes, 2 incomplete answers, 3 appropriate
abstentions, and 1 unsupported extrapolation.

## A future experiment

Create a separate batch JSON listing the specific trace files and hashes to
evaluate. Set `case_ids` to the declared set, the benchmark path/hash, and compute
`protocol_signature` using the reviewer's function on a trace from that protocol.
All selected traces must share that signature. Do not select the latest response
for each case across arbitrary protocols.

Create a separate annotation JSON with an `annotations` list. An empty list is
valid: all selected responses will be `unreviewed` while automatic checks still
run. For reviewed responses, use the existing annotation structure, bind the
actual file hash, and enter new manual judgments for those exact outputs.

```powershell
python scripts/review_generation.py --batch runs/generation/review_batches/NEW-BATCH.json --annotations runs/generation/annotations/NEW-BATCH.json
```

The obsolete optional positional run-directory argument is replaced by these
explicit batch and annotation options. The normal no-argument pilot command
continues to work.

## Top-five recovery experiment

Expected passage count belongs to the selected batch. The original pilot batch
pins `top_k: 3`; the separate B06 recovery batch pins `top_k: 5`. No global
five-passage default is applied to historical top-three traces. The optional
`--top-k` override must agree with the selected batch metadata.

The single-case top-five batch is deliberately separate and initially unreviewed:

```powershell
python scripts/review_generation.py --batch runs/generation/review_batches/evidence-first-v1-top5-b06.json --annotations runs/generation/annotations/evidence-first-v1-top5-b06.json
```

It contains only B06's `20261107T091251824537Z` trace. Do not interpret this
single-case report as a ten-case experiment result. The fresh annotation file
is empty so the earlier B06 abstention verdict cannot be applied to its new answer.

Validation after adding batch-specific passage counts: all 21 regression tests
passed; the original ten-case top-three audit and the separate one-case top-five
audit both completed. The new B06 trace has three exact quote matches and zero
length violations. Its semantic annotation remains `unreviewed`: the core scaling
explanation has relevant evidence now, but the feature-selection quote is unrelated
and the added leakage-prevention claim needs qualification. No new generation
requests or changes to original traces were made while validating the reviewer.

## Repair handoff

Applied the previously selected
[surgical-patch skill](C:/Users/Lenovo/.agents/skills/surgical-patch/SKILL.md):
reproduced the empty-quote, unretrieved-source, and boundary-count bugs, repaired
the reviewer, and added focused regression checks. The original reviewer is
archived at `docs/archive/review_generation_v1.py.txt`.

Validation: 18 standard-library unittest checks passed. The real ten-trace pilot
audit completed successfully. All migrated trace hashes still match the original
generation files. No model calls, generation-script edits, or regeneration were
needed for this repair.
