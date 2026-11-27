# Model Crash Lab: pilot benchmark design

Date: 2026-11-01
Status: proposed milestone design; benchmark-first approach selected by the user.

## Purpose

Build a small, auditable benchmark for a documentation assistant before building
the assistant. Eventually test whether a learned answer/retrieve-again/abstain
policy reduces incorrect answers at a fixed inference budget and minimum coverage.
This milestone only defines and reviews the benchmark.

Working mode: the learner writes application code; the assistant explains,
reviews, and helps verify it, following the workspace GEMINI.md convention.

## Approaches considered

| Approach | Benefit | Cost | Decision |
| --- | --- | --- | --- |
| Benchmark first | Defines correctness before implementation | Slower route to a demo | Selected by user |
| Assistant first | Produces an interactive demo sooner | Evaluation may become an afterthought | Deferred |
| Predictor on existing outputs | Starts model training sooner | Less ownership of the complete system | Deferred |

## Proposed scope and evidence boundary

Use three English documentation pages under the scikit-learn 1.7 documentation
branch. On inspection on 2026-11-01 their page titles identify version 1.7.2.
This is a reproducibility choice, not a claim that 1.7.2 is the latest release.

| Source ID | Official page | Relevant sections |
| --- | --- | --- |
| S1 | [train_test_split](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.train_test_split.html) | Parameters and Returns |
| S2 | [Pipeline](https://scikit-learn.org/1.7/modules/generated/sklearn.pipeline.Pipeline.html) | Class description, steps, Examples |
| S3 | [Common pitfalls](https://scikit-learn.org/1.7/common_pitfalls.html) | Inconsistent preprocessing; Data leakage; How to avoid data leakage |

Only the substantive body of these three pages is in scope: navigation,
external links, and linked pages are excluded. Questions and gold answers never
enter the retrieval corpus. Inspecting the web pages is complete; no local corpus
snapshot has been downloaded yet. Before experiments, save the selected content
with source URL, observed version, section identifiers, acquisition timestamp,
and SHA-256 hash. Versioned URLs alone do not freeze page contents.

The proposed library, page selection, and record fields are assistant defaults
for review, rather than additional choices already confirmed by the user.

## Deliverables

1. Ten pilot cases: three direct, three synthesis, two noisy, two unanswerable.
2. A reviewed reference answer and explicit pass/fail criteria for every case.
3. A source manifest and frozen corpus before the first assistant experiment.
4. A human review record, including unresolved disagreements.

The companion pilot-cases.md is a draft review document. It is not a completed,
human-validated benchmark or a model training dataset.

## Proposed case fields

| Field | Meaning |
| --- | --- |
| id | Stable case identifier |
| question | Text shown to the assistant |
| category | direct, synthesis, noisy, or unanswerable |
| family_id | Groups related facts and paraphrases to prevent split leakage |
| corpus_id | Exact source snapshot identifier; pending snapshot creation |
| answerable | Whether the frozen evidence can support a sufficient answer |
| reference_answer | Human-reviewed semantic answer, not a required exact string |
| required_claims | Claims needed to answer adequately |
| forbidden_claims | Specific misleading assertions that cause failure |
| evidence | Source IDs and section identifiers supporting the answer |
| absence_reason | For unanswerable cases, what evidence is missing and why |
| review_status | draft, needs_revision, or approved |
| reviewer_notes | Reason for approval or requested changes |

Reference answers, answerability, family labels, and grading criteria are
evaluation metadata. They must not be passed to the assistant or used as
prediction-time features for a later failure predictor.

## Grading contract

For each response, record answer versus abstain separately from its quality.
An answer passes only if it includes all required claims, no material incorrect
or unsupported claim, and valid citations supporting its factual claims.
Accept equivalent wording. Record correctness and citation support separately
so a correct but unsupported answer remains diagnosable. Keywords alone are
insufficient for grading. A contradictory extra claim can fail an otherwise
correct response.

For an unanswerable case, success means identifying the evidence gap without
inventing an answer; a sensible request for missing information is acceptable.
For an answerable case, abstention isx` a missed answer, even when cautious.
Absence is judged against the entire frozen corpus, not just retrieved chunks.
Correct general knowledge without corpus support does not satisfy this contract.

Later evaluation should report raw counts and:

- Coverage = substantive answers / all questions.
- Selective error = failed substantive answers / substantive answers; undefined
  when there are no substantive answers.
- Correct supported answer yield = passing substantive answers / all questions.
- Abstention behavior separately for answerable and unanswerable cases.
- End-to-end latency and inference cost, including recovery attempts.

An empty run has undefined rates. Preserve run configuration and model identity.
The fixed budget and minimum coverage are future experiment settings to select
using validation data, before final testing. No arbitrary target is approved yet.

## Splits and limitations

All ten cases are development-only. They check the workflow, not generalization.
They overlap in concepts and some synthesis cases may be answerable from one
page; do not call them verified multi-page reasoning cases. Later author genuinely
independent families and assign connected, overlapping cases to the same split.
Keep training, policy selection, and final evaluation separate.

Do not claim novelty, train a predictor, select thresholds, or report meaningful
performance from ten cases. Version conflicts, model routing, deployment, and a
dashboard are outside this milestone. This milestone needs no paid inference,
dependencies, environment activation, notebook, or executable script.

## Completion and first learner task

The milestone is complete when all ten cases have been reviewed, sources have
been frozen, and an independent reader could apply the rubric consistently.
Source inspection is complete; case review and corpus freezing remain open.

Start by reviewing B01: locate S1's test_size parameter and explain why omitting
both size parameters differs from explicitly specifying test_size. Then review
the other cases before translating the agreed format into a data file.
