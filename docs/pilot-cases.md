# Pilot cases for human review

Status: draft. These are proposed development cases, not approved gold labels.
Source IDs refer to benchmark-design.md. All cases use the proposed three-page
scikit-learn 1.7.2 corpus; snapshot identifier is pending. Evidence locations below
were checked against the official pages on 2026-11-01.

For every case, apply the design's general rule: reject material unsupported or
incorrect extra claims and require supporting citations for substantive answers.
The listed failure examples supplement that rule. Review status: draft for all.

## B01: default split size

- Category: direct. Family: split-configuration. Answerable: yes.
- Question: In scikit-learn 1.7.2, what test fraction does train_test_split use
  when I omit both test_size and train_size?
- Reference / required claim: 25% is assigned to the test split.
- Failure example: 20%, or treating the default as unconditional when train_size
  is specified.
- Evidence: S1, Parameters > test_size.

- Reviewer notes: 
  - When both size parameters are omitted, the expected test size is 25%
  - However when train_size is supplied, the test size's value should complement the given train_size
  - The 25% being default should always fail.

## B02: pipeline step requirements

- Category: direct. Family: pipeline-contract. Answerable: yes.
- Question: Which methods must an intermediate Pipeline step implement, and
  which method must its final estimator implement?
- Reference / required claims: Intermediate steps need fit and transform; the
  final estimator needs fit.
- Failure example: Requiring predict on every step.
- Evidence: S2, class description and Parameters > steps.

## B03: nested parameter syntax

- Category: direct. Family: pipeline-parameters. Answerable: yes.
- Question: My Pipeline has a step named svc. How do I set its C parameter to
  10 using the pipeline's set_params method?
- Reference / required claim: Use pipe.set_params(svc__C=10), allowing any
  equivalent pipeline variable name.
- Failure example: svc_C or svc.C as the parameter key.
- Evidence: S2, class description and Examples.

## B04: safe preprocessing workflow

- Category: synthesis. Family: preprocessing-leakage. Answerable: yes.
- Question: I scaled all my data before splitting. Using train_test_split and
  a scaler/classifier Pipeline, explain the corrected training and evaluation order.
- Reference / required claims: Split first; fit the pipeline on training data;
  use it for test prediction, without fitting preprocessing on test data.
- Failure example: Calling fit_transform on the full dataset before splitting.
- Evidence: S1, Returns; S2, Examples; S3, How to avoid data leakage.

## B05: feature selection during cross-validation

- Category: synthesis. Family: preprocessing-leakage. Answerable: yes.
- Question: I used SelectKBest on the full dataset before cross-validation.
  What is wrong, and how should a Pipeline be evaluated instead?
- Reference / required claims: 
  - Selecting features on the full dataset lets validation-fold target values
    influence feature selection, potentially producing overly optimistic scores.
  - Put SelectKBest and the estimator inside a Pipeline, then cross-validate
    the entire pipeline.
  - Within each fold, fit feature selection and the estimator using only that
    fold's training data; apply the fitted pipeline to its validation data.
- Failure example: Performing selection once before cross-validation is safe.
- Evidence: S3, Data leakage during pre-processing; S2, class description.


## B06: test-time scaling

- Category: synthesis. Family: preprocessing-leakage. Answerable: yes.
- Question: My classifier was trained on scaled features but receives raw test
  features. Explain the mismatch and how a fitted Pipeline addresses it.
- Reference / required claims: Training and test representations differ; a
  fitted pipeline applies learned preprocessing before prediction.
- Failure example: Refitting the scaler on the test set.
- Evidence: S3, Inconsistent preprocessing; S2, Examples.

## B07: misspelled function name

- Category: noisy. Family: split-configuration. Answerable: yes.
- Question: With train_tset_split, how do I get the same randomized split each
  time for identical inputs?
- Reference / required claims: Recognize train_test_split; set random_state
  to a fixed integer.
- Failure example: An integer seed guarantees every model training result.
- Evidence: S1, Parameters > random_state.

## B08: misspelled stratification option

- Category: noisy. Family: split-configuration. Answerable: yes.
- Question: Can I use shufle=False and stratfy=y together in train_test_split?
- Reference / required claims: Interpret shuffle and stratify; no, disabling
  shuffling requires stratify=None.
- Failure example: Claiming that combination performs stratified splitting.
- Evidence: S1, Parameters > shuffle.

## B09: missing experiment outcome

- Category: unanswerable. Family: unavailable-results. Answerable: no.
- Question: What exact test accuracy will my unpublished dataset achieve with
  a scaler and classifier Pipeline?
- Reference / required behavior: Abstain from a numerical prediction; explain
  that actual data, configuration, and evaluation results are needed.
- Failure example: Inventing an accuracy or presenting a documentation example
  score as the user's result.
- Evidence: None supporting the requested outcome.
- Absence reason: No user experiment is supplied; the three pages cannot
  establish its result. This is a deliberately easy abstention control.

## B10: unsupported parameter premise

- Category: unanswerable. Family: unsupported-api-premise. Answerable: no.
- Question: In scikit-learn 1.7.2, what value should I give train_test_split's
  leakage_guard parameter to automatically prevent preprocessing leakage?
- Reference / required behavior: Reject the unsupported parameter premise;
  do not invent a value. An optional explanation may cite S3's leakage guidance.
- Failure example: Recommending leakage_guard=True or fabricating its behavior.
- Evidence: S1, signature and Parameters (the complete parameter list omits
  leakage_guard); S3, How to avoid data leakage, for an optional correction.
- Absence reason: The requested parameter is absent from the documented
  signature; there is no supported value to provide.
- Label nuance: The requested setting is unanswerable, but the false premise
  can be corrected with evidence. This differs from B09's missing experiment.

## Review checklist

- [ ] Verify each answerable case's claims against the cited sections.
- [ ] Challenge each unanswerable label using all three corpus pages.
- [ ] Check whether another reasonable answer should pass.
- [ ] Record reviewer notes and revise ambiguous criteria.
- [ ] Confirm the deliberately narrow corpus is appropriate for the pilot.
- [ ] Freeze source content and record hashes before running an experiment.

The learner has not yet reviewed or approved these cases. No model has answered
them, and no scores or performance claims have been produced.
