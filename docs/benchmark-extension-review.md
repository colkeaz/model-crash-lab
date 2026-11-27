# B11–B20 draft review

The proposed extension is saved separately in `benchmarks/extension-v1.draft.json`. All ten cases remain draft. The original pilot and its hash are unchanged: `a2a3fdbf5e87d3a566b9585053d50ddc82597ee7153159099d661b71c839204c`.

## Changes to the proposals

| Case | Review decision |
| --- | --- |
| B11 | Keep the step replacement question. Skipping a transformer does not guarantee an already fitted downstream estimator will work without refitting. |
| B12 | Specify the output splits of the sparse input itself; accompanying labels need not become sparse matrices. |
| B13 | Keep the memory caching question and its directory/Memory interface requirement. |
| B14 | Classify named_steps access as a direct lookup. Include the property-specific anchor supporting its read-only description; this does not make returned estimators immutable. |
| B15 | Scope the comparison to stochastic training with otherwise identical conditions. An advancing RNG does not guarantee every pair of fitted outputs differs. |
| B16 | Replace the cloning premise with two estimators explicitly constructed using the exact same RandomState instance. Use the same estimator-randomness family as B15. |
| B17 | Keep as a paraphrase rather than a typo test. All inputs must have the same first dimension. |
| B18 | Keep training-only normalization statistics as the required claim. |
| B19 | Ask for documented auto_scale parameters and require declining to invent them. Missing retrieved evidence alone cannot establish that a method is globally absent. |
| B20 | Ask which seed is documented to guarantee the highest CV score. Require abstention from naming one and the source-supported caution about tuning the estimator RNG. This replaces the private-dataset question and avoids extending that caution to all split seeds. |

## B16 source conflict

The original proposal asks why estimators created by cloning one configured with a RandomState instance influence each other. The frozen S3 prose describes a shared RNG in this cloning example. However, the default implementation in [scikit-learn 1.7.2 base.py](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/base.py) recursively clones constructor parameters with `safe=False`, which deep-copies ordinary non-estimator objects. The [clone API documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.base.clone.html) also describes deep copying and allows custom clone delegation.

This is a source conflict identified by inspecting the pinned implementation, not a runtime reproduction. Do not make shared RNG identity under default cloning a required gold claim. The rewritten question uses explicit sharing, supported by [S3's estimator discussion](https://scikit-learn.org/1.7/common_pitfalls.html#estimators). External implementation evidence was used to review the benchmark; it has not been added to the frozen retrieval corpus.

## Validation and next experiment

Checked ten unique IDs B11–B20, schema field names against the pilot, corpus identity, draft status, answerability/absence-reason consistency, and every supplied anchor against the frozen HTML. These checks do not replace semantic review.

Several cases reuse existing development families. Treat this as a development extension, not an independent held-out test or evidence of broad generalization. Review the revised wording, freeze the extension, then generate paired top-3 and top-5 runs with the same model, prompt, generation settings, and corpus. Keep failed attempts and review annotations bound to their exact traces. Report completeness, grounding, abstention, validation failures, and observed generation costs separately.
