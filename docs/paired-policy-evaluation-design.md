# Exploratory paired policy evaluation

## Goal and scope

Review the existing B21–B30 paired traces, combine verified development records at one row per question, and compare fixed recovery baselines. The 30 paired questions are not 60 independent policy examples. No model training or claims of generalization are in scope.

## Audit and provenance

Review each B21–B30 response against the actual passages and benchmark rubric. Record completeness, grounding, abstention appropriateness, and protocol validity separately. Bind manual judgments to exact trace hashes and the current benchmark hash; distinguish this from automated quote and schema checks. Preserve failed attempts and original traces. Where historical benchmark bytes, model digest, retrieval metadata, or execution settings were not captured, mark provenance unknown rather than inferring it. Verify actual supplied passages and paired prefixes.

## Dataset contract

Each row represents one question and contains separately: trace-bound identifiers and provenance, runtime features available by the k=3 decision point, k=3 outcome, k=5 outcome, and observed costs. Never expose k=5 responses, quotes, outcomes, or labels as k=3 decision features. Exclude or explicitly mark unreviewed, hash-mismatched, and missing-provenance records; do not silently fill in labels. Keep family identifiers for grouping and auditing, not model features. Treat empty quote sets as not applicable rather than verified fidelity.

Exclude unreviewed or hash-mismatched records from scored results. Include reviewed records with unknown historical provenance only in a clearly identified exploratory cohort, with unknowns visible. Before pairing, check question identity, corpus and chunk bindings, prompt, schema, recorded settings, and the top-three prefix. Mark unavailable checks unknown; never treat unknown as a match. Normalize only request-specific schema chunk IDs when comparing schemas.

## Fixed baseline contract

Compare (1) use the k=3 response, (2) expand and use the k=5 response, and (3) abstain without generation. Respect each selected response's actual decision; invalid or missing output must be accounted for explicitly and cannot be silently converted into a correct answer. Report answer coverage, selective error (incomplete, materially incorrect, unsupported, or protocol-invalid substantive answers divided by substantive answers), completeness, abstentions, invalid outputs, and input/output tokens. Selective error is undefined with no answers. For a policy that first runs k=3 then expands, count both generations; distinguish that from direct k=5 inference. State the cost assumption for immediate abstention and flag missing token data rather than assuming zero.

Also report content error and protocol violations separately, including quote-length violations separately from incorrect answers. Unparseable responses are invalid attempts outside the substantive-answer denominator; report their rate and count separately. Immediate abstention assumes zero generation tokens, explicitly excluding any retrieval or infrastructure cost.

A substantive answer is a parsed response with decision `answer` and nonempty answer text; include it in the selective-error denominator even when evidence or protocol validation fails. All baselines use the same eligible questions; report excluded questions and reasons. Name the expansion baseline `k3→k5 recovery` and report direct k5 inference separately. For missing token observations, report the known subtotal and missing count and leave the complete total unknown.

Known pair-compatibility mismatches exclude a question from scored results; unknown checks permit inclusion only in the explicitly marked exploratory cohort. Restore historical reviewer signature compatibility without weakening trace or benchmark hash verification. Completeness is the number of complete responses divided by eligible questions, with counts for answers and abstentions reported separately. For k3→k5 recovery every eligible question incurs both generations and the selected response is k5.

## Evaluation limits and checks

Report counts and denominators with rates. No threshold tuning is required for fixed baselines; any future tuning is exploratory. Group related question families together in splits and reserve untouched families for later evaluation. Test hash mismatch handling, missing and invalid responses, no-answer metrics, paired cost accounting, provenance flags, and prevention of k=5 feature leakage. Run existing relevant tests without modifying historical artifacts.
