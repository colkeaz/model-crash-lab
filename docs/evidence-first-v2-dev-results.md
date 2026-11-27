# Evidence-first v2 development experiment results

**Verdict: Do not advance `evidence-first-v2` (`advance = false`).**

Across the frozen 12-case, fixed `k=3` development slice (24 fresh generations), `evidence-first-v2` eliminated false abstentions (`6 -> 0`) and doubled clean answers (`2/12 -> 4/12`), but failed five advance-gate criteria: targeted false-abstention-to-clean-answer recovery reached `1/4` (`B22` only; gate requires `>= 2`), `B26` and `B29` regressed from grounded complete abstentions to substantive answers (`B29` failed to reject `stratify_ratio`), cases with overlength quotes increased (`1 -> 5`), and substantive answers with combined selective errors increased (`1/3 -> 7/11`).

These results are strictly from a 12-case development slice with overlapping topic clusters. No policy training or generalization claims are made.

---

## 1. Run provenance and verified controls

- **Run ID:** `20261126T135939902519Z` (`2026-11-26T13:59:39.983984+00:00` to `2026-11-26T14:19:05.192438+00:00`)
- **Runner:** `scripts/run_evidence_first_v2.py` (persists `request.raw.json` and `response.raw.json` before parsing; preserves malformed outputs and transport failures without retry)
- **Scorer:** `scripts/grade_evidence_first_v2.py` (blinded opaque IDs via `runs/generation/evidence_first_v2/20261126T135939902519Z/blind/sheet.json`)
- **Frozen-input verification:** `passed` (`338/338` checks, `0` errors) in `runs/generation/evidence_first_v2/20261126T135939902519Z/verification.json`
- **Model and runtime:** `qwen3:4b`, Ollama `0.40.10`, digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7` (verified unchanged before and after all 24 attempts)
- **Fixed request controls (both arms):** `num_ctx=4096`, `num_predict=512`, `temperature=0`, `think=false`, `stream=false`
- **Historical provenance note:** Fresh generation provenance is fully recorded. Missing historical metadata from the source `k=3` traces (including `historical_model_digest` and missing historical retrieval metadata) is preserved explicitly as `unknown`.

### Pinned artifact hashes (SHA-256)

| Artifact | Path | SHA-256 |
| --- | --- | --- |
| Paired dataset | `runs/policy_data/paired_questions_v1.json` | `93c7d781ea9679dc0136e05203ad7292156b3d99dd6334b73feb6a6ad129d0bb` |
| Chunks corpus | `corpus/chunks_v2/20261105T054527423001Z/chunks.json` | `33850c53b27c80d9d4423cef635ddf5370bb731f549944c8c76582d6e1ea64ea` |
| Pilot benchmark (`B01`, `B05`) | `benchmarks/pilot.json` | `a2a3fdbf5e87d3a566b9585053d50ddc82597ee7153159099d661b71c839204c` |
| Extension-v1 benchmark (`B11`, `B13`, `B17`) | `benchmarks/extension-v1.json` | `e4c23556120ebeab76b5ce87c9dd5e26392cd387767391035472bbe6dcb66fba` |
| Extension-v2 benchmark (`B21`–`B30`) | `benchmarks/extension-v2.json` | `3475ccdbb5626fe922ca75461428dd774ead186d4582af9d94c7d037ed2d935b` |
| `evidence-first-v1` system prompt | `scripts/run_paired_evaluation.py` (`.strip()`) | `f14cd38c0b2babb3e759d039afb401c34b2d2162677d5e4d20e773f48f9dc684` |
| `evidence-first-v2` system prompt | `docs/evidence-first-v2-dev-experiment.md` (UTF-8, LF, no trailing newline) | `de5b38d7c7144a30e716a09128fe1f24a6c1024d14d6e218c9de924568b8d470` |
| Sealed blind arm mapping | `runs/generation/evidence_first_v2/20261126T135939902519Z/blind/sealed_map.json` | `0d5f3b8c7c68a32c5a280a170cd8ccc0b76ccf8080478b0efe9e65bc9722d86b` |
| Blinded semantic annotations | `semantic-audits/evidence-first-v2-dev-20261126.json` | `775ea1132f151286d3360041a085e82b8ba311d50feceb16b07df80ca86fee73` |

---

## 2. Aggregate comparison (`k=3`, fresh control vs. `evidence-first-v2`)

| Metric | Fresh `v1` control | `evidence-first-v2` | Delta (`v2 - v1`) |
| --- | ---: | ---: | ---: |
| Attempts | `12` | `12` | `0` |
| Transport failures | `0/12` | `0/12` | `0` |
| Invalid outputs (structural/quotation audit) | `0/12` | `0/12` | `0` |
| Substantive answers (coverage) | `3/12` (`25.0%`) | `11/12` (`91.7%`) | `+8` |
| Abstentions | `9/12` (`75.0%`) | `1/12` (`8.3%`) | `-8` |
| Clean answers (grounded + complete + valid + 0 overlength) | `2/12` (`16.7%`) | `4/12` (`33.3%`) | `+2` |
| Complete answers (`is_complete=true`) | `2/12` (`16.7%`) | `5/12` (`41.7%`) | `+3` |
| Complete abstentions (`abstention_complete=true`) | `2/12` (`B26`, `B29`) | `0/12` | `-2` |
| False abstentions | `6/12` | `0/12` | `-6` |
| Unsupported substantive answers | `0/3` | `1/11` (`B29`) | `+1` |
| Cases with overlength quote (`>= 30` words) | `1/12` (`B05`) | `5/12` (`B05`, `B13`, `B21`, `B28`, `B30`) | `+4` |
| Total overlength excerpts (`>= 30` words) | `3` | `8` | `+5` |
| Combined selective error (substantive answers) | `1/3` (`33.3%`) | `7/11` (`63.6%`) | `+6` count (`+30.3 pp`) |
| Input tokens (known / missing) | `13,020` (`0` missing) | `14,136` (`0` missing) | `+1,116` (`+8.6%`) |
| Output tokens (known / missing) | `1,464` (`0` missing) | `3,700` (`0` missing) | `+2,236` (`+152.7%`) |
| Observed elapsed seconds (summed)* | `421.223s` | `741.986s` | `+320.763s` |

\*Observed elapsed seconds are single-run wall-clock observations, not a controlled speed benchmark. Historical 12-case slice cost reference was `13,026` input / `2,073` output tokens.

---

## 3. Advance gate evaluation

| Gate criterion | Requirement | Observed | Status |
| --- | --- | --- | --- |
| Control reproduces `>= 2` targeted false abstentions (`B13`, `B17`, `B21`, `B22`, `B27`) | `>= 2` (else recovery inconclusive) | `4` (`B13`, `B17`, `B21`, `B22`; `B27` answered cleanly in fresh `v1`) | Conclusive |
| Targeted false abstentions recovered to clean answer | `>= 2` of reproduced targets | `1/4` (`B22` only; `B13` incomplete + overlength, `B17` incomplete, `B21` complete but 1 overlength excerpt) | **FAIL** |
| Total clean answers increase | `v2 > v1` | `4 > 2` | PASS |
| Total false abstentions decrease | `v2 < v1` | `0 < 6` | PASS |
| `B01` and `B11` produce clean answers in `v2` | Both clean | `B01`: clean; `B11`: clean (recovered from fresh `v1` false abstention) | PASS |
| `B26`, `B29`, `B30` abstain with grounded reasons in `v2` | All 3 justified abstentions | `B26`: answered (incomplete); `B29`: answered (unsupported); `B30`: justified abstention | **FAIL** (`B26`, `B29`) |
| `B29` explicitly rejects `stratify_ratio` in `v2` | Abstain + reject `stratify_ratio` | Answered about `stratify` without rejecting `stratify_ratio` | **FAIL** |
| Complete control abstentions remain complete in `v2` | `B26`, `B29` stay complete abstentions | `0/2` preserved (both flipped to answers) | **FAIL** |
| Overlength-quote cases decrease (or stay 0 if control is 0) | `v2 < 1` | `5` vs `1` (`+4` cases) | **FAIL** |
| Invalid outputs do not increase | `v2 <= v1` | `0 <= 0` | PASS |
| Substantive answers with combined selective errors do not increase | `v2 <= v1` | `7 > 1` | **FAIL** |
| **Overall gate** | **All criteria pass** | **5 criteria failed** | **DO NOT ADVANCE** |

---

## 4. Per-case transitions (frozen order)

| Case | Fresh `v1` outcome | `v2` outcome | Quote word counts (`v1` / `v2`) | Tokens in/out (`v1` -> `v2`) | Transition summary |
| --- | --- | --- | --- | --- | --- |
| **B01** | `answer` — **clean** | `answer` — **clean** | `[24, 14]` / `[24, 14]` | `1539/242` -> `1632/242` | Identical clean answer (`0.25`). |
| **B05** | `answer` — incomplete, 3 overlength | `answer` — incomplete, 2 overlength | `[40, 37, 41]` / `[45, 25, 32]` | `1685/446` -> `1778/425` | Both omit optimistic-score/validation-target consequence; overlength excerpts `3 -> 2` (still selective error). |
| **B11** | `abstain` — false abstention | `answer` — **clean** | `[]` / `[29, 5, 9]` | `983/49` -> `1076/362` | Recovered to clean grounded answer (`'passthrough'` or `None`). |
| **B13** | `abstain` — false abstention | `answer` — incomplete, 1 overlength | `[]` / `[31]` | `928/45` -> `1021/154` | Named `memory` but omitted accepted types (`str` / `joblib.Memory`) and exceeded 29 words (`31`). |
| **B17** | `abstain` — false abstention | `answer` — incomplete | `[]` / `[8, 10]` | `834/45` -> `927/203` | Answered `"No"` without stating the required matching length / `shape[0]` rule. |
| **B21** | `abstain` — false abstention | `answer` — complete, 1 overlength | `[]` / `[10, 37]` | `625/52` -> `718/309` | Semantically complete and grounded, but blocked from clean answer by a 37-word excerpt. |
| **B22** | `abstain` — false abstention | `answer` — **clean** | `[]` / `[9, 9]` | `733/57` -> `826/250` | **Targeted recovery:** clean contrast of `int` sample count vs. `float` proportion. |
| **B26** | `abstain` — justified, complete | `answer` — grounded, incomplete | `[]` / `[27, 19, 21]` | `869/76` -> `962/393` | **Regression:** answered with general `Pipeline` capabilities even though passages omit practical advantages over manual transforms. |
| **B27** | `answer` — **clean** | `answer` — **clean** | `[23, 9]` / `[1, 15]` | `1244/233` -> `1337/224` | Fresh `v1` did not reproduce historical false abstention; both cleanly answer `np.random.mtrand._rand`. |
| **B28** | `abstain` — false abstention | `answer` — incomplete, 2 overlength | `[]` / `[60, 37]` | `997/78` -> `1090/382` | Structurally valid (resolved historical invalid quote), but omitted "entire dataset before splitting" and emitted 60- and 37-word quotes. |
| **B29** | `abstain` — justified, complete | `answer` — **unsupported** | `[]` / `[26, 7]` | `1461/55` -> `1554/297` | **Regression:** described `stratify` without rejecting fictitious `stratify_ratio`. |
| **B30** | `abstain` — justified, partial | `abstain` — justified, partial, 2 overlength | `[]` / `[37, 46, 25]` | `1122/86` -> `1215/459` | Both decline numeric runtime and omit supported estimator dependency; **hardware rubric caveat recorded separately** (frozen passages do not address hardware). `v2` added 2 overlength quotes (`37`, `46` words) on an abstention. |

---

## 5. Diagnostic findings and next-step implications

1. **Prompt wording shifted the model from over-abstention to over-answering.** Removing false abstentions (`6 -> 0`) proved the model *can* find the evidence at `k=3` (`B11`, `B21`, `B22`, `B28` all located relevant passages), but the broader "different wording" permission caused `qwen3:4b` to substitute adjacent concepts on `B26` (capabilities for practical advantages) and `B29` (`stratify` for `stratify_ratio`).
2. **Conciseness and quote-length instructions did not hold under `qwen3:4b`.** `v2` produced terse incomplete answers on `B13` (`"memory"`) and `B17` (`"No"`) while still violating the `< 30`-word quote limit on `5/12` cases (`8` excerpts, up to `60` words on `B28` and `46` words on `B30`).
3. **Why not advance `v2`:** Advancing `v2` would trade false abstentions for a `63.6%` (`7/11`) selective error rate and break unanswerable/false-premise safety (`B26`, `B29`), while increasing output tokens `2.53x` (`1,464 -> 3,700`).
4. **Evidence-backed options from here:**
   - **Targeted `v2.1` prompt revision** (tightening exact-entity/premise refusal and completeness-of-answer instructions, plus removing `evidence` quotes on pure abstentions where no factual claim is made, which caused `B30`'s overlength flags), or
   - **Hardened programmatic post-check / repair loop** (e.g., deterministic quote-length / entity-mismatch gate or `k=3` abstention gate), or
   - **Model comparison** under the same hardened frozen harness (`runs/generation/evidence_first_v2/`) to separate `qwen3:4b` instruction-following limits from prompt design.
