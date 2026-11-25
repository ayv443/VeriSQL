# Results

Qwen2.5-Coder-1.5B-Instruct with LoRA, trained on Spider with three methods and evaluated on Spider dev,
Spider-Syn, Spider-DK and BIRD dev (Phase 1, sections 1 to 6), then trained and evaluated in a
multi-turn environment where the model sees the database's reply and can fix its query (Phase 2,
section 7). All training ran on AWS SageMaker Studio (one NVIDIA T4 each). All Phase 1 evaluation ran on
one machine, so every model was scored the same way.

Every number below comes from a file in this repo:

| What | Where |
|---|---|
| Training logs, per-step metrics, checkpoint states, GPU / versions / AWS identity | [`text2sql-rl/proof/`](text2sql-rl/proof/) (one folder per model) |
| Scores, official evaluation output, every prediction per question | [`text2sql-rl/evaluation/results/`](text2sql-rl/evaluation/results/) |
| Bootstrap confidence intervals and McNemar tests | [`text2sql-rl/evaluation/results/statistics/`](text2sql-rl/evaluation/results/statistics/) |
| Feasibility run, data filtering, DPO pairs | [`text2sql-rl/evaluation/feasibility/`](text2sql-rl/evaluation/feasibility/), [`text2sql-rl/data/processed/tags.json`](text2sql-rl/data/processed/tags.json) |
| Console logs of the evaluation runs | [`text2sql-rl/evaluation/logs/`](text2sql-rl/evaluation/logs/) |
| Code | [`text2sql-rl/`](text2sql-rl/) (Phase 1), [`extension/`](extension/) (Phase 2) |
| Phase 2 results, statistics and proof of training | [`extension/findings/`](extension/findings/), [`extension/proof/`](extension/proof/) |

## 1. Data and feasibility

- Spider train (+ others): 8659 questions. 1616 dropped because the gold query returns an empty result,
  3 because the gold query fails. 7040 kept: 6631 train, 409 validation (10 held-out databases).
- 899 training prompts are longer than 2048 tokens and are skipped by every method, leaving 5732.
- Feasibility check with the base model (8 samples per question, temperature 0.8):

| pass@1 | pass@4 | pass@8 | all 8 wrong | mixed | all 8 right | valid SQL |
|---|---|---|---|---|---|---|
| 0.573 | 0.786 | 0.845 | 890 (15.5%) | 3457 (60.3%) | 1385 (24.2%) | 0.760 |

  GRPO and RLOO train only on the 3457 mixed questions, where the group of answers has some signal.
  DPO uses 7917 preference pairs from 4347 questions (for the 890 all-wrong questions the gold SQL is
  the chosen answer).

Files: `evaluation/feasibility/passk_train.json`, `filter_log.json`, `dpo_pairs.jsonl`, `data/processed/tags.json`.

## 2. Training runs

Shared settings: LoRA r 16, alpha 32 on all 7 projection layers, learning rate 5e-5, seed 0, fp16.
GRPO and RLOO: beta 0.04, 4 answers per question, 4 questions per step, temperature 0.8, 600 steps.

| | Steps | Time | s/step | Peak GPU |
|---|---|---|---|---|
| DPO | 495 / 495 (1 epoch) | 4h 15m 12s | 30.83 | 11.87 GB |
| GRPO | 600 / 600 | 5h 47m 25s | 34.72 | 8.48 GB |
| RLOO | 600 / 600 | 6h 32m 44s | 39.25 | 8.48 GB |

Training signals (mean of the first 10 vs the last 10 logged steps):

| | |
|---|---|
| DPO | preference accuracy 0.51 -> 0.89, reward margin 0.09 -> 3.49, but log-prob of the chosen answer -27.2 -> -44.6 (rejected -27.3 -> -79.5) |
| GRPO | execution reward 0.61 -> 0.81, valid-SQL reward 0.74 -> 0.94, KL 0.001 -> 0.046 |
| RLOO | execution reward 0.51 -> 0.82, valid-SQL reward 0.71 -> 0.93, KL 0.001 -> 0.034 |

DPO learns to separate chosen from rejected, yet the chosen answers also become less likely. That is
likelihood displacement, and it shows up in the validation curve below.

Files: `proof/<model>/summary.txt`, `proof/<model>/logs/*_steps.jsonl` (every step), `*.log`, `checkpoints/*/trainer_state.json`.

## 3. Validation curve (409 held-out questions, our EX)

| Step | 0 (base) | 100 | 200 | 300 | 400 | 495 / 500 | 600 |
|---|---|---|---|---|---|---|---|
| DPO | 0.619 | **0.579** | 0.491 | 0.443 | 0.428 | 0.411 | |
| GRPO | 0.619 | 0.709 | 0.716 | 0.736 | **0.743** | 0.738 | 0.741 |
| RLOO | 0.619 | 0.707 | 0.726 | 0.719 | 0.736 | 0.736 | **0.743** |

Best checkpoints (bold), used for every result below: DPO step 100, GRPO step 400, RLOO step 600.
GRPO and RLOO improve steadily; DPO falls below the base model and keeps falling.

Files: `evaluation/results/val_curve.csv`, `val_curve.png`, `evaluation/results/curve/`.

## 4. Our execution accuracy, with confidence intervals

Strict EX (the result must match exactly, column order included), greedy decoding.

| | Spider dev | Spider-Syn | Spider-DK | BIRD dev | Valid SQL (Spider dev) |
|---|---|---|---|---|---|
| base | 0.617 [0.588, 0.647] | 0.498 [0.467, 0.529] | 0.548 [0.505, 0.591] | 0.268 [0.246, 0.290] | 0.837 |
| DPO | 0.596 [0.567, 0.626] | 0.467 [0.437, 0.497] | 0.542 [0.499, 0.583] | 0.244 [0.222, 0.266] | 0.894 |
| GRPO | 0.715 [0.688, 0.743] | 0.588 [0.558, 0.618] | 0.622 [0.581, 0.664] | 0.332 [0.309, 0.355] | 0.929 |
| RLOO | 0.721 [0.695, 0.749] | 0.585 [0.555, 0.614] | 0.617 [0.576, 0.658] | 0.337 [0.314, 0.360] | 0.939 |

95% bootstrap intervals (10,000 resamples). McNemar test p-values on the same questions:

| | Spider dev | Spider-Syn | Spider-DK | BIRD dev |
|---|---|---|---|---|
| GRPO vs base | < 0.0001 | < 0.0001 | < 0.0001 | < 0.0001 |
| RLOO vs base | < 0.0001 | < 0.0001 | < 0.0001 | < 0.0001 |
| GRPO vs RLOO | 0.48 | 0.84 | 0.75 | 0.62 |
| DPO vs base | 0.18 | 0.043 (worse) | 0.86 | 0.040 (worse) |

GRPO and RLOO beat the base model on every benchmark. The difference between them is never significant.
DPO is never better than base and is significantly worse on Spider-Syn and BIRD.

Files: `evaluation/results/<model>/<benchmark>/metrics.json`, `per_question.jsonl`, `evaluation/results/statistics/`.

## 5. Official Spider evaluation (test-suite-sql-eval)

| | Spider dev EX | Spider dev TS | Spider-Syn EX | Spider-Syn TS | Spider-DK EX |
|---|---|---|---|---|---|
| base | 0.652 | 0.561 | 0.531 | 0.431 | 0.590 |
| DPO | 0.627 | 0.529 | 0.488 | 0.398 | 0.560 |
| GRPO | 0.746 | 0.653 | 0.614 | 0.502 | 0.654 |
| RLOO | **0.756** | **0.661** | **0.619** | **0.516** | **0.661** |

Spider dev by hardness (easy / medium / hard / extra):

| | EX | TS |
|---|---|---|
| base | 0.827 / 0.704 / 0.471 / 0.440 | 0.815 / 0.583 / 0.374 / 0.319 |
| DPO | 0.835 / 0.657 / 0.460 / 0.410 | 0.810 / 0.545 / 0.362 / 0.241 |
| GRPO | 0.871 / 0.778 / 0.655 / 0.566 | 0.867 / 0.650 / 0.557 / 0.440 |
| RLOO | 0.875 / 0.783 / 0.661 / 0.608 | 0.871 / 0.650 / 0.575 / 0.464 |

Notes:
- The official EX ignores column order, so it is about 3 points higher than our strict EX. The ranking is the same.
- Spider dev and Spider-Syn keep all 1034 questions. Spider-DK keeps 534 of 535: one gold query is broken
  in the dataset (missing comma) and is left out for every model.
- The official script's timeout cannot stop a running SQLite query and it loads full results into
  memory. Predictions that took over 60 s or returned 100,000+ rows were scored as wrong before scoring:
  DPO 10 (dev) / 12 (Syn) / 8 (DK), every other model 0.
- Spider-DK has no TS score: the test-suite databases do not include its 3 extra databases.

Files: `evaluation/results/<model>/spider_*/official_ex.txt`, `official_ts.txt`, `official_kept.txt`.

## 6. Official BIRD evaluation (full dev set, 1534 questions, %)

| | EX | Soft-F1 | R-VES |
|---|---|---|---|
| base | 29.14 | 30.63 | 27.17 |
| DPO | 27.84 | 29.91 | 25.77 |
| GRPO | 36.11 | 37.51 | 34.16 |
| RLOO | **36.96** | **38.63** | **34.74** |

EX by difficulty (simple / moderate / challenging): base 37.19 / 16.59 / 17.93, DPO 35.78 / 17.24 / 11.03,
GRPO 45.73 / 21.34 / 22.07, RLOO 46.81 / 22.41 / 20.69.

Same picture as Spider: GRPO and RLOO about 7 points above base on every metric, DPO slightly below,
with its largest drop on the challenging questions. The BIRD scripts print "mini dev set" in their
output, but they were given the full dev set. Predictions over 30 s or with 100,000+ rows were scored as
wrong first: base 2, DPO 20, GRPO 5, RLOO 4.

Files: `evaluation/results/<model>/bird_dev/official_bird.txt`, `official_bird_kept.txt`.

## 7. Phase 2: multi-turn evaluation and training

After each answer the query is run and the model sees the error, or the number of rows and the first 5
rows, and can keep or fix its query (up to 3 turns, greedy). The gold SQL is never shown. MT-GRPO and
MT-RLOO were trained in this environment for 150 more steps from GRPO step 400 and RLOO step 600 (same
data, beta, learning rate and temperature as Phase 1; step 150 picked on validation). Strict EX (%).

Fixed = wrong at turn 1, right at the end. Broken = right at turn 1, wrong at the end. Changed = the final
query differs from the first.

| Split | Model | Turn 1 | After feedback | Fixed | Broken | Changed |
|---|---|---|---|---|---|---|
| Spider dev | base | 61.7 | 62.7 | 20 | 10 | 158 |
| Spider dev | DPO | 59.6 | 56.2 | 9 | 44 | 118 |
| Spider dev | GRPO | 71.5 | 71.5 | 6 | 6 | 96 |
| Spider dev | GRPO step 600 (control) | 71.3 | 71.0 | 6 | 9 | 92 |
| Spider dev | MT-GRPO | 71.8 | 72.2 | 5 | 0 | 21 |
| Spider dev | RLOO | 72.1 | 71.9 | 6 | 9 | 118 |
| Spider dev | MT-RLOO | 71.5 | 71.7 | 5 | 3 | 46 |
| BIRD dev | GRPO | 33.2 | 34.4 | 25 | 7 | 305 |
| BIRD dev | MT-GRPO | 34.8 | 36.5 | 26 | 0 | 191 |
| BIRD dev | RLOO | 33.7 | 34.2 | 24 | 16 | 330 |
| BIRD dev | MT-RLOO | 31.6 | 32.9 | 25 | 4 | 188 |
| Validation | GRPO | 74.3 | 74.1 | 0 | 1 | 16 |
| Validation | MT-GRPO | 76.0 | 76.3 | 1 | 0 | 13 |
| Validation | RLOO | 74.3 | 74.6 | 3 | 2 | 21 |
| Validation | MT-RLOO | 73.6 | 74.1 | 2 | 0 | 9 |

Exact McNemar tests (questions where only the first / only the second has the property):

| | Spider dev | BIRD dev |
|---|---|---|
| Broken: GRPO vs MT-GRPO | 6 vs 0 (p = 0.031) | 7 vs 0 (p = 0.016) |
| Broken: RLOO vs MT-RLOO | 7 vs 1 (p = 0.070) | 13 vs 1 (p = 0.002) |
| Broken: both methods pooled | 13 vs 1 (p = 0.002) | 20 vs 1 (p < 0.0001) |
| Broken: GRPO step 600 vs MT-GRPO | 9 vs 0 (p = 0.004) | not run |
| Correct after feedback: GRPO vs MT-GRPO | 20 vs 28 (p = 0.31) | 38 vs 70 (p = 0.003) |
| Correct after feedback: RLOO vs MT-RLOO | 39 vs 37 (p = 0.91) | 78 vs 58 (p = 0.10) |
| Correct at turn 1: RLOO vs MT-RLOO | 36 vs 29 (p = 0.46) | 74 vs 41 (p = 0.003) |
| base: broken vs fixed by feedback | 10 vs 20 (p = 0.099) | not run |
| DPO: broken vs fixed by feedback | 44 vs 9 (p < 0.0001) | not run |

- Feedback alone changes the RL models' Spider accuracy by half a point or less: they break about as
  many answers as they fix. It helps more on BIRD, where many first answers fail.
- Multi-turn training makes revision safe: almost no correct answer is broken any more, on both test
  sets. The control shows this is not just the effect of 200 more training steps.
- The accuracy effect depends on the method: MT-GRPO is better on BIRD, MT-RLOO is worse at turn 1 on
  BIRD and not different after feedback. One seed per run, so this is promising rather than established.
- DPO breaks 44 correct answers when it sees feedback, which fits its likelihood displacement.

Files: `extension/findings/multiturn/summary.md`, `extension/findings/multiturn_statistics.md`,
`extension/findings/multiturn/<model>/<split>/all/`, `extension/proof/`.

## Summary

Both online RL methods (GRPO and RLOO) improve the 1.5B model by roughly 10 points on Spider dev, 7 to 9
on Spider-Syn and Spider-DK, and 7 on BIRD. They are statistically tied with
each other. Offline DPO on self-generated pairs does not help: its best checkpoint is slightly below the
base model, and training longer makes it worse because the chosen answers lose probability along with
the rejected ones.

In the multi-turn setting, training with the execution feedback taught the agent when to change its
answer: over both methods, correct answers broken by the feedback fell from 13 to 1 on Spider dev and from
20 to 1 on BIRD dev.
