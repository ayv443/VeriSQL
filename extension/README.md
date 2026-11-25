# Phase 2: the multi-turn (self-checking) environment

Phase 1 is a single-turn task: the model writes one query, gets one reward and the episode ends. At the
level of whole queries that is a contextual bandit. Phase 2 makes it a real multi-step MDP. After every
answer the query is run and the model sees what the database returned, and it can fix its query. GRPO
and RLOO were then trained inside this environment.

The Phase 2 code only reads from `text2sql-rl/` (`common.py`, `reward.py`, `generation.py` and the
saved evaluation results); nothing there was changed.

## The environment

| | |
|---|---|
| State | question + schema + every earlier query and the database's reply to it |
| Action | a new query, or confirming the current one (repeating it, or replying with no new SQL) |
| Feedback | the error message, or "Your query ran and returned N rows. The first rows are: ..." plus "if this answers the question, write the same query again; if not, write a corrected query". The gold SQL is never shown |
| Reward | only at the end: score of the final query (+1 / 0 / -0.1) minus 0.05 for every change of query |
| End | the model keeps its query, or after 3 turns |

## Files

| File | What it does | GPU |
|---|---|---|
| `multiturn_eval.py` | Multi-turn evaluation of any model; `--summary` writes `findings/multiturn/summary.md` | Yes |
| `train_multiturn.py` | Multi-turn GRPO / RLOO training (`--method grpo` or `rloo`) | Yes |
| `check_ready.py` | Checks the data, the saved results and the adapters before a long run | No |
| `multiturn_stats.py` | Exact McNemar tests and pooled sign tests; writes `findings/multiturn_statistics.md` | No |
| `error_analysis.py` | Error categories of the saved Phase 1 predictions | No |
| `dpo_analysis.py` | Why DPO did not help, from its training log | No |
| `collect_proof_mt.sh` | Proof of training for a multi-turn run, written to `proof/<run>/` | No |

## The trainer

TRL 0.24 only trains on single-turn episodes, so `train_multiturn.py` is our own loop that copies TRL's
update. The only difference from the Phase 1 runs is the multi-turn episode.

- Log-probabilities from logits / temperature; KL to the untrained base model (LoRA switched off), beta 0.04.
- RLOO: leave-one-out baseline; the KL summed over the episode is subtracted from the reward; loss uses
  the summed log-probability of the episode.
- GRPO: advantage (R - group mean) / (group std + 1e-4); per-token k3 KL in the loss; loss divided by
  the number of model-written tokens in the step ("dapo").
- The episode is built from the exact sampled token ids. The gradient goes through every token the model
  wrote in every turn; the prompt and the feedback tokens are masked out.
- A reply with no new SQL (for example "Yes, the query returns the correct result") keeps the previous query.
- Same settings as Phase 1: LoRA r 16, learning rate 5e-5, temperature 0.8, 4 questions x 4 episodes per
  step, the same 3,457 mixed questions, seed 0.

The update was checked line by line against the TRL 0.24 source and numerically: the gradient matches an
independent implementation of TRL's formulas (cosine 1.00, norm ratio 1.00) for both methods.

## What was run

Training (one T4 each, about 2 h 10 min), checkpoints every 50 steps, step 150 picked on validation:

```bash
python extension/train_multiturn.py --method rloo --init_adapter <rloo-1.5b-s0/checkpoint-600> \
    --run_name mt-rloo-1.5b-s0 --max_steps 150
python extension/train_multiturn.py --method grpo --init_adapter <grpo-1.5b-s0/checkpoint-400> \
    --run_name mt-grpo-1.5b-s0 --max_steps 150
```

Evaluation (greedy, up to 3 turns, `--mode all`). Turn 1 of the Phase 1 models reuses their saved
answers, so it is exactly the Phase 1 result; the multi-turn models and the control generate turn 1:

| Split | Models |
|---|---|
| Spider dev | base, DPO, GRPO, RLOO, MT-GRPO, MT-RLOO, GRPO step 600 (control for the extra training) |
| BIRD dev | GRPO, RLOO, MT-GRPO, MT-RLOO |
| Validation | GRPO step 400, RLOO step 600, MT-GRPO and MT-RLOO at steps 50, 100, 150 |

```bash
python extension/multiturn_eval.py --tag mt-grpo --adapter <mt-grpo-1.5b-s0/checkpoint-150> --split bird_dev --regenerate
python extension/multiturn_eval.py --summary
python extension/multiturn_stats.py
```

## Results

The full tables are in [`../RESULTS.md`](../RESULTS.md) (section 7),
[`findings/multiturn/summary.md`](findings/multiturn/summary.md) and
[`findings/multiturn_statistics.md`](findings/multiturn_statistics.md).

- Multi-turn training makes self-correction safe. Over both methods, answers broken by the feedback
  (right at turn 1, wrong at the end) fell 13 -> 1 on Spider dev (p = 0.002) and 20 -> 1 on BIRD dev (p < 0.0001).
- It is not just extra training: GRPO step 600 (200 more single-turn steps) broke 9 answers on Spider
  dev against 0 for MT-GRPO (p = 0.004).
- Accuracy depends on the method: MT-GRPO is better than GRPO on BIRD (36.5% vs 34.4% after feedback,
  p = 0.003); MT-RLOO is not better than RLOO; neither changes significantly on Spider dev.

## Where things are

| | |
|---|---|
| Per-question results and metrics of every run | `findings/multiturn/<model>/<split>/all/` |
| Training logs of the two runs (every step, sample episodes, settings) | `findings/mt_training/grpo/`, `rloo/` |
| Proof of training (logs, checkpoint hashes, Studio space, AWS identity, versions) | `proof/mt-grpo-1.5b-s0/`, `proof/mt-rloo-1.5b-s0/` |
| Error analysis and DPO analysis | `findings/error_analysis.md`, `findings/dpo_analysis.md` |

The trained adapters are not in the repository (too large); they are backed up on the team Google Drive.
