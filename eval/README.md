# Evaluation suite

50 Spider dev questions (a fixed random sample, seed 0, every dev database except the 105 MB `wta_1`) and
their 18 small databases, so the suite runs anywhere with no downloads. Built by `build_eval_set.py`;
run by `run_eval.py`, which uses the same self-check loop as the API (`verisql/core.py`). Scoring is strict
execution accuracy: the final query's result must equal the gold query's result.

| Run | Accuracy | Final SQL error rate | Mean turns | Median latency | p95 latency |
|---|---|---|---|---|---|
| Replay of the saved MT-GRPO outputs (the CI gate) | 66.0% (33/50) | 12% | 2.02 | under 0.01 s | under 0.01 s |
| Live MT-GRPO step 150, CPU, bfloat16 (2 cores) | 70.0% (35/50) | 4% | 2.06 | 27.4 s | 83.2 s |

- The replay row reproduces the original evaluation run exactly (fp16 on a T4); `tests/test_eval_replay.py`
  checks every question. On the full Spider dev set the same run scores 72.2%.
- The live run agrees with the saved run on 48 of 50 questions; bf16 on the CPU got 2 more right. The
  report with every question is `live-cpu-bf16-report.json`.
- 50 questions give a wide margin (about ±13 points at 95%), so this suite is a regression check, not a
  benchmark. The benchmark numbers are in [`RESULTS.md`](../RESULTS.md).
- CPU latency is for a 2-core cloud machine without a GPU; on a GPU each turn takes a few seconds.

```bash
python eval/run_eval.py --engine replay --min-accuracy 0.66      # CI
python eval/run_eval.py --engine live --adapter SwordVer/verisql-adapters --subfolder mt-grpo-step150
```
