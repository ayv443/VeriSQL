# Significance tests for the multi-turn evaluation

Exact McNemar tests on paired per-question results (the same questions for both models). b = questions where only the first is true, c = only the second. Two-sided, p < 0.05 marked *.

## spider_dev

| Comparison | b | c | p |
|---|---|---|---|
| grpo: correct at turn 1 vs after feedback | 6 | 6 | 1.0000 |
| mt-grpo: correct at turn 1 vs after feedback | 0 | 5 | 0.0625 |
| grpo vs mt-grpo: correct at turn 1 | 19 | 22 | 0.7552 |
| grpo vs mt-grpo: correct after feedback | 20 | 28 | 0.3123 |
| grpo vs mt-grpo: broken by feedback | 6 | 0 | 0.0312 * |
| rloo: correct at turn 1 vs after feedback | 9 | 6 | 0.6072 |
| mt-rloo: correct at turn 1 vs after feedback | 3 | 5 | 0.7266 |
| rloo vs mt-rloo: correct at turn 1 | 36 | 29 | 0.4570 |
| rloo vs mt-rloo: correct after feedback | 39 | 37 | 0.9088 |
| rloo vs mt-rloo: broken by feedback | 7 | 1 | 0.0703 |
| grpo-600: correct at turn 1 vs after feedback | 9 | 6 | 0.6072 |
| mt-grpo: correct at turn 1 vs after feedback | 0 | 5 | 0.0625 |
| grpo-600 vs mt-grpo: correct at turn 1 | 19 | 24 | 0.5424 |
| grpo-600 vs mt-grpo: correct after feedback | 21 | 34 | 0.1048 |
| grpo-600 vs mt-grpo: broken by feedback | 9 | 0 | 0.0039 * |
| base: correct at turn 1 vs after feedback | 10 | 20 | 0.0987 |
| dpo: correct at turn 1 vs after feedback | 44 | 9 | < 0.0001 * |

## bird_dev

| Comparison | b | c | p |
|---|---|---|---|
| grpo: correct at turn 1 vs after feedback | 7 | 25 | 0.0021 * |
| mt-grpo: correct at turn 1 vs after feedback | 0 | 26 | < 0.0001 * |
| grpo vs mt-grpo: correct at turn 1 | 36 | 60 | 0.0184 * |
| grpo vs mt-grpo: correct after feedback | 38 | 70 | 0.0027 * |
| grpo vs mt-grpo: broken by feedback | 7 | 0 | 0.0156 * |
| rloo: correct at turn 1 vs after feedback | 16 | 24 | 0.2682 |
| mt-rloo: correct at turn 1 vs after feedback | 4 | 25 | 0.0001 * |
| rloo vs mt-rloo: correct at turn 1 | 74 | 41 | 0.0027 * |
| rloo vs mt-rloo: correct after feedback | 78 | 58 | 0.1029 |
| rloo vs mt-rloo: broken by feedback | 13 | 1 | 0.0018 * |

## val

| Comparison | b | c | p |
|---|---|---|---|
| grpo-400: correct at turn 1 vs after feedback | 1 | 0 | 1.0000 |
| mt-grpo-150: correct at turn 1 vs after feedback | 0 | 1 | 1.0000 |
| grpo-400 vs mt-grpo-150: correct at turn 1 | 8 | 15 | 0.2100 |
| grpo-400 vs mt-grpo-150: correct after feedback | 7 | 16 | 0.0931 |
| grpo-400 vs mt-grpo-150: broken by feedback | 1 | 0 | 1.0000 |
| rloo-600: correct at turn 1 vs after feedback | 2 | 3 | 1.0000 |
| mt-rloo-150: correct at turn 1 vs after feedback | 0 | 2 | 0.5000 |
| rloo-600 vs mt-rloo-150: correct at turn 1 | 16 | 13 | 0.7111 |
| rloo-600 vs mt-rloo-150: correct after feedback | 15 | 13 | 0.8506 |
| rloo-600 vs mt-rloo-150: broken by feedback | 2 | 0 | 0.5000 |

## Broken answers, GRPO and RLOO pooled

Pooling treats the GRPO and RLOO comparisons as one set of paired observations; it is a summary, not a separate experiment.

| Benchmark | Broken only by single-turn | Broken only by multi-turn | p |
|---|---|---|---|
| spider_dev | 13 | 1 | 0.0018 * |
| bird_dev | 20 | 1 | < 0.0001 * |
