# Exact McNemar tests on the multi-turn evaluation results (no GPU needed).
#   python extension/multiturn_stats.py
# Reads extension/findings/multiturn/<tag>/<split>/all/per_question.jsonl
# Writes extension/findings/multiturn_statistics.md
import json
import os
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "findings", "multiturn")
# (single-turn model, multi-turn model) on the same questions
PAIRS = {"spider_dev": [("grpo", "mt-grpo"), ("rloo", "mt-rloo"), ("grpo-600", "mt-grpo")],
         "bird_dev": [("grpo", "mt-grpo"), ("rloo", "mt-rloo")],
         "val": [("grpo-400", "mt-grpo-150"), ("rloo-600", "mt-rloo-150")]}
# models with no multi-turn version: only the effect of feedback (turn 1 vs final)
SINGLES = {"spider_dev": ["base", "dpo"]}
# pooled broken-answer test over the GRPO and RLOO pairs (not the step 600 control)
POOL = [("grpo", "mt-grpo"), ("rloo", "mt-rloo")]


def load(tag, split):
    path = os.path.join(F, tag, split, "all", "per_question.jsonl")
    return {r["qid"]: r["correct_per_turn"] for r in map(json.loads, open(path))}


def exact_p(b, c):
    """Two-sided exact McNemar test: binomial with p = 0.5 on the discordant pairs."""
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def fmt(p):
    return ("< 0.0001" if p < 0.0001 else f"{p:.4f}") + (" *" if p < 0.05 else "")


def discordant(a, b):
    return sum(a[q] and not b[q] for q in a), sum(b[q] and not a[q] for q in a)


def main():
    md = ["# Significance tests for the multi-turn evaluation", "",
          "Exact McNemar tests on paired per-question results (the same questions for both models). "
          "b = questions where only the first is true, c = only the second. Two-sided, p < 0.05 marked *.", ""]
    pooled = {}
    for split, pairs in PAIRS.items():
        md += [f"## {split}", "", "| Comparison | b | c | p |", "|---|---|---|---|"]
        for old, new in pairs:
            o, n = load(old, split), load(new, split)
            first = lambda d: {k: v[0] for k, v in d.items()}
            last = lambda d: {k: v[-1] for k, v in d.items()}
            broken = lambda d: {k: v[0] and not v[-1] for k, v in d.items()}
            rows = [(f"{old}: correct at turn 1 vs after feedback", first(o), last(o)),
                    (f"{new}: correct at turn 1 vs after feedback", first(n), last(n)),
                    (f"{old} vs {new}: correct at turn 1", first(o), first(n)),
                    (f"{old} vs {new}: correct after feedback", last(o), last(n)),
                    (f"{old} vs {new}: broken by feedback", broken(o), broken(n))]
            for name, a, b in rows:
                x, y = discordant(a, b)
                p = exact_p(x, y)
                md.append(f"| {name} | {x} | {y} | {fmt(p)} |")
            if split != "val" and (old, new) in POOL:
                x, y = discordant(broken(o), broken(n))
                pooled.setdefault(split, [0, 0])
                pooled[split][0] += x
                pooled[split][1] += y
        for tag in SINGLES.get(split, []):
            d = load(tag, split)
            x, y = discordant({k: v[0] for k, v in d.items()}, {k: v[-1] for k, v in d.items()})
            p = exact_p(x, y)
            md.append(f"| {tag}: correct at turn 1 vs after feedback | {x} | {y} | {fmt(p)} |")
        md.append("")
    md += ["## Broken answers, GRPO and RLOO pooled", "",
           "Pooling treats the GRPO and RLOO comparisons as one set of paired observations; it is a summary, "
           "not a separate experiment.", "", "| Benchmark | Broken only by single-turn | Broken only by multi-turn | p |",
           "|---|---|---|---|"]
    for split, (x, y) in pooled.items():
        p = exact_p(x, y)
        md.append(f"| {split} | {x} | {y} | {fmt(p)} |")
    md.append("")
    out = os.path.join(HERE, "findings", "multiturn_statistics.md")
    open(out, "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
