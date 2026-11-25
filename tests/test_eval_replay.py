"""Regression test: running the saved model outputs through today's loop, guardrails and scorer must give
exactly the results recorded in the original evaluation (extension/findings/multiturn/mt-grpo)."""
import json
import os

from conftest import ROOT
import core
import run_eval

SAVED = os.path.join(ROOT, "extension", "findings", "multiturn", "mt-grpo", "spider_dev", "all", "per_question.jsonl")


def test_replay_matches_the_original_evaluation():
    items = json.load(open(os.path.join(ROOT, "eval", "eval_set.json")))["questions"]
    saved = {}
    with open(SAVED) as f:
        for line in f:
            r = json.loads(line)
            saved[r["qid"]] = r
    per_q = run_eval.evaluate(core.ReplayEngine(), items)
    assert len(per_q) == 50
    for q in per_q:
        s = saved[q["qid"]]
        assert q["correct"] == s["correct_per_turn"][-1], q["qid"]
        assert q["turns_used"] == s["turns_used"], q["qid"]
    acc = sum(q["correct"] for q in per_q) / len(per_q)
    assert acc == 0.66  # the recorded accuracy of this 50-question set
