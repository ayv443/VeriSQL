"""Runs the fixed evaluation set (eval/eval_set.json) through the same self-check loop the API uses.

    python eval/run_eval.py --engine replay                                  # no model (CI): saved MT-GRPO outputs
    python eval/run_eval.py --engine live --adapter <checkpoint-150 folder>  # the real model (GPU, Apple MPS or CPU)
    python eval/run_eval.py --engine live --adapter SwordVer/verisql-adapters --subfolder mt-grpo-step150

Scoring is strict execution accuracy, as in the project: the final query's result must equal the gold
query's result (row order only checked when the gold query has ORDER BY). Prints one JSON report and
writes it to eval/reports/<engine>.json. --min-accuracy makes the run fail below a threshold (CI gate).
"""
import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "verisql"))
import core  # noqa: E402
from reward import has_order_by, results_match  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")


def db_file(db_id):
    return os.path.join(FIXTURES, db_id, f"{db_id}.sqlite")


def evaluate(engine, items):
    per_q = []
    for it in items:
        db = db_file(it["db_id"])
        t = time.perf_counter()
        res = core.solve(engine, db, it["question"], db_id=it["db_id"])
        latency = time.perf_counter() - t
        gold = core.run_sql(db, it["gold_sql"], timeout=30)
        correct = res["error"] is None and gold.error is None and \
            results_match(res["rows"], gold.rows, has_order_by(it["gold_sql"]))
        per_q.append({"qid": it["qid"], "correct": correct, "error": res["error"], "turns_used": res["turns_used"],
                      "fixed_errors": res["fixed_errors"], "first_turn_error": res["turns"][0]["error"],
                      "latency_s": round(latency, 3), "final_sql": res["final_sql"]})
    return per_q


def summarise(per_q, engine):
    n = len(per_q)
    lat = sorted(q["latency_s"] for q in per_q)
    return {
        "engine": engine.mode, "model": engine.label, "questions": n,
        "accuracy": round(sum(q["correct"] for q in per_q) / n, 4),
        "final_error_rate": round(sum(q["error"] is not None for q in per_q) / n, 4),
        "first_turn_error_rate": round(sum(q["first_turn_error"] is not None for q in per_q) / n, 4),
        "errors_fixed": sum(q["fixed_errors"] for q in per_q),
        "mean_turns": round(sum(q["turns_used"] for q in per_q) / n, 2),
        "latency_median_s": round(statistics.median(lat), 3),
        "latency_p95_s": round(lat[min(n - 1, int(0.95 * n))], 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="replay", choices=["replay", "test", "live"])
    ap.add_argument("--adapter", help="live: checkpoint folder or Hub repo id")
    ap.add_argument("--subfolder", help="live: folder inside the Hub repo")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], help="live: weights dtype (default fp16 on GPU, fp32 elsewhere)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--min-accuracy", type=float, help="exit with an error below this accuracy")
    args = ap.parse_args()

    items = json.load(open(os.path.join(HERE, "eval_set.json")))["questions"][:args.limit]
    engine = core.make_engine(args.engine, args.adapter, args.subfolder, args.dtype)
    per_q = evaluate(engine, items)
    report = summarise(per_q, engine)
    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    name = args.engine + (f"-{args.dtype}" if args.dtype else "")
    with open(os.path.join(HERE, "reports", f"{name}.json"), "w") as f:
        json.dump({"summary": report, "per_question": per_q}, f, indent=1)
    print(json.dumps(report, indent=2))
    if args.min_accuracy is not None and report["accuracy"] < args.min_accuracy:
        sys.exit(f"accuracy {report['accuracy']:.3f} is below the gate {args.min_accuracy:.3f}")


if __name__ == "__main__":
    main()
