"""Builds the fixed evaluation set: eval/eval_set.json + the small Spider databases it needs (eval/fixtures/).

Run once, after text2sql-rl/download_data.sh and prepare_data.py; the output is committed so CI needs no
downloads. 50 Spider dev questions, a plain random sample (fixed seed) from every dev database except wta_1
(105 MB), so the accuracy is not biased by hand-picked questions. The singer database (used in the README and
API examples) is always copied too. The databases are from Spider (Yu et al., 2018, CC BY-SA 4.0).

    python eval/build_eval_set.py
"""
import json
import os
import random
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "text2sql-rl"))
from common import DATA_DIR, PROCESSED_DIR, load_jsonl  # noqa: E402

N, SEED = 50, 0
SKIP_DBS = {"wta_1"}  # too large to commit
EXTRA_DBS = {"singer"}  # used in the README and API examples
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    rows = [r for r in load_jsonl(os.path.join(PROCESSED_DIR, "spider_dev.jsonl")) if r["db_id"] not in SKIP_DBS]
    picked = random.Random(SEED).sample(rows, N)
    picked.sort(key=lambda r: int(r["qid"].rsplit("_", 1)[1]))
    items = [{"qid": r["qid"], "db_id": r["db_id"], "question": r["question"], "gold_sql": r["gold_sql"]}
             for r in picked]
    fixtures = os.path.join(OUT, "fixtures")
    for db in sorted({r["db_id"] for r in picked} | EXTRA_DBS):
        src = os.path.join(DATA_DIR, next(r["db_path"] for r in rows if r["db_id"] == db))
        os.makedirs(os.path.join(fixtures, db), exist_ok=True)
        shutil.copy(src, os.path.join(fixtures, db, f"{db}.sqlite"))
    with open(os.path.join(OUT, "eval_set.json"), "w") as f:
        json.dump({"source": "Spider dev (Yu et al., 2018), CC BY-SA 4.0", "seed": SEED, "questions": items},
                  f, indent=1)
    print(f"{len(items)} questions over {len({r['db_id'] for r in picked})} databases -> eval/eval_set.json")


if __name__ == "__main__":
    main()
