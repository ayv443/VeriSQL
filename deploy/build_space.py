"""Assembles the Hugging Face upload folders in deploy/build/ (gitignored). Nothing is uploaded here.

    python deploy/build_space.py --mode replay     # free CPU Space: saved model outputs, databases queried live
    python deploy/build_space.py --mode zerogpu    # ZeroGPU Space: the real model, adapters from the Hub
    python deploy/build_space.py --adapters <dir>  # also stage the adapters for the Hub model repo

Then follow deploy/README.md to upload them with the `hf` CLI. Needs the data from
text2sql-rl/download_data.sh + prepare_data.py.
"""
import argparse
import json
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "text2sql-rl"))
from common import DATA_DIR, PROCESSED_DIR, load_jsonl  # noqa: E402

BUILD = os.path.join(ROOT, "deploy", "build")
HUB_REPO = "SwordVer/verisql-adapters"
# BIRD dev databases small enough for a Space (the other four are 69 to 571 MB)
BIRD_DBS = {"superhero", "student_club", "toxicology", "thrombosis_prediction", "california_schools",
            "formula_1", "debit_card_specializing"}
# adapter folder names in the Hub repo -> local checkpoint folders (relative to --adapters)
ADAPTERS = {"mt-grpo-step150": "mt-grpo-1.5b-s0/checkpoint-150", "rloo-step600": "rloo-1.5b-s0/checkpoint-600",
            "grpo-step400": "grpo-1.5b-s0/checkpoint-400", "dpo-step100": "dpo-1.5b-s0/checkpoint-100"}

SPACE_README = """---
title: VeriSQL
emoji: 🗄️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: {gradio}
app_file: app.py
pinned: false
license: mit
short_description: Self-checking text-to-SQL, trained with RL
---

# VeriSQL

Ask a database a question in plain English. The model writes SQL, runs it read-only, reads the database's
reply and fixes its own query (up to 3 turns). Qwen2.5-Coder-1.5B-Instruct + LoRA, trained with GRPO in a
multi-turn environment.

{mode_note}

Code, training logs and every result: https://github.com/ayv443/verisql
Adapters: https://huggingface.co/{hub}

Databases: Spider dev (Yu et al., 2018) and part of BIRD dev (Li et al., 2023), both CC BY-SA 4.0.
"""

MODE_NOTE = {
    "replay": "**Replay mode (free CPU Space).** The answers are the model's saved outputs from our evaluation "
              "runs, so only Spider dev and BIRD dev questions can be asked; the databases are queried live.",
    "zerogpu": "**Live model on ZeroGPU.** Each answer runs the real model on a shared GPU, within the daily "
               "ZeroGPU quota of your Hugging Face account.",
}

ENTRY = {
    "replay": 'import sys\nsys.argv = ["app.py", "--replay"]\nsys.path.insert(0, "verisql")\nimport app\napp.main()\n',
    "zerogpu": 'import spaces  # noqa: F401  (must be imported before torch on ZeroGPU)\nimport sys\n'
               f'sys.argv = ["app.py", "--adapters-repo", "{HUB_REPO}"]\n'
               'sys.path.insert(0, "verisql")\nimport app\napp.main()\n',
}

REQUIREMENTS = {
    "replay": "",
    "zerogpu": "torch==2.8.0\ntransformers==4.56.2\npeft==0.17.1\naccelerate==1.10.1\n",
}


def copy(rel, dst_root, src_root=ROOT):
    src, dst = os.path.join(src_root, rel), os.path.join(dst_root, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, dst)


def build_space(mode, gradio_version):
    out = os.path.join(BUILD, f"space-{mode}")
    shutil.rmtree(out, ignore_errors=True)
    for rel in ["verisql/app.py", "verisql/style.css", "text2sql-rl/common.py", "text2sql-rl/reward.py",
                "extension/multiturn_eval.py"]:
        copy(rel, out)
    # saved runs: VeriSQL replay + "Why trust it" (mt-grpo, grpo) and "Compare models" (base, dpo, rloo)
    for tag in ["mt-grpo", "grpo", "base", "dpo", "rloo"]:
        for split in ["spider_dev", "bird_dev"]:
            rel = f"extension/findings/multiturn/{tag}/{split}/all/per_question.jsonl"
            if os.path.exists(os.path.join(ROOT, rel)):
                copy(rel, out)
    for tag in ["base", "dpo", "rloo"]:
        for split in ["spider_dev", "bird_dev"]:
            rel = f"text2sql-rl/evaluation/results/{tag}/{split}/per_question.jsonl"
            if os.path.exists(os.path.join(ROOT, rel)):
                copy(rel, out)
    # databases and the question lists (without the long prompts, which the app rebuilds)
    data_out = os.path.join(out, "text2sql-rl", "data")
    os.makedirs(os.path.join(data_out, "processed"), exist_ok=True)
    for split, keep in [("spider_dev", None), ("bird_dev", BIRD_DBS)]:
        rows = [r for r in load_jsonl(os.path.join(PROCESSED_DIR, f"{split}.jsonl")) if keep is None or r["db_id"] in keep]
        with open(os.path.join(data_out, "processed", f"{split}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps({k: r.get(k) for k in ["qid", "db_id", "db_path", "question", "evidence", "gold_sql"]}) + "\n")
        for db_path in sorted({r["db_path"] for r in rows}):
            copy(db_path, data_out, DATA_DIR)
            desc = os.path.join(os.path.dirname(db_path), "database_description")  # BIRD column notes
            if os.path.isdir(os.path.join(DATA_DIR, desc)):
                copy(desc, data_out, DATA_DIR)
    with open(os.path.join(out, "app.py"), "w") as f:
        f.write(ENTRY[mode])
    with open(os.path.join(out, "requirements.txt"), "w") as f:
        f.write(REQUIREMENTS[mode])
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(SPACE_README.format(gradio=gradio_version, mode_note=MODE_NOTE[mode], hub=HUB_REPO))
    size = sum(os.path.getsize(os.path.join(d, x)) for d, _, fs in os.walk(out) for x in fs)
    print(f"Space ({mode}) -> {out}  ({size / 1e6:.0f} MB)")


def build_adapters(adapter_dir):
    out = os.path.join(BUILD, "adapters")
    shutil.rmtree(out, ignore_errors=True)
    for name, rel in ADAPTERS.items():
        src = os.path.join(adapter_dir, rel)
        if not os.path.isdir(src):
            print(f"  skipped {name}: {src} not found")
            continue
        os.makedirs(os.path.join(out, name))
        for f in ["adapter_config.json", "adapter_model.safetensors"]:
            shutil.copy(os.path.join(src, f), os.path.join(out, name, f))
    shutil.copy(os.path.join(ROOT, "deploy", "model_card.md"), os.path.join(out, "README.md"))
    print(f"Hub model repo -> {out}  ({', '.join(sorted(os.listdir(out)))})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["replay", "zerogpu"])
    ap.add_argument("--adapters", help="folder holding mt-grpo-1.5b-s0/, rloo-1.5b-s0/, grpo-1.5b-s0/, dpo-1.5b-s0/")
    ap.add_argument("--gradio", default="6.17.3", help="Gradio version for the Space")
    args = ap.parse_args()
    if args.mode:
        build_space(args.mode, args.gradio)
    if args.adapters:
        build_adapters(args.adapters)


if __name__ == "__main__":
    main()
