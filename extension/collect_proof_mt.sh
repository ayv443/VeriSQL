#!/usr/bin/env bash
# Collects proof that a multi-turn run was trained on AWS (inside a SageMaker Studio space), the same
# things text2sql-rl/collect_proof.sh collects for the single-turn runs.
#   bash extension/collect_proof_mt.sh <run folder> <console log>
#   e.g. bash extension/collect_proof_mt.sh ~/outputs/mt-rloo-1.5b-s0 ~/mt_rloo_train.log
# Writes extension/proof/<run name>/ (run from the repository root).
set -e
SRC=$1
LOG=$2
if [ -z "$SRC" ] || [ ! -d "$SRC" ]; then echo "usage: bash extension/collect_proof_mt.sh <run folder> <console log>"; exit 1; fi
RUN=$(basename "$SRC")
OUT=extension/proof/$RUN
mkdir -p "$OUT/logs" "$OUT/checkpoints"

# 1. step logs, sample episodes, settings, and the console log of the run
cp "$SRC"/logs/*.jsonl "$SRC"/logs/args.json "$OUT/logs/" 2>/dev/null || true
if [ -n "$LOG" ] && [ -f "$LOG" ]; then cp "$LOG" "$OUT/logs/console.log"; else echo "console log not found" > "$OUT/logs/console.log"; fi

# 2. every checkpoint: its LoRA config, and the size and SHA-256 of the saved weights (the weights
#    themselves are too large for GitHub; the hash shows which file the results came from)
for ck in "$SRC"/checkpoint-*; do
  [ -d "$ck" ] || continue
  name=$(basename "$ck")
  mkdir -p "$OUT/checkpoints/$name"
  cp "$ck/adapter_config.json" "$OUT/checkpoints/$name/" 2>/dev/null || true
  if [ -f "$ck/adapter_model.safetensors" ]; then
    ( cd "$ck" && ls -l --time-style=full-iso adapter_model.safetensors && sha256sum adapter_model.safetensors ) > "$OUT/checkpoints/$name/weights.txt"
  fi
done

# 3. where it ran: platform and region only (no account ID, role or user profile, the repo is public), the GPU
python3 -c "import json,re; m=json.load(open('/opt/ml/metadata/resource-metadata.json')); r=re.search(r'arn:aws:sagemaker:([a-z0-9-]+):', m.get('ResourceArn','')); json.dump({'platform':'SageMaker Studio','app_type':m.get('AppType',''),'region':r.group(1) if r else ''}, open('$OUT/platform.json','w'))" 2>/dev/null || echo '{"platform": "not a SageMaker Studio space"}' > "$OUT"/platform.json
nvidia-smi > "$OUT/nvidia_smi.txt" 2>&1 || true

# 4. which code and library versions were used
git log -1 --format='%H  %ad  %s' > "$OUT/git_commit.txt"
pip freeze 2>/dev/null | grep -iE "^(trl|transformers|peft|accelerate|datasets|torch|vllm)==" > "$OUT/versions.txt" || true

# 5. short summary
python - "$OUT" "$RUN" > "$OUT/summary.txt" <<'EOF'
import glob, json, os, sys
out, run = sys.argv[1], sys.argv[2]
print("run:", run)
args = json.load(open(f"{out}/logs/args.json")) if os.path.exists(f"{out}/logs/args.json") else {}
print("method:", args.get("method"), "| start adapter:", args.get("init_adapter"), "| max steps:", args.get("max_steps"))
steps = [json.loads(l) for l in open(f"{out}/logs/steps.jsonl")] if os.path.exists(f"{out}/logs/steps.jsonl") else []
if steps:
    secs = sum(r.get("step_s", 0) for r in steps)
    print(f"steps logged: {len(steps)} | last step: {steps[-1].get('step')} | total time: {secs / 3600:.2f} h "
          f"| avg s/step: {secs / len(steps):.1f}")
cks = sorted(glob.glob(f"{out}/checkpoints/checkpoint-*"), key=lambda p: int(p.split("-")[-1]))
print("checkpoints:", ", ".join(os.path.basename(c) for c in cks))
for c in cks:
    w = f"{c}/weights.txt"
    if os.path.exists(w):
        print(f"  {os.path.basename(c)} sha256: {open(w).read().split()[-2]}")
print("ran on:", " | ".join(str(v) for v in json.load(open(f"{out}/platform.json")).values()))
print("versions:", " ".join(open(f"{out}/versions.txt").read().split()))
EOF

echo "saved to $OUT:"
ls "$OUT"
cat "$OUT/summary.txt"
