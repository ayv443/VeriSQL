---
license: mit
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
  - text-to-sql
  - reinforcement-learning
  - grpo
  - rloo
  - dpo
  - lora
datasets:
  - xlangai/spider
---

# VeriSQL adapters

LoRA adapters (rank 16, alpha 32, all 7 projection layers, about 74 MB each) for
[Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct), trained on Spider
with an execution reward: the query is run on a read-only copy of the database and its result is compared
with the gold query's result (+1 match, 0 wrong, -0.1 error). Code, training logs and every result:
https://github.com/ayv443/verisql

| Folder | Training | Notes |
|---|---|---|
| `mt-grpo-step150` | GRPO step 400, then 150 steps of multi-turn GRPO | **VeriSQL.** Sees the database's reply and can fix its query (up to 3 turns, -0.05 per change) |
| `rloo-step600` | RLOO, 600 steps | Best single-turn model |
| `grpo-step400` | GRPO, 400 steps | Start point of `mt-grpo-step150` |
| `dpo-step100` | DPO, 100 steps | Offline baseline (does not beat the base model) |

## Results

Execution accuracy, official evaluation scripts (Phase 1, single turn):

| | Spider dev | BIRD dev |
|---|---|---|
| Base model | 65.2% | 29.1% |
| DPO | 62.7% | 27.8% |
| GRPO | 74.6% | 36.1% |
| RLOO | 75.6% | 37.0% |

Multi-turn, strict execution accuracy after feedback: MT-GRPO 72.2% on Spider dev and 36.5% on BIRD dev,
and it broke no correct answer when revising (GRPO without multi-turn training broke 6 on Spider dev and
7 on BIRD dev). One training seed per method.

SHA-256 of `mt-grpo-step150/adapter_model.safetensors`:
`fa41692b3046019ba4c4ff9404655624559c3876fabaaa5059e2f966539b92bf` (matches the proof file in the repo).

## Use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base)
model = PeftModel.from_pretrained(model, "ayv443/verisql-adapters", subfolder="mt-grpo-step150")
```

The prompt format (system prompt, schema with example values, `<answer></answer>` tags) and the
self-check loop are in `verisql/core.py` in the GitHub repo; use them to get the reported accuracy.

## Limitations

Trained on Spider (SQLite, English). BIRD dev accuracy is still under 40%. Execution accuracy can count a
wrong query as correct when it happens to return the right rows. Always run generated SQL read-only.

UTS 43008 Reinforcement Learning, Project 56: Aditya Verma, Eby Kurian, Pravicksha Paulsamy Vigneshkumar.
