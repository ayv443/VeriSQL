# Deploying to Hugging Face

Two things go on the Hub: the **adapters** (a model repo) and the **demo** (a Space). Weights never go in
git. The Hugging Face account is `SwordVer` (the GitHub account is `ayv443`). Run everything from the repository root on your laptop; you need a Hugging Face account and
`pip install -U huggingface_hub` (gives the `hf` command).

```bash
hf auth login                 # then `hf auth whoami` must print SwordVer; paste a token with "write" access (huggingface.co/settings/tokens)
```

## 1. Adapters → model repo `SwordVer/verisql-adapters`

Put the checkpoint folders side by side in one folder, e.g. `~/verisql-adapters-src/`:
`mt-grpo-1.5b-s0/checkpoint-150`, `rloo-1.5b-s0/checkpoint-600`, `grpo-1.5b-s0/checkpoint-400`,
`dpo-1.5b-s0/checkpoint-100`. Then:

```bash
python deploy/build_space.py --adapters ~/verisql-adapters-src     # -> deploy/build/adapters (4 folders + model card)
shasum -a 256 deploy/build/adapters/mt-grpo-step150/adapter_model.safetensors
#   must print fa41692b3046019ba4c4ff9404655624559c3876fabaaa5059e2f966539b92bf (extension/proof/mt-grpo-1.5b-s0)
hf repos create SwordVer/verisql-adapters --repo-type model --public
hf upload SwordVer/verisql-adapters deploy/build/adapters . --repo-type model
```

## 2. Demo Space on ZeroGPU → `SwordVer/verisql` (what is live)

Since late 2026, free accounts can no longer host Gradio Spaces on the CPU tier (`402 Payment Required`),
but a free account in good standing (verified email, older than 30 days) can still host up to 2 ZeroGPU
Spaces. Visitors use their own daily GPU quota: about 2 minutes logged out, 3.5 to 5 minutes with a free
account, more with PRO. Each generation asks for at most 30 s (`spaces.GPU(duration=30)` in
`verisql/app.py`), and one question is 1 to 3 generations.

```bash
python deploy/build_space.py --mode zerogpu       # -> deploy/build/space-zerogpu (about 210 MB, databases included)
find deploy/build/space-zerogpu -name .DS_Store -delete
hf repos create SwordVer/verisql --repo-type space --space-sdk gradio --flavor zero-a10g --public
hf upload SwordVer/verisql deploy/build/space-zerogpu . --repo-type space
```

The first start installs torch and downloads the base model and the adapters (a few minutes). Two things
ZeroGPU needs, both handled in `verisql/app.py`: build the model on the CPU and move it to the GPU once at
startup (weights cannot be loaded straight onto the GPU), and pass only text to the `@spaces.GPU`
function (its arguments are copied to a separate GPU worker). If the GPU does not answer within 60 s the
page shows the model's saved answer from our evaluation run and says so.

## 3. Replay-only Space (no GPU)

`python deploy/build_space.py --mode replay` builds the same page answering only from the saved runs. It
needs a CPU Space, which now requires Hugging Face PRO; locally it runs with
`python verisql/app.py --replay`.

## Where the URLs end up

- Demo: https://huggingface.co/spaces/SwordVer/verisql
- Adapters: https://huggingface.co/SwordVer/verisql-adapters
