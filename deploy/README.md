# Deploying to Hugging Face

Two things go on the Hub: the **adapters** (a model repo) and the **demo** (a Space). Weights never go in
git. Run everything from the repository root on your laptop; you need a Hugging Face account and
`pip install -U huggingface_hub` (gives the `hf` command).

```bash
hf auth login                 # paste a token with "write" access (huggingface.co/settings/tokens)
```

## 1. Adapters → model repo `ayv443/verisql-adapters`

Put the checkpoint folders side by side in one folder, e.g. `~/verisql-adapters-src/`:
`mt-grpo-1.5b-s0/checkpoint-150`, `rloo-1.5b-s0/checkpoint-600`, `grpo-1.5b-s0/checkpoint-400`,
`dpo-1.5b-s0/checkpoint-100`. Then:

```bash
python deploy/build_space.py --adapters ~/verisql-adapters-src     # -> deploy/build/adapters (4 folders + model card)
shasum -a 256 deploy/build/adapters/mt-grpo-step150/adapter_model.safetensors
#   must print fa41692b3046019ba4c4ff9404655624559c3876fabaaa5059e2f966539b92bf (extension/proof/mt-grpo-1.5b-s0)
hf repo create ayv443/verisql-adapters --repo-type model
hf upload ayv443/verisql-adapters deploy/build/adapters . --repo-type model
```

## 2. Free demo Space (replay, CPU) → `ayv443/verisql`

```bash
python deploy/build_space.py --mode replay        # -> deploy/build/space-replay (about 210 MB, databases included)
hf repo create ayv443/verisql --repo-type space --space_sdk gradio
hf upload ayv443/verisql deploy/build/space-replay . --repo-type space
```

The page says clearly that it is a replay of saved outputs. It runs on the free CPU tier and sleeps after
48 hours without visitors (it wakes up on the next visit).

## 3. Live model on ZeroGPU (optional upgrade of the same Space)

ZeroGPU rules (checked November 2026): a free account in good standing (verified email, older than 30 days)
can host up to 2 ZeroGPU Spaces. Visitors use their own daily GPU quota: about 2 minutes unauthenticated,
3.5 to 5 minutes with a free account, more with PRO. Each generation here asks for at most 30 s
(`spaces.GPU(duration=30)` in `verisql/app.py`), and one question is 1 to 3 generations, so a free
visitor gets a few questions a day.

```bash
python deploy/build_space.py --mode zerogpu
hf upload ayv443/verisql deploy/build/space-zerogpu . --repo-type space
```

Then on the Space page: Settings → Space hardware → **ZeroGPU** → save. If ZeroGPU is not offered for your
account, keep the replay Space; nothing else changes.

## Where the URLs end up

- Demo: https://huggingface.co/spaces/ayv443/verisql
- Adapters: https://huggingface.co/ayv443/verisql-adapters
