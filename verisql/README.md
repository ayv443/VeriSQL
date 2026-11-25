# VeriSQL demo

A search-style web app for the self-checking model. You pick a database, ask a question in plain English,
and watch the model write a query, run it, read the database's reply and fix the query when it fails.

The answer comes from **MT-GRPO step 150**, the GRPO model trained inside the multi-turn environment
(Phase 2). The app runs the same loop as `extension/multiturn_eval.py`: the same prompt, the same feedback
text, greedy decoding, up to 3 turns, and a reply with no new SQL keeps the previous query. The model never
sees the correct answer.

## What is on the page

| Part | What it shows |
|---|---|
| Search bar | Database picker, question box, example questions for the chosen database |
| Live steps | Each turn as it happens: writing the query, running it, the error or row count, the fix (as a diff), the final check |
| Status bar | "Ran successfully · self-checked in N turns · fixed N errors"; click it to see the steps again |
| Answer | The result table, "Copy SQL", "Download CSV", the SQL, follow-up questions |
| How it was checked | Every attempt, the database's reply, and what the model answered |
| Compare models | The untrained model, DPO and RLOO answering the same question once, with the reward each answer gets |
| Why trust it | VeriSQL's strict execution accuracy on Spider dev and BIRD dev, and answers broken after feedback (GRPO vs VeriSQL), read from `extension/findings/` |
| Side menu | Database list and search, tables with a 3-row preview, upload of your own `.sqlite` file, recent questions |
| Theme button | Light and dark mode (or open the page with `?__theme=dark`) |

"Matches the expected answer" and the +1 / 0 badges only appear for questions from Spider dev or BIRD dev,
where the correct query is known. For a question you type yourself, the app can only say whether the query
ran, so that is all it says.

## Run it (SageMaker Studio, one T4)

From the repository root, with the data prepared as for the main project (`text2sql-rl/download_data.sh`
and `prepare_data.py`):

```bash
pip install gradio
python verisql/app.py \
    --verisql ~/outputs/mt-grpo-1.5b-s0/checkpoint-150 \
    --dpo  text2sql-rl/outputs/dpo-1.5b-s0/checkpoint-100 \
    --rloo text2sql-rl/outputs/rloo-1.5b-s0/checkpoint-600 \
    --share
```

`--share` prints a public `https://….gradio.live` link. `--dpo` and `--rloo` are only used by "Compare
models" and can be left out. The adapters are not in the repository; they are on the team Google Drive
(`mt-grpo-1.5b-s0-adapters.tar.gz` and the Phase 1 adapters).

## Without a GPU

```bash
python verisql/app.py --replay    # saved answers from our evaluation runs, databases queried live
python verisql/app.py --test      # small untrained 0.5B model on CPU, only to check the app works
```

Replay mode answers Spider dev and BIRD dev questions with the outputs saved in
`extension/findings/multiturn/` and `text2sql-rl/evaluation/results/`, paced so the steps can be followed.
Both modes say so on the page.

## Files

| File | What it is |
|---|---|
| `app.py` | The app: the self-check loop, the model loading, the page and its events |
| `style.css` | The look, light and dark |

Uploaded databases are copied to a temporary folder and always opened read-only.
