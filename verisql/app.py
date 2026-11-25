# VeriSQL: a self-checking text-to-SQL assistant (demo app).
#
# The answer comes from the self-checking model (multi-turn GRPO, step 150). It writes a query, the query is
# run on a read-only copy of the database, and the model sees the error or the first rows and can fix its
# query, for up to 3 turns. This is the same loop as extension/multiturn_eval.py, with the same prompt,
# feedback text and greedy decoding.
#
# Run from the repository root, on a GPU (T4):
#   python verisql/app.py --verisql <mt-grpo-1.5b-s0/checkpoint-150> \
#       --dpo <dpo-1.5b-s0/checkpoint-100> --rloo <rloo-1.5b-s0/checkpoint-600> --share
#   python verisql/app.py --adapters-repo ayv443/verisql-adapters      (adapters from the Hugging Face Hub)
# --dpo and --rloo are only used by "Compare models" and can be left out. Without CUDA it runs on Apple MPS
# or the CPU (slower); on a Hugging Face ZeroGPU Space each generation gets a GPU for up to 30 s.
#
# Without a GPU:
#   python verisql/app.py --replay   answers are the saved outputs from our evaluation runs (Spider dev and
#                                    BIRD dev questions only); the databases are still queried live
#   python verisql/app.py --test     the small untrained 0.5B model on CPU, only to check that the app works
import argparse
import csv
import difflib
import html
import inspect
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time

import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "text2sql-rl"))
sys.path.insert(0, os.path.join(ROOT, "extension"))

from common import DATA_DIR, MAX_COMPLETION_LEN, PROCESSED_DIR, build_prompt, extract_sql, load_jsonl, schema_text  # noqa: E402
from reward import TIMEOUT_S, execute, has_order_by, results_match  # noqa: E402
from multiturn_eval import clean_sql, feedback, same_query  # noqa: E402

MAX_TURNS = 3
SHOW_ROWS = 20
FINDINGS = os.path.join(ROOT, "extension", "findings", "multiturn")
SAVED_PHASE1 = os.path.join(ROOT, "text2sql-rl", "evaluation", "results")
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "verisql_uploads")
E = html.escape

COMPARE = [("untrained", "Untrained", "the base model"), ("dpo", "DPO", "offline preference pairs"),
           ("rloo", "RLOO", "online, single answer"), ("verisql", "VeriSQL", "self-checking")]


# ------------------------------------------------------------------ databases and sample questions
def load_catalogue():
    dbs, questions = {}, {}
    for split, source in [("spider_dev", "Spider"), ("bird_dev", "BIRD")]:
        p = os.path.join(PROCESSED_DIR, f"{split}.jsonl")
        if not os.path.exists(p):
            continue
        for r in load_jsonl(p):
            path = os.path.join(DATA_DIR, r["db_path"])
            if not os.path.exists(path):
                continue
            name = r["db_id"] if r["db_id"] not in dbs or dbs[r["db_id"]]["source"] == source else f"{r['db_id']} ({source})"
            dbs.setdefault(name, {"path": path, "source": source})
            questions.setdefault(name, []).append({"question": r["question"], "gold_sql": r["gold_sql"],
                                                   "evidence": r.get("evidence", "") or ""})
    return dbs, questions


DBS, QUESTIONS = load_catalogue()
FEATURED = ["singer", "flight_2", "concert_singer", "california_schools"]


def db_names(filter_text=""):
    names = [n for n in FEATURED if n in DBS] + sorted(n for n in DBS if n not in FEATURED)
    f = (filter_text or "").strip().lower()
    return [n for n in names if f in n.lower()]


def find_example(db, question):
    q = " ".join((question or "").split()).lower()
    for ex in QUESTIONS.get(db, []):
        if " ".join(ex["question"].split()).lower() == q:
            return ex
    return None


def short_questions(db, n=3, exclude=""):
    exs = [e for e in QUESTIONS.get(db, []) if e["question"] != exclude]
    seen, out = set(), []
    for e in sorted(exs, key=lambda e: len(e["question"])):  # short questions read best as chips
        key = " ".join(e["gold_sql"].lower().replace('"', "'").split())  # one chip per distinct answer
        if key not in seen and len(e["question"]) <= 80:
            seen.add(key)
            out.append(e["question"])
    if db == "singer":  # the showcase database: the example from the report first
        lead = "Show the names of singers and the total sales of their songs."
        if any(e["question"] == lead for e in exs):
            out = [lead] + [q for q in out if q != lead]
    return out[:n]


def tables_of(path):
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        con.close()
        return names
    except sqlite3.Error:
        return []


def run_query(path, sql, limit=100_000):
    """Read-only run that also returns the column names (same limits as reward.execute)."""
    t = time.perf_counter()
    rows, err = execute(path, sql)
    cols = []
    if err is None:
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            con.text_factory = lambda b: b.decode(errors="ignore")
            deadline = time.monotonic() + TIMEOUT_S
            con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description or []]
            con.close()
        except Exception:
            cols = [f"column {i + 1}" for i in range(len(rows[0]) if rows else 0)]
    return cols, rows, err, time.perf_counter() - t


# ------------------------------------------------------------------ the models
class Engine:
    """generate(name, messages) -> text, or None when there is no answer (replay of an unknown question)."""
    mode = "live"
    available = {"verisql"}
    note = ""

    def generate(self, name, messages):
        raise NotImplementedError


# Folder names inside the Hugging Face Hub repo of adapters (see deploy/README.md)
HUB_SUBFOLDERS = {"verisql": "mt-grpo-step150", "dpo": "dpo-step100", "rloo": "rloo-step600"}

try:  # Hugging Face ZeroGPU: a GPU is attached only while a decorated call runs
    import spaces
    on_gpu = spaces.GPU(duration=30)
except ImportError:
    def on_gpu(f):
        return f


class LiveEngine(Engine):
    def __init__(self, args):
        import torch
        from peft import PeftModel
        from common import load_model
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
        base, self.tok = load_model()
        if self.device != "cuda":
            base = base.float()  # fp16 is slow or unsupported off the GPU

        def source(name):  # a local checkpoint folder, or a folder inside the Hub repo
            if args.adapters_repo:
                return args.adapters_repo, {"subfolder": HUB_SUBFOLDERS[name]}
            return getattr(args, name), {}

        path, kw = source("verisql")
        self.model = PeftModel.from_pretrained(base.to(self.device), path, adapter_name="verisql", **kw)
        self.available = {"verisql", "untrained"}
        for name in ["dpo", "rloo"]:
            if args.adapters_repo or getattr(args, name):
                path, kw = source(name)
                self.model.load_adapter(path, adapter_name=name, **kw)
                self.available.add(name)
        self.model.eval()

    def generate(self, name, messages):
        return _live_generate(self, name, messages)


@on_gpu
def _live_generate(engine, name, messages):
    text = engine.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = engine.tok(text, return_tensors="pt").to(engine.device)
    with engine.torch.no_grad():
        if name == "untrained":
            with engine.model.disable_adapter():
                out = engine.model.generate(**enc, max_new_tokens=MAX_COMPLETION_LEN, do_sample=False)
        else:
            engine.model.set_adapter(name)
            out = engine.model.generate(**enc, max_new_tokens=MAX_COMPLETION_LEN, do_sample=False)
    return engine.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


class TestEngine(Engine):
    mode = "test"
    note = "Test mode: every answer comes from the small untrained 0.5B model on CPU, not from VeriSQL."

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
        self.available = {"verisql", "untrained", "dpo", "rloo"}

    def generate(self, name, messages):
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=128, do_sample=False)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


class ReplayEngine(Engine):
    """Answers from the saved evaluation runs: VeriSQL = every turn of MT-GRPO's saved episode, the other
    models = their saved single-turn answer. Only questions from those runs can be answered."""
    mode = "replay"
    note = ("Replay mode: the answers are the saved outputs from our evaluation runs, "
            "and the databases are queried live. Only Spider dev and BIRD dev questions can be asked.")

    def __init__(self):
        self.saved = {"verisql": {}, "untrained": {}, "dpo": {}, "rloo": {}}
        for split in ["spider_dev", "bird_dev"]:
            p = os.path.join(FINDINGS, "mt-grpo", split, "all", "per_question.jsonl")
            if os.path.exists(p):
                for r in load_jsonl(p):
                    turns = [m["content"] for m in r["conversation"] if m["role"] == "assistant"]
                    self.saved["verisql"][self.key(r["db_id"], r["question"])] = turns
            for name, tag in [("untrained", "base"), ("dpo", "dpo"), ("rloo", "rloo")]:
                p = os.path.join(SAVED_PHASE1, tag, split, "per_question.jsonl")
                if not os.path.exists(p):
                    p = os.path.join(FINDINGS, tag, split, "all", "per_question.jsonl")
                if os.path.exists(p):
                    for r in load_jsonl(p):
                        out = r.get("raw_output") or (f"<answer>{r['sqls'][0]}</answer>" if r.get("sqls") else None)
                        if out:
                            self.saved[name][self.key(r["db_id"], r["question"])] = [out]
        self.available = {n for n, d in self.saved.items() if d} | {"verisql"}
        self.current_db = None

    @staticmethod
    def key(db, q):
        return (db.split(" (")[0], " ".join(q.split()).lower())

    pause = 0.9  # seconds per saved answer, so the self-check steps can be followed on screen

    def generate(self, name, messages):
        time.sleep(self.pause if name == "verisql" else self.pause / 3)
        q = messages[1]["content"].rsplit("Question:", 1)[-1].strip()
        turns = self.saved.get(name, {}).get(self.key(self.current_db or "", q))
        if not turns:
            return None
        k = sum(m["role"] == "assistant" for m in messages)
        return turns[min(k, len(turns) - 1)]


ENGINE = Engine()


# ------------------------------------------------------------------ numbers for "Why trust it"
def trust_numbers():
    def load(tag, split):
        p = os.path.join(FINDINGS, tag, split, "all", "per_question.jsonl")
        return load_jsonl(p) if os.path.exists(p) else None
    out = {}
    for split, label in [("spider_dev", "Spider"), ("bird_dev", "BIRD")]:
        rows = load("mt-grpo", split)
        if rows:
            out[label] = (100 * sum(r["correct_per_turn"][-1] for r in rows) / len(rows), len(rows))
    broken = {}
    for tag in ["grpo", "mt-grpo"]:
        rows = load(tag, "bird_dev")
        if rows:
            broken[tag] = sum(r["correct_per_turn"][0] and not r["correct_per_turn"][-1] for r in rows)
    out["broken"] = (broken.get("grpo"), broken.get("mt-grpo"))
    return out


TRUST = trust_numbers()


# ------------------------------------------------------------------ HTML pieces
LOGO = """<svg viewBox="0 0 48 48" width="{s}" height="{s}" aria-hidden="true">
<ellipse cx="22" cy="10" rx="16" ry="6" fill="#5EAAA0"/>
<path d="M6 10v9c0 3.3 7.2 6 16 6s16-2.7 16-6v-9c0 3.3-7.2 6-16 6S6 13.3 6 10z" fill="#0F766E"/>
<path d="M6 22v9c0 3.3 7.2 6 16 6s16-2.7 16-6v-9c0 3.3-7.2 6-16 6S6 25.3 6 22z" fill="#0F766E"/>
<path d="M6 34v4c0 3.3 7.2 6 16 6 3 0 5.8-.3 8.2-.9L36 37l2-2v-1c0 3.3-7.2 6-16 6S6 37.3 6 34z" fill="#115E59"/>
<path d="M25 33.5l5.5 5.5L42 27.5" fill="none" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""
WORD = '<span class="vs-word">Veri<b>SQL</b></span>'
ICON = {
    "ok": '<span class="vs-ic ok"><svg viewBox="0 0 24 24"><path d="M6 12.5l4 4 8-9"/></svg></span>',
    "warn": '<span class="vs-ic warn"><svg viewBox="0 0 24 24"><path d="M12 4 2.5 20h19L12 4z"/><path d="M12 10v4.5M12 17.5v.01"/></svg></span>',
    "err": '<span class="vs-ic err"><svg viewBox="0 0 24 24"><path d="M7 7l10 10M17 7 7 17"/></svg></span>',
    "spin": '<span class="vs-ic spin"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" class="trk"/><path d="M12 3a9 9 0 0 1 9 9" class="arc"/></svg></span>',
}
SVG = {
    "doc": '<svg viewBox="0 0 24 24"><rect x="5" y="3" width="14" height="18" rx="2"/><path d="M9 8h6M9 12h6M9 16h4"/></svg>',
    "bars": '<svg viewBox="0 0 24 24"><path d="M5 20V11M10 20V5M15 20v-8M20 20v-4"/></svg>',
    "shield": '<svg viewBox="0 0 24 24"><path d="M12 3 5 6v5c0 4.5 3 8.3 7 10 4-1.7 7-5.5 7-10V6l-7-3z"/><path d="m9 12 2 2 4-4"/></svg>',
    "db": '<svg viewBox="0 0 24 24"><ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6"/><path d="M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3"/></svg>',
    "trend": '<svg viewBox="0 0 24 24"><path d="M4 16l5-5 4 4 7-7"/><path d="M15 8h5v5"/></svg>',
}


def header_html():
    return ('<div class="vs-top"><div class="vs-badges">Read-only <span>·</span> Runs on 1 GPU '
            '<span>·</span> Data stays on your server</div></div>')


def hero_html():
    note = f'<div class="vs-mode">{E(ENGINE.note)}</div>' if ENGINE.note else ""
    return f'<div class="vs-hero">{LOGO.format(s=92)}{WORD}</div>{note}'


def fmt(v):
    # thousands separators from 10,000 up, so years and ids (1944, 2009) stay as they are
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e15:
            v = int(v)
        else:
            return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) >= 10_000 else f"{v:.4f}".rstrip("0").rstrip(".")
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}" if abs(v) >= 10_000 else str(v)
    return "NULL" if v is None else str(v)


def table_html(cols, rows):
    if not rows:
        return '<div class="vs-empty">The query ran but returned no rows.</div>'
    num = [len(cols) > 1 and all(isinstance(r[i], (int, float)) or r[i] is None for r in rows) for i in range(len(cols))]
    head = "".join(f'<th class="{"num" if num[i] else ""}">{E(str(c))}</th>' for i, c in enumerate(cols))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if num[i] else ""}">{E(fmt(v))}</td>' for i, v in enumerate(r))
                   + "</tr>" for r in rows[:SHOW_ROWS])
    shown = min(len(rows), SHOW_ROWS)
    foot = f"Showing all {len(rows)} rows" if shown == len(rows) else f"Showing {shown} of {len(rows):,} rows"
    if len(rows) == 1:
        foot = "1 row"
    return (f'<div class="vs-table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
            f'<div class="vs-tfoot">{foot}</div></div>')


def diff_html(old, new):
    tok = re.compile(r"\w+|[^\w\s]|\s+")
    a, b = tok.findall(old or ""), tok.findall(new or "")
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(E("".join(a[i1:i2])))
        else:
            if i2 > i1:
                out.append(f'<del>{E("".join(a[i1:i2]))}</del>')
            if j2 > j1:
                out.append(f'<ins>{E("".join(b[j1:j2]))}</ins>')
    return f'<pre class="vs-diff">{"".join(out)}</pre>'


def steps_html(steps, running=None):
    items = []
    for s in steps:
        t = f'<span class="vs-time">{E(s["time"])}</span>' if s.get("time") else ""
        detail = s.get("detail", "")
        items.append(f'<li>{ICON[s["icon"]]}<div class="vs-step"><div class="vs-step-line"><span>{E(s["text"])}</span>{t}</div>{detail}</div></li>')
    if running:
        items.append(f'<li class="now">{ICON["spin"]}<div class="vs-step"><div class="vs-step-line"><span>{E(running)}</span></div></div></li>')
    return f'<ol class="vs-steps">{"".join(items)}</ol>'


def secs(dt):
    return "" if dt is None else (f"{dt * 1000:.0f} ms" if dt < 1 else f"{dt:.1f} s")


def context_html(db, ex):
    hint = f'<span class="vs-hint">Hint used: {E(ex["evidence"])}</span>' if ex and ex.get("evidence") else ""
    mode = f'<span class="vs-modetag">{"Replay of saved answers" if ENGINE.mode == "replay" else "Test mode: small untrained model"}</span>' \
        if ENGINE.mode != "live" else ""
    return f'<div class="vs-context"><span class="vs-dbchip">{SVG["db"]}Database: {E(db)}</span>{mode}{hint}</div>'


# ------------------------------------------------------------------ the self-check loop
def self_check(db, question):
    """Generator: yields ("progress", steps, running_text) and finally ("done", result)."""
    info = DBS[db]
    ex = find_example(db, question)
    prompt = build_prompt(info["path"], question, ex["evidence"] if ex else "")
    if isinstance(ENGINE, ReplayEngine):
        ENGINE.current_db = db
    steps, attempts = [], []
    gen_time = (lambda dt: "") if ENGINE.mode == "replay" else secs

    yield ("progress", steps, "Writing the SQL query…")
    t = time.perf_counter()
    out = ENGINE.generate("verisql", prompt)
    if out is None:
        yield ("done", {"missing": True, "db": db, "question": question, "steps": steps, "ex": ex})
        return
    sql = extract_sql(out)
    steps.append({"icon": "ok" if sql else "warn", "text": "Wrote the SQL query" if sql else "Could not find a query in the answer",
                  "time": gen_time(time.perf_counter() - t)})
    attempts.append({"sql": sql, "raw": out})
    messages = list(prompt) + [{"role": "assistant", "content": out}]
    turns, fixed_errors = 1, 0

    def run_and_log(label_ok, label_err):
        cols, rows, err, dt = run_query(info["path"], sql)
        attempts[-1].update({"cols": cols, "rows": rows, "err": err})
        if err:
            steps.append({"icon": "warn", "text": f"{label_err}: error, {err}", "time": secs(dt)})
        else:
            steps.append({"icon": "ok", "text": f"{label_ok}: {len(rows):,} row{'s' if len(rows) != 1 else ''}", "time": secs(dt)})
        return err

    yield ("progress", steps, "Running the query…")
    err = run_and_log("Ran the query", "Ran the query")
    while turns < MAX_TURNS:
        yield ("progress", steps, "Fixing the query…" if err else "Checking the result…")
        _, msg = feedback(sql, info["path"])  # exactly what the model saw in training
        messages.append({"role": "user", "content": msg})
        t = time.perf_counter()
        out = ENGINE.generate("verisql", messages) or ""
        dt = time.perf_counter() - t
        messages.append({"role": "assistant", "content": out})
        turns += 1
        new = clean_sql(extract_sql(out)) or sql  # no new SQL = keeps the previous query
        if same_query(new, sql):
            attempts[-1]["reply"] = out
            if err:
                steps.append({"icon": "err", "text": "Could not fix the error", "time": gen_time(dt)})
            else:
                steps.append({"icon": "ok", "text": "Checked the result and kept the query", "time": gen_time(dt)})
            break
        steps.append({"icon": "ok", "text": "Fixed the query" if err else "Revised the query", "time": gen_time(dt),
                      "detail": diff_html(sql, new)})
        fixed_errors += bool(err)
        sql = new
        attempts.append({"sql": sql, "raw": out})
        yield ("progress", steps, "Running the fixed query…" if err else "Running the revised query…")
        err = run_and_log("Ran the fixed query" if fixed_errors else "Ran the revised query", "Ran it again")

    final = attempts[-1]
    match = None
    if ex:
        gold_rows, gold_err = execute(info["path"], ex["gold_sql"], timeout=30)
        if gold_err is None and final.get("err") is None and final.get("rows") is not None:
            match = results_match(final["rows"], gold_rows, has_order_by(ex["gold_sql"]))
        elif final.get("err"):
            match = False
    yield ("done", {"db": db, "question": question, "ex": ex, "steps": steps, "attempts": attempts, "turns": turns,
                    "fixed_errors": fixed_errors, "final": final, "match": match, "prompt": prompt})


def status_html(res):
    final = res["final"]
    n = res["turns"]
    parts = []
    if final.get("err"):
        kind, parts = "bad", ["Could not get a query that runs", E(final["err"])]
    else:
        kind = "good"
        parts = ["Ran successfully", f"self-checked in {n} turn{'s' if n != 1 else ''}"]
        if res["fixed_errors"]:
            parts.append(f"fixed {res['fixed_errors']} error{'s' if res['fixed_errors'] != 1 else ''}")
        if res["match"] is True:
            parts.append("matches the expected answer")
        elif res["match"] is False:
            kind = "mixed"
            parts.append("does not match the expected answer")
    icon = ICON["ok"] if kind == "good" else ICON["warn"] if kind == "mixed" else ICON["err"]
    summary = f'{icon}<span>{" · ".join(parts)}</span><span class="vs-caret"></span>'
    return (f'<details class="vs-status {kind}"><summary>{summary}</summary>'
            f'{steps_html(res["steps"])}</details>')


def meta_html(final):
    rows = final.get("rows") or []
    if final.get("err") or not rows:
        return '<div class="vs-meta"></div>'
    top = f" · top row: {E(fmt(rows[0][0]))}" if rows and rows[0] else ""
    return f'<div class="vs-meta">{len(rows):,} row{"s" if len(rows) != 1 else ""}{top}</div>'


def write_csv(final):
    if final.get("err") or final.get("rows") is None:
        return None
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, f"verisql-result-{int(time.time() * 1000)}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(final["cols"])
        w.writerows(final["rows"])
    return path


def checked_html(res):
    items = []
    for i, a in enumerate(res["attempts"], 1):
        if a.get("err"):
            reply = f'<div class="vs-reply err">Database: error, {E(a["err"])}</div>'
        elif a.get("rows") is not None:
            first = ", ".join("(" + ", ".join(fmt(v) for v in r) + ")" for r in a["rows"][:3])
            more = " …" if len(a["rows"]) > 3 else ""
            reply = f'<div class="vs-reply">Database: {len(a["rows"]):,} row{"s" if len(a["rows"]) != 1 else ""}. First rows: {E(first)}{more}</div>'
        else:
            reply = ""
        said = ""
        if a.get("reply"):
            if clean_sql(extract_sql(a["reply"])) is None:  # answered in words, e.g. "Yes, the query is correct"
                text = re.sub(r"</?answer>", "", a["reply"]).strip()
                said = f'<div class="vs-said">Model: “{E(text[:300])}” (no new query, so the query was kept)</div>'
            else:
                said = '<div class="vs-said">Model: wrote the same query again, so it was kept.</div>'
        items.append(f'<div class="vs-attempt"><div class="vs-att-h">Attempt {i}</div>'
                     f'<pre class="vs-code">{E(a["sql"] or "(no SQL found)")}</pre>{reply}{said}</div>')
    note = ("The model never sees the correct answer: it only sees what the database returned, "
            "and decides whether to keep or change its query.")
    return f'<div class="vs-panel"><p class="vs-panel-note">{note}</p>{"".join(items)}</div>'


def compare_html(cards, note=""):
    html_cards = []
    for c in cards:
        sql = f'<details class="vs-sql-peek"><summary>SQL</summary><pre class="vs-code">{E(c["sql"] or "(no SQL found)")}</pre></details>' if c.get("sql") is not None else ""
        html_cards.append(f'<div class="vs-mcard {"me" if c["key"] == "verisql" else ""}"><div class="vs-mhead">'
                          f'<div><div class="vs-mname">{E(c["name"])}</div><div class="vs-msub">{E(c["sub"])}</div></div>'
                          f'<span class="vs-pill {c["tone"]}">{E(c["badge"])}</span></div>{sql}</div>')
    foot = f'<p class="vs-panel-note">{E(note)}</p>' if note else ""
    return f'<div class="vs-panel"><div class="vs-mgrid">{"".join(html_cards)}</div>{foot}</div>'


def trust_html():
    tiles = []
    for label in ["Spider", "BIRD"]:
        if label in TRUST:
            v, n = TRUST[label]
            tip = f"Strict execution accuracy after the self-check, on {n:,} {label} dev questions."
            tiles.append(f'<div class="vs-tile" title="{E(tip)}">{SVG["bars" if label == "Spider" else "db"]}'
                         f'<div><div class="vs-big">{v:.1f}%</div><div class="vs-small">on {label} dev</div></div></div>')
    g, m = TRUST.get("broken", (None, None))
    if g is not None and m is not None:
        tip = ("Questions answered correctly at the first attempt but wrong after the feedback, on BIRD dev: "
               "GRPO trained on single answers vs VeriSQL trained with the self-check loop.")
        tiles.append(f'<div class="vs-tile" title="{E(tip)}">{SVG["trend"]}<div><div class="vs-big">{g} → {m}</div>'
                     f'<div class="vs-small">correct answers broken after feedback on BIRD dev (GRPO → VeriSQL)</div></div></div>')
    note = ("VeriSQL is Qwen2.5-Coder-1.5B with a LoRA adapter trained with GRPO inside the self-check loop. "
            "Strict execution accuracy: the result must match the correct query's result exactly. "
            "Full results and significance tests are in RESULTS.md.")
    return f'<div class="vs-panel"><div class="vs-tiles">{"".join(tiles)}</div><p class="vs-panel-note">{note}</p></div>'


def preview_html(db, table):
    if not db or not table or db not in DBS:
        return ""
    try:
        con = sqlite3.connect(f"file:{DBS[db]['path']}?mode=ro", uri=True)
        con.text_factory = lambda b: b.decode(errors="ignore")
        cur = con.execute(f'SELECT * FROM "{table}" LIMIT 3')
        cols = [d[0] for d in cur.description][:4]
        rows = [r[:4] for r in cur.fetchall()]
        n = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        con.close()
    except sqlite3.Error as e:
        return f'<div class="vs-preview">Could not read the table: {E(str(e))}</div>'
    head = "".join(f"<th>{E(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{E(fmt(v))}</td>" for v in r) + "</tr>" for r in rows)
    return (f'<div class="vs-preview"><div class="vs-preview-h">Preview: {E(table)} <span>{n:,} rows</span></div>'
            f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


# ------------------------------------------------------------------ the app
def build():
    start_db = next((n for n in FEATURED if n in DBS), next(iter(DBS), None))
    blocks_kw = {"title": "VeriSQL"}
    if "css" in inspect.signature(gr.Blocks.__init__).parameters:
        blocks_kw["css"] = CSS
        blocks_kw["theme"] = gr.themes.Base()
    has_sidebar = hasattr(gr, "Sidebar")

    with gr.Blocks(**blocks_kw) as demo:
        last = gr.State(None)
        recent_state = gr.State([])

        # ---------- side menu
        side = gr.Sidebar(open=False, elem_classes="vs-side") if has_sidebar else gr.Accordion("Databases", open=False)
        with side:
            gr.HTML('<div class="vs-side-h">Databases</div>')
            db_search = gr.Textbox(placeholder="Search databases…", show_label=False, container=False, elem_classes="vs-side-search")
            db_list = gr.Radio(db_names(), value=start_db, show_label=False, container=False, elem_classes="vs-dblist")
            tbl_list = gr.Radio(tables_of(DBS[start_db]["path"]) if start_db else [], show_label=False, container=False,
                                elem_classes="vs-tbllist")
            preview = gr.HTML()
            side_upload = gr.UploadButton("+  Upload .sqlite (read-only)", file_types=[".sqlite", ".db", ".sqlite3"],
                                          elem_classes="vs-upload")
            gr.HTML('<div class="vs-side-h small">Recent</div>')
            recent = gr.Dataset(components=[gr.Textbox(visible=False)], samples=[], elem_classes="vs-recent", label="Recent",
                                samples_per_page=6)

        # ---------- top bar
        with gr.Row(elem_classes="vs-toprow"):
            logo_btn = gr.Button("Veri", elem_classes="vs-logo-btn", visible=False, scale=0, min_width=150)
            gr.HTML(header_html(), elem_classes="vs-tophtml")
            theme_btn = gr.Button("Theme", elem_classes="vs-theme", scale=0, min_width=40)

        hero = gr.HTML(hero_html(), elem_classes="vs-herobox")

        # ---------- search bar
        with gr.Row(elem_classes="vs-search", equal_height=True):
            db_pick = gr.Dropdown(db_names(), value=start_db, show_label=False, container=False, scale=0, min_width=170,
                                  elem_classes="vs-dbpick", interactive=True)
            q_in = gr.Textbox(placeholder="Ask your database a question…", show_label=False, container=False, scale=1,
                              elem_classes="vs-q", lines=1, max_lines=1, autofocus=True)
            go = gr.Button("➜", elem_classes="vs-go", scale=0, min_width=48)

        first = short_questions(start_db)
        with gr.Row(elem_classes="vs-chips") as chips:
            chip_btns = [gr.Button(first[i] if i < len(first) else "", visible=i < len(first), elem_classes="vs-chip",
                                   scale=0, min_width=0) for i in range(3)]
        with gr.Row(elem_classes="vs-uploadline") as upline:
            gr.HTML(f'<div class="vs-upline">Try {len(db_names())} sample databases, or</div>')
            home_upload = gr.UploadButton("upload your own .sqlite file", file_types=[".sqlite", ".db", ".sqlite3"],
                                          elem_classes="vs-linkbtn", size="sm")

        context = gr.HTML(visible=False)
        live = gr.HTML(visible=False)

        # ---------- the answer
        with gr.Column(visible=False, elem_classes="vs-answer") as answer:
            with gr.Row(elem_classes="vs-metarow", equal_height=True):
                meta = gr.HTML()
                copy_btn = gr.Button("Copy SQL", elem_classes="vs-small-btn", scale=0, min_width=110)
                dl_btn = gr.DownloadButton("Download CSV", elem_classes="vs-small-btn", scale=0, min_width=140)
            table = gr.HTML()
            with gr.Accordion("SQL", open=False, elem_classes="vs-sqlacc"):
                sql_code = gr.Code(language="sql", show_label=False, interactive=False)
            sql_text = gr.Textbox(visible=False)
            with gr.Row(elem_classes="vs-chips vs-follow"):
                follow_btns = [gr.Button("", visible=False, elem_classes="vs-chip", scale=0, min_width=0) for _ in range(2)]
            with gr.Row(elem_classes="vs-cards"):
                b_checked = gr.Button("How it was checked", elem_classes="vs-card c-checked")
                b_compare = gr.Button("Compare models", elem_classes="vs-card c-compare")
                b_trust = gr.Button("Why trust it", elem_classes="vs-card c-trust")
            panel = gr.HTML(visible=False)
            panel_open = gr.State("")

        # ---------- events
        def pick_db(db):
            tables = tables_of(DBS[db]["path"]) if db in DBS else []
            qs = short_questions(db)
            return [gr.Button(value=qs[i] if i < len(qs) else "", visible=i < len(qs)) for i in range(3)] + [
                gr.Radio(value=db), gr.Radio(choices=tables, value=None), ""]

        db_pick.change(pick_db, db_pick, chip_btns + [db_list, tbl_list, preview])
        db_list.input(lambda db: gr.Dropdown(value=db), db_list, db_pick)
        db_search.change(lambda f: gr.Radio(choices=db_names(f)), db_search, db_list)
        tbl_list.change(preview_html, [db_pick, tbl_list], preview)

        def add_upload(file):
            if file is None:
                return gr.Dropdown(), gr.Radio()
            src = file if isinstance(file, str) else file.name
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            base = re.sub(r"[^\w.-]", "_", os.path.splitext(os.path.basename(src))[0]) or "uploaded"
            dst = os.path.join(UPLOAD_DIR, f"{base}.sqlite")
            shutil.copy(src, dst)
            try:
                schema_text.cache_clear()
                if not tables_of(dst):
                    raise sqlite3.Error("no tables")
                schema_text(dst)
            except Exception as e:
                gr.Warning(f"That file could not be read as a SQLite database ({e}).")
                return gr.Dropdown(), gr.Radio()
            name, k = base, 2
            while name in DBS:
                name, k = f"{base}_{k}", k + 1
            DBS[name] = {"path": dst, "source": "Uploaded"}
            gr.Info(f"Added {name}. It is opened read-only.")
            return gr.Dropdown(choices=db_names(), value=name), gr.Radio(choices=db_names(), value=name)

        for up in (side_upload, home_upload):
            up.upload(add_upload, up, [db_pick, db_list])

        def ask(db, question, recent_list):
            question = (question or "").strip()
            empty_panel = (gr.HTML(visible=False), "")
            if not question or db not in DBS:
                yield {live: gr.HTML(value='<div class="vs-empty">Pick a database and type a question.</div>', visible=True)}
                return
            base = {logo_btn: gr.Button(visible=True), hero: gr.HTML(visible=False), chips: gr.Row(visible=False),
                    upline: gr.Row(visible=False), answer: gr.Column(visible=False), panel: empty_panel[0],
                    panel_open: "", context: gr.HTML(value=context_html(db, find_example(db, question)), visible=True)}
            for kind, *payload in self_check(db, question):
                if kind == "progress":
                    steps, running = payload
                    yield {**base, live: gr.HTML(value=steps_html(steps, running), visible=True)}
                    continue
                res = payload[0]
                if res.get("missing"):
                    yield {**base, live: gr.HTML(value='<div class="vs-status bad"><div class="vs-missing">This question is not in the '
                                                 'saved results. Replay mode can only answer Spider dev and BIRD dev questions; '
                                                 'pick one of the suggested questions.</div></div>', visible=True)}
                    return
                final = res["final"]
                new_recent = ([[question, db]] + [r for r in recent_list if r[0] != question])[:6]
                more = short_questions(db, 2, exclude=question)
                has_rows = not final.get("err") and final.get("rows") is not None
                yield {**base, live: gr.HTML(value=status_html(res), visible=True),
                       answer: gr.Column(visible=True), meta: meta_html(final),
                       table: table_html(final["cols"], final["rows"]) if has_rows else
                       f'<div class="vs-empty err">The last query did not run: {E(final.get("err") or "no SQL")}</div>',
                       sql_code: final["sql"] or "", sql_text: final["sql"] or "",
                       dl_btn: gr.DownloadButton(value=write_csv(final), visible=has_rows),
                       **{b: gr.Button(value=more[i] if i < len(more) else "", visible=i < len(more))
                          for i, b in enumerate(follow_btns)},
                       recent: gr.Dataset(samples=[[r[0]] for r in new_recent]), recent_state: new_recent, last: res}

        ask_outputs = [logo_btn, hero, chips, upline, answer, panel, panel_open, context, live, meta, table, sql_code, sql_text,
                       dl_btn, *follow_btns, recent, recent_state, last]
        go.click(ask, [db_pick, q_in, recent_state], ask_outputs)
        q_in.submit(ask, [db_pick, q_in, recent_state], ask_outputs)

        def from_sample(sample):
            return sample[0] if isinstance(sample, (list, tuple)) else sample

        for b in chip_btns + follow_btns:
            b.click(lambda v: v, b, q_in).then(ask, [db_pick, q_in, recent_state], ask_outputs)

        def from_recent(sample, recent_list):
            q = from_sample(sample)
            db = next((r[1] for r in recent_list if r[0] == q), None)
            return q, (gr.Dropdown(value=db) if db else gr.Dropdown())

        recent.click(from_recent, [recent, recent_state], [q_in, db_pick]).then(ask, [db_pick, q_in, recent_state], ask_outputs)

        # detail cards
        def show_checked(res, opened):
            if opened == "checked" or res is None:
                return gr.HTML(visible=False), ""
            return gr.HTML(value=checked_html(res), visible=True), "checked"

        def show_trust(opened):
            if opened == "trust":
                return gr.HTML(visible=False), ""
            return gr.HTML(value=trust_html(), visible=True), "trust"

        def show_compare(res, opened):
            if opened == "compare" or res is None:
                yield gr.HTML(visible=False), ""
                return
            yield gr.HTML(value='<div class="vs-panel"><p class="vs-panel-note">Asking the other models the same question…</p></div>',
                          visible=True), "compare"
            info = DBS[res["db"]]
            gold = res["ex"]["gold_sql"] if res.get("ex") else None
            gold_rows = execute(info["path"], gold, timeout=30)[0] if gold else None
            cards = []
            for key, name, sub in COMPARE:
                if key == "verisql":
                    final = res["final"]
                    if final.get("err"):
                        cards.append({"key": key, "name": name, "sub": "still an error after the self-check", "badge": "−0.1",
                                      "tone": "bad", "sql": final["sql"]})
                    else:
                        how = (f"fixed after {res['fixed_errors']} error{'s' if res['fixed_errors'] != 1 else ''}"
                               if res["fixed_errors"] else "right first time" if res["turns"] <= 2 else "revised once")
                        if res["match"] is False:
                            cards.append({"key": key, "name": name, "sub": f"ran, wrong result ({how})", "badge": "0",
                                          "tone": "mid", "sql": final["sql"]})
                        else:
                            cards.append({"key": key, "name": name, "sub": how if res["match"] else f"ran ({how})",
                                          "badge": "+1" if res["match"] else "ran", "tone": "good" if res["match"] else "plain",
                                          "sql": final["sql"]})
                    continue
                if key not in ENGINE.available:
                    cards.append({"key": key, "name": name, "sub": "not loaded", "badge": "–", "tone": "plain", "sql": None})
                    continue
                out = ENGINE.generate(key, res["prompt"])
                if out is None:
                    cards.append({"key": key, "name": name, "sub": "no saved answer", "badge": "–", "tone": "plain", "sql": None})
                    continue
                sql = extract_sql(out)
                rows, err = execute(info["path"], sql)
                if err:
                    cards.append({"key": key, "name": name, "sub": "SQL error", "badge": "−0.1", "tone": "bad", "sql": sql})
                elif gold_rows is not None:
                    ok = results_match(rows, gold_rows, has_order_by(gold))
                    cards.append({"key": key, "name": name, "sub": "correct" if ok else "wrong result",
                                  "badge": "+1" if ok else "0", "tone": "good" if ok else "mid", "sql": sql})
                else:
                    cards.append({"key": key, "name": name, "sub": "ran", "badge": "ran", "tone": "plain", "sql": sql})
            note = ("The other models answer once, with no feedback; the badges are the reward each answer gets "
                    "(+1 correct, 0 wrong, −0.1 error).") if gold_rows is not None else \
                ("This question has no stored correct answer, so the answers are shown as ran or SQL error only.")
            if ENGINE.mode == "replay":
                note += " Answers are the saved outputs from our evaluation runs."
            yield gr.HTML(value=compare_html(cards, note), visible=True), "compare"

        b_checked.click(show_checked, [last, panel_open], [panel, panel_open])
        b_compare.click(show_compare, [last, panel_open], [panel, panel_open])
        b_trust.click(show_trust, panel_open, [panel, panel_open])

        def go_home(db):
            qs = short_questions(db) if db in DBS else []
            return [gr.Button(visible=False), gr.HTML(visible=True), gr.Row(visible=True), gr.Row(visible=True),
                    gr.HTML(visible=False), gr.HTML(visible=False), gr.Column(visible=False), gr.HTML(visible=False), "", ""] + \
                [gr.Button(value=qs[i] if i < len(qs) else "", visible=i < len(qs)) for i in range(3)]

        logo_btn.click(go_home, db_pick, [logo_btn, hero, chips, upline, context, live, answer, panel, panel_open, q_in] + chip_btns)

        copy_btn.click(None, sql_text, None, js="(s) => { navigator.clipboard.writeText(s || ''); }")
        theme_js = """() => {
            const dark = !document.body.classList.contains('dark');
            for (const el of [document.documentElement, document.body, document.querySelector('gradio-app')]) {
                if (el) el.classList.toggle('dark', dark);
            }
            try { localStorage.setItem('verisql-theme', dark ? 'dark' : 'light'); } catch (e) {}
        }"""
        load_js = """() => {
            let want = null;
            try { want = localStorage.getItem('verisql-theme'); } catch (e) {}
            const p = new URLSearchParams(location.search).get('__theme');
            if (p === 'dark' || p === 'light') want = p;
            if (want === null) want = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
            for (const el of [document.documentElement, document.body, document.querySelector('gradio-app')]) {
                if (el) el.classList.toggle('dark', want === 'dark');
            }
        }"""
        theme_btn.click(None, None, None, js=theme_js)
        demo.load(None, None, None, js=load_js)
    return demo


CSS = open(os.path.join(HERE, "style.css")).read()


def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--verisql", help="multi-turn GRPO adapter (mt-grpo-1.5b-s0/checkpoint-150)")
    ap.add_argument("--dpo", help="DPO adapter, for Compare models (optional)")
    ap.add_argument("--rloo", help="RLOO adapter, for Compare models (optional)")
    ap.add_argument("--adapters-repo", help="Hugging Face Hub repo with all adapters, e.g. ayv443/verisql-adapters")
    ap.add_argument("--replay", action="store_true", help="answer from the saved evaluation runs (no GPU)")
    ap.add_argument("--test", action="store_true", help="small untrained model on CPU, only to test the app")
    ap.add_argument("--share", action="store_true", help="public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    if args.replay:
        ENGINE = ReplayEngine()
    elif args.test:
        ENGINE = TestEngine()
    else:
        if not (args.verisql or args.adapters_repo):
            ap.error("--verisql <adapter> or --adapters-repo <hub repo> is needed (or use --replay / --test)")
        ENGINE = LiveEngine(args)
    if not DBS:
        print(f"No databases found under {DATA_DIR}: run text2sql-rl/download_data.sh and prepare_data.py first.")
    demo = build()
    launch_kw = {"share": args.share, "server_port": args.port}
    if "css" not in inspect.signature(gr.Blocks.__init__).parameters:
        launch_kw["css"] = CSS  # Gradio 6 takes the css in launch()
        launch_kw["theme"] = gr.themes.Base()
    demo.queue().launch(**launch_kw)


if __name__ == "__main__":
    main()
