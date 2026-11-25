"""VeriSQL core: the self-checking loop, the guarded SQL runner and the model engines.

Shared by the FastAPI service (service/api.py), the evaluation suite (eval/run_eval.py) and the Gradio
demo (verisql/app.py), so all three run exactly the loop the model was trained and evaluated with
(extension/multiturn_eval.py, --mode all):

    turn 1   the model writes a query
    run      the query runs on a read-only copy of the database (guardrails below)
    feedback the model is shown the error, or the row count and the first 5 rows (never the gold answer)
    turn 2+  it repeats the query (= keep it, stop) or writes a new one, for at most 3 turns

Engines (all expose generate(messages, db_id) -> text or None):
    LiveEngine    Qwen2.5-Coder-1.5B-Instruct + a LoRA adapter (local folder or Hugging Face Hub repo)
    TestEngine    the small untrained 0.5B model on CPU, only to check that the plumbing works
    ReplayEngine  the model's saved outputs from our evaluation runs, no model needed (CI and the free demo)
"""
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(ROOT, "text2sql-rl"), os.path.join(ROOT, "extension")):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import MAX_COMPLETION_LEN, MODEL_NAME, build_prompt, extract_sql, load_jsonl  # noqa: E402
from multiturn_eval import clean_sql, feedback, same_query  # noqa: E402
from reward import MAX_ROWS, TIMEOUT_S  # noqa: E402

MAX_TURNS = 3
FINDINGS = os.path.join(ROOT, "extension", "findings", "multiturn")
SAVED_PHASE1 = os.path.join(ROOT, "text2sql-rl", "evaluation", "results")


# ------------------------------------------------------------------ guardrails
# The same rules as reward.execute (what the model was trained against), plus an authorizer as a second
# line of defence. Layers, from outside in:
#   1. only statements that start with SELECT or WITH are run
#   2. the connection is opened read-only (mode=ro), so writes fail even if 1 is bypassed (WITH ... DELETE)
#   3. an authorizer denies ATTACH / DETACH / PRAGMA / transactions / writes inside the statement
#   4. sqlite runs one statement per call, so "SELECT 1; DROP TABLE t" is rejected
#   5. a progress handler stops the query after `timeout` seconds
#   6. at most `max_rows` rows are fetched
_ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}


def _authorizer(action, *_):
    return sqlite3.SQLITE_OK if action in _ALLOWED else sqlite3.SQLITE_DENY


@dataclass
class QueryResult:
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    error: str = None
    seconds: float = 0.0
    truncated: bool = False


def run_sql(db_file, sql, timeout=TIMEOUT_S, max_rows=MAX_ROWS):
    """Run one query under the guardrails above. Never raises: problems come back in .error."""
    t = time.perf_counter()
    if not sql:
        return QueryResult(error="no sql found", seconds=time.perf_counter() - t)
    if not sql.lstrip().lower().startswith(("select", "with")):
        return QueryResult(error="not a SELECT", seconds=time.perf_counter() - t)
    try:
        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True, check_same_thread=False)
    except sqlite3.Error as e:
        return QueryResult(error=f"cannot open db: {e}", seconds=time.perf_counter() - t)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    conn.set_authorizer(_authorizer)
    deadline = time.monotonic() + timeout
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        cols = [d[0] for d in cur.description or []]
        return QueryResult(cols, rows[:max_rows], None, time.perf_counter() - t, len(rows) > max_rows)
    except Exception as e:  # sqlite errors, interrupts, decode problems
        msg = "timeout" if "interrupted" in str(e) else str(e)
        return QueryResult(error=msg, seconds=time.perf_counter() - t)
    finally:
        conn.close()


# ------------------------------------------------------------------ engines
class Engine:
    mode = "base"
    label = ""

    def generate(self, messages, db_id=None):
        raise NotImplementedError


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"


class LiveEngine(Engine):
    """The trained model. `adapter` is a local checkpoint folder or a Hub repo id (with `subfolder`)."""
    mode = "live"

    def __init__(self, adapter, subfolder=None, base=MODEL_NAME, device=None, dtype=None):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or _device()
        # fp16 on a GPU (as in training); fp32 elsewhere unless asked (bf16 halves the memory on a small CPU box)
        dtype = getattr(torch, dtype) if dtype else (torch.float16 if self.device == "cuda" else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(self.device)
        kw = {"subfolder": subfolder} if subfolder else {}
        self.model = PeftModel.from_pretrained(model, adapter, **kw).eval()
        self.label = f"{base} + {adapter}{'/' + subfolder if subfolder else ''} on {self.device} ({str(dtype).split('.')[-1]})"

    def generate(self, messages, db_id=None):
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=MAX_COMPLETION_LEN, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


class TestEngine(LiveEngine):
    """Untrained 0.5B model, no adapter: checks the plumbing on a laptop CPU. Not VeriSQL's accuracy."""
    mode = "test"

    def __init__(self, base="Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device = torch, "cpu"
        self.tok = AutoTokenizer.from_pretrained(base)
        self.model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval()
        self.label = f"{base} (untrained, test mode)"


def _key(db_id, question):
    return ((db_id or "").split(" (")[0], " ".join(question.split()).lower())


class ReplayEngine(Engine):
    """Answers with the model's saved turns from our evaluation runs (MT-GRPO step 150 by default).
    The databases are still queried live, so the guardrails, feedback and scoring are exercised for real.
    Only Spider dev / BIRD dev questions can be answered; anything else returns None."""
    mode = "replay"

    def __init__(self, tag="mt-grpo"):
        self.saved = {}
        for split in ["spider_dev", "bird_dev"]:
            p = os.path.join(FINDINGS, tag, split, "all", "per_question.jsonl")
            if os.path.exists(p):
                for r in load_jsonl(p):
                    turns = [m["content"] for m in r["conversation"] if m["role"] == "assistant"]
                    self.saved[_key(r["db_id"], r["question"])] = turns
        self.label = f"replay of saved {tag} outputs ({len(self.saved)} questions)"

    def has(self, db_id, question):
        return _key(db_id, question) in self.saved

    def generate(self, messages, db_id=None):
        question = messages[1]["content"].rsplit("Question:", 1)[-1].strip()
        turns = self.saved.get(_key(db_id, question))
        if not turns:
            return None
        k = sum(m["role"] == "assistant" for m in messages)  # which turn we are on
        return turns[min(k, len(turns) - 1)]


class ScriptedEngine(Engine):
    """Returns fixed answers in order. For unit tests."""
    mode = "scripted"

    def __init__(self, answers):
        self.answers = list(answers)

    def generate(self, messages, db_id=None):
        k = sum(m["role"] == "assistant" for m in messages)
        return self.answers[min(k, len(self.answers) - 1)]


def make_engine(kind, adapter=None, subfolder=None, dtype=None):
    if kind == "replay":
        return ReplayEngine()
    if kind == "test":
        return TestEngine()
    if kind == "live":
        if not adapter:
            raise ValueError("live mode needs an adapter (local folder or Hub repo id)")
        return LiveEngine(adapter, subfolder, dtype=dtype)
    raise ValueError(f"unknown engine {kind!r}: use live, test or replay")


# ------------------------------------------------------------------ the self-check loop
class NoAnswer(Exception):
    """The engine has no answer (replay mode, question not in the saved runs)."""


def solve(engine, db_file, question, evidence="", db_id=None, max_turns=MAX_TURNS):
    """Ask the question, run the self-check loop, return everything that happened.

    Returns a dict: final_sql, columns, rows, error, turns (one entry per model turn), turns_used,
    fixed_errors, gen_seconds, sql_seconds. Raises NoAnswer if the engine has nothing to say.
    """
    messages = build_prompt(db_file, question, evidence)
    turns, gen_s = [], 0.0

    def ask():
        nonlocal gen_s
        t = time.perf_counter()
        out = engine.generate(messages, db_id)
        gen_s += time.perf_counter() - t
        return out

    sql_s = 0.0

    def run(q):
        nonlocal sql_s
        r = run_sql(db_file, q)
        sql_s += r.seconds
        return r

    out = ask()
    if out is None:
        raise NoAnswer(question)
    sql = extract_sql(out)  # turn 1 is read exactly as in the single-turn evaluation
    messages.append({"role": "assistant", "content": out})
    result = run(sql)
    turns.append({"sql": sql, "error": result.error, "row_count": len(result.rows), "kept": False})
    fixed = 0
    while len(turns) < max_turns:
        _, msg = feedback(sql, db_file)  # exactly the text the model saw in training
        messages.append({"role": "user", "content": msg})
        out = ask() or ""
        messages.append({"role": "assistant", "content": out})
        new = clean_sql(extract_sql(out)) or sql  # no new SQL = keep the previous query
        if same_query(new, sql):
            turns.append({"sql": sql, "error": result.error, "row_count": len(result.rows), "kept": True})
            break
        fixed += result.error is not None
        sql = new
        result = run(sql)
        turns.append({"sql": sql, "error": result.error, "row_count": len(result.rows), "kept": False})
    return {"final_sql": sql, "columns": result.columns, "rows": result.rows, "error": result.error,
            "truncated": result.truncated, "turns": turns, "turns_used": len(turns), "fixed_errors": fixed,
            "gen_seconds": gen_s, "sql_seconds": sql_s}
