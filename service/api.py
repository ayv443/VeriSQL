"""VeriSQL HTTP API.

    POST /query      {"question": "...", "db": "singer"} -> final SQL, rows and every self-check turn
    GET  /databases  the databases the service can query
    GET  /health     liveness + which engine is loaded
    GET  /metrics    request count, failure rate, latency percentiles (since start)

Configuration (environment variables):
    VERISQL_ENGINE     replay (default, no model) | live | test
    VERISQL_ADAPTER    live: LoRA checkpoint folder or Hugging Face Hub repo id
    VERISQL_SUBFOLDER  live: folder inside the Hub repo (e.g. mt-grpo-step150)
    VERISQL_DTYPE      live: float16 | bfloat16 | float32 (default: float16 on a GPU, float32 otherwise)
    VERISQL_DB_DIRS    colon-separated folders of <name>/<name>.sqlite databases (default: eval/fixtures)
    VERISQL_MAX_ROWS   rows returned to the client (default 200; the query itself is capped at 100,000)

Run:  uvicorn service.api:app --port 8000      (from the repository root)
"""
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "verisql"))
import core  # noqa: E402

log = logging.getLogger("verisql")
logging.basicConfig(level=logging.INFO, format="%(message)s")

MAX_ROWS_OUT = int(os.environ.get("VERISQL_MAX_ROWS", "200"))
DB_DIRS = os.environ.get("VERISQL_DB_DIRS", os.path.join(ROOT, "eval", "fixtures")).split(":")


def find_databases(dirs):
    """name -> path, for every <dir>/<name>/<name>.sqlite. Clients only ever send a name, never a path."""
    found = {}
    for d in dirs:
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                p = os.path.join(d, name, f"{name}.sqlite")
                if os.path.isfile(p):
                    found.setdefault(name, p)
    return found


# ------------------------------------------------------------------ metrics
class Metrics:
    """In-memory counters. Good enough for one process; a real deployment would export these to Prometheus."""

    def __init__(self, window=1000):
        self.lock = threading.Lock()
        self.started = time.time()
        self.requests = self.failures = self.sql_errors = 0
        self.latencies = deque(maxlen=window)  # seconds, last `window` /query calls

    def record(self, seconds, failed=False, sql_error=False):
        with self.lock:
            self.requests += 1
            self.failures += failed
            self.sql_errors += sql_error
            self.latencies.append(seconds)

    def snapshot(self):
        with self.lock:
            lat = sorted(self.latencies)
            pct = (lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))] * 1000, 1)) if lat else (lambda q: None)
            return {"uptime_s": round(time.time() - self.started), "queries": self.requests,
                    "failures": self.failures, "failure_rate": round(self.failures / self.requests, 4) if self.requests else 0.0,
                    "final_sql_errors": self.sql_errors,
                    "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "max": pct(1.0)}}


# ------------------------------------------------------------------ app
class QueryIn(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["Show the names of singers and the total sales of their songs."])
    db: str = Field(examples=["singer"])
    evidence: str = Field("", max_length=1000, description="optional hint, as in BIRD")


class Turn(BaseModel):
    sql: str | None
    error: str | None
    row_count: int
    kept: bool


class QueryOut(BaseModel):
    request_id: str
    sql: str | None
    columns: list[str]
    rows: list[list]
    row_count: int
    rows_truncated: bool
    error: str | None
    turns: list[Turn]
    turns_used: int
    fixed_errors: int
    latency_ms: float
    engine: str


def create_app(engine=None, databases=None):
    @asynccontextmanager
    async def lifespan(app):
        if app.state.engine is None:  # load the model once, at startup
            app.state.engine = core.make_engine(os.environ.get("VERISQL_ENGINE", "replay"),
                                                os.environ.get("VERISQL_ADAPTER"), os.environ.get("VERISQL_SUBFOLDER"),
                                                os.environ.get("VERISQL_DTYPE"))
        log.info(json.dumps({"event": "startup", "engine": app.state.engine.label, "databases": len(app.state.dbs)}))
        yield

    app = FastAPI(title="VeriSQL", version="1.0", lifespan=lifespan,
                  description="Self-checking text-to-SQL: writes a query, runs it read-only, fixes it from the database's reply.")
    app.state.engine = engine
    app.state.dbs = databases if databases is not None else find_databases(DB_DIRS)
    app.state.metrics = Metrics()
    app.state.lock = threading.Lock()  # one generation at a time: the model is not thread-safe

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        t = time.perf_counter()
        response = await call_next(request)
        log.info(json.dumps({"event": "http", "method": request.method, "path": request.url.path,
                             "status": response.status_code, "ms": round((time.perf_counter() - t) * 1000, 1)}))
        return response

    @app.get("/health")
    def health():
        e = app.state.engine
        return {"status": "ok" if e else "loading", "engine": e.mode if e else None,
                "model": e.label if e else None, "databases": len(app.state.dbs)}

    @app.get("/databases")
    def databases():
        return {"databases": sorted(app.state.dbs)}

    @app.get("/metrics")
    def metrics():
        return app.state.metrics.snapshot()

    @app.post("/query", response_model=QueryOut)
    def query(body: QueryIn):
        rid = uuid.uuid4().hex[:12]
        path = app.state.dbs.get(body.db)
        if path is None:
            raise HTTPException(404, f"unknown database {body.db!r}; see GET /databases")
        engine = app.state.engine
        if isinstance(engine, core.ReplayEngine) and not engine.has(body.db, body.question):
            raise HTTPException(422, "replay mode only answers Spider dev / BIRD dev questions from the saved runs")
        t = time.perf_counter()
        try:
            with app.state.lock:
                res = core.solve(engine, path, body.question.strip(), body.evidence, db_id=body.db)
        except Exception as e:  # model or loop failure: count it, log it, return 500
            app.state.metrics.record(time.perf_counter() - t, failed=True)
            log.exception(json.dumps({"event": "query_failed", "request_id": rid, "db": body.db, "error": str(e)}))
            raise HTTPException(500, f"internal error (request {rid})")
        dt = time.perf_counter() - t
        app.state.metrics.record(dt, sql_error=res["error"] is not None)
        log.info(json.dumps({"event": "query", "request_id": rid, "db": body.db, "turns": res["turns_used"],
                             "fixed_errors": res["fixed_errors"], "final_error": res["error"],
                             "gen_ms": round(res["gen_seconds"] * 1000, 1), "sql_ms": round(res["sql_seconds"] * 1000, 1),
                             "total_ms": round(dt * 1000, 1)}))
        rows = [list(r) for r in res["rows"][:MAX_ROWS_OUT]]
        return QueryOut(request_id=rid, sql=res["final_sql"], columns=res["columns"], rows=rows,
                        row_count=len(res["rows"]), rows_truncated=res["truncated"] or len(res["rows"]) > MAX_ROWS_OUT,
                        error=res["error"], turns=res["turns"], turns_used=res["turns_used"],
                        fixed_errors=res["fixed_errors"], latency_ms=round(dt * 1000, 1), engine=engine.mode)

    return app


app = create_app()
