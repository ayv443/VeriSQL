"""HTTP contract of the service, with the replay engine (no model needed)."""
import pytest
from fastapi.testclient import TestClient

import core
from service.api import create_app, find_databases
from conftest import FIXTURES

DEMO_Q = "Show the names of singers and the total sales of their songs."


@pytest.fixture(scope="module")
def client():
    app = create_app(engine=core.ReplayEngine(), databases=find_databases([FIXTURES]))
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["engine"] == "replay" and body["databases"] >= 18


def test_databases(client):
    assert "singer" in client.get("/databases").json()["databases"]


def test_query_fixes_its_own_error(client):
    r = client.post("/query", json={"question": DEMO_Q, "db": "singer"})
    assert r.status_code == 200
    body = r.json()
    assert body["turns"][0]["error"] == "no such column: s.Sales"
    assert body["fixed_errors"] == 1 and body["error"] is None and body["row_count"] == 6
    assert body["columns"][0] == "Name" and len(body["rows"]) == 6


def test_unknown_database_is_404(client):
    assert client.post("/query", json={"question": DEMO_Q, "db": "../../etc/passwd"}).status_code == 404


@pytest.mark.parametrize("payload", [{"question": "", "db": "singer"}, {"question": "x" * 501, "db": "singer"},
                                     {"db": "singer"}])
def test_bad_input_is_422(client, payload):
    assert client.post("/query", json=payload).status_code == 422


def test_replay_rejects_unknown_questions(client):
    assert client.post("/query", json={"question": "What is the meaning of life?", "db": "singer"}).status_code == 422


def test_metrics_count_queries(client):
    before = client.get("/metrics").json()["queries"]
    client.post("/query", json={"question": DEMO_Q, "db": "singer"})
    m = client.get("/metrics").json()
    assert m["queries"] == before + 1 and m["failure_rate"] == 0.0 and m["latency_ms"]["p50"] is not None
