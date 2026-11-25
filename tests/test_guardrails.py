"""The SQL runner must never change a database, read other files, or run forever."""
import os
import shutil
import sqlite3

import pytest

from conftest import FIXTURES
import core

SINGER = os.path.join(FIXTURES, "singer", "singer.sqlite")


@pytest.fixture
def db_copy(tmp_path):
    p = tmp_path / "singer.sqlite"
    shutil.copy(SINGER, p)
    return str(p)


def row_count(path):
    con = sqlite3.connect(path)
    n = con.execute("SELECT count(*) FROM singer").fetchone()[0]
    con.close()
    return n


def test_select_works():
    r = core.run_sql(SINGER, "SELECT Name FROM singer ORDER BY Name")
    assert r.error is None and r.columns == ["Name"] and len(r.rows) == 8


@pytest.mark.parametrize("sql", [
    "DELETE FROM singer",
    "DROP TABLE singer",
    "INSERT INTO singer (Name) VALUES ('x')",
    "UPDATE singer SET Name = 'x'",
    "PRAGMA writable_schema = 1",
    "ATTACH DATABASE '/tmp/other.db' AS other",
    "CREATE TABLE t (x)",
])
def test_non_select_statements_are_refused(db_copy, sql):
    before = row_count(db_copy)
    r = core.run_sql(db_copy, sql)
    assert r.error == "not a SELECT"
    assert row_count(db_copy) == before


def test_write_hidden_behind_with_is_blocked(db_copy):
    # passes the SELECT/WITH prefix check, so the read-only connection and the authorizer must stop it
    before = row_count(db_copy)
    r = core.run_sql(db_copy, "WITH x AS (SELECT 1) DELETE FROM singer")
    assert r.error is not None
    assert row_count(db_copy) == before


def test_second_statement_is_not_run(db_copy):
    before = row_count(db_copy)
    r = core.run_sql(db_copy, "SELECT 1; DELETE FROM singer")
    assert r.error is not None
    assert row_count(db_copy) == before


def test_timeout_stops_runaway_query():
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT max(x) FROM c"
    r = core.run_sql(SINGER, sql, timeout=0.5)
    assert r.error == "timeout"
    assert r.seconds < 3


def test_row_cap():
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 500) SELECT x FROM c"
    r = core.run_sql(SINGER, sql, max_rows=100)
    assert r.error is None and len(r.rows) == 100 and r.truncated


def test_missing_sql():
    assert core.run_sql(SINGER, None).error == "no sql found"
