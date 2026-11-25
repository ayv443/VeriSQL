"""The self-check loop: keep, fix, give up, and what counts as 'no new SQL'."""
import os

from conftest import FIXTURES
import core

SINGER = os.path.join(FIXTURES, "singer", "singer.sqlite")
Q = "How many singers are there?"
GOOD = "<answer>SELECT count(*) FROM singer</answer>"
BAD = "<answer>SELECT count(*) FROM singers</answer>"  # no such table


def solve(*answers):
    return core.solve(core.ScriptedEngine(answers), SINGER, Q)


def test_correct_answer_is_kept_after_one_check():
    r = solve(GOOD, GOOD)
    assert r["turns_used"] == 2 and r["turns"][-1]["kept"]
    assert r["error"] is None and r["rows"] == [(8,)] and r["fixed_errors"] == 0


def test_error_is_fixed_then_kept():
    r = solve(BAD, GOOD, GOOD)
    assert [t["error"] is not None for t in r["turns"]] == [True, False, False]
    assert r["turns_used"] == 3 and r["fixed_errors"] == 1 and r["rows"] == [(8,)]


def test_stops_after_three_turns():
    r = solve(BAD, "<answer>SELECT x FROM nowhere</answer>", "<answer>SELECT y FROM nowhere</answer>", GOOD)
    assert r["turns_used"] == 3 and r["error"] is not None


def test_reply_without_sql_keeps_previous_query():
    r = solve(GOOD, "Yes, that answers the question.")
    assert r["turns_used"] == 2 and r["final_sql"] == "SELECT count(*) FROM singer"


def test_feedback_never_contains_gold_answer():
    seen = []

    class Spy(core.ScriptedEngine):
        def generate(self, messages, db_id=None):
            seen.append(messages[-1]["content"])
            return super().generate(messages, db_id)

    core.solve(Spy([BAD, GOOD, GOOD]), SINGER, Q)
    feedback = seen[1:]
    assert "no such table: singers" in feedback[0]
    assert all("gold" not in f.lower() for f in feedback)
