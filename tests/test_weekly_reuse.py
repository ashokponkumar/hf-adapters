"""Unit tests for carrying unchanged weekly-scan verdicts forward.

Run with ``pytest --noconftest``, like ``test_weekly_prefilter.py``: everything tested here is
pure or takes a stub sink, so no torch, database driver or ingest library is needed.
"""

from __future__ import annotations

from datetime import date

import pytest

from tests.spyre.weekly_generation import reuse
from tests.spyre.weekly_generation.sink.capability_write import capability_results

# ISO week 41 of 2026, so the default every-4-weeks forced re-test is not due.
_WEEK = date(2026, 10, 5)


def _v2(model, backend, status, run="r1", reason="", sha="s1", origin=""):
    return (
        model,
        "causal_lm",
        backend,
        status,
        reason,
        "",
        sha,
        "2026-01-01",
        origin or run,
        run,
    )


def _prior(**kw):
    base = dict(
        run_id="r1",
        hub_sha="s1",
        adapter_name="causal_lm",
        verified_on_cpu=True,
        verified_on_spyre=True,
        failure_category="",
        error="",
        added_date="2026-01-01",
    )
    base.update(kw)
    return reuse.Prior(**base)


class _Sink:
    def __init__(self):
        self.rows = []
        self.v2_props = {}

    def add_entry(self, **kw):
        self.rows.append(kw)


def test_priors_need_both_backends_from_one_run():
    rows = [
        _v2("a", "cpu", "passed"),
        _v2("a", "spyre", "failed", reason="verification_failed"),
        _v2("b", "cpu", "passed", run="r1"),
        _v2("b", "spyre", "passed", run="r2"),
        _v2("c", "cpu", "passed"),
    ]
    priors = reuse.priors_from_rows(rows)
    assert set(priors) == {"a"}
    a = priors["a"]
    assert (a.verified_on_cpu, a.verified_on_spyre) == (True, False)
    assert a.failure_category == "verification_failed"


def test_reused_from_run_points_at_the_run_that_tested():
    priors = reuse.priors_from_rows(
        [
            _v2("a", "cpu", "passed", run="r2", origin="r1"),
            _v2("a", "spyre", "passed", run="r2", origin="r1"),
        ]
    )
    assert priors["a"].run_id == "r1"


def test_split_reuses_only_unchanged_reusable_verdicts():
    rows = [
        {"model_id": "same", "hub_sha": "s1"},
        {"model_id": "new-commit", "hub_sha": "s2"},
        {"model_id": "no-sha", "hub_sha": ""},
        {"model_id": "flaky", "hub_sha": "s1"},
        {"model_id": "unseen", "hub_sha": "s1"},
    ]
    priors = {
        "same": _prior(),
        "new-commit": _prior(),
        "no-sha": _prior(),
        "flaky": _prior(verified_on_spyre=False, failure_category="worker_timeout"),
    }
    to_test, carried = reuse.split(rows, priors)
    assert [r["model_id"] for r in to_test] == [
        "new-commit",
        "no-sha",
        "flaky",
        "unseen",
    ]
    assert [r["model_id"] for r, _ in carried] == ["same"]


def test_record_writes_carried_verdict_with_this_weeks_catalog(monkeypatch):
    monkeypatch.setenv("WEEKLY_STACK_KEY", "k1")
    sink = _Sink()
    row = {"model_id": "a", "hub_sha": "s1", "downloads": 42, "curated": False}
    reuse.record(
        sink, row, _prior(verified_on_spyre=False, failure_category="x"), _WEEK
    )
    (entry,) = sink.rows
    assert entry["snapshot_date"] == _WEEK
    assert entry["num_downloads"] == 42
    assert (entry["verified_on_cpu"], entry["verified_on_spyre"]) == (True, False)
    assert entry["failure_category"] == "x"
    assert sink.v2_props["a"] == {
        "stack_key": "k1",
        "hub_sha": "s1",
        "reused_from_run": "r1",
    }


def test_run_props_reach_the_v2_rows():
    rows = capability_results(
        [
            {
                "model_name": "a",
                "adapter_name": "causal_lm",
                "verified_on_cpu": True,
                "verified_on_spyre": True,
            }
        ],
        {"a": {"stack_key": "k1", "reused_from_run": "r1"}},
    )
    assert {r["props"]["reused_from_run"] for r in rows} == {"r1"}


@pytest.mark.parametrize(
    "env,expected",
    [
        ({"WEEKLY_STACK_KEY": "k1"}, True),
        ({}, False),
        ({"WEEKLY_STACK_KEY": "k1", "WEEKLY_REUSE": "0"}, False),
        ({"WEEKLY_STACK_KEY": "k1", "WEEKLY_FORCE_RETEST_EVERY_WEEKS": "41"}, False),
        ({"WEEKLY_STACK_KEY": "k1", "WEEKLY_FORCE_RETEST_EVERY_WEEKS": "0"}, True),
    ],
)
def test_enabled(monkeypatch, env, expected):
    for k in ("WEEKLY_STACK_KEY", "WEEKLY_REUSE", "WEEKLY_FORCE_RETEST_EVERY_WEEKS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert reuse.enabled(_WEEK) is expected
