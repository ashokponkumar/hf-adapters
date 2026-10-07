"""Unit tests for the weekly scan's schema-v2 capability rows.

Run with ``pytest --noconftest``, like ``test_weekly_prefilter.py``: ``capability_results`` is
pure, so no torch, database driver or ingest library is needed.
"""

from __future__ import annotations

import sys
import types

from tests.spyre.weekly_generation.sink import capability_write
from tests.spyre.weekly_generation.sink.capability_write import capability_results


def _row(**overrides):
    row = {
        "model_name": "org/model",
        "adapter_name": "causal_lm",
        "verified_on_cpu": True,
        "verified_on_gpu": False,
        "verified_on_spyre": False,
        "failure_category": "not-implemented-adapter",
    }
    row.update(overrides)
    return row


def test_no_gpu_verdict():
    backends = {r["backend"] for r in capability_results([_row()])}
    assert backends == {"cpu", "spyre"}


def test_status_and_fail_reason_per_backend():
    by_backend = {r["backend"]: r for r in capability_results([_row()])}
    assert by_backend["cpu"]["status"] == "passed"
    assert by_backend["cpu"]["fail_reason"] == ""
    assert by_backend["spyre"]["status"] == "failed"
    assert by_backend["spyre"]["fail_reason"] == "not-implemented-adapter"


def test_prefilter_skip_fails_tested_backends_only():
    rows = capability_results(
        [
            _row(
                adapter_name="",
                verified_on_cpu=False,
                failure_category="model_too_large",
            )
        ]
    )
    assert [(r["backend"], r["status"], r["name"]) for r in rows] == [
        ("cpu", "failed", "unknown"),
        ("spyre", "failed", "unknown"),
    ]


def test_the_v2_set_gets_its_own_connection(monkeypatch):
    opened: list[dict] = []
    shared, own = object(), object()
    lib = types.ModuleType("spyre_clickhouse_ingest")
    lib.get_client = lambda: shared  # type: ignore[attr-defined]
    lib.target_database = lambda: "shared_v2"  # type: ignore[attr-defined]
    lib.run_id_of = lambda *parts: "run"  # type: ignore[attr-defined]
    lib.tables_present = lambda client, db, tables: True  # type: ignore[attr-defined]
    lib.capabilities_already_ingested = lambda *a: False  # type: ignore[attr-defined]
    written: list[tuple] = []
    lib.insert_capabilities = lambda client, db, *a, **kw: written.append((client, db)) or 1  # type: ignore[attr-defined]
    schema = types.ModuleType("spyre_clickhouse_ingest.schema")
    schema.CAPABILITIES = schema.CAPABILITY_RUNS = None  # type: ignore[attr-defined]
    driver = types.ModuleType("clickhouse_connect")
    driver.get_client = lambda **kw: opened.append(kw) or own  # type: ignore[attr-defined]
    for name, mod in (
        ("spyre_clickhouse_ingest", lib),
        ("spyre_clickhouse_ingest.schema", schema),
        ("clickhouse_connect", driver),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("GITHUB_RUN_ID", "1")

    monkeypatch.delenv("CLICKHOUSE_V2_HOST", raising=False)
    capability_write.write([_row()])
    assert written == [(shared, "shared_v2")] and opened == []

    monkeypatch.setenv("CLICKHOUSE_V2_HOST", "v2.example")
    monkeypatch.setenv("CLICKHOUSE_V2_DB", "spyre_v2")
    monkeypatch.setenv("CLICKHOUSE_V2_USER", "v2user")
    capability_write.write([_row()])
    assert written[-1] == (own, "spyre_v2")
    assert (opened[0]["host"], opened[0]["user"], opened[0]["database"]) == (
        "v2.example",
        "v2user",
        "spyre_v2",
    )
