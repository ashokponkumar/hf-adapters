"""artifact_link: one artifact_results leg per weekly scan, over its capability verdicts."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from tests.spyre.weekly_generation.sink import artifact_link

ENV = {
    "GITHUB_RUN_ID": "123456",
    "GITHUB_RUN_ATTEMPT": "2",
    "GITHUB_REPOSITORY": "torch-spyre/hf-adapters",
    "GITHUB_REF_NAME": "main",
    "GITHUB_SHA": "a" * 40,
}


class _Client:
    def __init__(self, verdicts: int):
        self.verdicts = verdicts
        self.queries: list[dict[str, Any]] = []

    def query(self, sql: str, parameters: dict[str, Any]):
        self.queries.append(parameters)
        return types.SimpleNamespace(result_rows=[(self.verdicts,)])


@pytest.fixture
def written(monkeypatch) -> dict[str, list[dict[str, Any]]]:
    calls: dict[str, list[dict[str, Any]]] = {"ensure": [], "result": []}
    lib = types.ModuleType("spyre_clickhouse_ingest")
    lib.run_id_of = lambda *parts: "run-" + "-".join(parts)  # type: ignore[attr-defined]

    def ensure(client, db, spec, arch, **kw):
        if spec.count("|") != 2:
            raise ValueError(f"{spec!r} names no artifact")
        calls["ensure"].append({"db": db, "spec": spec, "arch": arch, **kw})
        return types.SimpleNamespace(
            artifact_id=spec.removeprefix("gha:").split("|")[0]
        )

    def insert(client, db, **kw):
        calls["result"].append({"db": db, **kw})
        return True

    lib.ensure_artifact = ensure  # type: ignore[attr-defined]
    lib.insert_artifact_result = insert  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spyre_clickhouse_ingest", lib)
    return calls


def test_the_leg_carries_the_record_and_the_scan_run(written):
    client = _Client(verdicts=42)
    record = "aid|base|hf-adapters@abc,torch-spyre@def"
    assert artifact_link.link(client, "spyre_v2", record, ENV)
    (art,) = written["ensure"]
    assert (art["spec"], art["arch"], art["component"]) == (
        f"gha:{record}",
        "x86_64",
        "hf-adapters",
    )
    assert art["sources"] == [("torch-spyre/hf-adapters", "main", "a" * 40)]
    (call,) = written["result"]
    assert call["artifact_id"] == "aid"
    # The run_id capability_write stamps on the verdicts, so the leg joins them.
    assert call["run_id"] == "run-gha-123456-x86_64-model_support"
    assert client.queries[0]["run_id"] == call["run_id"]
    assert (call["result_kind"], call["test_type"], call["state"]) == (
        "capability",
        "model_support",
        "passed",
    )
    assert call["attempt"] == 2
    assert call["props"] == {
        "run_url": "https://github.com/torch-spyre/hf-adapters/actions/runs/123456",
        "source": "gha",
    }


def test_a_scan_with_no_verdicts_is_an_error_leg(written):
    assert artifact_link.link(_Client(verdicts=0), "spyre_v2", "aid|base|", ENV)
    assert written["result"][0]["state"] == "error"


@pytest.mark.parametrize(
    "record, env",
    [
        ("", ENV),
        ("aid", ENV),
        ("aid|base|x", {k: v for k, v in ENV.items() if k != "GITHUB_RUN_ID"}),
    ],
)
def test_nothing_is_written_without_an_artifact_or_a_run(written, record, env):
    assert not artifact_link.link(_Client(verdicts=5), "spyre_v2", record, env)
    assert written == {"ensure": [], "result": []}
