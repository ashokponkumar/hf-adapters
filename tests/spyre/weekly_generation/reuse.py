"""Carry a model's last v2 verdict forward when nothing it depends on has changed.

A verdict depends on the checkpoint (its Hub commit, ``hub_sha``) and on the stack that
tested it (``WEEKLY_STACK_KEY``, a digest the workflow computes from the pinned
torch-spyre commit, the hf_adapters tree and the image's packages). When both match the
model's latest v2 verdict, re-testing it would only reproduce that verdict, so the scan
records it again marked ``reused_from_run`` and spends no card on it.

Reads v2 only, never v1: v1 has no column for either key half, so it cannot say whether a
verdict still applies. Every failure here falls back to testing, never to skipping.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any

from tests.spyre.weekly_generation.failure_categories import (
    FAILURE_CATEGORY_HARDWARE_EXCEPTION,
    FAILURE_CATEGORY_TEST_EXECUTION_EXCEPTION,
    FAILURE_CATEGORY_WORKER_CRASHED,
    FAILURE_CATEGORY_WORKER_TIMEOUT,
)

# Outcomes of the run, not of the model: carrying one forward would pin a transient
# failure onto every later week.
_NOT_REUSABLE = frozenset(
    {
        FAILURE_CATEGORY_HARDWARE_EXCEPTION,
        FAILURE_CATEGORY_TEST_EXECUTION_EXCEPTION,
        FAILURE_CATEGORY_WORKER_CRASHED,
        FAILURE_CATEGORY_WORKER_TIMEOUT,
    }
)

_DEFAULT_FORCE_EVERY_WEEKS = 4


@dataclass(frozen=True)
class Prior:
    """A model's latest v2 verdict under the current stack key."""

    run_id: str
    hub_sha: str
    adapter_name: str
    verified_on_cpu: bool
    verified_on_spyre: bool
    failure_category: str
    error: str
    added_date: str

    @property
    def reusable(self) -> bool:
        return self.failure_category not in _NOT_REUSABLE


def stack_key() -> str:
    return os.environ.get("WEEKLY_STACK_KEY", "").strip()


def enabled(snapshot_date: date) -> bool:
    """Reuse is on unless disabled, keyless, or this is a forced full re-test week.

    The forced week bounds how long a verdict can be carried without a card behind it,
    in case something outside the key changed (a Hub-side file served differently, say).
    """
    if os.environ.get("WEEKLY_REUSE", "1").strip().lower() in ("0", "false", "no"):
        return False
    if not stack_key():
        return False
    every = int(
        os.environ.get("WEEKLY_FORCE_RETEST_EVERY_WEEKS") or _DEFAULT_FORCE_EVERY_WEEKS
    )
    return every <= 0 or snapshot_date.isocalendar().week % every != 0


_PRIORS_SQL = """
SELECT
    c.subject,
    c.name,
    r.backend,
    r.status,
    r.fail_reason,
    r.props['error'],
    r.props['hub_sha'],
    r.props['added_date'],
    if(r.props['reused_from_run'] != '', r.props['reused_from_run'], toString(r.run_id)),
    toString(r.run_id)
FROM {db:Identifier}.capability_runs AS r
INNER JOIN {db:Identifier}.capabilities AS c ON c.capability_id = r.capability_id
WHERE r.component = {component:String}
  AND r.test_type = {test_type:String}
  AND r.props['stack_key'] = {key:String}
  AND c.subject IN {subjects:Array(String)}
ORDER BY r.ts DESC
LIMIT 1 BY c.subject, r.backend
"""


def priors_from_rows(rows: list[tuple[Any, ...]]) -> dict[str, Prior]:
    """Fold per-backend rows into one ``Prior`` per model.

    A model is kept only when its cpu and spyre rows come from the same run, so the two
    halves of a carried verdict were always observed together.
    """
    by_model: dict[str, dict[str, tuple[Any, ...]]] = {}
    for row in rows:
        by_model.setdefault(str(row[0]), {})[str(row[2])] = row
    out: dict[str, Prior] = {}
    for model, backends in by_model.items():
        cpu, spyre = backends.get("cpu"), backends.get("spyre")
        if cpu is None or spyre is None or cpu[9] != spyre[9]:
            continue
        failing = spyre if spyre[3] != "passed" else cpu
        out[model] = Prior(
            run_id=str(spyre[8]),
            hub_sha=str(spyre[6]),
            adapter_name=str(spyre[1]),
            verified_on_cpu=cpu[3] == "passed",
            verified_on_spyre=spyre[3] == "passed",
            failure_category=str(failing[4]) if failing[3] != "passed" else "",
            error=str(failing[5] or ""),
            added_date=str(spyre[7] or ""),
        )
    return out


def lookup(model_ids: list[str]) -> dict[str, Prior]:
    """Latest v2 verdict per model under the current stack key; ``{}`` on any failure."""
    if not model_ids:
        return {}
    try:
        from spyre_clickhouse_ingest import get_client, target_database

        from tests.spyre.weekly_generation.sink.capability_write import (
            COMPONENT,
            TEST_TYPE,
        )

        db = target_database()
        if not db:
            return {}
        result = get_client().query(
            _PRIORS_SQL,
            parameters={
                "db": db,
                "component": COMPONENT,
                "test_type": TEST_TYPE,
                "key": stack_key(),
                "subjects": model_ids,
            },
        )
        return priors_from_rows(list(result.result_rows))
    except Exception as exc:  # reuse is an optimisation: never fail the scan over it
        print(f"  reuse: v2 lookup failed ({exc}); testing every model.")
        return {}


def split(
    rows: list[dict[str, Any]], priors: dict[str, Prior]
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], Prior]]]:
    """Partition *rows* into (to test, to carry forward), preserving order."""
    to_test: list[dict[str, Any]] = []
    carried: list[tuple[dict[str, Any], Prior]] = []
    for row in rows:
        prior = priors.get(str(row["model_id"]))
        hub_sha = str(row.get("hub_sha") or "")
        if prior and hub_sha and prior.hub_sha == hub_sha and prior.reusable:
            carried.append((row, prior))
        else:
            to_test.append(row)
    return to_test, carried


def record(sink: Any, row: dict[str, Any], prior: Prior, snapshot_date: date) -> None:
    """Write the carried verdict to v1 and v2 as this week's row for *row*'s model.

    v1 gets it too, so the snapshot the HF adapter page reads stays complete; its
    catalog facts (downloads, size) are this week's, only the verdict is carried.
    """
    try:
        added: date | None = date.fromisoformat(prior.added_date)
    except ValueError:
        added = None
    model = str(row["model_id"])
    sink.add_entry(
        model_name=model,
        config_class=str(row.get("config_class") or ""),
        adapter_name="" if prior.adapter_name == "unknown" else prior.adapter_name,
        added_date=added,
        snapshot_date=snapshot_date,
        verified_on_cpu=prior.verified_on_cpu,
        verified_on_gpu=False,
        verified_on_spyre=prior.verified_on_spyre,
        curated=bool(row.get("curated")),
        num_downloads=int(row.get("downloads") or 0),
        family=str(row.get("model_type") or ""),
        architecture=str(row.get("architectures") or ""),
        parameters_number=int(row.get("parameters") or 0),
        failure_category=prior.failure_category or None,
        error=prior.error or None,
    )
    sink.v2_props[model] = run_props(row, prior)


def run_props(row: dict[str, Any], prior: Prior | None = None) -> dict[str, str]:
    """The v2 props that make this verdict findable by next week's ``lookup``."""
    props = {"stack_key": stack_key(), "hub_sha": str(row.get("hub_sha") or "")}
    if prior is not None:
        props["reused_from_run"] = prior.run_id
    return {k: v for k, v in props.items() if v}
