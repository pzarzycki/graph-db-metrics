"""Opt-in real Codex preparation followed by loading and credential-free reuse."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from graph_metrics.io import read_profile
from graph_metrics.models import (
    NodeSpec,
    Profile,
    PropertySpec,
    RelationshipSpec,
    Rule,
    Semantics,
    Targets,
)
from graph_metrics.preparation import _save_prepared, prepare

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("GRAPH_METRICS_RUN_CODEX") != "1",
        reason="set GRAPH_METRICS_RUN_CODEX=1 for live Codex",
    ),
]


def test_live_preparation_load_and_reuse_without_codex(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_METRICS_PROJECT", str(tmp_path))
    profile = Profile(
        semantics=Semantics(
            status="declared",
            concepts=[
                {
                    "cohort": "measurement",
                    "meaning": "Fictional benchmark measurements, not real people or a business domain. value is dimensionless and uniformly sampled in [1,2]. sample_id is a synthetic technical identity. Links are arbitrary directed benchmark connections with no additional semantic meaning.",
                }
            ],
            rules=[
                Rule(
                    id="identity",
                    kind="unique",
                    parameters={"cohort": "measurement", "property": "sample_id"},
                ),
                Rule(
                    id="value_range",
                    kind="range",
                    parameters={"cohort": "measurement", "property": "value", "min": 1, "max": 2},
                ),
            ],
        ),
        targets=Targets(
            nodes=[
                NodeSpec(
                    id="measurement",
                    labels=["SyntheticMeasurement"],
                    count=50,
                    properties={
                        "sample_id": PropertySpec(type="integer", distribution={"kind": "id"}),
                        "value": PropertySpec(
                            type="float", distribution={"kind": "uniform", "min": 1, "max": 2}
                        ),
                    },
                )
            ],
            relationships=[
                RelationshipSpec(
                    id="links",
                    type="SYNTHETIC_LINK",
                    source="measurement",
                    target="measurement",
                    count=100,
                    out_degrees=[{"degree": 2, "count": 50}],
                    in_degrees=[{"degree": 2, "count": 50}],
                )
            ],
            allow_self_loops=False,
            allow_parallel_edges=False,
        ),
    )
    prepared = prepare(
        profile, tmp_path, tmp_path / "live-preparation", mode="codex", max_helpers=0, timeout=300
    )
    assert not prepared.semantics.questions
    assert prepared.semantics.status in {"declared", "mixed", "inferred"}
    assert all(r.provenance in {"inferred", "declared"} for r in prepared.semantics.rules)
    frozen_path = tmp_path / "prepared-profile.yaml"
    _save_prepared(prepared, tmp_path, frozen_path)
    executable = Path(sys.executable).parent / "graph-metrics"
    # A valid saved preparation must work with no Codex executable on PATH.
    environment = dict(os.environ, PATH="/usr/bin:/bin")
    try:
        for name, target in [("loaded", "local"), ("reused", "stage")]:
            result = subprocess.run(
                [
                    str(executable),
                    "generate",
                    "--profile",
                    str(frozen_path),
                    "--target",
                    target,
                    "--output",
                    str(tmp_path / name),
                ],
                env=environment,
                capture_output=True,
                text=True,
                timeout=240,
            )
            assert result.returncode == 0, result.stderr
            summary = json.loads(result.stdout)
            assert summary["nodes"] == 50 and summary["relationships"] == 100
            assert summary["fidelity"] == "passed"
            actual = read_profile(tmp_path / name / "analysis" / "profile.yaml")
            assert not actual.semantics.questions
    finally:
        connection = tmp_path / "loaded" / "connection.json"
        if connection.exists():
            name = json.loads(connection.read_text())["container"]
            labels = subprocess.run(
                ["docker", "inspect", name, "--format", "{{json .Config.Labels}}"],
                capture_output=True,
                text=True,
            )
            if labels.returncode == 0 and json.loads(labels.stdout).get(
                "graph-db-metrics.run"
            ) == str(tmp_path / "loaded"):
                subprocess.run(
                    ["docker", "rm", "-f", name], capture_output=True, check=True, timeout=30
                )
