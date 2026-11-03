"""Full million-node verification; explicitly opted in because it creates a new database."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.scale,
    pytest.mark.skipif(
        os.environ.get("GRAPH_METRICS_RUN_SCALE") != "1",
        reason="set GRAPH_METRICS_RUN_SCALE=1 for the million-node benchmark",
    ),
]


def test_million_node_native_round_trip(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("GRAPH_METRICS_PROJECT", str(tmp_path))
    output = tmp_path / "benchmark"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "benchmark.py"),
                "--output",
                str(output),
                "--nodes",
                "1000000",
                "--relationships",
                "5000000",
                "--target",
                "local",
            ],
            capture_output=True,
            text=True,
            timeout=15000,
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout
        report = json.loads((output / "benchmark.json").read_text())
        assert report["status"] == "completed"
        assert report["pipeline"]["loaded"] is True
        assert report["pipeline"]["nodes"] == 1_000_000
        assert report["pipeline"]["relationships"] == 5_000_000
        assert report["child_peak_rss_mib"] > 0
        assert not report["fidelity"]["hard_violations"]
        assert not report["fidelity"]["unmet_targets"]
        assert all(check["passed"] for check in report["fidelity"]["checks"])
        measured_path = output / "run" / "analysis" / "profile.yaml"
        from graph_metrics.io import read_profile

        profile = read_profile(measured_path)
        assert profile.observations.node_count == 1_000_000
        assert profile.observations.relationship_count == 5_000_000
        properties = profile.observations.nodes[0].properties
        assert set(properties) == {"synthetic_id", "category", "value"}
        rules = next(metric.value for metric in profile.metrics if metric.id == "semantics.rules")
        assert all(rule["violations"] == 0 and rule["status"] == "measured" for rule in rules)
    finally:
        connection = output / "run" / "connection.json"
        if connection.exists():
            details = json.loads(connection.read_text())
            name = details.get("container", "")
            if name.startswith("graph-db-metrics-generated-"):
                inspected = subprocess.run(
                    ["docker", "inspect", name, "--format", "{{json .Config.Labels}}"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if inspected.returncode == 0:
                    labels = json.loads(inspected.stdout)
                    assert labels.get("graph-db-metrics.run") == str(output / "run")
                    subprocess.run(
                        ["docker", "rm", "-f", name], check=True, capture_output=True, timeout=30
                    )
