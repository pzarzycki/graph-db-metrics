"""Real Community Docker round trips; enabled explicitly, never use the development database."""

import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import timezone
from pathlib import Path

import pytest
from neo4j.spatial import CartesianPoint, WGS84Point
from neo4j.time import Date, DateTime, Duration, Time

from graph_metrics.io import dumps, read_profile
from graph_metrics.local import load_local, local_password
from graph_metrics.neo4j_io import connect, load_online, scan
from graph_metrics.store import GraphStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("GRAPH_METRICS_RUN_INTEGRATION") != "1",
        reason="set GRAPH_METRICS_RUN_INTEGRATION=1 to run real Docker imports",
    ),
]


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_METRICS_PROJECT", str(tmp_path))
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("Docker is unavailable")
    yield tmp_path
    # Remove only containers carrying the exact run directory from this test's artifacts.
    # Keep database files and logs for diagnosis; no shared instance is touched.
    for connection_file in tmp_path.rglob("connection.json"):
        connection = json.loads(connection_file.read_text())
        name = connection.get("container", "")
        if not name.startswith("graph-db-metrics-generated-"):
            continue
        result = subprocess.run(
            ["docker", "inspect", name, "--format", "{{json .Config.Labels}}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode:
            continue
        labels = json.loads(result.stdout)
        if labels.get("graph-db-metrics.run") != str(connection_file.parent):
            raise AssertionError("test refused to clean up a container owned by another run")
        subprocess.run(["docker", "rm", "-f", name], check=True, capture_output=True, timeout=30)


def canonical(store):
    nodes = [node for batch in store.iter_nodes() for node in batch]
    identities = {node["id"]: dumps([sorted(node["labels"]), node["properties"]]) for node in nodes}
    return (
        Counter(identities.values()),
        Counter(
            (
                identities[edge["source"]],
                identities[edge["target"]],
                edge["type"],
                dumps(edge["properties"]),
            )
            for batch in store.iter_edges()
            for edge in batch
        ),
    )


def live_snapshot(uri, password):
    with connect(uri, password=password) as driver, driver.session(database="neo4j") as session:
        nodes = [
            dumps([row["id"], sorted(row["labels"]), row["properties"]])
            for row in session.run(
                "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties"
            )
        ]
        edges = [
            dumps(dict(row))
            for row in session.run(
                "MATCH (a)-[r]->(b) RETURN elementId(r) AS id, elementId(a) AS source, "
                "elementId(b) AS target, type(r) AS type, properties(r) AS properties"
            )
        ]
        constraints = [dumps(dict(row)) for row in session.run("SHOW CONSTRAINTS")]
    return sorted(nodes), sorted(edges), sorted(constraints)


def native_graph(path, online):
    store = GraphStore(path)
    props = {
        "identity": "first",
        "empty_text": "",
        "text": 'Comma, newline\nquotes " and 世界',
        "flag": True,
        "number": 7,
        "decimal": 3.125,
        "date": Date(1990, 2, 3),
        "datetime": DateTime(2026, 1, 2, 3, 4, 5, nanosecond=123456789, tzinfo=timezone.utc),
        "local_datetime": DateTime(2026, 1, 2, 3, 4, 5, nanosecond=123456789),
        "time": Time(3, 4, 5, nanosecond=123456789, tzinfo=timezone.utc),
        "local_time": Time(3, 4, 5, nanosecond=123456789),
        "duration": Duration(months=2, days=3, seconds=4, nanoseconds=5),
        "location": WGS84Point((-123.1, 49.2)),
        "position": CartesianPoint((1.2, 3.4)),
        "scores": [1, 2, 3],
        "tags": ["quoted,value", "世界"],
        "flags": [True, False],
    }
    if online:
        props.update(byte_array=b"\x00\x01\xfe\xff", empty_list=[])
    store.add_nodes(
        [
            {"id": 0, "cohort": "person", "labels": ["Person", "Employee"], "properties": props},
            {"id": 1, "cohort": "unlabeled", "labels": [], "properties": {"identity": "second"}},
            {
                "id": 2,
                "cohort": "isolated",
                "labels": ["Isolate"],
                "properties": {"identity": "third"},
            },
        ]
    )
    store.add_edges(
        [
            {
                "id": 0,
                "cohort": "link",
                "source": 0,
                "target": 1,
                "type": "LINK",
                "properties": {"since": Date(2025, 1, 1), "note": "primary"},
            },
            {
                "id": 1,
                "cohort": "link",
                "source": 0,
                "target": 1,
                "type": "LINK",
                "properties": {"since": Date(2025, 1, 1), "note": "parallel"},
            },
            {"id": 2, "cohort": "self", "source": 1, "target": 1, "type": "SELF", "properties": {}},
        ]
    )
    return store


@pytest.mark.parametrize("offline", [True, False], ids=["native-offline", "driver-online"])
def test_native_types_multigraph_and_read_only_scan(project, offline):
    store = native_graph(project / "source", online=not offline)
    loaded = scanned = None
    try:
        loaded = load_local(store, project / "target", offline=offline)
        password = local_password(project / "target")
        before = live_snapshot(loaded["uri"], password)
        scanned, source = scan(loaded["uri"], project / "scan", password=password, batch_size=2)
        assert canonical(store) == canonical(scanned)
        assert source["consistency"] == "live_non_atomic"
        assert source["observed_count_change"] is False
        assert live_snapshot(loaded["uri"], password) == before
        # A second load must fail before adding data or temporary constraints.
        with pytest.raises(ValueError, match="populated"):
            load_online(store, loaded["uri"], password=password, batch_size=2)
        assert live_snapshot(loaded["uri"], password) == before
        assert not any("_GraphMetrics_" in value for value in before[0])
    finally:
        store.close()
        if scanned:
            scanned.close()


@pytest.mark.parametrize("domain,loader", [("aviation", "offline"), ("employment", "online")])
def test_domain_cli_round_trip_and_fidelity(project, domain, loader):
    repository = Path(__file__).resolve().parents[1]
    profile_path = project / f"{domain}.yaml"
    shutil.copy2(repository / "examples" / f"{domain}.yaml", profile_path)
    requested = read_profile(profile_path)
    run_dir = project / "run"
    executable = Path(sys.executable).parent / "graph-metrics"
    completed = subprocess.run(
        [
            str(executable),
            "generate",
            "--profile",
            str(profile_path),
            "--target",
            "local",
            "--loader",
            loader,
            "--prepare",
            "off",
            "--output",
            str(run_dir),
            "--sample-roots",
            "8",
            "--exact-threshold",
            "20",
        ],
        capture_output=True,
        text=True,
        timeout=360,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["loaded"] is True
    assert summary["nodes"] == sum(n.count for n in requested.targets.nodes)
    assert summary["relationships"] == sum(r.count for r in requested.targets.relationships)
    fidelity = json.loads((run_dir / "fidelity.json").read_text())
    assert not fidelity["hard_violations"]
    assert all(
        check["passed"] for check in fidelity["checks"] if check["metric"].endswith(".count")
    )
    actual = read_profile(run_dir / "analysis" / "profile.yaml")
    rules = next(metric.value for metric in actual.metrics if metric.id == "semantics.rules")
    assert all(rule["status"] == "measured" and rule["violations"] == 0 for rule in rules)
    # Native import and online driver loading preserve every generated property and edge.
    with GraphStore(run_dir / "staging", read_only=True) as staged:
        with GraphStore(run_dir / "verification-store", read_only=True) as scanned:
            assert canonical(staged) == canonical(scanned)


@pytest.mark.parametrize("target_mode", ["local", "uri"])
def test_analyze_cli_then_generate_from_existing_database(project, monkeypatch, target_mode):
    executable = Path(sys.executable).parent / "graph-metrics"
    with native_graph(project / "source-staging", online=False) as original:
        source = load_local(original, project / "source-target")
        source_password = local_password(project / "source-target")
        monkeypatch.setenv("TEST_SOURCE_PASSWORD", source_password)
        before = live_snapshot(source["uri"], source_password)
        profile_path = project / "extracted" / "profile.yaml"
        analyzed = subprocess.run(
            [
                str(executable),
                "analyze",
                "--uri",
                source["uri"],
                "--password-env",
                "TEST_SOURCE_PASSWORD",
                "--output",
                str(profile_path),
                "--semantics",
                "off",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert analyzed.returncode == 0, analyzed.stderr
        measured = read_profile(profile_path)
        assert (
            measured.observations.node_count == 3 and measured.observations.relationship_count == 3
        )
        assert measured.source["system"] == "neo4j"
        assert live_snapshot(source["uri"], source_password) == before
        run_dir = project / "regenerated"
        command = [
            str(executable),
            "generate",
            "--profile",
            str(profile_path),
            "--output",
            str(run_dir),
            "--prepare",
            "off",
        ]
        if target_mode == "uri":
            with GraphStore(project / "empty-target-staging") as empty:
                remote = load_local(empty, project / "remote-target", offline=False)
            monkeypatch.setenv("TEST_TARGET_PASSWORD", local_password(project / "remote-target"))
            command += ["--target-uri", remote["uri"], "--password-env", "TEST_TARGET_PASSWORD"]
        generated = subprocess.run(command, capture_output=True, text=True, timeout=240)
        assert generated.returncode == 0, generated.stderr
        summary = json.loads(generated.stdout)
        assert summary["loaded"] is True
        assert summary["fidelity"] == "passed"
        with GraphStore(run_dir / "verification-store", read_only=True) as actual:
            assert canonical(actual) == canonical(original)
        assert live_snapshot(source["uri"], source_password) == before
        frozen = read_profile(run_dir / "requested-profile.yaml")
        assert frozen.observations == measured.observations
