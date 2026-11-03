import pytest

from graph_metrics.analyze import analyze_store
from graph_metrics.io import decode, encode, read_profile, read_table
from graph_metrics.models import Rule
from graph_metrics.store import GraphStore
from graph_metrics.validation import check_rules


def measure(tmp_path, monkeypatch, nodes, edges=(), **kwargs):
    monkeypatch.setenv("GRAPH_METRICS_PROJECT", str(tmp_path))
    store = GraphStore(tmp_path / "graph")
    store.add_nodes(nodes)
    store.add_edges(list(edges))
    profile = analyze_store(store, tmp_path / "profile", **kwargs)
    return store, profile, {m.id: m for m in profile.metrics}


def node(i, properties=None, labels=None):
    labels = ["Device"] if labels is None else labels
    return {"id": i, "cohort": "device", "labels": labels, "properties": properties or {}}


def test_directed_multigraph_incidence_and_projection(tmp_path, monkeypatch):
    edges = [
        {"id": i, "cohort": "link", "source": a, "target": b, "type": "LINK", "properties": {}}
        for i, (a, b) in enumerate([(0, 0), (0, 1), (0, 1), (1, 0)])
    ]
    store, p, metrics = measure(tmp_path, monkeypatch, [node(i) for i in range(3)], edges)
    try:
        degrees = metrics["connectivity.degrees"].value
        assert degrees["distribution"] == [
            {"in": 0, "out": 0, "count": 1},
            {"in": 2, "out": 1, "count": 1},
            {"in": 2, "out": 3, "count": 1},
        ]
        mult = metrics["connectivity.multiplicity"].value
        assert mult["self_loops"] == 1 and mult["parallel_excess"] == 1
        assert mult["pair_occupancy"] == pytest.approx(1 / 3)
        assert metrics["connectivity.components.strong"].value["sizes"] == [
            {"size": 1, "count": 1},
            {"size": 2, "count": 1},
        ]
        assert metrics["reachability.paths"].value["unreachable_fraction"] == pytest.approx(2 / 3)
        assert metrics["topology.clustering"].scope["multiplicity"] == "collapse"
        assert metrics["semantics.conceptual_diversity"].status == "ontology_required"
    finally:
        store.close()


@pytest.mark.parametrize(
    "patterns,expected_h,expected_tc",
    [
        ([(1, 0), (1, 0), (0, 1), (0, 1)], 1, 1),
        ([(0, 0), (0, 0), (1, 1), (1, 1)], 1, 1),
        ([(0, 0), (0, 1), (1, 0), (1, 1)], 2, 0),
    ],
)
def test_presence_counterexamples(tmp_path, monkeypatch, patterns, expected_h, expected_tc):
    nodes = [
        node(i, {k: 1 for k, enabled in zip(("A", "B"), pattern) if enabled})
        for i, pattern in enumerate(patterns)
    ]
    store, p, metrics = measure(tmp_path, monkeypatch, nodes)
    try:
        summary = metrics["schema.node.device.dependence"].value
        assert summary["joint_entropy_bits"] == expected_h
        assert summary["total_correlation_bits"] == expected_tc
        assert p.observations.nodes[0].properties["A"].presence == 0.5
        actual = read_table(p.observations.nodes[0].patterns, tmp_path / "profile")
        assert sum(x["count"] for x in actual) == 4
    finally:
        store.close()


def test_parquet_distributions_and_yaml_round_trip(tmp_path, monkeypatch):
    store, p, _ = measure(
        tmp_path, monkeypatch, [node(i, {"x": i}) for i in range(8)], table_threshold=2
    )
    try:
        reloaded = read_profile(tmp_path / "profile" / "profile.yaml")
        rows = read_table(
            reloaded.observations.nodes[0].properties["x"].distribution["table"],
            tmp_path / "profile",
        )
        assert {x["value"] for x in rows} == set(range(8))
        assert p.observations.node_count == 8
    finally:
        store.close()


def test_empty_graph_reports_undefined_quantities(tmp_path, monkeypatch):
    store, p, metrics = measure(tmp_path, monkeypatch, [])
    try:
        assert p.observations.node_count == 0
        assert metrics["reachability.paths"].value["unreachable_fraction"] is None
        assert metrics["topology.communities"].status == "not_applicable"
    finally:
        store.close()


def test_sampled_bfs_agrees_with_complete_graph(tmp_path, monkeypatch):
    edges = [
        {"id": i, "cohort": "link", "source": a, "target": b, "type": "LINK", "properties": {}}
        for i, (a, b) in enumerate(((a, b) for a in range(8) for b in range(8) if a != b))
    ]
    store, p, metrics = measure(
        tmp_path, monkeypatch, [node(i) for i in range(8)], edges, exact_threshold=2, sample_roots=4
    )
    try:
        paths = metrics["reachability.paths"]
        assert paths.status == "estimated" and paths.value["unreachable_fraction"] == 0
        assert metrics["reachability.neighborhoods"].value["1"] == [{"reachable": 8, "count": 4}]
        assert paths.uncertainty["reachable_fraction_ci95"] == [1, 1]
    finally:
        store.close()


def test_rules_validate_values_independently(tmp_path, monkeypatch):
    store, p, _ = measure(
        tmp_path,
        monkeypatch,
        [node(0, {"age": 20, "id": "a"}), node(1, {"age": -2, "id": "a"}), node(2, {})],
    )
    try:
        rules = [
            Rule(
                id="age", kind="range", parameters={"cohort": "device", "property": "age", "min": 0}
            ),
            Rule(id="id", kind="unique", parameters={"cohort": "device", "property": "id"}),
            Rule(id="req", kind="required", parameters={"cohort": "device", "property": "age"}),
        ]
        checks = check_rules(store, rules)
        assert [r["violations"] for r in checks] == [1, 1, 1]
        assert [r["eligible"] for r in checks] == [2, 2, 3]
    finally:
        store.close()


def test_native_properties_keep_types():
    from neo4j.spatial import WGS84Point
    from neo4j.time import Date, DateTime, Duration, Time

    values = [
        Date(2024, 2, 29),
        DateTime(2024, 1, 2, 3, 4, 5),
        Time(3, 4, 5),
        Duration(months=2, days=3),
        WGS84Point((1.5, 2.5)),
        b"\x00\xff",
    ]
    for value in values:
        recovered = decode(encode(value))
        assert type(recovered) is type(value)
        assert recovered == value


def test_yaml_12_domain_words(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_METRICS_PROJECT", str(tmp_path))
    path = tmp_path / "profile.yaml"
    path.write_text(
        "schema_version: '1'\ntargets:\n  nodes:\n  - id: signal\n    labels: [Signal]\n    count: 10\n    properties:\n      status:\n        distribution:\n          kind: categorical\n          values: [ON, OFF, YES, NO]\n"
    )
    values = read_profile(path).targets.nodes[0].properties["status"].distribution["values"]
    assert values == ["ON", "OFF", "YES", "NO"]
