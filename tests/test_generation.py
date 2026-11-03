from collections import Counter
from pathlib import Path

import pytest

from graph_metrics.generation import generate
from graph_metrics.io import dumps, read_profile
from graph_metrics.models import (
    Metric,
    NodeSpec,
    Observations,
    Profile,
    PropertySpec,
    RelationshipSpec,
    Rule,
    Targets,
)
from graph_metrics.validation import check_rules


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_METRICS_PROJECT", str(tmp_path))
    return tmp_path


def records(store):
    return (
        [row for batch in store.iter_nodes() for row in batch],
        [row for batch in store.iter_edges() for row in batch],
    )


def test_joint_presence_patterns_and_repeatability_across_batches(project):
    specs = {
        key: PropertySpec(type="integer", distribution={"kind": "constant", "value": 1})
        for key in ("A", "B")
    }
    profile = Profile(
        targets=Targets(
            nodes=[
                NodeSpec(
                    id="n",
                    labels=["Person", "Employee"],
                    count=100,
                    properties=specs,
                    patterns=[{"keys": ["A"], "count": 50}, {"keys": ["B"], "count": 50}],
                )
            ]
        )
    )
    first = generate(profile, project / "first", base_dir=project, seed=73, batch_size=7)
    second = generate(profile, project / "second", base_dir=project, seed=73, batch_size=31)
    try:
        a, b = records(first), records(second)
        assert dumps(a) == dumps(b)
        assert Counter(tuple(sorted(node["properties"])) for node in a[0]) == {
            ("A",): 50,
            ("B",): 50,
        }
    finally:
        first.close()
        second.close()


def test_joint_typed_role_vectors_are_retained(project):
    roles = [
        {"roles": {"a": {"out": 2, "in": 0}, "b": {"out": 0, "in": 1}}, "count": 4},
        {"roles": {"a": {"out": 0, "in": 2}, "b": {"out": 1, "in": 0}}, "count": 4},
    ]
    profile = Profile(
        targets=Targets(
            nodes=[NodeSpec(id="n", labels=["N"], count=8, roles=roles)],
            relationships=[
                RelationshipSpec(id="a", type="A", source="n", target="n", count=8),
                RelationshipSpec(id="b", type="B", source="n", target="n", count=4),
            ],
            allow_self_loops=True,
            allow_parallel_edges=True,
        )
    )
    store = generate(profile, project / "roles", base_dir=project)
    try:
        nodes, edges = records(store)
        vectors = {node["id"]: [0, 0, 0, 0] for node in nodes}
        for edge in edges:
            offset = 0 if edge["cohort"] == "a" else 2
            vectors[edge["source"]][offset] += 1
            vectors[edge["target"]][offset + 1] += 1
        assert Counter(tuple(vector) for vector in vectors.values()) == {
            (2, 0, 0, 1): 4,
            (0, 2, 1, 0): 4,
        }
    finally:
        store.close()


def test_degree_sequences_and_simple_endpoint_restrictions(project):
    profile = Profile(
        targets=Targets(
            nodes=[NodeSpec(id="n", labels=["N"], count=12)],
            relationships=[
                RelationshipSpec(
                    id="r",
                    type="R",
                    source="n",
                    target="n",
                    count=24,
                    out_degrees=[{"degree": 2, "count": 12}],
                    in_degrees=[{"degree": 2, "count": 12}],
                )
            ],
            allow_self_loops=False,
            allow_parallel_edges=False,
        )
    )
    store = generate(profile, project / "simple", base_dir=project)
    try:
        _, edges = records(store)
        assert all(edge["source"] != edge["target"] for edge in edges)
        assert len({(edge["source"], edge["target"]) for edge in edges}) == 24
        assert set(Counter(edge["source"] for edge in edges).values()) == {2}
        assert set(Counter(edge["target"] for edge in edges).values()) == {2}
    finally:
        store.close()


def test_hard_cardinality_and_capacity_conflicts_fail_before_staging(project):
    profile = Profile(
        targets=Targets(
            nodes=[NodeSpec(id="n", labels=["N"], count=2)],
            relationships=[RelationshipSpec(id="r", type="R", source="n", target="n", count=3)],
            rules=[
                Rule(
                    id="one",
                    kind="cardinality",
                    parameters={"relationship": "r", "min": 1, "max": 1},
                )
            ],
        )
    )
    with pytest.raises(ValueError, match="infeasible.*cardinalities"):
        generate(profile, project / "invalid", base_dir=project)
    assert not (project / "invalid" / "graph.duckdb").exists()
    profile.targets.rules = []
    with pytest.raises(ValueError, match="endpoint capacity"):
        generate(profile, project / "invalid", base_dir=project)


def test_scaling_preserves_exact_total_population(project):
    profile = Profile(
        targets=Targets(
            nodes=[
                NodeSpec(id="a", labels=["A"], count=3),
                NodeSpec(id="b", labels=["B"], count=1),
            ],
            scale=1.5,
        )
    )
    store = generate(profile, project / "scaled", base_dir=project, scale=2)
    try:
        assert store.counts() == (12, 0)
        assert store.db.execute(
            "SELECT cohort,count(*) FROM nodes GROUP BY cohort ORDER BY cohort"
        ).fetchall() == [("a", 9), ("b", 3)]
    finally:
        store.close()


def test_small_populations_preserve_categorical_counts_and_rare_values(project):
    profile = Profile(
        targets=Targets(
            nodes=[
                NodeSpec(
                    id="n",
                    labels=["N"],
                    count=24,
                    properties={
                        "city": PropertySpec(
                            distribution={
                                "kind": "categorical",
                                "values": ["A", "B", "C", "D", "E", "F", "G", "H"],
                            }
                        )
                    },
                )
            ]
        )
    )
    store = generate(profile, project / "categories", base_dir=project, batch_size=5)
    try:
        nodes, _ = records(store)
        assert Counter(node["properties"]["city"] for node in nodes) == {
            city: 3 for city in "ABCDEFGH"
        }
    finally:
        store.close()
    profile.targets.nodes[0].count = 100
    profile.targets.nodes[0].properties["city"].distribution = {
        "kind": "empirical",
        "values": ["common", "rare_a", "rare_b"],
        "weights": [98, 1, 1],
    }
    store = generate(profile, project / "rare", base_dir=project)
    try:
        nodes, _ = records(store)
        assert Counter(node["properties"]["city"] for node in nodes) == {
            "common": 98,
            "rare_a": 1,
            "rare_b": 1,
        }
    finally:
        store.close()


def test_observed_multigraph_self_loop_policy_survives_extraction(project):
    profile = Profile(
        observations=Observations(
            node_count=1,
            relationship_count=1,
            nodes=[NodeSpec(id="n", labels=["N"], count=1)],
            relationships=[RelationshipSpec(id="r", type="R", source="n", target="n", count=1)],
        ),
        metrics=[
            Metric(
                id="connectivity.multiplicity",
                definition="observed endpoint multiplicity",
                value={"self_loops": 1, "parallel_excess": 0},
                status="exact",
            )
        ],
    )
    store = generate(profile, project / "loop", base_dir=project)
    try:
        _, edges = records(store)
        assert len(edges) == 1 and edges[0]["source"] == edges[0]["target"] == 0
    finally:
        store.close()


def test_observed_dependency_is_used_and_identifier_values_are_replaced(project):
    node = NodeSpec(
        id="n",
        labels=["N"],
        count=500,
        properties={
            "group": PropertySpec(distribution={"kind": "categorical", "values": ["A", "B"]}),
            "salary": PropertySpec(
                type="integer", distribution={"kind": "categorical", "values": [10, 20]}
            ),
            "user_id": PropertySpec(
                distribution={
                    "kind": "categorical",
                    "values": ["original-secret-1", "original-secret-2"],
                }
            ),
        },
    )
    profile = Profile(
        observations=Observations(node_count=500, nodes=[node]),
        metrics=[
            Metric(
                id="dependencies.n.group.salary",
                definition="measured contingency",
                status="estimated",
                value={
                    "mutual_information_bits": 1,
                    "contingency": [
                        {"a": "A", "b": 10, "count": 250},
                        {"a": "B", "b": 20, "count": 250},
                    ],
                },
            ),
            Metric(
                id="properties.node.n.user_id",
                definition="identifier hint",
                status="exact",
                value={"possible_identifier": True},
            ),
        ],
    )
    before = profile.model_dump(mode="json")
    store = generate(profile, project / "dependent", base_dir=project)
    try:
        nodes, _ = records(store)
        assert all(
            n["properties"]["salary"] == (10 if n["properties"]["group"] == "A" else 20)
            for n in nodes
        )
        ids = [n["properties"]["user_id"] for n in nodes]
        assert len(set(ids)) == 500
        assert all(value.startswith("synthetic-") for value in ids)
        assert profile.model_dump(mode="json") == before
    finally:
        store.close()


def test_external_provider_gets_linked_context_and_invalid_output_stops(project):
    profile = Profile(
        targets=Targets(
            nodes=[
                NodeSpec(
                    id="n",
                    labels=["N"],
                    count=10,
                    properties={
                        "value": PropertySpec(type="integer", distribution={"kind": "id"}),
                    },
                )
            ],
            relationships=[RelationshipSpec(id="r", type="R", source="n", target="n", count=10)],
            allow_parallel_edges=True,
        )
    )

    def provider(kind, cohort, rows, seed):
        if kind == "relationship":
            return [
                {"endpoint_sum": r["source_properties"]["value"] + r["target_properties"]["value"]}
                for r in rows
            ]
        return [{} for _ in rows]

    store = generate(profile, project / "provider", base_dir=project, provider=provider)
    try:
        _, edges = records(store)
        assert all(e["properties"]["endpoint_sum"] == e["source"] + e["target"] for e in edges)
    finally:
        store.close()
    with pytest.raises(ValueError, match="one property mapping"):
        generate(profile, project / "broken-provider", base_dir=project, provider=lambda *args: [])


@pytest.mark.parametrize("example", ["aviation", "employment"])
def test_domain_examples_are_diverse_and_pass_semantic_rules(tmp_path, monkeypatch, example):
    repository = Path(__file__).resolve().parents[1]
    profile = read_profile(repository / "examples" / f"{example}.yaml")
    monkeypatch.setenv("GRAPH_METRICS_PROJECT", str(tmp_path))
    store = generate(profile, tmp_path / example, base_dir=repository / "examples")
    try:
        checks = check_rules(store, profile.semantics.rules)
        assert checks and all(r["status"] == "measured" and r["violations"] == 0 for r in checks)
        nodes, edges = records(store)
        if example == "aviation":
            flights = [n["properties"] for n in nodes if n["cohort"] == "flight"]
            assert all(p["origin_code"] != p["destination_code"] for p in flights)
            assert all(p["arrival"] > p["departure"] for p in flights)
            assert all(0 <= p["passengers"] <= p["capacity"] for p in flights)
            assert len({p["origin_code"] for p in flights}) >= 15
        else:
            employees = [n["properties"] for n in nodes if n["cohort"] == "employee"]
            assert len({p["occupation"] for p in employees}) == 4
            assert len({p["name"] for p in employees}) > 950
            assert all(p["hired_year"] >= p["birth_year"] + 18 for p in employees)
    finally:
        store.close()


def test_extracted_profile_recipes_preserve_endpoint_role_coupling(project):
    from graph_metrics.analyze import analyze_store

    repository = Path(__file__).resolve().parents[1]
    # Read the authored fixture before setting the isolated artifact root.
    import os

    saved = os.environ.pop("GRAPH_METRICS_PROJECT", None)
    try:
        authored = read_profile(repository / "examples" / "employment.yaml")
    finally:
        if saved:
            os.environ["GRAPH_METRICS_PROJECT"] = saved
    source = generate(authored, project / "source", base_dir=repository / "examples")
    try:
        measured = analyze_store(source, project / "measured", exact_threshold=20, sample_roots=8)
    finally:
        source.close()
    measured.semantics = authored.semantics.model_copy(deep=True)
    measured.preparation.recipe = {
        "property_overrides": [
            {
                "cohort": "employee",
                "property": "hired_year",
                "distribution": {
                    "kind": "derived",
                    "on": "birth_year",
                    "random_min": 18,
                    "random_max": 40,
                    "max": 2026,
                },
            },
            {
                "entity": "relationship",
                "cohort": "works_at",
                "property": "since_year",
                "distribution": {"kind": "copy", "on": "source_properties.hired_year"},
            },
        ]
    }
    before = measured.observations.model_dump()
    target = generate(measured, project / "regenerated", base_dir=project / "measured")
    try:
        assert measured.observations.model_dump() == before
        assert all(
            r["violations"] == 0 and r["status"] == "measured"
            for r in check_rules(target, measured.semantics.rules)
        )
        nodes, edges = records(target)
        props = {n["id"]: n["properties"] for n in nodes}
        assert all(e["properties"]["since_year"] == props[e["source"]]["hired_year"] for e in edges)
        assert target.counts() == (1040, 1000)
    finally:
        target.close()
