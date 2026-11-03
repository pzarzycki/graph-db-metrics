import pytest
import yaml

from graph_metrics.analyze import analyze_store
from graph_metrics.io import read_profile, read_table, write_table
from graph_metrics.models import (
    NodeSpec,
    Profile,
    PropertySpec,
    RelationshipSpec,
    Rule,
    Semantics,
    Targets,
)
from graph_metrics.store import GraphStore
from graph_metrics.validation import attach_semantic_metrics, check_property_types, compare_profiles


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_METRICS_PROJECT", str(tmp_path))
    return tmp_path


def test_malformed_cohorts_cannot_collapse_silently():
    with pytest.raises(ValueError, match="distinct complete label sets"):
        Profile(
            targets=Targets(
                nodes=[
                    NodeSpec(id="a", labels=["Person"], count=1),
                    NodeSpec(id="b", labels=["Person"], count=1),
                ]
            )
        )
    with pytest.raises(ValueError, match="unknown node cohort"):
        Profile(
            targets=Targets(
                nodes=[NodeSpec(id="a", labels=["Person"], count=1)],
                relationships=[
                    RelationshipSpec(id="r", type="LINK", source="a", target="missing", count=1)
                ],
            )
        )


def test_profile_rejects_executable_yaml_and_tampered_companions(project):
    path = project / "unsafe.yaml"
    path.write_text("!!python/object/apply:os.system ['touch should-not-exist']")
    with pytest.raises(yaml.constructor.ConstructorError):
        read_profile(path)
    assert not (project / "should-not-exist").exists()
    reference = write_table([{"value": "A", "count": 2}], project / "values.parquet", project)
    (project / "values.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        read_table(reference, project)
    reference.path = "../../outside.parquet"
    with pytest.raises(ValueError, match="within the project"):
        read_table(reference, project)


def test_invalid_values_and_type_changing_provider_fail_before_loading(project):
    profile = Profile(
        targets=Targets(
            nodes=[
                NodeSpec(
                    id="a",
                    labels=["Person"],
                    count=1,
                    properties={"age": PropertySpec(type="integer")},
                )
            ]
        )
    )
    with GraphStore(project / "invalid") as store:
        store.add_nodes(
            [{"id": 0, "cohort": "a", "labels": ["Person"], "properties": {"age": "twenty"}}]
        )
        with pytest.raises(ValueError, match="declared type"):
            check_property_types(store, profile)
    with GraphStore(project / "overflow") as store:
        store.add_nodes(
            [{"id": 0, "cohort": "a", "labels": ["Person"], "properties": {"age": 2**63}}]
        )
        with pytest.raises(ValueError, match="out-of-range"):
            check_property_types(store, profile)


def test_declared_dag_depth_includes_isolates(project):
    with GraphStore(project / "dag") as store:
        store.add_nodes(
            [{"id": i, "cohort": "a", "labels": ["Topic"], "properties": {}} for i in range(4)]
        )
        store.add_edges(
            [
                {
                    "id": i,
                    "cohort": "r",
                    "source": a,
                    "target": b,
                    "type": "PARENT",
                    "properties": {},
                }
                for i, (a, b) in enumerate([(0, 1), (1, 2)])
            ]
        )
        profile = analyze_store(store, project / "profile")
        profile.semantics = Semantics(
            status="declared",
            rules=[Rule(id="hierarchy", kind="dag", parameters={"relationship": "r"})],
        )
        attach_semantic_metrics(profile, store)
        result = next(m for m in profile.metrics if m.id == "topology.hierarchy")
        assert result.status == "exact"
        assert result.value[0]["max_depth"] == 2
        assert result.value[0]["depths"] == [
            {"depth": 0, "count": 2},
            {"depth": 1, "count": 1},
            {"depth": 2, "count": 1},
        ]


def test_fidelity_detects_values_and_preserves_observed_profile(project):
    with GraphStore(project / "graph") as store:
        store.add_nodes(
            [
                {"id": i, "cohort": "a", "labels": ["Device"], "properties": {"kind": "B"}}
                for i in range(10)
            ]
        )
        actual = analyze_store(store, project / "actual")
        expected = Profile(
            targets=Targets(
                nodes=[
                    NodeSpec(
                        id="different-id",
                        labels=["Device"],
                        count=10,
                        properties={
                            "kind": PropertySpec(
                                distribution={"kind": "empirical", "values": ["A"], "weights": [10]}
                            )
                        },
                    )
                ]
            )
        )
        before = expected.model_dump()
        report = compare_profiles(
            expected, actual, expected_dir=project, actual_dir=project / "actual"
        )
        assert report["status"] == "partial"
        assert report["unmet_targets"][0]["value"] == 1
        assert expected.model_dump() == before
        assert all(c["passed"] for c in report["checks"] if c["metric"].startswith("population."))
