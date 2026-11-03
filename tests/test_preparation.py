import io
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from graph_metrics.io import profile_digest, read_profile, sha256
from graph_metrics.models import Metric, NodeSpec, Profile, PropertySpec, Rule, Semantics, Targets
from graph_metrics.preparation import (
    PreparationError,
    PreparationRequired,
    ProviderRunner,
    bounded_context,
    prepare,
    validate_provider,
)

PROVIDER = """import random
def provide(kind, cohort, rows, seed):
    rng = random.Random(seed)
    return [{"generated": rng.randint(1, 1000000)} for row in rows]
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_METRICS_PROJECT", str(tmp_path))
    return tmp_path


def profile():
    return Profile(
        source={"uri": "bolt://PRIVATE_SOURCE", "password": "source-secret"},
        targets=Targets(nodes=[NodeSpec(id="employee", labels=["Person", "Employee"], count=10)]),
        semantics=Semantics(
            status="declared",
            rules=[
                Rule(
                    id="age",
                    kind="range",
                    parameters={"cohort": "employee", "property": "age", "min": 18},
                )
            ],
        ),
        samples={"employee": [{"properties": {"name": "An example", "password": "sample-secret"}}]},
    )


def proposal(**updates):
    result = {
        "concepts": [
            {
                "name": "Employee",
                "labels": ["Employee"],
                "description": "An employed person",
                "properties": [],
            }
        ],
        "rules": [
            {
                "id": "salary",
                "kind": "range",
                "parameters_json": '{"cohort":"employee","property":"salary","min":0}',
                "description": "Proposed nonnegative salary",
                "hard": True,
            }
        ],
        "assumptions": ["Employment is interpreted from the label; this is inferred."],
        "questions": [],
        "recipe_json": '{"vocabularies":{"jobs":["engineer","nurse"]}}',
        "provider_code": PROVIDER,
    }
    result.update(updates)
    return result


def fake_codex(monkeypatch, response, calls):
    monkeypatch.setattr("graph_metrics.preparation.shutil.which", lambda _: "/bin/codex")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        destination = Path(command[command.index("-o") + 1])
        destination.write_text(json.dumps(response))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("graph_metrics.preparation.subprocess.run", run)


def test_preparation_preserves_observations_and_declared_rules(project, monkeypatch):
    calls = []
    fake_codex(monkeypatch, proposal(), calls)
    monkeypatch.setenv("NEO4J_PASSWORD", "database-secret")
    original = profile()
    before = original.model_dump(mode="json")
    prepared = prepare(original, project, project / "prep")
    assert original.model_dump(mode="json") == before
    assert prepared.observations == original.observations
    assert prepared.semantics.rules[0] == original.semantics.rules[0]
    assert prepared.semantics.rules[1].provenance == "inferred"
    assert prepared.semantics.concepts[0]["provenance"] == "inferred"
    assert prepared.semantics.status == "mixed"
    assert prepared.preparation.profile_digest == profile_digest(prepared)
    command, kwargs = calls[0]
    assert "NEO4J_PASSWORD" not in kwargs["env"]
    assert "PRIVATE_SOURCE" not in kwargs["input"]
    assert "source-secret" not in kwargs["input"]
    assert "sample-secret" not in kwargs["input"]
    assert "read-only" in command
    assert command[command.index("--disable") + 1] == "shell_tool"
    # The persisted preparation has rebased references and can be loaded independently.
    saved = read_profile(project / "prep" / "prepared-profile.yaml")
    assert saved.preparation.provider_path == "provider.py"
    assert saved.preparation.profile_digest == profile_digest(saved)


def test_cache_reuses_only_compatible_profile_and_provider(project, monkeypatch):
    calls = []
    fake_codex(monkeypatch, proposal(), calls)
    prepared = prepare(profile(), project, project / "prep")
    assert prepare(prepared, project, project / "unused") == prepared
    assert len(calls) == 1
    changed = prepared.model_copy(deep=True)
    changed.targets.nodes[0].count = 20
    prepare(changed, project, project / "changed")
    assert len(calls) == 2
    (project / "prep" / "provider.py").write_text(PROVIDER + "\n# modified\n")
    prepare(prepared, project, project / "modified")
    assert len(calls) == 3


def test_questions_are_saved_and_block_generation(project, monkeypatch):
    fake_codex(monkeypatch, proposal(questions=["Does salary mean annual or monthly pay?"]), [])
    with pytest.raises(PreparationRequired, match="annual or monthly") as caught:
        prepare(profile(), project, project / "prep")
    assert caught.value.path.is_file()
    saved = read_profile(caught.value.path)
    assert saved.semantics.questions == ["Does salary mean annual or monthly pay?"]
    assert saved.observations == profile().observations


def test_declared_rule_collision_becomes_a_question(project, monkeypatch):
    response = proposal()
    response["rules"][0]["id"] = "age"
    fake_codex(monkeypatch, response, [])
    with pytest.raises(PreparationRequired, match="collide") as caught:
        prepare(profile(), project, project / "prep")
    assert caught.value.profile.semantics.rules == profile().semantics.rules


def test_identical_echo_of_declared_rule_is_not_an_ambiguity(project, monkeypatch):
    response = proposal(
        rules=[
            {
                "id": "age",
                "kind": "range",
                "parameters_json": '{"cohort":"employee","property":"age","min":18}',
                "description": "Different wording of the same existing rule",
                "hard": True,
            }
        ]
    )
    fake_codex(monkeypatch, response, [])
    prepared = prepare(profile(), project, project / "prep")
    assert not prepared.semantics.questions
    assert prepared.semantics.rules == profile().semantics.rules


def test_repeated_generation_reuses_frozen_preparation_without_rewriting_input(
    project, monkeypatch
):
    calls = []
    fake_codex(monkeypatch, proposal(), calls)
    source = profile()
    original = source.model_dump(mode="json")
    first = prepare(source, project, project / "run-1" / "prep")
    # Deleting the first run's code must not discard the frozen preparation cache.
    (project / first.preparation.provider_path).unlink()
    second = prepare(source, project, project / "run-2" / "prep")
    assert len(calls) == 1
    assert source.model_dump(mode="json") == original
    assert second.semantics == first.semantics
    assert second.preparation.provider_path.startswith(".cache/")
    assert (project / second.preparation.provider_path).is_file()
    saved = read_profile(project / "run-2" / "prep" / "prepared-profile.yaml")
    assert saved.preparation.profile_digest == profile_digest(saved)


@pytest.mark.parametrize(
    "response",
    [
        proposal(recipe_json="[]"),
        proposal(
            rules=[
                {
                    "id": "oops",
                    "kind": "range",
                    "parameters_json": "[]",
                    "description": "invalid parameters",
                    "hard": True,
                }
            ]
        ),
        {**proposal(), "unrecognized": "field"},
    ],
)
def test_malformed_structured_response_is_rejected(project, monkeypatch, response):
    fake_codex(monkeypatch, response, [])
    with pytest.raises(PreparationError):
        prepare(profile(), project, project / "prep")


def test_context_is_bounded_and_source_credentials_are_absent():
    original = profile()
    original.samples = {
        str(i): [{"properties": {str(k): "x" * 10000 for k in range(200)}}] for i in range(100)
    }
    context = bounded_context(original)
    assert len(json.dumps(context, ensure_ascii=False)) <= 48000
    assert "source-secret" not in json.dumps(context)
    assert "PRIVATE_SOURCE" not in json.dumps(context)
    assert context["omitted"]


def test_distribution_excerpts_are_marked_and_secret_metrics_redacted():
    original = profile()
    original.targets.nodes[0].properties["category"] = PropertySpec(
        distribution={"kind": "empirical", "values": list(range(30)), "weights": [1] * 30}
    )
    original.metrics = [
        Metric(
            id="sensitive",
            definition="Example property",
            scope={"property": "api_token"},
            value={"distribution": ["actual-secret-value"]},
            status="exact",
        )
    ]
    context = bounded_context(original)
    excerpt = context["node_cohorts"][0]["properties"]["category"]["distribution"]["values"]
    assert (
        excerpt["truncated"] is True
        and excerpt["total_items"] == 30
        and len(excerpt["items"]) == 12
    )
    assert "actual-secret-value" not in json.dumps(context)


def test_disabling_codex_does_not_ignore_pending_semantic_questions(project):
    original = profile()
    original.semantics.questions = ["Which unit applies?"]
    with pytest.raises(PreparationRequired, match="resolve"):
        prepare(original, project, project / "prepared", mode="off")


def test_provider_rejects_host_access_and_checksum_changes(project):
    with pytest.raises(PreparationError, match="standard library"):
        validate_provider("import subprocess\ndef provide(kind, cohort, rows, seed): return [{}]")
    with pytest.raises(PreparationError, match="unsupported call"):
        validate_provider("def provide(kind, cohort, rows, seed): return [eval('1')]")
    path = project / "provider.py"
    path.write_text(PROVIDER)
    with pytest.raises(PreparationError, match="checksum"):
        ProviderRunner(path, "0" * 64)
    with pytest.raises(ValueError, match="within the project"):
        ProviderRunner(project.parent / "provider.py", "0" * 64)


def test_worker_timeout_and_bad_output_are_actionable(project, monkeypatch):
    path = project / "provider.py"
    path.write_text(PROVIDER)
    runner = ProviderRunner(path, sha256(path), timeout=1)
    runner.process = SimpleNamespace(stdin=io.BytesIO())
    monkeypatch.setattr(runner, "_readline", lambda: {"overrides": []})
    with pytest.raises(PreparationError, match="one property dictionary"):
        runner("node", "employee", [{"id": 1}], 42)
    runner.process = None
    read_fd, write_fd = os.pipe()
    try:
        runner.process = SimpleNamespace(stdout=os.fdopen(read_fd, "rb"))
        monkeypatch.undo()
        with pytest.raises(PreparationError, match="execution limit"):
            runner._readline()
        runner.process.stdout.close()
    finally:
        runner.process = None
        os.close(write_fd)


@pytest.mark.parametrize(
    "bad_value",
    [
        {"not": "a property scalar"},
        [1, "mixed"],
        [[1]],
        None,
        2**80,
        {"__neo4j_type__": "unknown", "value": "x"},
    ],
)
def test_provider_invalid_native_property_is_rejected_before_loading(
    project, monkeypatch, bad_value
):
    path = project / "provider.py"
    path.write_text(PROVIDER)
    runner = ProviderRunner(path, sha256(path))
    runner.process = SimpleNamespace(stdin=io.BytesIO())
    monkeypatch.setattr(runner, "_readline", lambda: {"overrides": [{"invalid": bad_value}]})
    with pytest.raises(PreparationError):
        runner("node", "employee", [{"id": 1}], 42)
    runner.process = None


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("GRAPH_METRICS_RUN_INTEGRATION") != "1", reason="Docker opt-in")
def test_real_worker_is_isolated_and_batch_independent(project):
    path = project / "provider.py"
    path.write_text(PROVIDER)
    rows = [{"id": i, "properties": {}} for i in range(4)]
    with ProviderRunner(path, sha256(path), timeout=60) as runner:
        together = runner("node", "employee", rows, 42)
        separate = [runner("node", "employee", [row], 42)[0] for row in rows]
        assert together == separate
        inspected = json.loads(
            subprocess.check_output(["docker", "inspect", runner._container_name])
        )[0]
        assert inspected["HostConfig"]["NetworkMode"] == "none"
        assert inspected["HostConfig"]["ReadonlyRootfs"] is True
        assert all("NEO4J" not in item for item in inspected["Config"]["Env"])
        assert len(inspected["Mounts"]) == 2


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("GRAPH_METRICS_RUN_INTEGRATION") != "1", reason="Docker opt-in")
def test_real_worker_failure_and_hang_cannot_leave_running_container(project):
    path = project / "provider.py"
    path.write_text("def provide(kind, cohort, rows, seed):\n    while True: pass\n")
    with pytest.raises(PreparationError, match="execution limit"):
        with ProviderRunner(path, sha256(path), timeout=2) as runner:
            name = runner._container_name
            runner("node", "employee", [{"id": 1}], 42)
    assert subprocess.run(["docker", "inspect", name], capture_output=True).returncode != 0
