"""Two commands: analyze a source, or generate and verify a separate target."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import sys
import time
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from neo4j.exceptions import DriverError, Neo4jError

from .analyze import analyze_store
from .io import dumps, profile_digest, project_root, read_profile, safe_path, sha256, write_profile
from .local import load_local, local_password
from .neo4j_io import connect, load_online, scan
from .preparation import (
    PreparationError,
    PreparationRequired,
    ProviderRunner,
    _save_prepared,
    prepare,
)
from .store import GraphStore
from .validation import attach_semantic_metrics, check_property_types, check_rules, compare_profiles


def _env():
    """Read local values without executing shell syntax or overwriting the environment."""
    path = project_root() / ".env"
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key in {"NEO4J_PASSWORD", "NEO4J_USERNAME", "NEO4J_URI", "NEO4J_DATABASE"}:
                os.environ.setdefault(key, value.strip().strip("\"'"))


def _report(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _log(message):
    print(message, file=sys.stderr, flush=True)


def _write_semantic_profile(profile, store, path):
    checks = attach_semantic_metrics(profile, store)
    # An inferred rule contradicted by the observed source is a question, not a new fact.
    for result in checks:
        if result["provenance"] == "inferred" and (
            result["violations"] or result["status"] != "measured"
        ):
            question = f"Inferred rule {result['id']!r} conflicts with measured evidence or cannot be checked: {result.get('reason', str(result['violations']) + ' violations')}"
            if question not in profile.semantics.questions:
                profile.semantics.questions.append(question)
    if profile.preparation.prepared_at:
        profile.preparation.profile_digest = profile_digest(profile)
    write_profile(profile, path)
    if profile.semantics.questions:
        raise PreparationRequired(
            "semantic questions need resolution in " + str(path), profile, path
        )


def _remap_semantics(expected, actual):
    nodes, rels = expected.generation_specs()
    actual_nodes = {tuple(sorted(n.labels)): n.id for n in actual.observations.nodes}
    node_map = {n.id: actual_nodes.get(tuple(sorted(n.labels)), n.id) for n in nodes}
    actual_rels = {(r.source, r.type, r.target): r.id for r in actual.observations.relationships}
    edge_map = {
        r.id: actual_rels.get((node_map[r.source], r.type, node_map[r.target]), r.id) for r in rels
    }
    actual.semantics = expected.semantics.model_copy(deep=True)
    actual.semantics.rules += [r.model_copy(deep=True) for r in expected.targets.rules]
    for rule in actual.semantics.rules:
        p = rule.parameters
        if "cohort" in p:
            p["cohort"] = (edge_map if p.get("entity") == "relationship" else node_map).get(
                p["cohort"], p["cohort"]
            )
        if "relationship" in p:
            p["relationship"] = edge_map.get(p["relationship"], p["relationship"])
        if "signatures" in p:
            for signature in p["signatures"]:
                signature["source"] = node_map.get(signature["source"], signature["source"])
                signature["target"] = node_map.get(signature["target"], signature["target"])
    # Supply eligible endpoint cohorts even for zero-edge signatures.
    for rule in actual.semantics.rules:
        if rule.kind == "cardinality" and "cohort" not in rule.parameters:
            relationship = next(
                (r for r in rels if edge_map[r.id] == rule.parameters["relationship"]), None
            )
            if relationship:
                rule.parameters["cohort"] = node_map[
                    relationship.source
                    if rule.parameters.get("direction", "out") == "out"
                    else relationship.target
                ]
    if actual.semantics.ontology and "concept_map" in actual.semantics.ontology:
        actual.semantics.ontology["concept_map"] = {
            node_map.get(k, k): v for k, v in actual.semantics.ontology["concept_map"].items()
        }


def analyze_command(args):
    output = safe_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("output profile exists; choose a new path to preserve its provenance")
    started = time.monotonic()
    if args.store:
        store = GraphStore(safe_path(args.store), read_only=True)
        source = {"system": "staged", "consistency": "immutable_staging"}
    else:
        _log("Scanning Neo4j read-only into project-local staging…")
        store, source = scan(
            args.uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            output.parent / "source-store",
            database=args.database,
            username=args.username,
            password=os.environ.get(args.password_env),
            batch_size=args.batch_size,
        )
    try:
        _log("Measuring graph characteristics…")
        profile = analyze_store(
            store,
            output.parent,
            source=source,
            seed=args.seed,
            sample_roots=args.sample_roots,
            exact_threshold=args.exact_threshold,
        )
        if args.semantics != "off":
            _log("Preparing semantic proposals with Codex…")
        try:
            profile = prepare(
                profile,
                output.parent,
                output.parent / "preparation",
                mode=args.semantics,
                timeout=args.codex_timeout,
            )
        except PreparationRequired as error:
            write_profile(error.profile, output)
            raise PreparationRequired(str(error), error.profile, output) from error
        _write_semantic_profile(profile, store, output)
        summary = {
            "profile": str(output),
            "nodes": profile.observations.node_count,
            "relationships": profile.observations.relationship_count,
            "metrics": len(profile.metrics),
            "seconds": time.monotonic() - started,
            "semantics": profile.semantics.status,
        }
        _report(summary, output.parent / "analysis-run.json")
        print(json.dumps(summary, indent=2))
    finally:
        store.close()


def generate_command(args):
    from .generation import generate

    path = safe_path(args.profile)
    base = path.parent
    profile = read_profile(path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = safe_path(args.output or Path("data/runs") / run_id)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("run output is not empty")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if args.target_uri:
        with connect(
            args.target_uri, username=args.username, password=os.environ.get(args.password_env)
        ) as driver:
            with driver.session(database=args.database) as session:
                if session.run("MATCH (n) RETURN count(n) AS n").single()["n"]:
                    raise ValueError("target database is populated; use an empty target")
    _log("Checking or preparing domain logic…")
    profile = prepare(
        profile, base, run_dir / "preparation", mode=args.prepare, timeout=args.codex_timeout
    )
    prep = profile.preparation
    if prep.provider_path:
        frozen_provider = run_dir / "domain-provider.py"
        shutil.copyfile(safe_path(prep.provider_path, base), frozen_provider)
        prep.provider_path = os.path.relpath(frozen_provider, base)
    _save_prepared(profile, base, run_dir / "requested-profile.yaml")
    runner = (
        ProviderRunner(
            safe_path(prep.provider_path, base), prep.provider_sha256, timeout=args.provider_timeout
        )
        if prep.provider_path
        else nullcontext(None)
    )
    _log("Generating nodes, relationships, and conditional properties…")
    with runner as provider:
        store = generate(
            profile,
            run_dir / "staging",
            base_dir=base,
            scale=args.scale,
            seed=args.seed,
            batch_size=args.batch_size,
            provider=provider,
        )
    try:
        check_property_types(store, profile)
        checks = check_rules(store, profile.semantics.rules + profile.targets.rules)
        failures = [
            r for r in checks if r["hard"] and (r["violations"] or r["status"] != "measured")
        ]
        _report({"rules": checks, "passed": not failures}, run_dir / "preload-validation.json")
        if failures:
            raise ValueError(
                "generated graph violates hard constraints; see "
                + str(run_dir / "preload-validation.json")
            )
        _log("Pre-load checks passed.")
        password = None
        if args.target_uri:
            target = load_online(
                store,
                args.target_uri,
                database=args.database,
                username=args.username,
                password=os.environ.get(args.password_env),
                batch_size=args.batch_size,
            )
            target["uri"] = args.target_uri
        elif args.target == "local":
            _log("Loading a fresh local Neo4j Community instance…")
            target = load_local(store, run_dir, offline=args.loader == "offline")
            password = local_password(run_dir)
        else:
            target = {"mode": "stage", "status": "not_loaded"}
        if target.get("uri"):
            _log("Re-scanning the target database to verify the loaded graph…")
            verify, source = scan(
                target["uri"],
                run_dir / "verification-store",
                database=target.get("database", args.database),
                username=target.get("username", args.username),
                password=password or os.environ.get(args.password_env),
                batch_size=args.batch_size,
            )
        else:
            verify = store
            source = {"system": "staged", "consistency": "immutable_staging"}
        try:
            measured = analyze_store(
                verify,
                run_dir / "analysis",
                source=source,
                seed=args.seed,
                sample_roots=args.sample_roots,
                exact_threshold=args.exact_threshold,
            )
            _remap_semantics(profile, measured)
            attach_semantic_metrics(measured, verify)
            write_profile(measured, run_dir / "analysis" / "profile.yaml")
            fidelity = compare_profiles(
                profile,
                measured,
                expected_dir=base,
                actual_dir=run_dir / "analysis",
                scale=args.scale,
            )
        finally:
            if verify is not store:
                verify.close()
        _report(fidelity, run_dir / "fidelity.json")
        n, m = store.counts()
        elapsed = time.monotonic() - started
        summary = {
            "run": str(run_dir),
            "target": target,
            "nodes": n,
            "relationships": m,
            "seconds": elapsed,
            "entities_per_second": (n + m) / elapsed,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "seed": args.seed,
            "scale": args.scale,
            "profile_digest": profile_digest(profile),
            "requested_profile": str(run_dir / "requested-profile.yaml"),
            "recipe_digest": hashlib.sha256(dumps(prep.recipe).encode()).hexdigest(),
            "provider_sha256": prep.provider_sha256,
            "uv_lock_sha256": sha256(project_root() / "uv.lock")
            if (project_root() / "uv.lock").is_file()
            else None,
            "fidelity": fidelity["status"],
            "fidelity_report": str(run_dir / "fidelity.json"),
            "loaded": target["status"] == "loaded",
        }
        _report(summary, run_dir / "run.json")
        print(json.dumps(summary, indent=2))
        if fidelity["status"] == "failed":
            raise ValueError("post-load validation found hard violations; see fidelity report")
    finally:
        store.close()


def parser():
    result = argparse.ArgumentParser(
        description="Characterize Neo4j graphs and synthesize measured counterparts."
    )
    commands = result.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser(
        "analyze", help="read-only deep characterization into a YAML profile"
    )
    source = analyze.add_mutually_exclusive_group()
    source.add_argument("--uri")
    source.add_argument(
        "--store", help="analyze an existing project-local staging directory instead of Neo4j"
    )
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--semantics", choices=["codex", "auto", "off"], default="codex")
    generate = commands.add_parser(
        "generate", help="generate, load, and remeasure a separate graph"
    )
    generate.add_argument("--profile", type=Path, required=True)
    generate.add_argument("--target", choices=["local", "stage"], default="local")
    generate.add_argument("--target-uri")
    generate.add_argument("--loader", choices=["offline", "online"], default="offline")
    generate.add_argument("--output", type=Path)
    generate.add_argument("--scale", type=float, default=1)
    generate.add_argument("--prepare", choices=["auto", "codex", "off"], default="auto")
    generate.add_argument("--provider-timeout", type=int, default=30)
    for cmd in (analyze, generate):
        cmd.add_argument("--database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
        cmd.add_argument("--username", default=os.environ.get("NEO4J_USERNAME", "neo4j"))
        cmd.add_argument("--password-env", default="NEO4J_PASSWORD")
        cmd.add_argument("--seed", type=int, default=42)
        cmd.add_argument("--sample-roots", type=int, default=128)
        cmd.add_argument("--exact-threshold", type=int, default=2000)
        cmd.add_argument("--batch-size", type=int, default=10000)
        cmd.add_argument("--codex-timeout", type=int, default=300)
    return result


def main():
    _env()
    args = parser().parse_args()
    if args.batch_size < 1 or args.sample_roots < 1 or args.exact_threshold < 0:
        parser().error(
            "batch size and root sample must be positive; exact threshold cannot be negative"
        )
    try:
        needs_connection = (args.command == "analyze" and not args.store) or (
            args.command == "generate" and args.target_uri
        )
        if needs_connection and not os.environ.get(args.password_env):
            raise ValueError(f"set {args.password_env} before connecting")
        if args.command == "generate" and args.target == "stage" and args.target_uri:
            raise ValueError("--target stage cannot be combined with --target-uri")
        if args.command == "analyze":
            analyze_command(args)
        else:
            generate_command(args)
    except PreparationRequired as error:
        _log(str(error))
        if error.path:
            _log(f"Resolve questions in {error.path}, then rerun with that profile.")
        raise SystemExit(2) from error
    except (ValueError, RuntimeError, OSError, PreparationError, DriverError, Neo4jError) as error:
        _log(f"Error: {error}")
        raise SystemExit(1) from error
