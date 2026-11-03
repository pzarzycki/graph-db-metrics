"""Opt-in full CLI benchmark. All input, data, logs, and reports remain project-local."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from graph_metrics.io import project_root, safe_path, write_profile
from graph_metrics.models import (
    NodeSpec,
    Profile,
    PropertySpec,
    RelationshipSpec,
    Rule,
    Semantics,
    Targets,
)


def benchmark_profile(nodes: int, relationships: int) -> Profile:
    if nodes < 2 or relationships < 0 or relationships > nodes * (nodes - 1):
        raise ValueError("benchmark needs at least two nodes and feasible nonnegative simple edges")
    quotient, remainder = divmod(relationships, nodes)
    degrees = [{"degree": quotient, "count": nodes - remainder}]
    if remainder:
        degrees.append({"degree": quotient + 1, "count": remainder})
    return Profile(
        semantics=Semantics(
            status="declared",
            concepts=[
                {
                    "cohort": "entity",
                    "meaning": "Synthetic benchmark entities; no domain semantics inferred",
                }
            ],
            rules=[
                Rule(
                    id="identity",
                    kind="unique",
                    parameters={"cohort": "entity", "property": "synthetic_id"},
                ),
                Rule(
                    id="category_required",
                    kind="required",
                    parameters={"cohort": "entity", "property": "category"},
                ),
                Rule(
                    id="numeric_range",
                    kind="range",
                    parameters={"cohort": "entity", "property": "value", "min": 0, "max": 1000},
                ),
                Rule(
                    id="outgoing_cardinality",
                    kind="cardinality",
                    parameters={
                        "relationship": "links",
                        "cohort": "entity",
                        "direction": "out",
                        "min": quotient,
                        "max": quotient + bool(remainder),
                    },
                ),
            ],
        ),
        targets=Targets(
            nodes=[
                NodeSpec(
                    id="entity",
                    labels=["BenchmarkEntity"],
                    count=nodes,
                    properties={
                        "synthetic_id": PropertySpec(type="integer", distribution={"kind": "id"}),
                        "category": PropertySpec(
                            type="string",
                            distribution={
                                "kind": "categorical",
                                "values": [f"category_{i}" for i in range(16)],
                            },
                        ),
                        "value": PropertySpec(
                            type="integer", distribution={"kind": "integer", "min": 0, "max": 1000}
                        ),
                    },
                )
            ],
            relationships=[
                RelationshipSpec(
                    id="links",
                    type="LINKS",
                    source="entity",
                    target="entity",
                    count=relationships,
                    out_degrees=degrees,
                    in_degrees=degrees,
                )
            ],
            allow_self_loops=False,
            allow_parallel_edges=False,
        ),
    )


def run(args):
    root = project_root()
    directory = safe_path(
        args.output
        or Path("data/benchmarks")
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    )
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("benchmark output must be empty; existing runs are preserved")
    directory.mkdir(parents=True, exist_ok=True)
    profile_path = directory / "target.yaml"
    write_profile(benchmark_profile(args.nodes, args.relationships), profile_path)
    run_dir = directory / "run"
    command = [
        str(Path(sys.executable).parent / "graph-metrics"),
        "generate",
        "--profile",
        str(profile_path),
        "--output",
        str(run_dir),
        "--prepare",
        "off",
        "--target",
        args.target,
        "--loader",
        args.loader,
        "--seed",
        str(args.seed),
        "--sample-roots",
        str(args.sample_roots),
        "--exact-threshold",
        "2000",
        "--batch-size",
        str(args.batch_size),
    ]
    started = time.monotonic()
    with (
        (directory / "stdout.json").open("w") as out,
        (directory / "progress.log").open("w") as err,
    ):
        result = subprocess.run(command, cwd=root, stdout=out, stderr=err, timeout=args.timeout)
    elapsed = time.monotonic() - started
    report = {
        "schema_version": "1",
        "status": "completed" if result.returncode == 0 else "failed",
        "command": command,
        "returncode": result.returncode,
        "seconds": elapsed,
        "nodes": args.nodes,
        "relationships": args.relationships,
        "seed": args.seed,
        "entities_per_second": (args.nodes + args.relationships) / elapsed,
        "child_peak_rss_mib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
            "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        },
        "scope": "Peak RSS is the CLI child process; Neo4j Docker memory is separate.",
        "run": str(run_dir.relative_to(root)),
    }
    if (run_dir / "run.json").exists():
        report["pipeline"] = json.loads((run_dir / "run.json").read_text())
    if (run_dir / "fidelity.json").exists():
        report["fidelity"] = json.loads((run_dir / "fidelity.json").read_text())
    (directory / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(directory / "benchmark.json"),
                "seconds": elapsed,
            },
            indent=2,
        )
    )
    return result.returncode


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--nodes", type=int, default=1_000_000)
    result.add_argument("--relationships", type=int, default=5_000_000)
    result.add_argument("--output", type=Path)
    result.add_argument("--target", choices=["local", "stage"], default="local")
    result.add_argument("--loader", choices=["offline", "online"], default="offline")
    result.add_argument("--sample-roots", type=int, default=32)
    result.add_argument("--batch-size", type=int, default=10000)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--timeout", type=int, default=14400)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
