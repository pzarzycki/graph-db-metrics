# Graph Database Metrics

A measurement-first framework for **characterizing arbitrary existing property graphs**, including structural organization, typed relationships, property-schema heterogeneity, value distributions, and domain-dependent semantics.

**Primary use:** analyze and compare existing graph databases. **Secondary use:** use measured profiles as targets for synthetic graph generation and evaluate generated-graph fidelity.

## Core distinction

1. **Observed graph:** counts, label combinations, directed edges, typed relationship signatures, property presence and observed values.
2. **Derived measurements:** paths, components, motifs, communities, association statistics. These require explicit projections, estimators and scope.
3. **Interpreted semantics:** ontology, units, domain constraints, valid signatures and rule conformity. These cannot be fully inferred from labels and topology alone.

See [the technical framework (PDF)](graph_characterization.pdf) for formulas, measurement contracts, interpretation, counterexamples, and validation requirements.

## Metric families

| Layer | Expected reporting |
| --- | --- |
| Population | Nodes/relationships; label-set and relationship-type frequencies |
| Connectivity | Joint in/out-degree distribution; parallel edges; loops; components |
| Reachability | BFS layer sizes; distributions of k-hop neighborhoods; path lengths |
| Local structure | Triangles / motifs, clustering, projection assumptions |
| Relational heterogeneity | Edge-signature distribution; typed directional degree / cardinality distributions |
| Property schema | Presence masks, key co-occurrence, complete signature frequencies |
| Property values | Missingness, type mix, frequencies, quantiles, conditional distributions |
| Dependencies | Key co-occurrence, conditional mutual information, topology-attribute coupling |
| Semantics, optional | Concept map, units, permitted signatures, validated constraints |

Applicability depends on graph scope, projection, computational budget and available domain information. Report each metric as exact, estimated, not applicable, not measured or ontology required, as appropriate.

## Measurement invariant

Every metric must include its **mathematical definition, graph scope, denominator, projection, value, estimator, uncertainty and status**. Missing domain semantics must be indicated as *unavailable*, never inferred without qualification.

## Report vs generator

Characterization: `graph -> measured profile`.

Optional generation: `target profile + semantic contract + seed -> graph`.

Validation uses the **same metrics** for source and generated graphs. Statistical distributions and constraints are independently reported, rather than collapsed into one complexity score.

## Status

The `graph-metrics` Python CLI analyzes existing Neo4j databases and generates
synthetic graphs from measured or authored YAML profiles. It uses Neo4j Community
Edition, disk-backed distribution tables, and independent post-load validation.

## Run the tools

Install the locked Python environment with [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked
```

The project keeps the virtual environment and uv cache locally. Set connection
credentials in `.env` or environment variables; passwords never go into profiles.
The default username and database are both `neo4j`.

Analyze an existing database:

```sh
uv run graph-metrics analyze --uri bolt://localhost:7687 \
  --output data/profiles/source/profile.yaml
```

Analysis reads the source without adding nodes, properties, constraints, or plugins.
It stages records locally and writes a versioned YAML profile plus checksummed
Parquet distribution files. Source queries are read-committed and do not constitute
an atomic snapshot; use a quiescent source when snapshot consistency matters.

By default, analysis invokes your installed/authenticated Codex CLI to propose
semantic meanings and generation logic from bounded summaries and samples. Use
`--semantics off` for observation-only analysis. Inferred rules retain their
provenance. Questions and contradictions are saved for resolution, with exit code
2; successful measurements remain available even if semantic preparation fails.

Generate a separate graph from that profile:

```sh
uv run graph-metrics generate --profile data/profiles/source/profile.yaml \
  --target local --scale 10 --seed 42
```

Local generation exports CSV, runs a native offline import, starts a fresh
Neo4j Community container, and re-scans it. The result prints its connection URI,
Browser address, and the path to its private credentials file. The original
development database stays separate. Each generated instance has project-local
data and logs; `connection.json` identifies its container for later management.

For a running local or remote **empty** database, export its password in a named
environment variable and use:

```sh
uv run graph-metrics generate --profile examples/employment.yaml \
  --target-uri bolt://target-host:7687 --password-env TARGET_NEO4J_PASSWORD \
  --prepare off
```

Populated targets are refused. Online loading uses batched driver writes and
removes temporary loader metadata before measurement. A failed load leaves its
partial target and diagnostic artifacts intact; it never silently clears data.
Use `--loader online` with `--target local` for a fresh local container loaded
through the driver, or `--target stage` for a procedural preview without a database.

Native CSV import supports ordinary scalar, array, temporal, and spatial values.
Use online loading for byte arrays or property names containing `:` or backticks.
Arrays containing the internal CSV array delimiter also require online loading.

Try the two authored domain examples without Codex:

```sh
uv run graph-metrics generate --profile examples/aviation.yaml --prepare off
uv run graph-metrics generate --profile examples/employment.yaml --prepare off
```

The examples include dependent values and cross-entity rules: aviation coordinates,
routes, distance and flight times; employment roles, salaries, skills and employer
locations. See [profile format and supported recipes](docs/profile-format.md).

## Semantic preparation and reproducibility

Generation automatically prepares missing domain logic and caches frozen recipes
under `.cache/graph-metrics/`. Compatible later runs reuse those recipes. Explicit
`--prepare off` uses declared procedural providers only; `--prepare codex` requests
a new preparation. The Codex coordinator may use up to two reasoning/review helpers.

Generated Python providers run in a pinned Docker worker with no network, database
credentials, Docker socket, or writable project mount. Only the loader writes to
Neo4j. Providers receive bounded batches and linked-entity context; the LLM is not
called once per generated record. The same seed, frozen recipe, and locked
dependencies reproduce generated content independently of batch size.

Generation validates hard rules before loading, then re-analyzes the loaded target.
Each run saves `preload-validation.json`, `analysis/profile.yaml`, `fidelity.json`,
and `run.json`. The fidelity report lists achieved distributions, unmet targets,
unsupported comparisons, and semantic violations separately. `partial` means soft
targets differ or could not be compared; hard violations fail the run.

The topology generator preserves feasible typed degrees and joint role assignments
and uses bounded degree-preserving community refinement. It cannot promise exact
components, paths, clustering, or motifs simultaneously. Distribution summaries
do not uniquely identify the source graph or its meaning. Arbitrary business rules
also require an appropriate declared recipe; unsupported hard rules stop generation.

## Validation

```sh
uv run pytest -m 'not integration and not scale'
uv run ruff check src tests scripts
GRAPH_METRICS_RUN_INTEGRATION=1 uv run pytest -m integration
```

Run the complete end-to-end suite, including live Codex preparation and the
million-node import, with an installed/authenticated Codex CLI and Docker:

```sh
GRAPH_METRICS_RUN_INTEGRATION=1 GRAPH_METRICS_RUN_CODEX=1 \
  GRAPH_METRICS_RUN_SCALE=1 uv run pytest
```

Integration tests use disposable, separate Community containers and verify native
types, multigraph structure, both loaders, source preservation, and both domains.
Run the opt-in million-node, five-million-relationship benchmark with:

```sh
uv run python scripts/benchmark.py --target local
```

It records hardware, Python peak memory, elapsed time, throughput, and fidelity
under `data/benchmarks/`. Neo4j Docker memory is separate from Python peak memory.

## Authoring

The complete LaTeX-typeset PDF was produced as a companion technical reference. The repository documentation is a textual presentation of the same measurement-first principles.

## Local database

This project uses Neo4j Community Edition through Docker Compose.

For a fresh checkout, copy `.env.example` to `.env` and set a strong
`NEO4J_PASSWORD` before starting the database. The initial setup in this workspace
already has a generated password in `.env`.

```sh
mkdir -p data/neo4j/data data/neo4j/logs data/neo4j/import
docker compose up -d --wait
```

- Browser: <http://localhost:7474>
- Bolt connection: `bolt://localhost:7687`
- Username: `neo4j`
- Password: `NEO4J_PASSWORD` in `.env`

Ports are bound to the Docker host's local interface. For access from another
machine, forward ports 7474 and 7687 over SSH to the Docker host, then use the
addresses above.

Database files, logs, and import files live under `data/neo4j/` in this project.
Git ignores this directory and `.env`; source files and dependency lockfiles
should stay tracked. Changing `.env` after database initialization does not
change the existing database password.

```sh
docker compose ps            # Check status
docker compose logs -f neo4j # Follow logs
docker compose stop         # Stop without removing the container
docker compose down         # Remove the container; retain project data
```

The image version is pinned in `compose.yaml`. Docker recreates the container
using the same project data when you run `docker compose up -d --wait` again.

Setup follows the [official Neo4j Docker guide](https://neo4j.com/docs/operations-manual/current/docker/introduction/).
