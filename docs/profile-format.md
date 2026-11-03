# Profile and recipe format

`schema_version: '1'` is validated with Pydantic. Unknown schema fields, duplicate
cohort IDs, duplicate complete label sets, duplicate relationship signatures, and
relationships referencing unknown cohorts are rejected. YAML uses safe constructors
and YAML 1.2 boolean behavior: words such as `ON` and `YES` remain strings.

## Sections

| Section | Meaning |
| --- | --- |
| `source` | Measurement interval, database, consistency notes and observed constraints; no credentials |
| `scope` | Direction, edge multiplicity, and complete-label-set cohort policy |
| `observations` | Population counts, node and relationship cohort specifications measured from the source |
| `metrics` | Definitions, scope, denominator, estimator, value, uncertainty, and status |
| `semantics` | Declared or inferred concepts, meanings, rules, assumptions, unresolved questions, optional ontology |
| `targets` | Optional authored/overridden cohorts, rules, scale, metric targets and tolerances |
| `preparation` | Frozen recipe, provider code reference and checksum, profile digest, preparation version information |
| `samples` | Bounded representative source examples for interpretation |

Authored specifications can omit observations and metrics. The generator uses
`targets.nodes` and `targets.relationships` when supplied, otherwise their observed
counterparts. Target and command-line scale multiply. Largest-remainder allocation
keeps total population counts exact; degree sequences must still obey incidence
and cardinality constraints. Observations are never edited to look like targets.

Node cohorts have `id`, `labels`, `count`, and optional `properties`, `patterns`,
`degrees`, and `roles`. An empty labels list represents unlabeled nodes. Relationship
cohorts have `id`, `type`, `source`, `target`, `count`, optional properties/patterns,
and `out_degrees` / `in_degrees` that include eligible zero-degree nodes.

```yaml
schema_version: '1'
semantics:
  status: declared
  rules:
    - id: identity
      kind: unique
      parameters: {cohort: device, property: device_id}
targets:
  allow_self_loops: false
  allow_parallel_edges: false
  nodes:
    - id: device
      labels: [Device]
      count: 10000
      properties:
        device_id: {type: integer, distribution: {kind: id}}
        model:
          type: string
          distribution: {kind: categorical, values: [A, B], weights: [3, 1]}
        temperature:
          type: float
          presence: 0.9
          distribution: {kind: normal, mean: 20, std: 3}
  relationships:
    - id: connects
      type: CONNECTS_TO
      source: device
      target: device
      count: 30000
      out_degrees: [{degree: 3, count: 10000}]
```

## Distributions and native values

A property specification has `type`, `presence` (0 through 1), and `distribution`.
Complete presence patterns take priority over independent missingness probabilities.
Built-in distribution kinds include `constant`, `categorical`, `empirical`,
`integer`, `uniform`, `normal`, `quantiles`, `boolean`, `id`, `faker`, `template`,
`conditional`, `derived`, `copy`, `lookup`, `reference`, `date`, `datetime`,
`datetime_offset`, `list`, `bytes`, and `distance_km`.

- Categorical/empirical distributions use `values` and optional nonnegative `weights`,
  or a `table` reference containing `value` / `count` rows.
- `conditional` selects a case using `on` and `cases`; `copy` reads a field from the
  current record or `source_properties` / `target_properties` for relationships.
- `lookup` and `reference` access already-generated cohorts. Cyclic provider
  dependencies are rejected. See the example profiles for full parameter shapes.
- Declared unique identifiers and recognized source ID fields are regenerated;
  `preserve_values: true` explicitly requests retaining empirical identifiers.

Empirical generation reuses the measured value support. Domain recipes and Python
providers can replace it with new names, descriptions, or domain vocabularies.

`preparation.recipe.property_overrides` is a list of entries containing `entity`
(`node` or `relationship`), `cohort`, `property`, and a procedural `distribution`.
These recipes override generation providers without editing observations. A
`preserve_values: true` entry can retain a public domain code's measured support
without listing a potentially truncated vocabulary again. Mandatory endpoint
matching retains measured role/attribute coupling; explicit authored node targets
take precedence over these measured conditional tables.

Large tables are relative Parquet references with `path`, `rows`, `sha256`, and
`format: parquet`. Each Parquet row contains a JSON `record`, preserving mixed
native types. Checksums and project boundaries are checked before reading. Move a
profile together with its companion tables, or preserve their relative paths.

Temporal, spatial and byte values use explicit JSON-compatible wrappers rather
than executable YAML tags, for example:

```yaml
__neo4j_type__: date
value: '2026-01-01'
```

Null properties, nested maps, nested arrays, mixed nonnumeric arrays, and integers
outside Neo4j's signed 64-bit range are rejected before loading. Absent properties
remain distinct from domain-specific sentinel strings such as `UNKNOWN`.

## Rule parameters

Rules carry `id`, `kind`, `parameters`, `provenance` (`declared`, `inferred`, or
`database`), and `hard` (default true). Property rules use node cohorts unless
`entity: relationship` is specified.

| Kind | Required parameters |
| --- | --- |
| `required`, `unique` | `cohort`, `property` |
| `range` | `cohort`, `property`, `min` and/or `max` |
| `enum` | `cohort`, `property`, `values` |
| `temporal_order` | `cohort`, `before`, `after` (inclusive order) |
| `cardinality` | `relationship`, `direction: out/in`, `min` and/or `max`; optional explicit `cohort` |
| `same_property` | `relationship`, `source_property`, `target_property` |
| `dag` | `relationship` |
| `allowed_signature` | `signatures: [{source, type, target}]` using cohort IDs |
| `property_dependency` | `cohort`, `if_present`, `then_present` |
| `functional_dependency` | `cohort`, `determinant`, `dependent` |

Checks report eligible denominators and violations. Passing a limited collection
of checks does not prove complete domain correctness. Generation may reject a
combination it cannot construct even when that combination is theoretically feasible.

An optional ontology has `concept_map: {cohort: concept}` and pairwise
`distances: {conceptA: {conceptB: distance}}`. Conceptual dispersion is computed
only for complete supplied mappings/distances; it is unavailable without them.

## Measurement and fidelity

Metric statuses are `exact`, `estimated`, `not_applicable`, `not_measured`, and
`ontology_required`. Expensive topology uses documented projections and seeded
sampling. Uncertainty is included where implemented and otherwise marked explicitly
as unavailable, with the reason. No meaningful universal graph-depth or semantic
distance is fabricated from labels alone.

The default discrete total-variation tolerance is 0.1 and normalized numeric
Wasserstein tolerance is 0.15 (`targets.tolerances.numeric_nrmse`, retained as the
configuration key). Counts are checked separately. `targets.metrics` can override
observed structural targets; corresponding tolerance keys use metric IDs. Reports
retain unsupported comparisons and all unachieved targets.

Prepared Python defines `provide(kind, cohort, rows, seed)` and returns one property
override dictionary per input row. The runner uses stable per-record seeds and
JSONL transport, so batch size cannot change results. Only supported standard-library
imports are allowed; Docker isolation is the execution boundary. Provider code and
distribution files must pass their saved checksums before reuse.
