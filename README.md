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

Documentation foundation, not an implemented Neo4j analyzer. No claim of empirical benchmark validation.

## Authoring

The complete LaTeX-typeset PDF was produced as a companion technical reference. The repository documentation is a textual presentation of the same measurement-first principles.
