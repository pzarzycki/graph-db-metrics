"""Independent rule checks and transparent fidelity comparison."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from .io import dumps, loads, read_table
from .models import Metric, Profile, Rule
from .store import GraphStore

SUPPORTED_RULES = {
    "required",
    "unique",
    "enum",
    "range",
    "temporal_order",
    "cardinality",
    "same_property",
    "dag",
    "allowed_signature",
    "property_dependency",
    "functional_dependency",
}


def check_property_types(store: GraphStore, profile: Profile | None = None) -> None:
    """Reject invalid authored/provider values before creating or touching a target."""
    from neo4j.spatial import CartesianPoint, WGS84Point
    from neo4j.time import Date, DateTime, Duration, Time

    scalar_types = (
        str,
        bool,
        int,
        float,
        bytes,
        Date,
        DateTime,
        Time,
        Duration,
        CartesianPoint,
        WGS84Point,
    )
    specifications = (
        {s.id: s.properties for group in profile.generation_specs() for s in group}
        if profile
        else {}
    )
    expected_types = {
        "string": str,
        "str": str,
        "integer": int,
        "int": int,
        "float": float,
        "boolean": bool,
        "bool": bool,
        "date": Date,
        "datetime": DateTime,
        "time": Time,
        "duration": Duration,
        "point": (CartesianPoint, WGS84Point),
        "bytes": bytes,
        "list": list,
    }

    def scalar(value):
        if not isinstance(value, scalar_types):
            return False
        return not isinstance(value, int) or isinstance(value, bool) or -(2**63) <= value < 2**63

    for iterator in (store.iter_nodes, store.iter_edges):
        for batch in iterator():
            for row in batch:
                for key, value in row["properties"].items():
                    declared = specifications.get(row["cohort"], {}).get(key)
                    if declared and declared.type != "mixed":
                        expected_type = expected_types.get(declared.type)
                        if (
                            expected_type is None
                            or not isinstance(value, expected_type)
                            or (declared.type in ("integer", "int") and isinstance(value, bool))
                        ):
                            raise ValueError(
                                f"property {row['cohort']}.{key} does not match declared type {declared.type!r}"
                            )
                    if isinstance(value, list):
                        valid = all(scalar(v) and not isinstance(v, bytes) for v in value)
                        kinds = {
                            "number"
                            if isinstance(v, (float, int)) and not isinstance(v, bool)
                            else type(v).__name__
                            for v in value
                        }
                        valid = valid and len(kinds) <= 1
                    else:
                        valid = scalar(value)
                    if not valid:
                        raise ValueError(
                            f"invalid Neo4j property {row['cohort']}.{key}: null, maps, nested/mixed arrays and out-of-range integers cannot be loaded"
                        )


def check_rules(store: GraphStore, rules: list[Rule]) -> list[dict]:
    results = []
    for rule in rules:
        p = rule.parameters
        result = {
            "id": rule.id,
            "kind": rule.kind,
            "provenance": rule.provenance,
            "hard": rule.hard,
            "eligible": 0,
            "violations": 0,
            "status": "measured",
        }
        try:
            if rule.kind not in SUPPORTED_RULES:
                raise ValueError(f"unsupported rule kind: {rule.kind}")
            if rule.kind == "cardinality":
                rel = p["relationship"]
                direction = p.get("direction", "out")
                if direction not in ("out", "in"):
                    raise ValueError("cardinality direction must be out or in")
                column = "source" if direction == "out" else "target"
                cohorts = store.db.execute(
                    f"SELECT DISTINCT n.cohort FROM edges e JOIN nodes n ON n.id=e.{column} WHERE e.cohort=?",
                    [rel],
                ).fetchall()
                cohort = p.get("cohort")
                if cohort is None:
                    if not cohorts:
                        raise ValueError(
                            "cardinality on an empty relationship cohort requires parameters.cohort"
                        )
                    cohort = cohorts[0][0]
                rows = store.db.execute(
                    f"""SELECT coalesce(k,0),count(*) FROM nodes n
                    LEFT JOIN (SELECT {column} id,count(*) k FROM edges WHERE cohort=? GROUP BY {column}) e USING(id)
                    WHERE n.cohort=? GROUP BY coalesce(k,0)""",
                    [rel, cohort],
                ).fetchall()
                result["eligible"] = sum(c for _, c in rows)
                result["violations"] = sum(
                    c for k, c in rows if k < p.get("min", 0) or k > p.get("max", float("inf"))
                )
            elif rule.kind == "dag":
                import igraph as ig

                rows = store.db.execute(
                    "SELECT source,target FROM edges WHERE cohort=?", [p["relationship"]]
                ).fetchall()
                vertices = {v for row in rows for v in row}
                if p.get("node_cohorts"):
                    vertices.update(
                        v
                        for (v,) in store.db.execute(
                            "SELECT id FROM nodes WHERE cohort=ANY(?)", [p["node_cohorts"]]
                        ).fetchall()
                    )
                ids = {v: i for i, v in enumerate(sorted(vertices))}
                graph = ig.Graph(
                    n=len(ids), edges=[(ids[a], ids[b]) for a, b in rows], directed=True
                )
                result["eligible"] = 1
                result["violations"] = int(not graph.is_dag())
                if not result["violations"]:
                    depths = np.zeros(len(ids), dtype=np.int64)
                    for vertex in graph.topological_sorting(mode="out"):
                        for target in graph.neighbors(vertex, mode="out"):
                            depths[target] = max(depths[target], depths[vertex] + 1)
                    result["hierarchy"] = {
                        "relationship": p["relationship"],
                        "root_policy": "all zero-indegree vertices",
                        "max_depth": int(depths.max()) if len(depths) else None,
                        "depths": [
                            {"depth": int(d), "count": c}
                            for d, c in sorted(Counter(depths).items())
                        ],
                        "nodes": len(ids),
                    }
            elif rule.kind == "same_property":
                rows = store.db.execute(
                    """SELECT s.properties,t.properties FROM edges e
                    JOIN nodes s ON s.id=e.source JOIN nodes t ON t.id=e.target WHERE e.cohort=?""",
                    [p["relationship"]],
                )
                while batch := rows.fetchmany(10000):
                    for src, dst in batch:
                        a, b = (
                            loads(src).get(p["source_property"]),
                            loads(dst).get(p["target_property"]),
                        )
                        if a is not None and b is not None:
                            result["eligible"] += 1
                            result["violations"] += int(a != b)
            elif rule.kind == "allowed_signature":
                allowed = {(x["source"], x["type"], x["target"]) for x in p["signatures"]}
                for (
                    src,
                    kind,
                    dst,
                    count,
                ) in store.db.execute("""SELECT s.cohort,e.type,t.cohort,count(*) FROM edges e
                    JOIN nodes s ON s.id=e.source JOIN nodes t ON t.id=e.target GROUP BY s.cohort,e.type,t.cohort""").fetchall():
                    result["eligible"] += count
                    result["violations"] += count * int((src, kind, dst) not in allowed)
            else:
                cohort = p["cohort"]
                table = "edges" if p.get("entity") == "relationship" else "nodes"
                if rule.kind in ("unique", "functional_dependency"):
                    key = p["property"] if rule.kind == "unique" else p["determinant"]
                    path = "/" + key.replace("~", "~0").replace("/", "~1")
                    if rule.kind == "unique":
                        eligible, distinct = store.db.execute(
                            f"SELECT count(json_extract(properties,?)),count(DISTINCT json_extract(properties,?)) FROM {table} WHERE cohort=?",
                            [path, path, cohort],
                        ).fetchone()
                        result["eligible"] = eligible
                        result["violations"] = eligible - distinct
                    else:
                        dep = "/" + p["dependent"].replace("~", "~0").replace("/", "~1")
                        rows = store.db.execute(
                            f"SELECT count(*),count(DISTINCT json_extract(properties,?)) FROM {table} WHERE cohort=? AND json_extract(properties,?) IS NOT NULL AND json_extract(properties,?) IS NOT NULL GROUP BY json_extract(properties,?)",
                            [dep, cohort, path, dep, path],
                        ).fetchall()
                        result["eligible"] = sum(c for c, _ in rows)
                        result["violations"] = sum(c for c, d in rows if d > 1)
                else:
                    cursor = store.db.cursor()
                    cursor.execute(f"SELECT properties FROM {table} WHERE cohort=?", [cohort])
                    try:
                        while batch := cursor.fetchmany(10000):
                            for (raw,) in batch:
                                props = loads(raw)
                                if rule.kind == "required":
                                    result["eligible"] += 1
                                    result["violations"] += int(p["property"] not in props)
                                elif rule.kind in ("range", "enum"):
                                    if p["property"] in props:
                                        value = props[p["property"]]
                                        result["eligible"] += 1
                                        failed = (
                                            (
                                                value < p.get("min", float("-inf"))
                                                or value > p.get("max", float("inf"))
                                            )
                                            if rule.kind == "range"
                                            else dumps(value) not in {dumps(v) for v in p["values"]}
                                        )
                                        result["violations"] += int(failed)
                                elif rule.kind == "temporal_order":
                                    if p["before"] in props and p["after"] in props:
                                        result["eligible"] += 1
                                        result["violations"] += int(
                                            props[p["before"]] > props[p["after"]]
                                        )
                                elif rule.kind == "property_dependency":
                                    trigger = p.get("if_present", p.get("if"))
                                    required = p.get("then_present", p.get("then"))
                                    if not isinstance(trigger, str) or not isinstance(
                                        required, str
                                    ):
                                        raise ValueError(
                                            "property_dependency requires if_present and then_present"
                                        )
                                    if trigger in props:
                                        result["eligible"] += 1
                                        result["violations"] += int(required not in props)
                    finally:
                        cursor.close()
        except (KeyError, ValueError, TypeError) as error:
            result.update(status="not_measured", reason=str(error))
        result["violation_rate"] = (
            result["violations"] / result["eligible"] if result["eligible"] else None
        )
        results.append(result)
    return results


def attach_semantic_metrics(profile: Profile, store: GraphStore) -> list[dict]:
    rules = profile.semantics.rules + profile.targets.rules
    # Enrich empty-cohort cardinality tests using the declared relationship signatures.
    relationships = {r.id: r for r in profile.generation_specs()[1]}
    enriched = []
    for rule in rules:
        if rule.kind == "cardinality" and "cohort" not in rule.parameters:
            rel = relationships.get(rule.parameters.get("relationship"))
            if rel:
                rule = rule.model_copy(deep=True)
                rule.parameters["cohort"] = (
                    rel.source if rule.parameters.get("direction", "out") == "out" else rel.target
                )
        if rule.kind == "dag":
            rel = relationships.get(rule.parameters.get("relationship"))
            if rel:
                rule = rule.model_copy(deep=True)
                rule.parameters["node_cohorts"] = list(dict.fromkeys([rel.source, rel.target]))
        enriched.append(rule)
    results = check_rules(store, enriched)
    hierarchies = [r["hierarchy"] for r in results if "hierarchy" in r]
    if hierarchies:
        profile.metrics = [m for m in profile.metrics if m.id != "topology.hierarchy"]
        profile.metrics.append(
            Metric(
                id="topology.hierarchy",
                definition="Longest directed distance from any zero-indegree root within declared acyclic relationship projections",
                scope=profile.scope
                | {
                    "root_policy": "all zero-indegree vertices",
                    "relationship_filter": "per declared DAG rule",
                },
                value=hierarchies,
                denominator={h["relationship"]: h["nodes"] for h in hierarchies},
                status="exact",
            )
        )
    if rules:
        profile.metrics = [m for m in profile.metrics if m.id != "semantics.rules"]
        profile.metrics.append(
            Metric(
                id="semantics.rules",
                definition="Violation counts divided by eligible tests for each explicitly attributed semantic rule",
                scope=profile.scope,
                denominator={r["id"]: r["eligible"] for r in results},
                value=results,
                status="exact"
                if all(r["status"] == "measured" for r in results)
                else "not_measured",
                reason=None
                if all(r["status"] == "measured" for r in results)
                else "Some rule kinds or parameterizations could not be validated; see per-rule results.",
            )
        )
    # Ontology distances are usable only when an explicit mapping and complete distances exist.
    ontology = profile.semantics.ontology
    if ontology:
        mapping = ontology.get("concept_map", {})
        distances = ontology.get("distances", {})
        weights = Counter()
        for node in profile.observations.nodes:
            if node.id in mapping:
                weights[mapping[node.id]] += node.count
        total = profile.observations.node_count
        concepts = sorted(weights)
        if (
            sum(weights.values()) == total
            and total
            and all(
                b in distances.get(a, {}) for i, a in enumerate(concepts) for b in concepts[i + 1 :]
            )
        ):
            dispersion = sum(
                2 * weights[a] * weights[b] / total**2 * distances[a][b]
                for i, a in enumerate(concepts)
                for b in concepts[i + 1 :]
            )
            profile.metrics = [
                m for m in profile.metrics if m.id != "semantics.conceptual_diversity"
            ]
            profile.metrics.append(
                Metric(
                    id="semantics.conceptual_diversity",
                    definition="2 sum_{i<j} p_i p_j d(i,j) under supplied ontology",
                    scope=profile.scope,
                    denominator=total,
                    value=dispersion,
                    status="exact",
                )
            )
    return results


def _frequencies(rows, *, key=None):
    result = Counter()
    for row in rows:
        category = row.get(key) if key else {k: v for k, v in row.items() if k != "count"}
        result[dumps(category)] += row["count"]
    total = sum(result.values())
    return {k: v / total for k, v in result.items()} if total else {}


def _tv(expected, actual):
    keys = expected.keys() | actual.keys()
    return sum(abs(expected.get(k, 0) - actual.get(k, 0)) for k in keys) / 2


def compare_profiles(
    expected: Profile, actual: Profile, *, expected_dir: Path, actual_dir: Path, scale=1.0
) -> dict:
    """Compare observations without mutating either profile or collapsing to one score."""
    report = {
        "schema_version": "1",
        "checks": [],
        "unsupported_targets": [],
        "hard_violations": [],
        "status": "passed",
    }
    nodes, relationships = expected.generation_specs()
    checks = report["checks"]
    tolerance = expected.targets.tolerances.get("discrete_tv", 0.1)
    effective_scale = scale * expected.targets.scale
    for entity, specs, count in (
        ("nodes", nodes, actual.observations.node_count),
        ("relationships", relationships, actual.observations.relationship_count),
    ):
        target_count = round(sum(s.count for s in specs) * effective_scale)
        checks.append(
            {
                "metric": f"population.{entity}",
                "expected": target_count,
                "actual": count,
                "distance": "absolute",
                "tolerance": 0,
                "passed": target_count == count,
            }
        )
    expected_edges = {r.id: r for r in relationships}
    actual_node_signatures = {n.id: tuple(sorted(n.labels)) for n in actual.observations.nodes}
    actual_edges = {r.id: r for r in actual.observations.relationships}
    expected_node_signatures = {n.id: tuple(sorted(n.labels)) for n in nodes}

    def normalize_roles(rows, edge_specs, node_signatures):
        normalized = []
        for row in rows:
            roles = {}
            for identifier, degree in row.get("roles", {}).items():
                rel = edge_specs.get(identifier)
                if rel:
                    signature = dumps(
                        [node_signatures[rel.source], rel.type, node_signatures[rel.target]]
                    )
                    roles[signature] = degree
            normalized.append({"roles": roles, "count": row["count"]})
        return normalized

    for entity, specs, actual_specs in [
        ("node", nodes, actual.observations.nodes),
        ("relationship", relationships, actual.observations.relationships),
    ]:
        # Cohort identifiers in authored specs differ from analyzer hashes: align by label/signature.
        if entity == "node":
            signatures = {s.id: tuple(sorted(s.labels)) for s in nodes}
            lookup = {tuple(sorted(s.labels)): s for s in actual_specs}

            def align(s):
                return tuple(sorted(s.labels))
        else:
            actual_nodes = {s.id: tuple(sorted(s.labels)) for s in actual.observations.nodes}
            lookup = {
                (actual_nodes[s.source], s.type, actual_nodes[s.target]): s for s in actual_specs
            }

            def align(s):
                return (signatures[s.source], s.type, signatures[s.target])

        for spec in specs:
            measured = lookup.get(align(spec))
            target_count = int(round(spec.count * effective_scale))
            actual_count = measured.count if measured else 0
            checks.append(
                {
                    "metric": f"{entity}.{spec.id}.count",
                    "expected": target_count,
                    "actual": actual_count,
                    "passed": abs(target_count - actual_count) <= 1,
                    "tolerance": 1,
                    "distance": "absolute",
                }
            )
            if measured is None:
                continue
            for field in (
                ("patterns", "degrees", "roles")
                if entity == "node"
                else ("patterns", "out_degrees", "in_degrees")
            ):
                lhs = read_table(getattr(spec, field), expected_dir)
                rhs = read_table(getattr(measured, field), actual_dir)
                if not lhs:
                    continue
                # Role vectors contain source cohort ids; comparing these by raw identifier is invalid.
                if field == "roles":
                    lhs = normalize_roles(lhs, expected_edges, expected_node_signatures)
                    rhs = normalize_roles(rhs, actual_edges, actual_node_signatures)
                value = _tv(_frequencies(lhs), _frequencies(rhs))
                checks.append(
                    {
                        "metric": f"{entity}.{spec.id}.{field}",
                        "distance": "total_variation",
                        "value": value,
                        "tolerance": tolerance,
                        "passed": value <= tolerance,
                    }
                )
            for key, prop in spec.properties.items():
                got = measured.properties.get(key)
                presence = abs(prop.presence - (got.presence if got else 0))
                checks.append(
                    {
                        "metric": f"{entity}.{spec.id}.{key}.presence",
                        "distance": "absolute",
                        "value": presence,
                        "tolerance": tolerance,
                        "passed": presence <= tolerance,
                    }
                )
                if got is None:
                    continue

                def values(distribution, base):
                    if distribution.get("table"):
                        return read_table(distribution["table"], base)
                    return [
                        {"value": v, "count": w}
                        for v, w in zip(
                            distribution.get("values", []),
                            distribution.get("weights", [1] * len(distribution.get("values", []))),
                            strict=True,
                        )
                    ]

                lhs = values(prop.distribution, expected_dir)
                rhs = values(got.distribution, actual_dir)
                if not lhs:
                    # Authored providers do not assert an empirical distribution.
                    continue
                if prop.distribution.get("identity", False) and not prop.distribution.get(
                    "preserve_values", False
                ):
                    continue
                if prop.type in ("integer", "float"):
                    try:
                        a = [v["value"] for v in lhs]
                        b = [v["value"] for v in rhs]
                        value = float(
                            wasserstein_distance(
                                a, b, [v["count"] for v in lhs], [v["count"] for v in rhs]
                            )
                        )
                        spread = max(max(a) - min(a), abs(float(np.mean(a))) * 1e-9, 1e-9)
                        normalized = value / spread
                        tol = expected.targets.tolerances.get("numeric_nrmse", 0.15)
                        checks.append(
                            {
                                "metric": f"{entity}.{spec.id}.{key}.values",
                                "distance": "wasserstein / observed range",
                                "value": normalized,
                                "raw_wasserstein": value,
                                "tolerance": tol,
                                "passed": normalized <= tol,
                            }
                        )
                    except (ValueError, TypeError):
                        report["unsupported_targets"].append(
                            {
                                "metric": key,
                                "reason": "Numeric target has mixed or nonfinite values.",
                            }
                        )
                elif (
                    prop.type not in ("string", "date", "datetime", "time", "bytes")
                    or prop.distribution.get("preserve_values", False)
                    or len(lhs) <= 20
                ):
                    value = _tv(_frequencies(lhs, key="value"), _frequencies(rhs, key="value"))
                    checks.append(
                        {
                            "metric": f"{entity}.{spec.id}.{key}.values",
                            "distance": "total_variation",
                            "value": value,
                            "tolerance": tolerance,
                            "passed": value <= tolerance,
                        }
                    )
    # Structural targets are soft comparisons, with matching projections and estimator policies.
    measured = {m.id: m for m in actual.metrics}
    targets = {
        m.id: m
        for m in expected.metrics
        if m.id.startswith(("connectivity.components.", "reachability.", "topology."))
        or m.id == "connectivity.multiplicity"
    }
    for identifier, target in expected.targets.metrics.items():
        if identifier in targets:
            targets[identifier] = targets[identifier].model_copy(update={"value": target})
        else:
            targets[identifier] = Metric(
                id=identifier, definition="Authored metric target", value=target, status="exact"
            )
    for identifier, metric in targets.items():
        other = measured.get(identifier)
        if metric.status in {"not_applicable", "not_measured", "ontology_required"}:
            continue
        if not other or other.status not in {"exact", "estimated"}:
            report["unsupported_targets"].append(
                {
                    "metric": identifier,
                    "reason": "Target metric could not be measured on the generated graph.",
                }
            )
            continue
        lhs, rhs = metric.value, other.value
        tol = expected.targets.tolerances.get(identifier, tolerance)

        def record(suffix, value, distance="absolute"):
            checks.append(
                {
                    "metric": identifier + suffix,
                    "distance": distance,
                    "value": value,
                    "tolerance": tol,
                    "passed": value <= tol,
                    "expected_status": metric.status,
                    "actual_status": other.status,
                }
            )

        if identifier == "connectivity.multiplicity":
            for key in ("self_loops", "parallel_excess", "distinct_nonloop_pairs"):
                record(
                    "." + key,
                    abs(lhs[key] * effective_scale - rhs[key]) / max(lhs[key] * effective_scale, 1),
                    "relative count error",
                )
        elif identifier.startswith("connectivity.components."):
            record(
                ".sizes",
                _tv(_frequencies(lhs["sizes"], key="size"), _frequencies(rhs["sizes"], key="size")),
                "total_variation",
            )
            if lhs["largest_share"] is not None and rhs["largest_share"] is not None:
                record(".largest_share", abs(lhs["largest_share"] - rhs["largest_share"]))
        elif identifier == "reachability.paths":
            record(
                ".finite_distances",
                _tv(
                    _frequencies(lhs["distances"], key="distance"),
                    _frequencies(rhs["distances"], key="distance"),
                ),
                "total_variation",
            )
            if lhs["unreachable_fraction"] is not None and rhs["unreachable_fraction"] is not None:
                record(
                    ".unreachable_fraction",
                    abs(lhs["unreachable_fraction"] - rhs["unreachable_fraction"]),
                )
        elif identifier == "reachability.neighborhoods":
            for k, rows in lhs.items():
                record(
                    f".k{k}",
                    _tv(
                        _frequencies(rows, key="reachable"),
                        _frequencies(rhs.get(k, []), key="reachable"),
                    ),
                    "total_variation",
                )
        elif identifier == "topology.clustering":
            if lhs["mean"] is not None and rhs["mean"] is not None:
                record(".mean", abs(lhs["mean"] - rhs["mean"]))
        elif (
            identifier == "topology.triangles"
            and isinstance(lhs, (int, float))
            and isinstance(rhs, (int, float))
        ):
            record(
                "",
                abs(lhs * effective_scale - rhs) / max(lhs * effective_scale, 1),
                "relative count error",
            )
        elif (
            identifier == "topology.communities" and isinstance(lhs, dict) and isinstance(rhs, dict)
        ):
            record(".modularity", abs(lhs["modularity"] - rhs["modularity"]))
            total_a = sum(lhs["sizes"].values())
            total_b = sum(rhs["sizes"].values())
            record(
                ".sizes",
                _tv(
                    {str(k): v / total_a for k, v in lhs["sizes"].items()},
                    {str(k): v / total_b for k, v in rhs["sizes"].items()},
                ),
                "total_variation",
            )
        elif identifier == "topology.directed_triads":
            a = np.array(lhs.get("proportions", lhs.get("counts", [])), dtype=float)
            b = np.array(rhs.get("proportions", rhs.get("counts", [])), dtype=float)
            if a.sum() and b.sum():
                record("", float(np.abs(a / a.sum() - b / b.sum()).sum() / 2), "total_variation")
        elif identifier == "reachability.layers":
            a = {str(k): v / sum(lhs.values()) for k, v in lhs.items()} if sum(lhs.values()) else {}
            b = {str(k): v / sum(rhs.values()) for k, v in rhs.items()} if sum(rhs.values()) else {}
            record("", _tv(a, b), "total_variation")
        else:
            report["unsupported_targets"].append(
                {
                    "metric": identifier,
                    "reason": "No compatible distance implemented for this metric target.",
                    "expected": lhs,
                    "actual": rhs,
                }
            )
    semantic = next((m.value for m in actual.metrics if m.id == "semantics.rules"), [])
    if isinstance(semantic, list):
        report["hard_violations"] = [
            r for r in semantic if r["hard"] and (r["violations"] or r["status"] != "measured")
        ]
    report["unmet_targets"] = [c for c in checks if not c["passed"]]
    if report["hard_violations"]:
        report["status"] = "failed"
    elif report["unmet_targets"] or report["unsupported_targets"]:
        report["status"] = "partial"
    return report
