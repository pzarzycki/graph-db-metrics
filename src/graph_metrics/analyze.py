"""Measurement-first characterization of a disk-backed directed property multigraph."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import igraph as ig
import numpy as np

from .io import dumps, encode, write_profile, write_query_table, write_table
from .models import Metric, NodeSpec, Observations, Profile, PropertySpec, RelationshipSpec
from .store import GraphStore


def entropy(counts):
    values = np.asarray(list(counts), dtype=float)
    if not len(values) or values.sum() == 0:
        return 0.0
    p = values / values.sum()
    return float(-(p[p > 0] * np.log2(p[p > 0])).sum())


def _type(value):
    if isinstance(value, dict) and "__neo4j_type__" in value:
        return value["__neo4j_type__"]
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    return "unsupported"


def _distribution_shape(value):
    if not isinstance(value, str):
        return None
    return re.sub(r"\d+", "#", re.sub(r"[^\W\d_]+", "A", value))[:200]


def analyze_store(
    store: GraphStore,
    output_dir: Path,
    *,
    source=None,
    seed=42,
    sample_roots=128,
    exact_threshold=2000,
    table_threshold=128,
    max_dependency_pairs=32,
) -> Profile:
    output_dir.mkdir(parents=True, exist_ok=True)
    db = store.db
    n, m = store.counts()
    profile = Profile(
        source=source or {"system": "staged", "consistency": "immutable_staging"},
        observations=Observations(node_count=n, relationship_count=m),
    )
    scope = profile.scope

    def metric(
        identifier,
        definition,
        value=None,
        *,
        status="exact",
        denominator=None,
        estimator=None,
        uncertainty=None,
        reason=None,
        extra_scope=None,
    ):
        item = Metric(
            id=identifier,
            definition=definition,
            scope=scope | (extra_scope or {}),
            denominator=denominator,
            value=encode(value),
            status=status,
            estimator=estimator or {"method": "exact staged scan"},
            uncertainty=uncertainty,
            reason=reason,
        )
        profile.metrics.append(item)
        return item

    def dist(identifier, rows):
        if len(rows) <= table_threshold:
            return rows
        name = hashlib.sha256(identifier.encode()).hexdigest()[:20]
        return write_table(rows, output_dir / "distributions" / f"{name}.parquet", output_dir)

    metric(
        "population.counts",
        "Number of nodes and directed relationships in the selected graph",
        {"nodes": n, "relationships": m},
    )
    db.execute("""CREATE OR REPLACE TEMP TABLE degree AS
        SELECT n.id,n.cohort,coalesce(o.k,0) AS kout,coalesce(i.k,0) AS kin FROM nodes n
        LEFT JOIN (SELECT source,count(*) k FROM edges GROUP BY source) o ON o.source=n.id
        LEFT JOIN (SELECT target,count(*) k FROM edges GROUP BY target) i ON i.target=n.id""")
    node_specs = []
    for cohort, labels, count in db.execute(
        "SELECT cohort,labels,count(*) FROM nodes GROUP BY cohort,labels ORDER BY cohort"
    ).fetchall():
        patterns = [
            {"keys": json.loads(keys), "count": c}
            for keys, c in db.execute(
                "SELECT to_json(list_sort(json_keys(properties))) k,count(*) FROM nodes WHERE cohort=? GROUP BY k ORDER BY k",
                [cohort],
            ).fetchall()
        ]
        degrees = [
            {"in": a, "out": b, "count": c}
            for a, b, c in db.execute(
                "SELECT kin,kout,count(*) FROM degree WHERE cohort=? GROUP BY kin,kout ORDER BY kin,kout",
                [cohort],
            ).fetchall()
        ]
        spec = NodeSpec(
            id=cohort,
            labels=json.loads(labels),
            count=count,
            patterns=dist(cohort + ".patterns", patterns),
            degrees=dist(cohort + ".degrees", degrees),
        )
        node_specs.append(spec)
        metric(
            f"schema.nodes.{cohort}.patterns",
            "Complete property-presence signature frequencies within an exact label-set cohort",
            {"distribution": spec.patterns, "entropy_bits": entropy(p["count"] for p in patterns)},
            denominator=count,
            extra_scope={"node_cohort": cohort},
        )
    profile.observations.nodes = node_specs
    metric(
        "population.label_sets",
        "Frequency of complete node label sets, including unlabeled nodes",
        [{"cohort": x.id, "labels": x.labels, "count": x.count} for x in node_specs],
        denominator=n,
    )
    label_counts = Counter()
    label_pairs = Counter()
    for spec in node_specs:
        for label in spec.labels:
            label_counts[label] += spec.count
            for other in spec.labels:
                label_pairs[label, other] += spec.count
    metric(
        "population.labels",
        "Per-label prevalence and conditional label implication probabilities",
        {
            "counts": dict(label_counts),
            "implications": [
                {"given": a, "label": b, "probability": c / label_counts[a]}
                for (a, b), c in sorted(label_pairs.items())
            ],
        },
        denominator=n,
    )
    joint = [
        {"in": a, "out": b, "count": c}
        for a, b, c in db.execute(
            "SELECT kin,kout,count(*) FROM degree GROUP BY kin,kout ORDER BY kin,kout"
        ).fetchall()
    ]
    isolated = db.execute("SELECT count(*) FROM degree WHERE kin=0 AND kout=0").fetchone()[0]
    metric(
        "connectivity.degrees",
        "Joint empirical in/out degree distribution; every self-loop contributes one to each degree",
        {
            "distribution": dist("joint_degree", joint),
            "isolates": isolated,
            "isolate_fraction": isolated / n if n else None,
            "sum_in": m,
            "sum_out": m,
        },
        denominator=n,
    )
    loops = db.execute("SELECT count(*) FROM edges WHERE source=target").fetchone()[0]
    distinct_pairs = db.execute(
        "SELECT count(*) FROM (SELECT DISTINCT source,target FROM edges WHERE source<>target)"
    ).fetchone()[0]
    parallel = db.execute(
        "SELECT coalesce(sum(c-1),0) FROM (SELECT count(*) c FROM edges GROUP BY source,target,type)"
    ).fetchone()[0]
    metric(
        "connectivity.multiplicity",
        "Distinct ordered nonloop pair occupancy and mean multiplicity; parallel excess is counted within endpoint/type groups",
        {
            "self_loops": loops,
            "parallel_excess": parallel,
            "distinct_nonloop_pairs": distinct_pairs,
            "pair_occupancy": distinct_pairs / (n * (n - 1)) if n > 1 else None,
            "mean_pair_multiplicity": (m - loops) / distinct_pairs if distinct_pairs else None,
        },
        denominator={"ordered_pairs": n * (n - 1)},
    )
    rel_specs = []
    signatures = db.execute("""SELECT e.cohort,e.type,s.cohort,t.cohort,count(*) FROM edges e
        JOIN nodes s ON s.id=e.source JOIN nodes t ON t.id=e.target
        GROUP BY e.cohort,e.type,s.cohort,t.cohort ORDER BY e.cohort""").fetchall()
    for cohort, kind, src, dst, count in signatures:
        degree_sets = []
        for endpoint, node_cohort in [("source", src), ("target", dst)]:
            rows = db.execute(
                f"""SELECT coalesce(k,0),count(*) FROM nodes n
                LEFT JOIN (SELECT {endpoint} id,count(*) k FROM edges WHERE cohort=? GROUP BY {endpoint}) d USING(id)
                WHERE n.cohort=? GROUP BY coalesce(k,0) ORDER BY coalesce(k,0)""",
                [cohort, node_cohort],
            ).fetchall()
            degree_sets.append(
                dist(cohort + endpoint, [{"degree": k, "count": c} for k, c in rows])
            )
        patterns = [
            {"keys": json.loads(keys), "count": c}
            for keys, c in db.execute(
                "SELECT to_json(list_sort(json_keys(properties))) k,count(*) FROM edges WHERE cohort=? GROUP BY k ORDER BY k",
                [cohort],
            ).fetchall()
        ]
        rel_specs.append(
            RelationshipSpec(
                id=cohort,
                type=kind,
                source=src,
                target=dst,
                count=count,
                out_degrees=degree_sets[0],
                in_degrees=degree_sets[1],
                patterns=dist(cohort + ".patterns", patterns),
            )
        )
        metric(
            f"connectivity.typed.{cohort}",
            "Directional degree distributions within an edge-signature cohort, including eligible zero-degree nodes",
            {"out": degree_sets[0], "in": degree_sets[1]},
            denominator={
                "out": next(x.count for x in node_specs if x.id == src),
                "in": next(x.count for x in node_specs if x.id == dst),
            },
        )
    profile.observations.relationships = rel_specs
    metric(
        "relations.signatures",
        "Frequency of (complete source label set, relationship type, complete target label set)",
        [
            {
                "cohort": x.id,
                "source": x.source,
                "type": x.type,
                "target": x.target,
                "count": x.count,
            }
            for x in rel_specs
        ],
        denominator=m,
    )
    db.execute("""CREATE OR REPLACE TEMP TABLE incidence AS SELECT id,cohort,sum(kout)::BIGINT kout,sum(kin)::BIGINT kin
        FROM (SELECT source id,cohort,count(*) kout,0 kin FROM edges GROUP BY source,cohort
        UNION ALL SELECT target id,cohort,0 kout,count(*) kin FROM edges GROUP BY target,cohort)
        GROUP BY id,cohort""")
    db.execute("""CREATE OR REPLACE TEMP TABLE role_maps AS SELECT id,
        to_json(map(list(cohort ORDER BY cohort),list(json_object('out',kout,'in',kin) ORDER BY cohort))) roles
        FROM incidence GROUP BY id""")
    for spec in node_specs:
        rows = [
            {"roles": json.loads(role), "count": count}
            for role, count in db.execute(
                "SELECT coalesce(r.roles,'{}') vector,count(*) FROM nodes n LEFT JOIN role_maps r USING(id) WHERE n.cohort=? GROUP BY vector ORDER BY vector",
                [spec.id],
            ).fetchall()
        ]
        spec.roles = dist(spec.id + ".roles", rows)
    metric(
        "relations.roles",
        "Joint typed directional role-vector distribution conditional on complete node label set",
        {spec.id: spec.roles for spec in node_specs},
        denominator=n,
    )

    # Extract typed property records into a disk-backed long table for exact grouped frequencies.
    db.execute(
        "CREATE OR REPLACE TEMP TABLE props(entity VARCHAR,cohort VARCHAR,id BIGINT,key VARCHAR,type VARCHAR,value VARCHAR,number DOUBLE)"
    )
    for entity, iterator in [("node", store.iter_nodes), ("relationship", store.iter_edges)]:
        for batch in iterator():
            rows = []
            for record in batch:
                for key, value in record["properties"].items():
                    encoded = encode(value)
                    kind = _type(encoded)
                    number = (
                        float(value)
                        if kind in ("integer", "float")
                        and isinstance(value, (int, float))
                        and math.isfinite(value)
                        else None
                    )
                    rows.append(
                        (entity, record["cohort"], record["id"], key, kind, dumps(value), number)
                    )
            if rows:
                db.execute(
                    "INSERT INTO props SELECT unnest(?),unnest(?),unnest(?),unnest(?),unnest(?),unnest(?),unnest(?)",
                    [list(x) for x in zip(*rows)],
                )
    for entity, specs in [("node", node_specs), ("relationship", rel_specs)]:
        for spec in specs:
            key_entropies = []
            for key, present in db.execute(
                "SELECT key,count(*) FROM props WHERE entity=? AND cohort=? GROUP BY key ORDER BY key",
                [entity, spec.id],
            ).fetchall():
                parameters = [entity, spec.id, key]
                distinct, value_entropy = db.execute(
                    """SELECT count(*),coalesce(-sum(c/?*log2(c/?)),0)
                    FROM (SELECT count(*)::DOUBLE c FROM props WHERE entity=? AND cohort=? AND key=? GROUP BY value)""",
                    [present, present, *parameters],
                ).fetchone()
                if distinct <= table_threshold:
                    values = [
                        {"value": json.loads(v), "count": c}
                        for v, c in db.execute(
                            "SELECT value,count(*) FROM props WHERE entity=? AND cohort=? AND key=? GROUP BY value ORDER BY value",
                            parameters,
                        ).fetchall()
                    ]
                    frequencies = values
                else:
                    name = hashlib.sha256(dumps(parameters).encode()).hexdigest()[:20]
                    frequencies = write_query_table(
                        db,
                        """SELECT json_object('value',json(value),'count',count(*))::VARCHAR record
                        FROM props WHERE entity=? AND cohort=? AND key=? GROUP BY value ORDER BY value""",
                        parameters,
                        output_dir / "distributions" / f"{name}.parquet",
                        output_dir,
                        distinct,
                    )
                    values = []
                types = dict(
                    db.execute(
                        "SELECT type,count(*) FROM props WHERE entity=? AND cohort=? AND key=? GROUP BY type ORDER BY type",
                        [entity, spec.id, key],
                    ).fetchall()
                )
                common = next(iter(types)) if len(types) == 1 else "mixed"
                distribution = {"kind": "empirical"}
                if isinstance(frequencies, list):
                    distribution.update(
                        values=[v["value"] for v in values], weights=[v["count"] for v in values]
                    )
                else:
                    distribution["table"] = frequencies.model_dump()
                spec.properties[key] = PropertySpec(
                    type=common,
                    presence=present / spec.count if spec.count else 0,
                    distribution=distribution,
                )
                q = present / spec.count if spec.count else 0
                key_entropies.append(entropy([present, spec.count - present]))
                summary = {
                    "present": present,
                    "missing": spec.count - present,
                    "presence": q,
                    "type_mix": types,
                    "distinct": distinct,
                    "entropy_bits": value_entropy,
                    "distribution": frequencies,
                    "possible_identifier": distinct == present and present > 1,
                }
                numeric = db.execute(
                    "SELECT quantile_cont(number, [0,0.01,0.05,0.25,0.5,0.75,0.95,0.99,1]),stddev_pop(number),count(number) FROM props WHERE entity=? AND cohort=? AND key=?",
                    [entity, spec.id, key],
                ).fetchone()
                if numeric[2]:
                    summary["numeric"] = {
                        "quantile_levels": [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1],
                        "quantiles": numeric[0],
                        "stddev": numeric[1],
                        "finite_count": numeric[2],
                    }
                strings = dict(
                    db.execute(
                        "SELECT length(json_extract_string(value,'$')),count(*) FROM props WHERE entity=? AND cohort=? AND key=? AND type='string' GROUP BY 1 ORDER BY 1",
                        parameters,
                    ).fetchall()
                )
                format_rows = [
                    {"format": f, "count": c}
                    for f, c in db.execute(
                        """SELECT left(regexp_replace(regexp_replace(json_extract_string(value,'$'),'[a-zA-Z]+','A','g'),'[0-9]+','#','g'),200),count(*)
                    FROM props WHERE entity=? AND cohort=? AND key=? AND type='string' GROUP BY 1 ORDER BY 1""",
                        parameters,
                    ).fetchall()
                ]
                lists = dict(
                    db.execute(
                        "SELECT json_array_length(value),count(*) FROM props WHERE entity=? AND cohort=? AND key=? AND type='list' GROUP BY 1 ORDER BY 1",
                        parameters,
                    ).fetchall()
                )
                elements = (
                    dict(
                        db.execute(
                            """SELECT coalesce(json_extract_string(j.value,'/__neo4j_type__'),json_type(j.value)),count(*)
                    FROM props p,json_each(p.value) j WHERE p.entity=? AND p.cohort=? AND p.key=? AND p.type='list' GROUP BY 1 ORDER BY 1""",
                            parameters,
                        ).fetchall()
                    )
                    if lists
                    else {}
                )
                if strings:
                    summary["strings"] = {
                        "lengths": strings,
                        "formats": dist(f"{entity}.{spec.id}.{key}.formats", format_rows),
                    }
                if lists:
                    summary["lists"] = {"lengths": dict(lists), "element_types": dict(elements)}
                metric(
                    f"properties.{entity}.{spec.id}.{key}",
                    "Conditional value distribution given property presence and entity cohort; type mixtures remain separate",
                    summary,
                    denominator=spec.count,
                    extra_scope={"entity": entity, "cohort": spec.id, "property": key},
                )
            patterns = spec.patterns if isinstance(spec.patterns, list) else []
            if not patterns:
                table = "nodes" if entity == "node" else "edges"
                counts = [
                    r[0]
                    for r in db.execute(
                        f"SELECT count(*) FROM {table} WHERE cohort=? GROUP BY to_json(list_sort(json_keys(properties)))",
                        [spec.id],
                    ).fetchall()
                ]
            else:
                counts = [p["count"] for p in patterns]
            joint_h = entropy(counts)
            metric(
                f"schema.{entity}.{spec.id}.dependence",
                "Joint presence entropy and total correlation (sum of marginal binary entropies minus joint entropy)",
                {
                    "joint_entropy_bits": joint_h,
                    "total_correlation_bits": max(0, sum(key_entropies) - joint_h),
                },
                denominator=spec.count,
            )
    _dependencies(store, profile, metric, max_dependency_pairs, seed, dist)
    _topology(store, profile, metric, dist, seed, sample_roots, exact_threshold)
    # SQL bounds samples before materializing values; cohort-stratified examples include rare types.
    sample_rows = db.execute(
        """SELECT cohort,labels,properties FROM nodes
        QUALIFY row_number() OVER(PARTITION BY cohort ORDER BY hash(id,?)) <= 3
        ORDER BY hash(cohort,properties,?) LIMIT 30""",
        [seed, seed],
    ).fetchall()
    profile.samples["nodes"] = [
        {"cohort": c, "labels": json.loads(labels), "properties": json.loads(props)}
        for c, labels, props in sample_rows
    ]
    profile.samples["relationships"] = (
        [
            {"cohort": r["cohort"], "type": r["type"], "properties": encode(r["properties"])}
            for batch in store.iter_edges(20)
            for r in batch
        ][:20]
        if m < 10000
        else [
            {"cohort": c, "type": t, "properties": json.loads(p)}
            for c, t, p in db.execute(
                "SELECT cohort,type,properties FROM edges ORDER BY hash(id,?) LIMIT 20", [seed]
            ).fetchall()
        ]
    )
    metric(
        "semantics.conceptual_diversity",
        "Concept-distance dispersion requires an aligned ontology and concept mapping",
        status="ontology_required",
        reason="No declared ontology supplied; label names do not establish semantic distance.",
    )
    metric(
        "semantics.rules",
        "Eligible rule tests and violation counts require a semantic contract",
        status="ontology_required",
        reason="Attach declared rules or explicitly attributed inferred proposals.",
    )
    write_profile(profile, output_dir / "profile.yaml")
    return profile


def _dependencies(store, profile, metric, max_pairs, seed, dist):
    db = store.db
    tested = 0
    for spec in profile.observations.nodes:
        eligible_keys = db.execute(
            """SELECT key,count(DISTINCT value) distinct_count,count(*) present
            FROM props WHERE entity='node' AND cohort=? GROUP BY key
            HAVING distinct_count<=100 AND distinct_count<=present/3 ORDER BY key LIMIT 32""",
            [spec.id],
        ).fetchall()
        keys = [key for key, _, _ in eligible_keys]
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                if tested >= max_pairs:
                    break
                tested += 1
                contingency = db.execute(
                    """SELECT a.value,b.value,count(*) FROM props a JOIN props b
                    ON a.entity=b.entity AND a.cohort=b.cohort AND a.id=b.id
                    WHERE a.entity='node' AND a.cohort=? AND a.key=? AND b.key=?
                    GROUP BY a.value,b.value""",
                    [spec.id, a, b],
                ).fetchall()
                eligible = sum(c for _, _, c in contingency)
                ca = Counter()
                cb = Counter()
                for va, vb, c in contingency:
                    ca[va] += c
                    cb[vb] += c
                # Unique IDs and sparse categories do not yield an interpretable plug-in MI.
                if (
                    not eligible
                    or len(ca) > min(100, eligible // 3)
                    or len(cb) > min(100, eligible // 3)
                ):
                    continue
                mi = sum(
                    c / eligible * math.log2(c * eligible / (ca[va] * cb[vb]))
                    for va, vb, c in contingency
                )
                presence = db.execute(
                    """SELECT sum(CASE WHEN json_exists(properties,?) AND json_exists(properties,?) THEN 1 ELSE 0 END)
                    FROM nodes WHERE cohort=?""",
                    [
                        '$."' + a.replace('"', '\\"') + '"',
                        '$."' + b.replace('"', '\\"') + '"',
                        spec.id,
                    ],
                ).fetchone()[0]
                metric(
                    f"dependencies.{spec.id}.{a}.{b}",
                    "Plug-in mutual information on jointly present categorical values; association is not causality",
                    {
                        "mutual_information_bits": mi,
                        "joint_presence": presence,
                        "contingency": [
                            {"a": json.loads(va), "b": json.loads(vb), "count": c}
                            for va, vb, c in contingency
                        ],
                    },
                    status="estimated",
                    denominator=eligible,
                    estimator={
                        "method": "categorical plug-in MI",
                        "missingness": "complete cases",
                        "selection": "lexicographic eligible key pairs",
                        "seed": seed,
                    },
                    uncertainty={
                        "confidence_interval": None,
                        "reason": "No iid assumption for graph-coupled records; finite-sample MI is biased.",
                    },
                )
        if tested >= max_pairs:
            break
    metric(
        "dependencies.coverage",
        "Number and policy of targeted conditional association tests",
        {
            "tested": tested,
            "pair_limit": max_pairs,
            "policy": "within exact label sets; exclude sparse/high-cardinality pairs; complete cases",
        },
    )
    numeric_pairs = []
    for spec in profile.observations.nodes:
        keys = [
            k
            for k, v in spec.properties.items()
            if v.type in ("integer", "float")
            and k.lower() not in {"id", "uuid"}
            and not k.lower().endswith("_id")
        ]
        for a, b in itertools.combinations(sorted(keys), 2):
            count, correlation = db.execute(
                """SELECT count(*),corr(a.number,b.number) FROM props a JOIN props b
                ON a.entity=b.entity AND a.cohort=b.cohort AND a.id=b.id
                WHERE a.entity='node' AND a.cohort=? AND a.key=? AND b.key=? AND a.number IS NOT NULL AND b.number IS NOT NULL""",
                [spec.id, a, b],
            ).fetchone()
            numeric_pairs.append(
                {
                    "cohort": spec.id,
                    "a": a,
                    "b": b,
                    "complete_cases": count,
                    "pearson_correlation": correlation
                    if correlation is not None and math.isfinite(correlation)
                    else None,
                }
            )
            if len(numeric_pairs) >= min(8, max_pairs):
                break
        if len(numeric_pairs) >= min(8, max_pairs):
            break
    metric(
        "dependencies.numeric",
        "Finite-population Pearson associations within exact label-set cohorts, excluding missing/nonfinite values and ID-like fields",
        numeric_pairs,
        denominator={
            f"{p['cohort']}.{p['a']}.{p['b']}": p["complete_cases"] for p in numeric_pairs
        },
        extra_scope={"interpretation": "association is not causality"},
    )
    # Degree distributions by property category expose topology/attribute coupling without causal claims.
    coupling = []
    for spec in profile.observations.nodes:
        for key, prop in spec.properties.items():
            distinct = db.execute(
                "SELECT count(DISTINCT value) FROM props WHERE entity='node' AND cohort=? AND key=?",
                [spec.id, key],
            ).fetchone()[0]
            if 1 < distinct <= 20:
                rows = db.execute(
                    """SELECT p.value,count(*),avg(d.kin),avg(d.kout) FROM props p JOIN degree d USING(id)
                    WHERE p.entity='node' AND p.cohort=? AND p.key=? GROUP BY p.value ORDER BY p.value""",
                    [spec.id, key],
                ).fetchall()
                item = {
                    "cohort": spec.id,
                    "property": key,
                    "groups": [
                        {"value": json.loads(v), "count": c, "mean_in": a, "mean_out": b}
                        for v, c, a, b in rows
                    ],
                }
                if prop.presence == 1:
                    role_rows = db.execute(
                        """SELECT p.value,coalesce(r.roles,'{}'),count(*) FROM props p
                        LEFT JOIN role_maps r ON r.id=p.id WHERE p.entity='node' AND p.cohort=? AND p.key=?
                        GROUP BY p.value,r.roles ORDER BY p.value,r.roles""",
                        [spec.id, key],
                    ).fetchmany(10001)
                    if len(role_rows) > 10000:
                        item["role_distribution_status"] = "not_measured"
                        item["role_distribution_reason"] = (
                            "Joint table exceeds the 10,000-row per-property budget; conditional means remain exact."
                        )
                    else:
                        item["role_distribution"] = dist(
                            f"{spec.id}.{key}.role_values",
                            [
                                {"value": json.loads(v), "roles": json.loads(r), "count": c}
                                for v, r, c in role_rows
                            ],
                        )
                coupling.append(item)
                if len(coupling) >= 16:
                    break
        if len(coupling) >= 16:
            break
    metric(
        "dependencies.topology_attributes",
        "Conditional directed-degree means by low-cardinality property value",
        coupling,
        denominator=profile.observations.node_count,
    )


def _topology(store, profile, metric, dist, seed, sample_roots, exact_threshold):
    db = store.db
    n, m = store.counts()
    db.execute(
        "CREATE OR REPLACE TEMP TABLE node_index AS SELECT id,row_number() OVER(ORDER BY id)-1 idx FROM nodes"
    )
    endpoints = db.execute(
        "SELECT s.idx AS src,t.idx AS dst FROM edges e JOIN node_index s ON s.id=e.source JOIN node_index t ON t.id=e.target ORDER BY e.id"
    ).fetchnumpy()
    pairs = np.column_stack((endpoints["src"], endpoints["dst"])) if m else []
    graph = ig.Graph(n=n, edges=pairs, directed=True)
    for mode in ("weak", "strong"):
        sizes = graph.connected_components(mode=mode).sizes()
        frequencies = Counter(sizes)
        metric(
            f"connectivity.components.{mode}",
            f"{mode.capitalize()} component size frequencies, retaining isolated nodes",
            {
                "count": len(sizes),
                "sizes": [{"size": k, "count": v} for k, v in sorted(frequencies.items())],
                "largest_share": max(sizes) / n if n else None,
            },
            denominator=n,
        )
    simple = graph.as_undirected(mode="collapse")
    simple.simplify(multiple=True, loops=True)
    topology_exact = n <= exact_threshold or n < 3
    triangle_estimate = 0.0
    if topology_exact:
        clustering = simple.transitivity_local_undirected(mode="zero")
        cluster_estimator = {"method": "exact local triangle incidence"}
    else:
        rng = np.random.default_rng(seed)
        sampled = np.sort(rng.choice(n, size=min(n, sample_roots), replace=False))
        clustering = []
        triangle_incidence = []
        for vertex in sampled:
            neighbors = simple.neighbors(int(vertex))
            wedges = math.comb(len(neighbors), 2)
            if not wedges:
                clustering.append(0.0)
                triangle_incidence.append(0.0)
                continue
            if wedges <= 256:
                pairs_to_check = itertools.combinations(neighbors, 2)
                checked = wedges
            else:
                pairs_to_check = (rng.choice(neighbors, size=2, replace=False) for _ in range(256))
                checked = 256
            closed = sum(simple.are_adjacent(int(a), int(b)) for a, b in pairs_to_check)
            probability = closed / checked
            clustering.append(probability)
            triangle_incidence.append(probability * wedges)
        triangle_estimate = (
            float(n * np.mean(triangle_incidence) / 3) if triangle_incidence else 0.0
        )
        cluster_estimator = {
            "method": "uniform roots with up to 256 uniform neighbor pairs per root",
            "sample_size": len(sampled),
            "seed": seed,
        }
    metric(
        "topology.clustering",
        "Local undirected clustering after collapsing parallel edges and removing self-loops",
        {
            "mean": float(np.mean(clustering)) if n else None,
            "distribution": dist(
                "clustering",
                [{"value": k, "count": v} for k, v in sorted(Counter(clustering).items())],
            ),
        },
        denominator=len(clustering),
        status="exact" if topology_exact else "estimated",
        estimator=cluster_estimator,
        uncertainty=None
        if topology_exact
        else {
            "confidence_interval": None,
            "reason": "Two-stage root/wedge estimator; interval calibration is not implemented.",
        },
        extra_scope={"direction": "undirected", "multiplicity": "collapse", "loops": "remove"},
    )
    # Triangle counts are O(sum degree^2) on pathological hubs; sample induced neighborhoods at scale.
    if topology_exact:
        triangles = len(simple.list_triangles())
        metric(
            "topology.triangles",
            "Number of unordered triangles in the loopless simple undirected projection",
            triangles,
            denominator=n,
            extra_scope={"direction": "undirected", "multiplicity": "collapse"},
        )
        census = list(graph.copy().simplify(multiple=True, loops=True).triad_census())
        metric(
            "topology.directed_triads",
            "Induced three-node directed motif census on a loopless simple directed projection",
            {
                "counts": census,
                "order": [
                    "003",
                    "012",
                    "102",
                    "021D",
                    "021U",
                    "021C",
                    "111D",
                    "111U",
                    "030T",
                    "030C",
                    "201",
                    "120D",
                    "120U",
                    "120C",
                    "210",
                    "300",
                ],
            },
            denominator=math.comb(n, 3) if n >= 3 else 0,
            extra_scope={"multiplicity": "collapse", "loops": "remove"},
        )
    else:
        metric(
            "topology.triangles",
            "Horvitz-Thompson triangle total from sampled local triangle incidences, divided by three",
            triangle_estimate,
            status="estimated",
            denominator=n,
            estimator=cluster_estimator,
            uncertainty={
                "confidence_interval": None,
                "reason": "Two-stage root/wedge estimator; interval calibration is not implemented.",
            },
            extra_scope={"direction": "undirected", "multiplicity": "collapse", "loops": "remove"},
        )
        rng = np.random.default_rng(seed)
        samples = min(10000, math.comb(n, 3))
        census = Counter()
        for _ in range(samples):
            chosen = rng.choice(n, size=3, replace=False)
            census[
                tuple(
                    graph.induced_subgraph(chosen)
                    .simplify(multiple=True, loops=True)
                    .triad_census()
                )
            ] += 1
        estimates = np.zeros(16)
        for counts, frequency in census.items():
            estimates += np.asarray(counts) * frequency / samples
        metric(
            "topology.directed_triads",
            "Estimated proportions of induced directed triad types under uniform vertex-triple sampling",
            {"proportions": estimates.tolist()},
            status="estimated",
            denominator=samples,
            estimator={"method": "uniform vertex triples", "sample_size": samples, "seed": seed},
            uncertainty={
                "method": "marginal Wilson 95% intervals; independent uniform vertex-triple draws",
                "standard_errors": np.sqrt(estimates * (1 - estimates) / samples).tolist(),
                "confidence_intervals95": [
                    [
                        (
                            p
                            + 1.96**2 / (2 * samples)
                            - 1.96 * math.sqrt(p * (1 - p) / samples + 1.96**2 / (4 * samples**2))
                        )
                        / (1 + 1.96**2 / samples),
                        (
                            p
                            + 1.96**2 / (2 * samples)
                            + 1.96 * math.sqrt(p * (1 - p) / samples + 1.96**2 / (4 * samples**2))
                        )
                        / (1 + 1.96**2 / samples),
                    ]
                    for p in estimates
                ],
            },
            extra_scope={"multiplicity": "collapse", "loops": "remove"},
        )
    ig.set_random_number_generator(random.Random(seed))
    if simple.ecount():
        partition = simple.community_leiden(
            objective_function="modularity", resolution=1.0, n_iterations=2
        )
        metric(
            "topology.communities",
            "Leiden community-size frequencies and modularity on a simple undirected projection",
            {"sizes": dict(Counter(partition.sizes())), "modularity": partition.modularity},
            estimator={"method": "Leiden", "seed": seed, "resolution": 1.0, "iterations": 2},
            extra_scope={"direction": "undirected", "multiplicity": "collapse"},
        )
    else:
        metric(
            "topology.communities",
            "Community partition and modularity",
            status="not_applicable",
            reason="No nonloop connections in the simple projection.",
        )
    rng = np.random.default_rng(seed)
    roots = (
        np.arange(n)
        if n <= exact_threshold
        else np.sort(rng.choice(n, size=min(sample_roots, n), replace=False))
    )
    reachable = []
    distances = Counter()
    expansion = {k: [] for k in (1, 2, 3, 5)}
    layers = Counter()
    for root in roots:
        d = np.asarray(graph.distances(source=[int(root)], mode="out")[0])
        finite = d[np.isfinite(d)]
        reachable.append(int(len(finite) - 1))
        distances.update(int(x) for x in finite if x > 0)
        layers.update(int(x) for x in finite)
        for k in expansion:
            expansion[k].append(int((finite <= k).sum()))
    status = "exact" if len(roots) == n else "estimated"
    estimator = {
        "method": "all-root BFS" if status == "exact" else "uniform-root directed BFS",
        "sample_size": len(roots),
        "seed": seed,
        "root_policy": "uniform without replacement",
    }
    unreachable = 1 - sum(reachable) / (len(roots) * (n - 1)) if n > 1 and len(roots) else None
    uncertainty = None
    if status == "estimated" and len(roots) > 1 and n > 1:
        fractions = np.asarray(reachable) / (n - 1)
        se = float(
            np.std(fractions, ddof=1) / np.sqrt(len(roots)) * np.sqrt((n - len(roots)) / (n - 1))
        )
        uncertainty = {
            "method": "root-cluster normal approximation with finite-population correction",
            "reachable_fraction_ci95": [
                max(0, 1 - unreachable - 1.96 * se),
                min(1, 1 - unreachable + 1.96 * se),
            ],
        }
    metric(
        "reachability.paths",
        "Finite directed shortest-path length frequencies and unreachable ordered-pair fraction, excluding self-pairs",
        {
            "distances": [{"distance": k, "count": v} for k, v in sorted(distances.items())],
            "unreachable_fraction": unreachable,
            "sampled_roots": len(roots),
        },
        status=status,
        denominator=len(roots) * max(n - 1, 0),
        estimator=estimator,
        uncertainty=uncertainty,
    )
    metric(
        "reachability.neighborhoods",
        "Distribution of distinct vertices within k outgoing hops, including the root",
        {
            str(k): [{"reachable": a, "count": b} for a, b in sorted(Counter(values).items())]
            for k, values in expansion.items()
        },
        status=status,
        denominator=len(roots),
        estimator=estimator,
        uncertainty=uncertainty,
    )
    metric(
        "reachability.layers",
        "Pooled BFS layer counts for the documented root policy",
        dict(layers),
        status=status,
        denominator=len(roots),
        estimator=estimator,
    )
    metric(
        "topology.hierarchy",
        "Longest directed depth for acyclic relation projections; cyclic relations require a declared root/scope",
        status="not_measured",
        reason="No hierarchy relationship subset/root contract supplied; arbitrary graphs have no canonical depth.",
    )
