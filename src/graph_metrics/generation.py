"""Procedural synthesis into a disk-backed staging graph, before any database writes."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from .io import decode, dumps, loads, read_table, safe_path
from .models import Profile
from .providers import ValueProviders, stable_seed
from .store import GraphStore

SUPPORTED_RULES = {
    "cardinality",
    "required",
    "unique",
    "enum",
    "range",
    "temporal_order",
    "same_property",
    "dag",
    "allowed_signature",
    "functional_dependency",
    "property_dependency",
}


def allocate(weights, total: int) -> np.ndarray:
    """Largest remainders keep population totals exact during scaling."""
    weights = np.asarray(weights, dtype=float)
    if total < 0 or np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError("invalid population weights")
    if not len(weights):
        if total:
            raise ValueError("cannot allocate a population with no cohorts")
        return np.zeros(0, dtype=np.int64)
    if weights.sum() == 0:
        if total:
            raise ValueError("cannot allocate a population from zero weights")
        return np.zeros(len(weights), dtype=np.int64)
    exact = weights / weights.sum() * total
    counts = np.floor(exact).astype(np.int64)
    extra = total - int(counts.sum())
    if extra:
        counts[np.argsort(-(exact - counts), kind="stable")[:extra]] += 1
    return counts


def _counts(specs, scale):
    total = round(sum(spec.count for spec in specs) * scale)
    return {
        spec.id: int(count)
        for spec, count in zip(specs, allocate([spec.count for spec in specs], total), strict=True)
    }


def _rules(profile):
    combined = [*profile.semantics.rules, *profile.targets.rules]
    ids = [rule.id for rule in combined]
    if len(ids) != len(set(ids)):
        raise ValueError("semantic and target rule ids must be distinct")
    for rule in combined:
        if rule.hard and rule.kind not in SUPPORTED_RULES:
            raise ValueError(
                f"unsupported hard rule {rule.id!r}: {rule.kind}; add a supported validator"
            )
    return combined


def _bounds(rules, rel_id, direction, n, m):
    minimum, maximum = 0, m
    for rule in rules:
        params = rule.parameters
        if (
            rule.hard
            and rule.kind == "cardinality"
            and params.get("relationship") == rel_id
            and params.get("direction", "out") == direction
        ):
            minimum = max(minimum, int(params.get("min", 0)))
            maximum = min(maximum, int(params.get("max", m)))
    if minimum < 0 or maximum < minimum or m < minimum * n or m > maximum * n:
        raise ValueError(
            f"infeasible {direction} cardinalities for {rel_id!r}: "
            f"{m} relationships across {n} nodes with bounds [{minimum}, {maximum}]"
        )
    return minimum, maximum


def _sequence(rows, n, m, rng, minimum=0, maximum=None):
    maximum = m if maximum is None else maximum
    if n == 0:
        if m:
            raise ValueError("nonempty relationships require nonempty endpoint cohorts")
        return np.zeros(0, dtype=np.int64)
    if rows:
        degrees = np.asarray([int(r["degree"]) for r in rows], dtype=np.int64)
        if np.any(degrees < 0):
            raise ValueError("degree distributions cannot contain negative degrees")
        counts = allocate([r["count"] for r in rows], n)
        sequence = np.repeat(degrees, counts)
        rng.shuffle(sequence)
        sequence = np.clip(sequence, minimum, maximum)
    else:
        sequence = np.full(n, minimum, dtype=np.int64)
        remaining = m - int(sequence.sum())
        if maximum >= m:
            sequence += rng.multinomial(remaining, np.full(n, 1 / n))
        else:
            fill = min(maximum - minimum, remaining // n)
            sequence += fill
    difference = m - int(sequence.sum())
    # Water filling changes scaled histograms only as much as needed to satisfy incidence.
    while difference:
        eligible = np.flatnonzero(sequence < maximum if difference > 0 else sequence > minimum)
        if not len(eligible):
            raise ValueError("degree targets conflict with required relationship count")
        rng.shuffle(eligible)
        amount = min(abs(difference), len(eligible))
        direction = 1 if difference > 0 else -1
        sequence[eligible[:amount]] += direction
        difference -= direction * amount
    return sequence


def _role_histogram(node, relationship, direction, base_dir):
    rows = read_table(node.roles, base_dir)
    if not rows:
        return []
    result = []
    for row in rows:
        role = row.get("roles", {}).get(relationship, {})
        result.append({"degree": int(role.get(direction, 0)), "count": row["count"]})
    return result


def _pattern_assignments(spec, n, base_dir, seed):
    patterns = read_table(spec.patterns, base_dir)
    if not patterns:
        return None, None
    for row in patterns:
        if not set(row["keys"]).issubset(spec.properties):
            raise ValueError(f"property presence pattern in {spec.id!r} references unknown keys")
    assignment = np.repeat(np.arange(len(patterns)), allocate([r["count"] for r in patterns], n))
    np.random.default_rng(stable_seed(seed, spec.id, "presence")).shuffle(assignment)
    return [set(row["keys"]) for row in patterns], assignment


class _PropertySchedule:
    """Finite categorical counts and presence masks, shuffled independently of batch boundaries."""

    def __init__(self, spec, n, patterns, assignments, providers, seed, required=()):
        self.masks, self.presets, self.generated = {}, {}, {}
        for key, prop in spec.properties.items():
            rng = np.random.default_rng(stable_seed(seed, spec.id, key, "finite-allocation"))
            if key in required:
                mask = None
            elif patterns:
                present_patterns = np.asarray([key in pattern for pattern in patterns], dtype=bool)
                mask = None if np.all(present_patterns) else present_patterns[assignments]
            elif prop.presence == 1:
                mask = None
            else:
                present = int(round(n * prop.presence))
                mask = np.zeros(n, dtype=bool)
                mask[rng.permutation(n)[:present]] = True
            self.masks[key] = mask
            dist = prop.distribution
            kind = dist.get("kind")
            if kind in {"integer", "uniform_int"}:
                low, high = int(dist.get("min", 0)), int(dist.get("max", 100))
                if high < low:
                    raise ValueError("integer range is empty")
                self.generated[key] = rng.integers(low, high, endpoint=True, size=n)
            elif kind == "uniform":
                self.generated[key] = rng.uniform(dist.get("min", 0), dist.get("max", 1), size=n)
            elif kind == "boolean":
                self.generated[key] = rng.random(n) < dist.get("probability", 0.5)
            elif kind == "normal":
                data = rng.normal(dist.get("mean", 0), dist.get("std", 1), size=n)
                data = np.clip(data, dist.get("min", -math.inf), dist.get("max", math.inf))
                self.generated[key] = (
                    np.rint(data).astype(np.int64) if prop.type == "integer" else data
                )
            elif kind in {"id", "unique"} and prop.type in {"integer", "int"}:
                self.generated[key] = np.arange(n, dtype=np.int64) + int(dist.get("start", 0))
            if dist.get("kind") not in {"categorical", "empirical"} or dist.get("exclude_property"):
                continue
            values, weights, _ = providers.categorical(dist)
            eligible = np.arange(n) if mask is None else np.flatnonzero(mask)
            allocations = allocate(weights, len(eligible))
            categories = np.repeat(np.arange(len(values), dtype=np.int64), allocations)
            rng.shuffle(categories)
            sequence = np.full(n, -1, dtype=np.int64)
            sequence[eligible] = categories
            self.presets[key] = (values, sequence)

    def keys(self, index):
        return {key for key, mask in self.masks.items() if mask is None or mask[index]}

    def values(self, index):
        result = {
            key: decode(values[sequence[index]])
            for key, (values, sequence) in self.presets.items()
            if sequence[index] >= 0
        }
        result.update(
            {
                key: values[index].item()
                for key, values in self.generated.items()
                if self.masks[key] is None or self.masks[key][index]
            }
        )
        return result


def _loop_policy(profile, name):
    explicit = getattr(profile.targets, name)
    if explicit is not None:
        return explicit
    fragments = (
        ("loop", "multiplicity") if name == "allow_self_loops" else ("parallel", "multiplicity")
    )
    for metric in profile.metrics:
        if any(fragment in metric.id for fragment in fragments) and metric.status == "exact":
            if isinstance(metric.value, (int, float)):
                return metric.value > 0
            if isinstance(metric.value, dict):
                keys = (
                    ("self_loops", "loops", "count")
                    if name == "allow_self_loops"
                    else ("parallel_excess", "parallel_edges", "excess_edges", "count")
                )
                for key in keys:
                    if key in metric.value:
                        return metric.value[key] > 0
    return False


def _repair_pairing(sources, targets, n, rng, *, loops, parallel, dag, rounds):
    """Endpoint swaps preserve directed degree sequences and typed incidences."""
    m = len(sources)
    for _ in range(max(10, rounds * 10)):
        invalid = np.zeros(m, dtype=bool)
        if not loops:
            invalid |= sources == targets
        if dag:
            invalid |= sources >= targets
        if not parallel and m:
            codes = sources * np.int64(n) + targets
            _, positions, counts = np.unique(codes, return_index=True, return_counts=True)
            duplicated = counts > 1
            if np.any(duplicated):
                order = np.argsort(codes, kind="stable")
                duplicate_positions = np.flatnonzero(codes[order][1:] == codes[order][:-1]) + 1
                invalid[order[duplicate_positions]] = True
        bad = np.flatnonzero(invalid)
        if not len(bad):
            return
        partners = rng.integers(0, m, size=len(bad))
        for left, right in zip(bad, partners, strict=True):
            targets[left], targets[right] = targets[right], targets[left]
    raise ValueError(
        "could not satisfy endpoint restrictions within bounded refinement; "
        "degree targets may be infeasible or require a different structural recipe"
    )


def _community_plan(profile, n, seed):
    metric = next((m for m in profile.metrics if m.id == "topology.communities"), None)
    if not metric or not isinstance(metric.value, dict) or not n:
        return None
    sizes = metric.value.get("sizes", {})
    modularity = metric.value.get("modularity")
    if (
        not isinstance(sizes, dict)
        or not sizes
        or modularity is None
        or not math.isfinite(modularity)
    ):
        return None
    weights = [int(size) for size, count in sizes.items() for _ in range(int(count))]
    if len(weights) < 2:
        return None
    counts = allocate(weights, n)
    membership = np.repeat(np.arange(len(counts)), counts)
    np.random.default_rng(stable_seed(seed, "community-membership")).shuffle(membership)
    desired_cross = float(np.clip(1 - modularity - np.sum((counts / n) ** 2), 0, 1))
    return membership, desired_cross


def _community_pairing(sources, targets, plan, rng, rounds):
    """A bounded, degree-preserving block-mixing refinement for community targets.

    This is a synthesis heuristic; fidelity is decided by re-analysis, not this proxy.
    """
    membership, desired = plan
    if not len(sources) or not rounds:
        return {"status": "not_refined", "reason": "empty edge cohort or zero refinement budget"}
    source_blocks, target_blocks = membership[sources], membership[targets]
    before = float(np.mean(source_blocks != target_blocks))
    if before <= desired:
        return {
            "status": "not_refined",
            "reason": "initial block mixing already meets target proxy",
        }
    source_order = np.argsort(source_blocks, kind="stable")
    target_order = np.argsort(target_blocks, kind="stable")
    source_sizes = np.bincount(source_blocks, minlength=len(membership))
    target_sizes = np.bincount(target_blocks, minlength=len(membership))
    source_offsets = np.concatenate(([0], np.cumsum(source_sizes)))
    target_offsets = np.concatenate(([0], np.cumsum(target_sizes)))
    proposed = np.empty_like(targets)
    unmatched_source, unmatched_target = [], []
    for block in np.flatnonzero(source_sizes + target_sizes):
        si = source_order[source_offsets[block] : source_offsets[block + 1]]
        ti = target_order[target_offsets[block] : target_offsets[block + 1]]
        paired = min(len(si), len(ti))
        proposed[si[:paired]] = targets[ti[:paired]]
        unmatched_source.append(si[paired:])
        unmatched_target.append(targets[ti[paired:]])
    left = np.concatenate(unmatched_source)
    right = np.concatenate(unmatched_target)
    rng.shuffle(right)
    proposed[left] = right
    minimum = float(np.mean(source_blocks != membership[proposed]))
    best = proposed
    best_error = abs(minimum - desired)
    for _ in range(rounds):
        if minimum >= desired:
            break
        # Randomizing only a subset continuously interpolates between strong blocks and random mixing.
        proportion = min(1, max(0, (desired - minimum) / max(1e-9, before - minimum)))
        selected = rng.choice(len(targets), size=round(proportion * len(targets)), replace=False)
        candidate = proposed.copy()
        shuffled = candidate[selected].copy()
        rng.shuffle(shuffled)
        candidate[selected] = shuffled
        error = abs(float(np.mean(source_blocks != membership[candidate])) - desired)
        if error < best_error:
            best, best_error = candidate, error
    if best_error < abs(before - desired):
        targets[:] = best
    return {
        "status": "refined",
        "method": "degree-preserving synthetic community block mixing",
        "rounds": rounds,
        "target_cross_edge_fraction_proxy": desired,
        "before": before,
        "after": float(np.mean(source_blocks != membership[targets])),
        "caveat": "Leiden modularity and actual components must be remeasured; this proxy is not a fidelity guarantee.",
    }


def _lookup_provider(store, ranges):
    tables = {}
    cache = OrderedDict()

    def lookup(cohort, match_property, match_value, property_name):
        if match_property == "__random__":
            start, stop = ranges[cohort]
            if stop == start:
                raise ValueError(f"reference provider needs a nonempty {cohort!r} cohort")
            identifier = start + match_value % (stop - start)
            key = ("reference", cohort, identifier, property_name)
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            row = store.db.execute(
                "SELECT properties FROM nodes WHERE id=?", [identifier]
            ).fetchone()
            if row is None:
                raise ValueError(f"reference provider requires {cohort!r} to be generated first")
            value = loads(row[0])[property_name]
            cache[key] = value
            if len(cache) > 4096:
                cache.popitem(last=False)
            return value
        table_key = (cohort, match_property, property_name)
        if table_key not in tables:
            table = f"provider_lookup_{len(tables)}"
            match_path, value_path = (
                "$." + json.dumps(match_property),
                "$." + json.dumps(property_name),
            )
            store.db.execute(
                f"CREATE TEMP TABLE {table} AS SELECT "
                "json_extract(properties, ?)::VARCHAR AS match, "
                "json_extract(properties, ?)::VARCHAR AS value FROM nodes WHERE cohort = ?",
                [match_path, value_path, cohort],
            )
            duplicate = store.db.execute(
                f"SELECT match FROM {table} GROUP BY match HAVING count(*)>1 LIMIT 1"
            ).fetchone()
            if duplicate:
                raise ValueError(f"lookup key {cohort}.{match_property} must be unique")
            store.db.execute(f"CREATE UNIQUE INDEX {table}_idx ON {table}(match)")
            tables[table_key] = table
        key = (*table_key, dumps(match_value))
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        row = store.db.execute(
            f"SELECT value FROM {tables[table_key]} WHERE match = ?", [dumps(match_value)]
        ).fetchone()
        if row is None or row[0] is None:
            raise ValueError(f"lookup has no value for {cohort}.{match_property}={match_value!r}")
        value = loads(row[0])
        cache[key] = value
        if len(cache) > 4096:
            cache.popitem(last=False)
        return value

    return lookup


def _dependencies(distribution):
    dependencies = set()
    if isinstance(distribution, dict):
        if distribution.get("kind") in {"lookup", "reference"}:
            dependencies.add(distribution["cohort"])
        for value in distribution.values():
            dependencies.update(_dependencies(value))
    elif isinstance(distribution, list):
        for value in distribution:
            dependencies.update(_dependencies(value))
    return dependencies


def _node_order(specs):
    pending = list(specs)
    ready, complete = [], set()
    while pending:
        candidates = [
            node
            for node in pending
            if set().union(
                *(_dependencies(prop.distribution) for prop in node.properties.values()), set()
            )
            <= complete
        ]
        if not candidates:
            raise ValueError("cyclic or unknown cross-cohort lookup dependencies")
        for node in candidates:
            pending.remove(node)
            ready.append(node)
            complete.add(node.id)
    return ready


def _generation_specs(profile):
    """Compile observed dependencies into procedural recipes without editing observations."""
    raw_nodes, raw_relationships = profile.generation_specs()
    nodes = [node.model_copy(deep=True) for node in raw_nodes]
    relationships = [rel.model_copy(deep=True) for rel in raw_relationships]
    overrides = profile.preparation.recipe.get("property_overrides", [])
    if not isinstance(overrides, list):
        raise ValueError("recipe property_overrides must be a list")
    recipe_properties = []
    for override in overrides:
        entity = override.get("entity", "node")
        specs = nodes if entity == "node" else relationships if entity == "relationship" else []
        spec = next((s for s in specs if s.id == override.get("cohort")), None)
        key = override.get("property")
        if spec is None or key not in spec.properties:
            raise ValueError("recipe property override references an unknown cohort/property")
        if override.get("preserve_values"):
            spec.properties[key].distribution["preserve_values"] = True
        recipe_properties.append((spec, key, override))
    metrics = {metric.id: metric for metric in profile.metrics}
    unique = {
        (r.parameters.get("cohort"), r.parameters.get("property"))
        for r in _rules(profile)
        if r.kind == "unique" and r.hard
    }
    for entity, specs, authored in [
        ("node", nodes, bool(profile.targets.nodes)),
        ("relationship", relationships, bool(profile.targets.relationships)),
    ]:
        for spec in specs:
            for key, prop in spec.properties.items():
                if authored or prop.distribution.get("preserve_values"):
                    continue
                summary = metrics.get(f"properties.{entity}.{spec.id}.{key}")
                possible_id = bool(
                    summary
                    and isinstance(summary.value, dict)
                    and summary.value.get("possible_identifier")
                )
                identifier_name = bool(re.search(r"(^id$|_id$|^uuid$|^guid$|^email$)", key, re.I))
                if (spec.id, key) in unique or (possible_id and identifier_name):
                    if prop.type in {"integer", "string"}:
                        prop.distribution = {"kind": "id", "prefix": f"synthetic-{spec.id}-"}
            if entity != "node" or authored:
                continue
            keys = sorted(spec.properties)
            best = {}
            for i, a in enumerate(keys):
                for b in keys[i + 1 :]:
                    metric = metrics.get(f"dependencies.{spec.id}.{a}.{b}")
                    if not metric or not isinstance(metric.value, dict):
                        continue
                    information = metric.value.get("mutual_information_bits", 0)
                    if information <= best.get(b, (0, None))[0]:
                        continue
                    if spec.properties[a].distribution.get("kind") == "id":
                        continue
                    groups = {}
                    for row in metric.value.get("contingency", []):
                        group = groups.setdefault(
                            dumps(row["a"]), {"value": row["a"], "values": [], "weights": []}
                        )
                        group["values"].append(row["b"])
                        group["weights"].append(row["count"])
                    if not groups:
                        continue
                    conditional = {
                        "kind": "conditional",
                        "on": a,
                        "default_on_missing": True,
                        "default": spec.properties[b].distribution,
                        "cases": [
                            {
                                "value": group["value"],
                                "distribution": {
                                    "kind": "empirical",
                                    "values": group["values"],
                                    "weights": group["weights"],
                                },
                            }
                            for group in groups.values()
                        ],
                    }
                    best[b] = (information, conditional)
            for key, (_, distribution) in best.items():
                spec.properties[key].distribution = distribution
    for spec, key, override in recipe_properties:
        if "distribution" in override:
            if not isinstance(override["distribution"], dict):
                raise ValueError("recipe distribution must be a mapping")
            spec.properties[key].distribution = dict(override["distribution"])
        if override.get("preserve_values"):
            spec.properties[key].distribution["preserve_values"] = True
    return nodes, relationships


def _role_property_schedule(profile, node, n, schedule, base_dir, seed, rules):
    """Retain measured role/attribute coupling for mandatory endpoint matching."""
    if profile.targets.nodes:
        return  # Explicit authored targets take precedence over measured conditional tables.
    needed = set()
    _, relationships = profile.generation_specs()
    rels = {r.id: r for r in relationships}
    for rule in rules:
        if rule.hard and rule.kind == "same_property":
            rel = rels[rule.parameters["relationship"]]
            if rel.source == node.id:
                needed.add(rule.parameters["source_property"])
            if rel.target == node.id:
                needed.add(rule.parameters["target_property"])
    role_rows = read_table(node.roles, base_dir)
    if not needed or not role_rows:
        return
    assignments = np.repeat(np.arange(len(role_rows)), allocate([r["count"] for r in role_rows], n))
    np.random.default_rng(stable_seed(seed, node.id, "roles", "sequence")).shuffle(assignments)
    metric = next((m for m in profile.metrics if m.id == "dependencies.topology_attributes"), None)
    if metric is None:
        return
    for item in metric.value:
        key = item.get("property")
        if item.get("cohort") != node.id or key not in needed or "role_distribution" not in item:
            continue
        joint = read_table(item["role_distribution"], base_dir)
        by_role = {}
        for row in joint:
            by_role.setdefault(dumps(row["roles"]), []).append(row)
        values = []
        value_indices = {}
        sequence = np.full(n, -1, dtype=np.int64)
        for role_index, role in enumerate(role_rows):
            positions = np.flatnonzero(assignments == role_index)
            rows = by_role.get(dumps(role["roles"]), [])
            if not len(positions):
                continue
            if not rows:
                raise ValueError("role/attribute joint distribution is incomplete")
            slots = []
            for row, count in zip(
                rows, allocate([r["count"] for r in rows], len(positions)), strict=True
            ):
                value_key = dumps(row["value"])
                if value_key not in value_indices:
                    value_indices[value_key] = len(values)
                    values.append(row["value"])
                slots.append(np.full(int(count), value_indices[value_key], dtype=np.int64))
            allocated = np.concatenate(slots)
            np.random.default_rng(
                stable_seed(seed, node.id, key, role_index, "role-values")
            ).shuffle(allocated)
            sequence[positions] = allocated
        schedule.presets[key] = (values, sequence)
        schedule.generated.pop(key, None)


def _endpoint_properties(store, ids):
    if not len(ids):
        return {}
    rows = store.db.execute(
        "SELECT id, properties FROM nodes WHERE id IN (SELECT unnest(?))", [np.unique(ids).tolist()]
    ).fetchall()
    return {identifier: loads(properties) for identifier, properties in rows}


def _same_property_pairing(
    store, rule, rel, sources, targets, rng, target_range, explicit_in_degrees, bounds
):
    params = rule.parameters
    source_key, target_key = params["source_property"], params["target_property"]
    source_properties = _endpoint_properties(store, np.unique(sources))
    target_ids = np.arange(*target_range, dtype=np.int64)
    target_properties = _endpoint_properties(store, target_ids)
    groups = {}
    for node_id in target_ids:
        value = target_properties[int(node_id)].get(target_key)
        if value is not None:
            groups.setdefault(dumps(value), []).append(int(node_id))
    source_groups = {}
    for index, source in enumerate(sources):
        value = source_properties[int(source)].get(source_key)
        if value is None:
            raise ValueError(f"same_property rule {rule.id!r} requires {source_key!r}")
        source_groups.setdefault(dumps(value), []).append(index)
    target_degrees = Counter(int(node_id) for node_id in targets)
    for value, indices in source_groups.items():
        candidates = groups.get(value, [])
        if not candidates:
            raise ValueError(f"same_property rule {rule.id!r} has no compatible target for {value}")
        if explicit_in_degrees:
            stubs = [node_id for node_id in candidates for _ in range(target_degrees[node_id])]
            if len(stubs) != len(indices):
                raise ValueError(
                    f"same_property rule {rule.id!r} conflicts with incoming degree targets"
                )
            rng.shuffle(stubs)
            targets[indices] = stubs
        else:
            minimum, maximum = bounds
            sequence = _sequence([], len(candidates), len(indices), rng, minimum, maximum)
            stubs = np.repeat(np.asarray(candidates, dtype=np.int64), sequence)
            rng.shuffle(stubs)
            targets[indices] = stubs


def generate(
    profile: Profile,
    output_dir: Path,
    *,
    base_dir: Path,
    scale: float = 1,
    seed: int = 42,
    batch_size: int = 10000,
    provider=None,
) -> GraphStore:
    """Generate to local staging only. The caller validates before loading Neo4j.

    The effective scale is the command's scale multiplied by profile.targets.scale.
    Generated provider hooks may override properties, never topology or identifiers.
    """
    if not math.isfinite(scale) or scale <= 0 or batch_size <= 0:
        raise ValueError("scale and batch_size must be positive")
    rules = _rules(profile)
    nodes, relationships = _generation_specs(profile)
    effective_scale = scale * profile.targets.scale
    node_counts, relationship_counts = (
        _counts(nodes, effective_scale),
        _counts(relationships, effective_scale),
    )
    node_specs = {node.id: node for node in nodes}
    ranges = {}
    n = 0
    for node in nodes:
        ranges[node.id] = (n, n + node_counts[node.id])
        n += node_counts[node.id]
    loops, parallel = (
        _loop_policy(profile, "allow_self_loops"),
        _loop_policy(profile, "allow_parallel_edges"),
    )
    for rel in relationships:
        m, ns, nt = relationship_counts[rel.id], node_counts[rel.source], node_counts[rel.target]
        if m and (not ns or not nt):
            raise ValueError(f"relationship {rel.id!r} has an empty endpoint cohort after scaling")
        _bounds(rules, rel.id, "out", ns, m)
        _bounds(rules, rel.id, "in", nt, m)
        capacity = ns * nt - (ns if rel.source == rel.target and not loops else 0)
        if not parallel and m > capacity:
            raise ValueError(
                f"infeasible simple endpoint capacity for {rel.id!r}: {m} > {capacity}"
            )
    for rule in rules:
        if not rule.hard:
            continue
        params = rule.parameters
        if rule.kind == "required":
            spec = node_specs.get(params.get("cohort")) or next(
                (r for r in relationships if r.id == params.get("cohort")), None
            )
            if spec is None or params.get("property") not in spec.properties:
                raise ValueError(f"required property rule {rule.id!r} has no generation provider")
    store = GraphStore(safe_path(output_dir))
    if store.counts() != (0, 0):
        store.close()
        raise ValueError("generation staging directory must be empty")
    values = ValueProviders(base_dir, lookup=_lookup_provider(store, ranges))
    community_plan = _community_plan(profile, n, seed)
    refinement = []
    try:
        for node in _node_order(nodes):
            start, stop = ranges[node.id]
            patterns, assignments = _pattern_assignments(node, stop - start, base_dir, seed)
            required = {
                r.parameters["property"]
                for r in rules
                if r.hard and r.kind == "required" and r.parameters.get("cohort") == node.id
            }
            schedule = _PropertySchedule(
                node, stop - start, patterns, assignments, values, seed, required
            )
            _role_property_schedule(profile, node, stop - start, schedule, base_dir, seed, rules)
            for lo in range(start, stop, batch_size):
                batch = []
                for identifier in range(lo, min(stop, lo + batch_size)):
                    context = {
                        "id": identifier,
                        "index": identifier - start,
                        "cohort": node.id,
                        "kind": "node",
                    }
                    properties = values.properties(
                        node.properties,
                        seed,
                        context,
                        selected_keys=schedule.keys(identifier - start),
                        presets=schedule.values(identifier - start),
                    )
                    if required - properties.keys():
                        raise ValueError(f"required properties missing in {node.id!r}")
                    batch.append(
                        {
                            "id": identifier,
                            "cohort": node.id,
                            "labels": node.labels,
                            "properties": properties,
                        }
                    )
                if provider is not None:
                    overrides = provider("node", node.id, batch, seed)
                    _merge_overrides(batch, overrides)
                store.add_nodes(batch)
            values._categoricals.clear()
        offset = 0
        for rel in relationships:
            m = relationship_counts[rel.id]
            rng = np.random.default_rng(stable_seed(seed, rel.id, "topology"))
            out_bounds = _bounds(rules, rel.id, "out", node_counts[rel.source], m)
            in_bounds = _bounds(rules, rel.id, "in", node_counts[rel.target], m)
            out_rows = read_table(rel.out_degrees, base_dir)
            in_rows = read_table(rel.in_degrees, base_dir)
            source_roles = _role_histogram(node_specs[rel.source], rel.id, "out", base_dir)
            target_roles = _role_histogram(node_specs[rel.target], rel.id, "in", base_dir)
            outgoing = _sequence(
                source_roles or out_rows,
                node_counts[rel.source],
                m,
                np.random.default_rng(stable_seed(seed, rel.source, "roles", "sequence")),
                *out_bounds,
            )
            incoming = _sequence(
                target_roles or in_rows,
                node_counts[rel.target],
                m,
                np.random.default_rng(stable_seed(seed, rel.target, "roles", "sequence")),
                *in_bounds,
            )
            sources = np.repeat(np.arange(*ranges[rel.source], dtype=np.int64), outgoing)
            targets = np.repeat(np.arange(*ranges[rel.target], dtype=np.int64), incoming)
            rng.shuffle(targets)
            dag = any(
                r.hard and r.kind == "dag" and r.parameters.get("relationship") == rel.id
                for r in rules
            )
            if (
                dag
                and rel.source == rel.target
                and not out_rows
                and not in_rows
                and not source_roles
                and not target_roles
            ):
                # A topological order is available without constraining unrequested degree marginals.
                if (
                    out_bounds[0]
                    or in_bounds[0]
                    or m > node_counts[rel.source] * (node_counts[rel.source] - 1) // 2
                ):
                    raise ValueError(f"DAG rule for {rel.id!r} conflicts with counts/cardinalities")
                start, stop = ranges[rel.source]
                if m and stop - start < 2:
                    raise ValueError("a DAG relationship requires at least two nodes")
                sources = rng.integers(start, max(start + 1, stop - 1), size=m, dtype=np.int64)
                targets = np.asarray(
                    [rng.integers(source + 1, stop) for source in sources], dtype=np.int64
                )
                if out_bounds[1] < m or in_bounds[1] < m:
                    raise ValueError(
                        "DAG generation with constrained cardinalities requires explicit feasible degree sequences"
                    )
            matching_rules = [
                r
                for r in rules
                if r.hard
                and r.kind == "same_property"
                and r.parameters.get("relationship") == rel.id
            ]
            if community_plan is not None and not matching_rules and not dag:
                refinement.append(
                    {
                        "relationship": rel.id,
                        **_community_pairing(
                            sources, targets, community_plan, rng, profile.targets.refinement_rounds
                        ),
                    }
                )
            elif community_plan is not None:
                refinement.append(
                    {
                        "relationship": rel.id,
                        "status": "not_refined",
                        "reason": "hard semantic endpoint constraints take precedence over community mixing",
                    }
                )
            if len(matching_rules) > 1:
                raise ValueError(
                    "multiple same_property restrictions require a combined domain recipe"
                )
            for rule in matching_rules:
                _same_property_pairing(
                    store,
                    rule,
                    rel,
                    sources,
                    targets,
                    rng,
                    ranges[rel.target],
                    bool(in_rows or target_roles),
                    in_bounds,
                )
            if matching_rules and (dag or (rel.source == rel.target and not loops)):
                raise ValueError(
                    "combined same_property and loop/DAG restrictions require a domain recipe"
                )
            if matching_rules:
                # Repair separately within semantic groups so swaps cannot violate equality.
                props = _endpoint_properties(store, np.unique(targets))
                key = matching_rules[0].parameters["target_property"]
                groups = {}
                for index, target in enumerate(targets):
                    groups.setdefault(dumps(props[int(target)][key]), []).append(index)
                for indices in groups.values():
                    subset = np.asarray(indices, dtype=np.int64)
                    t = targets[subset].copy()
                    _repair_pairing(
                        sources[subset],
                        t,
                        n,
                        rng,
                        loops=loops,
                        parallel=parallel,
                        dag=dag,
                        rounds=profile.targets.refinement_rounds,
                    )
                    targets[subset] = t
            else:
                _repair_pairing(
                    sources,
                    targets,
                    n,
                    rng,
                    loops=loops,
                    parallel=parallel,
                    dag=dag,
                    rounds=profile.targets.refinement_rounds,
                )
            patterns, assignments = _pattern_assignments(rel, m, base_dir, seed)
            required = {
                r.parameters["property"]
                for r in rules
                if r.hard
                and r.kind == "required"
                and r.parameters.get("cohort") == rel.id
                and r.parameters.get("entity") == "relationship"
            }
            schedule = _PropertySchedule(rel, m, patterns, assignments, values, seed, required)
            needs_context = provider is not None or any(
                "source_properties" in json.dumps(prop.distribution)
                or "target_properties" in json.dumps(prop.distribution)
                for prop in rel.properties.values()
            )
            for lo in range(0, m, batch_size):
                hi = min(m, lo + batch_size)
                endpoint_props = (
                    _endpoint_properties(store, np.concatenate((sources[lo:hi], targets[lo:hi])))
                    if needs_context
                    else {}
                )
                batch, contexts = [], []
                for index in range(lo, hi):
                    identifier = offset + index
                    source, target = int(sources[index]), int(targets[index])
                    context = {
                        "id": identifier,
                        "index": index,
                        "cohort": rel.id,
                        "kind": "relationship",
                        "source": source,
                        "target": target,
                        "source_properties": endpoint_props.get(source, {}),
                        "target_properties": endpoint_props.get(target, {}),
                    }
                    properties = values.properties(
                        rel.properties,
                        seed,
                        context,
                        selected_keys=schedule.keys(index),
                        presets=schedule.values(index),
                    )
                    row = {
                        "id": identifier,
                        "cohort": rel.id,
                        "source": source,
                        "target": target,
                        "type": rel.type,
                        "properties": properties,
                    }
                    batch.append(row)
                    contexts.append(
                        {
                            **row,
                            "source_properties": context["source_properties"],
                            "target_properties": context["target_properties"],
                        }
                    )
                if provider is not None:
                    _merge_overrides(batch, provider("relationship", rel.id, contexts, seed))
                store.add_edges(batch)
            values._categoricals.clear()
            offset += m
        (store.path / "generation.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "effective_scale": effective_scale,
                    "nodes": n,
                    "relationships": sum(relationship_counts.values()),
                    "community_refinement": refinement,
                    "limitations": [
                        "Feasible typed degree marginals and observed role assignments are retained subject to integer rounding and hard constraints.",
                        "Sampled categorical dependencies use one strongest measured parent per property; higher-order dependencies require domain providers.",
                        "Components, clustering, paths and motifs are soft targets and require independent re-analysis.",
                        "Observed identifier values are replaced unless preserve_values is explicitly requested.",
                    ],
                },
                indent=2,
            )
        )
        return store
    except BaseException:
        store.close()
        raise


def _merge_overrides(batch, overrides):
    if not isinstance(overrides, list) or len(overrides) != len(batch):
        raise ValueError("domain provider must return one property mapping for each input row")
    for row, override in zip(batch, overrides, strict=True):
        if not isinstance(override, dict) or any(value is None for value in override.values()):
            raise ValueError("domain provider returned invalid property overrides")
        row["properties"].update(override)
