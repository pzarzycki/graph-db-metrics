"""Seeded procedural values, with explicit dependencies instead of executable YAML."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
from datetime import date, datetime, timedelta
from typing import Any, Callable

import numpy as np
from faker import Faker

from .io import decode, read_table


def stable_seed(seed: int, *parts: Any) -> int:
    text = json.dumps([seed, *parts], sort_keys=True, default=str, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _field(name: str, properties: dict, context: dict) -> Any:
    if name in properties:
        return properties[name]
    current: Any = context
    for part in name.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(name)
        current = current[part]
    return current


class ValueProviders:
    def __init__(self, base_dir, lookup: Callable | None = None):
        self.base_dir = base_dir
        self.lookup = lookup
        self._categoricals: dict[int, tuple] = {}
        self._faker: dict[str, Faker] = {}

    def value(self, spec, rng: random.Random, properties: dict, context: dict) -> Any:
        distribution = spec.distribution
        kind = distribution.get("kind", "")
        if not kind:
            kind = {
                "integer": "integer",
                "int": "integer",
                "float": "uniform",
                "boolean": "boolean",
                "date": "date",
                "datetime": "datetime",
                "list": "list",
                "bytes": "bytes",
            }.get(spec.type, "id")
        return self._draw(kind, distribution, spec.type, rng, properties, context)

    def categorical(self, dist):
        """Compile a finite distribution once; every subsequent draw is O(log categories)."""
        key = id(dist)
        if key not in self._categoricals:
            values = dist.get("values", [])
            weights = dist.get("weights")
            if "table" in dist:
                rows = read_table(dist["table"], self.base_dir)
                values, weights = [r["value"] for r in rows], [r["count"] for r in rows]
            if not values:
                raise ValueError("categorical/empirical distribution requires values")
            if weights is None:
                weights = np.ones(len(values), dtype=np.float64)
            else:
                weights = np.asarray(weights, dtype=np.float64)
            if (
                len(weights) != len(values)
                or np.any(weights < 0)
                or not np.all(np.isfinite(weights))
                or not weights.sum()
            ):
                raise ValueError(
                    "distribution weights must be finite, nonnegative and have positive total"
                )
            cumulative = np.cumsum(weights)
            # Retain the distribution object to prevent identity reuse in the cache.
            self._categoricals[key] = (dist, values, weights, cumulative)
        _, values, weights, cumulative = self._categoricals[key]
        return values, weights, cumulative

    def _draw(self, kind, dist, value_type, rng, properties, context):
        if kind in {"categorical", "empirical"}:
            values, weights, cumulative = self.categorical(dist)
            if exclude := dist.get("exclude_property"):
                forbidden = _field(exclude, properties, context)
                eligible = [i for i, value in enumerate(values) if decode(value) != forbidden]
                if not eligible:
                    raise ValueError(f"no categorical value remains after excluding {exclude}")
                values = [values[i] for i in eligible]
                cumulative = np.cumsum(weights[eligible])
            index = min(
                len(values) - 1, bisect.bisect_right(cumulative, rng.random() * cumulative[-1])
            )
            return decode(values[index])
        if kind in {"constant", "const"}:
            return decode(dist.get("value"))
        if kind in {"integer", "uniform_int"}:
            low, high = int(dist.get("min", 0)), int(dist.get("max", 100))
            if high < low:
                raise ValueError("integer range is empty")
            return rng.randint(low, high)
        if kind == "uniform":
            low, high = float(dist.get("min", 0)), float(dist.get("max", 1))
            if high < low:
                raise ValueError("uniform range is empty")
            return rng.uniform(low, high)
        if kind == "normal":
            value = rng.gauss(dist.get("mean", 0), dist.get("std", 1))
            value = min(dist.get("max", math.inf), max(dist.get("min", -math.inf), value))
            return round(value) if value_type in {"integer", "int"} else value
        if kind == "quantiles":
            values = dist.get("values", dist.get("quantiles", []))
            if isinstance(values, dict):
                values = [v for _, v in sorted(values.items(), key=lambda item: float(item[0]))]
            if not values:
                raise ValueError("quantiles distribution requires values")
            position = rng.random() * max(0, len(values) - 1)
            lo = int(position)
            value = values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (
                position - lo
            )
            return round(value) if value_type in {"integer", "int"} else value
        if kind == "boolean":
            return rng.random() < dist.get("probability", 0.5)
        if kind in {"id", "unique"}:
            identifier = context.get("index", context["id"])
            if value_type in {"integer", "int"}:
                return int(dist.get("start", 0)) + identifier
            return f"{dist.get('prefix', context.get('cohort', 'entity') + '-')}{identifier:0{int(dist.get('width', 0))}d}"
        if kind == "faker":
            locale = dist.get("locale", "en_US")
            if locale not in self._faker:
                self._faker[locale] = Faker(locale)
            fake = self._faker[locale]
            fake.seed_instance(rng.getrandbits(64))
            provider = dist.get("provider", "name")
            if provider.startswith("_") or not callable(getattr(fake, provider, None)):
                raise ValueError(f"unknown Faker provider: {provider}")
            return getattr(fake, provider)(**dist.get("arguments", {}))
        if kind == "template":
            return dist["template"].format_map({**context, **properties})
        if kind == "conditional":
            try:
                value = _field(dist["on"], properties, context)
            except KeyError:
                if not dist.get("default_on_missing") or dist["on"] in context.get(
                    "pending_properties", []
                ):
                    raise
                selected = dist["default"]
                return self._draw(
                    selected.get("kind", "constant"), selected, value_type, rng, properties, context
                )
            cases = dist.get("cases", {})
            if isinstance(cases, dict):
                selected = cases.get(str(value), dist.get("default"))
            else:
                selected = next(
                    (r["distribution"] for r in cases if decode(r["value"]) == value),
                    dist.get("default"),
                )
            if selected is None:
                raise ValueError(f"conditional distribution has no case for {value!r}")
            return self._draw(
                selected.get("kind", "constant"), selected, value_type, rng, properties, context
            )
        if kind == "derived":
            value = _field(dist["on"], properties, context)
            value = value * dist.get("multiplier", 1) + dist.get("offset", 0)
            if "random_min" in dist or "random_max" in dist:
                value += rng.uniform(dist.get("random_min", 0), dist.get("random_max", 0))
            value = min(dist.get("max", math.inf), max(dist.get("min", -math.inf), value))
            return round(value) if value_type in {"integer", "int"} else value
        if kind == "copy":
            return _field(dist["on"], properties, context)
        if kind == "lookup":
            if self.lookup is None:
                raise ValueError("lookup provider requires a graph context")
            match_value = _field(dist["on"], properties, context)
            return self.lookup(
                dist["cohort"], dist["match_property"], match_value, dist["property"]
            )
        if kind == "reference":
            if self.lookup is None:
                raise ValueError("reference provider requires a graph context")
            forbidden = (
                _field(dist["exclude_property"], properties, context)
                if "exclude_property" in dist
                else object()
            )
            for _ in range(100):
                value = self.lookup(
                    dist["cohort"], "__random__", rng.getrandbits(64), dist["property"]
                )
                if value != forbidden:
                    return value
            raise ValueError("reference provider cannot satisfy distinct-value restriction")
        if kind in {"date", "datetime"}:
            cls = date if kind == "date" else datetime
            low = cls.fromisoformat(str(dist.get("min", "2020-01-01")))
            high = cls.fromisoformat(str(dist.get("max", "2026-01-01")))
            if high < low:
                raise ValueError("temporal range is empty")
            delta = high - low
            return low + timedelta(seconds=rng.randint(0, max(0, int(delta.total_seconds()))))
        if kind == "datetime_offset":
            value = _field(dist["on"], properties, context)
            if isinstance(value, str):
                value = datetime.fromisoformat(value)
            if hasattr(value, "to_native"):
                value = value.to_native()
            minutes = (
                _field(dist["minutes_property"], properties, context)
                if "minutes_property" in dist
                else dist.get("minutes", 0)
            )
            return value + timedelta(minutes=minutes)
        if kind == "distance_km":
            latitude1, longitude1, latitude2, longitude2 = [
                math.radians(_field(field, properties, context)) for field in dist["coordinates"]
            ]
            a = (
                math.sin((latitude2 - latitude1) / 2) ** 2
                + math.cos(latitude1)
                * math.cos(latitude2)
                * math.sin((longitude2 - longitude1) / 2) ** 2
            )
            return 6371.0088 * 2 * math.asin(math.sqrt(min(1, max(0, a))))
        if kind == "list":
            low, high = int(dist.get("min_length", 0)), int(dist.get("max_length", 5))
            element = dist.get("element", {"kind": "integer"})
            return [
                self._draw(
                    element.get("kind", "integer"), element, "integer", rng, properties, context
                )
                for _ in range(rng.randint(low, high))
            ]
        if kind == "bytes":
            return rng.randbytes(int(dist.get("length", 16)))
        raise ValueError(
            f"unsupported procedural distribution: {kind!r}; prepare a domain provider"
        )

    def properties(
        self,
        specs,
        seed: int,
        context: dict,
        *,
        selected_keys: set[str] | None = None,
        presets: dict | None = None,
    ):
        properties = dict(presets or {})
        pending = {key: spec for key, spec in specs.items() if key not in properties}
        while pending:
            progress = False
            for key, spec in list(pending.items()):
                rng = random.Random(
                    stable_seed(seed, context.get("kind"), context["cohort"], context["id"], key)
                )
                include = (
                    key in selected_keys
                    if selected_keys is not None
                    else rng.random() < spec.presence
                )
                if not include:
                    del pending[key]
                    progress = True
                    continue
                try:
                    properties[key] = self.value(
                        spec, rng, properties, {**context, "pending_properties": list(pending)}
                    )
                except KeyError:
                    continue
                if properties[key] is None:
                    raise ValueError(
                        f"property {key!r} generated null; use presence for missing values"
                    )
                del pending[key]
                progress = True
            if not progress:
                raise ValueError(
                    f"unresolved or cyclic property dependencies: {', '.join(pending)}"
                )
        return properties
