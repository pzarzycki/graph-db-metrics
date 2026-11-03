"""Portable profile files, typed property values, and project-local artifacts."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import duckdb
import yaml
from neo4j.spatial import CartesianPoint, WGS84Point
from neo4j.time import Date, DateTime, Duration, Time
from pydantic import BaseModel

from .models import Profile, TableRef


class ProfileLoader(yaml.SafeLoader):
    """YAML 1.2 booleans keep domain words such as ON and YES as strings."""


ProfileLoader.yaml_implicit_resolvers = copy.deepcopy(yaml.SafeLoader.yaml_implicit_resolvers)
for first, resolvers in ProfileLoader.yaml_implicit_resolvers.items():
    ProfileLoader.yaml_implicit_resolvers[first] = [
        (tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"
    ]
ProfileLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$", re.I), list("tTfF")
)


def project_root() -> Path:
    configured = os.environ.get("GRAPH_METRICS_PROJECT")
    if configured:
        return Path(configured).resolve()
    for path in [Path.cwd(), *Path.cwd().parents]:
        if (path / ".git").exists() or (path / "pyproject.toml").exists():
            return path.resolve()
    return Path.cwd().resolve()


def safe_path(path: str | Path, base_dir: Path | None = None) -> Path:
    root = project_root()
    result = Path(path)
    if not result.is_absolute():
        result = (base_dir or root) / result
    result = result.resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"artifacts must stay within the project: {result}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode(value: Any) -> Any:
    """Preserve Neo4j temporal/spatial/byte types without unsafe YAML constructors."""
    if isinstance(value, BaseModel):
        return encode(value.model_dump(mode="json"))
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"__neo4j_type__": "float", "value": str(value)}
    if isinstance(value, (DateTime, datetime)):
        return {
            "__neo4j_type__": "datetime",
            "value": value.iso_format() if isinstance(value, DateTime) else value.isoformat(),
        }
    if isinstance(value, (Date, date)):
        return {"__neo4j_type__": "date", "value": str(value)}
    if isinstance(value, (Time, time)):
        return {
            "__neo4j_type__": "time",
            "value": value.iso_format() if isinstance(value, Time) else value.isoformat(),
        }
    if isinstance(value, Duration):
        return {
            "__neo4j_type__": "duration",
            "value": [value.months, value.days, value.seconds, value.nanoseconds],
        }
    if isinstance(value, timedelta):
        return encode(
            Duration(days=value.days, seconds=value.seconds, nanoseconds=value.microseconds * 1000)
        )
    if isinstance(value, (CartesianPoint, WGS84Point)):
        return {"__neo4j_type__": "point", "srid": value.srid, "value": list(value)}
    if isinstance(value, (bytes, bytearray)):
        return {"__neo4j_type__": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    raise TypeError(f"unsupported property value: {type(value).__name__}")


def decode(value: Any) -> Any:
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("__neo4j_type__")
    if not kind:
        return {k: decode(v) for k, v in value.items()}
    raw = value["value"]
    if kind == "float":
        return float(raw)
    if kind == "datetime":
        return DateTime.from_iso_format(raw)
    if kind == "date":
        return Date.from_iso_format(raw)
    if kind == "time":
        return Time.from_iso_format(raw)
    if kind == "duration":
        return Duration(months=raw[0], days=raw[1], seconds=raw[2], nanoseconds=raw[3])
    if kind == "point":
        return (WGS84Point if value["srid"] in (4326, 4979) else CartesianPoint)(raw)
    if kind == "bytes":
        return base64.b64decode(raw)
    raise ValueError(f"unsupported encoded property type: {kind}")


def dumps(value: Any) -> str:
    return json.dumps(
        encode(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def loads(value: str) -> Any:
    return decode(json.loads(value))


def read_profile(path: Path) -> Profile:
    with safe_path(path).open() as stream:
        raw = yaml.load(stream, Loader=ProfileLoader)
    return Profile.model_validate(raw)


def write_profile(profile: Profile, path: Path) -> None:
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(
        yaml.safe_dump(profile.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    )
    temp.replace(path)


def profile_digest(profile: Profile) -> str:
    raw = profile.model_dump(mode="json", exclude={"preparation", "samples"})
    return hashlib.sha256(dumps(raw).encode()).hexdigest()


def read_table(value: list[dict] | TableRef | dict, base_dir: Path) -> list[dict]:
    if isinstance(value, list):
        return value
    ref = value if isinstance(value, TableRef) else TableRef.model_validate(value)
    path = safe_path(ref.path, base_dir)
    if sha256(path) != ref.sha256:
        raise ValueError(f"distribution checksum mismatch: {path}")
    with duckdb.connect() as db:
        rows = db.execute("SELECT record FROM read_parquet(?)", [str(path)]).fetchall()
    if len(rows) != ref.rows:
        raise ValueError(f"distribution row count mismatch: {path}")
    return [json.loads(row[0]) for row in rows]


def write_table(rows: list[dict], path: Path, base_dir: Path) -> TableRef:
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as db:
        db.execute("CREATE TABLE records (record VARCHAR)")
        for offset in range(0, len(rows), 10000):
            db.execute(
                "INSERT INTO records SELECT unnest(?)",
                [[dumps(row) for row in rows[offset : offset + 10000]]],
            )
        db.execute("COPY records TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)])
    return TableRef(path=os.path.relpath(path, base_dir), rows=len(rows), sha256=sha256(path))


def write_query_table(
    db, query: str, parameters: list, path: Path, base_dir: Path, row_count: int
) -> TableRef:
    """Spill grouped frequencies straight to Parquet without materializing Python dictionaries."""
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # COPY binds its filename before SELECT parameters; quote the validated local path explicitly.
    destination = str(path).replace("'", "''")
    db.execute(f"COPY ({query}) TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD)", parameters)
    return TableRef(path=os.path.relpath(path, base_dir), rows=row_count, sha256=sha256(path))
