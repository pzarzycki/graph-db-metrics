"""Project-local CSV export, offline Community import, and fresh Docker targets."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import secrets
import socket
import subprocess
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from neo4j.spatial import CartesianPoint, WGS84Point
from neo4j.time import Date, DateTime, Duration, Time

from .io import dumps, safe_path
from .neo4j_io import connect
from .store import GraphStore

IMAGE = "neo4j:2026.10.1-community"
ARRAY_DELIMITER = "\x1f"


def _csv_type(value):
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "long"
    if isinstance(value, float):
        return "double"
    if isinstance(value, str):
        return "string"
    if isinstance(value, DateTime):
        return "datetime" if value.tzinfo else "localdatetime"
    if isinstance(value, Date):
        return "date"
    if isinstance(value, Time):
        return "time" if value.tzinfo else "localtime"
    if isinstance(value, Duration):
        return "duration"
    if isinstance(value, (CartesianPoint, WGS84Point)):
        return "point"
    if isinstance(value, list):
        kinds = {_csv_type(v) for v in value}
        if len(kinds) > 1:
            raise ValueError("mixed-type arrays cannot be imported as Neo4j properties")
        return (next(iter(kinds)) if kinds else "string") + "[]"
    raise ValueError(
        f"offline import does not support {type(value).__name__}; use --target-uri for online loading"
    )


def _csv_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        parts = [str(_csv_value(v)) for v in value]
        if any(ARRAY_DELIMITER in p for p in parts):
            raise ValueError("array value contains the export delimiter; use online loading")
        return ARRAY_DELIMITER.join(parts)
    if isinstance(value, (CartesianPoint, WGS84Point)):
        names = (
            ("longitude", "latitude", "height")
            if isinstance(value, WGS84Point)
            else ("x", "y", "z")
        )
        crs = {4326: "wgs-84", 4979: "wgs-84-3d", 7203: "cartesian", 9157: "cartesian-3d"}[
            value.srid
        ]
        return (
            "{" + ", ".join(f"{key}: {v}" for key, v in zip(names, value)) + f", crs: '{crs}'" + "}"
        )
    if isinstance(value, (DateTime, Time)):
        return value.iso_format()
    return str(value)


def export_csv(store: GraphStore, output_dir: Path) -> dict:
    output_dir = safe_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {"nodes": [], "relationships": []}
    handles = OrderedDict()
    headers = {}
    try:
        for entity, iterator in [("nodes", store.iter_nodes), ("relationships", store.iter_edges)]:
            for batch in iterator():
                for row in batch:
                    props = row["properties"]
                    keys = sorted(props)
                    if any(":" in k or "`" in k for k in keys):
                        raise ValueError(
                            "offline import requires simple property names; use online loading for names containing ':' or '`'"
                        )
                    signature = [(k, _csv_type(props[k])) for k in keys]
                    digest = hashlib.sha256(
                        dumps([entity, row["cohort"], signature]).encode()
                    ).hexdigest()[:24]
                    filename = f"{entity}-{digest}.csv"
                    new = filename not in headers
                    if filename not in handles:
                        if len(handles) >= 64:
                            _, (handle, _) = handles.popitem(last=False)
                            handle.close()
                        handle = (output_dir / filename).open("a", newline="")
                        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
                        handles[filename] = (handle, writer)
                    handle, writer = handles[filename]
                    handles.move_to_end(filename)
                    if new:
                        if entity == "nodes":
                            header = [":ID", ":LABEL"]
                        else:
                            header = [":START_ID", ":END_ID", ":TYPE"]
                        header += [f"{k}:{t}" for k, t in signature]
                        writer.writerow(header)
                        headers[filename] = header
                        files[entity].append(filename)
                    if entity == "nodes":
                        if any(ARRAY_DELIMITER in label for label in row["labels"]):
                            raise ValueError("label contains import delimiter")
                        fields = [row["id"], ARRAY_DELIMITER.join(row["labels"])]
                    else:
                        fields = [row["source"], row["target"], row["type"]]
                    writer.writerow(fields + [_csv_value(props[k]) for k in keys])
    finally:
        for handle, _ in handles.values():
            handle.close()
    if not files["nodes"]:
        (output_dir / "nodes-empty.csv").write_text(":ID,:LABEL\n")
        files["nodes"].append("nodes-empty.csv")
    (output_dir / "manifest.json").write_text(json.dumps(files, indent=2))
    return files


def _docker(args, *, log=None, timeout=600):
    command = ["docker", *args]
    if log:
        with log.open("a") as stream:
            completed = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout
            )
        if completed.returncode:
            raise RuntimeError(f"Docker import failed; see {log}")
        return ""
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError("Docker command failed: " + completed.stderr.strip()[:1000])
    return completed.stdout.strip()


def load_local(store: GraphStore, run_dir: Path, *, offline=True) -> dict:
    run_dir = safe_path(run_dir)
    db_dir = run_dir / "neo4j"
    data = db_dir / "data"
    logs = db_dir / "logs"
    imports = run_dir / "import"
    for directory in (data, logs, imports):
        directory.mkdir(parents=True, exist_ok=True)
    if any(data.iterdir()):
        raise ValueError("local target directory is populated")
    started = time.monotonic()
    mounts = [
        "--mount",
        f"type=bind,src={data},dst=/data",
        "--mount",
        f"type=bind,src={logs},dst=/logs",
        "--mount",
        f"type=bind,src={imports},dst=/import,readonly",
    ]
    identity = ["--user", f"{os.getuid()}:{os.getgid()}"]
    if offline:
        manifest = export_csv(store, imports)
        options = [
            "--id-type=integer",
            "--array-delimiter=U+001F",
            "--bad-tolerance=0",
            "--max-off-heap-memory=2G",
            "--threads=4",
            "--multiline-fields=true",
            "--report-file=/logs/import.report",
        ]
        options += [f"--nodes=/import/{name}" for name in manifest["nodes"]]
        options += [f"--relationships=/import/{name}" for name in manifest["relationships"]]
        _docker(
            [
                "run",
                "--rm",
                *identity,
                *mounts,
                "--env",
                "NEO4J_server_memory_heap_max__size=1G",
                IMAGE,
                "neo4j-admin",
                "database",
                "import",
                "full",
                "neo4j",
                *options,
            ],
            log=run_dir / "import.log",
            timeout=3600,
        )
    name = "graph-db-metrics-generated-" + uuid.uuid4().hex[:12]
    password = secrets.token_hex(24)
    credential_file = db_dir / ".env"
    descriptor = os.open(credential_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(f"NEO4J_AUTH=neo4j/{password}\n")
    with socket.socket() as http_socket, socket.socket() as bolt_socket:
        http_socket.bind(("127.0.0.1", 0))
        bolt_socket.bind(("127.0.0.1", 0))
        http_port = http_socket.getsockname()[1]
        bolt_port = bolt_socket.getsockname()[1]
    _docker(
        [
            "run",
            "-d",
            "--name",
            name,
            "--restart",
            "unless-stopped",
            *identity,
            *mounts,
            "--label",
            "graph-db-metrics.managed=true",
            "--label",
            f"graph-db-metrics.run={run_dir}",
            "--publish",
            f"127.0.0.1:{http_port}:7474",
            "--publish",
            f"127.0.0.1:{bolt_port}:7687",
            "--env",
            f"NEO4J_server_bolt_advertised__address=localhost:{bolt_port}",
            "--env",
            f"NEO4J_server_http_advertised__address=localhost:{http_port}",
            "--env-file",
            str(credential_file),
            "--env",
            "NEO4J_server_memory_heap_initial__size=512m",
            "--env",
            "NEO4J_server_memory_heap_max__size=1G",
            "--env",
            "NEO4J_server_memory_pagecache_size=512m",
            IMAGE,
        ]
    )
    ports = json.loads(_docker(["inspect", name, "--format", "{{json .NetworkSettings.Ports}}"]))
    bolt_port = ports["7687/tcp"][0]["HostPort"]
    http_port = ports["7474/tcp"][0]["HostPort"]
    uri = f"bolt://127.0.0.1:{bolt_port}"
    result = {
        "mode": "offline" if offline else "online",
        "container": name,
        "uri": uri,
        "browser": f"http://127.0.0.1:{http_port}",
        "database": "neo4j",
        "username": "neo4j",
        "credentials_file": str(credential_file.relative_to(run_dir)),
        "status": "starting",
    }
    (run_dir / "connection.json").write_text(json.dumps(result, indent=2))
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            with connect(uri, password=password) as driver:
                driver.verify_connectivity()
                with driver.session(database="neo4j") as session:
                    session.run("RETURN 1").consume()
            break
        except Exception:
            state = _docker(["inspect", name, "--format", "{{.State.Status}}"])
            if state in ("exited", "dead", "restarting"):
                raise RuntimeError(
                    f"generated Neo4j container stopped; inspect docker logs {name}"
                ) from None
            time.sleep(1)
    else:
        raise TimeoutError(f"Neo4j did not become ready; inspect docker logs {name}")
    if not offline:
        from .neo4j_io import load_online

        load_online(store, uri, password=password)
    result.update(status="loaded", seconds=time.monotonic() - started)
    (run_dir / "connection.json").write_text(json.dumps(result, indent=2))
    return result


def local_password(run_dir: Path) -> str:
    for line in (run_dir / "neo4j" / ".env").read_text().splitlines():
        if line.startswith("NEO4J_AUTH=neo4j/"):
            return line.removeprefix("NEO4J_AUTH=neo4j/")
        if line.startswith("NEO4J_PASSWORD="):
            return line.partition("=")[2]
    raise ValueError("missing local target credentials")
