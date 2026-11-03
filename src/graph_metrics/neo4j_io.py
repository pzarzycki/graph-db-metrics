"""Read-only source scans and batched target writes using the official driver."""

from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from neo4j import READ_ACCESS, GraphDatabase

from .io import dumps, loads
from .store import GraphStore


def cohort_id(labels: list[str]) -> str:
    return "n_" + hashlib.sha256(dumps(sorted(labels)).encode()).hexdigest()[:16]


def relationship_id(source: str, kind: str, target: str) -> str:
    return "r_" + hashlib.sha256(dumps([source, kind, target]).encode()).hexdigest()[:16]


def connect(uri: str, *, username: str | None = None, password: str | None = None):
    parsed = urlsplit(uri)
    if parsed.username or parsed.password:
        raise ValueError("put Neo4j credentials in environment variables, not the URI")
    password = password if password is not None else os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise ValueError("set NEO4J_PASSWORD before connecting")
    return GraphDatabase.driver(
        uri, auth=(username or os.environ.get("NEO4J_USERNAME", "neo4j"), password)
    )


def scan(
    uri: str,
    output_dir: Path,
    *,
    database: str = "neo4j",
    batch_size: int = 10000,
    username: str | None = None,
    password: str | None = None,
) -> tuple[GraphStore, dict]:
    store = GraphStore(output_dir)
    if any(store.counts()):
        store.close()
        raise ValueError("source staging directory is not empty; choose a new output directory")
    started = datetime.now(timezone.utc).isoformat()
    store.db.execute("CREATE TABLE source_ids (element_id VARCHAR PRIMARY KEY, id BIGINT)")
    source = {
        "system": "neo4j",
        "database": database,
        "started_at": started,
        "consistency": "live_non_atomic",
        "snapshot": None,
        "consistency_note": "Read-committed streaming queries are not an atomic database snapshot.",
    }
    try:
        with connect(uri, username=username, password=password) as driver:
            driver.verify_connectivity()
            with driver.session(
                database=database, default_access_mode=READ_ACCESS, fetch_size=batch_size
            ) as session:
                before = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
                pending, identities = [], []
                result = session.run(
                    "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties"
                )
                for index, row in enumerate(result):
                    labels = sorted(row["labels"])
                    pending.append(
                        {
                            "id": index,
                            "cohort": cohort_id(labels),
                            "labels": labels,
                            "properties": row["properties"],
                        }
                    )
                    identities.append((row["id"], index))
                    if len(pending) >= batch_size:
                        store.add_nodes(pending)
                        store.db.execute(
                            "INSERT INTO source_ids SELECT unnest(?),unnest(?)",
                            [list(x) for x in zip(*identities)],
                        )
                        pending, identities = [], []
                if pending:
                    store.add_nodes(pending)
                    store.db.execute(
                        "INSERT INTO source_ids SELECT unnest(?),unnest(?)",
                        [list(x) for x in zip(*identities)],
                    )
                edge_index = 0
                pending = []
                result = session.run(
                    "MATCH (a)-[r]->(b) RETURN elementId(a) AS source, elementId(b) AS target, type(r) AS type, properties(r) AS properties"
                )
                for row in result:
                    pending.append(
                        {
                            "source": row["source"],
                            "target": row["target"],
                            "type": row["type"],
                            "properties": row["properties"],
                        }
                    )
                    if len(pending) >= batch_size:
                        edge_index = _scan_edges(store, pending, edge_index)
                        pending = []
                if pending:
                    _scan_edges(store, pending, edge_index)
                after = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
                source["node_count_before"] = before
                source["node_count_after"] = after
                source["observed_count_change"] = before != after
                try:
                    source["constraints"] = [
                        dict(row)
                        for row in session.run(
                            "SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties RETURN *"
                        )
                    ]
                except Exception as error:
                    source["constraint_introspection"] = type(error).__name__
        source["finished_at"] = datetime.now(timezone.utc).isoformat()
        return store, source
    except BaseException:
        store.close()
        raise


def _scan_edges(store, rows, first_id):
    store.db.execute(
        "CREATE OR REPLACE TEMP TABLE incoming AS SELECT unnest(?) AS source,unnest(?) AS target,unnest(?) AS type,unnest(?) AS properties",
        [
            [r["source"] for r in rows],
            [r["target"] for r in rows],
            [r["type"] for r in rows],
            [dumps(r["properties"]) for r in rows],
        ],
    )
    resolved = store.db.execute("""SELECT s.id,t.id,ns.cohort,nt.cohort,i.type,i.properties
        FROM incoming i JOIN source_ids s ON s.element_id=i.source
        JOIN source_ids t ON t.element_id=i.target
        JOIN nodes ns ON ns.id=s.id JOIN nodes nt ON nt.id=t.id""").fetchall()
    if len(resolved) != len(rows):
        raise ValueError(
            "source changed during scan: relationship endpoint is missing; retry on a quiescent source"
        )
    store.add_edges(
        [
            {
                "id": first_id + i,
                "cohort": relationship_id(a, b, c),
                "source": s,
                "target": t,
                "type": b,
                "properties": loads(props),
            }
            for i, (s, t, a, c, b, props) in enumerate(resolved)
        ]
    )
    return first_id + len(rows)


def _identifier(value: str) -> str:
    return "`" + value.replace("`", "``") + "`"


def load_online(
    store: GraphStore, uri: str, *, database="neo4j", username=None, password=None, batch_size=5000
) -> dict:
    """Only empty targets are accepted. Loader IDs are removed before fidelity analysis."""
    started = time.monotonic()
    with connect(uri, username=username, password=password) as driver:
        with driver.session(database=database) as session:
            if session.run("MATCH (n) RETURN count(n) AS n").single()["n"]:
                raise ValueError(
                    "target database is populated; generation requires an empty target"
                )
            # Choose a loader label/property absent from every generated record.
            import uuid

            marker = "_GraphMetrics_" + uuid.uuid4().hex
            label, key = _identifier(marker), _identifier(marker)
            session.run(
                f"CREATE CONSTRAINT {label} FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
            ).consume()
            try:
                for batch in store.iter_nodes(batch_size):
                    groups = {}
                    for row in batch:
                        groups.setdefault(tuple(sorted(row["labels"])), []).append(
                            {"id": row["id"], "properties": row["properties"]}
                        )
                    for labels, rows in groups.items():
                        label_text = ":".join(_identifier(x) for x in labels)
                        if label_text:
                            label_text = ":" + label_text
                        session.run(
                            f"UNWIND $rows AS row CREATE (n:{label}{label_text}) SET n = row.properties, n.{key}=row.id",
                            rows=rows,
                        ).consume()
                for batch in store.iter_edges(batch_size):
                    groups = {}
                    for row in batch:
                        groups.setdefault(row["type"], []).append(
                            {k: row[k] for k in ("source", "target", "properties")}
                        )
                    for kind, rows in groups.items():
                        session.run(
                            f"UNWIND $rows AS row MATCH (a:{label} {{{key}:row.source}}), (b:{label} {{{key}:row.target}}) CREATE (a)-[r:{_identifier(kind)}]->(b) SET r=row.properties",
                            rows=rows,
                        ).consume()
                # Bounded cleanup transactions avoid one database-sized transaction.
                session.run(
                    f"MATCH (n:{label}) CALL (n) {{ REMOVE n.{key}, n:{label} }} IN TRANSACTIONS OF {int(batch_size)} ROWS"
                ).consume()
                session.run(f"DROP CONSTRAINT {label}").consume()
            except BaseException:
                # Preserve partial data for diagnosis; never silently delete a target.
                raise
    return {
        "mode": "online",
        "database": database,
        "seconds": time.monotonic() - started,
        "nodes": store.counts()[0],
        "relationships": store.counts()[1],
        "status": "loaded",
    }
