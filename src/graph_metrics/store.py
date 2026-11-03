"""Disk-backed graph staging. Whole graph property objects never need to fit in RAM."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from .io import dumps, loads, safe_path


class GraphStore:
    def __init__(self, output_dir: Path, *, read_only: bool = False):
        self.path = safe_path(output_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.db = duckdb.connect(str(self.path / "graph.duckdb"), read_only=read_only)
        if not read_only:
            self.db.execute("SET memory_limit = '2GB'")
            self.db.execute("SET threads = 4")
            self.db.execute("SET temp_directory = ?", [str(self.path / "spill")])
            self.db.execute("""CREATE TABLE IF NOT EXISTS nodes (
                id BIGINT PRIMARY KEY, cohort VARCHAR, labels VARCHAR, properties VARCHAR)""")
            self.db.execute("""CREATE TABLE IF NOT EXISTS edges (
                id BIGINT PRIMARY KEY, cohort VARCHAR, source BIGINT, target BIGINT,
                type VARCHAR, properties VARCHAR)""")

    def _insert(self, table: str, rows: list[dict], columns: list[str]):
        if not rows:
            return
        # A registered dataframe-free column relation keeps dependencies and batching small.
        arrays = [
            [dumps(row[k]) if k in ("properties", "labels") else row[k] for row in rows]
            for k in columns
        ]
        select = ", ".join("unnest(?)" for _ in columns)
        self.db.execute(f"INSERT INTO {table} SELECT {select}", arrays)

    def add_nodes(self, rows: list[dict]):
        self._insert("nodes", rows, ["id", "cohort", "labels", "properties"])

    def add_edges(self, rows: list[dict]):
        self._insert("edges", rows, ["id", "cohort", "source", "target", "type", "properties"])

    def _iter(self, table: str, batch_size: int):
        cursor = self.db.cursor()
        cursor.execute(f"SELECT * FROM {table} ORDER BY id")
        names = [c[0] for c in cursor.description]
        try:
            while batch := cursor.fetchmany(batch_size):
                records = []
                for row in batch:
                    item = dict(zip(names, row, strict=True))
                    item["properties"] = loads(item["properties"])
                    if table == "nodes":
                        item["labels"] = json.loads(item["labels"])
                    records.append(item)
                yield records
        finally:
            cursor.close()

    def iter_nodes(self, batch_size=10000):
        return self._iter("nodes", batch_size)

    def iter_edges(self, batch_size=10000):
        return self._iter("edges", batch_size)

    def counts(self) -> tuple[int, int]:
        return (
            self.db.execute("SELECT count(*) FROM nodes").fetchone()[0],
            self.db.execute("SELECT count(*) FROM edges").fetchone()[0],
        )

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
