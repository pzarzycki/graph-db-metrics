"""The shared, versioned interchange format. YAML never contains executable tags."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TableRef(Model):
    path: str
    rows: int = Field(ge=0)
    sha256: str
    format: Literal["parquet"] = "parquet"


class Metric(Model):
    id: str
    definition: str
    scope: dict[str, Any] = Field(default_factory=dict)
    denominator: Any = None
    value: Any = None
    estimator: dict[str, Any] = Field(default_factory=lambda: {"method": "exact scan"})
    uncertainty: dict[str, Any] | None = None
    status: Literal["exact", "estimated", "not_applicable", "not_measured", "ontology_required"]
    reason: str | None = None


class PropertySpec(Model):
    type: str = "string"
    presence: float = Field(default=1.0, ge=0, le=1)
    distribution: dict[str, Any] = Field(default_factory=dict)


class NodeSpec(Model):
    id: str
    labels: list[str]
    count: int = Field(ge=0)
    properties: dict[str, PropertySpec] = Field(default_factory=dict)
    patterns: list[dict[str, Any]] | TableRef = Field(default_factory=list)
    degrees: list[dict[str, Any]] | TableRef = Field(default_factory=list)
    roles: list[dict[str, Any]] | TableRef = Field(default_factory=list)

    @model_validator(mode="after")
    def distinct_labels(self):
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("node labels must be distinct")
        if not self.id or any(not label for label in self.labels):
            raise ValueError("node cohort ids and label names must be nonempty")
        return self


class RelationshipSpec(Model):
    id: str
    type: str
    source: str
    target: str
    count: int = Field(ge=0)
    properties: dict[str, PropertySpec] = Field(default_factory=dict)
    patterns: list[dict[str, Any]] | TableRef = Field(default_factory=list)
    out_degrees: list[dict[str, Any]] | TableRef = Field(default_factory=list)
    in_degrees: list[dict[str, Any]] | TableRef = Field(default_factory=list)


class Observations(Model):
    node_count: int = Field(default=0, ge=0)
    relationship_count: int = Field(default=0, ge=0)
    nodes: list[NodeSpec] = Field(default_factory=list)
    relationships: list[RelationshipSpec] = Field(default_factory=list)


class Rule(Model):
    id: str
    kind: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    provenance: Literal["declared", "inferred", "database"] = "declared"
    hard: bool = True
    description: str = ""


class Semantics(Model):
    status: Literal["unavailable", "inferred", "declared", "mixed"] = "unavailable"
    concepts: list[dict[str, Any]] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    ontology: dict[str, Any] | None = None


class Targets(Model):
    nodes: list[NodeSpec] = Field(default_factory=list)
    relationships: list[RelationshipSpec] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    scale: float = Field(default=1, gt=0)
    tolerances: dict[str, float] = Field(
        default_factory=lambda: {"discrete_tv": 0.1, "numeric_nrmse": 0.15}
    )
    allow_self_loops: bool | None = None
    allow_parallel_edges: bool | None = None
    refinement_rounds: int = Field(default=3, ge=0, le=20)


class Preparation(Model):
    provider_path: str | None = None
    provider_sha256: str | None = None
    profile_digest: str | None = None
    recipe: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None
    prepared_at: str | None = None


class Profile(Model):
    schema_version: Literal["1"] = "1"
    source: dict[str, Any] = Field(default_factory=dict)
    scope: dict[str, Any] = Field(
        default_factory=lambda: {
            "direction": "directed",
            "multiplicity": "retain",
            "cohorts": "exact_label_set",
        }
    )
    observations: Observations = Field(default_factory=Observations)
    metrics: list[Metric] = Field(default_factory=list)
    semantics: Semantics = Field(default_factory=Semantics)
    targets: Targets = Field(default_factory=Targets)
    preparation: Preparation = Field(default_factory=Preparation)
    samples: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_cohorts(self):
        for nodes, relationships in [
            (self.observations.nodes, self.observations.relationships),
            (
                self.targets.nodes or self.observations.nodes,
                self.targets.relationships or self.observations.relationships,
            ),
        ]:
            identifiers = [n.id for n in nodes]
            if len(set(identifiers)) != len(identifiers):
                raise ValueError("duplicate node cohort ids")
            label_sets = [tuple(sorted(n.labels)) for n in nodes]
            if len(set(label_sets)) != len(label_sets):
                raise ValueError("node cohorts must have distinct complete label sets")
            edge_ids = [r.id for r in relationships]
            if len(set(edge_ids)) != len(edge_ids):
                raise ValueError("duplicate relationship cohort ids")
            signatures = [(r.source, r.type, r.target) for r in relationships]
            if len(set(signatures)) != len(signatures):
                raise ValueError(
                    "relationship cohorts must have distinct source/type/target signatures"
                )
            for rel in relationships:
                if not rel.id or not rel.type:
                    raise ValueError("relationship cohort ids and types must be nonempty")
                if rel.source not in identifiers or rel.target not in identifiers:
                    raise ValueError(f"relationship {rel.id!r} references an unknown node cohort")
        return self

    def generation_specs(self) -> tuple[list[NodeSpec], list[RelationshipSpec]]:
        return (
            self.targets.nodes or self.observations.nodes,
            self.targets.relationships or self.observations.relationships,
        )
