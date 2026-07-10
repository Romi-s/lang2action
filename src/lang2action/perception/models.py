"""Scene-graph schema shared by every perception backend and the agent.

World-frame convention (matches the PyBullet tabletop):
    +x = right, +y = away from the viewer, +z = up.
So "in front of" means smaller y, "behind" means larger y.
"""

from typing import Literal

from pydantic import BaseModel

Relation = Literal["left_of", "right_of", "in_front_of", "behind", "on_top_of"]


class SceneObject(BaseModel):
    id: str
    category: str  # e.g. "cube", "box", "cylinder"
    color: str  # e.g. "red", "blue"
    position: tuple[float, float, float]
    extents: tuple[float, float, float] | None = None  # bounding-box half-extents


class SpatialRelation(BaseModel):
    subject_id: str
    relation: Relation
    object_id: str


class SceneGraph(BaseModel):
    objects: list[SceneObject] = []
    relations: list[SpatialRelation] = []

    def has_object(self, object_id: str) -> bool:
        return any(o.id == object_id for o in self.objects)

    def object_by_id(self, object_id: str) -> SceneObject:
        for o in self.objects:
            if o.id == object_id:
                return o
        raise KeyError(object_id)

    def relations_of(self, subject_id: str) -> list[SpatialRelation]:
        return [r for r in self.relations if r.subject_id == subject_id]
