"""Geometric inference of pairwise spatial relations from object positions.

Used by the sim ground-truth backend, and by the eval harness to verify task
success from final poses. Pure functions - no simulator dependency.
"""

import math
from collections.abc import Iterable

from lang2action.perception.models import SceneObject, SpatialRelation


def infer_relations(
    objects: Iterable[SceneObject],
    *,
    xy_gap: float = 0.03,
    stack_xy_tol: float = 0.05,
    stack_z_min: float = 0.01,
) -> list[SpatialRelation]:
    """Return all pairwise relations that hold between the given objects.

    xy_gap: minimum axis separation before left/right/front/behind is asserted,
        so near-aligned objects don't generate noisy relations.
    stack_xy_tol: maximum horizontal center distance for "on_top_of".
    stack_z_min: minimum height difference for "on_top_of".
    """
    objs = list(objects)
    relations: list[SpatialRelation] = []
    for a in objs:
        for b in objs:
            if a.id == b.id:
                continue
            ax, ay, az = a.position
            bx, by, bz = b.position

            if az > bz + stack_z_min and math.hypot(ax - bx, ay - by) <= stack_xy_tol:
                relations.append(
                    SpatialRelation(subject_id=a.id, relation="on_top_of", object_id=b.id)
                )
                continue  # a stacked pair is not also left/right of each other

            if ax < bx - xy_gap:
                relations.append(
                    SpatialRelation(subject_id=a.id, relation="left_of", object_id=b.id)
                )
            elif ax > bx + xy_gap:
                relations.append(
                    SpatialRelation(subject_id=a.id, relation="right_of", object_id=b.id)
                )

            if ay < by - xy_gap:
                relations.append(
                    SpatialRelation(subject_id=a.id, relation="in_front_of", object_id=b.id)
                )
            elif ay > by + xy_gap:
                relations.append(
                    SpatialRelation(subject_id=a.id, relation="behind", object_id=b.id)
                )
    return relations
