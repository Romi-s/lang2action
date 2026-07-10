"""Deterministic queries over a scene graph.

These are lexical helpers exposed as MCP tools. The LLM agent does the actual
referring-expression grounding; the hallucination guard then verifies that the
chosen id exists in the graph, so these helpers never need to be clever - only
predictable.
"""

import re

from lang2action.perception.models import Relation, SceneGraph, SceneObject


def _stems(token: str):
    yield token
    if token.endswith("es"):
        yield token[:-2]
    if token.endswith("s"):
        yield token[:-1]


def _mentioned(description: str, vocabulary: set[str]) -> set[str]:
    tokens = re.findall(r"[a-z]+", description.lower())
    return {stem for token in tokens for stem in _stems(token) if stem in vocabulary}


def find_objects(graph: SceneGraph, description: str) -> list[SceneObject]:
    """Objects whose color/category are compatible with the description.

    Unrecognized descriptions return no matches (never "everything"), so a
    query for an absent object comes back empty - which is what the agent's
    refusal path needs to see.
    """
    scene_colors = {o.color for o in graph.objects}
    scene_categories = {o.category for o in graph.objects}
    colors = _mentioned(description, scene_colors)
    categories = _mentioned(description, scene_categories)
    if not colors and not categories:
        return []
    return [
        o
        for o in graph.objects
        if (not colors or o.color in colors) and (not categories or o.category in categories)
    ]


def spatial_query(graph: SceneGraph, reference_id: str, relation: Relation) -> list[str]:
    """Ids of objects standing in `relation` to the reference object.

    Example: spatial_query(g, "blue_box", "left_of") -> everything left of the blue box.
    """
    return [
        r.subject_id
        for r in graph.relations
        if r.object_id == reference_id and r.relation == relation
    ]
