import pytest
from pydantic import ValidationError

from lang2action.perception import SceneGraph, SceneObject, SpatialRelation


def make_graph() -> SceneGraph:
    return SceneGraph(
        objects=[
            SceneObject(id="red_cube", category="cube", color="red", position=(0.0, 0.0, 0.02)),
            SceneObject(id="blue_box", category="box", color="blue", position=(0.2, 0.0, 0.03)),
        ],
        relations=[
            SpatialRelation(subject_id="red_cube", relation="left_of", object_id="blue_box"),
        ],
    )


def test_lookup():
    g = make_graph()
    assert g.has_object("red_cube")
    assert not g.has_object("green_cup")  # the hallucination-guard check
    assert g.object_by_id("blue_box").color == "blue"
    with pytest.raises(KeyError):
        g.object_by_id("green_cup")


def test_relations_of():
    g = make_graph()
    rels = g.relations_of("red_cube")
    assert len(rels) == 1
    assert rels[0].relation == "left_of"


def test_invalid_relation_rejected():
    with pytest.raises(ValidationError):
        SpatialRelation(subject_id="a", relation="inside_of", object_id="b")


def test_round_trip_json():
    g = make_graph()
    assert SceneGraph.model_validate_json(g.model_dump_json()) == g
