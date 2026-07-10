from lang2action.perception import SceneGraph, SceneObject, SpatialRelation
from lang2action.perception.queries import find_objects, spatial_query


def graph() -> SceneGraph:
    return SceneGraph(
        objects=[
            SceneObject(id="red_cube", category="cube", color="red", position=(0, 0, 0.02)),
            SceneObject(id="blue_cube", category="cube", color="blue", position=(0.2, 0, 0.02)),
            SceneObject(id="blue_box", category="box", color="blue", position=(0.4, 0, 0.03)),
        ],
        relations=[
            SpatialRelation(subject_id="red_cube", relation="left_of", object_id="blue_box"),
            SpatialRelation(subject_id="blue_cube", relation="left_of", object_id="blue_box"),
        ],
    )


def test_find_by_color_and_category():
    assert [o.id for o in find_objects(graph(), "the blue cube")] == ["blue_cube"]


def test_find_by_category_only():
    assert [o.id for o in find_objects(graph(), "a cube")] == ["red_cube", "blue_cube"]


def test_find_plural():
    assert [o.id for o in find_objects(graph(), "the cubes")] == ["red_cube", "blue_cube"]


def test_find_absent_object_returns_empty():
    assert find_objects(graph(), "the green cup") == []
    assert find_objects(graph(), "something") == []


def test_find_color_mismatch_still_filters():
    # "red box": red is mentioned and box is mentioned, nothing satisfies both
    assert find_objects(graph(), "the red box") == []


def test_spatial_query():
    assert set(spatial_query(graph(), "blue_box", "left_of")) == {"red_cube", "blue_cube"}
    assert spatial_query(graph(), "red_cube", "on_top_of") == []
