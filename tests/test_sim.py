import math

import pytest

pytest.importorskip("pybullet")

from lang2action.action.base import PickPlace  # noqa: E402
from lang2action.perception.sim_backend import SimGroundTruthBackend  # noqa: E402
from lang2action.sim import (  # noqa: E402
    COLORS,
    SIZES,
    ObjectSpec,
    PyBulletExecutor,
    TabletopWorld,
    generate_scene,
)


def spec(id: str, x: float, y: float) -> ObjectSpec:
    color, category = id.split("_")
    he = SIZES[category]
    return ObjectSpec(
        id=id,
        category=category,
        color=color,
        rgba=COLORS[color],
        half_extents=he,
        position=(x, y, he[2]),
    )


@pytest.fixture()
def world():
    w = TabletopWorld()
    w.spawn([spec("red_cube", -0.1, 0.0), spec("blue_box", 0.1, 0.0)])
    yield w
    w.disconnect()


# -- scene generation ----------------------------------------------------------


def test_scene_gen_deterministic():
    assert generate_scene(seed=5) == generate_scene(seed=5)
    assert generate_scene(seed=5) != generate_scene(seed=6)


def test_scene_gen_unique_ids_and_separation():
    specs = generate_scene(seed=11, n_objects=5)
    ids = [s.id for s in specs]
    assert len(set(ids)) == 5
    for i, a in enumerate(specs):
        for b in specs[i + 1 :]:
            assert math.dist(a.position[:2], b.position[:2]) >= 0.13


# -- ground-truth perception -----------------------------------------------------


def test_ground_truth_backend(world):
    graph = SimGroundTruthBackend(world).get_scene_graph()
    assert {o.id for o in graph.objects} == {"red_cube", "blue_box"}
    red = graph.object_by_id("red_cube")
    assert math.dist(red.position[:2], (-0.1, 0.0)) < 0.01  # settled where spawned
    rels = {(r.subject_id, r.relation, r.object_id) for r in graph.relations}
    assert ("red_cube", "left_of", "blue_box") in rels


# -- executor --------------------------------------------------------------------


def test_stack_on_top(world):
    executor = PyBulletExecutor(world)
    result = executor.execute(
        PickPlace(target_id="red_cube", relation="on_top_of", reference_id="blue_box")
    )
    assert result.success, result.message
    graph = SimGroundTruthBackend(world).get_scene_graph()
    rels = {(r.subject_id, r.relation, r.object_id) for r in graph.relations}
    assert ("red_cube", "on_top_of", "blue_box") in rels


def test_place_behind(world):
    executor = PyBulletExecutor(world)
    result = executor.execute(
        PickPlace(target_id="red_cube", relation="behind", reference_id="blue_box")
    )
    assert result.success, result.message


def test_unknown_object_rejected(world):
    executor = PyBulletExecutor(world)
    result = executor.execute(
        PickPlace(target_id="green_cup", relation="on_top_of", reference_id="blue_box")
    )
    assert not result.success
    assert "green_cup" in result.message


def test_reset_restores_scene(world):
    executor = PyBulletExecutor(world)
    executor.execute(
        PickPlace(target_id="red_cube", relation="on_top_of", reference_id="blue_box")
    )
    executor.reset()
    graph = SimGroundTruthBackend(world).get_scene_graph()
    red = graph.object_by_id("red_cube")
    assert math.dist(red.position[:2], (-0.1, 0.0)) < 0.01


# -- camera ----------------------------------------------------------------------


def test_camera_render(world):
    from lang2action.sim.camera import render

    image = render(world, view="top", width=320, height=240)
    assert image.shape == (240, 320, 3)
    assert image.std() > 1.0  # not a blank frame
