"""End-to-end: agent graph + real PyBullet world (LLM still faked)."""

import pytest

pytest.importorskip("pybullet")
pytest.importorskip("langgraph")

from tests.test_agent import FakeLLM  # noqa: E402

from lang2action.agent import GroundedStep, Grounding, InProcessRobot, run_agent  # noqa: E402
from lang2action.perception.sim_backend import SimGroundTruthBackend  # noqa: E402
from lang2action.perception.spatial import infer_relations  # noqa: E402
from lang2action.sim import COLORS, SIZES, ObjectSpec, PyBulletExecutor, TabletopWorld  # noqa: E402


def spec(id: str, x: float, y: float) -> ObjectSpec:
    color, category = id.split("_")
    he = SIZES[category]
    return ObjectSpec(
        id=id, category=category, color=color, rgba=COLORS[color],
        half_extents=he, position=(x, y, he[2]),
    )


def test_agent_stacks_in_real_sim():
    with TabletopWorld() as world:
        world.spawn([spec("red_cube", -0.1, 0.0), spec("blue_box", 0.1, 0.0)])
        robot = InProcessRobot(SimGroundTruthBackend(world), PyBulletExecutor(world))
        llm = FakeLLM(
            Grounding(
                feasible=True,
                steps=[
                    GroundedStep(
                        target_id="red_cube", relation="on_top_of", reference_id="blue_box"
                    )
                ],
            )
        )
        state = run_agent(robot, llm, "stack the red cube on the blue box")
        assert state["outcome"] == "success", state["message"]
        rels = {
            (r.subject_id, r.relation, r.object_id)
            for r in infer_relations(world.scene_objects())
        }
        assert ("red_cube", "on_top_of", "blue_box") in rels


def test_agent_two_step_sequence_in_real_sim():
    with TabletopWorld() as world:
        world.spawn(
            [
                spec("red_cube", -0.1, 0.0),
                spec("blue_box", 0.1, 0.0),
                spec("green_cylinder", 0.0, 0.18),
            ]
        )
        robot = InProcessRobot(SimGroundTruthBackend(world), PyBulletExecutor(world))
        llm = FakeLLM(
            Grounding(
                feasible=True,
                steps=[
                    GroundedStep(
                        target_id="red_cube", relation="on_top_of", reference_id="blue_box"
                    ),
                    GroundedStep(
                        target_id="green_cylinder", relation="behind", reference_id="blue_box"
                    ),
                ],
            )
        )
        state = run_agent(robot, llm, "stack the red cube on the blue box, then ...")
        assert state["outcome"] == "success", state["message"]
        rels = {
            (r.subject_id, r.relation, r.object_id)
            for r in infer_relations(world.scene_objects())
        }
        assert ("red_cube", "on_top_of", "blue_box") in rels
        assert ("green_cylinder", "behind", "blue_box") in rels
        # the final state's re-perceived graph is in the agent state
        assert state["scene_graph"].has_object("green_cylinder")
