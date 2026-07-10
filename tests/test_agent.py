"""Agent graph tests with a fake LLM and fake robot - no API key, no simulator."""

import pytest

pytest.importorskip("langgraph")

from lang2action.action.base import ActionResult  # noqa: E402
from lang2action.agent import Grounding, run_agent  # noqa: E402
from lang2action.perception import SceneGraph, SceneObject  # noqa: E402


class FakeLLM:
    """Mimics the .with_structured_output(...).invoke(...) surface."""

    def __init__(self, grounding: Grounding):
        self.grounding = grounding

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return self.grounding


class FakeRobot:
    def __init__(self, graph: SceneGraph, succeed: bool = True):
        self.graph = graph
        self.succeed = succeed
        self.executed = []

    def get_scene_graph(self) -> SceneGraph:
        return self.graph

    def execute(self, action) -> ActionResult:
        self.executed.append(action)
        return ActionResult(success=self.succeed, message="" if self.succeed else "toppled")


def graph() -> SceneGraph:
    return SceneGraph(
        objects=[
            SceneObject(id="red_cube", category="cube", color="red", position=(0, 0, 0.02)),
            SceneObject(id="blue_box", category="box", color="blue", position=(0.2, 0, 0.03)),
        ]
    )


def test_success_path():
    robot = FakeRobot(graph())
    llm = FakeLLM(
        Grounding(
            feasible=True, target_id="red_cube", relation="on_top_of", reference_id="blue_box"
        )
    )
    state = run_agent(robot, llm, "stack the red cube on the blue box")
    assert state["outcome"] == "success"
    assert len(robot.executed) == 1
    assert robot.executed[0].target_id == "red_cube"


def test_hallucinated_id_is_refused():
    robot = FakeRobot(graph())
    llm = FakeLLM(
        Grounding(
            feasible=True, target_id="green_cup", relation="on_top_of", reference_id="blue_box"
        )
    )
    state = run_agent(robot, llm, "put the green cup on the blue box")
    assert state["outcome"] == "refused"
    assert "green_cup" in state["message"]
    assert robot.executed == []  # the guard fired before the executor


def test_llm_reported_infeasible_is_refused():
    robot = FakeRobot(graph())
    llm = FakeLLM(Grounding(feasible=False, reason="there is no green cup in the scene"))
    state = run_agent(robot, llm, "put the green cup on the blue box")
    assert state["outcome"] == "refused"
    assert "green cup" in state["message"]
    assert robot.executed == []


def test_same_target_and_reference_refused():
    robot = FakeRobot(graph())
    llm = FakeLLM(
        Grounding(
            feasible=True, target_id="red_cube", relation="on_top_of", reference_id="red_cube"
        )
    )
    state = run_agent(robot, llm, "put the red cube on itself")
    assert state["outcome"] == "refused"
    assert robot.executed == []


def test_execution_failure_reported():
    robot = FakeRobot(graph(), succeed=False)
    llm = FakeLLM(
        Grounding(
            feasible=True, target_id="red_cube", relation="on_top_of", reference_id="blue_box"
        )
    )
    state = run_agent(robot, llm, "stack the red cube on the blue box")
    assert state["outcome"] == "failed"
    assert "toppled" in state["message"]
