"""What the agent needs from a robot: perceive and act.

The agent never touches PyBullet or HTTP directly - it programs against this
protocol. InProcessRobot composes a PerceptionBackend with an ActionExecutor;
an MCP-backed implementation can satisfy the same protocol later.
"""

from typing import Protocol

from lang2action.action.base import ActionExecutor, ActionResult, PickPlace
from lang2action.perception.base import PerceptionBackend
from lang2action.perception.models import SceneGraph


class Robot(Protocol):
    def get_scene_graph(self) -> SceneGraph: ...

    def execute(self, action: PickPlace) -> ActionResult: ...


class InProcessRobot:
    def __init__(self, perception: PerceptionBackend, executor: ActionExecutor):
        self._perception = perception
        self._executor = executor

    def get_scene_graph(self) -> SceneGraph:
        return self._perception.get_scene_graph()

    def execute(self, action: PickPlace) -> ActionResult:
        return self._executor.execute(action)
