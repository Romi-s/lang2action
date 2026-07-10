"""The action layer is an interface so v1 (PyBullet) can later be swapped for
ROS2 + MoveIt (v3) without touching the agent.

Implementations:
  - PyBulletExecutor (milestone 1/2): tabletop pick-and-place in simulation.
  - Ros2MoveItExecutor (v3, planned): real motion-planning stack in Gazebo.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel

from lang2action.perception.models import Relation


class PickPlace(BaseModel):
    """Pick `target_id` and place it in `relation` to `reference_id`."""

    target_id: str
    relation: Relation
    reference_id: str


class ActionResult(BaseModel):
    success: bool
    message: str = ""


class ActionExecutor(ABC):
    @abstractmethod
    def execute(self, action: PickPlace) -> ActionResult:
        """Carry out one pick-and-place step in the world."""

    @abstractmethod
    def reset(self) -> None:
        """Restore the world to a fresh state."""
