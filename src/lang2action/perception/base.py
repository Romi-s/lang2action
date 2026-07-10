"""Perception is an interface: the agent never knows which backend produced the graph.

Implementations (milestone 2):
  - SimGroundTruthBackend: reads object poses directly from the PyBullet world and
    infers relations geometrically (always correct; unblocks agent + eval work).
  - SggHttpBackend: sends a camera render to the Lightweight SGG FastAPI service
    and converts its response (the real perception path used in the demo).
"""

from abc import ABC, abstractmethod

from lang2action.perception.models import SceneGraph


class PerceptionBackend(ABC):
    @abstractmethod
    def get_scene_graph(self) -> SceneGraph:
        """Return the current scene as a structured graph."""
