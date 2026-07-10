"""Ground-truth perception: read poses straight from the simulator.

This backend is always correct, which lets the eval isolate reasoning errors
(agent) from perception errors (the real SGG backend, milestone 3).
"""

from lang2action.perception.base import PerceptionBackend
from lang2action.perception.models import SceneGraph
from lang2action.perception.spatial import infer_relations
from lang2action.sim.world import TabletopWorld


class SimGroundTruthBackend(PerceptionBackend):
    def __init__(self, world: TabletopWorld):
        self._world = world

    def get_scene_graph(self) -> SceneGraph:
        objects = self._world.scene_objects()
        return SceneGraph(objects=objects, relations=infer_relations(objects))
