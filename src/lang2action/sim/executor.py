"""Pick-and-place in PyBullet via a "magic grasp".

A fixed constraint attaches the target to a virtual gripper point, which is
moved along a lift -> traverse -> lower trajectory while physics steps. After
release the world settles and the executor *verifies* that the requested
relation actually holds, so ActionResult.success reflects physical reality,
not intent. (v3 swaps this class for a ROS2 + MoveIt implementation.)
"""

import math

import pybullet as pb

from lang2action.action.base import ActionExecutor, ActionResult, PickPlace
from lang2action.perception.models import Relation
from lang2action.perception.spatial import infer_relations
from lang2action.sim.world import TabletopWorld

PLACE_GAP = 0.12  # planar center-to-center distance for left_of/right_of/...
CARRY_HEIGHT = 0.25
STEP_SIZE = 0.01  # meters of gripper travel per simulation step


class PyBulletExecutor(ActionExecutor):
    def __init__(self, world: TabletopWorld):
        self._world = world

    def execute(self, action: PickPlace) -> ActionResult:
        for object_id in (action.target_id, action.reference_id):
            if not self._world.has_object(object_id):
                return ActionResult(
                    success=False, message=f"object not in scene: {object_id}"
                )
        if action.target_id == action.reference_id:
            return ActionResult(success=False, message="target and reference are the same object")

        destination = self._destination(action)
        self._carry(action.target_id, destination)
        self._world.settle()

        if self._relation_holds(action):
            return ActionResult(success=True)
        return ActionResult(
            success=False,
            message=f"placed, but '{action.target_id} {action.relation} "
            f"{action.reference_id}' does not hold after settling",
        )

    def reset(self) -> None:
        self._world.reset()

    # -- internals -------------------------------------------------------------

    def _destination(self, action: PickPlace) -> tuple[float, float, float]:
        ref = self._world.object_position(action.reference_id)
        target_hz = self._world.spec(action.target_id).half_extents[2]
        ref_hz = self._world.spec(action.reference_id).half_extents[2]
        relation: Relation = action.relation
        if relation == "on_top_of":
            return (ref[0], ref[1], ref[2] + ref_hz + target_hz + 0.005)
        offsets = {
            "left_of": (-PLACE_GAP, 0.0),
            "right_of": (PLACE_GAP, 0.0),
            "in_front_of": (0.0, -PLACE_GAP),
            "behind": (0.0, PLACE_GAP),
        }
        dx, dy = offsets[relation]
        return (ref[0] + dx, ref[1] + dy, target_hz)

    def _carry(self, object_id: str, destination: tuple[float, float, float]) -> None:
        body = self._world.body_id(object_id)
        start = self._world.object_position(object_id)
        grip = pb.createConstraint(
            parentBodyUniqueId=body,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=pb.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=start,
            physicsClientId=self._world.client,
        )
        try:
            waypoints = [
                (start[0], start[1], CARRY_HEIGHT),
                (destination[0], destination[1], CARRY_HEIGHT),
                (destination[0], destination[1], destination[2] + 0.01),
            ]
            current = start
            for waypoint in waypoints:
                for point in _interpolate(current, waypoint, STEP_SIZE):
                    pb.changeConstraint(
                        grip, jointChildPivot=point, maxForce=50,
                        physicsClientId=self._world.client,
                    )
                    self._world.step()
                current = waypoint
            self._world.step(30)
        finally:
            pb.removeConstraint(grip, physicsClientId=self._world.client)

    def _relation_holds(self, action: PickPlace) -> bool:
        relations = infer_relations(self._world.scene_objects())
        return any(
            r.subject_id == action.target_id
            and r.relation == action.relation
            and r.object_id == action.reference_id
            for r in relations
        )


def _interpolate(a, b, step_size):
    distance = math.dist(a, b)
    steps = max(1, int(distance / step_size))
    for i in range(1, steps + 1):
        t = i / steps
        yield tuple(a[k] + (b[k] - a[k]) * t for k in range(3))
