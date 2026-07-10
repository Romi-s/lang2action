"""Headless PyBullet tabletop world.

The table surface is the plane z=0; the world frame follows the convention in
`perception.models` (+x right, +y away from the viewer, +z up).
"""

import pybullet as pb

from lang2action.perception.models import SceneObject
from lang2action.sim.scene_gen import ObjectSpec


class TabletopWorld:
    def __init__(self, gui: bool = False):
        self._client = pb.connect(pb.GUI if gui else pb.DIRECT)
        pb.setGravity(0.0, 0.0, -9.81, physicsClientId=self._client)
        plane = pb.createCollisionShape(pb.GEOM_PLANE, physicsClientId=self._client)
        pb.createMultiBody(0, plane, physicsClientId=self._client)
        self._specs: list[ObjectSpec] = []
        self._bodies: dict[str, int] = {}
        self.on_step = None  # optional callable, invoked after every physics step

    # -- lifecycle -----------------------------------------------------------

    def spawn(self, specs: list[ObjectSpec]) -> None:
        self._specs = list(specs)
        for spec in specs:
            self._bodies[spec.id] = self._create_body(spec)
        self.settle()

    def reset(self) -> None:
        """Remove all objects and respawn the original scene."""
        for body in self._bodies.values():
            pb.removeBody(body, physicsClientId=self._client)
        self._bodies.clear()
        specs, self._specs = self._specs, []
        self.spawn(specs)

    def disconnect(self) -> None:
        if self._client >= 0:
            pb.disconnect(physicsClientId=self._client)
            self._client = -1

    def __enter__(self) -> "TabletopWorld":
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    # -- simulation ----------------------------------------------------------

    def settle(self, steps: int = 240) -> None:
        """Step physics until objects come to rest (240 steps = 1 s)."""
        self.step(steps)

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            pb.stepSimulation(physicsClientId=self._client)
            if self.on_step is not None:
                self.on_step()

    # -- state ---------------------------------------------------------------

    @property
    def client(self) -> int:
        return self._client

    def body_id(self, object_id: str) -> int:
        return self._bodies[object_id]

    def has_object(self, object_id: str) -> bool:
        return object_id in self._bodies

    def object_position(self, object_id: str) -> tuple[float, float, float]:
        pos, _ = pb.getBasePositionAndOrientation(
            self._bodies[object_id], physicsClientId=self._client
        )
        return tuple(pos)

    def scene_objects(self) -> list[SceneObject]:
        """Current object states as scene-graph nodes (ground truth)."""
        return [
            SceneObject(
                id=spec.id,
                category=spec.category,
                color=spec.color,
                position=self.object_position(spec.id),
                extents=spec.half_extents,
            )
            for spec in self._specs
        ]

    def spec(self, object_id: str) -> ObjectSpec:
        for spec in self._specs:
            if spec.id == object_id:
                return spec
        raise KeyError(object_id)

    # -- internals ------------------------------------------------------------

    def _create_body(self, spec: ObjectSpec) -> int:
        he = spec.half_extents
        if spec.category == "cylinder":
            col = pb.createCollisionShape(
                pb.GEOM_CYLINDER, radius=he[0], height=2 * he[2], physicsClientId=self._client
            )
            vis = pb.createVisualShape(
                pb.GEOM_CYLINDER,
                radius=he[0],
                length=2 * he[2],
                rgbaColor=spec.rgba,
                physicsClientId=self._client,
            )
        else:
            col = pb.createCollisionShape(
                pb.GEOM_BOX, halfExtents=he, physicsClientId=self._client
            )
            vis = pb.createVisualShape(
                pb.GEOM_BOX, halfExtents=he, rgbaColor=spec.rgba, physicsClientId=self._client
            )
        body = pb.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=spec.position,
            physicsClientId=self._client,
        )
        pb.changeDynamics(body, -1, lateralFriction=1.0, physicsClientId=self._client)
        return body
