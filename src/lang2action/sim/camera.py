"""Camera renders of the tabletop, for the SGG perception path and demo GIFs."""

import numpy as np
import pybullet as pb
from PIL import Image

from lang2action.sim.world import TabletopWorld

VIEWS = {
    # eye position, look-at target
    "top": ((0.0, 0.0, 0.9), (0.0, 0.0, 0.0)),
    "side": ((0.0, -0.85, 0.45), (0.0, 0.0, 0.05)),
}


def render(
    world: TabletopWorld, view: str = "top", width: int = 640, height: int = 480
) -> np.ndarray:
    """Render the scene to an RGB uint8 array of shape (height, width, 3)."""
    eye, target = VIEWS[view]
    view_matrix = pb.computeViewMatrix(
        cameraEyePosition=eye,
        cameraTargetPosition=target,
        cameraUpVector=(0.0, 1.0, 0.0) if view == "top" else (0.0, 0.0, 1.0),
        physicsClientId=world.client,
    )
    proj_matrix = pb.computeProjectionMatrixFOV(
        fov=60.0, aspect=width / height, nearVal=0.01, farVal=2.0
    )
    _, _, rgba, _, _ = pb.getCameraImage(
        width,
        height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=pb.ER_TINY_RENDERER,
        physicsClientId=world.client,
    )
    rgba = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)
    return rgba[:, :, :3].copy()


def save_png(image: np.ndarray, path: str) -> None:
    Image.fromarray(image).save(path)
