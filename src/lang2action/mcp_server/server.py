"""Scene-Graph MCP server: the simulated robot as a set of MCP tools.

The server owns the PyBullet world. Any MCP client (the LangGraph agent,
Claude Code, the MCP inspector) perceives through the scene-graph tools and
acts through the pick-and-place tool - so the client is the brain, this
process is the robot.

Environment:
    LANG2ACTION_SCENE_SEED (default 42)     seed for the generated tabletop
    LANG2ACTION_SCENE_OBJECTS (default 4)   number of objects
    LANG2ACTION_PERCEPTION (default "sim")  "sim" ground truth; "sgg" (real
                                            perception over HTTP) lands with
                                            milestone 4
"""

import os
from typing import Literal

from mcp.server.fastmcp import FastMCP

from lang2action.config import load_settings

mcp = FastMCP(
    "lang2action-robot",
    instructions=(
        "A simulated tabletop robot. Perceive the scene with get_scene_graph / "
        "find_object / spatial_query, then act with execute_pick_place. Object ids "
        "look like 'red_cube'. Only reference ids that exist in the scene graph."
    ),
)

_state: dict = {}


def _robot() -> dict:
    """Lazily build the world, perception backend, and executor (once)."""
    if not _state:
        from lang2action.perception.factory import make_perception
        from lang2action.sim import PyBulletExecutor, TabletopWorld, generate_scene

        settings = load_settings()
        seed = int(os.getenv("LANG2ACTION_SCENE_SEED", "42"))
        n_objects = int(os.getenv("LANG2ACTION_SCENE_OBJECTS", "4"))
        world = TabletopWorld()
        world.spawn(generate_scene(seed=seed, n_objects=n_objects))
        _state["world"] = world
        _state["backend"] = make_perception(settings, world)
        _state["executor"] = PyBulletExecutor(world)
    return _state


@mcp.tool()
def get_scene_graph() -> dict:
    """Return the current tabletop scene as a structured graph:
    objects (id, category, color, position) and pairwise spatial relations."""
    return _robot()["backend"].get_scene_graph().model_dump()


@mcp.tool()
def find_object(description: str) -> dict:
    """Find objects matching a natural-language description (color/category match,
    e.g. 'the blue cube'). Returns {"matches": [...]}; empty when nothing fits."""
    from lang2action.perception.queries import find_objects

    graph = _robot()["backend"].get_scene_graph()
    return {"matches": [o.model_dump() for o in find_objects(graph, description)]}


@mcp.tool()
def spatial_query(
    reference_id: str,
    relation: Literal["left_of", "right_of", "in_front_of", "behind", "on_top_of"],
) -> dict:
    """Ids of objects standing in `relation` to the reference object
    (e.g. reference_id='blue_box', relation='left_of' -> everything left of it)."""
    from lang2action.perception.queries import spatial_query as query

    graph = _robot()["backend"].get_scene_graph()
    if not graph.has_object(reference_id):
        return {"error": f"object not in scene: {reference_id}", "matches": []}
    return {"matches": query(graph, reference_id, relation)}


@mcp.tool()
def execute_pick_place(
    target_id: str,
    relation: Literal["left_of", "right_of", "in_front_of", "behind", "on_top_of"],
    reference_id: str,
) -> dict:
    """Pick `target_id` and place it in `relation` to `reference_id`
    (e.g. red_cube on_top_of blue_box). Success is physically verified after
    the world settles."""
    from lang2action.action.base import PickPlace

    result = _robot()["executor"].execute(
        PickPlace(target_id=target_id, relation=relation, reference_id=reference_id)
    )
    return result.model_dump()


@mcp.tool()
def reset_scene() -> dict:
    """Restore the tabletop to its initial (seeded) configuration."""
    _robot()["executor"].reset()
    return {"ok": True}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
