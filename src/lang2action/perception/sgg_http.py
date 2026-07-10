"""Real perception: render a camera view, send it to the Lightweight SGG
service (my thesis pipeline) over HTTP, convert its response to our schema.

Notes on the mapping:
  - Ids are rebuilt as "{color}_{label}" from the detector label + the color
    attribute, so a correctly perceived object gets the same id the simulator
    uses ("red_cube") and the executor can act on it. A misdetection produces
    an id the world doesn't know - which surfaces as an honest refusal or
    failure, never as silent wrong behavior.
  - Object positions are pseudo-positions from bbox centers (image space
    mapped to table-ish coordinates). The agent grounds on ids + relations;
    the executor reads real poses from the simulator, so these are
    informational only.
  - SGG predicates are canonicalized to our five relations; unknown
    predicates are dropped.
"""

import io

import httpx
from PIL import Image

from lang2action.perception.base import PerceptionBackend
from lang2action.perception.models import Relation, SceneGraph, SceneObject

PREDICATE_MAP: dict[str, Relation] = {
    "on": "on_top_of",
    "on top of": "on_top_of",
    "on the top of": "on_top_of",
    "on the left of": "left_of",
    "left of": "left_of",
    "on the right of": "right_of",
    "right of": "right_of",
    "in front of": "in_front_of",
    "in the front of": "in_front_of",
    "behind": "behind",
}

RENDER_WIDTH = 640
RENDER_HEIGHT = 480


def sgg_to_scene_graph(payload: dict, width: int, height: int) -> SceneGraph:
    """Convert an /analyze response body to our SceneGraph schema."""
    graph = payload["graph"]
    id_map: dict[int, str] = {}
    objects: list[SceneObject] = []
    for obj in graph["objects"]:
        label = str(obj["label"]).strip().lower().replace(" ", "_")
        color = (obj.get("attributes") or {}).get("color", "").strip().lower()
        object_id = f"{color}_{label}" if color else f"{label}_{obj['id']}"
        if any(o.id == object_id for o in objects):  # disambiguate duplicates
            object_id = f"{object_id}_{obj['id']}"
        id_map[obj["id"]] = object_id
        x1, y1, x2, y2 = obj["box_xyxy"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        objects.append(
            SceneObject(
                id=object_id,
                category=label,
                color=color or "unknown",
                # pseudo-position: image center -> origin, ~0.6 m visible table span
                position=((cx / width - 0.5) * 0.6, (0.5 - cy / height) * 0.6, 0.0),
            )
        )
    relations = []
    for rel in graph["relations"]:
        predicate = PREDICATE_MAP.get(str(rel["predicate"]).strip().lower())
        if predicate is None:
            continue
        subject, object_ = id_map.get(rel["subject"]), id_map.get(rel["object"])
        if subject is None or object_ is None or subject == object_:
            continue
        relations.append(
            {"subject_id": subject, "relation": predicate, "object_id": object_}
        )
    return SceneGraph(objects=objects, relations=relations)


class SggHttpBackend(PerceptionBackend):
    def __init__(
        self, world, base_url: str, view: str = "side", client: httpx.Client | None = None
    ):
        self._world = world
        self._base_url = base_url.rstrip("/")
        self._view = view
        self._client = client or httpx.Client(timeout=120.0)

    def get_scene_graph(self) -> SceneGraph:
        from lang2action.sim.camera import render

        image = render(self._world, view=self._view, width=RENDER_WIDTH, height=RENDER_HEIGHT)
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        buffer.seek(0)
        response = self._client.post(
            f"{self._base_url}/analyze",
            files={"file": ("scene.png", buffer, "image/png")},
        )
        response.raise_for_status()
        return sgg_to_scene_graph(response.json(), RENDER_WIDTH, RENDER_HEIGHT)
