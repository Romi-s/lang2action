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
import numpy as np
from PIL import Image, ImageFilter

from lang2action.perception.base import PerceptionBackend
from lang2action.perception.models import Relation, SceneGraph, SceneObject

# The detector runs with an expanded open-vocabulary prompt list (synonyms
# raise recall on synthetic renders); canonicalize its labels back to the
# scene vocabulary so ids align with the simulator's.
CANONICAL_LABELS: dict[str, str] = {
    "block": "cube",
    "toy cube": "cube",
    "dice": "cube",
    "crate": "box",
    "cuboid": "box",
    "can": "cylinder",
    "barrel": "cylinder",
    "toy cylinder": "cylinder",
}

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

RENDER_WIDTH = 1280
RENDER_HEIGHT = 960


def sgg_to_scene_graph(payload: dict, width: int, height: int) -> SceneGraph:
    """Convert an /analyze response body to our SceneGraph schema.

    Synonym prompts make the detector fire multiple boxes on one object, so
    detections are deduplicated per canonical id keeping the highest score
    (scenes in this domain never contain two objects with the same
    color+category, so a duplicate id is always the same physical object).
    """
    graph = payload["graph"]
    id_map: dict[int, str] = {}
    objects: list[SceneObject] = []
    seen_ids: set[str] = set()
    for obj in sorted(graph["objects"], key=lambda o: o.get("score", 0.0), reverse=True):
        raw_label = str(obj["label"]).strip().lower()
        label = CANONICAL_LABELS.get(raw_label, raw_label).replace(" ", "_")
        color = (obj.get("attributes") or {}).get("color", "").strip().lower()
        object_id = f"{color}_{label}" if color else f"{label}_{obj['id']}"
        if object_id in seen_ids:  # duplicate detection of the same object
            continue
        seen_ids.add(object_id)
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
        self, world, base_url: str, view: str = "sgg", client: httpx.Client | None = None
    ):
        self._world = world
        self._base_url = base_url.rstrip("/")
        self._view = view
        self._client = client or httpx.Client(timeout=120.0)

    def get_scene_graph(self) -> SceneGraph:
        from lang2action.sim.camera import render

        image = render(self._world, view=self._view, width=RENDER_WIDTH, height=RENDER_HEIGHT)
        buffer = _photo_like_jpeg(image)
        response = self._client.post(
            f"{self._base_url}/analyze",
            files={"file": ("scene.jpg", buffer, "image/jpeg")},
        )
        response.raise_for_status()
        return sgg_to_scene_graph(response.json(), RENDER_WIDTH, RENDER_HEIGHT)


def _photo_like_jpeg(image: np.ndarray) -> io.BytesIO:
    """Soften the synthetic look before detection: mild blur + sensor noise +
    JPEG compression. Measured effect (10 scenes x 4 objects, with the expanded
    prompt list): 12% -> 75% object id-recall vs sending the raw render."""
    blurred = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(0.8)))
    noise = np.random.default_rng(0).normal(0.0, 7.0, blurred.shape)
    noisy = np.clip(blurred.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(noisy).save(buffer, format="JPEG", quality=70)
    buffer.seek(0)
    return buffer
