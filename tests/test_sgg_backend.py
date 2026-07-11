import httpx
import pytest

from lang2action.perception.sgg_http import sgg_to_scene_graph

PAYLOAD = {
    "summary": "2 objects, 1 relation",
    "graph": {
        "objects": [
            {
                "id": 0,
                "label": "cube",
                "score": 0.91,
                "box_xyxy": [100, 100, 160, 160],
                "attributes": {"color": "red"},
            },
            {
                "id": 1,
                "label": "box",
                "score": 0.88,
                "box_xyxy": [400, 120, 500, 200],
                "attributes": {"color": "blue"},
            },
            {
                "id": 2,
                "label": "cylinder",
                "score": 0.75,
                "box_xyxy": [300, 300, 340, 380],
                "attributes": {},
            },
        ],
        "relations": [
            {"subject": 0, "predicate": "on the left of", "object": 1, "score": 0.9},
            {"subject": 0, "predicate": "touching", "object": 1, "score": 0.8},  # unmapped
            {"subject": 1, "predicate": "on top of", "object": 2, "score": 0.7},
        ],
        "triplets": [],
    },
}


def test_mapping_ids_and_predicates():
    graph = sgg_to_scene_graph(PAYLOAD, width=640, height=480)
    ids = {o.id for o in graph.objects}
    assert {"red_cube", "blue_box"} <= ids
    assert any(o.id.startswith("cylinder_") for o in graph.objects)  # no color attribute
    rels = {(r.subject_id, r.relation, r.object_id) for r in graph.relations}
    assert ("red_cube", "left_of", "blue_box") in rels
    assert ("blue_box", "on_top_of") in {(r.subject_id, r.relation) for r in graph.relations}
    assert len(graph.relations) == 2  # "touching" dropped


def test_synonym_labels_canonicalized_and_deduplicated():
    payload = {
        "graph": {
            "objects": [
                # same physical object fired twice under synonym prompts:
                # keep the higher-scoring detection, canonicalize the label
                {"id": 0, "label": "block", "score": 0.4, "box_xyxy": [100, 100, 160, 160],
                 "attributes": {"color": "red"}},
                {"id": 1, "label": "cube", "score": 0.9, "box_xyxy": [102, 101, 158, 161],
                 "attributes": {"color": "red"}},
                {"id": 2, "label": "can", "score": 0.5, "box_xyxy": [300, 300, 340, 380],
                 "attributes": {"color": "green"}},
            ],
            "relations": [],
            "triplets": [],
        }
    }
    graph = sgg_to_scene_graph(payload, width=640, height=480)
    assert [o.id for o in graph.objects] == ["red_cube", "green_cylinder"]
    assert graph.object_by_id("green_cylinder").category == "cylinder"


def test_mapping_pseudo_positions_ordered():
    graph = sgg_to_scene_graph(PAYLOAD, width=640, height=480)
    red = graph.object_by_id("red_cube")
    blue = graph.object_by_id("blue_box")
    assert red.position[0] < blue.position[0]  # left in image -> smaller x


def test_http_backend_round_trip():
    pytest.importorskip("pybullet")
    from lang2action.perception.sgg_http import SggHttpBackend
    from lang2action.sim import TabletopWorld, generate_scene

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/analyze"
        assert b"image/jpeg" in request.read()  # photo-like JPEG, not the raw render
        return httpx.Response(200, json=PAYLOAD)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with TabletopWorld() as world:
        world.spawn(generate_scene(seed=1, n_objects=3))
        backend = SggHttpBackend(world, "http://sgg:8000", client=client)
        graph = backend.get_scene_graph()
    assert graph.has_object("red_cube")
