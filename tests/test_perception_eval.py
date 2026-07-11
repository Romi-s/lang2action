"""Pure grading tests for the perception eval - no service, no simulator."""

from lang2action.eval.perception import PerceptionReport, score_scene
from lang2action.perception import SceneGraph, SceneObject, SpatialRelation


def obj(id: str, x: float = 0.0) -> SceneObject:
    color, category = id.split("_")
    return SceneObject(id=id, category=category, color=color, position=(x, 0.0, 0.02))


def rel(subject: str, relation: str, object_: str) -> SpatialRelation:
    return SpatialRelation(subject_id=subject, relation=relation, object_id=object_)


def test_scoring():
    truth = SceneGraph(
        objects=[obj("red_cube", 0.0), obj("blue_box", 0.2), obj("green_cylinder", -0.2)],
        relations=[
            rel("red_cube", "left_of", "blue_box"),
            rel("red_cube", "behind", "blue_box"),
            rel("green_cylinder", "left_of", "red_cube"),  # involves an undetected object
        ],
    )
    perceived = SceneGraph(
        objects=[obj("red_cube", 0.0), obj("blue_box", 0.2), obj("purple_box", 0.1)],
        relations=[
            rel("red_cube", "left_of", "blue_box"),  # true positive
            rel("red_cube", "in_front_of", "blue_box"),  # false positive (truth says behind)
            rel("red_cube", "on_top_of", "purple_box"),  # ignored: purple_box not co-detected
        ],
    )
    report = PerceptionReport()
    score_scene(report, truth, perceived)

    assert report.gt_objects == 3
    assert report.detected_true == 2  # red_cube, blue_box
    assert report.detected_false == 1  # purple_box hallucinated detection
    assert report.object_recall == 2 / 3
    assert report.object_precision == 2 / 3

    assert report.tp["left_of"] == 1
    assert report.fp["in_front_of"] == 1
    assert report.fn["behind"] == 1
    # relation on the undetected green_cylinder is excluded from grading
    assert report.fn["left_of"] == 0
    assert report.predicate_precision("left_of") == 1.0
    assert report.predicate_recall("behind") == 0.0


def test_report_serialization():
    report = PerceptionReport()
    score_scene(
        report,
        SceneGraph(objects=[obj("red_cube")]),
        SceneGraph(objects=[obj("red_cube")]),
    )
    assert "Object id-recall" in report.as_markdown()
    assert '"object_recall": 1.0' in report.to_json()
