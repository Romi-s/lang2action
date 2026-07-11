"""Perception eval: score the real SGG backend against simulator ground truth.

Two questions, PredCls-style (the thesis eval discipline):
  1. Detection: which objects does the pipeline find? (id recall / precision)
  2. Relations: on pairs where BOTH objects were detected, does the predicted
     predicate set match geometric ground truth? Per-predicate precision and
     recall - the in_front_of/behind rows are the depth-predicate validation.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field

from lang2action.perception.models import Relation, SceneGraph

PREDICATES: list[Relation] = ["left_of", "right_of", "in_front_of", "behind", "on_top_of"]


@dataclass
class PerceptionReport:
    n_scenes: int = 0
    gt_objects: int = 0
    detected_true: int = 0  # detected ids that exist in ground truth
    detected_false: int = 0  # detected ids that do not exist (false positives)
    # per-predicate counts over ordered co-detected pairs
    tp: dict = field(default_factory=lambda: defaultdict(int))
    fp: dict = field(default_factory=lambda: defaultdict(int))
    fn: dict = field(default_factory=lambda: defaultdict(int))

    @property
    def object_recall(self) -> float:
        return self.detected_true / self.gt_objects if self.gt_objects else 0.0

    @property
    def object_precision(self) -> float:
        found = self.detected_true + self.detected_false
        return self.detected_true / found if found else 0.0

    def predicate_precision(self, predicate: Relation) -> float:
        found = self.tp[predicate] + self.fp[predicate]
        return self.tp[predicate] / found if found else 0.0

    def predicate_recall(self, predicate: Relation) -> float:
        actual = self.tp[predicate] + self.fn[predicate]
        return self.tp[predicate] / actual if actual else 0.0

    def as_markdown(self) -> str:
        lines = [
            "| Measure | Value |",
            "| --- | --- |",
            f"| Object id-recall | {self.object_recall:.0%} "
            f"({self.detected_true}/{self.gt_objects}) |",
            f"| Object id-precision | {self.object_precision:.0%} |",
            "",
            "| Predicate (on co-detected pairs) | Precision | Recall |",
            "| --- | --- | --- |",
        ]
        for predicate in PREDICATES:
            support = self.tp[predicate] + self.fn[predicate]
            lines.append(
                f"| {predicate} (n={support}) | {self.predicate_precision(predicate):.0%} "
                f"| {self.predicate_recall(predicate):.0%} |"
            )
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "n_scenes": self.n_scenes,
                "object_recall": self.object_recall,
                "object_precision": self.object_precision,
                "predicates": {
                    p: {
                        "precision": self.predicate_precision(p),
                        "recall": self.predicate_recall(p),
                        "tp": self.tp[p],
                        "fp": self.fp[p],
                        "fn": self.fn[p],
                    }
                    for p in PREDICATES
                },
            },
            indent=2,
        )


def score_scene(report: PerceptionReport, truth: SceneGraph, perceived: SceneGraph) -> None:
    """Accumulate one scene's scores into the report (pure - no I/O)."""
    report.n_scenes += 1
    true_ids = {o.id for o in truth.objects}
    perceived_ids = {o.id for o in perceived.objects}
    report.gt_objects += len(true_ids)
    report.detected_true += len(true_ids & perceived_ids)
    report.detected_false += len(perceived_ids - true_ids)

    co_detected = true_ids & perceived_ids
    truth_rels = {
        (r.subject_id, r.relation, r.object_id)
        for r in truth.relations
        if r.subject_id in co_detected and r.object_id in co_detected
    }
    perceived_rels = {
        (r.subject_id, r.relation, r.object_id)
        for r in perceived.relations
        if r.subject_id in co_detected and r.object_id in co_detected
    }
    for _, predicate, _ in perceived_rels & truth_rels:
        report.tp[predicate] += 1
    for _, predicate, _ in perceived_rels - truth_rels:
        report.fp[predicate] += 1
    for _, predicate, _ in truth_rels - perceived_rels:
        report.fn[predicate] += 1


def run_perception_eval(
    sgg_url: str, n_scenes: int = 10, base_seed: int = 1000, n_objects: int = 4
) -> PerceptionReport:
    """Score the live SGG service against ground truth over seeded scenes."""
    from lang2action.perception.sgg_http import SggHttpBackend
    from lang2action.perception.sim_backend import SimGroundTruthBackend
    from lang2action.sim import TabletopWorld, generate_scene

    report = PerceptionReport()
    for seed in range(base_seed, base_seed + n_scenes):
        with TabletopWorld() as world:
            world.spawn(generate_scene(seed=seed, n_objects=n_objects))
            truth = SimGroundTruthBackend(world).get_scene_graph()
            perceived = SggHttpBackend(world, sgg_url).get_scene_graph()
        score_scene(report, truth, perceived)
    return report
