"""Run the eval set through the agent and grade against ground truth.

Metrics:
  - grounding accuracy: execute-cases where the LLM's (target, relation,
    reference) triple matches ground truth exactly
  - task success rate: execute-cases that ended in a physically verified
    successful placement (the executor re-checks the relation after settling)
  - hallucination-refusal rate: refuse-cases the agent actually refused
  - over-refusal rate (diagnostic): execute-cases wrongly refused
"""

import json
from dataclasses import asdict, dataclass, field

from lang2action.agent import InProcessRobot, run_agent
from lang2action.eval.cases import EvalCase
from lang2action.perception.sim_backend import SimGroundTruthBackend
from lang2action.sim import PyBulletExecutor, TabletopWorld, generate_scene


@dataclass
class CaseResult:
    case_id: str
    instruction: str
    expected: str
    outcome: str
    message: str
    grounding_correct: bool | None  # None for refuse-cases
    task_success: bool | None  # None for refuse-cases
    refused_correctly: bool | None  # None for execute-cases


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)

    @property
    def grounding_accuracy(self) -> float:
        rows = [r.grounding_correct for r in self.results if r.grounding_correct is not None]
        return sum(rows) / len(rows) if rows else 0.0

    @property
    def task_success_rate(self) -> float:
        rows = [r.task_success for r in self.results if r.task_success is not None]
        return sum(rows) / len(rows) if rows else 0.0

    @property
    def hallucination_refusal_rate(self) -> float:
        rows = [r.refused_correctly for r in self.results if r.refused_correctly is not None]
        return sum(rows) / len(rows) if rows else 0.0

    @property
    def over_refusal_rate(self) -> float:
        rows = [r for r in self.results if r.expected == "execute"]
        return sum(r.outcome == "refused" for r in rows) / len(rows) if rows else 0.0

    def as_markdown(self) -> str:
        n_exec = sum(r.expected == "execute" for r in self.results)
        n_refuse = len(self.results) - n_exec
        return "\n".join(
            [
                "| Metric | Value |",
                "| --- | --- |",
                f"| Grounding accuracy ({n_exec} execute-cases) "
                f"| {self.grounding_accuracy:.0%} |",
                f"| Task success rate | {self.task_success_rate:.0%} |",
                f"| Hallucination-refusal rate ({n_refuse} refuse-cases) "
                f"| {self.hallucination_refusal_rate:.0%} |",
                f"| Over-refusal rate (diagnostic) | {self.over_refusal_rate:.0%} |",
            ]
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "metrics": {
                    "grounding_accuracy": self.grounding_accuracy,
                    "task_success_rate": self.task_success_rate,
                    "hallucination_refusal_rate": self.hallucination_refusal_rate,
                    "over_refusal_rate": self.over_refusal_rate,
                },
                "cases": [asdict(r) for r in self.results],
            },
            indent=2,
        )


def run_case(case: EvalCase, llm) -> CaseResult:
    with TabletopWorld() as world:
        world.spawn(generate_scene(seed=case.scene_seed, n_objects=case.n_objects))
        robot = InProcessRobot(SimGroundTruthBackend(world), PyBulletExecutor(world))
        state = run_agent(robot, llm, case.instruction)

    outcome = state["outcome"]
    grounding = state.get("grounding")
    if case.expected == "refuse":
        return CaseResult(
            case_id=case.case_id,
            instruction=case.instruction,
            expected=case.expected,
            outcome=outcome,
            message=state.get("message", ""),
            grounding_correct=None,
            task_success=None,
            refused_correctly=outcome == "refused",
        )
    grounding_correct = (
        grounding is not None
        and grounding.feasible
        and grounding.target_id == case.expected_target
        and grounding.relation == case.expected_relation
        and grounding.reference_id == case.expected_reference
    )
    return CaseResult(
        case_id=case.case_id,
        instruction=case.instruction,
        expected=case.expected,
        outcome=outcome,
        message=state.get("message", ""),
        grounding_correct=grounding_correct,
        task_success=outcome == "success",  # executor already verified physically
        refused_correctly=None,
    )


def run_eval(cases: list[EvalCase], llm, on_result=None) -> EvalReport:
    report = EvalReport()
    for case in cases:
        result = run_case(case, llm)
        report.results.append(result)
        if on_result is not None:
            on_result(result)
    return report
