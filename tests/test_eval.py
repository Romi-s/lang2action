"""Eval harness tests: fake LLMs, real sim - verifies grading, not the model."""

import json

import pytest

pytest.importorskip("pybullet")
pytest.importorskip("langgraph")

from lang2action.agent import Grounding  # noqa: E402
from lang2action.eval import build_cases, run_eval  # noqa: E402
from lang2action.eval.runner import run_case  # noqa: E402


class ScriptedLLM:
    """Returns pre-scripted groundings in order (run_eval processes cases in order)."""

    def __init__(self, groundings):
        self._iterator = iter(groundings)

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return next(self._iterator)


def oracle_grounding(case) -> Grounding:
    if case.expected == "refuse":
        return Grounding(feasible=False, reason="referenced object is not in the scene")
    return Grounding(
        feasible=True,
        target_id=case.expected_target,
        relation=case.expected_relation,
        reference_id=case.expected_reference,
    )


# -- case generation (pure) ------------------------------------------------------


def test_cases_are_deterministic():
    assert build_cases(n_cases=30, base_seed=0) == build_cases(n_cases=30, base_seed=0)
    assert build_cases(n_cases=30, base_seed=0) != build_cases(n_cases=30, base_seed=1)


def test_case_mix():
    cases = build_cases(n_cases=30)
    refusals = [c for c in cases if c.expected == "refuse"]
    executes = [c for c in cases if c.expected == "execute"]
    assert len(refusals) == 8
    assert len(executes) == 22
    for case in executes:
        assert case.expected_target and case.expected_relation and case.expected_reference
        assert case.expected_target != case.expected_reference


# -- grading with an oracle (real sim) ---------------------------------------------


def test_oracle_scores_perfectly_on_grounding_and_refusal():
    cases = build_cases(n_cases=6, base_seed=3)
    llm = ScriptedLLM([oracle_grounding(c) for c in cases])
    report = run_eval(cases, llm)
    assert report.grounding_accuracy == 1.0
    assert report.hallucination_refusal_rate == 1.0
    assert report.over_refusal_rate == 0.0
    # physical execution can occasionally be imperfect; it must mostly work
    assert report.task_success_rate > 0.5
    parsed = json.loads(report.to_json())
    assert set(parsed["metrics"]) == {
        "grounding_accuracy",
        "task_success_rate",
        "hallucination_refusal_rate",
        "over_refusal_rate",
    }
    assert "Grounding accuracy" in report.as_markdown()


def test_hallucinated_grounding_is_graded_as_refused():
    case = next(c for c in build_cases(n_cases=10) if c.expected == "execute")
    llm = ScriptedLLM(
        [
            Grounding(
                feasible=True,
                target_id="imaginary_thing",
                relation="on_top_of",
                reference_id=case.expected_reference,
            )
        ]
    )
    result = run_case(case, llm)
    assert result.outcome == "refused"  # guard fired
    assert result.grounding_correct is False
    assert result.task_success is False
