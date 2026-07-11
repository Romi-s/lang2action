"""Structured output of the grounding LLM call."""

from pydantic import BaseModel, Field

from lang2action.perception.models import Relation


class GroundedStep(BaseModel):
    """One pick-and-place step, grounded in the scene."""

    target_id: str = Field(
        description="Id of the object to pick, copied VERBATIM from the scene graph."
    )
    relation: Relation = Field(
        description="Where to place the target relative to the reference object."
    )
    reference_id: str = Field(
        description="Id of the reference object, copied VERBATIM from the scene graph."
    )


class Grounding(BaseModel):
    """The ordered pick-and-place steps an instruction asks for (v2: multi-step)."""

    feasible: bool = Field(
        description=(
            "True only if every object the instruction references exists in the scene "
            "graph and the reference is unambiguous. False otherwise."
        )
    )
    steps: list[GroundedStep] = Field(
        default_factory=list,
        description=(
            "The steps in execution order. A single-step instruction produces one step; "
            "'do X, then Y' produces two. Empty when feasible is false."
        ),
    )
    reason: str = Field(
        default="",
        description=(
            "When feasible is false: which referenced object is missing or why the "
            "instruction is ambiguous, phrased as a message to the user."
        ),
    )
