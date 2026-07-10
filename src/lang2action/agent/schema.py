"""Structured output of the grounding LLM call."""

from pydantic import BaseModel, Field

from lang2action.perception.models import Relation


class Grounding(BaseModel):
    """The single pick-and-place step an instruction asks for, grounded in the scene."""

    feasible: bool = Field(
        description=(
            "True only if every object the instruction references exists in the scene "
            "graph and the reference is unambiguous. False otherwise."
        )
    )
    target_id: str | None = Field(
        default=None,
        description="Id of the object to pick, copied VERBATIM from the scene graph.",
    )
    relation: Relation | None = Field(
        default=None,
        description="Where to place the target relative to the reference object.",
    )
    reference_id: str | None = Field(
        default=None,
        description="Id of the reference object, copied VERBATIM from the scene graph.",
    )
    reason: str = Field(
        default="",
        description=(
            "When feasible is false: which referenced object is missing or why the "
            "instruction is ambiguous, phrased as a message to the user."
        ),
    )
