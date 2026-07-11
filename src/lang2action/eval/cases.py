"""Auto-generated eval cases with programmatic ground truth.

Scenes come from the seeded generator, so the true target/relation/reference
triple of every instruction is known by construction - no hand labeling, and
the whole set regenerates deterministically from a base seed.

Case kinds:
  - "execute" cases: the instruction refers to present objects; ground truth is
    the (target_id, relation, reference_id) triple.
  - "refuse" cases: the instruction references an object that is NOT in the
    scene; the correct behavior is refusal (the hallucination-guard metric).
    Ambiguous-instruction refusals are deliberately excluded from v1 scoring -
    grading them requires judgment calls; absent-object cases grade exactly.
"""

import random
from dataclasses import dataclass
from typing import Literal

from lang2action.perception.models import Relation
from lang2action.perception.spatial import infer_relations
from lang2action.sim.scene_gen import COLORS, SIZES, ObjectSpec, generate_scene

RELATION_PHRASES: dict[Relation, str] = {
    "left_of": "to the left of",
    "right_of": "to the right of",
    "in_front_of": "in front of",
    "behind": "behind",
    "on_top_of": "on top of",
}
VERBS = ["put", "place", "move"]


Step = tuple[str, Relation, str]  # (target_id, relation, reference_id)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    scene_seed: int
    n_objects: int
    instruction: str
    expected: Literal["execute", "refuse"]
    expected_steps: tuple[Step, ...] = ()  # empty for refuse-cases


def _noun(spec: ObjectSpec) -> str:
    return f"the {spec.color} {spec.category}"


def _target_phrase(rng: random.Random, specs: list[ObjectSpec], target: ObjectSpec) -> str:
    """Choose a referring expression for the target, from easy to relational."""
    same_category = [s for s in specs if s.category == target.category]
    options = [_noun(target)]
    if len(same_category) == 1:
        options.append(f"the {target.category}")
    else:
        # relational reference: "the cube that is left of the blue box" - only
        # when the relation identifies the target uniquely among its category
        objects = [
            # positions at generation time are the spawn positions
            _spec_to_scene_object(s)
            for s in specs
        ]
        relations = infer_relations(objects)
        for r in relations:
            if r.subject_id != target.id:
                continue
            competitors = [
                other.subject_id
                for other in relations
                if other.relation == r.relation
                and other.object_id == r.object_id
                and other.subject_id != target.id
                and _category_of(specs, other.subject_id) == target.category
            ]
            if not competitors:
                ref_spec = next(s for s in specs if s.id == r.object_id)
                options.append(
                    f"the {target.category} that is "
                    f"{RELATION_PHRASES[r.relation]} {_noun(ref_spec)}"
                )
                break
    return rng.choice(options)


def _category_of(specs: list[ObjectSpec], object_id: str) -> str:
    return next(s.category for s in specs if s.id == object_id)


def _spec_to_scene_object(spec: ObjectSpec):
    from lang2action.perception.models import SceneObject

    return SceneObject(
        id=spec.id,
        category=spec.category,
        color=spec.color,
        position=spec.position,
        extents=spec.half_extents,
    )


def _execute_case(index: int, rng: random.Random) -> EvalCase:
    scene_seed = 1000 + index
    n_objects = rng.choice([3, 4, 5])
    specs = generate_scene(seed=scene_seed, n_objects=n_objects)
    target, reference = rng.sample(specs, 2)
    relation: Relation = rng.choice(list(RELATION_PHRASES))
    target_phrase = _target_phrase(rng, specs, target)
    if relation == "on_top_of" and rng.random() < 0.5:
        instruction = f"stack {target_phrase} on {_noun(reference)}"
    else:
        verb = rng.choice(VERBS)
        instruction = f"{verb} {target_phrase} {RELATION_PHRASES[relation]} {_noun(reference)}"
    return EvalCase(
        case_id=f"case{index:03d}_execute",
        scene_seed=scene_seed,
        n_objects=n_objects,
        instruction=instruction,
        expected="execute",
        expected_steps=((target.id, relation, reference.id),),
    )


def _execute2_case(index: int, rng: random.Random) -> EvalCase:
    """Two-step sequential instruction: '..., then ...' (v2 multi-step)."""
    scene_seed = 1000 + index
    n_objects = rng.choice([4, 5])
    specs = generate_scene(seed=scene_seed, n_objects=n_objects)
    a, b, c = rng.sample(specs, 3)
    relation1: Relation = rng.choice(list(RELATION_PHRASES))
    # step 2 places relative to the object step 1 just moved; keep it planar
    # so the sequence stays physically stable
    relation2: Relation = rng.choice(["left_of", "right_of", "in_front_of", "behind"])
    verb1, verb2 = rng.choice(VERBS), rng.choice(VERBS)
    instruction = (
        f"{verb1} {_noun(a)} {RELATION_PHRASES[relation1]} {_noun(b)}, "
        f"then {verb2} {_noun(c)} {RELATION_PHRASES[relation2]} {_noun(a)}"
    )
    return EvalCase(
        case_id=f"case{index:03d}_execute2",
        scene_seed=scene_seed,
        n_objects=n_objects,
        instruction=instruction,
        expected="execute",
        expected_steps=((a.id, relation1, b.id), (c.id, relation2, a.id)),
    )


def _refuse_case(index: int, rng: random.Random) -> EvalCase:
    scene_seed = 1000 + index
    n_objects = rng.choice([3, 4, 5])
    specs = generate_scene(seed=scene_seed, n_objects=n_objects)
    present_ids = {s.id for s in specs}
    absent = rng.choice(
        [
            (color, category)
            for color in COLORS
            for category in SIZES
            if f"{color}_{category}" not in present_ids
        ]
    )
    absent_noun = f"the {absent[0]} {absent[1]}"
    present = rng.choice(specs)
    relation: Relation = rng.choice(list(RELATION_PHRASES))
    if rng.random() < 0.5:  # absent object as the target...
        instruction = f"put {absent_noun} {RELATION_PHRASES[relation]} {_noun(present)}"
    else:  # ...or as the reference
        instruction = f"put {_noun(present)} {RELATION_PHRASES[relation]} {absent_noun}"
    return EvalCase(
        case_id=f"case{index:03d}_refuse",
        scene_seed=scene_seed,
        n_objects=n_objects,
        instruction=instruction,
        expected="refuse",
    )


def build_cases(
    n_cases: int = 30,
    refusal_fraction: float = 0.27,
    multi_step_fraction: float = 0.17,
    base_seed: int = 0,
):
    """Deterministically build the eval set.

    Default mix for 30 cases: 8 refusals, 5 two-step sequences, 17 single steps.
    """
    rng = random.Random(base_seed)
    n_refuse = round(n_cases * refusal_fraction)
    n_multi = round(n_cases * multi_step_fraction)
    kinds = (
        ["refuse"] * n_refuse
        + ["execute2"] * n_multi
        + ["execute"] * (n_cases - n_refuse - n_multi)
    )
    rng.shuffle(kinds)
    builders = {"refuse": _refuse_case, "execute": _execute_case, "execute2": _execute2_case}
    return [builders[kind](i, rng) for i, kind in enumerate(kinds)]
