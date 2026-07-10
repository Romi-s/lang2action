"""Seeded random tabletop scenes.

Each object is a colored primitive with a human-readable id like "red_cube".
(color, category) pairs are unique within a scene, so the id doubles as the
referring-expression ground truth used by the eval harness.
"""

import random
from dataclasses import dataclass

COLORS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.85, 0.10, 0.10, 1.0),
    "green": (0.10, 0.65, 0.15, 1.0),
    "blue": (0.15, 0.25, 0.85, 1.0),
    "yellow": (0.95, 0.85, 0.10, 1.0),
    "purple": (0.55, 0.15, 0.70, 1.0),
    "orange": (0.95, 0.55, 0.10, 1.0),
}

# Half-extents per category; for cylinders x is the radius and z the half-height.
SIZES: dict[str, tuple[float, float, float]] = {
    "cube": (0.025, 0.025, 0.025),
    "box": (0.045, 0.035, 0.030),
    "cylinder": (0.025, 0.025, 0.035),
}

TABLE_BOUND = 0.22  # objects spawn with |x|,|y| below this
MIN_SEPARATION = 0.13  # center-to-center distance between spawned objects


@dataclass(frozen=True)
class ObjectSpec:
    id: str
    category: str
    color: str
    rgba: tuple[float, float, float, float]
    half_extents: tuple[float, float, float]
    position: tuple[float, float, float]


def generate_scene(seed: int, n_objects: int = 4) -> list[ObjectSpec]:
    """Deterministically generate a scene of n unique (color, category) objects."""
    rng = random.Random(seed)
    combos = [(color, cat) for color in COLORS for cat in SIZES]
    chosen = rng.sample(combos, n_objects)

    positions: list[tuple[float, float]] = []
    while len(positions) < n_objects:
        x = rng.uniform(-TABLE_BOUND, TABLE_BOUND)
        y = rng.uniform(-TABLE_BOUND, TABLE_BOUND)
        if all((x - px) ** 2 + (y - py) ** 2 >= MIN_SEPARATION**2 for px, py in positions):
            positions.append((x, y))

    specs = []
    for (color, category), (x, y) in zip(chosen, positions, strict=True):
        he = SIZES[category]
        specs.append(
            ObjectSpec(
                id=f"{color}_{category}",
                category=category,
                color=color,
                rgba=COLORS[color],
                half_extents=he,
                position=(x, y, he[2]),  # resting on the table surface (z=0)
            )
        )
    return specs
