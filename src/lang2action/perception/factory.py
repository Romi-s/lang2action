"""Pick the perception backend from settings - the flag behind the eval's
sim-vs-sgg comparison."""

from lang2action.config import Settings
from lang2action.perception.base import PerceptionBackend


def make_perception(settings: Settings, world) -> PerceptionBackend:
    if settings.perception_backend == "sim":
        from lang2action.perception.sim_backend import SimGroundTruthBackend

        return SimGroundTruthBackend(world)
    if settings.perception_backend == "sgg":
        from lang2action.perception.sgg_http import SggHttpBackend

        return SggHttpBackend(world, settings.sgg_url)
    raise ValueError(
        f"unknown perception backend '{settings.perception_backend}' (expected 'sim' or 'sgg')"
    )
