from __future__ import annotations

import numpy as np

from navcore.scenario.generator import ScenarioGenerator
from navcore.scenario.scenario import Scenario


class RandomScenarioGenerator(ScenarioGenerator):
    def __init__(
        self,
        seed: int = 42,
    ):
        self.seed = seed

    def generate(
        self,
        episode: int,
    ) -> Scenario:

        rng = np.random.default_rng(self.seed + episode)

        raise NotImplementedError
