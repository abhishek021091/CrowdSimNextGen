from __future__ import annotations

from abc import ABC, abstractmethod

from .scenario import Scenario


class ScenarioGenerator(ABC):
    """Creates simulation scenarios."""

    @abstractmethod
    def generate(self, episode: int) -> Scenario:
        """Generate one scenario."""
        raise NotImplementedError
