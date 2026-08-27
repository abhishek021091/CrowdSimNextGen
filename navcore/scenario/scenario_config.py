"""Scenario configuration: population dynamics over the course of one episode.

A ``ScenarioConfig`` describes the population *recipe* the robot must
contend with during a single, continuous episode -- not a fixed
headcount, but a sequence of ``PopulationStage``s (e.g. "quiet
morning" -> "lunch rush" -> "empty again") that plays out on an
episode-relative clock while the robot is still doing its job. This is
what lets the same robot policy be benchmarked against qualitatively
different crowd conditions -- no groups, all groups, a rush that fades
-- as a config change rather than a code change.

Design notes:
    - Stage boundaries are compositional, not a hard cutover: agents
      spawned under an earlier stage finish their own lifecycle (reach
      goal, or time out) under that stage's terms. Only *new* spawns
      pick up the newly active stage's parameters.
    - Stage timing is measured from the start of the *current episode*
      (episode-relative elapsed time), not across episodes.
      ``Simulation`` resets this clock to zero on every episode reset,
      so consecutive training episodes each replay the same staged
      recipe from stage 0 -- with different random draws each time.
    - ``num_people`` is the population's target *headcount*
      (individuals plus everyone currently inside an active group),
      not a count of spawn events.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SpawnMode = Literal["fixed", "random"]


@dataclass(slots=True, frozen=True)
class PopulationStage:
    """One segment of a scenario's population recipe.

    Attributes:
        num_people: Target steady-state headcount for this stage
            (individuals + everyone in an active group). In
            ``"fixed"`` mode the spawner holds to this value exactly;
            in ``"random"`` mode it is the mean of a wandering
            setpoint.
        std: Standard deviation used to sample a new target headcount
            when ``spawn_mode == "random"``. Ignored in ``"fixed"``
            mode.
        spawn_mode: ``"fixed"`` replaces despawned agents 1-for-1 to
            hold ``num_people`` exactly. ``"random"`` instead chases a
            target resampled from ``Normal(num_people, std)`` each
            time the population catches up to it, so the long-run
            headcount settles to approximately ``num_people +/- std``
            rather than an exact value.
        group_ratio: Fraction of *new spawns* that arrive as a group
            rather than a solo pedestrian, in ``[0.0, 1.0]``.
            ``0.0`` means this stage never spawns groups; ``1.0``
            means it never spawns solo pedestrians.
        duration: How long this stage lasts, in seconds of
            episode-relative time. ``None`` means "runs until the
            episode ends" -- valid only for the last stage in a
            ``ScenarioConfig.stages`` sequence.
        group_size_min: Minimum members in a newly spawned group,
            inclusive. Must be at least 2.
        group_size_max: Maximum members in a newly spawned group,
            inclusive. Must be >= ``group_size_min``.

    Raises:
        ValueError: If any field is out of its documented range.
    """

    num_people: int
    std: float
    spawn_mode: SpawnMode
    group_ratio: float
    duration: float | None = None
    group_size_min: int = 2
    group_size_max: int = 5

    def __post_init__(self) -> None:
        if self.duration is not None and self.duration <= 0.0:
            raise ValueError(f"duration must be positive, got {self.duration!r}.")
        if self.num_people < 0:
            raise ValueError(f"num_people must be non-negative, got {self.num_people!r}.")
        if self.std < 0.0:
            raise ValueError(f"std must be non-negative, got {self.std!r}.")
        if not (0.0 <= self.group_ratio <= 1.0):
            raise ValueError(f"group_ratio must be in [0.0, 1.0], got {self.group_ratio!r}.")
        if self.group_size_min < 2:
            raise ValueError(f"group_size_min must be >= 2, got {self.group_size_min!r}.")
        if self.group_size_max < self.group_size_min:
            raise ValueError(
                f"group_size_max ({self.group_size_max!r}) must be >= "
                f"group_size_min ({self.group_size_min!r})."
            )


@dataclass(slots=True, frozen=True)
class ScenarioConfig:
    """The full population recipe for one episode.

    Attributes:
        stages: The ordered sequence of ``PopulationStage``s played out
            over the episode, on an episode-relative clock. Must
            contain at least one stage. Only the last stage may leave
            ``duration`` as ``None``.
        pedestrian_max_lifetime: Optional safety-net timeout, in
            seconds, after which a pedestrian that has neither reached
            its goal nor otherwise been removed is force-despawned and
            counted as a fresh vacancy. Prevents a pedestrian stuck
            behind an obstacle from permanently shrinking the
            effective headcount. ``None`` disables the safety net.

    Raises:
        ValueError: If ``stages`` is empty, or if any stage other than
            the last has ``duration=None``.
    """

    stages: tuple[PopulationStage, ...]
    pedestrian_max_lifetime: float | None = None

    def __post_init__(self) -> None:
        if len(self.stages) == 0:
            raise ValueError("ScenarioConfig requires at least one stage.")
        for stage in self.stages[:-1]:
            if stage.duration is None:
                raise ValueError(
                    "Only the last stage in a ScenarioConfig may have "
                    "duration=None (open-ended)."
                )

    def stage_at(self, episode_time: float) -> PopulationStage:
        """Return the stage active at ``episode_time`` seconds into the episode.

        Args:
            episode_time: Elapsed time since the current episode began,
                in seconds. Must be non-negative.

        Returns:
            The active ``PopulationStage``. If ``episode_time`` exceeds
            the sum of all finite-duration stages, the last stage is
            returned regardless of its own ``duration``.

        Raises:
            ValueError: If ``episode_time`` is negative.
        """
        if episode_time < 0.0:
            raise ValueError(f"episode_time must be non-negative, got {episode_time!r}.")

        elapsed = 0.0
        for stage in self.stages[:-1]:
            assert stage.duration is not None  # guaranteed by __post_init__
            if episode_time < elapsed + stage.duration:
                return stage
            elapsed += stage.duration
        return self.stages[-1]

    @classmethod
    def from_toml(cls, path: Path) -> "ScenarioConfig":
        """Load a ``ScenarioConfig`` from a TOML file.

        Expects an optional top-level ``pedestrian_max_lifetime`` and
        one or more ``[[stage]]`` tables, in the order they play out.
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)

        stages = tuple(
            PopulationStage(
                num_people=stage["num_people"],
                std=stage["std"],
                spawn_mode=stage["spawn_mode"],
                group_ratio=stage["group_ratio"],
                duration=stage.get("duration"),
                group_size_min=stage.get("group_size_min", 2),
                group_size_max=stage.get("group_size_max", 5),
            )
            for stage in data.get("stage", [])
        )
        return cls(stages=stages, pedestrian_max_lifetime=data.get("pedestrian_max_lifetime"))
