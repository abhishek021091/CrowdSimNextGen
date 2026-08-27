"""RenderFrame: the one-way, read-only snapshot handed to a Renderer.

A ``Renderer`` never sees ``Agent``, ``Policy``, config dicts, or
anything else internal to the simulation -- only geometry and labels,
captured once per tick by ``Simulation``. This is what lets the exact
same renderer implementation draw a headless-kinematic run, a
Gazebo-physics-validated run, and (eventually) a live feed from a real
robot without modification: whichever ``ExecutionBackend`` produced a
pose, it ends up in the same array, in the same shape.

Positions are stored as a NumPy structure-of-arrays rather than a list
of per-agent objects. At mall/hospital/house scale (tens of agents,
not thousands) this isn't a performance necessity -- it would be a
premature optimization at that scale -- but it costs nothing extra to
write it this way now and keeps the boundary consistent if a scenario
ever needs a much larger crowd later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from navcore.entities.obstacles.obstacle import Obstacle


@dataclass(slots=True, frozen=True)
class RenderFrame:
    """An immutable, renderer-agnostic snapshot of one simulation tick.

    Attributes:
        sim_time: Elapsed time since the current episode began, in
            seconds.
        step: The tick count since the current episode began.
        episode: The episode index since this run started.
        agent_positions: ``(N, 2)`` array of ``(x, y)`` world
            positions, one row per agent.
        agent_radii: ``(N,)`` array of agent radii, aligned with
            ``agent_positions``.
        agent_kinds: Length-``N`` tuple of human-readable agent kinds
            (``"Robot"``, ``"Pedestrian"``), aligned with
            ``agent_positions``. Used for renderer-side styling only.
        obstacles: The episode's static obstacles. Safe to treat as
            constant across an entire episode -- obstacle layout never
            changes mid-episode by design.
    """

    sim_time: float
    step: int
    episode: int
    agent_positions: np.ndarray
    agent_radii: np.ndarray
    agent_kinds: tuple[str, ...]
    obstacles: tuple[Obstacle, ...]
