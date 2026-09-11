"""Training-only future pedestrian trajectory labels.

The Gym observation deliberately contains no future state.  A rollout
collector can pass complete per-tick pedestrian positions to
``TrajectoryTargetRecorder`` and receive delayed supervision targets for a
past tick.  Those labels are suitable for an auxiliary prediction loss, but
are never returned by :class:`~navcore.gym_wrapper.crowd_sim_env.CrowdSimEnv`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class FutureTrajectoryTargets:
    """Future displacement labels for pedestrians seen at one source tick.

    ``displacements[i, k]`` is pedestrian ``pedestrian_ids[i]``'s position at
    source tick + ``k + 1``, minus its position at the source tick. A zero in
    ``valid_mask[i, k]`` means that position was unavailable and the loss for
    that entry must be ignored.
    """

    source_step: int
    pedestrian_ids: tuple[int, ...]
    displacements: np.ndarray
    valid_mask: np.ndarray


class TrajectoryTargetRecorder:
    """Turn a stream of training-only positions into fixed-horizon labels.

    Positions should come from the simulator's complete pedestrian state
    after each tick, rather than from the policy observation.  This lets a
    training pipeline label pedestrians that temporarily leave the robot's
    sensor range without exposing that information to the actor.
    """

    def __init__(self, horizon: int) -> None:
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon!r}.")
        self.horizon = horizon
        self._frames: deque[dict[int, np.ndarray]] = deque(maxlen=horizon + 1)
        self._next_source_step = 0

    def reset(self) -> None:
        """Discard buffered frames at an episode boundary."""
        self._frames.clear()
        self._next_source_step = 0

    def record(
        self, positions: Mapping[int, tuple[float, float] | np.ndarray]
    ) -> FutureTrajectoryTargets | None:
        """Record one simulator frame and return labels once enough exist.

        The first ``horizon`` calls return ``None``. Thereafter each call
        returns labels for exactly one source tick, preserving a one-to-one
        correspondence between policy samples and auxiliary targets.
        """
        frame = {
            pedestrian_id: np.asarray(position, dtype=np.float32).reshape(2).copy()
            for pedestrian_id, position in positions.items()
        }
        self._frames.append(frame)
        if len(self._frames) < self.horizon + 1:
            return None

        source = self._frames[0]
        pedestrian_ids = tuple(sorted(source))
        displacements = np.zeros(
            (len(pedestrian_ids), self.horizon, 2), dtype=np.float32
        )
        valid_mask = np.zeros((len(pedestrian_ids), self.horizon), dtype=np.int8)

        for pedestrian_index, pedestrian_id in enumerate(pedestrian_ids):
            source_position = source[pedestrian_id]
            for future_index, future_frame in enumerate(list(self._frames)[1:]):
                future_position = future_frame.get(pedestrian_id)
                if future_position is None:
                    continue
                displacements[pedestrian_index, future_index] = (
                    future_position - source_position
                )
                valid_mask[pedestrian_index, future_index] = 1

        targets = FutureTrajectoryTargets(
            source_step=self._next_source_step,
            pedestrian_ids=pedestrian_ids,
            displacements=displacements,
            valid_mask=valid_mask,
        )
        self._next_source_step += 1
        return targets
