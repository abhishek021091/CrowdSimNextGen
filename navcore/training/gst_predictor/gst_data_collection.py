# navcore/training/gst_data_collection.py
"""GSTDataCollector: builds (history, future) pedestrian trajectory
windows from ground-truth ORCA-only crowd simulation, for pretraining
GSTPredictor.

Ground-truth, not sensor-limited, on purpose:
    The project's "planners only see ObservableState" boundary (see
    LocalAvoidancePlanner's module docstring) governs what a *policy*
    may read at decision time -- it says nothing about what a
    *separately trained* prediction network may be trained against.
    GST is pretrained offline on full ground-truth trajectories, then
    frozen before ever touching the RL policy (see GSTPredictorTrainer).

Robot-free by design:
    Episodes here run a pure ORCA-driven crowd with no robot policy in
    the loop -- predicting "how humans move around each other" doesn't
    depend on a not-yet-trained robot, and training against one would
    create a chicken-and-egg dependency this pipeline deliberately avoids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from navcore.builder.environment_builder import EnvironmentBuilder
from navcore.middleware.orca_middleware import DecentralizedORCAPlanner
from navcore.step.step import Step


@dataclass(slots=True)
class GSTSample:
    history_positions: np.ndarray  # [max_agents, obs_length, 2]
    history_velocity: np.ndarray  # [max_agents, obs_length, 2]
    history_mask: np.ndarray  # [max_agents, obs_length]
    future_displacement: np.ndarray  # [max_agents, pred_length, 2]
    future_mask: np.ndarray  # [max_agents, pred_length]


class GSTDataCollector:
    """Rolls out ORCA-only crowd episodes and slices fixed-length windows.

    Attributes:
        max_agents: Fixed pedestrian-slot count, same padding convention
            as ObservationEncoder's max_neighbors.
        obs_length: History window length.
        pred_length: Future window length.
    """

    def __init__(
        self,
        max_agents: int = 20,
        obs_length: int = 8,
        pred_length: int = 5,
        orca_config_file: str = "orca.toml",
        rand: np.random.Generator | None = None,
    ) -> None:
        self.max_agents = max_agents
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.orca_config_file = orca_config_file
        self.rand = rand if rand is not None else np.random.default_rng()

    def _new_episode(self):
        builder = EnvironmentBuilder(
            rand=np.random.default_rng(seed=int(self.rand.integers(0, 2**31 - 1))),
            include_static_obstacles=False,
        )
        env = builder.build_environment()
        robot_planner = DecentralizedORCAPlanner(
            config_file=self.orca_config_file, obstacles=env.obstacles
        )
        crowd_planner = DecentralizedORCAPlanner(config_file=self.orca_config_file)
        step_driver = Step(
            robot_planner=robot_planner,
            crowd_planner=crowd_planner,
            env=env,
            robot_visible=False,
        )
        return env, step_driver

    def collect(self, n_episodes: int, ticks_per_episode: int = 200) -> list[GSTSample]:
        """Run ``n_episodes`` ORCA-only crowd rollouts and slice every
        valid ``(obs_length, pred_length)`` window from each.
        """
        samples: list[GSTSample] = []
        window = self.obs_length + self.pred_length

        for _ in range(n_episodes):
            env, step_driver = self._new_episode()
            ped_ids = sorted(env.crowd.keys())[: self.max_agents]

            # frame log: position + velocity, read straight off each
            # pedestrian's actual state -- not finite-differenced from
            # position, which would be a noisier, indirect proxy for the
            # same quantity Step already integrates exactly.
            frame_log: list[dict[int, tuple[float, float, float, float]]] = []

            for _tick in range(ticks_per_episode):
                step_driver.step()
                frame = {
                    ped_id: (
                        env.crowd[ped_id].pose.px,
                        env.crowd[ped_id].pose.py,
                        env.crowd[ped_id].velocity.vx,
                        env.crowd[ped_id].velocity.vy,
                    )
                    for ped_id in ped_ids
                    if ped_id in env.crowd and env.crowd[ped_id].pose is not None
                }
                frame_log.append(frame)

            samples.extend(self._slice_windows(frame_log, ped_ids, window))

        return samples

    def _slice_windows(
        self,
        frame_log: list[dict[int, tuple[float, float, float, float]]],
        ped_ids: list[int],
        window: int,
    ) -> list[GSTSample]:
        samples: list[GSTSample] = []
        T = len(frame_log)

        for start in range(0, T - window + 1):
            history_positions = np.zeros(
                (self.max_agents, self.obs_length, 2), dtype=np.float32
            )
            history_velocity = np.zeros(
                (self.max_agents, self.obs_length, 2), dtype=np.float32
            )
            history_mask = np.zeros(
                (self.max_agents, self.obs_length), dtype=np.float32
            )
            future_displacement = np.zeros(
                (self.max_agents, self.pred_length, 2), dtype=np.float32
            )
            future_mask = np.zeros(
                (self.max_agents, self.pred_length), dtype=np.float32
            )

            for slot, ped_id in enumerate(ped_ids):
                for t in range(self.obs_length):
                    frame = frame_log[start + t]
                    if ped_id not in frame:
                        continue
                    px, py, vx, vy = frame[ped_id]
                    history_positions[slot, t] = (px, py)
                    history_velocity[slot, t] = (vx, vy)
                    history_mask[slot, t] = 1.0

                if history_mask[slot, -1] == 0.0:
                    continue  # need a real last-observed position to anchor displacement
                last_px, last_py = history_positions[slot, -1]

                for k in range(self.pred_length):
                    frame = frame_log[start + self.obs_length + k]
                    if ped_id not in frame:
                        continue
                    px, py, _, _ = frame[ped_id]
                    future_displacement[slot, k] = (px - last_px, py - last_py)
                    future_mask[slot, k] = 1.0

            if history_mask.sum() == 0.0:
                continue
            samples.append(
                GSTSample(
                    history_positions=history_positions,
                    history_velocity=history_velocity,
                    history_mask=history_mask,
                    future_displacement=future_displacement,
                    future_mask=future_mask,
                )
            )

        return samples


def samples_to_batch(
    samples: list[GSTSample], device: torch.device
) -> dict[str, Tensor]:
    """Stack a list of GSTSample into batched ``[B, max_agents, ...]`` tensors."""

    def stack(field_name: str) -> Tensor:
        arr = np.stack([getattr(s, field_name) for s in samples])
        return torch.as_tensor(arr, dtype=torch.float32, device=device)

    return {
        "history_positions": stack("history_positions"),
        "history_velocity": stack("history_velocity"),
        "history_mask": stack("history_mask"),
        "future_displacement": stack("future_displacement"),
        "future_mask": stack("future_mask"),
    }
