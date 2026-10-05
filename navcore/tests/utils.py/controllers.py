# tests/utils/controllers.py
"""Robot controllers under test.

``PolicyController`` is the default: it runs the trained CrowdNav++ policy
through the same ``ObservationEncoder`` / action decoding used in training and
evaluation. ``OrcaController`` is a sanity baseline for validating the suite.

Environment variables:
    NAVCORE_TEST_CONTROLLER   "policy" (default) | "orca"
    NAVCORE_TEST_CHECKPOINT   checkpoint path (default: searched 41200.pt)
    NAVCORE_TEST_DEVICE       torch device (default "cpu")
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np
import torch

from navcore.entities.components.velocity import Velocity
from navcore.entities.environment.environment import Environment
from navcore.gym_wrapper.crowd_sim_env import CrowdSimEnv
from navcore.gym_wrapper.observation_encoder import ObservationEncoder
from navcore.middleware.orca_middleware import DecentralizedORCAPlanner

ORCA_CONFIG_FILE = "orca.toml"


class RobotController(Protocol):
    name: str

    def reset(self, env: Environment) -> None: ...

    def act(self, env: Environment) -> Velocity | None:
        """Velocity override for this tick, or ``None`` to let ``Step`` plan."""
        ...

    def make_robot_planner(self, env: Environment): ...


class PolicyController:
    name = "policy"

    def __init__(
        self,
        policy,
        encoder: ObservationEncoder,
        device: torch.device,
        deterministic: bool = True,
    ) -> None:
        self.policy = policy
        self.encoder = encoder
        self.device = device
        self.deterministic = deterministic
        self._uses_rays = bool(policy.config.uses_ray_features)
        self._hidden = policy.initial_hidden_state(nenv=1, device=device)
        self._not_done = torch.zeros(1, device=device)

    @classmethod
    def from_checkpoint(
        cls, path: str | None = None, device: str = "cpu"
    ) -> "PolicyController":
        from navcore.training.original.checkpoint import (
            inspect_checkpoint,
            load_policy_checkpoint,
            resolve_pretrained,
        )
        from navcore.training.original.common import (
            build_policy,
            resolve_policy_config,
            resolve_ray_spec,
        )

        checkpoint = resolve_pretrained(path)
        torch_device = torch.device(device)
        ray_spec = resolve_ray_spec(None, None)
        _, info = inspect_checkpoint(checkpoint)
        config = resolve_policy_config(
            obstacle_mode="auto",
            max_neighbors=None,
            ray_spec=ray_spec,
            checkpoint_info=info,
        )
        policy = build_policy(config, torch_device)
        load_policy_checkpoint(policy, checkpoint, verbose=False)
        policy.eval()
        encoder = ObservationEncoder(
            max_neighbors=config.max_neighbors, history_steps=8, ray_spec=ray_spec
        )
        return cls(policy, encoder, torch_device)

    def reset(self, env: Environment) -> None:
        self.encoder.reset(env)
        self._hidden = self.policy.initial_hidden_state(nenv=1, device=self.device)
        self._not_done = torch.zeros(1, device=self.device)

    def act(self, env: Environment) -> Velocity:
        obs = self.encoder.encode(env)
        batch = {
            key: torch.as_tensor(
                value, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            for key, value in obs.items()
        }
        with torch.no_grad():
            distribution, _, self._hidden = self.policy.forward(
                batch["robot"],
                batch["neighbors"],
                batch["neighbor_mask"],
                batch["neighbor_history"],
                batch["neighbor_history_mask"],
                self._hidden,
                self._not_done,
                ray_features=batch["ray_features"] if self._uses_rays else None,
            )
            action = distribution.mean if self.deterministic else distribution.sample()
        self._not_done = torch.ones(1, device=self.device)
        action_np = action.squeeze(0).float().cpu().numpy().astype(np.float32)
        return CrowdSimEnv._decode_velocity_action(action_np)

    def make_robot_planner(self, env: Environment):
        return None


class OrcaController:
    """Baseline: the robot is planned by ORCA exactly like a pedestrian.

    Built without obstacles on purpose: ``obstacle_to_vertices`` returns
    local-frame vertices for ``Rectangle`` geometry (documented upstream bug),
    which would plant phantom obstacles at the origin, inside the robot corridor.
    """

    name = "orca"

    def reset(self, env: Environment) -> None:
        pass

    def act(self, env: Environment) -> Velocity | None:
        return None

    def make_robot_planner(self, env: Environment):
        return DecentralizedORCAPlanner(config_file=ORCA_CONFIG_FILE)


def make_controller_from_env() -> RobotController:
    kind = os.environ.get("NAVCORE_TEST_CONTROLLER", "policy").lower()
    if kind == "orca":
        return OrcaController()
    if kind == "policy":
        return PolicyController.from_checkpoint(
            os.environ.get("NAVCORE_TEST_CHECKPOINT"),
            os.environ.get("NAVCORE_TEST_DEVICE", "cpu"),
        )
    raise ValueError(f"Unknown NAVCORE_TEST_CONTROLLER={kind!r}.")
