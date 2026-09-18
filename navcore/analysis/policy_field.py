"""PolicyFieldVisualizer: spatial value-function and greedy-action fields
for a trained CrowdNav++-style policy, evaluated over one frozen crowd
snapshot.

What this answers, precisely:
    "If the robot were standing at (x, y) instead of wherever it currently
    is -- holding every pedestrian's position/velocity, every obstacle, and
    the robot's goal and heading fixed at this snapshot -- what value does
    the critic assign, and what action does the actor take?" That is a
    genuinely useful diagnostic (does the value surface look sane near
    obstacles and pedestrians? does the action field route around the
    crowd toward the goal?), but it is NOT "the value function" in the
    tabular-RL sense of one plot covering the whole state space --
    CrowdNav's true state includes the entire crowd configuration, which
    this sweeps robot position over while freezing everything else. Re-run
    against a different `Environment` snapshot (different crowd layout,
    different goal) to see a different slice. Robot orientation (`theta`)
    is likewise held fixed at the snapshot's original value for every grid
    cell -- this is "value/action if the robot were here, facing the way
    it currently is," not marginalized over heading.

Temporal-history caveat (flagged, not glossed over):
    The policy's `TemporalEncoder` conditions each neighbor's embedding on
    a recent motion-history window (see that module's docstring). A
    one-shot spatial sweep has no real history to give it. This class
    resets `ObservationEncoder`'s history buffer before every grid cell,
    so every neighbor's temporal embedding is exactly what the network
    already produces for "just became visible, no motion history yet" (an
    all-zero-mask window -- a legitimate input the policy sees at every
    episode's first tick, not an out-of-distribution one). It does mean
    this field reflects the policy's response to instantaneous crowd state
    only, not to an observed motion trend.

Recurrence caveat:
    Evaluated with a freshly zeroed recurrent hidden state at every cell --
    a single-tick snapshot, not a rollout, so `RecurrentNodeUpdate`'s
    cross-tick memory plays no role here.

Environment mutation:
    Evaluating a cell means temporarily overwriting `env.robot.pose` and
    restoring it in a `finally` block once the sweep finishes. Pass a
    dedicated snapshot built for this purpose (e.g.
    `EnvironmentBuilder().build_environment()`), not a live
    training/rollout `Environment` -- the mutation is real for the
    duration of the call, and nothing here makes it safe to run
    concurrently with another consumer of the same instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor

from navcore.entities.environment.environment import Environment
from navcore.gym_wrapper.observation_encoder import ObservationEncoder
from navcore.visualization.entities.crowd_visualizer import CrowdVisualizer
from navcore.visualization.entities.obstacle_visualizer import ObstacleVisualizer


class RecurrentActorCritic(Protocol):
    """The minimal interface `PolicyFieldVisualizer` needs from a policy.

    Deliberately narrower than `CrowdNavPPPolicy`'s full surface, so this
    visualizer stays usable against any future policy sharing this same
    per-tick, explicit-hidden-state contract (the project's
    RL-framework-agnostic principle) rather than hard-importing
    `CrowdNavPPPolicy`, even though that is, today, the only
    implementation.
    """

    def initial_hidden_state(
        self, nenv: int, device: torch.device | None = None
    ) -> Tensor: ...

    def forward(
        self,
        robot_features: Tensor,
        neighbor_features: Tensor,
        neighbor_mask: Tensor,
        neighbor_history: Tensor,
        neighbor_history_mask: Tensor,
        hidden_state: Tensor,
        not_done_mask: Tensor,
    ) -> tuple[object, Tensor, Tensor]: ...


@dataclass(slots=True, frozen=True)
class PolicyField:
    """One evaluated spatial field: value and greedy action per grid cell.

    Attributes:
        xs: Grid x-coordinates, shape `(nx,)`, world units.
        ys: Grid y-coordinates, shape `(ny,)`, world units.
        value: Critic value estimate per cell, shape `(ny, nx)`.
        action_x: Greedy vx per cell, shape `(ny, nx)`.
        action_y: Greedy vy per cell, shape `(ny, nx)`.
        reachable: Boolean mask, `(ny, nx)`. False for any cell that fell
            inside a non-traversable static obstacle -- a robot pose there
            is physically meaningless, so `value`/`action_x`/`action_y`
            are left as `nan` rather than a spurious network output.
    """

    xs: np.ndarray
    ys: np.ndarray
    value: np.ndarray
    action_x: np.ndarray
    action_y: np.ndarray
    reachable: np.ndarray


class PolicyFieldVisualizer:
    """Evaluates and plots a `RecurrentActorCritic`'s value/action fields
    over robot position, against one frozen `Environment` snapshot.

    Attributes:
        policy: The trained policy to query. Callers are responsible for
            `policy.eval()` and loading weights -- this class never
            mutates the policy.
        encoder: An `ObservationEncoder` matching the policy's configured
            `robot_feature_dim`/`neighbor_feature_dim`. This class calls
            `encoder.reset()` before every grid cell regardless of the
            instance's prior state -- see module docstring's
            temporal-history caveat -- so passing a dedicated instance
            (not one shared with a live rollout) avoids any confusion
            about whose history is being reset.
        device: Torch device to run inference on.
    """

    def __init__(
        self,
        policy: RecurrentActorCritic,
        encoder: ObservationEncoder,
        device: torch.device | None = None,
    ) -> None:
        self.policy = policy
        self.encoder = encoder
        self.device = device if device is not None else torch.device("cpu")

    # -- evaluation -----------------------------------------------------

    def evaluate(
        self,
        env: Environment,
        resolution: float = 0.25,
        margin: float = 0.5,
    ) -> PolicyField:
        """Sweep robot position over `env`'s arena and query the policy at each cell.

        Args:
            env: The frozen scenario (pedestrians, obstacles, robot goal)
                to evaluate against. See module docstring's "Environment
                mutation" note.
            resolution: Grid spacing, world units. Cost is
                `O(1 / resolution**2)` inference calls with no caching
                across cells -- each is an independent hypothetical robot
                position.
            margin: How far outside the nominal arena half-extents to
                extend the grid, world units. A small positive margin
                shows how the value surface behaves right up to (and just
                past, for context) the boundary.

        Returns:
            The evaluated `PolicyField`.

        Raises:
            RuntimeError: If `env.robot.pose`, `env.robot.goal`, or
                `env.robot.sensor` is unset.
        """
        if env.robot.pose is None or env.robot.goal is None:
            raise RuntimeError(
                "PolicyFieldVisualizer.evaluate() requires env.robot.pose "
                "and env.robot.goal to already be set."
            )
        if env.robot.sensor is None:
            raise RuntimeError(
                "PolicyFieldVisualizer.evaluate() requires env.robot.sensor "
                "to be initialized (build env via the normal builder path)."
            )

        half_width = float(env.info.arena_width) / 2.0 + margin
        half_height = float(env.info.arena_height) / 2.0 + margin
        xs = np.arange(-half_width, half_width + resolution, resolution)
        ys = np.arange(-half_height, half_height + resolution, resolution)

        value = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
        action_x = np.full_like(value, np.nan)
        action_y = np.full_like(value, np.nan)
        reachable = np.zeros_like(value, dtype=bool)

        original_pose = env.robot.pose
        original_px, original_py, original_theta = (
            original_pose.px,
            original_pose.py,
            original_pose.theta,
        )
        try:
            for iy, y in enumerate(ys):
                for ix, x in enumerate(xs):
                    if self._inside_any_obstacle(env, float(x), float(y)):
                        continue
                    reachable[iy, ix] = True
                    env.robot.pose.px = float(x)
                    env.robot.pose.py = float(y)
                    env.robot.pose.theta = original_theta
                    v, ax_, ay_ = self._query_policy(env)
                    value[iy, ix] = v
                    action_x[iy, ix] = ax_
                    action_y[iy, ix] = ay_
        finally:
            env.robot.pose.px = original_px
            env.robot.pose.py = original_py
            env.robot.pose.theta = original_theta

        return PolicyField(xs, ys, value, action_x, action_y, reachable)

    def _inside_any_obstacle(self, env: Environment, x: float, y: float) -> bool:
        """Return whether `(x, y)` falls inside a non-traversable obstacle.

        Uses `containment.point_in_geometry`, which only supports
        area-enclosing geometry (`Circle`/`Rectangle`/`Polygon`). A `Line`
        obstacle geometry (none exist in `navcore.entities.obstacles`
        today, but the type permits one) would raise `TypeError` here --
        left uncaught deliberately, since that is a real modeling gap this
        class should surface loudly rather than silently treat as
        reachable.
        """
        from navcore.entities.components.geometry.containment import (
            point_in_geometry,
        )
        from navcore.entities.components.geometry.vector2 import Vector2

        point = Vector2(x, y)
        for obstacle in env.obstacles.values():
            if obstacle.traversable:
                continue
            if point_in_geometry(point, obstacle.geometry):
                return True
        return False

    def _query_policy(self, env: Environment) -> tuple[float, float, float]:
        """Run one forward pass for the robot's current pose in `env`.

        Returns:
            `(value, greedy_vx, greedy_vy)`. The action is the
            distribution's mean, not a stochastic sample -- this is a
            "what would the trained policy do" field, so the deterministic
            action is the right choice, matching the project's existing
            `deterministic=True` evaluation convention (see
            `evaluate.py`/`test_crowdnav_pp.py`).
        """
        self.encoder.reset()
        obs = self.encoder.encode(env)

        batch = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)
            for k, v in obs.items()
        }
        hidden = self.policy.initial_hidden_state(nenv=1, device=self.device)
        not_done = torch.ones(1, device=self.device)

        with torch.no_grad():
            distribution, value, _ = self.policy.forward(
                batch["robot"],
                batch["neighbors"],
                batch["neighbor_mask"],
                batch["neighbor_history"],
                batch["neighbor_history_mask"],
                hidden,
                not_done,
            )
            action = distribution.mean

        return (
            float(value.squeeze().item()),
            float(action[0, 0].item()),
            float(action[0, 1].item()),
        )

    # -- plotting ---------------------------------------------------------

    def plot(
        self,
        field: PolicyField,
        env: Environment,
        quiver_stride: int = 2,
    ) -> Figure:
        """Render `field` as a value heatmap with a greedy-action quiver overlay.

        Obstacles and the frozen pedestrian snapshot are drawn on top for
        context by reusing the same `ObstacleVisualizer`/`CrowdVisualizer`
        sub-visualizers the live simulation viewer uses, rather than
        duplicating that drawing logic here.

        Args:
            field: The result of `evaluate()`.
            env: The same `Environment` passed to `evaluate()` -- used
                only for drawing obstacles/pedestrians/goal, never
                re-queried against the policy.
            quiver_stride: Draw one action arrow every `quiver_stride`
                grid cells per axis, so the field stays legible instead of
                a solid mat of overlapping arrows.

        Returns:
            The created `Figure`.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        extent = (field.xs[0], field.xs[-1], field.ys[0], field.ys[-1])
        mesh = ax.imshow(
            field.value,
            extent=extent,
            origin="lower",
            cmap="viridis",
            aspect="equal",
        )
        fig.colorbar(mesh, ax=ax, label="critic value")

        stride = max(1, quiver_stride)
        xs_grid, ys_grid = np.meshgrid(field.xs, field.ys)
        ax.quiver(
            xs_grid[::stride, ::stride],
            ys_grid[::stride, ::stride],
            field.action_x[::stride, ::stride],
            field.action_y[::stride, ::stride],
            color="white",
            width=0.003,
        )

        ObstacleVisualizer(env, ax).draw()
        CrowdVisualizer(env, ax).draw()

        if env.robot.goal is not None:
            ax.plot(
                env.robot.goal.gx,
                env.robot.goal.gy,
                marker="*",
                markersize=16,
                color="red",
                label="goal",
            )

        ax.set_title("Critic value + greedy action field")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        return fig
