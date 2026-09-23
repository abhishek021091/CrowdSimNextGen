# CrowdNav++ on navcore/CrowdSimNextGen — Technical System Design Report

*This report is derived exclusively from the source code provided. Wherever a
detail (hyperparameter, shape, algorithmic choice, or design motivation) is
not explicitly present in the code, this document states:
"Not determinable from the available code." No behavior, tensor shape, or
rationale is invented.*

---

# 1. Executive Summary

**Problem solved.** The repository implements a crowd-navigation
reinforcement-learning system: a holonomic mobile robot must reach a goal
position inside a bounded 2D arena that also contains a dynamic crowd of
pedestrians (optionally organized into social groups) and, optionally,
static furniture obstacles (tables). The robot must do this while avoiding
collisions with pedestrians, obstacles, and the arena boundary.

**Environment.** `navcore.gym_wrapper.crowd_sim_env.CrowdSimEnv` is a
Gymnasium (`gym.Env`) environment. It wraps a lower-level, RL-framework-
agnostic simulation core (`navcore.entities.environment.environment.Environment`,
driven tick-by-tick by `navcore.step.step.Step`). Pedestrians are always
driven by a decentralized ORCA (Optimal Reciprocal Collision Avoidance)
planner built on the `rvo2` library
(`navcore.middleware.orca_middleware.DecentralizedORCAPlanner` →
`navcore.policies.base_orca_planner.BaseORCAPlanner`). The robot can either
be driven by the same ORCA machinery (`ActionMode.WAYPOINT`) or have its
velocity supplied directly by the RL policy every tick
(`ActionMode.VELOCITY`, the mode used by the PPO training pipeline).

**Agent (robot).** `navcore.entities.agents.robot.Robot`, holonomic
(config `robot.toml`: `kinematics.chassis = "holonomic"`), with a fixed
physical radius, preferred speed, and a 360° range sensor.

**Observation pipeline.** `navcore.gym_wrapper.observation_encoder.
ObservationEncoder` turns the live `Environment` into a fixed-shape
Gymnasium `Dict` observation: an 8-dimensional robot-state vector, a
padded/masked set of neighbor (pedestrian) feature vectors plus their
short motion-history buffer, and a binary "range image" built from a
ray-casting obstacle sensor.

**Neural network.** `navcore.policies.crowdnav_pp.policy.CrowdNavPPPolicy`
is a ported/adapted version of the CrowdNav++ interaction-graph
architecture (Liu et al., ICRA 2023, "Intention Aware Robot Crowd
Navigation with Attention-Based Interaction Graph", per the module's own
docstring), consisting of: a robot-state encoder, a masked per-neighbor
temporal LSTM encoder, a human-human self-attention layer, a robot→human
cross-attention layer, an *optional* obstacle branch (range-image residual
CNN → robot→obstacle cross-attention → learned gate fusion with the human
context), a recurrent GRU-based node update, and separate actor/critic
MLP towers feeding a diagonal-Gaussian continuous action head.

**Action generation.** `navcore.policies.crowdnav_pp.actiton_distribution.
DiagGaussianHead` produces `Independent(Normal(mean, std), 1)` over a
2-D continuous `(vx, vy)` action space; `select_action()` either samples
(training rollout) or takes the distribution mean (`deterministic=True`,
evaluation).

**PPO training.** `navcore.training.crowd_nav_pp.crowd_nav_pp_trainer.
CrowdNavPPTrainer` implements a clipped-surrogate PPO loop with GAE
(`navcore.training.crowd_nav_pp.rollout_buffer.RecurrentRolloutBuffer`)
over a synchronous vectorized environment
(`navcore.training.crowd_nav_pp.vec_env.VecCrowdSimEnv`). There is no
minibatching over time — each PPO epoch reprocesses the *entire* stored
rollout sequentially, one tick at a time, to correctly thread the GRU's
recurrent hidden state (see §9, §22).

**Reward system.** `navcore.gym_wrapper.goal_reaching_task.
GoalReachingTask` computes a dense, per-tick reward: a small constant
step penalty, a distance-progress term, a one-time goal bonus, and
one-time penalties for collision and for leaving the arena bounds.

**Evaluation.** `navcore.training.crowd_nav_pp.evaluate.evaluate()` runs a
deterministic evaluation harness over a fixed number of seeded episodes
and reports outcome rates (success/collision/out_of_bounds/timeout),
reward statistics, path efficiency, minimum separation from pedestrians,
and policy behavior diagnostics (action saturation, commanded/actual
speed).

---

# 2. Repository Architecture

Folder-by-folder responsibilities, restricted to what appears in the
provided source:

| Folder | Purpose | Key files | Interacts with |
|---|---|---|---|
| `navcore/entities/` | Domain model: agents, static obstacles, geometry, shared value types (`Pose`, `Goal`, `Velocity`, `FullState`, `ObservableState`), sensors, groups, the `Environment` container and its `CollisionChecker`. | `agents/agent.py`, `agents/robot.py`, `agents/pedestrians.py`, `components/*`, `obstacles/*`, `environment/environment.py`, `environment/collision_checker.py`, `groups/group.py` | Everything else imports from here; it has no dependency on planners/policies/training. |
| `navcore/missions/` | The "where should this agent move right now" abstraction (`Mission` protocol), decoupled from "how to avoid collisions". | `mission.py`, `goal_reaching.py`, `group_goal_reaching.py`, `sweeping.py` | Consumed by `Step` and by `SweepingMission`/coverage planners. |
| `navcore/policies/` | ORCA planner core (`base_orca_planner.py`) plus the CrowdNav++ neural-network submodules and the GST trajectory predictor. | `base_orca_planner.py`, `crowdnav_pp/*`, `gst_predictor/*` | `middleware/orca_middleware.py` wraps `base_orca_planner`; `gym_wrapper`/`training` consume `crowdnav_pp`. |
| `navcore/middleware/` | Reconciles `Step`'s per-agent, sensor-limited calling convention with `BaseORCAPlanner`'s centralized, fixed-population API. | `orca_middleware.py` (`DecentralizedORCAPlanner`) | Used by `Step`, `GlobalPlanner`, `GSTDataCollector`, test scripts. |
| `navcore/step/` | Advances the simulation by exactly one tick (Mission → VelocityPlanner → integration), with two-phase compute/apply ordering. | `step.py` (`Step`, `StepResult`) | Central orchestrator used by `CrowdSimEnv`, `GlobalPlanner`, `test_sweep.py`, `task_planners.py`. |
| `navcore/gym_wrapper/` | Gymnasium RL interface: environment, observation encoding, task abstraction, RL-specific missions, action decoding. | `crowd_sim_env.py`, `observation_encoder.py`, `task.py`, `goal_reaching_task.py`, `rl_missions.py` | Wraps `Environment`/`Step`/`Task`; is what `training/` trains against. |
| `navcore/training/` | Everything training/offline-only: PPO trainer, vectorized env, rollout buffer, CLI entry points, GST predictor pretraining, trajectory-target labels. | `crowd_nav_pp/*`, `gst_predictor/*`, `trajectory_targets.py` | Consumes `gym_wrapper` and `policies.crowdnav_pp`; nothing in `gym_wrapper`/`entities` imports from here. |
| `navcore/builder/` | Constructs a fresh `Environment` (or refreshes part of one) per episode: robot placement, crowd placement/grouping, static obstacle placement. | `environment_builder.py`, `crowd_builder.py`, `robot_builder.py`, `obstacle_builder.py` | Used by `CrowdSimEnv.reset`, `GlobalPlanner`, test scripts. |
| `navcore/avoidace_planner/` *(sic, misspelled in repo)* | Candidate-based local safe-point selection for reactive obstacle/human avoidance (non-RL, used by the coverage/global-planner pipeline). | `local_avoidace_planner.py` | Used by `SweepingMission.avoid_crowd`, `GlobalPlanner`. |
| `navcore/collision_predictor/` | Predictive (not ground-truth) collision/intrusion checks from sensor-limited observations, plus circle-vs-polygon SAT collision tests. | `sat.py` (`SAT`, `ObstacleCollisionDetector`) | Used by `SweepingMission`, `GlobalPlanner`, `test_sweep.py`. |
| `navcore/boustropheden/` | Convex decomposition of free space (Boustrophedon Cellular Decomposition) for full-coverage planning — unrelated to the RL/PPO pipeline. | `boustropheden.py` | Used by `GlobalPlanner`, `graph_traversal`. |
| `navcore/graph_traversal/` | DFS traversal sequencing over the BCD adjacency graph. | `traversal.py` | Used by `GlobalPlanner`. |
| `navcore/planner/` | Orchestrates the non-RL coverage task (`GlobalPlanner`) and simple goal/waypoint task runners (`task_planners.py`) — not part of the RL/PPO stack. | `global_planner.py`, `task_planners.py` | Uses `boustropheden`, `graph_traversal`, `missions.sweeping`, `step`. |
| `navcore/visualization/` | Live Matplotlib rendering of a running `Environment` (crowd, obstacles, robot, decomposition cells). | `visualizer.py`, `entities/*` | Used by test scripts and `GlobalPlanner`; not used inside the training loop. |
| `navcore/analysis/` | Offline, post-hoc analysis: JSONL metrics logging, training-curve plotting, and a spatial value/action field visualizer for a trained policy. | `metrics_logger.py`, `training_curves.py`, `policy_field.py` | `metrics_logger.py` is imported by the PPO trainer and evaluator; the rest are standalone offline tools. |
| `navcore/configs/` | TOML configuration: `env.toml`, `robot.toml`, `pedestrians.toml`, `orca.toml`, `obstacle.toml`. | — | Read at class-construction time by many modules via `tomllib`. |

---

# 3. Environment

## 3.1 `Environment` (data container)

`navcore.entities.environment.environment.Environment` is a plain
dataclass: `info: EnvironmentInfo`, `obstacles: dict[str, Obstacle]`,
`crowd: dict[int, Pedestrian]`, `groups: dict[int, Group]`,
`robot: Robot`. `EnvironmentInfo` is populated from `env.toml`:
`arena_width`, `arena_height`, `random_seed`, `collision_counter`,
`goal_reach_tolerance` (`tolerance.goal_reach = 0.1`), `safety_distance`
(`safety.distance = 0.1`).

`Environment.did_collision_happened()` constructs a
`CollisionChecker(self.robot, self)` and calls `check_collision()`,
incrementing `info.collision_counter` on a hit. `Environment.
out_of_bounds()` checks the robot's pose plus radius plus safety distance
against the arena half-extents.

## 3.2 `EnvironmentBuilder`

`navcore.builder.environment_builder.EnvironmentBuilder` owns one
`CrowdBuilder`, one `RobotBuilder`, one `ObstacleBuilder`, and an
`np.random.Generator`.

- `build_environment()`: if `include_static_obstacles` (constructor flag,
  default `True`), calls `obstacle_builder.build_table()` then
  `build_boundary()`; then `crowd_builder.build_crowd()` and
  `build_groups()`; then `robot_builder.build_robot(obstacle_builder.
  obstacles)`; returns a new `Environment`.
- `reset(random_seed)`: builds a **fresh** `np.random.Generator`, fresh
  `CrowdBuilder`/`RobotBuilder`/`ObstacleBuilder` seeded from it, and a
  fresh `EnvironmentInfo(random_seed=random_seed)`; used by
  `CrowdSimEnv.reset()`.
- `rebuild_pedestrian(env, ped_id, random_seed)`: reseeds a fresh
  `CrowdBuilder` from `random_seed` and replaces exactly one pedestrian
  in `env.crowd` (used to keep the crowd dynamic across an episode — see
  §14).

## 3.3 Robot placement (`RobotBuilder`)

`generate_pose()`: rejection-samples `(px, py, theta)` uniformly over the
arena rectangle, up to `MAX_PLACEMENT_ATTEMPTS = 200` times, rejecting any
point inside a non-traversable, non-"boundary" obstacle
(`_inside_obstacle`, via `point_in_geometry`). Raises `RuntimeError` if no
valid pose is found within the attempt budget.

`generate_goal()`: shrinks the sampling rectangle inward by
`_boundary_clearance() = robot.radius + env.info.safety_distance` (so a
sampled goal cannot legally coincide with a collision-flagging position
right at the wall), then rejection-samples the same way as `generate_pose`
against non-traversable obstacles.

## 3.4 Crowd placement (`CrowdBuilder`)

`build_crowd()`: for `pedestrians.toml`'s `num_pedestrians` (10 by
default), constructs a `Pedestrian`, assigns an id, samples a pose via
`generate_pose()` (a random point on one of the four arena edges, offset
outward by up to 2 units, with a random heading), and a goal via
`generate_goal(method="opposite")` (one of three reflected corners of the
spawn point, chosen uniformly), then calls `pedestrian.set_state(...)`.

`build_groups()`: if `pedestrians.toml`'s `num_groups`/`group_size` are
nonzero, partitions the built crowd list into contiguous chunks of
`group_size`, builds a `Group` per chunk (leader = first member), and for
each member constructs a (locally-scoped, not retained) `GroupGoalReaching
Mission` to set the member's `group_id`, formation position, and goal.
**Default `env.toml` values are `group_size = 0`, `num_groups = 0`**, so
by default no groups are built.

`build_single_pedestrian(ped_id)`: used by `rebuild_pedestrian` to respawn
one pedestrian with a fresh pose/goal under a given id, without touching
the rest of the crowd.

## 3.5 Obstacle placement (`ObstacleBuilder`)

`build_boundary()`: always registers a `"boundary"` `Obstacle` — a closed
`Polygon` at the arena's outer rectangle, with a `BoundaryGate` if
`obstacle.toml`'s `boundary.gate.enabled` is `True` (it is `False` in the
provided config, so no gates by default).

`build_table()`: builds `circular_table.num_tables` (0 by default) and
`rectangular_table.num_tables` (7 by default, positions/sizes given
explicitly in `obstacle.toml` rather than randomized, since
`dimensions = "given"` / `randomize_dimensions = false`).

**Note:** `EnvironmentBuilder.build_environment()`'s docstring/behavior
and `obstacle.toml`'s comments both indicate static obstacles (tables) are
built by default when `include_static_obstacles=True`. `CrowdSimEnvConfig.
include_static_obstacles` defaults to `True`, but
`GSTDataCollector._new_episode()` explicitly passes
`include_static_obstacles=False`, and `test_crowdnav_pp_smoke.py`'s
`_build_env_and_obs()` also passes `include_static_obstacles=False`. So
whether obstacles are present is configuration-dependent, not fixed.

## 3.6 `CrowdSimEnv.reset()`

1. `super().reset(seed=seed)` (Gymnasium's own RNG seeding).
2. `episode_seed = seed if seed is not None else self.np_random.integers(0, 2**31 - 1)`.
3. `self.env = self._env_builder.reset(random_seed=episode_seed)`.
4. `self.task.reset(self.env)` — rebuilds the task's `Mission` and its
   internal `_prev_distance` (see §10).
5. `self._obs_encoder.reset(self.env)` — clears neighbor-history buffers
   and rebuilds the cached obstacle/boundary shapely geometry used by ray
   casting.
6. If `action_mode is ActionMode.WAYPOINT`: constructs an
   `RLWaypointMission(initial_target = robot's current pose)` and a
   `DecentralizedORCAPlanner` for the robot (so the robot is still ORCA-
   driven, targeting the RL-selected waypoint). Otherwise
   `robot_planner = None` and `robot_mission = None`.
7. Constructs a `crowd_planner = DecentralizedORCAPlanner(config_file=
   self.config.orca_config_file)` — **always** present, since pedestrians
   are always ORCA-driven.
8. Constructs `self._step_driver = Step(crowd_planner=..., env=self.env,
   robot_visible=self.config.robot_visible, robot_planner=...,
   robot_mission=...)`.
9. Resets `self._velocity_override = None`, `self._elapsed_steps = 0`.
10. Returns `(self._obs_encoder.encode(self.env), {})`.

## 3.7 `CrowdSimEnv.step(action)`

1. `self._apply_action(action)`:
   - `ActionMode.VELOCITY`: `self._velocity_override =
     self._decode_velocity_action(action)` — clips the *magnitude* of
     `(vx, vy)` to `v_pref` (from `robot.toml`'s `kinematics.v_pref`),
     preserving direction, since `spaces.Box` allows an independent
     per-axis bound that can exceed `v_pref` in Euclidean norm.
   - `ActionMode.WAYPOINT`: `self._velocity_override = None`;
     `self._waypoint_mission.set_target(self._decode_waypoint_action(action))`,
     where the `[-1, 1]`-normalized action is scaled by the arena's
     half-width/half-height.
2. `step_result = self._step_driver.step(robot_velocity_override=
   self._velocity_override)` — see §3.8/§13 for `Step`'s internals.
3. `self._respawn_pedestrians(step_result)` — any pedestrian whose
   `pedestrian_reached_goals[id]` is `True` this tick is rebuilt via
   `EnvironmentBuilder.rebuild_pedestrian(random_seed = env.info.
   random_seed + self._elapsed_steps)`.
4. `collided = self.env.did_collision_happened()`;
   `out_of_bounds = self.env.out_of_bounds()`.
5. `observation = self._obs_encoder.encode(self.env)`.
6. `reward = self.task.reward(self.env, collided, out_of_bounds)`.
7. `terminated = self.task.is_terminated(self.env, collided, out_of_bounds)`.
8. `self._elapsed_steps += 1`; `truncated = self._elapsed_steps >=
   self.config.max_episode_steps` (default 1500).
9. Returns `(observation, reward, terminated, truncated, info)`, with
   `info = {"collision", "out_of_bounds", "truncated",
   "robot_reached_goal", "terminated"}`.

## 3.8 `Step.step()` — one simulation tick

`navcore.step.step.Step.step(robot_velocity_override=None)`:

1. `self._validate()` — asserts pose/velocity/goal/sensor are set for the
   robot and every pedestrian.
2. `self._change_group_goals()` — refreshes each group member's
   `GroupGoalReachingMission` (goal only; positions are not force-set here
   at every tick, only at group construction, per `CrowdBuilder.
   build_groups`'s one-time `set_position()` call). **Note:** the default
   configuration has zero groups, so this is a no-op in the default RL
   setup.
3. `result = self._compute_velocities(robot_velocity_override)`:
   - `_compute_robot_velocity`: if an override was supplied, uses it
     directly (bypassing `robot_planner` entirely — this is the path RL
     `ActionMode.VELOCITY` takes). Otherwise requires `self.robot_planner`
     to be set and calls its `compute_velocities(ROBOT_KEY, robot_full_state,
     robot_obs)`.
   - `_compute_crowd_velocities`: for each pedestrian, with **1%
     per-tick probability** (`self.rand.random() < 0.01`) the pedestrian's
     velocity for this tick is forced to `Velocity(0, 0)` (a "pause"
     event) instead of being computed by the planner; otherwise its
     observation is gathered via its sensor, its planning target resolved
     via `_target_for` (mission or own goal), and `self.crowd_planner.
     compute_velocities(ped_id, ped_full_state, ped_obs)` is called.
   - Also computes `robot_reached_goal` (Euclidean distance between
     robot's pose and its **real** `robot.goal`, not any mission target,
     `<= robot.radius + goal_reach_tolerance`) and, per pedestrian,
     `pedestrian_reached_goals[id]`.
4. `self._set_velocities(...)` writes the computed velocities onto
   `robot.velocity` / each pedestrian's `.velocity`.
5. `self._advance_agent(robot)` then, for each pedestrian,
   `self._advance_agent(ped)`: Euler integration,
   `pose.px += velocity.vx * dt`, `pose.py += velocity.vy * dt`
   (`dt = env.toml`'s `policy.time_step = 0.1`), and `pose.theta =
   atan2(vy, vx)` if speed² > 1e-12.
6. Returns the `StepResult`.

Two-phase compute-then-apply ordering (compute every agent's velocity
before mutating any pose) is stated in the module docstring as
deliberate, to avoid first-mover bias between agents.

## 3.9 Randomization / seed handling

- `EnvironmentBuilder.__init__` and every `Builder` take an optional
  `np.random.Generator`; if omitted, each seeds independently from
  `env.toml`'s `random.seed = 15`.
- `EnvironmentBuilder.reset(random_seed)` creates **one** fresh
  `np.random.Generator(seed=random_seed)` and passes it into fresh
  `CrowdBuilder`/`RobotBuilder`/`ObstacleBuilder` instances — so within one
  episode, the same RNG stream is shared/advanced sequentially across
  obstacle placement → crowd placement → robot placement, in that call
  order (this ordering makes the episode fully reproducible for a given
  seed, but a change to the placement call order or count would change
  every downstream draw).
- `CrowdSimEnv.reset(seed=...)` derives its own `episode_seed` either from
  the caller's `seed` or Gymnasium's own `self.np_random` (itself seeded
  by `super().reset(seed=seed)`).
- `Step`'s own `self.rand` (used for the 1%-pause draw) is a **separate**
  `np.random.Generator`, defaulting to an unseeded
  `np.random.default_rng()` unless explicitly passed in — `CrowdSimEnv`
  does **not** pass a `rand` into its `Step` construction in `reset()`,
  so the per-tick pause draw is **not deterministic** under a fixed
  `CrowdSimEnv` seed. *(Compare: `GlobalPlanner`/`test_sweep.py` do pass
  `rand=self.rand`/`self.env_builder.rand` into their `Step` instances.)*

## 3.10 Parallel environments / VecEnv

`navcore.training.crowd_nav_pp.vec_env.VecCrowdSimEnv` owns `n_envs`
independent `CrowdSimEnv` instances (one `Task` each, since
`GoalReachingTask` carries per-episode state). It is a **synchronous,
single-process** Python-loop wrapper — no multiprocessing — per the
module docstring, explicitly to avoid pickling `rvo2`'s C extension
across process boundaries. `reset(seed)` offsets each sub-env's seed by
its index (`seed + i`). `step(actions)` steps every sub-env, auto-resets
any env whose `done = terminated or truncated`, and stashes the pre-reset
terminal observation under `info["terminal_observation"]`. Observations
are stacked with `np.stack` along a new leading axis across `_OBS_KEYS`.

---

# 4. Observation Pipeline

`navcore.gym_wrapper.observation_encoder.ObservationEncoder.encode(env)`
returns a `dict[str, np.ndarray]` with exactly six keys. All dtypes are
`np.float32` except the two `*_mask` keys, which are `np.int8`
(`spaces.MultiBinary` in the declared Gym space).

## 4.1 `"robot"` — shape `(8,)`, `np.float32`

Constructed as (in order):
`[goal.gx - pose.px, goal.gy - pose.py, velocity.vx, velocity.vy,
v_pref, radius, cos(pose.theta), sin(pose.theta)]`.

Origin: `robot.goal`, `robot.pose`, `robot.velocity`, `robot.v_pref`,
`robot.radius` at encode time (**post**-movement, since `encode()` is
called after `Step.step()` in `CrowdSimEnv.step()`). Destination:
`RobotStateEncoder` (see §5.1).

## 4.2 `"neighbors"` — shape `(max_neighbors, 5)`, `np.float32`

Per visible pedestrian (from `robot.sensor.observe(env, robot_visible=
False)`, a distance-based query within `robot.sensor_range` — see §12):
`[rel_px, rel_py, vx, vy, radius]` where `rel_px = ped.pose.px -
robot.pose.px`, `rel_py = ped.pose.py - robot.pose.py`, `vx/vy` are the
pedestrian's own velocity (**not** relative to the robot), `radius` is
the pedestrian's own radius.

Selection/padding: the `max_neighbors` (default 10, configurable via
`CrowdSimEnvConfig.max_neighbors`) **nearest** visible pedestrians by
Euclidean distance to the robot are placed at slots `0..k-1`; remaining
slots (if `k < max_neighbors`) stay zero. `"neighbor_mask"` (shape
`(max_neighbors,)`, `int8`) is `1` at a real slot, `0` at padding.

## 4.3 `"neighbor_history"` — shape `(history_steps, max_neighbors, 5)`, `np.float32`

`ObservationEncoder` keeps a `dict[int, deque[np.ndarray]]`
(`self._neighbor_history`, `maxlen=history_steps`, default `history_steps
=8`) keyed by pedestrian id. Every `encode()` call appends the current
tick's 5-D feature vector (same layout as §4.2) for **every currently
observed** pedestrian id (not just the nearest `max_neighbors`), so a
pedestrian's history keeps accumulating even on a tick it is not among
the `max_neighbors` selected for `"neighbors"`. For each of the selected
nearest `max_neighbors` this tick, its deque's contents are written into
`history[start:, i]` where `start = history_steps - len(track)` — i.e.
history is **left-padded with zeros** (older-than-available steps are
zero), most-recent step in the **last** row. `"neighbor_history_mask"`
(shape `(history_steps, max_neighbors)`, `int8`) mirrors this: `1` where
a real (non-padding) history entry exists.

**Note:** the neighbor-history buffer is cleared only on `reset()`, so it
does not track "was this pedestrian in the selected `max_neighbors` set
last tick" — it tracks "was this pedestrian *observed* at all" (within
`robot.sensor_range`), independent of the top-`max_neighbors` selection
used for the instantaneous `"neighbors"` tensor.

## 4.4 `"range_image"` — shape `(1, num_range_bins, num_rays)`, `np.float32`, values in `{0.0, 1.0}`

Pipeline: `ObstacleDetector.sense(robot.pose.px, robot.pose.py,
self._cached_obstacle_polygons, heading=0.0)` (world-frame ray fan,
`heading` fixed at `0.0` since the robot is holonomic — see §12) →
`RangeImageBuilder.build(scan)`.

`self._cached_obstacle_polygons` is built **once per episode**, in
`ObservationEncoder.reset(env)`: every non-`"boundary"` obstacle
converted to a world-frame shapely `Polygon`
(`obstacle_to_shapely_polygon`), plus the arena's boundary ring
(`arena_boundary_ring(env)`), as a flat list — the boundary is not
distinguished from any other obstacle geometry at ray-casting time (see
§12 for the caveat this implies for `HitType`).

`RangeImageBuilder` default config: `num_rays=180`,
`num_range_bins=128`, `max_range=5.0` (kept in sync with
`ObstacleDetectorConfig` at `ObservationEncoder.__init__` time, raising
`ValueError` on mismatch). Binary semantics: `1.0` = free/traversable,
`0.0` = blocked (or "unknown-beyond-first-hit", treated identically —
see the module's own docstring for this explicit simplification). Row 0
= far edge of the sensing square (`max_range`); the last row = the
robot's own position. Column 0 = left-most ray (columns are reversed from
`ObstacleDetector`'s native increasing-angle ray order — see module
docstring).

## 4.5 Full flow diagram

```
Environment (ground-truth state)
   │
   ├─ robot.pose / robot.goal / robot.velocity / robot.v_pref / robot.radius
   │        └──► "robot"  (8,)  float32
   │
   ├─ robot.sensor.observe(env, robot_visible=False)  ── distance-gated dict[int, ObservableState]
   │        ├──► nearest max_neighbors → "neighbors" (max_neighbors,5) float32
   │        │                          → "neighbor_mask" (max_neighbors,) int8
   │        └──► every observed id, appended to a per-id deque(maxlen=history_steps)
   │                                  → "neighbor_history" (history_steps,max_neighbors,5) float32
   │                                  → "neighbor_history_mask" (history_steps,max_neighbors) int8
   │
   └─ ObstacleDetector.sense(robot pose, cached obstacle/boundary polygons)
            └─ RangeImageBuilder.build(scan)
                     └──► "range_image" (1, num_range_bins, num_rays) float32 {0,1}
```

All six arrays are stacked by `VecCrowdSimEnv._stack` across the `n_envs`
axis (new leading dim), then converted to `torch.Tensor` via
`torch.as_tensor(..., dtype=torch.float32)` inside `CrowdNavPPTrainer`
(`_to_tensor_batch`) and inside the rollout buffer's `observations()`
(stacked additionally across the time axis, `[T, n_envs, ...]`) before
reaching the network. **`"neighbor_mask"`/`"neighbor_history_mask"` are
cast to `.bool()` inside `CrowdNavPPPolicy.forward()`.**

---

# 5. Neural Network Architecture

All modules below are instantiated inside
`navcore.policies.crowdnav_pp.policy.CrowdNavPPPolicy.__init__`, driven by
`CrowdNavPPPolicyConfig`. Defaults quoted are the dataclass field
defaults unless a caller overrides them (the training CLI
`train_crowdnav_pp.py` does not override any of the interaction/attention
widths, and does not enable the obstacle branch — `use_obstacle_encoder`
is commented out in that CLI's `CrowdNavPPPolicy(...)` construction, so
`use_range_image_obstacles` stays at its dataclass default, `False`).

## 5.1 `RobotStateEncoder`

File: `navcore/policies/crowdnav_pp/robot_state_encoder.py`.
`self.project = nn.Sequential(nn.Linear(robot_feature_dim, embedding_dim), nn.ReLU())`.

- Input: `[..., robot_feature_dim]` (default `8`).
- Output: `[..., embedding_dim]` (default `interaction_embedding_dim =
  256`).
- One `Linear` layer + `ReLU`, no normalization, no dropout, no residual.
- Weight/bias initialization: `Not determinable from the available code`
  (uses PyTorch's default `nn.Linear` init; no explicit `nn.init.*` call
  in this class).

## 5.2 `TemporalEncoder`

File: `navcore/policies/crowdnav_pp/temporal_encoder.py`.
`self.cell = nn.LSTMCell(motion_feature_dim, hidden_size)`.

- `motion_feature_dim`: in `CrowdNavPPPolicy`, set to `_NEIGHBOR_MOTION_
  SLICE.stop - .start = 2` (slices `(vx, vy)`, index 2:4, out of the 5-D
  neighbor feature).
- `hidden_size`: `CrowdNavPPPolicyConfig.temporal_hidden_size`, default
  `32`.
- Forward loop: explicit Python `for t in range(history_steps)`, **not**
  a single batched `nn.LSTM` call (module docstring explains this is a
  deliberate correctness choice — see §19). At each step:
  `new_hidden, new_cell = self.cell(flat_motion[t], (hidden, cell))`;
  then `step_mask = flat_mask[t].unsqueeze(-1)`; `hidden = new_hidden *
  step_mask + hidden * (1 - step_mask)` (and identically for `cell`) —
  i.e. hidden/cell state is **frozen**, not updated, on any timestep the
  agent was not actually visible.
- Input shapes: `motion_history: [history_steps, nenv, max_agents, 2]`,
  `history_mask: [history_steps, nenv, max_agents]`.
- Initial hidden/cell state: zeros, `motion_history.new_zeros(nenv *
  max_agents, hidden_size)`.
- Output: `[nenv, max_agents, hidden_size]` (`= [nenv, 10, 32]` under
  defaults).
- An agent never visible across the *entire* window produces an
  all-zero embedding (state never leaves its zero initialization).

## 5.3 `HumanHumanAttention`

File: `navcore/policies/crowdnav_pp/human_human_attention.py`.

- `self.embed = nn.Sequential(nn.Linear(spatial_edge_feature_dim, 128),
  nn.ReLU(), nn.Linear(128, embedding_size), nn.ReLU())`.
- `self.query_proj / key_proj / value_proj = nn.Linear(embedding_size,
  embedding_size)` (three separate projections).
- `self.attention = nn.MultiheadAttention(embedding_size,
  num_attention_heads)` — **not** `batch_first` (uses the
  `(seq_len, batch, embed)` PyTorch convention).
- `spatial_edge_feature_dim` (from `CrowdNavPPPolicyConfig.
  spatial_edge_feature_dim` property) = `neighbor_feature_dim (5) +
  temporal_hidden_size (32) [+ gst_pred_length*2 if use_gst_prediction]`.
  Default (no GST): `5 + 32 = 37`.
- `embedding_size` default `512` (`human_human_embedding_size`).
- `num_attention_heads` default `8` (`human_human_num_heads`); must
  evenly divide `embedding_size` (validated in `__post_init__`; `512 / 8
  = 64` — satisfied by default).
- Input: `human_features [seq_len, nenv, max_human_num,
  spatial_edge_feature_dim]`, `visible_mask [seq_len, nenv,
  max_human_num]` bool.
- Forward: flattens `seq_len*nenv` into one batch axis, embeds,
  transposes to `[max_human_num, batch, embedding_size]`, computes
  `key_padding_mask = ~flat_mask`, runs `Q/K/V` projections, then
  `nn.MultiheadAttention`, transposes back.
- **Raises `ValueError` if any `(seq, env)` slot has zero visible
  humans** — `nn.MultiheadAttention` would otherwise produce NaNs on a
  fully-masked row. `CrowdNavPPPolicy` works around this upstream (see
  §5.9 `_substitute_dummy_human`).
- Output: `[seq_len, nenv, max_human_num, embedding_size]`.

## 5.4 Human-embedding down-projection

`self._human_embed_down = nn.Identity()` if `human_human_embedding_size
== interaction_embedding_dim`, else `nn.Linear(human_human_embedding_size,
interaction_embedding_dim)`. Under the stated defaults (`512` vs `256`),
this is a `Linear(512, 256)`.

## 5.5 `RobotHumanAttention`

File: `navcore/policies/crowdnav_pp/robot_human_attention.py`.

- `self.attention = nn.MultiheadAttention(embed_dim=embedding_dim,
  num_heads=num_attention_heads, batch_first=True)`.
- `embedding_dim` = `interaction_embedding_dim` (default `256`).
- `num_attention_heads` default `8` (`robot_human_num_heads`); must
  evenly divide `256` (satisfied).
- Input: `robot_embedding [seq_len, nenv, 1, embedding_dim]`,
  `human_embeddings [seq_len, nenv, human_count, embedding_dim]`,
  `visible_mask [seq_len, nenv, human_count]` bool.
- Forward: reshapes both to `[batch=seq_len*nenv, ...]`, runs
  `nn.MultiheadAttention(query=robot, key=humans, value=humans,
  key_padding_mask=~mask)`, reshapes back to `[seq_len, nenv,
  embedding_dim]`.
- Module docstring notes the *original* paper's attention scaling
  (`human_count / sqrt(attention_size)` instead of standard
  `1/sqrt(d_k)`) is **not** reproduced here — this port instead delegates
  scaling entirely to `nn.MultiheadAttention`'s own standard scaled
  dot-product, per the code as written (the docstring's claim about
  matching the paper's *unusual* scaling is describing intent for a
  sibling design note, not this class's actual `nn.MultiheadAttention`
  call, which uses PyTorch's standard `1/sqrt(d_k)`).
- Output: `[seq_len, nenv, embedding_dim]` (crowd context).

## 5.6 `human_context_norm`

`nn.LayerNorm(interaction_embedding_dim)` — applied to the squeezed
`crowd_context` (`[nenv, 256]`).

## 5.7 Obstacle branch (present only when `config.use_range_image_obstacles`)

### 5.7.1 `RangeImageEncoder`

File: `navcore/policies/crowdnav_pp/range_image_encoder.py`.

- **Stem**: `_AngularConv2d(1, stem_channels=32, kernel_size=3,
  stride=1)` → `BatchNorm2d(32)` → `ReLU`.
- **Stages** (`stage_channels=(32,64,128)`, `blocks_per_stage=(2,2,2)`,
  `downsample_after_stage=(True,True,False)`): each stage is a
  `nn.Sequential` of `_ResidualBlock2D`s; the first block of a
  downsampling stage uses `stride=2`.
- `_ResidualBlock2D`: `conv1 = _AngularConv2d(in, out, 3, stride)` →
  `BN` → `ReLU` → `conv2 = _AngularConv2d(out, out, 3, stride=1)` → `BN`
  → add identity/1×1-`Conv2d`+`BN` skip (skip only present when
  `stride != 1` or `in_channels != out_channels`) → `ReLU`.
- `_AngularConv2d`: raw `nn.Parameter` weight `[out, in, k, k]` (`kaiming_
  normal_(mode="fan_out", nonlinearity="relu")` init) + zero-init bias;
  forward pads width circularly and height with zeros (`F.pad`), then
  `F.conv2d` with `padding=0`.
- **Height collapse**: `nn.Conv2d(128, token_embedding_dim=128,
  kernel_size=(out_height, 1), stride=(out_height, 1))` — a single
  strided conv spanning the full remaining feature-map height, producing
  `[B, 128, 1, backbone_output_width]`.
- **Token compress**: `nn.Conv2d(128, 128, kernel_size=(1,
  tokens_per_group), stride=(1, tokens_per_group))`, where
  `tokens_per_group = backbone_output_width // num_obstacle_tokens
  (=15)` — exact, non-overlapping tiling (validated to divide evenly in
  `__post_init__`).
- `self.token_norm = nn.LayerNorm(128)`;
  `self.angular_position_embedding = nn.Parameter(zeros(15, 128))`
  (`normal_(std=0.02)` init), added after `token_norm`.
- Default config: `in_height=128, in_width=180, stem_channels=32,
  stage_channels=(32,64,128), blocks_per_stage=(2,2,2),
  downsample_after_stage=(True,True,False), token_embedding_dim=128,
  num_obstacle_tokens=15`. With two downsampling stages, backbone output
  width = `180 // 2 // 2 = 45`; `45 / 15 = 3` (`tokens_per_group=3`).
  Backbone output height = `128 // 2 // 2 = 32`.
- Output: `[B, 15, 128]` obstacle tokens (each carrying its learned
  angular position embedding).

### 5.7.2 `RobotObstacleAttention`

File: `navcore/policies/crowdnav_pp/robot_obstacle_attention.py`.

- `self.query_proj = nn.Linear(robot_embedding_dim=256,
  obstacle_embedding_dim=128)`.
- `self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4
  (default `num_attention_heads`), batch_first=True)`.
- Forward: `query = query_proj(robot_embedding).unsqueeze(1)` →
  `[B,1,128]`; `key=value=obstacle_tokens [B,15,128]`; optional
  `key_padding_mask` (unused by default — `RangeImageEncoder` always
  produces a fixed, meaningful token set); output `context.squeeze(1)`
  → `[B, 128]`.

### 5.7.3 Fusion path

`obstacle_context_proj = nn.Linear(128, 256)`; `obstacle_context_norm =
nn.LayerNorm(256)`; then `ContextFusionGate`
(`navcore/policies/crowdnav_pp/fusion_gate.py`):
`gate_mlp = nn.Sequential(nn.Linear(2*256, hidden_size=128), nn.ReLU(),
nn.Linear(128, 256))`; `gate = sigmoid(gate_mlp(cat(human_context,
obstacle_context)))`; `fused = gate * human_context + (1 - gate) *
obstacle_context` (a **feature-wise**, `[B,256]`-shaped gate, not a
scalar). `self.last_gate` stores the most recent gate tensor for
diagnostics.

## 5.8 `RecurrentNodeUpdate`

File: `navcore/policies/crowdnav_pp/recurrent_node_update.py`.

- `robot_embed = nn.Sequential(nn.Linear(input_dim=256,
  node_embedding_size=64), nn.ReLU())`.
- `context_embed = nn.Sequential(nn.Linear(256, 64), nn.ReLU())`.
- `gru_cell = nn.GRUCell(64*2=128, rnn_hidden_size=128)`. Initialization:
  biases `nn.init.constant_(param, 0)`, weights
  `nn.init.orthogonal_(param)`.
- `output_linear = nn.Linear(rnn_hidden_size=128, output_size=256)`.
- `initial_hidden_state(nenv, device) = torch.zeros(nenv,
  rnn_hidden_size)`.
- Forward: `concat = cat(robot_embed(robot_embedding),
  context_embed(crowd_context))` → `[nenv, 128]`; `reset_hidden =
  hidden_state * not_done_mask.unsqueeze(-1)` (zeroes the incoming
  hidden state on an episode-restart tick); `new_hidden =
  gru_cell(concat, reset_hidden)`; `output = output_linear(new_hidden)`.
- Returns `(output [nenv, 256], new_hidden [nenv, 128])`.
- **Scope note (from the module docstring):** this implements only the
  single-timestep GRU path (matching the original's rollout-collection
  branch). The original's separate multi-timestep batched-GRU training
  path (with mid-sequence hidden-state resets inside one call) is **not
  ported** — see §22 for the consequence this has for PPO's recompute
  cost.

## 5.9 `_substitute_dummy_human` (in `CrowdNavPPPolicy`)

Static method: given `visible_mask [1, nenv, max_neighbors]`, computes
`no_humans_visible = ~visible_mask.any(dim=-1)`; if any `(seq, env)` has
zero visible neighbors, clones the mask and forces slot `0` visible for
those entries (`visible_mask[..., 0] |= no_humans_visible`). This is the
project's own resolution of `HumanHumanAttention`'s "raise on fully
masked row" behavior, applied once, right before `human_human_attention`
is called. **Caveat (from the code's own comments elsewhere in the
class):** the substituted slot-0 feature vector is whatever zero-padding
already sits there (an all-zero neighbor feature), not a genuinely
synthetic "dummy human" state distinct from ordinary padding.

## 5.10 `ActorCriticHeads`

File: `navcore/policies/crowdnav_pp/actor_crititc_head.py`.

- `actor = nn.Sequential(orthogonal_linear(input_dim=256, hidden_size=256,
  gain=sqrt(2)), nn.Tanh(), orthogonal_linear(256, 256, gain=sqrt(2)),
  nn.Tanh())`.
- `critic` = an **independently parameterized** but structurally
  identical two-layer `Linear→Tanh→Linear→Tanh` tower (no shared trunk
  with `actor` — stated as a deliberate faithfulness-to-the-original
  choice in the module docstring).
- `critic_linear = orthogonal_linear(256, 1, gain=0.01)` — small gain to
  keep the initial value-head output near-linear/small.
- `_orthogonal_linear(in, out, gain)`: `nn.Linear` with
  `nn.init.orthogonal_(weight, gain=gain)`, `nn.init.constant_(bias, 0)`.
- Forward: `actor_features = actor(node_output)` `[..., 256]`;
  `value = critic_linear(critic(node_output))` `[..., 1]`.

## 5.11 `DiagGaussianHead`

File: `navcore/policies/crowdnav_pp/actiton_distribution.py`.

- `mean_linear = nn.Linear(input_dim=256 (`actor_critic_hidden_size`),
  action_dim=2)`.
- `log_std = nn.Parameter(torch.full((action_dim,), -1.0))` — a single
  **state-independent** learnable vector, not a function of the input
  features.
- `LOG_STD_MIN = -3.0`, `LOG_STD_MAX = 0.5` (class constants).
- Forward: `mean = mean_linear(actor_features)`; `log_std =
  self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)`; `std =
  log_std.exp().expand_as(mean)`; returns
  `torch.distributions.Independent(torch.distributions.Normal(mean,
  std), 1)` — the `Independent` wrapper sums log-probability over the
  action dimension.

## 5.12 Module inventory summary table

| # | Module | File | In/Out shape (default config) |
|---|---|---|---|
| 1 | `RobotStateEncoder` | `robot_state_encoder.py` | `[nenv,8] → [nenv,256]` |
| 2 | `TemporalEncoder` | `temporal_encoder.py` | `[8,nenv,10,2] → [nenv,10,32]` |
| 3 | `HumanHumanAttention` | `human_human_attention.py` | `[1,nenv,10,37] → [1,nenv,10,512]` |
| 4 | `_human_embed_down` | `policy.py` | `[1,nenv,10,512] → [1,nenv,10,256]` |
| 5 | `RobotHumanAttention` | `robot_human_attention.py` | `[1,nenv,1,256]×[1,nenv,10,256] → [1,nenv,256]` |
| 6 | `human_context_norm` | `policy.py` | `[nenv,256] → [nenv,256]` |
| 7 | `RangeImageEncoder` *(optional)* | `range_image_encoder.py` | `[B,1,128,180] → [B,15,128]` |
| 8 | `RobotObstacleAttention` *(optional)* | `robot_obstacle_attention.py` | `[B,256]×[B,15,128] → [B,128]` |
| 9 | `obstacle_context_proj`+`_norm` *(optional)* | `policy.py` | `[B,128] → [B,256]` |
| 10 | `ContextFusionGate` *(optional)* | `fusion_gate.py` | `[B,256]×[B,256] → [B,256]` |
| 11 | `RecurrentNodeUpdate` | `recurrent_node_update.py` | `[nenv,256]×[nenv,256]×[nenv,128] → [nenv,256],[nenv,128]` |
| 12 | `ActorCriticHeads` | `actor_crititc_head.py` | `[nenv,256] → [nenv,1],[nenv,256]` |
| 13 | `DiagGaussianHead` | `actiton_distribution.py` | `[nenv,256] → Independent(Normal, [nenv,2])` |

---

# 6. Data Flow Through the Network

One tick, `CrowdNavPPPolicy.forward(...)` (file: `navcore/policies/
crowdnav_pp/policy.py`):

```
robot_features            [nenv, 8]
neighbor_features         [nenv, 10, 5]
neighbor_mask             [nenv, 10]            (cast to bool)
neighbor_history          [nenv, 8, 10, 5]
neighbor_history_mask     [nenv, 8, 10]         (cast to bool)
hidden_state              [nenv, 128]
not_done_mask             [nenv]
range_image (optional)    [nenv, 1, 128, 180]

  1. motion_history = neighbor_history[..., 2:4].transpose(0,1)      → [8, nenv, 10, 2]
     history_mask    = neighbor_history_mask.transpose(0,1).float()  → [8, nenv, 10]
  2. temporal_embedding = TemporalEncoder(motion_history, history_mask)      → [nenv, 10, 32]
  3. spatial_edge_features = cat(neighbor_features, temporal_embedding, -1)  → [nenv, 10, 37]
  4. human_features  = spatial_edge_features.unsqueeze(0)             → [1, nenv, 10, 37]
     visible_mask    = neighbor_mask.unsqueeze(0)                     → [1, nenv, 10]
     visible_mask    = _substitute_dummy_human(visible_mask)
  5. human_embeddings = HumanHumanAttention(human_features, visible_mask)    → [1, nenv, 10, 512]
     human_embeddings = _human_embed_down(human_embeddings)                  → [1, nenv, 10, 256]
  6. robot_embedding_seq = RobotStateEncoder(robot_features)
                              .unsqueeze(0).unsqueeze(-2)              → [1, nenv, 1, 256]
     crowd_context = RobotHumanAttention(robot_embedding_seq,
                        human_embeddings, visible_mask).squeeze(0)     → [nenv, 256]
     robot_embedding = robot_embedding_seq.squeeze(0).squeeze(-2)      → [nenv, 256]
  7. human_context = LayerNorm(crowd_context)                          → [nenv, 256]
  8. if use_range_image_obstacles:
       obstacle_tokens = RangeImageEncoder(range_image)                → [nenv, 15, 128]
       obstacle_context_raw = RobotObstacleAttention(
                                  robot_embedding, obstacle_tokens)    → [nenv, 128]
       obstacle_context = obstacle_context_norm(
                              obstacle_context_proj(obstacle_context_raw)) → [nenv, 256]
       fused_context = ContextFusionGate(human_context, obstacle_context)  → [nenv, 256]
     else:
       fused_context = human_context                                  → [nenv, 256]
  9. node_output, new_hidden_state = RecurrentNodeUpdate(
                       robot_embedding, fused_context,
                       hidden_state, not_done_mask)                    → [nenv, 256], [nenv, 128]
 10. value, actor_features = ActorCriticHeads(node_output)             → [nenv, 1], [nenv, 256]
 11. distribution = DiagGaussianHead(actor_features)                   → Independent(Normal([nenv,2],[nenv,2]),1)

 12. action, log_prob = select_action(distribution, deterministic)     → [nenv,2], [nenv]
 13. env.step(action) → CrowdSimEnv._decode_velocity_action clips |v| ≤ v_pref, applies to robot
 14. reward = GoalReachingTask.reward(...) → scalar per env
 15. next observation (§4) fed back in as step 12's input for the next tick,
     new_hidden_state carried forward as hidden_state.
```

---

# 7. Actor Network

- **Layers**: `ActorCriticHeads.actor` (2× `Linear(256,256)` with
  orthogonal init `gain=sqrt(2)`, each followed by `Tanh`) feeding
  `DiagGaussianHead.mean_linear` (`Linear(256, 2)`, default PyTorch
  init — no explicit `nn.init.*` call for this layer).
- **Distribution**: `Independent(Normal(mean, std), 1)`; `mean_linear`
  output is the raw Gaussian mean with **no bounding function
  (no tanh-squashing) applied to the mean itself** — the code does not
  apply `torch.tanh` anywhere in `DiagGaussianHead` or `select_action`.
- **Log-std**: a single learnable `nn.Parameter` per action dimension
  (`[2]`), **not** conditioned on the input (state-independent), clamped
  to `[-3.0, 0.5]` before `exp()`. So `std ∈ [e^-3, e^0.5] ≈ [0.0498,
  1.6487]`.
- **Action bounds / clipping**: not enforced inside the network or the
  distribution. Enforcement happens **downstream**, in
  `CrowdSimEnv._decode_velocity_action`, which clips the sampled
  `(vx, vy)`'s **Euclidean magnitude** (not each axis independently) to
  `v_pref` before it is applied to the simulator: `if speed > v_max:
  scale = v_max / speed; vx, vy = vx*scale, vy*scale`. This clipping is
  **not** accounted for in the stored `log_prob` used by PPO — the
  trainer's own module docstring (`crowd_nav_pp_trainer.py`) explicitly
  states log-probs are computed on the raw, pre-clip sample, calling this
  "the standard PPO-on-continuous-control convention", with no
  change-of-variables correction applied.
- **Sampling**: `select_action(distribution, deterministic)` (file
  `actiton_distribution.py`) — `action = distribution.mean if
  deterministic else distribution.sample()`; `log_prob =
  distribution.log_prob(action)`. `distribution.sample()` (not
  `.rsample()`) is used for actual action selection (non-differentiable
  draw, as appropriate for rollout collection); the smoke test
  `test_crowdnav_pp_smoke.py` separately verifies gradients using
  `distribution.rsample()` for its own gradient-flow check only, not as
  part of the production rollout path.
- **Deterministic inference**: `deterministic=True` (used by
  `evaluate.py` and the live demo `test_crowdnav_pp.py`) takes the
  distribution mean, i.e. `mean_linear`'s raw output, with the fixed
  learned `std` playing no role in the chosen action (but still
  contributing to `log_prob`, which is unused at evaluation time).
- **Entropy**: `distribution.entropy()` — the closed-form entropy of an
  `Independent(Normal,1)` (used in the PPO loss's entropy bonus, and
  reported as `entropy` in training stats; see §9). No explicit formula
  is written in the code — it is PyTorch's built-in
  `torch.distributions.Normal.entropy()` summed over the action
  dimension by `Independent`.
- **Variance**: `std**2`, per-dimension, identical across the batch and
  across time (state-independent parameter), only two scalars total
  (`action_dim=2`).

## 7.1 Equations from implementation

For action dimension `i ∈ {0,1}` (i.e. `vx`, `vy`):

```
mean_i   = (W_mean · actor_features + b_mean)_i
log_std_i = clamp(log_std_param_i, -3.0, 0.5)
std_i     = exp(log_std_i)
π(a | s)  = ∏_i  N(a_i ; mean_i, std_i²)
log π(a|s) = Σ_i [ -0.5*((a_i - mean_i)/std_i)² - log(std_i) - 0.5*log(2π) ]   (PyTorch's Normal.log_prob, summed by Independent)
a_sampled = mean + std ⊙ ε,   ε ~ N(0, I)        (Normal.sample()/rsample())
```

---

# 8. Critic Network

- **Architecture**: `ActorCriticHeads.critic` — an independent two-layer
  `Linear(256,256)+Tanh` tower (same shape as `actor`, separate weights,
  no sharing), followed by `critic_linear = Linear(256, 1)` with
  orthogonal init `gain=0.01`.
- **State-value estimation**: `value = critic_linear(critic(node_output))`
  `[nenv, 1]` — a single scalar per environment per tick, from the same
  `node_output` the actor consumes (i.e. both heads read the recurrent
  `RecurrentNodeUpdate` output, not raw observations directly).
- **Advantages / GAE**: computed in
  `RecurrentRolloutBuffer.compute_returns_and_advantages` (see §9.2),
  **not** inside the network itself.
- **Loss**: computed in `CrowdNavPPTrainer.update()` (see §9.3): either
  plain MSE `((new_values - returns)**2).mean()`, or, when
  `PPOConfig.clip_range_vf` is set (default `0.2`), a clipped value loss
  `max((new_values - returns)**2, (values_clipped - returns)**2).mean()`
  where `values_clipped = old_values + clamp(new_values - old_values,
  -clip_range_vf, clip_range_vf)`.
- **Output**: `[nenv, 1]` at every tick; squeezed to `[nenv]` by the
  trainer (`value_t.squeeze(-1)`) before being stored in the rollout
  buffer.

---

# 9. PPO Algorithm

File: `navcore/training/crowd_nav_pp/crowd_nav_pp_trainer.py`
(`CrowdNavPPTrainer`), `PPOConfig`. **Only what is implemented is
described; no unimplemented "standard PPO" feature is asserted.**

## 9.1 Rollout collection (`collect_rollout`)

- `self.policy.eval()`; `self.buffer.start(self._hidden_state)`.
- For `config.n_steps` (default `512`) ticks:
  - `not_done_mask = where(self._prev_done, 0.0, 1.0)`.
  - `action_t, log_prob_t, value_t, new_hidden = self.policy.act(...,
    deterministic=False)` (no gradient — `torch.no_grad()`).
  - `next_obs, reward, done, infos = self.env.step(action_np)`
    (`VecCrowdSimEnv`, `done = terminated OR truncated` per sub-env).
  - `self.buffer.add(obs=self._obs, not_done_mask, action, log_prob,
    value, reward, done)` — note: `obs` stored is the **pre-step**
    observation (the one the action was actually computed from).
  - Running per-env `episode_reward`/`episode_length` accumulators;
    on `done`, appended to `episode_rewards`/`episode_lengths` and
    classified via `_classify_outcome(info)` into
    `success/collision/out_of_bounds/timeout` (precedence: collision →
    out_of_bounds → terminated(=success) → timeout).
  - `self._hidden_state = new_hidden`; `self._prev_done = done`.
- After the loop: bootstraps `last_value` by one extra
  `self.policy.forward(...)` call on the post-loop observation with
  `not_done_mask` derived from the still-current `self._prev_done`
  (`range_image` passed if the obstacle branch is enabled).
- `self.buffer.compute_returns_and_advantages(last_value, gamma,
  gae_lambda)`.
- Returns a stats dict: `episodes_completed`, `mean/std/min/max_episode_
  reward`, `mean_episode_length`, and `{outcome}_rate` for each of the
  four outcome classes.

## 9.2 GAE (`RecurrentRolloutBuffer.compute_returns_and_advantages`)

```
T = len(buffer)
rewards = stack(self._rewards)                          [T, n_envs]
values  = concat(stack(self._values), last_value[None])  [T+1, n_envs]
dones   = stack(self._dones).astype(float32)             [T, n_envs]

advantages = zeros(T, n_envs)
gae = zeros(n_envs)
for t in reversed(range(T)):
    next_non_terminal = 1.0 - dones[t]
    delta = rewards[t] + gamma * values[t+1] * next_non_terminal - values[t]
    gae   = delta + gamma * gae_lambda * next_non_terminal * gae
    advantages[t] = gae

returns = advantages + values[:T]
```

This is a per-env, elementwise/vectorized recursion over `numpy` arrays —
each env-column's accumulator naturally resets at its own episode
boundaries via `dones[t]`.

## 9.3 Update (`update`)

- `self.policy.train()`.
- `explained_var = _explained_variance(old_values, returns)` =
  `1 - Var(returns - old_values) / Var(returns)` (reported, not used in
  the loss).
- If `config.normalize_advantage` (default `True`): `advantages =
  (advantages - advantages.mean()) / (advantages.std() + 1e-8)` —
  computed once over the **whole** `T*n_envs` batch (not per-epoch, not
  per-minibatch — there are no minibatches; see §9.4).
- For `config.n_epochs` (default `4`) full passes:
  1. `new_log_probs, new_values, entropies = self._recompute_sequence()`
     (§9.4).
  2. `ratio = exp(new_log_probs - old_log_probs)`.
  3. `surr1 = ratio * advantages`; `surr2 = clamp(ratio, 1-clip_range,
     1+clip_range) * advantages`; `policy_loss = -min(surr1,
     surr2).mean()`.
  4. Value loss: clipped (if `clip_range_vf` set) or plain MSE (§8).
  5. `entropy_loss = entropies.mean()`.
  6. `loss = policy_loss + vf_coef * value_loss - ent_coef *
     entropy_loss`.
  7. `optimizer.zero_grad(); loss.backward()`.
  8. Gradient clipping: `torch.nn.utils.clip_grad_norm_(policy.
     parameters(), max_grad_norm)` if `max_grad_norm` is set (default
     `0.5`), **else** still computes the norm with `clip_grad_norm_(...,
     float("inf"))` purely for logging (no actual clipping in that
     branch, since `float("inf")` never triggers rescaling).
  9. `optimizer.step()`.
  10. `approx_kl = (old_log_probs - new_log_probs).mean()`;
      `clip_fraction = (|ratio - 1| > clip_range).float().mean()`.
- Stats averaged over the `n_epochs` epochs: `policy_loss, value_loss,
  entropy, approx_kl, clip_fraction, grad_norm`, plus
  `explained_variance` and `action_std_mean` (`policy.action_head.
  log_std.clamp(...).exp().mean()`), computed once (not averaged).

## 9.4 Sequential recompute — no minibatching, no chunked BPTT

`_recompute_sequence()` calls `self.policy.forward(...)` **once per
buffered timestep, sequentially**, `for t in range(T)`, threading `hidden
= self.buffer.initial_hidden_state` forward tick-by-tick and using the
**stored** `not_done_masks[t]` (so hidden-state reset points during
recompute exactly match collection time). It accumulates per-tick
`log_prob`, `value`, `entropy` and stacks them to `[T, n_envs]`. This
means one PPO `update()` call performs `n_epochs * T` sequential forward
passes (`4 * 512 = 2048` by default), each over a `[n_envs, ...]`-shaped
batch — there is **no** minibatch subsampling of `(t, env)` pairs, and no
truncated backpropagation-through-time chunking; the module docstrings
(`rollout_buffer.py`, `recurrent_node_update.py`) explicitly flag this as
the accepted scope for "Slice 2", with the batched multi-timestep GRU
training path deferred/unimplemented.

## 9.5 Optimizer / LR schedule

`self.optimizer = torch.optim.Adam(self.policy.parameters(),
lr=config.learning_rate)` (default `3e-4`). **No learning-rate
scheduler is present anywhere in `CrowdNavPPTrainer` or `PPOConfig`** —
`Not determinable from the available code` beyond "fixed learning rate
for the whole run".

## 9.6 Checkpointing / logging

- `save_checkpoint(path)`: `torch.save({"policy_state_dict",
  "optimizer_state_dict", "total_steps", "total_updates", "ppo_config"
  (the `PPOConfig` dataclass), "policy_config" (the
  `CrowdNavPPPolicyConfig` dataclass)}, path)`.
- `load_checkpoint(path)`: restores `policy_state_dict`,
  `optimizer_state_dict`, `total_steps`, `total_updates` (does **not**
  restore `ppo_config`/`policy_config` into `self` — those keys are only
  written, never read back by `load_checkpoint`).
- `train(total_timesteps, log_every=1, checkpoint_every=None,
  checkpoint_dir=None)`: loops `collect_rollout()` → `update()` until
  `self.total_steps >= total_timesteps`; logs (print + optional
  `TrainingMetricsLogger.log(self.total_steps, {**rollout_stats,
  **update_stats})`) every `log_every` updates; saves a checkpoint every
  `checkpoint_every` updates if `checkpoint_dir` is given, named
  `crowdnav_pp_step{total_steps}.pt`.
- **Parallel rollout**: achieved via `VecCrowdSimEnv`'s `n_envs`
  synchronous sub-environments (§3.10); there is no additional
  multi-process/distributed training code in this trainer.

---

# 10. Reward Function

File: `navcore/gym_wrapper/goal_reaching_task.py`, class
`GoalReachingTask`. Constructor defaults: `collision_penalty=-25.0`,
`out_bound_penalty=-25.0`, `goal_bonus=50.0`, `step_penalty=-0.01`,
`progress_weight=5.0`.

`reset(env)`: rebuilds `self._mission = GoalReachingMission()`; sets
`self._prev_distance = self._distance_to_goal(env)` (Euclidean distance,
robot pose to robot goal, at episode start).

`reward(env, collided, out_of_bounds)`:

```
distance  = _distance_to_goal(env)
progress  = self._prev_distance - distance
self._prev_distance = distance

reward = step_penalty + progress_weight * progress
if collided:       reward += collision_penalty
if out_of_bounds:   reward += out_bound_penalty
if reached_goal():  reward += goal_bonus
```

`_reached_goal(env)`: `distance_to_goal(env) <= robot.radius +
env.info.goal_reach_tolerance`.

`is_terminated(env, collided, out_of_bounds)`: `reached_goal() or
collided or out_of_bounds`.

## 10.1 Equation

```
R_t = -0.01 + 5.0 * (d_{t-1} - d_t)
      + (-25.0) * 1[collision_t]
      + (-25.0) * 1[out_of_bounds_t]
      + (+50.0) * 1[reached_goal_t]
```

where `d_t = ||robot.goal - robot.pose||_2` measured **after** the
tick's movement has been integrated (`reward()` is called in
`CrowdSimEnv.step()` after `Step.step()`).

There is no reward for pedestrian near-misses/clearance, no time-to-goal
shaping beyond the constant step penalty, and no separate boundary
"soft" penalty distinct from `out_bound_penalty` — `Not determinable
from the available code` beyond what is listed above.

`WaypointPlanner`/`GoalPlanner` (`navcore/planner/task_planners.py`) are
a **separate, non-Gym** evaluation harness that computes its own
`MissionMetrics` (success/collision/timeout/steps/path_length/
min_separation) — it does not compute or use `GoalReachingTask`'s
reward at all.

---

# 11. Training Pipeline

Entry point: `navcore/training/crowd_nav_pp/train_crowdnav_pp.py`,
`main()`.

- `argparse` CLI flags map directly onto `PPOConfig` and
  `CrowdSimEnvConfig` fields (`--total-timesteps`, `--n-envs`,
  `--n-steps`, `--n-epochs`, `--learning-rate`, `--gamma`,
  `--gae-lambda`, `--clip-range`, `--clip-range-vf`, `--ent-coef`,
  `--vf-coef`, `--max-grad-norm`/`--no-grad-clip`, `--max-neighbors`,
  `--history-steps`, `--max-episode-steps`, `--obstacle-num-rays`,
  `--obstacle-max-range`, `--seed`, `--device`, `--checkpoint-dir`,
  `--checkpoint-every`, `--resume`, `--use-gst-prediction`,
  `--gst-checkpoint`, `--metrics-path`, `--render`).
- `set_seed(seed)`: seeds `random`, `numpy`, `torch.manual_seed`,
  `torch.cuda.manual_seed_all` — a global, process-wide seed (separate
  from the per-env seeding described in §3.9).
- `log_run_config(args, checkpoint_dir)` writes `run_config.json`
  (command line + args) to `checkpoint_dir` at start of every run.
- Builds `env_config = CrowdSimEnvConfig(action_mode=ActionMode.VELOCITY,
  ...)`; builds `n_envs` `CrowdSimEnv(GoalReachingTask(), env_config,
  render_mode=...)` factories (`env_fns`), one per parallel slot (fresh
  `GoalReachingTask` per slot, per its statefulness — §3.10); wraps in
  `VecCrowdSimEnv(env_fns)`.
- Optionally loads a frozen GST predictor via
  `GSTPredictorTrainer.load_predictor(gst_checkpoint, device=...)` if
  `--use-gst-prediction` (requires `--gst-checkpoint`, enforced by
  `parser.error`).
- Constructs `CrowdNavPPPolicy(CrowdNavPPPolicyConfig(use_gst_prediction=
  args.use_gst_prediction), gst_predictor=gst_predictor)` — **note:**
  the `use_obstacle_encoder=args.use_obstacle_encoder` line is commented
  out in this CLI, so the obstacle branch is **not** wired up by this
  entry point regardless of the (also present but unused)
  `--use-obstacle-encoder` CLI flag; `use_range_image_obstacles` stays
  at its dataclass default (`False`).
- Builds `PPOConfig` from the parsed args, `CrowdNavPPTrainer(env,
  policy, ppo_config, metrics_path=..., render=args.render,
  seed=args.seed)`.
- `trainer.load_checkpoint(args.resume)` if `--resume` given.
- `trainer.train(total_timesteps=..., checkpoint_every=...,
  checkpoint_dir=...)`.

**Validation / best-model selection:** `Not determinable from the
available code` — there is no held-out validation loop or
best-checkpoint-selection logic inside `train_crowdnav_pp.py` or
`CrowdNavPPTrainer`; checkpoints are saved purely on a fixed
`checkpoint_every`-update cadence. A separate `evaluate.py` script exists
for post-hoc deterministic evaluation of a saved checkpoint but is not
invoked automatically during training.

**Device selection:** `--device` defaults to `"cuda" if
torch.cuda.is_available() else "cpu"`; passed straight through to
`PPOConfig.device` and `torch.device(...)`.

**Mixed precision:** `Not determinable from the available code` — no
`torch.cuda.amp`/`autocast`/`GradScaler` usage anywhere in the training
code.

---

# 12. Sensors

## 12.1 `RangeSensor` (crowd visibility)

File: `navcore/entities/components/sensors/sensor.py`. Constructed per
agent (`self.range = agent.config["sensors"]["sensor_range"]`, `self.fov
= agent.config["sensors"]["sensor_fov"] * π`). `observe(environment,
robot_visible)`: for a `Pedestrian` observer, iterates `environment.crowd`
excluding self, includes any pedestrian within `self.range`
(Euclidean distance, **not** FOV-restricted — `self.fov` is computed but
not used inside `observe()`'s distance test); if the observer is a
`Pedestrian` and `robot_visible` is `True`, also injects the robot's
`ObservableState` under key `-1`. For a `Robot` observer, the same
range-gated loop over `environment.crowd` applies (robot's own
`robot_visible` argument is irrelevant to what the robot itself sees —
that flag only governs whether *pedestrians* can see the robot).
`robot.toml`: `sensor_range = 5.0`, `sensor_fov = 360.0` (degrees, times
π gives `2π` — i.e. the config's own units are degrees, multiplied by π
rather than converted to radians correctly, but since it is unused in the
distance-only `observe()` this has no behavioral effect for the robot's
neighbor query).

## 12.2 `ObstacleDetector` (ray-casting range sensor)

File: `navcore/entities/components/sensors/obstacle_detector.py`.
`ObstacleDetectorConfig`: `num_rays=180`, `max_range=5.0`,
`fov_radians=2π`. `_build_ray_offsets()`: for a full `2π` FOV, `n` evenly
spaced angles via `np.linspace(0, 2π, n, endpoint=False)` (drops the
duplicate final ray); for a partial FOV, both endpoints are kept.
`sense(robot_x, robot_y, obstacles, heading=0.0)`: for each ray angle
`heading + offset`, casts a `LineString` from the robot position out to
`max_range`, and, over the supplied `obstacles` sequence (every entry
tested identically — the arena boundary is folded in by the caller,
**not** treated specially inside `sense()` despite `HitType` declaring a
`NONE`/`OBSTACLE` split; **the module docstring claims a `HitType.
BOUNDARY` exists, but the actual `HitType` enum in this version of the
file only defines `NONE = 0` and `OBSTACLE = 1`** — a boundary hit is
reported as `HitType.OBSTACLE`, not a distinct value; this is a
discrepancy between the module docstring and the enum's actual members
as provided), finds the nearest intersection (`_nearest_hit_distance` /
`_ray_geometry_distance` / `_nearest_point_distance`, handling `Point`,
`MultiPoint`, `LineString`, `MultiLineString`, `GeometryCollection`
intersection results). A ray with no hit within `max_range` gets
`distances[i] = max_range`, `hit_mask[i] = False`,
`relative_positions[i] = (0, 0)`.

Returns `ObstacleScan(hit_mask, hit_type, distances,
relative_positions, ray_angles)`, each of shape `(num_rays,)` except
`relative_positions` `(num_rays, 2)`.

`scan_to_features(scan, max_range)` (`RAY_FEATURE_DIM = 6`): per ray,
`[hit_mask, distance/max_range, dx/max_range, dy/max_range,
sin(ray_angle), cos(ray_angle)]`. **This feature vector feeds only the
legacy `ObstacleEncoder` (1D CNN) branch**, which — per §5, §19 — is no
longer constructed by `CrowdNavPPPolicy`'s default architecture; the
default obstacle path instead consumes `RangeImageBuilder`'s binary
image directly (§4.4, §5.7.1).

## 12.3 Human observations

Derived entirely from `RangeSensor.observe()` (§12.1) as consumed by
`ObservationEncoder._encode_neighbors` (§4.2–4.3); there is no separate
"human sensor" module distinct from `RangeSensor`.

## 12.4 Boundary detection

For the range-image obstacle branch, the arena boundary ring
(`arena_boundary_ring(env)`, from `navcore.entities.obstacles.
geometry_conversion`) is included in the same flat list of shapely
geometries passed to `ObstacleDetector.sense()` as every other static
obstacle — see §12.2's noted discrepancy regarding `HitType.BOUNDARY`
not actually existing as a distinct enum member in the provided
`obstacle_detector.py`.

---

# 13. Robot Pipeline

- **State**: `Robot(Agent)` (`navcore/entities/agents/robot.py`) — from
  `robot.toml`: `sensor_range=5.0`, `sensor_fov` (`360.0° * π`,
  see §12.1's caveat), `kinematics = {"v_pref": 1, "chassis":
  "holonomic"}`, `policy = "ORCA"`, `physical.observable = False`,
  `physical.radius = 0.3`. `Agent.__init__` sets `v_pref`, `radius` from
  config; `pose/goal/velocity/sensor` start `None` until `set_state(...)`
  is called by `RobotBuilder.build_robot`.
- **Velocity**: computed per-tick by `Step._compute_robot_velocity` —
  either a direct RL-supplied override (`ActionMode.VELOCITY`, clipped by
  magnitude to `v_pref` in `CrowdSimEnv._decode_velocity_action` before
  being passed down), or `self.robot_planner.compute_velocities(...)`
  (`DecentralizedORCAPlanner`, when `ActionMode.WAYPOINT` or when a
  non-RL script constructs `Step` with a `robot_planner`).
- **Kinematics**: holonomic — `Step._advance_agent` integrates position
  directly from `(vx, vy)` with no unicycle/differential-drive
  constraint; `theta` is derived post-hoc from the velocity heading
  (`atan2`) purely for rendering/orientation, not as a kinematic
  constraint on motion.
- **Goal**: `robot.goal` (a persistent `Goal(gx, gy)`), distinct from
  any transient `Mission` target (see `Step`'s module docstring, §3.8) —
  goal-reached checks and reward always read `robot.goal`, never a
  mission's waypoint.
- **Planner**: `DecentralizedORCAPlanner` (`navcore/middleware/
  orca_middleware.py`) — builds a **fresh, throwaway**
  `BaseORCAPlanner` per call (one small RVO2 simulator per tick, per
  agent), populated with the caller's own state plus its visible
  neighbors (each neighbor given a synthetic "continue current velocity"
  goal — see §14). `BaseORCAPlanner` (`navcore/policies/
  base_orca_planner.py`) wraps `rvo2.PyRVOSimulator`; per-agent
  radius/`preferred_speed`/velocity are taken from each `FullState`
  entry (not a single global config value); reasoning parameters
  (`neighbor_dist, max_neighbors, time_horizon, time_horizon_obst`) come
  from `orca.toml`'s `[orca]` table (`neighbor_dist=5.0, max_neighbors=
  10, time_horizon=5.0, time_horizon_obst=5.0`; the file's
  `agent_radius`/`max_speed` keys are present in `orca.toml` but are
  **not read** by `BaseORCAPlanner.__init__`, which only reads
  `orca["neighbor_dist"|"max_neighbors"|"time_horizon"|
  "time_horizon_obst"]`).
- **Action execution**: for RL training, the network's sampled/clipped
  `(vx, vy)` becomes `robot_velocity_override`, applied verbatim by
  `Step._apply_robot_velocity`/`_advance_agent` — the ORCA planner plays
  no role in the robot's own motion under `ActionMode.VELOCITY`
  (pedestrians are still ORCA-driven and will react to the robot's
  motion via their own sensor observations, since `robot_visible`
  controls whether pedestrians can see the robot).
- **Collision checking**: `Environment.did_collision_happened()` →
  `CollisionChecker.check_collision()` (§3.1, and detailed geometry
  handling per obstacle type, including inverted boundary semantics —
  the boundary `Polygon` is "solid outside", so a robot whose point falls
  outside it, or too near its edge from the inside, is flagged).
- **State update**: `Step._advance_agent` (§3.8) — Euler integration at
  `dt = 0.1` s, plus the heading update.

---

# 14. Human Pipeline

- **Generation**: `CrowdBuilder.build_crowd()`/`build_groups()` (§3.4).
  `Pedestrian.__init__` (`navcore/entities/agents/pedestrians.py`) takes
  an injected `np.random.Generator`; if
  `pedestrians.toml["Randomization"]["randomize_pedestrian_radius"]`
  **and** `"randomize_pedestrian_v_pref"` are both `True` (they are, by
  default), `radius *= rand.uniform(0.8, 1.2)` and `v_pref *=
  rand.uniform(0.8, 1.2)`; otherwise both fall back to the config's
  fixed `physical.radius`/`kinematics.v_pref`.
- **ORCA**: every pedestrian is driven by `self.crowd_planner` (always a
  `DecentralizedORCAPlanner`) inside `Step._compute_crowd_velocities` —
  see §13's planner description, shared code path with the robot's
  ORCA-driven mode.
- **Planner target**: `Step._target_for(ped, mission, neighbors)` — uses
  `crowd_missions.get(ped_id)` if present (only populated by callers that
  explicitly pass `crowd_missions`; `CrowdSimEnv` does **not** pass any,
  so pedestrians in the RL training loop always fall back to targeting
  their own persistent `ped.goal` directly), else `ped.goal`.
- **Velocity update / "pause" event**: with 1% per-tick probability
  (`Step._compute_crowd_velocities`), a pedestrian's velocity for that
  tick is forced to `(0,0)` instead of being ORCA-computed — see §3.8.
- **Goal update / respawn**: on reaching its goal (within `radius +
  goal_reach_tolerance`), a pedestrian is **not** given a new goal in
  place — instead `CrowdSimEnv._respawn_pedestrians` (or the equivalent
  in `GlobalPlanner`/`test_sweep.py`) calls `EnvironmentBuilder.
  rebuild_pedestrian`, which reseeds and completely replaces that
  pedestrian object (`CrowdBuilder.build_single_pedestrian`) with a fresh
  edge-spawned pose and opposite-corner goal, keeping the same `ped_id`.
- **Neighbor interactions (ORCA input)**: each pedestrian's own
  `RangeSensor.observe(env, robot_visible)` determines its visible
  neighbor set (other pedestrians within `sensor_range=10.0` per
  `pedestrians.toml`, plus the robot under key `-1` if `robot_visible`);
  this set (plus the pedestrian's own `ObservableState`, explicitly added
  under its own id — `ped_obs[ped_id] = ped.get_observable_state()`) is
  the local population `DecentralizedORCAPlanner` solves ORCA over for
  that pedestrian this tick.
- **Groups**: `Group` (`navcore/entities/groups/group.py`) is a frozen
  dataclass (`id, member_ids, goal, leader_id`); `GroupGoalReachingMission`
  (`navcore/missions/group_goal_reaching.py`) sets a follower's goal to a
  fixed `formation_offset` (default `Vector2(0.5, 0.5)`) from the
  leader's current position if the follower has drifted more than
  `FOLLOW_DISTANCE_THRESHOLD = 0.5` m from its formation slot, else
  targets the group's actual shared goal directly. **Under the
  default `env.toml` (`group_size=0, num_groups=0`), no groups exist in
  the RL training configuration**, so this code path is inert by
  default.

---

# 15. Complete Computational Graph

```
                         Environment (ground-truth sim state)
                                   │
                     ObservationEncoder.encode(env)
        ┌───────────┬──────────────┼──────────────┬──────────────┐
        │           │              │              │              │
     "robot"   "neighbors"  "neighbor_mask" "neighbor_history" "range_image"
     [nenv,8]  [nenv,10,5]   [nenv,10]     [nenv,8,10,5]      [nenv,1,128,180]
        │           │              │              │              │
        │           │       "neighbor_history_mask" [nenv,8,10]  │
        │           │              │              │              │
        │           └──────┐  ┌────┴──────────────┘              │
        │                  ▼  ▼                                  │
        │        slice [...,2:4] → TemporalEncoder (LSTMCell×8)   │
        │                  │  → [nenv,10,32]                      │
        │                  ▼                                      │
        │        cat(neighbors, temporal) → [nenv,10,37]           │
        │                  │                                      │
        │                  ▼                                      │
        │        HumanHumanAttention (MHA, 8 heads, D=512)        │
        │                  │ → [nenv,10,512]                       │
        │                  ▼                                      │
        │        Linear(512→256) → [nenv,10,256]                   │
        ▼                  │                                      │
RobotStateEncoder           │                                      │
(Linear+ReLU)               │                                      │
   │ → [nenv,256]           │                                      │
   └──────► RobotHumanAttention (MHA, 8 heads, D=256) ◄────────────┘
                  │ → crowd_context [nenv,256]
                  ▼
            LayerNorm → human_context [nenv,256]
                  │                                    [if obstacle branch enabled]
                  │                    range_image ──► RangeImageEncoder (residual CNN)
                  │                                          │ → [nenv,15,128]
                  │                    robot_embedding ──► RobotObstacleAttention
                  │                                          │ → [nenv,128]
                  │                                    Linear(128→256)+LayerNorm
                  │                                          │ → obstacle_context [nenv,256]
                  │                                          ▼
                  └────────────────► ContextFusionGate(human, obstacle) → fused [nenv,256]
                  │  (else: fused_context = human_context)
                  ▼
     RecurrentNodeUpdate (2×Linear+ReLU → concat[128] → GRUCell(128→128) → Linear(128→256))
        inputs: robot_embedding[nenv,256], fused_context[nenv,256],
                hidden_state[nenv,128], not_done_mask[nenv]
                  │ → node_output [nenv,256], new_hidden_state [nenv,128]
                  ▼
            ActorCriticHeads
        ┌─────────┴─────────┐
        ▼                   ▼
  actor tower           critic tower
  (2×Linear+Tanh)      (2×Linear+Tanh)
        │                   │
        ▼                   ▼
  actor_features        Linear(256→1)
  [nenv,256]                │
        │                   ▼
        ▼                value [nenv,1]
  DiagGaussianHead
  (Linear(256→2), learned log_std[2])
        │
        ▼
  Independent(Normal(mean,std),1)
        │
   sample()/mean  ──►  action [nenv,2]
        │
        ▼
  CrowdSimEnv._decode_velocity_action (clip |v|≤v_pref)
        │
        ▼
  Step (robot_velocity_override) → integrate pose, update crowd via ORCA
        │
        ▼
  reward = GoalReachingTask.reward(...)   next observation ─┐
                                                             │
                                        (feeds back into the graph above)
```

---

# 16. Tensor Shape Table

| Tensor | Shape | dtype | Source | Destination | Description |
|---|---|---|---|---|---|
| `robot` (obs) | `[nenv,8]` | float32 | `ObservationEncoder.encode` | `RobotStateEncoder` | goal-rel. x/y, vx, vy, v_pref, radius, cosθ, sinθ |
| `neighbors` (obs) | `[nenv,10,5]` | float32 | `ObservationEncoder.encode` | concat → temporal enc + attention | rel_px, rel_py, vx, vy, radius per neighbor slot |
| `neighbor_mask` (obs) | `[nenv,10]` | int8→bool | `ObservationEncoder.encode` | attention masks | 1 = real neighbor slot |
| `neighbor_history` (obs) | `[nenv,8,10,5]` | float32 | `ObservationEncoder.encode` | `TemporalEncoder` (sliced [...,2:4]) | 8-step per-neighbor history |
| `neighbor_history_mask` (obs) | `[nenv,8,10]` | int8→float | `ObservationEncoder.encode` | `TemporalEncoder` | 1 = real history entry |
| `range_image` (obs) | `[nenv,1,128,180]` | float32 {0,1} | `ObservationEncoder.encode` | `RangeImageEncoder` (optional branch) | binary occupancy image |
| `motion_history` | `[8,nenv,10,2]` | float32 | `policy.forward` (transpose of history) | `TemporalEncoder` | (vx,vy) slice, time-major |
| `temporal_embedding` | `[nenv,10,32]` | float32 | `TemporalEncoder` | `spatial_edge_features` concat | per-neighbor motion embedding |
| `spatial_edge_features` | `[nenv,10,37]` | float32 | concat(neighbors, temporal) | `HumanHumanAttention` | 5+32 per-neighbor feature |
| `human_features` | `[1,nenv,10,37]` | float32 | `.unsqueeze(0)` | `HumanHumanAttention` | seq-len=1 wrapper |
| `human_embeddings` (raw) | `[1,nenv,10,512]` | float32 | `HumanHumanAttention` | `_human_embed_down` | self-attended humans |
| `human_embeddings` (down) | `[1,nenv,10,256]` | float32 | `Linear(512,256)` | `RobotHumanAttention` | projected to shared width |
| `robot_embedding_seq` | `[1,nenv,1,256]` | float32 | `RobotStateEncoder`+unsqueeze | `RobotHumanAttention` | robot query |
| `crowd_context` | `[nenv,256]` | float32 | `RobotHumanAttention` | `human_context_norm` | robot's attended crowd context |
| `robot_embedding` | `[nenv,256]` | float32 | squeeze of `robot_embedding_seq` | `RecurrentNodeUpdate`, `RobotObstacleAttention` | robot's own embedding |
| `human_context` | `[nenv,256]` | float32 | `LayerNorm` | fusion / `RecurrentNodeUpdate` | normalized human branch |
| `obstacle_tokens` | `[nenv,15,128]` | float32 | `RangeImageEncoder` | `RobotObstacleAttention` | angular obstacle tokens |
| `obstacle_context_raw` | `[nenv,128]` | float32 | `RobotObstacleAttention` | proj+norm | robot-attended obstacle context |
| `obstacle_context` | `[nenv,256]` | float32 | `Linear(128,256)`+`LayerNorm` | `ContextFusionGate` | projected obstacle context |
| `fused_context` | `[nenv,256]` | float32 | `ContextFusionGate` or passthrough | `RecurrentNodeUpdate` | gated human/obstacle blend |
| `hidden_state` | `[nenv,128]` | float32 | prev tick / `initial_hidden_state` | `RecurrentNodeUpdate` | GRU hidden state |
| `not_done_mask` | `[nenv]` | float32 | rollout / bootstrap logic | `RecurrentNodeUpdate` | 1=carry hidden, 0=reset |
| `node_output` | `[nenv,256]` | float32 | `RecurrentNodeUpdate` | `ActorCriticHeads` | fused recurrent output |
| `new_hidden_state` | `[nenv,128]` | float32 | `RecurrentNodeUpdate` | next tick / buffer | updated GRU hidden state |
| `value` | `[nenv,1]` | float32 | `ActorCriticHeads` | GAE / value loss | critic estimate |
| `actor_features` | `[nenv,256]` | float32 | `ActorCriticHeads` | `DiagGaussianHead` | pre-distribution features |
| `mean` | `[nenv,2]` | float32 | `DiagGaussianHead.mean_linear` | `Normal` | Gaussian mean action |
| `std` | `[nenv,2]` (broadcast) | float32 | `log_std.clamp().exp()` | `Normal` | Gaussian std (state-independent) |
| `action` | `[nenv,2]` | float32 | `select_action` | `CrowdSimEnv.step` | (vx,vy) sample or mean |
| `log_prob` | `[nenv]` | float32 | `distribution.log_prob(action)` | rollout buffer / PPO ratio | summed over action dim |
| `advantages` | `[T,nenv]` | float32 | GAE recursion | PPO surrogate loss | per-(t,env) advantage |
| `returns` | `[T,nenv]` | float32 | `advantages + values[:T]` | value loss target | GAE-λ return |

---

# 17. Parameter Count

**Not determinable from the available code.** No parameter-count
computation, `sum(p.numel() for p in ...)` call, or logged parameter
total appears anywhere in the provided source (training loop, trainer,
or test scripts). Computing exact counts would require instantiating the
modules and running `numel()` over their state dicts, which is outside
what can be derived from static reading of the code alone; this report
does not fabricate a number. The architectural pieces that *would*
dominate the count, per the shapes in §5/§16, are:
`HumanHumanAttention`'s `nn.MultiheadAttention(512, 8)` and its
`Linear(37→128→512)` embed stack, `RangeImageEncoder`'s residual CNN
stages (when the obstacle branch is enabled), and `ActorCriticHeads`'
two independent `256×256` towers — but no code path in this repository
reports their parameter counts.

---

# 18. Mathematical Formulation

All equations below are transcribed directly from the operations present
in the code (§5–§10); no textbook substitute is used where the code
diverges or where a detail is absent.

**Attention (both `HumanHumanAttention` and `RobotHumanAttention`,
`RobotObstacleAttention`)**: standard PyTorch `nn.MultiheadAttention`
scaled dot-product, i.e. `softmax(QKᵀ/√d_k)V` per head, heads
concatenated and linearly projected — this is `nn.MultiheadAttention`'s
built-in computation; the project code supplies `Q` (from a
`query_proj`/robot embedding), `K`/`V` (from `key_proj`/`value_proj` or
the raw obstacle tokens), and a `key_padding_mask` to exclude padded
slots, but does not re-derive the attention formula itself in Python —
`Not determinable` beyond "PyTorch's standard implementation is used
as-is."

**Masked LSTM (`TemporalEncoder`)**:
```
(h_t, c_t) = LSTMCell(x_t, (h_{t-1}, c_{t-1}))
h_t ← h_t * m_t + h_{t-1} * (1 - m_t)
c_t ← c_t * m_t + c_{t-1} * (1 - m_t)
```
where `m_t ∈ {0,1}` is the per-agent visibility mask at step `t`, and
`LSTMCell` follows PyTorch's standard LSTM cell equations (not re-derived
in the codebase itself).

**GRU (`RecurrentNodeUpdate`)**:
```
z = cat(ReLU(W_r · robot_embedding), ReLU(W_c · crowd_context))   [dim 128]
h_reset = hidden_state * not_done_mask
h' = GRUCell(z, h_reset)                     (PyTorch's standard GRUCell equations)
output = W_o · h' + b_o
```

**Actor distribution / equations**: given in §7.1.

**Critic**: `V(s) = W_v · Tanh(W_2 · Tanh(W_1 · node_output))`
(two-layer critic tower + linear head, §8).

**Reward**: given in §10.1.

**GAE**: given in §9.2.

**PPO loss**:
```
L = -E[min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)]
    + c_vf · E[max((V_θ-R)², (V_clip-R)²)]     (if clip_range_vf set, else plain MSE)
    - c_ent · E[H(π_θ(·|s_t))]
r_t = exp(log π_θ(a_t|s_t) - log π_θ_old(a_t|s_t))
```

---

# 19. Design Decisions

For each decision: **What / Where / Rationale (if documented in code) /
Advantages / Disadvantages / Alternatives**. Rationale is quoted or
paraphrased **only** where the code's own docstrings state it; otherwise
this report states "The rationale is not documented in the code."

1. **Two-tower actor/critic with no shared trunk.**
   Where: `actor_crititc_head.py`. Rationale (documented): "more
   parameters than a shared-trunk architecture, but it is what the
   original publishes and benchmarks, so it is kept faithfully rather
   than 'optimized' into a shared trunk here." Advantages: no
   interference between policy and value gradients. Disadvantages: more
   parameters, more compute, no shared representation learning.
   Alternatives: a shared feature trunk with two small heads.

2. **State-independent log-std.**
   Where: `actiton_distribution.py`. Rationale: not documented beyond
   the module's note that this is a simplification of the original's
   `AddBias`-based approach into "a plain `nn.Parameter`" (mathematically
   equivalent to the original, per the docstring — a faithfulness
   choice, not an exploration-strategy choice). "The rationale [for
   using state-independent std at all, vs. a state-conditioned std] is
   not documented in the code" — the docstring only explains *why this
   port's implementation of it* differs mechanically from the original,
   not why a state-independent std was chosen in the first place.
   Advantages: fewer parameters, simpler; exploration noise is uniform
   across states. Disadvantages: cannot learn to be more/less
   deterministic in easy vs. hard states. Alternatives: a state-dependent
   std head.

3. **No tanh-squashing on the action mean.**
   Where: `actiton_distribution.py`/`select_action`. Not documented.
   Advantage: simpler log-prob math (no Jacobian correction needed).
   Disadvantage: the raw Gaussian sample can exceed the physical
   `v_pref` bound and is corrected only *after* the fact by magnitude
   clipping in `CrowdSimEnv`, with **no correction to the stored
   log-prob** — the trainer's own docstring calls this "the standard
   PPO-on-continuous-control convention" and explicitly accepts the
   approximation. Alternative: `tanh`-squashed Gaussian (SAC-style) with
   a proper log-prob Jacobian correction.

4. **Humans and obstacles attend in two completely separate modules,
   fused only afterward by a learned gate.**
   Where: `policy.py` module docstring, `fusion_gate.py`. Rationale
   (documented, at length): the previous design shared one attention
   module between humans and obstacle ray features; this was replaced
   because "nothing about a human's identity/visibility/motion history
   has anything in common with a ray-cast occupancy sector's, and the
   previous sharing was an artifact of reusing whatever attention
   module already existed, not a deliberate choice." Advantages: each
   branch can specialize; feature-wise (not scalar) gating lets the
   network lean on whichever branch actually carries signal per feature,
   per tick. Disadvantages: doubles the attention-layer parameter/compute
   cost when the obstacle branch is enabled. Alternatives: one combined
   attention pass over a heterogeneous token set (the rejected prior
   design).

5. **Range-image residual CNN instead of a 1D-CNN over per-ray
   features, for the obstacle branch.**
   Where: `policy.py`, `range_image_encoder.py`, `obstacle_encoder.py`
   (superseded). Rationale (documented): a diagnostic probe
   (`probe_obstacle_encoder_mirror.py`) found the old 1D-CNN + global
   average-pool design destroyed left/right directional information
   about nearby obstacles, even in an untrained network. The new design
   replaces global pooling with strided, *learned* spatial-compression
   convolutions all the way to a token sequence, explicitly to preserve
   angular/directional structure. Advantages: preserves spatial
   direction; produces a fixed, meaningful token set with no padding
   case. Disadvantages: heavier compute (residual CNN vs. a shallow 1D
   CNN); the obstacle branch defaults to **disabled**
   (`use_range_image_obstacles=False`), so this improved design is not
   exercised by the default training CLI. Alternatives: the legacy
   1D-CNN branch (still present in the codebase, unused by default).

6. **No minibatching over time in PPO's update; whole-sequence recompute
   every epoch.**
   Where: `rollout_buffer.py`, `recurrent_node_update.py`,
   `crowd_nav_pp_trainer.py`. Rationale (documented): the batched
   multi-timestep GRU training path from the original paper's
   implementation, which handles mid-sequence episode-boundary resets
   inside one call, "can only be verified against a real training loop
   that stores multi-step rollouts a specific way — and no such loop
   exists in navcore yet." Porting it now "would mean writing untestable
   code that also silently commits navcore to the original project's
   specific rollout-storage layout." Advantages: simple, exactly
   correct hidden-state threading (reproduces collection-time resets
   exactly). Disadvantages: `n_epochs × T` **sequential** forward passes
   per update — no parallelism across time, and no gradient reuse across
   epochs; explicitly flagged in the code as a performance concern (see
   §22). Alternatives: chunked BPTT with the original's batched-GRU
   `has_zeros`-splitting logic.

7. **Static obstacles' collision-geometry conversion has a documented,
   un-fixed local/world-frame bug in one code path
   (`base_orca_planner.obstacle_to_vertices`) but a correct
   implementation exists elsewhere
   (`geometry_conversion.obstacle_to_shapely_polygon`).**
   Where: both files' docstrings explicitly cross-reference this;
   `sat.py`'s `ObstacleCollisionDetector` inherits the buggy version's
   output. Rationale (documented): consolidating the four call sites
   that independently do this conversion is flagged as a real,
   pre-existing duplication problem, deliberately not fixed as a side
   effect of an unrelated change ("Flagged as a follow-up, not attempted
   as a side effect of wiring ray_features"). This is a known,
   acknowledged weakness (see §22).

8. **`GoalReachingMission`/`Mission` protocol kept separate from
   `Task`'s reward/termination logic.**
   Where: `mission.py`, `task.py`. Rationale (documented): "Reward and
   termination are RL concerns, not simulation concerns, so they don't
   belong on Mission or on Environment... lets `CrowdSimEnv` stay
   task-agnostic: swap the Task implementation and the same env class
   supports goal-reaching, coverage, or any future task without touching
   `CrowdSimEnv` itself." Advantage: reusability across tasks.
   Disadvantage: none stated in code.

---

# 20. Code Traceability

*Line numbers are not available in the provided source snapshot (no
line-numbered listings were supplied for these files); file, class, and
function names are given precisely as they appear in the code.*

| Concept | File | Class / Function |
|---|---|---|
| Gym environment | `navcore/gym_wrapper/crowd_sim_env.py` | `CrowdSimEnv`, `CrowdSimEnvConfig`, `ActionMode` |
| One tick | `navcore/step/step.py` | `Step.step`, `Step._compute_velocities`, `Step._advance_agent` |
| Observation encoding | `navcore/gym_wrapper/observation_encoder.py` | `ObservationEncoder.encode`, `_encode_neighbors`, `_encode_range_image` |
| Reward/termination | `navcore/gym_wrapper/goal_reaching_task.py` | `GoalReachingTask.reward`, `is_terminated` |
| Env builder | `navcore/builder/environment_builder.py` | `EnvironmentBuilder.build_environment`, `.reset`, `.rebuild_pedestrian` |
| Robot placement | `navcore/builder/robot_builder.py` | `RobotBuilder.generate_pose`, `.generate_goal` |
| Crowd placement | `navcore/builder/crowd_builder.py` | `CrowdBuilder.build_crowd`, `.build_groups` |
| Obstacle placement | `navcore/builder/obstacle_builder.py` | `ObstacleBuilder.build_table`, `.build_boundary` |
| ORCA core | `navcore/policies/base_orca_planner.py` | `BaseORCAPlanner.initialize`, `.compute_velocities` |
| ORCA adapter | `navcore/middleware/orca_middleware.py` | `DecentralizedORCAPlanner.compute_velocities`, `_as_full_state` |
| Full network | `navcore/policies/crowdnav_pp/policy.py` | `CrowdNavPPPolicy.forward`, `.act`, `_substitute_dummy_human` |
| Robot encoder | `navcore/policies/crowdnav_pp/robot_state_encoder.py` | `RobotStateEncoder` |
| Temporal encoder | `navcore/policies/crowdnav_pp/temporal_encoder.py` | `TemporalEncoder.forward` |
| Human-human attention | `navcore/policies/crowdnav_pp/human_human_attention.py` | `HumanHumanAttention.forward` |
| Robot-human attention | `navcore/policies/crowdnav_pp/robot_human_attention.py` | `RobotHumanAttention.forward` |
| Range-image encoder | `navcore/policies/crowdnav_pp/range_image_encoder.py` | `RangeImageEncoder.forward` |
| Robot-obstacle attention | `navcore/policies/crowdnav_pp/robot_obstacle_attention.py` | `RobotObstacleAttention.forward` |
| Fusion gate | `navcore/policies/crowdnav_pp/fusion_gate.py` | `ContextFusionGate.forward` |
| Recurrent update | `navcore/policies/crowdnav_pp/recurrent_node_update.py` | `RecurrentNodeUpdate.forward` |
| Actor/critic heads | `navcore/policies/crowdnav_pp/actor_crititc_head.py` | `ActorCriticHeads.forward` |
| Action distribution | `navcore/policies/crowdnav_pp/actiton_distribution.py` | `DiagGaussianHead.forward`, `select_action` |
| PPO trainer | `navcore/training/crowd_nav_pp/crowd_nav_pp_trainer.py` | `CrowdNavPPTrainer.collect_rollout`, `.update`, `._recompute_sequence` |
| Rollout buffer / GAE | `navcore/training/crowd_nav_pp/rollout_buffer.py` | `RecurrentRolloutBuffer.compute_returns_and_advantages` |
| Vectorized env | `navcore/training/crowd_nav_pp/vec_env.py` | `VecCrowdSimEnv.step`, `.reset` |
| CLI entry point | `navcore/training/crowd_nav_pp/train_crowdnav_pp.py` | `main` |
| Evaluation harness | `navcore/training/crowd_nav_pp/evaluate.py` | `evaluate`, `classify_outcome` |
| Ray sensor | `navcore/entities/components/sensors/obstacle_detector.py` | `ObstacleDetector.sense`, `scan_to_features` |
| Range image | `navcore/entities/components/sensors/range_image.py` | `RangeImageBuilder.build` |
| Crowd sensor | `navcore/entities/components/sensors/sensor.py` | `RangeSensor.observe` |
| Collision checking | `navcore/entities/environment/collision_checker.py` | `CollisionChecker.check_collision` |

---

# 21. Professor Questions (100)

**Environment & simulation loop**

1. *Q: What integration scheme advances agent positions each tick?*
   A: Explicit Euler, `pose += velocity * dt` with `dt = 0.1 s`
   (`Step._advance_agent`).
2. *Q: Is the simulation tick ordering compute-then-apply, or
   apply-as-you-go?* A: Compute-then-apply — every agent's velocity is
   computed before any pose is mutated (`Step._compute_velocities` then
   `_set_velocities`/`_advance_agent`), to avoid first-mover bias.
3. *Q: How is the robot's heading (`theta`) determined?* A: `atan2(vy,
   vx)` of its own velocity, only when speed² > 1e-12 — it is a
   rendering/orientation artifact, not a kinematic state variable that
   constrains motion (the robot is holonomic).
4. *Q: What triggers episode truncation vs. termination?* A: Truncation
   is a fixed step budget (`max_episode_steps=1500`) checked in
   `CrowdSimEnv.step`; termination is `GoalReachingTask.is_terminated`
   (goal reached, collision, or out-of-bounds).
5. *Q: Does the robot see the boundary the same way it sees a table
   obstacle in the range image?* A: Yes — the boundary ring is folded
   into the same flat list of shapely geometries passed to
   `ObstacleDetector.sense`; there is no `HitType.BOUNDARY` distinguishing
   it in the `hit_type` array in this version of the code.
6. *Q: What probability does a pedestrian "pause" (zero velocity) on any
   given tick?* A: 1% (`self.rand.random() < 0.01` in
   `Step._compute_crowd_velocities`).
7. *Q: Is that pause probability affected by the RL policy's seed?* A:
   Not necessarily — `Step`'s own RNG defaults to an unseeded
   `np.random.default_rng()` unless explicitly passed a `rand`, and
   `CrowdSimEnv` does not pass one into its `Step` construction.
8. *Q: How does a pedestrian get a new goal after reaching its old one?*
   A: It isn't reassigned a goal in place — the whole pedestrian object
   is rebuilt with a fresh pose and goal via
   `EnvironmentBuilder.rebuild_pedestrian`.
9. *Q: What arena size is used by default?* A: `arenaSize.width=14`,
   `height=75` (`env.toml`).
10. *Q: How many pedestrians are in the crowd by default?* A: 10
    (`pedestrians.toml.num_pedestrians`).
11. *Q: Are pedestrian groups active by default?* A: No —
    `pedestrians.toml` sets `group_size=0`, `num_groups=0`.
12. *Q: What is `goal_reach_tolerance`?* A: `0.1` (`env.toml`
    `tolerance.goal_reach`), added to the agent's radius for the
    reach-goal distance test.
13. *Q: How is out-of-bounds detected?* A: `Environment.out_of_bounds()`
    checks the robot pose ± `radius + safety_distance` against the arena
    half-extents.
14. *Q: Does `CrowdSimEnv` support parallel environments in-process
    with true multiprocessing?* A: No — `VecCrowdSimEnv` is a
    synchronous, single-process Python loop, explicitly to avoid
    pickling `rvo2`'s C extension.
15. *Q: What happens to a `VecCrowdSimEnv` sub-environment when it's
    done?* A: It is auto-reset immediately, and the terminal observation
    is stashed in `info["terminal_observation"]`.

**Observation pipeline**

16. *Q: What is the robot feature vector's dimensionality and content?*
    A: 8-D: goal-relative x/y, vx, vy, v_pref, radius, cos(theta),
    sin(theta).
17. *Q: How is the neighbor feature vector constructed, and is velocity
    robot-relative?* A: `[rel_px, rel_py, vx, vy, radius]` — position is
    robot-relative, but velocity is the pedestrian's **own** (world-frame)
    velocity, not relative to the robot.
18. *Q: How are neighbors selected when more than `max_neighbors` are
    visible?* A: The `max_neighbors` nearest by Euclidean distance to the
    robot.
19. *Q: How is neighbor history padded when a pedestrian has fewer than
    `history_steps` recorded frames?* A: Left-padded with zeros;
    `start = history_steps - len(track)`.
20. *Q: Does the neighbor-history buffer track only the currently
    selected `max_neighbors`, or every observed pedestrian?* A: Every
    pedestrian ever observed within sensor range, keyed by id — broader
    than the instantaneous top-`max_neighbors` selection.
21. *Q: What triggers a full reset of the neighbor-history buffer?*
    A: `ObservationEncoder.reset()`, called at episode start.
22. *Q: What is the range image's binary semantics?* A: 1.0 = free/
    traversable, 0.0 = blocked (no separate "unknown" channel).
23. *Q: Where in the range image is the robot's own position?* A: The
    last row.
24. *Q: Is the range image cast in a robot-heading-relative frame or a
    fixed world frame?* A: World frame — `heading=0.0` is always passed,
    since the robot is holonomic.
25. *Q: Where is the obstacle/boundary geometry for ray-casting cached,
    and when is it rebuilt?* A: `ObservationEncoder._cached_obstacle_
    polygons`, rebuilt once per episode in `reset(env)`.

**Network architecture**

26. *Q: What type of recurrent cell does the temporal encoder use, and
    why not a batched `nn.LSTM`?* A: `nn.LSTMCell`, looped explicitly
    per timestep, so hidden/cell state can be exactly **frozen** (not
    merely fed a zero input) on any timestep a neighbor wasn't visible —
    a documented correctness choice over the faster batched
    "faster_lstm" approximation.
27. *Q: What happens to a neighbor's temporal embedding if it was never
    visible during the whole history window?* A: All-zero (the LSTM
    state never leaves its zero initialization).
28. *Q: What is `spatial_edge_feature_dim` under default config, and how
    is it composed?* A: `37 = neighbor_feature_dim(5) +
    temporal_hidden_size(32)` (GST prediction disabled by default).
29. *Q: What happens if a `(seq, env)` slot has zero visible humans when
    it reaches `HumanHumanAttention`?* A: It would raise `ValueError` —
    but `CrowdNavPPPolicy._substitute_dummy_human` forces slot 0 visible
    beforehand specifically to prevent this.
30. *Q: Is the "dummy human" substitution a genuinely synthetic
    human state, or repurposed padding?* A: Repurposed padding — slot 0's
    already-zero feature vector is simply marked visible.
31. *Q: What width does `HumanHumanAttention` operate at internally, and
    how does it reach the shared `interaction_embedding_dim`?* A: 512
    internally (`human_human_embedding_size`), down-projected via a
    `Linear(512,256)` (`_human_embed_down`) since 512 ≠ 256.
32. *Q: Why are humans and obstacles never attended over in one shared
    module?* A: Per the code's own documented rationale, a human's
    identity/visibility/motion history has nothing in common with a
    ray-cast occupancy sector's, and prior sharing was an artifact of
    reuse, not deliberate design.
33. *Q: What projects the robot embedding into the obstacle branch's own
    latent space, and why is this projection applied *before* attention
    rather than after?* A: `RobotObstacleAttention.query_proj`
    (`Linear(256,128)`), applied before attention so the obstacle branch
    can develop its own representation rather than being constrained to
    live in the 256-d human-branch space from the start.
34. *Q: How many obstacle tokens does `RangeImageEncoder` produce by
    default, and how is that count guaranteed to tile the backbone's
    output width exactly?* A: 15, guaranteed via a divisibility check
    (`__post_init__`) between `backbone_output_width` and
    `num_obstacle_tokens`, so `token_compress`'s kernel/stride tiles
    exactly with no padding needed.
35. *Q: What padding convention does `_AngularConv2d` use, and why does
    it differ per spatial axis?* A: Circular padding along width (the
    ray-fan/angular axis, since ray 0 and ray W-1 are angularly
    adjacent), zero padding along height (a genuinely bounded axis — no
    "beyond max range").
36. *Q: Does `RangeImageEncoder` use any pooling operation?* A: No —
    the docstring states pooling would destroy left/right directional
    information (per the project's own diagnostic probe), so every
    spatial reduction is a strided, learned convolution instead.
37. *Q: What fuses the human and obstacle context vectors, and is the
    gate scalar or feature-wise?* A: `ContextFusionGate`, a
    feature-wise sigmoid gate, shape `[B, 256]` — not a single scalar.
38. *Q: What is `RecurrentNodeUpdate`'s hidden size, and how is the
    incoming hidden state reset at an episode boundary?* A: 128
    (`rnn_hidden_size`); `reset_hidden = hidden_state *
    not_done_mask.unsqueeze(-1)`.
39. *Q: Is `RecurrentNodeUpdate` ported for both the single-tick rollout
    path and the multi-timestep batched training path from the original
    paper?* A: Only the single-tick path; the batched path is explicitly
    deferred, per the module docstring.
40. *Q: Do the actor and critic towers share any layers?* A: No —
    two entirely independent MLP towers from the recurrent output
    onward, by deliberate faithfulness to the original.
41. *Q: What gain is used for orthogonal initialization of the actor/
    critic hidden layers vs. the final value projection?* A: `sqrt(2)`
    for hidden layers, `0.01` for the final value projection (keeps
    initial value estimates small).
42. *Q: Is the action distribution's std a function of the current
    state?* A: No — a single learnable parameter vector, independent of
    input, clamped to `[e^-3, e^0.5]`.
43. *Q: Is there any tanh-squashing applied to the sampled action?*
    A: No.
44. *Q: What library primitive computes the summed log-probability over
    action dimensions?* A: `torch.distributions.Independent(Normal(...),
    1)`.

**Data flow / shapes**

45. *Q: What shape is the observation just before it enters
    `TemporalEncoder`?* A: `motion_history: [8, nenv, 10, 2]`,
    `history_mask: [8, nenv, 10]` — time-major, sliced to (vx, vy).
46. *Q: Why does `CrowdNavPPPolicy.forward` insert a `seq_len=1` axis
    before calling the attention modules?* A: Because
    `HumanHumanAttention`/`RobotHumanAttention` carry an explicit
    `seq_len` axis in their contract (for potential future multi-step
    use), but this policy is single-tick-only, so `seq_len` is always 1.
47. *Q: What is `robot_embedding_seq`'s shape right before
    `RobotHumanAttention`?* A: `[1, nenv, 1, 256]`.
48. *Q: What is the output shape of `RecurrentNodeUpdate`, and where
    does it go next?* A: `[nenv, 256]` (`node_output`), into
    `ActorCriticHeads`.
49. *Q: Where does the value estimate's final shape come from?* A:
    `ActorCriticHeads.critic_linear` outputs `[nenv, 1]`; the trainer
    squeezes the last dim to `[nenv]` before storing it.

**PPO / training**

50. *Q: How many PPO epochs are run per update, and what does each epoch
    actually recompute?* A: 4 by default; each epoch reruns the **entire
    stored sequence**, one timestep at a time, through `policy.forward`
    to get fresh `log_prob`/`value`/`entropy`.
51. *Q: Is there minibatch sampling over `(timestep, env)` pairs during
    the PPO update?* A: No — the whole `T × n_envs` batch is used every
    epoch, in strict time order.
52. *Q: How is the GRU hidden state kept consistent between rollout
    collection and the PPO recompute pass?* A: The buffer stores
    `initial_hidden_state` (the hidden state right before the rollout's
    first step) and each step's `not_done_mask`; recompute starts from
    that same seed and replays the same reset points.
53. *Q: What is the advantage normalization scope — per epoch, per
    minibatch, or once per rollout?* A: Once, over the whole batch,
    before the epoch loop begins (not renormalized per epoch).
54. *Q: What PPO clip range is used for the surrogate objective vs. the
    value function?* A: `clip_range=0.15` (policy), `clip_range_vf=0.2`
    (value), both defaults.
55. *Q: What happens to the value loss if `clip_range_vf` is `None`?*
    A: It falls back to plain MSE against the returns.
56. *Q: Is entropy regularization active by default?* A: The default
    `ent_coef=0.0`, so the entropy term is computed and logged but
    contributes zero weight to the loss by default.
57. *Q: What happens to gradient clipping if `max_grad_norm=None`?* A:
    The clip function is still called, but with `float("inf")` as the
    max norm — the norm is computed and logged, but no actual clipping
    occurs.
58. *Q: What optimizer and learning rate are used?* A: `Adam`,
    `lr=3e-4` by default; no learning-rate scheduler exists in the code.
59. *Q: Does the trainer implement automatic mixed precision?* A: Not
    determinable from the available code — no AMP/autocast usage exists.
60. *Q: How is `explained_variance` computed, and is it part of the
    loss?* A: `1 - Var(returns - old_values)/Var(returns)`; it is
    reported for diagnostics only, not used in the loss.
61. *Q: What is stored in a PPO checkpoint?* A: Policy state dict,
    optimizer state dict, `total_steps`, `total_updates`, and the
    `PPOConfig`/`CrowdNavPPPolicyConfig` dataclasses.
62. *Q: Does `load_checkpoint` restore the saved hyperparameter
    configs into the live trainer?* A: No — only
    `policy_state_dict`/`optimizer_state_dict`/`total_steps`/
    `total_updates` are read back.
63. *Q: On what cadence are checkpoints saved during training?* A: Every
    `checkpoint_every` PPO updates (not every rollout, not by wall-clock
    time), if `checkpoint_dir` is given.
64. *Q: Does `train_crowdnav_pp.py`'s CLI actually enable the obstacle
    branch?* A: No — the line that would pass
    `use_obstacle_encoder=args.use_obstacle_encoder` is commented out,
    so `use_range_image_obstacles` stays at its default (`False`).
65. *Q: What global seeding does `set_seed()` perform, and is it
    sufficient to make a full training run reproducible?* A: It seeds
    `random`, `numpy`, and both `torch.manual_seed`/
    `cuda.manual_seed_all` — but `Step`'s own pause-event RNG is not
    seeded through this path in `CrowdSimEnv`'s default construction
    (see Q7), so full end-to-end determinism is not guaranteed by this
    alone.

**Reward**

66. *Q: What is the exact per-tick reward equation implemented?* A:
    `-0.01 + 5.0*(d_{t-1}-d_t) + (-25 if collision) + (-25 if
    out_of_bounds) + (+50 if goal reached)`.
67. *Q: Is progress measured before or after the tick's movement is
    integrated?* A: After — `reward()` is called following `Step.step()`
    in `CrowdSimEnv.step()`.
68. *Q: Does the reward function reward keeping distance from
    pedestrians?* A: Not determinable / not present — no separate
    clearance term exists in `GoalReachingTask.reward`.
69. *Q: Can the collision penalty and the goal bonus both apply on the
    same tick?* A: Yes, if `collided` and `_reached_goal()` are both
    true that tick, both terms are added (no precedence logic in the
    reward function itself — precedence only appears in the *evaluation*
    harness's `classify_outcome`, not in the reward computation).

**Sensors**

70. *Q: Does `RangeSensor.observe()` actually restrict neighbors by
    field-of-view, or only by distance?* A: Only by Euclidean distance —
    `self.fov` is computed at construction but not used inside the
    distance-only test in `observe()`.
71. *Q: How many rays does the default `ObstacleDetector` cast, and over
    what field of view?* A: 180 rays, over a full `2π` (360°) field of
    view.
72. *Q: What happens to a ray that hits nothing within `max_range`?* A:
    Its recorded distance is set to `max_range`, `hit_mask` is `False`,
    and `relative_positions` is `(0,0)`.
73. *Q: Does the code's `HitType` enum actually distinguish an obstacle
    hit from a boundary hit?* A: No — despite the module docstring
    describing a `HitType.BOUNDARY`, the enum in the provided file only
    defines `NONE` and `OBSTACLE`; a boundary hit is reported as
    `OBSTACLE`.
74. *Q: What is `RAY_FEATURE_DIM`, and which network branch actually
    consumes `scan_to_features`'s output?* A: 6; only the legacy 1D-CNN
    `ObstacleEncoder`, which is not constructed by `CrowdNavPPPolicy`'s
    default architecture.

**Robot / human pipelines**

75. *Q: Is the robot's motion constrained by any non-holonomic
    kinematics?* A: No — it integrates `(vx, vy)` directly with no
    heading-dependent motion constraint; `robot.toml` explicitly sets
    `chassis = "holonomic"`.
76. *Q: When the robot's velocity is overridden by the RL action, does
    the ORCA planner run at all for the robot that tick?* A: No —
    `_compute_robot_velocity` returns the override immediately, bypassing
    `self.robot_planner` entirely.
77. *Q: How does a neighboring pedestrian's "intended" velocity enter
    the robot's local ORCA solve when the robot itself IS ORCA-driven
    (waypoint mode)?* A: Via a synthetic one-step-ahead goal (`pose +
    velocity`) constructed for each neighbor, since a sensor observation
    carries no real goal/intent information.
78. *Q: Does `BaseORCAPlanner` read `orca.toml`'s `agent_radius`/
    `max_speed` keys?* A: No — only `neighbor_dist`, `max_neighbors`,
    `time_horizon`, `time_horizon_obst` are read; per-agent radius/speed
    come from each tick's `FullState` instead.
79. *Q: Is a fresh RVO2 simulator constructed every tick for the
    decentralized planner, or is one simulator reused?* A: A fresh one
    every call — RVO2 cannot add/remove agents from a live simulator, and
    the neighbor set changes every call.
80. *Q: How is a pedestrian's radius/preferred-speed randomized, and
    under what condition does that randomization apply?* A:
    `radius *= U(0.8,1.2)`, `v_pref *= U(0.8,1.2)`, only if both
    `randomize_pedestrian_radius` and `randomize_pedestrian_v_pref` are
    `True` in `pedestrians.toml` (both are, by default).
81. *Q: What happens to a pedestrian's goal once reached — is a new goal
    assigned, or is the pedestrian entirely rebuilt?* A: Entirely
    rebuilt (new pose and goal), via `rebuild_pedestrian`, keeping the
    same id.
82. *Q: Are pedestrian groups active in the RL training configuration by
    default?* A: No — `env.toml` sets `group_size=0`, `num_groups=0`.
83. *Q: What formation offset does a default `GroupGoalReachingMission`
    use for a follower?* A: `Vector2(0.5, 0.5)`, if none is given.

**Miscellaneous / cross-cutting**

84. *Q: Does the `Step` class ever mutate `agent.goal` for planning
    purposes?* A: No — a throwaway `FullState` with `goal` set to the
    mission's target is used for planning only; `agent.goal` itself is
    never touched, so downstream goal-reached checks/reward always see
    the agent's real destination.
85. *Q: What discount factor and GAE λ are used by default?* A:
    `gamma=0.99`, `gae_lambda=0.95`.
86. *Q: Is `torch.distributions.Normal.sample()` or `.rsample()` used
    for actual rollout action selection?* A: `.sample()` (inside
    `select_action`), via `distribution.sample()` when not deterministic
    — non-differentiable, as appropriate for rollout collection.
87. *Q: Where does the network's `entropy()` computation come from,
    mathematically?* A: PyTorch's built-in `Normal.entropy()`, summed
    over the action dimension by the `Independent` wrapper — no custom
    formula is written in the codebase.
88. *Q: What does `_classify_outcome`'s precedence order imply about how
    a tick satisfying both collision and out-of-bounds is scored?* A:
    Collision takes precedence over out-of-bounds, which takes precedence
    over success, which takes precedence over timeout.
89. *Q: Does `CrowdSimEnv` ever render during training by default?* A:
    No — `render_mode` defaults to `None`; the trainer only calls
    `_render_env()` if `--render` was passed, targeting only the first
    sub-environment.
90. *Q: What is `_boundary_clearance()` used for in `RobotBuilder.
    generate_goal`, and what value does it use?* A: Shrinks the goal
    sampling rectangle inward by `robot.radius + safety_distance`, so a
    sampled goal cannot coincide with a wall-adjacent, collision-flagging
    position.
91. *Q: What rejection-sampling attempt limit does `RobotBuilder` use
    before raising an error?* A: `MAX_PLACEMENT_ATTEMPTS = 200`.
92. *Q: Is the `"boundary"` obstacle tested the same way as a table when
    checking whether a sampled point is inside an obstacle?* A: No — it
    is explicitly excluded in `_inside_obstacle`, since its polygon
    represents the *allowed* interior region, not a forbidden footprint.
93. *Q: Does `EnvironmentBuilder.reset()` reuse the same RNG stream
    across obstacle, crowd, and robot placement, or three independent
    ones?* A: One shared `np.random.Generator`, passed to all three
    builders and advanced sequentially in call order (obstacles → crowd
    → robot).
94. *Q: What does `not_done_mask` represent, and where does its value
    come from during rollout collection?* A: `1.0` to carry the GRU
    hidden state forward, `0.0` to reset it; derived from `1 -
    previous_step_done` for each env.
95. *Q: Is the bootstrap value for GAE computed with gradient tracking
    enabled?* A: No — `collect_rollout`'s bootstrap `policy.forward` call
    is wrapped in `torch.no_grad()`.
96. *Q: What is `RecurrentRolloutBuffer._OBS_KEYS`, and does it include
    `range_image` even when the obstacle branch is disabled?* A: Yes —
    all six observation keys, including `range_image`, are always stored
    per step regardless of whether the obstacle branch is enabled.
97. *Q: How does `CrowdNavPPTrainer` decide whether to pass
    `range_image` into `policy.forward`/`policy.act`?* A: Conditionally,
    based on `self.policy.config.use_range_image_obstacles`.
98. *Q: What happens if `range_image` is passed to `forward()` while the
    obstacle branch is disabled?* A: `ValueError` is raised.
99. *Q: What happens if the obstacle branch is enabled but `range_image`
    is omitted from `forward()`?* A: `ValueError` is raised.
100. *Q: Does anything in the codebase compute or log the total
    parameter count of `CrowdNavPPPolicy`?* A: Not determinable from the
    available code — no such computation exists anywhere in the provided
    source.

---

# 22. Weaknesses

**Architectural:**
- The obstacle branch (range-image CNN + cross-attention + gated fusion)
  is fully implemented but **disabled by default** in the actual training
  CLI (`train_crowdnav_pp.py`'s line wiring `use_obstacle_encoder` is
  commented out), so the shipped training configuration never exercises
  it, despite the codebase's own diagnostic work
  (`probe_obstacle_encoder_mirror.py`) motivating its design.
- `HumanHumanAttention`'s "raise on fully masked row" behavior is worked
  around by repurposing an already-zero padding slot as a fake "visible"
  human (`_substitute_dummy_human`), rather than a genuinely synthetic
  dummy state — the network's true behavior on a zero-human tick is
  therefore governed by whatever an all-zero feature vector happens to
  produce through the attention/embedding stack, not by a deliberately
  designed placeholder.
- `ObstacleDetector`'s `HitType` enum does not actually implement the
  `BOUNDARY` distinction its own module docstring claims exists — a
  boundary hit and an obstacle hit are indistinguishable in `hit_type`.
- A documented, unfixed local-vs-world-frame bug exists in
  `base_orca_planner.obstacle_to_vertices` for `Rectangle` geometry
  (returns local-origin, not world-translated vertices), inherited by
  `SAT.ObstacleCollisionDetector`; a correct implementation exists in
  parallel (`geometry_conversion.obstacle_to_shapely_polygon`) but the
  four call sites that each independently re-derive obstacle-to-vertices
  conversion are not consolidated — this is explicitly flagged in the
  code as an acknowledged, unresolved duplication/correctness risk.

**Computational / training bottlenecks:**
- `_recompute_sequence()` performs one `policy.forward()` call **per
  buffered timestep, per PPO epoch**, sequentially (`n_epochs × n_steps`
  = `4 × 512 = 2048` sequential forward passes per update by default) —
  no time-axis batching, no truncated BPTT chunking. This is a direct
  consequence of `RecurrentNodeUpdate` only implementing the
  single-timestep GRU path (documented as a deliberate scope choice, not
  an oversight, but still a real cost).
- `DecentralizedORCAPlanner.compute_velocities` constructs a **brand-new
  RVO2 simulator on every call**, for every agent, every tick (robot
  and each of up to 10 pedestrians) — i.e. potentially 11 fresh
  `rvo2.PyRVOSimulator` constructions per simulation tick.
- `VecCrowdSimEnv` is single-process and sequential across `n_envs` — no
  process- or thread-level parallelism, so wall-clock throughput scales
  linearly with `n_envs` on a single core for the environment-stepping
  portion of the loop.

**Memory:**
- `RecurrentRolloutBuffer` stores every observation key (including the
  `[1,128,180]` range image) for every one of `n_steps × n_envs` steps in
  Python lists of `numpy` arrays before stacking — `Not determinable`
  whether this is a measured problem in practice, but the range-image
  tensor is by far the largest per-step observation component and is
  stored unconditionally regardless of whether the obstacle branch is
  even enabled.

**Possible bugs / design limitations (as documented in the code
itself):**
- `Step`'s per-tick pedestrian-pause RNG (`self.rand`) is not seeded
  through `CrowdSimEnv`'s construction path, so full determinism from a
  fixed `CrowdSimEnv`/training seed is not guaranteed for that specific
  source of randomness (see Q7, §21).
- The action's log-probability used throughout PPO is computed on the
  **raw, pre-magnitude-clip** sampled action, while the actually-applied
  action to the simulator is the clipped one — an approximation the
  trainer's own docstring acknowledges without a Jacobian correction.
- `evaluate.py` is hardcoded to assume `use_obstacle_encoder=True` for
  its default checkpoint family (`USE_OBSTACLE_ENCODER` comment /
  `evaluate()`'s `use_obstacle_encoder=True` default parameter), which is
  inconsistent with the training CLI's default of leaving the obstacle
  branch disabled — evaluating a checkpoint trained under the CLI's
  defaults with `evaluate.py`'s own defaults would mismatch architecture
  configuration unless explicitly overridden with `--no-obstacle-encoder`.

---

# 23. Suggested Improvements

Ranked informally by (expected gain / implementation difficulty /
novelty / risk), as *suggestions* — none of these are implemented in the
current code, and this section is explicitly separated from the
descriptive report above.

1. **Enable and evaluate the obstacle branch in the actual training
   CLI.** Gain: potentially large (the codebase's own diagnostic work
   argues the previous obstacle representation was structurally blind to
   direction; the replacement exists but is unused). Difficulty: low
   (uncomment/wire one constructor argument, add the CLI flag path that
   already exists as `--use-obstacle-encoder` but is not consumed).
   Novelty: none (already implemented). Risk: low, but increases compute
   cost per rollout.
2. **Batch the PPO recompute pass across the time axis** (chunked BPTT,
   or the originally-deferred batched-GRU path) instead of `T` sequential
   single-step calls per epoch. Gain: likely large training-speed
   improvement. Difficulty: moderate-to-high (requires designing and
   testing the mid-sequence episode-reset splitting logic the code
   currently defers). Novelty: low (well-established RL-infrastructure
   technique). Risk: moderate — correctness of hidden-state resets inside
   a batched call is exactly the subtlety the code's own docstring flags
   as needing careful, tested design.
3. **Reuse (or pool) RVO2 simulators instead of constructing a fresh one
   per agent per tick.** Gain: moderate simulation-speed improvement.
   Difficulty: moderate (requires reconciling with `BaseORCAPlanner`'s
   fixed-population invariant). Novelty: low. Risk: low-to-moderate
   (must preserve exact ORCA semantics).
4. **Fix/consolidate the `Rectangle` obstacle world-frame conversion
   bug** (`base_orca_planner.obstacle_to_vertices`) into the already-
   correct `geometry_conversion.obstacle_to_shapely_polygon`. Gain:
   correctness fix for rectangular obstacles away from the origin, in
   both ORCA obstacle avoidance and SAT collision detection. Difficulty:
   low-to-moderate (touches three already-tested files). Novelty: none.
   Risk: low, contained to obstacle-geometry code paths.
5. **Add a state-dependent (or otherwise adaptive) action standard
   deviation**, and/or a `tanh`-squashed Gaussian with a proper log-prob
   Jacobian correction for action bounds. Gain: uncertain (research
   question) — could improve exploration/action-bound consistency.
   Difficulty: moderate. Novelty: standard technique, not novel. Risk:
   moderate — changes the action distribution's mathematics and would
   require re-deriving/re-validating the PPO ratio computation.
6. **Add a validation/held-out evaluation loop with best-checkpoint
   selection** inside the training pipeline (currently checkpoints are
   saved purely on a fixed update cadence, with no automatic evaluation
   or best-model tracking). Gain: moderate (better model selection).
   Difficulty: low (compose the existing `evaluate.py` harness into the
   training loop periodically). Novelty: none. Risk: low.
