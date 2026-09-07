"""LocalAvoidancePlanner: candidate-based safe-point selection for the robot.

Design context
--------------
The robot's global planners (``BoustrophedonPlanner``, ``GraphSearchPlanner``,
...) are built around static assumptions -- fixed obstacles, a lane pattern, a
discrete waypoint. None of them react to human motion tick-by-tick; that
reaction is deliberately kept out of the global planner's stateful lane
bookkeeping (see ``BoustrophedonPlanner.pauses_during_avoidance``). This module
is the missing local-reaction piece: given the humans currently visible to the
robot's sensor, it samples a local ring of candidate points around the robot,
scores each one by predicted safety, and returns the least-dangerous candidate
as a temporary detour target for whichever ``Mission``/state machine owns the
robot's ``AVOIDING`` state.

This class produces *targets only* -- never velocities or poses -- consistent
with the project's ``Mission``/``Policy`` split (see
``navcore.missions.mission``). It does not mutate ``agent`` itself; the caller
(the mission's ``AVOIDING`` state handler) is responsible for writing the
returned point wherever the active mission reads its target from.

Ground-truth vs. observed state (explicit, on purpose)
--------------------------------------------------------
Project convention is that *collision detection* -- did a collision actually
happen -- must use ground-truth crowd state (see
``navcore.entities.environment.collision_checker.CollisionChecker``). This
module is not collision detection; it is planning, and a robot can only plan
around what its own sensor reports. It is intentionally built on
``dict[int, ObservableState]`` -- the robot's currently visible neighbors, as
produced by ``agent.sensor.observe(...)`` -- not ground truth. Do not widen
this to accept ground-truth state without re-deriving that decision; doing so
would let the planner react to humans the robot cannot actually perceive.

Open architectural question (flagged, not resolved here)
----------------------------------------------------------
Whether this planner should bias candidate selection toward the active global
planner's current waypoint ("coupled concurrency") or score candidates on
safety alone, letting the global planner replan once control returns to it
("cosmetic concurrency"), is still an open project decision. This
implementation takes the *cosmetic* path: ``reference_goal`` is accepted as an
optional soft pull toward progress, but nothing here reaches into a live
``Mission`` or ``Planner`` instance. Revisit ``_compute_efficiency`` if coupled
concurrency is chosen instead.

Duplicate-logic note
---------------------
``navcore.collision_predictor.sat.SAT.checkIntrusionSAT`` already computes a
closed-form worst-case-separation interval, but for the *agent's actual* pose
and velocity -- its signature has no way to ask "what if the agent moved
toward this candidate instead?" Rather than silently re-deriving near-identical
CPA math a third time in this file, the shared clearance computation is
factored out below as ``closest_approach_clearance``, a candidate for
extraction into ``navcore/collision_predictor/`` so ``SAT`` could eventually
call the same core math. Flagging this rather than refactoring ``SAT``
unilaterally, since which of the project's three parallel collision-detection
paths (``entities.CollisionChecker``, ``step.CollisionChecker``,
``SAT.check_collision``) is authoritative is still an open decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from navcore.entities.agents.agent import Agent
from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.components.state import ObservableState


@dataclass(slots=True)
class Candidate:
    """One sampled escape point and its evaluated safety cost.

    Mutable by design -- ``LocalAvoidancePlanner`` fills in ``risk``/``valid``
    across the scoring pipeline rather than rebuilding candidates each stage.

    Attributes:
        position: The candidate point, in world coordinates.
        risk: The candidate's combined safety + efficiency cost. Only
            meaningful when ``valid`` is ``True``.
        valid: Whether this candidate survived hard rejection (a predicted
            collision within the planning horizon sets this ``False``).
    """

    position: Vector2
    risk: float = 0.0
    valid: bool = True


def closest_approach_clearance(
    candidate_pos: Vector2,
    candidate_vel: Vector2,
    robot_radius: float,
    human_pos: Vector2,
    human_vel: Vector2,
    human_radius: float,
    safety_margin: float,
    horizon: float,
) -> float:
    """Return the worst-case (minimum) separation over ``[0, horizon]``.

    Closed-form constant-velocity closest-point-of-approach: solves for the
    time at which a hypothetical candidate moving in a straight line at
    ``candidate_vel`` comes closest to one human continuing in a straight
    line at ``human_vel``, clamped to ``horizon`` (a human already receding
    isn't penalized for a future beyond the horizon; a human that would only
    close in after the horizon isn't yet this candidate's problem).

    Args:
        candidate_pos: The candidate's world position.
        candidate_vel: The robot's hypothetical velocity if it moved toward
            this candidate (see ``LocalAvoidancePlanner.compute_candidate_velocity``).
        robot_radius: The robot's physical radius.
        human_pos: The human's current world position.
        human_vel: The human's current velocity.
        human_radius: The human's physical radius.
        safety_margin: Additional buffer added to both radii.
        horizon: Planning horizon, in seconds.

    Returns:
        Distance between the two bodies' surfaces at the time of closest
        approach. Negative means the bodies overlap at some point within
        the horizon -- a predicted collision, not merely "risky"; callers
        should hard-reject on this rather than treat it as a high score.
    """
    combined_radius = robot_radius + human_radius + safety_margin

    rel_pos = candidate_pos - human_pos
    rel_vel = candidate_vel - human_vel
    rel_speed_sq = rel_vel.magnitude_squared()

    if rel_speed_sq < 1e-12:
        # No relative motion: separation never changes, so t*=0 is exact
        # rather than an arbitrary convention.
        t_star = 0.0
    else:
        t_star = -rel_pos.dot(rel_vel) / rel_speed_sq
        t_star = min(max(t_star, 0.0), horizon)

    closest_sep = rel_pos + rel_vel * t_star
    return closest_sep.magnitude() - combined_radius


class LocalAvoidancePlanner:
    """Candidate-based local safe-point selection for one agent (the robot).

    Samples a ring of candidate positions around ``agent``, scores each by a
    safety-dominant cost derived from every currently visible human's
    predicted closest approach plus a narrow-corridor gap penalty, and
    returns the least-dangerous candidate as a temporary escape target.

    Attributes:
        agent: The agent this planner selects escape points for (the robot).
        arena_width: Arena half-extent bound on the x-axis (candidates
            outside this are discarded during sampling).
        arena_height: Same, for the y-axis.
        config: Local-avoidance tunables, merged over ``DEFAULTS`` so
            nothing breaks if a config file doesn't define these keys yet.
    """

    #: Sane fallback values for every tunable, so a config file that omits
    #: some or all of these keys still produces a working planner. Mirrors
    #: the getattr-with-default pattern used for optional tunables
    #: elsewhere in the project, adapted to navcore's dict-based TOML config
    #: convention rather than attribute access.
    DEFAULTS: dict[str, float] = {
        "search_radius": 2.0,
        "radial_resolution": 0.2,
        "angular_resolution_deg": 15.0,
        "horizon": 3.0,
        "k_v": 0.5,
        "beta": 4.0,
        "lambda_gap": 1.0,
        "gap_scale": 1.0,
        "barrier_eps": 1e-3,
        "w_travel_time": 0.1,
        "w_progress": 0.1,
        "hysteresis_bonus": 0.05,
        "hysteresis_radius": 0.05,
    }

    def __init__(
        self,
        agent: Agent,
        arena_width: float,
        arena_height: float,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.agent = agent
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.config: dict[str, Any] = {**self.DEFAULTS, **(config or {})}
        self.candidates: list[Candidate] = []
        self._prev_safe_point: Vector2 | None = None

    # -- candidate generation ---------------------------------------------

    def build_local_risk_map(self) -> list[Candidate]:
        """Sample a polar ring of candidates around the agent's current pose.

        Returns:
            The freshly sampled candidates (also stored on
            ``self.candidates``). Candidates outside the arena bounds are
            skipped rather than clamped, since a clamped-to-boundary
            candidate would misrepresent its actual sampled position.

        Raises:
            RuntimeError: If ``agent.pose`` has not been set yet.
        """
        if self.agent.pose is None:
            raise RuntimeError("Agent pose has not been initialized.")

        robot_pos = Vector2(self.agent.pose.px, self.agent.pose.py)
        radius = self.config["search_radius"]
        radial_step = self.config["radial_resolution"]
        angular_step_deg = self.config["angular_resolution_deg"]

        candidates: list[Candidate] = []
        r = radial_step
        while r <= radius + 1e-9:
            angle_deg = 0.0
            while angle_deg < 360.0:
                theta = math.radians(angle_deg)
                point = robot_pos + Vector2(math.cos(theta), math.sin(theta)) * r
                if (
                    -self.arena_width <= point.x <= self.arena_width
                    and -self.arena_height <= point.y <= self.arena_height
                ):
                    candidates.append(Candidate(position=point))
                angle_deg += angular_step_deg
            r += radial_step

        self.candidates = candidates
        return candidates

    # -- per-candidate kinematics ------------------------------------------

    def compute_candidate_velocity(self, point: Vector2) -> Vector2:
        """Return the straight-line velocity from the agent's pose to ``point``.

        Returns the zero vector if ``point`` coincides with the agent's
        current position, mirroring
        ``BaseORCAPlanner._preferred_velocity``'s divide-by-zero guard.

        Raises:
            RuntimeError: If ``agent.pose`` has not been set yet.
        """
        if self.agent.pose is None:
            raise RuntimeError("Agent pose has not been initialized.")
        robot_pos = Vector2(self.agent.pose.px, self.agent.pose.py)
        direction = point - robot_pos
        distance = direction.magnitude()
        if distance < 1e-8:
            return Vector2.zero()
        return direction * (self.agent.v_pref / distance)

    def compute_travel_time(self, point: Vector2) -> float:
        """Return time-to-reach ``point`` at the agent's preferred speed.

        Raises:
            RuntimeError: If ``agent.pose`` has not been set yet.
        """
        if self.agent.pose is None:
            raise RuntimeError("Agent pose has not been initialized.")
        robot_pos = Vector2(self.agent.pose.px, self.agent.pose.py)
        return robot_pos.distance_to(point) / max(self.agent.v_pref, 1e-8)

    # -- scoring -----------------------------------------------------------

    def compute_point_risk(
        self,
        point: Vector2,
        human_states: dict[int, ObservableState],
        reference_goal: Vector2 | None = None,
    ) -> tuple[bool, float]:
        """Score one candidate point.

        Args:
            point: The candidate position to evaluate.
            human_states: The robot's currently visible neighbors.
            reference_goal: Optional soft pull toward this point, used only
                to break near-ties between otherwise-safe candidates.

        Returns:
            ``(valid, risk)``. ``valid`` is ``False`` (and ``risk`` is
            ``math.inf``) whenever any visible human's predicted closest
            approach overlaps the robot within ``horizon`` -- a hard
            rejection, since that is a provable predicted collision under
            the constant-velocity model, not merely "risky."
        """
        safety_cost = self._compute_safety_cost(point, human_states)
        if not math.isfinite(safety_cost):
            return False, math.inf

        efficiency = self._compute_efficiency(point, reference_goal)
        return True, safety_cost + efficiency

    def _compute_safety_cost(
        self, point: Vector2, human_states: dict[int, ObservableState]
    ) -> float:
        """Safety-only cost for one candidate.

        Pipeline:
          1. For each visible human, compute the closed-form minimum
             separation over ``horizon`` (see ``closest_approach_clearance``).
             Non-positive clearance is a provable predicted collision within
             the horizon -> hard reject immediately, not "high risk."
          2. Turn surviving clearances into bounded per-human risks whose
             effective danger radius grows with closing speed -- a human
             closing fast is dangerous farther out than one moving parallel
             or away.
          3. Aggregate across humans with a soft-max (log-sum-exp), not
             ``max()`` and not a plain sum: ``max()`` discards information
             about multiple simultaneous mild threats; a sum lets many
             far-away, individually-safe humans outvote one genuinely close
             one. ``beta`` interpolates between the two.
          4. Add a narrow-corridor penalty for candidates that look fine
             per-human but sit in a gap too tight to actually pass through.
          5. Pass the combined danger signal through a barrier so it
             diverges as danger saturates -- this makes the safety term
             structurally dominate the bounded efficiency term without a
             hand-tuned weight ratio.
        """
        robot_radius = self.agent.radius
        safety_margin = float(self.agent.config["safety"]["safety_margin"])
        horizon = self.config["horizon"]
        k_v = self.config["k_v"]
        beta = self.config["beta"]

        candidate_vel = self.compute_candidate_velocity(point)

        risks: list[float] = []
        clearances: list[float] = []
        visible: list[ObservableState] = []

        for human in human_states.values():
            human_pos = Vector2(human.pose.px, human.pose.py)
            human_vel = Vector2(human.velocity.vx, human.velocity.vy)

            clearance = closest_approach_clearance(
                point,
                candidate_vel,
                robot_radius,
                human_pos,
                human_vel,
                human.radius,
                safety_margin,
                horizon,
            )
            if clearance <= 0.0:
                return math.inf  # Predicted collision within the horizon.

            combined_radius = robot_radius + human.radius + safety_margin
            rel_speed = (candidate_vel - human_vel).magnitude()
            danger_radius = combined_radius + k_v * rel_speed
            risks.append(math.exp(-clearance / danger_radius))
            clearances.append(clearance)
            visible.append(human)

        if not risks:
            return 0.0  # No visible humans -- safety term is zero.

        # Numerically stable log-sum-exp: subtract the max before
        # exponentiating so beta * risk doesn't overflow for large beta.
        m = max(risks)
        aggregate_risk = m + (1.0 / beta) * math.log(
            sum(math.exp(beta * (r - m)) for r in risks)
        )

        gap_penalty = self._compute_gap_penalty(
            point, candidate_vel, visible, clearances
        )

        phi = aggregate_risk + self.config["lambda_gap"] * gap_penalty
        eps = self.config["barrier_eps"]
        phi_clipped = min(phi, 1.0 - eps)
        return phi_clipped / (1.0 - phi_clipped + eps)

    def _compute_gap_penalty(
        self,
        point: Vector2,
        candidate_vel: Vector2,
        humans: list[ObservableState],
        clearances: list[float],
    ) -> float:
        """Penalize corridors too narrow to pass, even when each flanking
        human looks individually safe.

        Finds the closest human ahead-and-left and ahead-and-right of the
        candidate's direction of travel (within ``search_radius``); if both
        exist, the passable gap is bounded by their *combined* clearance --
        two humans each individually a meter clear can still leave a gap
        only slightly wider than the robot. Returns 0.0 if no such
        straddling pair exists.
        """
        speed = candidate_vel.magnitude()
        if speed < 1e-8:
            return 0.0

        direction = candidate_vel * (1.0 / speed)
        perp = Vector2(-direction.y, direction.x)
        search_radius = self.config["search_radius"]

        left_best: tuple[float, float] | None = None
        right_best: tuple[float, float] | None = None

        for human, clearance in zip(humans, clearances):
            human_pos = Vector2(human.pose.px, human.pose.py)
            rel = human_pos - point
            parallel = rel.dot(direction)
            if parallel <= 0.0 or parallel > search_radius:
                continue  # Not ahead of the candidate, within range.

            lateral = rel.dot(perp)
            if lateral > 0.0 and (left_best is None or parallel < left_best[0]):
                left_best = (parallel, clearance)
            elif lateral < 0.0 and (right_best is None or parallel < right_best[0]):
                right_best = (parallel, clearance)

        if left_best is None or right_best is None:
            return 0.0

        combined_clearance = left_best[1] + right_best[1]
        return math.exp(-combined_clearance / self.config["gap_scale"])

    def _compute_efficiency(
        self, point: Vector2, reference_goal: Vector2 | None
    ) -> float:
        """Small, normalized preference for faster/more-progress candidates.

        Both terms are scaled into roughly ``[0, 1]`` before weighting, so
        ``w_travel_time``/``w_progress`` express genuine relative preference
        rather than an accidental unit conversion (raw seconds vs. raw
        meters added straight into a cost dominated by an exponential in
        the [0, 1] range would make the weights meaningless).
        """
        search_radius = self.config["search_radius"]
        tau_max = search_radius / max(self.agent.v_pref, 1e-8)
        normalized_travel_time = min(self.compute_travel_time(point) / tau_max, 1.0)
        efficiency = self.config["w_travel_time"] * normalized_travel_time

        if reference_goal is not None:
            d_max = 2.0 * max(self.arena_width, self.arena_height)
            normalized_progress = min(point.distance_to(reference_goal) / d_max, 1.0)
            efficiency += self.config["w_progress"] * normalized_progress

        return efficiency

    # -- pipeline orchestration ---------------------------------------------

    def compute_candidate_risks(
        self,
        human_states: dict[int, ObservableState],
        reference_goal: Vector2 | None = None,
    ) -> None:
        """Score every candidate in ``self.candidates``, in place."""
        for candidate in self.candidates:
            if not candidate.valid:
                continue
            valid, risk = self.compute_point_risk(
                candidate.position, human_states, reference_goal
            )
            candidate.valid = valid
            candidate.risk = risk

    def choose_safe_point(self) -> Vector2 | None:
        """Return the minimum-risk valid candidate.

        Applies hysteresis toward the previously chosen safe point: any
        candidate within ``hysteresis_radius`` of the last pick gets a small
        cost discount before the argmin, preventing flicker between
        near-tied candidates across consecutive ticks.

        Returns:
            The chosen candidate's position, or ``None`` if every candidate
            was hard-rejected this tick (a predicted collision was found
            everywhere in the sampled ring) -- callers should treat this as
            "no safe escape found this tick," not as "no humans nearby."
        """
        valid = [c for c in self.candidates if c.valid and math.isfinite(c.risk)]
        if not valid:
            return None

        bonus = self.config["hysteresis_bonus"]
        hysteresis_radius = self.config["hysteresis_radius"]
        prev = self._prev_safe_point

        def effective_risk(candidate: Candidate) -> float:
            risk = candidate.risk
            if (
                prev is not None
                and candidate.position.distance_to(prev) <= hysteresis_radius
            ):
                risk -= bonus
            return risk

        best = min(valid, key=effective_risk)
        self._prev_safe_point = best.position
        return best.position

    def plan_escape_point(
        self,
        human_states: dict[int, ObservableState],
        reference_goal: Vector2 | None = None,
    ) -> Vector2 | None:
        """Full pipeline: sample candidates, score them, pick the safest.

        Args:
            human_states: The robot's currently visible neighbors, as
                produced by ``agent.sensor.observe(...)``. Deliberately not
                ground-truth crowd state -- see module docstring.
            reference_goal: Optional soft pull toward this point (e.g. the
                mission's real target), used only to break near-ties
                between otherwise-safe candidates.

        Returns:
            The chosen escape point, or ``None`` if no candidate survived
            hard rejection this tick.
        """
        self.build_local_risk_map()
        self.compute_candidate_risks(human_states, reference_goal)
        return self.choose_safe_point()
