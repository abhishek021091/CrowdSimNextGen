"""Lightweight, renderer-agnostic snapshots of simulation state.

Keeping a plain-data snapshot (rather than reading ``env`` directly from
every renderer, every sub-frame) is what makes interpolation and the
pedestrian inspector cheap: we snapshot once per simulation step and
interpolate/query against that, instead of re-touching the live env
objects on every draw call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AgentSnapshot:
    id: int
    x: float
    y: float
    radius: float
    heading: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    pref_vx: float | None = None
    pref_vy: float | None = None
    goal_x: float | None = None
    goal_y: float | None = None
    pref_speed: float | None = None
    group_id: int | None = None
    is_leader: bool = False
    name: str | None = None
    state: str | None = None


@dataclass(slots=True)
class SceneSnapshot:
    """Everything a renderer needs for one simulation step."""

    step: int
    sim_time: float
    robot: AgentSnapshot | None
    pedestrians: dict[int, AgentSnapshot] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    """Optional payloads: sensor data, planning data, ORCA data, metrics.
    Kept as a free-form dict so new env capabilities don't require
    changing this dataclass's shape."""


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def build_agent_snapshot(agent_id: int, agent: Any) -> AgentSnapshot | None:
    pose = _get(agent, "pose")
    if pose is None:
        return None
    velocity = _get(agent, "velocity")
    pref_velocity = _get(agent, "pref_velocity") or _get(agent, "preferred_velocity")
    goal = _get(agent, "goal")
    return AgentSnapshot(
        id=agent_id,
        x=float(pose.px),
        y=float(pose.py),
        radius=float(_get(agent, "radius", 0.3)),
        heading=float(_get(pose, "theta", _get(agent, "theta", 0.0)) or 0.0),
        vx=float(_get(velocity, "vx", 0.0) or 0.0) if velocity is not None else 0.0,
        vy=float(_get(velocity, "vy", 0.0) or 0.0) if velocity is not None else 0.0,
        pref_vx=(float(_get(pref_velocity, "vx", 0.0)) if pref_velocity is not None else None),
        pref_vy=(float(_get(pref_velocity, "vy", 0.0)) if pref_velocity is not None else None),
        goal_x=float(_get(goal, "gx")) if goal is not None else None,
        goal_y=float(_get(goal, "gy")) if goal is not None else None,
        pref_speed=_get(agent, "pref_speed") or _get(agent, "preferred_speed"),
        state=_get(agent, "state"),
    )


def build_scene_snapshot(env: Any, step: int, sim_time: float) -> SceneSnapshot:
    robot_snap = build_agent_snapshot(-1, env.robot) if getattr(env, "robot", None) else None

    peds: dict[int, AgentSnapshot] = {}
    group_map, leader_map = _group_lookup(env)
    for ped_id, ped in (getattr(env, "crowd", {}) or {}).items():
        snap = build_agent_snapshot(ped_id, ped)
        if snap is None:
            continue
        snap.group_id = group_map.get(ped_id)
        snap.is_leader = leader_map.get(ped_id, False)
        peds[ped_id] = snap

    extra: dict[str, Any] = {}
    for key in ("sensor_observation", "sensor_data", "lidar", "visibility_polygon"):
        val = getattr(getattr(env, "robot", None), key, None)
        if val is not None:
            extra["sensor"] = val
            break
    for key in ("planned_path", "path", "waypoints"):
        val = getattr(getattr(env, "robot", None), key, None)
        if val is not None:
            extra["path"] = val
            break
    orca_data = getattr(getattr(env, "robot", None), "orca_debug", None) or getattr(
        getattr(env, "robot", None), "orca_data", None
    )
    if orca_data is not None:
        extra["orca"] = orca_data
    predictions = getattr(env, "predicted_trajectories", None)
    if predictions is not None:
        extra["predictions"] = predictions

    return SceneSnapshot(step=step, sim_time=sim_time, robot=robot_snap, pedestrians=peds, extra=extra)


def _group_lookup(env: Any) -> tuple[dict[int, int], dict[int, bool]]:
    groups_obj = getattr(env, "groups", None)
    group_map: dict[int, int] = {}
    leader_map: dict[int, bool] = {}
    if not groups_obj:
        return group_map, leader_map
    items = groups_obj.items() if isinstance(groups_obj, dict) else enumerate(groups_obj)
    for group_id, group in items:
        gid = int(getattr(group, "id", group_id))
        leader_id = getattr(group, "leader_id", None)
        for member_id in getattr(group, "member_ids", ()):
            member_id = int(member_id)
            group_map[member_id] = gid
            leader_map[member_id] = leader_id is not None and member_id == int(leader_id)
    return group_map, leader_map


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def interpolate_agent(prev: AgentSnapshot, curr: AgentSnapshot, t: float) -> AgentSnapshot:
    """Blend two snapshots of the *same* agent for smooth sub-stepping."""
    if prev.id != curr.id:
        return curr
    return AgentSnapshot(
        id=curr.id,
        x=lerp(prev.x, curr.x, t),
        y=lerp(prev.y, curr.y, t),
        radius=lerp(prev.radius, curr.radius, t),
        heading=_lerp_angle(prev.heading, curr.heading, t),
        vx=lerp(prev.vx, curr.vx, t),
        vy=lerp(prev.vy, curr.vy, t),
        pref_vx=curr.pref_vx,
        pref_vy=curr.pref_vy,
        goal_x=curr.goal_x,
        goal_y=curr.goal_y,
        pref_speed=curr.pref_speed,
        group_id=curr.group_id,
        is_leader=curr.is_leader,
        name=curr.name,
        state=curr.state,
    )


def _lerp_angle(a: float, b: float, t: float) -> float:
    import math

    diff = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + diff * t


def interpolate_scene(prev: SceneSnapshot, curr: SceneSnapshot, t: float) -> SceneSnapshot:
    robot = None
    if curr.robot is not None:
        robot = interpolate_agent(prev.robot, curr.robot, t) if prev.robot is not None else curr.robot

    peds: dict[int, AgentSnapshot] = {}
    for pid, snap in curr.pedestrians.items():
        prev_snap = prev.pedestrians.get(pid)
        peds[pid] = interpolate_agent(prev_snap, snap, t) if prev_snap is not None else snap

    sim_time = lerp(prev.sim_time, curr.sim_time, t)
    return SceneSnapshot(step=curr.step, sim_time=sim_time, robot=robot, pedestrians=peds, extra=curr.extra)
