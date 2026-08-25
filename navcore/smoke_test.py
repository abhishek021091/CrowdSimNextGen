"""Manual smoke test for the first architectural walking-slice.

Not a real pytest suite yet -- just enough to prove the pieces actually
compose: GoalReachingMission, GroupGoalReachingMission,
AsymmetricVisibility + NeighborQuery, and StraightLinePolicy driving
plain Agent instances for a few ticks.
"""

from navcore.entities.agents.agent import Agent
from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.components.goal import Goal
from navcore.entities.components.pose import Pose
from navcore.entities.components.velocity import Velocity
from navcore.entities.groups.group import Group
from navcore.environment.neighbor_query import NeighborQuery
from navcore.missions.goal_reaching import GoalReachingMission
from navcore.missions.group_goal_reaching import GroupGoalReachingMission
from navcore.policies.straight_line import StraightLinePolicy
from navcore.visibility.visibility_policy import AsymmetricVisibility, FullVisibility

MINIMAL_CONFIG = {
    "kinematics": {"v_pref": 1.0},
    "physical": {"radius": 0.3},
}


def make_agent(agent_type: str, px: float, py: float, gx: float, gy: float) -> Agent:
    agent = Agent(MINIMAL_CONFIG, agent_type)
    agent.set_state(Pose(px, py, 0.0), Goal(gx, gy), v_pref=1.0, radius=0.3)
    agent.velocity = Velocity(0.0, 0.0)
    return agent


def test_goal_reaching_mission_moves_agent_toward_goal():
    agent = make_agent("Pedestrian", 0.0, 0.0, 5.0, 0.0)
    mission = GoalReachingMission()
    policy = StraightLinePolicy()

    target = mission.get_target(agent, neighbors=[])
    assert target == Vector2(5.0, 0.0)

    velocity = policy.compute_velocity(agent, target, neighbors=[])
    assert velocity.vx > 0.0
    assert velocity.vy == 0.0
    print("goal-reaching: target =", target, "velocity =", velocity)


def test_asymmetric_visibility_hides_robot_from_pedestrian():
    pedestrian = make_agent("Pedestrian", 0.0, 0.0, 5.0, 0.0)
    robot = make_agent("Robot", 1.0, 0.0, 0.0, 5.0)
    other_pedestrian = make_agent("Pedestrian", 2.0, 0.0, 0.0, 5.0)

    query = NeighborQuery(visibility_policy=AsymmetricVisibility())
    candidates = [pedestrian, robot, other_pedestrian]

    pedestrian_neighbors = query.neighbors_of(pedestrian, candidates, radius=10.0)
    assert len(pedestrian_neighbors) == 1  # sees the other pedestrian, not the robot

    robot_neighbors = query.neighbors_of(robot, candidates, radius=10.0)
    assert len(robot_neighbors) == 2  # robot sees both pedestrians
    print(
        "asymmetric visibility: pedestrian sees",
        len(pedestrian_neighbors),
        "| robot sees",
        len(robot_neighbors),
    )


def test_full_visibility_toggle():
    pedestrian = make_agent("Pedestrian", 0.0, 0.0, 5.0, 0.0)
    robot = make_agent("Robot", 1.0, 0.0, 0.0, 5.0)

    query = NeighborQuery(visibility_policy=FullVisibility())
    pedestrian_neighbors = query.neighbors_of(
        pedestrian, [pedestrian, robot], radius=10.0
    )
    assert len(pedestrian_neighbors) == 1  # now sees the robot too
    print("full visibility: pedestrian sees", len(pedestrian_neighbors))


def test_group_goal_reaching_leader_and_follower():
    leader = make_agent("Pedestrian", 0.0, 0.0, 0.0, 0.0)
    follower = make_agent("Pedestrian", -1.0, 0.0, 0.0, 0.0)

    registry = {"leader": leader, "follower": follower}
    group = Group(
        id="family_1",
        member_ids=("leader", "follower"),
        goal=Vector2(10.0, 0.0),
        leader_id="leader",
    )

    leader_mission = GroupGoalReachingMission(
        agent_id="leader", group=group, agent_lookup=registry.__getitem__
    )
    follower_mission = GroupGoalReachingMission(
        agent_id="follower",
        group=group,
        agent_lookup=registry.__getitem__,
        formation_offset=Vector2(-1.0, 0.0),
    )

    leader_target = leader_mission.get_target(leader, neighbors=[])
    assert leader_target == Vector2(10.0, 0.0)

    follower_target = follower_mission.get_target(follower, neighbors=[])
    assert follower_target == Vector2(-1.0, 0.0)  # leader position (0,0) + offset

    print(
        "group mission: leader target =",
        leader_target,
        "| follower target =",
        follower_target,
    )


if __name__ == "__main__":
    test_goal_reaching_mission_moves_agent_toward_goal()
    test_asymmetric_visibility_hides_robot_from_pedestrian()
    test_full_visibility_toggle()
    test_group_goal_reaching_leader_and_follower()
    print("\nAll smoke tests passed.")
