import numpy as np
from physics_sim import PhysicsSim

"""
TAKE OFF

The copter starts slightly above ground (0.1m) and is supposed to fly up
to 10 meters and stop there.
To assure nice take-off (straight line up) we are going to penalize straying
from the horizontal center.
To assure that the copter is not taking too long, we are going to penalize
each frame before arriving at target location.
To show the copter the way we are going to terminate the episode only when
the target location is reached.
To make the copter hover at target location we are going to terminate the
episode only when the copter is at target and has zero velocity.
"""


def euclid_distance(point, target=None):
    relevant_point = point[:len(target)]
    return vector_length(relevant_point - target)


def vector_length(point):
    return np.sqrt(np.sum(point ** 2))


class Task():
    """
    Task (environment) that defines the goal and provides feedback to the agent
    """

    def __init__(self):
        # Start
        # 10 cm above ground
        init_pose = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])

        # Start still
        init_velocities = None  # Will become zeros
        init_angle_velocities = None  # Will become zeros

        # Goal
        # almost 10m above starting point
        self.target_pos = np.array([0.0, 0.0, 10.0])
        self.horizonal_target_pos = self.target_pos[:2]

        # time limit for each episode
        runtime = 50.

        # Simulation
        self.sim = PhysicsSim(
            init_pose, init_velocities, init_angle_velocities, runtime
        )

        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

    def get_crashed_reward(self):
        return -1000

    def get_reward(self):
        """Uses current pose of sim to return reward."""

        current_position = self.sim.pose[:3]

        reward = (
            # Penalize each frame it takes us to get to the target
            -1 +
            # Penalize straying from horizontal center
            -euclid_distance(current_position, self.horizonal_target_pos)
        )
        return reward

    def get_target_reached_reward(self):
        return +100

    def is_target_reached(self):
        current_position = self.sim.pose[:3]
        current_velocity = self.sim.v
        return (
            # Within one cm from the target
            abs(euclid_distance(current_position, self.target_pos)) < 0.01 and
            # Almost zero velocity
            abs(vector_length(current_velocity)) < 0.01
        )

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds)
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        if done:
            # Penalize crashing, going off-limits or taking too long
            reward += -1000
        elif self.is_target_reached():
            # Reward reaching the target
            reward += 1000
            done = True

        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
