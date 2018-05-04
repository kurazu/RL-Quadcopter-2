import math

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


def _inverse_exponential(x):
    return math.exp(-x)


def inverse_exponential(x):
    return 1.0 / (x + 1)


Z_AXIS = 2


class Task():
    """
    Task (environment) that defines the goal and provides feedback to the agent
    """

    ACTION_LOW = 404 - 25
    ACTION_HIGH = 404 + 50

    def __init__(self):
        # Start
        # 10 cm above ground
        init_pose = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])

        # Start still
        init_velocities = None  # Will become zeros
        init_angle_velocities = None  # Will become zeros

        # Goal
        # almost 10m above starting point
        self.target_pos = np.array([0.0, 0.0, 10.0])
        self.horizonal_target_pos = self.target_pos[:2]

        # time limit for each episode
        runtime = 5000.

        # Simulation
        self.sim = PhysicsSim(
            init_pose, init_velocities, init_angle_velocities, runtime
        )

        self.action_repeat = 3

        # 6 - pose
        # 3 - v
        # 3- angular_v
        # 3 - linear_accel
        # 3 - angular_accel
        self.state_size = self.action_repeat * (6 + 3 + 3 + 3 + 3)
        self.state_size = self.action_repeat * (1 + 1 + 1)
        self.action_low = self.ACTION_LOW
        self.action_high = self.ACTION_HIGH
        self.action_size = 1

    def get_reward(self):
        """Uses current pose of sim to return reward."""

        current_z = self.sim.pose[Z_AXIS]
        target_z = self.target_pos[Z_AXIS]
        z_speed = self.sim.v[Z_AXIS]
        z_accel = self.sim.linear_accel[Z_AXIS]

        distance_to_target = abs(current_z - target_z)
        reward = (
            0.1 +
            # Penalize each frame it takes us to get to the target
            # -1 +
            # Penalize straying from horizontal center
            # 0.2 * inverse_exponential(
            #     euclid_distance(current_position, self.horizonal_target_pos)
            # ) +
            # Penalize straying from target
            0.6 * inverse_exponential(distance_to_target) +
            0.2 * inverse_exponential(abs(z_speed)) +
            0.1 * inverse_exponential(abs(z_accel))
        )
        return reward

    def get_target_reached_reward(self):
        return +100

    def get_episode_finished_reward(self, time_exceeded):
        if time_exceeded:
            return 0
        else:  # Crashed or went off limits
            return -1

    # def is_target_reached(self):
    #     current_position = self.sim.pose[:3]
    #     current_velocity = self.sim.v
    #     return (
    #         # Within one cm from the target
    #         abs(euclid_distance(current_position, self.target_pos)) < 0.01 and
    #         # Almost zero velocity
    #         abs(vector_length(current_velocity)) < 0.01
    #     )

    def sim_to_state(self):
        return [
            self.sim.pose[Z_AXIS],
            self.sim.v[Z_AXIS],
            self.sim.linear_accel[Z_AXIS]
        ]
        return [
            *self.sim.pose,
            *self.sim.v,
            *self.sim.angular_v,
            *self.sim.linear_accel,
            *self.sim.angular_accels
        ]

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        rotor_speeds = rotor_speeds * 4
        reward = 0
        state_all = []
        for _ in range(self.action_repeat):
            # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds)
            reward += self.get_reward()
            state_all.append(self.sim_to_state())

        if done:
            time_exceeded = self.sim.time > self.sim.runtime
            reward += self.get_episode_finished_reward(time_exceeded)

        next_state = np.concatenate(state_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim_to_state()] * self.action_repeat)
        return state
