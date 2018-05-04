import math

import numpy as np

from physics_sim import PhysicsSim

"""
HOVER TASK

The quadcopter starts still at 10m above ground.
It's aim is to stay in the air for as long as possible,
aiming to be at 10m elevation.
"""


def inverse_exponential(x):
    """
    Yields the highest value (1) when x == 0.

    Works for positive numbers only.
    """
    return math.exp(-x)


def shifted_reciprocal(x):
    """
    Yields the highest value (1) when x == 0.

    Works for positive numbers only.
    """
    return 1.0 / (x + 1)


Z_AXIS = 2


class Task():
    """
    Task (environment) that defines the goal and provides feedback to the agent
    """

    # Reduced range of actions to make it easier to control the vehicle
    ACTION_LOW = 404 - 25
    ACTION_HIGH = 404 + 50
    # The copter should hover around 10m above ground.
    TARGET_HEIGHT = 10.0

    def __init__(self):
        # Start 10m above ground
        init_pose = np.array([0.0, 0.0, self.TARGET_HEIGHT, 0.0, 0.0, 0.0])

        # Start still
        init_velocities = None  # Will become zeros
        init_angle_velocities = None  # Will become zeros

        # time limit for each episode
        # Allow long episodes that promote keeping the vehicle long in the air
        runtime = 5000.

        # Simulation
        self.sim = PhysicsSim(
            init_pose, init_velocities, init_angle_velocities, runtime
        )

        self.action_repeat = 3

        # To make that task easier we will steer all rotors with one value.
        self.action_size = 1

        # Since the rotors have joined speeds the aircraft can only move in the
        # z axis (go higher or lower) and will not rotate.
        # That means that the only state we need to feed to the algorithm
        # is:
        # * z position
        # * velocity along z axis
        # * acceleration along z axis
        self.state_size = self.action_repeat * (1 + 1 + 1)
        self.action_low = self.ACTION_LOW
        self.action_high = self.ACTION_HIGH

    def get_reward(self):
        """Reward the aircraft"""

        current_z = self.sim.pose[Z_AXIS]
        z_speed = self.sim.v[Z_AXIS]
        z_accel = self.sim.linear_accel[Z_AXIS]

        distance_to_target = abs(current_z - self.TARGET_HEIGHT)
        reward = (
            # Promote keeping in the air by giving a small reward
            # for not crashing
            0.1 +
            # Promote staying at 10m height
            # (the highest reward will be given when the aircraft
            # is at exactly 10m).
            0.6 * shifted_reciprocal(distance_to_target) +
            # Promote staying still
            # To reduce the aircrafts tendency to go outside bounds and promote
            # keeping minimal height variations we reward it for keeping
            # zero z-axis velocity.
            0.2 * shifted_reciprocal(abs(z_speed)) +
            # Promote avoiding high acceleration
            # Rapid acceleration leads to instability of flight, we reward
            # the aircraft for avoiding it.
            0.1 * shifted_reciprocal(abs(z_accel))
        )
        return reward

    def get_episode_finished_reward(self, time_exceeded):
        """Special rewards when the simulation is terminated."""
        if time_exceeded:
            # If we determine the simulation finished because the time limit
            # was exceeded, it means the aircraft managed to stay in the air
            # the whole time and we should not punish it for it.
            return 0
        else:
            # Otherwise the aircraft crashed into the floor or went outside
            # the limited space. We punish it for it.
            return -1

    def sim_to_state(self):
        """Construct state to be used as experience from simulator state."""
        # As mentioned earlier, we only need z-axis variables.
        # This will keep our state nice and small.
        return [
            self.sim.pose[Z_AXIS],
            self.sim.v[Z_AXIS],
            self.sim.linear_accel[Z_AXIS]
        ]

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        # We use only one action to control all 4 rotors.
        # Here we copy it 4 times to conform with simulator input requirements.
        rotor_speeds = rotor_speeds * 4
        reward = 0
        state_all = []
        for _ in range(self.action_repeat):
            # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds)
            reward += self.get_reward()
            state_all.append(self.sim_to_state())

        if done:
            # Determine the cause of simulation termination
            time_exceeded = self.sim.time > self.sim.runtime
            # Add additional reward
            reward += self.get_episode_finished_reward(time_exceeded)

        next_state = np.concatenate(state_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim_to_state()] * self.action_repeat)
        return state
