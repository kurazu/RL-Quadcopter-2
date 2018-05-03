import numpy as np

import gym


class GymTask():

    def __init__(self, task_name='MountainCarContinuous-v0'):
        self.env = gym.make(task_name)

        self.action_repeat = 3

        self.state_size = self.action_repeat * (
            self.env.observation_space.shape[0]
        )
        self.action_low = self.env.action_space.low[0]
        self.action_high = self.env.action_space.high[0]
        self.action_size = self.env.action_space.shape[0]

    def reset(self):
        """Reset the sim to start a new episode."""
        state = self.env.reset()
        state = np.concatenate([state] * self.action_repeat)
        return state

    def step(self, action):
        reward = 0
        state_all = []
        for _ in range(self.action_repeat):
            # update the sim pose and velocities
            next_state, step_reward, done, _ = self.env.step(action)
            state_all.append(next_state)
            reward += step_reward

        next_state = np.concatenate(state_all)
        return next_state, reward, done
