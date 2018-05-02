import math

from agents.base import BaseAgent


class UpAgent(BaseAgent):

    def act(self, state):
        current_z = state[2]
        target_z = self.task.target_pos[2]
        power = (math.cos(2 * math.pi * current_z / (2 * target_z)) + 1) / 2 * 900
        print('POWER', power)
        return [power] * 4
