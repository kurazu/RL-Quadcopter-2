import random

from agents.base import BaseAgent


class RandomAgent(BaseAgent):

    def act(self):
        new_thrust = random.gauss(450., 25.)
        return [new_thrust + random.gauss(0., 1.) for x in range(4)]
