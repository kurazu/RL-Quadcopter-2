from agents.base import BaseAgent


class DeadAgent(BaseAgent):

    def act(self):
        return [0.] * 4
