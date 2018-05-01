from agents.base import BaseAgent


class UpAgent(BaseAgent):

    def act(self, state):
        return [450.] * 4
