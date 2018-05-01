class BaseAgent:
    def __init__(self, task):
        self.task = task

    def act(self, state):
        return [0.] * 4

    def reset_episode(self):
        state = self.task.reset()
        return state

    def step(self, reward, done):
        pass
