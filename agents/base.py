class BaseAgent:
    def __init__(self, task):
        self.task = task
        self.min_z = 1000

    def act(self, state):
        return [0.] * 4

    def reset_episode(self):
        state = self.task.reset()
        return state

    def step(self, action, reward, next_state, done):
        current_z = next_state[2]
        print('z', next_state[2], 'reward', reward)
        if current_z < self.min_z:
            print('LOW', self.min_z, '=>', current_z)
            self.min_z = current_z
