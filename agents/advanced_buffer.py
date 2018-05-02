import numpy as np

from agents.buffer import ReplayBuffer


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exp = np.exp(x)
    return exp / np.sum(exp, axis=0)


class AdvancedReplayBuffer(ReplayBuffer):

    def sample(self, batch_size=64):
        epsilon = 1e-6
        rewards = np.array([e.reward for e in self.memory]) + epsilon
        p = softmax(rewards)
        indices = np.random.choice(len(self.memory), batch_size, p=p)
        return [self.memory[idx] for idx in indices]
