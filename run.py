import io
import pickle
import os.path

import numpy as np

from tqdm import tqdm

from task import Task
from agents.agent import DDPG


HERE = os.path.dirname(__file__)


def run_episode(episode_number, task, agent):
    experiences = []
    state = agent.reset_episode()
    episode_rewards = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        experience = agent.step(action, reward, next_state, done)
        experiences.append(experience)
        state = next_state
        episode_rewards += reward

    save_episode_dump(episode_number, episode_rewards, experiences)
    return episode_rewards


def save_episode_dump(episode_number, episode_rewards, experiences):
    dump = {
        'episode_number': episode_number,
        'total_reward': episode_rewards,
        'experiences': experiences
    }
    filename = os.path.join(
        HERE, 'episodes', f'episode-{episode_number}.pickle'
    )
    with io.open(filename, 'wb') as f:
        pickle.dump(dump, f)


def fly(num_episodes=2000, mean_every=10):
    task = Task()
    agent = DDPG(task)
    rewards = []
    episode_number = 1
    while episode_number <= num_episodes:
        batch_rewards = []
        for _ in tqdm(range(mean_every)):
            reward = run_episode(episode_number, task, agent)
            rewards.append(reward)
            episode_number += 1
            batch_rewards.append(reward)
        average_batch_reward = np.mean(batch_rewards)
        print(
            'AVG Reward', average_batch_reward,
            'after', episode_number - 1, 'episodes /', num_episodes
        )
    return rewards


def main():
    rewards = fly()
    print('Rewards', rewards)


if __name__ == '__main__':
    main()
