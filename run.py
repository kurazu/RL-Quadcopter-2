import numpy as np

from tqdm import tqdm

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa
import matplotlib.pyplot as plt
from matplotlib import cm

from task import Task
from gym_task import GymTask


def draw(states, mode='velocity'):
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    time = np.linspace(0, 1, len(states))

    from task import euclid_distance, inverse_exponential
    for x, y, z, *_ in states:
        print(
            'z', z, 'exp',
            inverse_exponential(
                euclid_distance(
                    np.array([x, y, z]),
                    np.array([0.0, 0.0, 10.0])
                )
            )
        )

    x = [state[0] for state in states]
    y = [state[1] for state in states]
    z = [state[2] for state in states]
    ax.scatter(
        x, y, z,
        c=cm.cool(time),
        label=f'Quadcopter position/{mode}'
    )
    ax.legend()

    plt.show()


def run_episode(episode_number, task, agent):
    states = []
    state = agent.reset_episode()
    states.append(state)
    episode_rewards = 0
    # results = defaultdict(list)
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        states.append(state)
        episode_rewards += reward
        # results['x'].append(task.sim.pose[0])
        # results['y'].append(task.sim.pose[1])
        # results['z'].append(task.sim.pose[2])
        # results['x_velocity'].append(task.sim.v[0])
        # results['y_velocity'].append(task.sim.v[1])
        # results['z_velocity'].append(task.sim.v[2])
    # print(
    #     'Episode', episode_number,
    #     'finished in', task.sim.time,
    #     'with reward', episode_rewards
    # )
    return episode_rewards, states


def fly(agent_class):
    quad_task = Task()
    pendulum_task = GymTask('Pendulum-v0')
    mountain_car_task = GymTask()
    task = quad_task
    agent = agent_class(task)
    rewards = []
    num_episodes = 2000
    mean_every = 20
    draw_every_n_batches = 100
    episode_number = 1
    while episode_number <= num_episodes:
        batch_rewards = []
        for _ in tqdm(range(mean_every)):
            reward, states = run_episode(episode_number, task, agent)
            episode_number += 1
            batch_rewards.append(reward)
        average_batch_reward = np.mean(batch_rewards)
        print(
            'AVG Reward', average_batch_reward,
            'after', episode_number - 1, 'episodes /', num_episodes
        )
        if (episode_number - 1) % (mean_every * draw_every_n_batches) == 0:
            draw(states, mode='time')
    return rewards


def main():
    from agents.agent import DDPG
    # from agents.random import RandomAgent
    from agents.up import UpAgent
    from agents.policy_search import PolicySearchAgent

    agent_class = DDPG
    rewards = fly(agent_class)
    print('Rewards', rewards)


if __name__ == '__main__':
    main()
