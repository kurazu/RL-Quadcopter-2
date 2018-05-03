from collections import defaultdict

import numpy as np

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa
import matplotlib.pyplot as plt
from matplotlib import cm

from task import Task
from gym_task import GymTask


def draw(results, mode='velocity'):
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    combined_velocity = np.array(list(zip(
        results['x_velocity'], results['y_velocity'], results['z_velocity']
    )))
    velocity_value = np.sum(combined_velocity ** 2, axis=1)
    min = np.min(velocity_value)
    max = np.max(velocity_value)
    range = max - min
    velocity_factor = (velocity_value - min) / range
    time = np.linspace(0, 1, len(velocity_factor))

    color_dimension = {
        'velocity': velocity_factor,
        'time': time
    }[mode]

    ax.scatter(
        results['x'], results['y'], results['z'],
        c=cm.cool(color_dimension),
        label=f'Quadcopter position/{mode}'
    )
    ax.legend()

    plt.show()


def fly(agent_class):
    task = Task()
    task = GymTask('Pendulum-v0')
    agent = agent_class(task)
    rewards = []
    num_episodes = 10000
    draw_every = 500
    mean_every = 10
    for episode_number in range(1, num_episodes + 1):
        state = agent.reset_episode()
        episode_rewards = 0
        results = defaultdict(list)
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = task.step(action)
            agent.step(action, reward, next_state, done)
            state = next_state
            episode_rewards += reward
            # results['x'].append(task.sim.pose[0])
            # results['y'].append(task.sim.pose[1])
            # results['z'].append(task.sim.pose[2])
            # results['x_velocity'].append(task.sim.v[0])
            # results['y_velocity'].append(task.sim.v[1])
            # results['z_velocity'].append(task.sim.v[2])
            if done:
                break
        True or print(
            'Episode', episode_number,
            'finished in', task.sim.time,
            'with reward', episode_rewards
        )
        if episode_number % mean_every == 0:
            avg_mean = np.mean(rewards[-draw_every:])
            print('AVG Reward', avg_mean, 'after', episode_number, 'episodes')
        if episode_number % draw_every == 0:
            draw(results, mode='time')
        rewards.append(episode_rewards)
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
