import io
import pickle
import os.path

import numpy as np

from tqdm import tqdm

# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D  # noqa
# import matplotlib.pyplot as plt
# from matplotlib import cm

from task import Task
# from gym_task import GymTask


# def draw(states, mode='velocity'):
#     mpl.rcParams['legend.fontsize'] = 10

#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     time = np.linspace(0, 1, len(states))

#     from task import euclid_distance, inverse_exponential
#     for x, y, z, *_ in states:
#         print(
#             'z', z, 'exp',
#             inverse_exponential(
#                 euclid_distance(
#                     np.array([x, y, z]),
#                     np.array([0.0, 0.0, 10.0])
#                 )
#             )
#         )

#     x = [state[0] for state in states]
#     y = [state[1] for state in states]
#     z = [state[2] for state in states]
#     ax.scatter(
#         x, y, z,
#         c=cm.cool(time),
#         label=f'Quadcopter position/{mode}'
#     )
#     ax.legend()

#     plt.show()


def run_episode(episode_number, task, agent):
    experiences = []
    state = agent.reset_episode()
    episode_rewards = 0
    # results = defaultdict(list)
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        experience = agent.step(action, reward, next_state, done)
        experiences.append(experience)
        state = next_state
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
    save_episode_dump(episode_number, episode_rewards, experiences)
    # save_agent(episode_number, agent)
    return episode_rewards


HERE = os.path.dirname(__file__)


def save_episode_dump(episode_number, episode_rewards, experiences):
    dump = {
        'episode_number': episode_number,
        'total_reward': episode_rewards,
        'experiences': experiences
    }
    filename = os.path.join(
        HERE, 'episodes', 'episode-{episode_number}.pickle'.format(
            episode_number=episode_number
        )
    )
    with io.open(filename, 'wb') as f:
        pickle.dump(dump, f)


def save_agent(episode_number, agent):
    for model_name in [
        'critic_target', 'critic_local', 'actor_target', 'actor_local'
    ]:
        weights = getattr(agent, model_name).model.get_weights()
        filename = os.path.join(
            HERE, 'episodes', '{model_name}-{episode_number}.pickle'.format(
                model_name=model_name, episode_number=episode_number
            )
        )
        with io.open(filename, 'wb') as f:
            pickle.dump(weights, f)

    # filename = os.path.join(
    #     HERE, 'episodes', f'memory-{episode_number}.pickle'
    # )
    # with io.open(filename, 'wb') as f:
    #     pickle.dump(agent.memory, f)

    filename = os.path.join(
        HERE, 'episodes', 'noise-{episode_number}.pickle'.format(
            episode_number=episode_number
        )
    )
    with io.open(filename, 'wb') as f:
        pickle.dump(agent.noise, f)


def fly(agent_class):
    quad_task = Task()
    # pendulum_task = GymTask('Pendulum-v0')
    # mountain_car_task = GymTask()
    task = quad_task
    agent = agent_class(task)
    rewards = []
    num_episodes = 2000
    mean_every = 10
    episode_number = 1
    while episode_number <= num_episodes:
        batch_rewards = []
        for _ in tqdm(range(mean_every)):
            reward = run_episode(episode_number, task, agent)
            episode_number += 1
            batch_rewards.append(reward)
        average_batch_reward = np.mean(batch_rewards)
        print(
            'AVG Reward', average_batch_reward,
            'after', episode_number - 1, 'episodes /', num_episodes
        )
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
