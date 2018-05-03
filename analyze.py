from glob import glob
import os.path
import io
import pickle

import matplotlib.pyplot as plt


HERE = os.path.dirname(__file__)


def read_data(filename):
    with io.open(filename, 'rb') as f:
        return pickle.load(f)


def main():
    path = os.path.join(HERE, 'episodes', 'episode-*.pickle')
    filenames = glob(path)
    raw_data = (read_data(filename) for filename in filenames)
    data = {
        d['episode_number']: d
        for d in raw_data
    }
    best_episode = max(data.values(), key=lambda d: d['total_reward'])
    print(
        'Best episode', best_episode['episode_number'],
        'reward', best_episode['total_reward']
    )
    best_experiences = best_episode['experiences']
    xs = [experience.next_state[0] for experience in best_experiences]
    ys = [experience.next_state[1] for experience in best_experiences]
    zs = [experience.next_state[2] for experience in best_experiences]
    rewards = [experience.reward for experience in best_experiences]
    frames = list(range(len(best_experiences)))

    plt.plot(frames, xs, label='x')
    plt.plot(frames, ys, label='y')
    plt.plot(frames, zs, label='z')
    plt.plot(frames, rewards, label='rewards')

    plt.legend()
    plt.ylim()

    plt.show()


if __name__ == '__main__':
    main()
