from glob import glob
import os.path
import io
import pickle
import sys

import numpy as np

import matplotlib.pyplot as plt

from task import Task


HERE = os.path.dirname(__file__)
FOLDER = os.environ.get('FOLDER', 'episodes')


def read_data(filename):
    with io.open(filename, 'rb') as f:
        return pickle.load(f)


def get_episode_filename(number):
    return os.path.join(HERE, FOLDER, 'episode-{number}.pickle'.format(
        number=number
    ))


def get_scores():
    path = get_episode_filename('*')
    filenames = glob(path)
    raw_data = (read_data(filename) for filename in filenames)
    scores = {d['episode_number']: d['total_reward'] for d in raw_data}
    return scores


def find_best_idx():
    scores = get_scores()
    best_episode_idx = max(scores, key=lambda idx: scores[idx])
    return best_episode_idx


def find_last_idx():
    scores = get_scores()
    return max(scores)


def show_all():
    scores = get_scores()
    xs = sorted(scores)
    ys = [scores[idx] for idx in xs]
    mean_window = 25
    means = [np.mean(ys[idx - mean_window:idx]) for idx in xs]
    plt.semilogy(xs, ys, label='reward')
    plt.semilogy(xs, means, label='reward mean (over last {})'.format(mean_window))

    plt.legend()
    plt.ylim()

    plt.show()


def normalize(min_, max_, x):
    range_ = max_ - min_
    x = np.array(x)
    return (x - min_) / range_


def show_episode(episode_idx):
    episode = read_data(get_episode_filename(episode_idx))
    print(
        'Episode', episode['episode_number'],
        'reward', episode['total_reward']
    )
    experiences = episode['experiences']
    targets = [10 for _ in experiences]
    zs = [experience.next_state[0] for experience in experiences]
    vzs = [experience.next_state[1] for experience in experiences]
    # azs = [experience.next_state[2] for experience in experiences]
    actions = normalize(
        Task.ACTION_LOW, Task.ACTION_HIGH,
        [experience.action for experience in experiences]
    ) * 10
    rewards = [experience.reward for experience in experiences]
    frames = list(range(len(experiences)))

    plt.plot(frames, targets, label='target z')
    plt.plot(frames, zs, label='z')
    plt.plot(frames, vzs, label='vz')
    # plt.plot(frames, azs, label='az')
    plt.plot(frames, rewards, label='rewards')
    plt.plot(frames, actions, label='action')

    plt.legend()
    plt.ylim()

    plt.show()


def main():
    idx = sys.argv[1]
    if idx == 'all':
        show_all()
        return

    if idx == 'best':
        idx = find_best_idx()
    elif idx == 'last':
        idx = find_last_idx()
    else:
        idx = int(idx)
    show_episode(idx)


if __name__ == '__main__':
    main()
