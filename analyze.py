from glob import glob
import os.path
import io
import pickle
import sys

import numpy as np

import matplotlib.pyplot as plt


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


def plot_all():
    scores = get_scores()
    xs = sorted(scores)
    ys = [scores[idx] for idx in xs]
    mean_window = 25
    means = [np.mean(ys[max(idx - mean_window, 0):idx]) for idx in xs]
    plt.semilogy(xs, ys, label='reward')
    plt.semilogy(
        xs, means, label='reward mean (over last {})'.format(mean_window)
    )

    plt.legend()
    plt.ylim()

    plt.show()


def show_episode_rewards(episode_idx):
    episode = read_data(get_episode_filename(episode_idx))
    print(
        'Episode', episode['episode_number'],
        'reward', episode['total_reward']
    )
    experiences = episode['experiences']
    targets = [10 for _ in experiences]
    ground = [0 for _ in experiences]
    zs = [experience.next_state[0] for experience in experiences]
    rewards = [experience.reward for experience in experiences]
    frames = list(range(len(experiences)))

    fig, ax1 = plt.subplots()
    ax1.plot(frames, zs, color='b', label='z')
    ax1.plot(frames, targets, color='g', label='target z')
    ax1.plot(frames, ground, color='r', label='ground z')
    # ax1.plot(frames, vzs, label='vz')
    ax1.set_xlabel('frame')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('z', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()

    # ax2.plot(frames, actions, label='action')
    ax2.plot(frames, rewards, color='r', label='rewards', alpha=0.2)
    ax2.set_ylabel('rewards', color='r')
    ax2.tick_params('y', colors='r')

    ax1.legend()
    fig.tight_layout()
    plt.ylim()
    plt.show()


def main():
    idx = sys.argv[1]
    if idx == 'all':
        plot_all()
        return

    if idx == 'best':
        idx = find_best_idx()
    elif idx == 'last':
        idx = find_last_idx()
    else:
        idx = int(idx)
    show_episode_rewards(idx)


if __name__ == '__main__':
    main()
