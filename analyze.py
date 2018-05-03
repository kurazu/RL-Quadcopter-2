from glob import glob
import os.path
import io
import pickle
import sys

import matplotlib.pyplot as plt


HERE = os.path.dirname(__file__)


def read_data(filename):
    with io.open(filename, 'rb') as f:
        return pickle.load(f)


def get_episode_filename(number):
    return os.path.join(HERE, 'episodes', f'episode-{number}.pickle')


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


def show_all():
    scores = get_scores()
    xs = sorted(scores)
    ys = [scores[idx] for idx in xs]
    plt.plot(xs, ys, label='reward')

    plt.legend()
    plt.ylim()

    plt.show()


def show_episode(episode_idx):
    episode = read_data(get_episode_filename(episode_idx))
    print(
        'Episode', episode['episode_number'],
        'reward', episode['total_reward']
    )
    experiences = episode['experiences']
    zs = [experience.next_state[0] for experience in experiences]
    vzs = [experience.next_state[1] for experience in experiences]
    azs = [experience.next_state[2] for experience in experiences]
    rewards = [experience.reward for experience in experiences]
    frames = list(range(len(experiences)))

    plt.plot(frames, zs, label='z')
    plt.plot(frames, vzs, label='vz')
    plt.plot(frames, azs, label='az')
    plt.plot(frames, rewards, label='rewards')

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
    else:
        idx = int(idx)
    show_episode(idx)


if __name__ == '__main__':
    main()
