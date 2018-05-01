import csv
import numpy as np

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa
import matplotlib.pyplot as plt
from matplotlib import cm

from task import Task


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
    file_output = 'data.txt'  # file name for saved results

    # Setup
    task = Task()
    agent = agent_class(task)
    done = False
    labels = [
        'time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
        'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
        'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3',
        'rotor_speed4'
    ]
    results = {x: [] for x in labels}
    reward_sum = 0
    # Run the simulation, and save the results.
    with open(file_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
        while True:
            rotor_speeds = agent.act()
            _, reward, done = task.step(rotor_speeds)
            reward_sum += reward
            to_write = (
                [task.sim.time] +
                list(task.sim.pose) +
                list(task.sim.v) +
                list(task.sim.angular_v) +
                list(rotor_speeds)
            )
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])
            writer.writerow(to_write)
            if done:
                break

    print('Finished in', task.sim.time, 'with reward', reward_sum)
    return results


def main():
    from agents.random import RandomAgent
    from agents.up import UpAgent

    agent_class = UpAgent
    results = fly(agent_class)
    draw(results, mode='time')


if __name__ == '__main__':
    main()
