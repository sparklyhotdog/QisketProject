from simulations_with_practical_issues.Rotation import Rotation
import numpy as np
import math
import matplotlib.pyplot as plt


def plot_visibility_darkcounts(start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = [[], [], [], []]

    # delta = phase difference
    delta = .25
    entangled_state = [1 / math.sqrt(2), 0, 0, np.exp(1j * delta) / math.sqrt(2)]

    for dc in x_val:
        sim = Rotation('config.yaml', entangled_state, 200)
        sim.set_dc_rate(dc)
        visibility = sim.run()
        for i in range(4):
            y_val[i].append(visibility[i])

    for i in range(4):
        plt.plot(x_val, y_val[i], label=states[i])
    plt.legend()
    plt.xlabel('Dark Count Rate (counts/second)')
    plt.ylabel('Visibility')
    plt.savefig('plots\\visibility_vs_darkcounts', dpi=1000)
    plt.show()


def plot_visibility_phasediff(start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = [[], [], [], []]

    for delta in x_val:
        entangled_state = [1 / math.sqrt(2), 0, 0, np.exp(1j * delta) / math.sqrt(2)]
        sim = Rotation('config.yaml', entangled_state, 200)
        visibility = sim.run()
        for i in range(4):
            y_val[i].append(visibility[i])

    for i in range(4):
        plt.plot(x_val, y_val[i], label=states[i])
    plt.legend()
    plt.xlabel('Phase Difference in the Entangled State')
    plt.ylabel('Visibility')
    plt.savefig('plots\\visibility_vs_phasediff', dpi=1000)
    plt.show()


def plot_visibility_loss(start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = [[], [], [], []]

    # delta = phase difference
    delta = .25
    entangled_state = [1 / math.sqrt(2), 0, 0, np.exp(1j * delta) / math.sqrt(2)]

    for loss in x_val:
        sim = Rotation('config.yaml', entangled_state, 200)
        sim.set_loss_signal(loss)
        sim.set_loss_idler(loss)
        visibility = sim.run()
        for i in range(4):
            y_val[i].append(visibility[i])

    for i in range(4):
        plt.plot(x_val, y_val[i], label=states[i])
    plt.legend()
    plt.xlabel('Optical Loss')
    plt.ylabel('Visibility')
    plt.savefig('plots\\visibility_vs_loss', dpi=1000)
    plt.show()


def plot_visibility_jitter(start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = [[], [], [], []]

    # delta = phase difference
    delta = .25
    entangled_state = [1 / math.sqrt(2), 0, 0, np.exp(1j * delta) / math.sqrt(2)]

    for jitter in x_val:
        sim = Rotation('config.yaml', entangled_state, 200)
        sim.set_jitter(jitter)
        visibility = sim.run()
        for i in range(4):
            y_val[i].append(visibility[i])

    for i in range(4):
        plt.plot(x_val, y_val[i], label=states[i])
    plt.legend()
    plt.xlabel('Jitter FWHM (ps)')
    plt.ylabel('Visibility')
    plt.savefig('plots\\visibility_vs_jitter', dpi=1000)
    plt.show()


def plot_visibility_deadtime(start, stop, num_points):
    x_val = np.linspace(start, stop, num_points)
    y_val = [[], [], [], []]

    # delta = phase difference
    delta = .25
    entangled_state = [1 / math.sqrt(2), 0, 0, np.exp(1j * delta) / math.sqrt(2)]

    for deadtime in x_val:
        sim = Rotation('config.yaml', entangled_state, 200)
        sim.set_deadtime(deadtime)
        visibility = sim.run()
        for i in range(4):
            y_val[i].append(visibility[i])

    for i in range(4):
        plt.plot(x_val, y_val[i], label=states[i])
    plt.legend()
    plt.xlabel('Deadtime (ps)')
    plt.ylabel('Visibility')
    plt.savefig('plots\\visibility_vs_deadtime', dpi=1000)
    plt.show()


if __name__ == '__main__':
    # plot_visibility_darkcounts(0, 100000, 4)
    # plot_visibility_phasediff(0, 2*math.pi, 8)
    # plot_visibility_loss(0, 1, 4)
    # plot_visibility_jitter(0, 1000, 4)
    plot_visibility_deadtime(0, 1000000, 4)
