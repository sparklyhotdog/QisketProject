from Rotation import Rotation
from Simulator import Simulator
import numpy as np
import math
import matplotlib.pyplot as plt
from alive_progress import alive_bar


rotations = 50
states = ['H', 'D', 'V', 'A']


def plot_car_darkcounts(yaml_fn, start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = []

    with alive_bar(len(x_val), force_tty=True) as bar:
        for dc in x_val:
            sim = Simulator(yaml_fn)
            sim.set_dc_rate(dc)
            sim.run()
            car = sim.get_car()
            y_val.append(car)
            bar()

    plt.plot(x_val, y_val)
    plt.xlabel('Dark Count Rate (counts/second)')
    plt.ylabel('Coincidence-to-Accidental Rate')
    plt.savefig('plots\\car_vs_darkcounts\\car_vs_darkcounts', dpi=1000, bbox_inches='tight')
    plt.show()


def plot_car_deadtime(yaml_fn, start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = []

    with alive_bar(len(x_val), force_tty=True) as bar:
        for deadtime in x_val:
            sim = Simulator(yaml_fn)
            sim.set_deadtime(deadtime)
            sim.run()
            car = sim.get_car()
            y_val.append(car)
            bar()

    plt.plot(x_val, y_val)
    plt.xlabel('Dead Time (ps)')
    plt.ylabel('Coincidence-to-Accidental Rate')
    plt.savefig('plots\\car_vs_deadtime\\car_vs_deadtime', dpi=1000, bbox_inches='tight')
    plt.show()


def plot_car_jitter(yaml_fn, start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = []

    with alive_bar(len(x_val), force_tty=True) as bar:
        for jitter in x_val:
            sim = Simulator(yaml_fn)
            sim.set_jitter(jitter)
            sim.run()
            car = sim.get_car()
            y_val.append(car)
            bar()

    plt.plot(x_val, y_val)
    plt.xlabel('Jitter FWHM (ps)')
    plt.ylabel('Coincidence-to-Accidental Rate')
    plt.savefig('plots\\car_vs_jitter\\car_vs_jitter', dpi=1000, bbox_inches='tight')
    plt.show()


def plot_car_loss(yaml_fn, start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = []

    with alive_bar(len(x_val), force_tty=True) as bar:
        for loss in x_val:
            sim = Simulator(yaml_fn)
            sim.set_loss_signal(loss)
            sim.set_loss_idler(loss)
            sim.run()
            car = sim.get_car()
            y_val.append(car)
            bar()

    plt.plot(x_val, y_val)
    plt.xlabel('Optical Loss')
    plt.ylabel('Coincidence-to-Accidental Rate')
    plt.xscale('log')
    plt.savefig('plots\\car_vs_loss\\car_vs_loss', dpi=1000, bbox_inches='tight')
    plt.show()


def plot_visibility_darkcounts(yaml_fn, state, start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = [[], [], [], []]

    with alive_bar(len(x_val), force_tty=True) as bar:
        for dc in x_val:
            sim = Rotation(yaml_fn, state, rotations)
            sim.set_dc_rate(dc)
            sim.run()
            visibility = sim.get_visibility()
            for i in range(4):
                y_val[i].append(visibility[i])
            bar()

    for i in range(4):
        plt.plot(x_val, y_val[i], label=states[i])
    plt.legend()
    plt.xlabel('Dark Count Rate (counts/second)')
    plt.ylabel('Visibility')
    plt.ylim(0, 1)
    plt.savefig('plots\\visibility_vs_darkcounts\\visibility_vs_darkcounts', dpi=1000)
    plt.show()


def plot_visibility_deadtime(yaml_fn, state, start, stop, num_points):
    x_val = np.linspace(start, stop, num_points)
    y_val = [[], [], [], []]

    with alive_bar(len(x_val), force_tty=True) as bar:
        for deadtime in x_val:
            sim = Rotation(yaml_fn, state, rotations)
            sim.set_deadtime(deadtime)
            sim.run()
            visibility = sim.get_visibility()
            for i in range(4):
                y_val[i].append(visibility[i])
            bar()

    for i in range(4):
        plt.plot(x_val, y_val[i], label=states[i])
    plt.legend()
    plt.xlabel('Dead Time (ps)')
    plt.ylabel('Visibility')
    plt.ylim(0, 1)
    plt.savefig('plots\\visibility_vs_deadtime\\visibility_vs_deadtime', dpi=1000)
    plt.show()


def plot_visibility_jitter(yaml_fn, state, start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = [[], [], [], []]

    with alive_bar(len(x_val), force_tty=True) as bar:
        for jitter in x_val:
            sim = Rotation(yaml_fn, state, rotations)
            sim.set_jitter(jitter)
            sim.run()
            visibility = sim.get_visibility()
            for i in range(4):
                y_val[i].append(visibility[i])
            bar()

    for i in range(4):
        plt.plot(x_val, y_val[i], label=states[i])
    plt.legend()
    plt.xlabel('Jitter FWHM (ps)')
    plt.ylabel('Visibility')
    plt.ylim(0, 1)
    plt.savefig('plots\\visibility_vs_jitter\\visibility_vs_jitter', dpi=1000)
    plt.show()


def plot_visibility_loss(yaml_fn, state, start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = [[], [], [], []]

    with alive_bar(len(x_val), force_tty=True) as bar:
        for loss in x_val:
            sim = Rotation(yaml_fn, state, rotations)
            sim.set_loss_signal(loss)
            sim.set_loss_idler(loss)
            sim.run()
            visibility = sim.get_visibility()
            for i in range(4):
                y_val[i].append(visibility[i])
            bar()

    for i in range(4):
        plt.plot(x_val, y_val[i], label=states[i])
    plt.legend()
    plt.xlabel('Optical Loss')
    plt.ylabel('Visibility')
    plt.ylim(0, 1)
    plt.savefig('plots\\visibility_vs_loss\\visibility_vs_loss', dpi=1000)
    plt.show()


def plot_visibility_phasediff(yaml_fn, start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = [[], [], [], []]

    with alive_bar(len(x_val), force_tty=True) as bar:
        for delta in x_val:
            entangled_state = [1 / math.sqrt(2), 0, 0, np.exp(1j * delta) / math.sqrt(2)]
            sim = Rotation(yaml_fn, entangled_state, rotations)
            sim.run()
            visibility = sim.get_visibility()
            for i in range(4):
                y_val[i].append(visibility[i])
            bar()

    for i in range(4):
        plt.plot(x_val, y_val[i], label=states[i])
    plt.legend()
    plt.xlabel('Phase Difference in the Entangled State')
    plt.ylabel('Visibility')
    plt.ylim(0, 1)
    plt.savefig('plots\\visibility_vs_phasediff\\visibility_vs_phasediff', dpi=1000)
    plt.show()


if __name__ == '__main__':
    d = .25
    qubit_state = [1 / math.sqrt(2), 0, 0, np.exp(1j * d) / math.sqrt(2)]

    # plot_car_darkcounts('config.yaml', 0, 90000, 256)
    # plot_car_deadtime('config.yaml', 0, 1000000, 256)
    plot_car_jitter('config.yaml', 0, 8000, 1024)
    # plot_car_loss('config.yaml', 0, 1, 256)

    # plot_visibility_darkcounts('config.yaml', qubit_state, 0, 1000000, 4)
    # plot_visibility_deadtime('config.yaml', qubit_state, 0, 1000000, 4)
    # plot_visibility_jitter('config.yaml', qubit_state, 0, 100000, 64)
    # plot_visibility_loss('config.yaml', qubit_state, 0, 1, 4)
    # plot_visibility_phasediff('config.yaml', 0, 2*math.pi, 64)
