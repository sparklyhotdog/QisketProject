from simulations_with_practical_issues.Simulator import Simulator
import math
import numpy as np
from qiskit import *
from qiskit.quantum_info import Statevector
from alive_progress import alive_bar
import matplotlib.pyplot as plt
import yaml


class EntanglementSim:

    def __init__(self, yaml_fn, entangled_state, rotations):
        self.yaml_fn = yaml_fn
        y_fn = open(self.yaml_fn, 'r')
        self.dicty = yaml.load(y_fn, Loader=yaml.SafeLoader)
        y_fn.close()

        self.lambd = self.dicty['lambd']
        self.total_time = self.dicty['total_time']
        self.n = self.lambd * self.total_time
        self.lag = self.dicty['lag']
        self.optical_loss_signal = self.dicty['optical_loss_signal']
        self.optical_loss_idler = self.dicty['optical_loss_idler']
        self.dark_count_rate = self.dicty['dark_count_rate']
        self.deadtime = self.dicty['deadtime']
        self.jitter_fwhm = self.dicty['jitter_fwhm']
        self.coincidence_interval = self.dicty['coincidence_interval']
        self.entangled_state = entangled_state
        self.rotations = rotations
        self.x_val = []
        self.y_val = [[], [], [], []]

    def run(self):
        states = ['H', 'D', 'V', 'A']

        qc = QuantumCircuit(2, 2)
        with alive_bar(4 * self.rotations, force_tty=True) as bar:

            for i in range(4):

                for j in range(self.rotations):
                    # build the circuit
                    qc.clear()
                    qc = QuantumCircuit(2, 2)

                    qc.initialize(self.entangled_state, [0, 1])

                    # keep  A the same, and rotate B
                    qc.ry(i * math.pi / 2, 0)
                    delta = 4 * j * math.pi / self.rotations
                    qc.ry(delta, 1)
                    pr_00 = abs(Statevector(qc)[0]) ** 2

                    sim = Simulator(pr_00, 'config.yaml')
                    bar()
                    self.y_val[i].append(sim.simulate())

                    if i == 0:
                        self.x_val.append(delta)

        # calculate visibility
        visibility = []
        for i in range(4):
            i_max = max(self.y_val[i])
            i_min = min(self.y_val[i])
            visibility.append((i_max - i_min) / (i_max + i_min))
            print(states[i] + ': ' + str(visibility[i]))

        return visibility

    def plot_correlation(self):
        states = ['H', 'D', 'V', 'A']

        fig, ax = plt.subplots()

        for i in range(4):
            ax.plot(self.x_val, self.y_val[i], label=states[i])

        ax.set_xlabel("Angle")
        ax.set_ylabel("Counts")
        ax.set_ylim(0 - 40, self.n + 40)
        ax.set_xticks([0, math.pi, 2 * math.pi, 3 * math.pi, 4 * math.pi], ['0', 'π', '2π', '3π', '4π'])
        plt.legend()
        plt.savefig('plots\polarization_correlation.png', dpi=1000)
        plt.show()

    def set_dc_rate(self, dc):
        self.dark_count_rate = dc


def plot_visibility_darkcounts():
    states = ['H', 'D', 'V', 'A']

    x_val = range(0, 100000, 50000)
    y_val = [[], [], [], []]

    # delta = phase difference
    delta = .25
    entangled_state = [1 / math.sqrt(2), 0, 0, np.exp(1j * delta) / math.sqrt(2)]

    for dc in x_val:
        sim = EntanglementSim('config.yaml', entangled_state, 200)
        sim.set_dc_rate(dc)
        visibility = sim.run()
        for i in range(4):
            y_val[i].append(visibility[i])

    for i in range(4):
        plt.plot(x_val, y_val[i], label=states[i])
    plt.legend()
    plt.xlabel('Dark Count Rate (counts/second)')
    plt.ylabel('Visibility')
    plt.savefig('plots\visibility_vs_darkcounts', dpi=1000)
    plt.show()


if __name__ == '__main__':
    d = .25
    state = [1 / math.sqrt(2), 0, 0, np.exp(1j * d) / math.sqrt(2)]
    a = EntanglementSim('config.yaml', state, 200)
    a.run()
    a.plot_correlation()
