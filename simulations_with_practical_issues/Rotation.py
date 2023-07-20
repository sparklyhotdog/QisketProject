from Simulator import Simulator
import math
import numpy as np
from qiskit import *
from qiskit.quantum_info import Statevector
from alive_progress import alive_bar
import matplotlib.pyplot as plt
import yaml

states = ('H', 'D', 'V', 'A')


class Rotation:

    def __init__(self, yaml_fn, entangled_state, rotations=200):
        self.yaml_fn = yaml_fn
        y_fn = open(self.yaml_fn, 'r')
        self.dicty = yaml.load(y_fn, Loader=yaml.SafeLoader)
        y_fn.close()

        self.lambd = self.dicty['lambd']
        self.total_time = self.dicty['total_time']
        self.n = self.lambd * self.total_time
        self.entangled_state = entangled_state
        self.rotations = rotations
        self.x_val = []
        self.y_val = [[], [], [], []]
        self.state_latex = ''

    def run(self):
        qc = QuantumCircuit(2, 2)

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

                if i + j == 0:
                    self.state_latex = Statevector(qc).draw(output='latex_source')

                sim = Simulator(self.yaml_fn, pr_00)
                sim.generate_timestamps()
                sim.cross_corr()
                self.y_val[i].append(sim.max_counts)

                if i == 0:
                    self.x_val.append(delta)

    def calc_visibility(self):

        visibility = []
        for i in range(4):
            i_max = max(self.y_val[i])
            i_min = min(self.y_val[i])
            visibility.append((i_max - i_min) / (i_max + i_min))
            # print(states[i] + ': ' + str(visibility[i]))

        return visibility

    def plot_correlation(self, path=None, plot_title=None):

        fig, ax = plt.subplots()

        for i in range(4):
            ax.plot(self.x_val, self.y_val[i], label=states[i])

        ax.set_xlabel('Polarizer Angle for the 2nd Photon')
        ax.set_ylabel('Counts')
        ax.set_ylim(0 - 40, self.n + 40)

        # default is to put the entangled state as the title
        if plot_title is None:
            ax.set_title('$' + self.state_latex + '$')
        else:
            ax.set_title(plot_title)

        ax.set_xticks([0, math.pi, 2 * math.pi, 3 * math.pi, 4 * math.pi], ['0', 'π', '2π', '3π', '4π'])
        plt.legend()
        if path is not None:
            plt.savefig(path, dpi=1000, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    d = .25
    state = [1 / math.sqrt(2), 0, 0, np.exp(1j * d) / math.sqrt(2)]
    # default title expands the complex number into floats
    # we set the title to keep it in its exponential form
    a = Rotation('config.yaml', state, rotations=50)
    a.run()
    print(a.calc_visibility())
    titl = '$\\frac{\sqrt{2}}{2} |00\\rangle+e^{' + str(d) + 'i}|11\\rangle$'
    a.plot_correlation('plots\\polarization_correlation.png', titl)
