from Simulator import Simulator
import math
import numpy as np
from qiskit import *
from qiskit.quantum_info import Statevector
from alive_progress import alive_bar
import matplotlib.pyplot as plt


def entangled_protocol():
    lambd = 100000  # average count rate (counts/second)
    total_time = 1
    n = total_time * lambd  # total number of events
    lag = 100  # difference between idler and signal (ps)
    optical_loss_signal = 0.1  # probability of not being detected for the signal photons
    optical_loss_idler = 0.1  # probability of not being detected for the idler photons
    dark_count_rate = 10000  # (counts/second)
    deadtime = 1000000  # (picoseconds)
    jitter_fwhm = 100  # (picoseconds)
    coincidence_interval = 10000  # (picoseconds)

    # rotations = how many rotation steps
    rotations = 200

    x_val = []
    y_val = [[], [], [], []]
    states = ['H', 'D', 'V', 'A']

    qc = QuantumCircuit(2, 2)
    with alive_bar(4 * rotations, force_tty=True) as bar:

        for i in range(4):

            for j in range(rotations):
                # build the circuit
                qc.clear()
                qc = QuantumCircuit(2, 2)

                # delta = phase difference
                delta = .25
                entangled_state = [1 / math.sqrt(2), 0, 0, np.exp(1j * delta) / math.sqrt(2)]
                qc.initialize(entangled_state, [0, 1])

                # keep  A the same, and rotate B
                qc.ry(i * math.pi / 2, 0)
                delta = 4 * j * math.pi / rotations
                qc.ry(delta, 1)
                pr_00 = abs(Statevector(qc)[0]) ** 2

                a = Simulator(pr_00,
                              lambd,
                              total_time,
                              lag,
                              optical_loss_signal,
                              optical_loss_idler,
                              dark_count_rate,
                              deadtime,
                              jitter_fwhm,
                              coincidence_interval)
                bar()
                y_val[i].append(a.simulate())

                if i == 0:
                    x_val.append(delta)

    fig, ax = plt.subplots()

    for i in range(4):
        ax.plot(x_val, y_val[i], label=states[i])

    ax.set_xlabel("Angle")
    ax.set_ylabel("Counts")
    ax.set_ylim(0 - 40, n + 40)
    ax.set_xticks([0, math.pi, 2 * math.pi, 3 * math.pi, 4 * math.pi], ['0', 'π', '2π', '3π', '4π'])
    plt.legend()
    plt.savefig('plot.png', dpi=1000)
    plt.show()


entangled_protocol()
