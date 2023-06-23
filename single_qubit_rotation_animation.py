import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from qiskit import *
from math import pi, sqrt

state_H = [1, 0]
state_V = [0, 1]
state_P = [1/sqrt(2), 1/sqrt(2)]
state_M = [1/sqrt(2), -1/sqrt(2)]
state_L = [1/sqrt(2), 1j/sqrt(2)]
state_R = [1/sqrt(2), -1j/sqrt(2)]

# start state
state_a = state_H

# end state
state_b = state_L

# n = how many rotation steps
n = 20

# num_shots = how many times to run each circuit
num_shots = 512

# num_frames = how many frames in the animation
num_frames = 50


fig = plt.figure()
axis = plt.axes(xlim=(0, 4*pi), ylim=(0 - 40, num_shots + 40))
plt.xlabel("Angle")
plt.ylabel("Counts")
plt.ylim(0 - 40, num_shots + 40)
plt.xticks([0, pi, 2*pi, 3*pi, 4*pi], ['0', 'π', '2π', '3π', '4π'])
line, = axis.plot([], [], lw=1)


def init():
    line.set_data([], [])
    return line,


dy = (state_b[1]-state_a[1])/num_frames
dx = (state_b[0]-state_a[0])/num_frames


def animate(i):
    x_val, y_val = [], []
    current_state = [state_a[0] + i*dx, state_a[1] + i*dy]

    # normalize so that it is a quantum state vector
    euclidean_norm = sqrt(np.abs(current_state[0]) ** 2 + np.abs(current_state[1]) ** 2)
    new_state = [current_state[0] / euclidean_norm, current_state[1] / euclidean_norm]
    qc = QuantumCircuit(1, 1)

    for j in range(n+1):
        qc.clear()
        qc = QuantumCircuit(1, 1)
        qc.initialize(new_state, 0)

        # rotate the qubit (equivalent to rotating the polarizer)
        delta = 4 * j * pi / n
        qc.ry(delta, 0)
        qc.measure(0, 0)

        # simulate
        backend = Aer.get_backend('qasm_simulator')
        job = backend.run(transpile(qc, backend), shots=num_shots)
        result = job.result()
        counts = result.get_counts(qc)

        if '0' in counts:
            out0 = counts['0']
        else:
            out0 = 0

        x_val.append(delta)
        y_val.append(out0)

    line.set_data(x_val, y_val)

    return line,


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=2, blit=True)
plt.show()
