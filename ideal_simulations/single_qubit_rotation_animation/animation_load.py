import numpy as np
from qiskit import *
from math import pi, sqrt
import pickle

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
n = 200

# num_shots = how many times to run each circuit
num_shots = 1024

# num_frames = how many frames in the animation
num_frames = 50

x_val, y_val = [], []
dy = (state_b[1]-state_a[1])/num_frames
dx = (state_b[0]-state_a[0])/num_frames

for i in range(num_frames):
    temp_x_val, temp_y_val = [], []
    # add dx and dy
    current_state = [state_a[0] + i*dx, state_a[1] + i*dy]

    # normalize so that it is a quantum state vector
    euclidean_norm = sqrt(np.abs(current_state[0]) ** 2 + np.abs(current_state[1]) ** 2)
    new_state = [current_state[0] / euclidean_norm, current_state[1] / euclidean_norm]
    qc = QuantumCircuit(1, 1)

    for j in range(n+1):
        # build the circuit with the new start state
        qc.clear()
        qc = QuantumCircuit(1, 1)
        qc.initialize(new_state, 0)

        # rotate
        delta = 4 * j * pi / n
        qc.ry(delta, 0)
        qc.measure(0, 0)

        # simulate with the aer simulator
        backend = Aer.get_backend('qasm_simulator')
        job = backend.run(transpile(qc, backend), shots=num_shots)
        result = job.result()
        counts = result.get_counts(qc)

        # put in a zero for the count if needed
        if '0' in counts:
            out0 = counts['0']
        else:
            out0 = 0

        temp_x_val.append(delta)
        temp_y_val.append(out0)
    x_val.append(temp_x_val)
    y_val.append(temp_y_val)

with open("x_val", "wb") as x:
    pickle.dump(x_val, x)
with open("y_val", "wb") as y:
    pickle.dump(y_val, y)


