from qutip import *
import numpy as np
from matplotlib import pyplot, animation
from mpl_toolkits.mplot3d import Axes3D

'''
Saves the frames of a rotation in the Bloch sphere to be made into a gif.

This script creates each frames using the Qutip library and a rotation matrix, and saves them in a folder called 
'temp_animation_frames'. The quantum state is rotated around the cos(φ)x + sin(φ)y axis. This could also be done with 
Qiskit using a rotation gate and returning the Statevector. Then, you can make the gif with ffmpeg in the shell.

Example shell command:
ffmpeg framerate 10 -i ideal_simulations/bloch_sphere_rotation_animation/temp_animation_frames/bloch_%d.png title.gif

The default frame rate is 25 fps
'''

num_frames = 100
theta = np.linspace(0, 2 * np.pi, num_frames)
b = qutip.Bloch()
state = qutip.basis(2, 0)
phi = 1

for i in range(num_frames):
    b.clear()
    ry = Qobj([[np.cos(theta[i] / 2), -1j * np.exp(1j*phi) * np.sin(theta[i] / 2)],
               [-1j * np.exp(-1j*phi) * np.sin(theta[i] / 2), np.cos(theta[i] / 2)]])
    b.add_states(ry * state)
    b.save(dirc='temp_animation_frames')

