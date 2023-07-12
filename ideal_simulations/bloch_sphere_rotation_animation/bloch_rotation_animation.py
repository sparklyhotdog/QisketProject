from qutip import *
import numpy as np
from matplotlib import pyplot, animation
from mpl_toolkits.mplot3d import Axes3D

# saves the frames into a folder
# then you can make the .gif with ffmpeg in the shell
# ffmpeg framerate 10 -i ideal_simulations/bloch_sphere_rotation_animation/temp_animation_frames/bloch_%d.png output.gif
# default frame rate is 25fps

# Rotation θ around the cos(φ)x + sin(φ)y axis

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

