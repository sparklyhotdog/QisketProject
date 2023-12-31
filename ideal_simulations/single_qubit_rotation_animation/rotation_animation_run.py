import pickle
import matplotlib.pyplot as plt
from math import pi, sqrt
import matplotlib.animation as animation

'''
Runs the animation using preloaded data.

The data is loaded into x_val and y_val from rotation_animation_load.py, because running the simulation on the spot 
takes too long. The animation is diplayed in a new window and saved in a gif using pillow.
'''

# num_shots = how many times to run each circuit
num_shots = 1024

# num_frames = how many frames in the animation
num_frames = 50

fig = plt.figure()
axis = plt.axes(xlim=(0, 4*pi), ylim=(0 - 40, num_shots + 40))
plt.xlabel("Polarizer Angle")
plt.ylabel("Counts")
plt.ylim(0 - 40, num_shots + 40)
plt.xticks([0, pi, 2*pi, 3*pi, 4*pi], ['0', 'π', '2π', '3π', '4π'])
line, = axis.plot([], [], lw=1)

with open("x_val", "rb") as x:
    x_val = pickle.load(x)

with open("y_val", "rb") as y:
    y_val = pickle.load(y)


def init():
    line.set_data([], [])
    return line,


def animate(frame):
    line.set_data(x_val[frame], y_val[frame])

    return line,


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=20, blit=True)

writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
anim.save('plot.gif', writer=writer)
plt.show()
