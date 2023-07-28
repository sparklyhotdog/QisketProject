import math
import random
import numpy as np
from IPython.display import display, Math, Latex
import matplotlib.pyplot as plt
import copy
import matplotlib.markers

'''
Displays a plot showing the effects of loss, jitter, dark counts, and dead time.

This script generates a small number of timestamps with exaggerated parameters, and plots them in a makeshift stem plot.
The plot is then displayed and saved in a png file.
'''

lambd = 100000              # average count rate (100k counts/second)
n = 10                      # total number of events (1 mil)
optical_loss = 0.3          # probability of not being detected for the signal photons
dark_count_rate = 50000     # (counts/second)
ambient_light = 50000       # (counts/second)
deadtime = 3000000          # (picoseconds)
jitter_fwhm = 1500000       # (picoseconds)

# ___________________________________________________
# generating timestamps

# (picoseconds)
timestamps = []

# generate pseudo timestamps following an exponential distribution
t = 0
for i in range(n):
    dt = math.floor(random.expovariate(lambd) * 1e12)
    t += dt
    timestamps.append(t)
original_timestamps = copy.deepcopy(timestamps)

# generate ambient light
t_stop = timestamps[-1]
for i in range(math.floor(n*ambient_light/lambd)):
    timestamps.append(math.floor(random.random()*t_stop))
timestamps.sort()
ambient_timestamps = copy.deepcopy(timestamps)

# optical loss
for x in timestamps:
    if random.random() < optical_loss:
        timestamps.remove(x)

optical_loss_timestamps = copy.deepcopy(timestamps)

# jitter
sigma = jitter_fwhm/(2*math.sqrt(2*math.log(2)))
for i in range(len(timestamps)):
    timestamps[i] += math.floor(random.gauss(0, sigma))
    while timestamps[i] < 0:
        timestamps[i] += math.floor(random.gauss(0, sigma))
jitter_timestamps = copy.deepcopy(timestamps)

# generate dark counts
t_stop = timestamps[-1]
for i in range(math.floor(n*dark_count_rate/lambd)):
    timestamps.append(math.floor(random.random()*t_stop))
timestamps.sort()
darkcounts_timestamps = copy.deepcopy(timestamps)

# deadtime
index = 0
while index < len(timestamps) - 1:
    if timestamps[index + 1] - timestamps[index] < deadtime:
        del timestamps[index + 1]
    else:
        index += 1

# _______________________________________________________
# plotting
x_lim_l = original_timestamps[0] - 1000000
x_lim_r = original_timestamps[-1] + 1000000

fig, axs = plt.subplots(6, 1)
fig.tight_layout(pad=1)
fig.set_size_inches(6, 6)

original = plt.subplot(6, 1, 1)
al = plt.subplot(6, 1, 2)
optical_loss = plt.subplot(6, 1, 3)
jitter = plt.subplot(6, 1, 4)
dc = plt.subplot(6, 1, 5)
dt = plt.subplot(6, 1, 6)
subplots = [original, al, optical_loss, jitter, dc, dt]
for subplot in subplots:
    subplot.set_xlim(x_lim_l, x_lim_r)
    subplot.set_ylim(0, 1.5)
    subplot.set_yticks([])
    subplot.set_yticks([])
    subplot.set_xticks([])
    subplot.set_frame_on(False)
    subplot.set_xlabel('Time', loc='right')
    subplot.hlines(0, original_timestamps[0], original_timestamps[-1], colors='C3')

original.set_title('Original Arrival Times of Photons')
original.stem(original_timestamps, [1]*len(original_timestamps))

al.set_title('After Adding Ambient Light')
al.stem(original_timestamps, [1]*len(original_timestamps))
# the added points get pluses
added_points = copy.deepcopy(ambient_timestamps)
for x in original_timestamps:
    added_points.remove(x)
if len(added_points) > 0:
    al.stem(added_points, [1] * len(added_points), markerfmt='P', linefmt='-')

optical_loss.set_title('After Optical Loss')
optical_loss.stem(optical_loss_timestamps, [1]*len(optical_loss_timestamps))
# the missing points get crossed out
missing_points = copy.deepcopy(ambient_timestamps)
for x in optical_loss_timestamps:
    missing_points.remove(x)
if len(missing_points) > 0:
    optical_loss.stem(missing_points, [1]*len(missing_points), markerfmt='x', linefmt='--')

jitter.set_title('After Incorporating Jitter')
for mu in optical_loss_timestamps:
    # make the normal distribution curve
    x_val = np.linspace(mu - 3*sigma, mu + 3*sigma)
    y_val = math.e ** ((x_val - mu)**2/(-2 * sigma**2))
    jitter.fill_between(x_val, y_val, [0]*len(x_val), color='#a7cef1')
jitter.stem(jitter_timestamps, [1]*len(jitter_timestamps))

dc.set_title('After Adding Dark Counts')
dc.stem(jitter_timestamps, [1]*len(jitter_timestamps))
# the added points get pluses
added_points = copy.deepcopy(darkcounts_timestamps)
for x in jitter_timestamps:
    added_points.remove(x)
if len(added_points) > 0:
    dc.stem(added_points, [1] * len(added_points), markerfmt='P', linefmt='-')

dt.set_title('After Accounting for Dead Time')
for x in timestamps:
    x_val = [x, x + deadtime, x + deadtime, x]
    y_val = [0, 0, 1, 1]
    dt.fill(x_val, y_val, c='#a7cef1')
dt.stem(timestamps, [1]*len(timestamps))

plt.savefig('time_tagger_plot.png', dpi=1000)
plt.show()
