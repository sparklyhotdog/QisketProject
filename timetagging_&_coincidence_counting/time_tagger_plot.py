import math
import random
import numpy as np
from IPython.display import display, Math, Latex
import matplotlib.pyplot as plt
import copy

lambd = 100000              # average count rate (100k counts/second)
n = 15                      # total number of events (1 mil)
optical_loss_signal = 0.1   # probability of not being detected for the signal photons
optical_loss_idler = 0.1    # probability of not being detected for the idler photons
dark_count_rate = 10000     # (counts/second)
deadtime = 1000000          # (picoseconds)
jitter_fwhm = 100           # (picoseconds)

# (picoseconds)
timestamps = []
original_timestamps = []

# generate pseudo timestamps following an exponential distribution
t = 0
for i in range(n):
    dt = math.floor(random.expovariate(lambd) * 1e12)
    t += dt
    original_timestamps.append(t)
    # optical loss
    if random.random() > optical_loss_signal:
        timestamps.append(t)
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

print(original_timestamps)
print(optical_loss_timestamps)
print(jitter_timestamps)
print(darkcounts_timestamps)
print(timestamps)

x_lim_l = original_timestamps[0] - 1000000
x_lim_r = original_timestamps[-1] + 1000000

fig, axs = plt.subplots(5, 1)
fig.tight_layout(pad=2)
fig.set_size_inches(10, 10)

original = plt.subplot(5, 1, 1)
original.stem(original_timestamps, [1]*len(original_timestamps))
original.set_xlim(x_lim_l, x_lim_r)
original.set_ylim(0, 1.5)
original.set_yticks([])
original.set_xticks([])
original.set_frame_on(False)
original.set_title('Original Arrival Times of Photons')
original.set_xlabel('Time', loc='right')

optical_loss = plt.subplot(5, 1, 2)
optical_loss.stem(optical_loss_timestamps, [1]*len(optical_loss_timestamps))
optical_loss.set_xlim(x_lim_l, x_lim_r)
optical_loss.set_ylim(0, 1.5)
optical_loss.set_yticks([])
optical_loss.set_xticks([])
optical_loss.set_frame_on(False)
optical_loss.set_title('After Optical Loss')
optical_loss.set_xlabel('Time', loc='right')

jitter = plt.subplot(5, 1, 3)
jitter.stem(jitter_timestamps, [1]*len(jitter_timestamps))
jitter.set_xlim(x_lim_l, x_lim_r)
jitter.set_ylim(0, 1.5)
jitter.set_yticks([])
jitter.set_yticks([])
jitter.set_xticks([])
jitter.set_frame_on(False)
jitter.set_title('After Incorporating Jitter')
jitter.set_xlabel('Time', loc='right')

dc = plt.subplot(5, 1, 4)
dc.stem(darkcounts_timestamps, [1]*len(darkcounts_timestamps))
dc.set_xlim(x_lim_l, x_lim_r)
dc.set_ylim(0, 1.5)
dc.set_yticks([])
dc.set_yticks([])
dc.set_xticks([])
dc.set_frame_on(False)
dc.set_title('After Adding Dark Counts')
dc.set_xlabel('Time', loc='right')

dt = plt.subplot(5, 1, 5)
dt.stem(timestamps, [1]*len(timestamps))
dt.set_xlim(x_lim_l, x_lim_r)
dt.set_ylim(0, 1.5)
dt.set_yticks([])
dt.set_yticks([])
dt.set_xticks([])
dt.set_frame_on(False)
dt.set_title('After Accounting for Dead Time')
dt.set_xlabel('Time', loc='right')

plt.show()

