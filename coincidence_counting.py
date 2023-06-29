from qiskit import *
from qiskit.quantum_info import Statevector
import math
import random
import numpy as np
from IPython.display import display, Math, Latex

lambd = 100000                  # average count rate (100k counts/second)
n = 1000000                     # total number of events (1 mil)
t_difference = 0                # difference between idler and signal
optical_loss_signal = 0.1       # probability of not being detected for the signal photons
optical_loss_idler = 0.1        # probability of not being detected for the idler photons
dark_count_rate = 10000         # (counts/second)
deadtime = 1000000              # (picoseconds)
jitter_fwhm = 100               # (picoseconds)
coincidence_interval = 10000    # (picoseconds)

# (picoseconds)
timestamps_signal = []
timestamps_idler = []

# generate pseudo timestamps following an exponential distribution
t = 0
for i in range(n):
    dt = math.floor(random.expovariate(lambd) * 1e12)
    t += dt

    # optical loss
    if random.random() > optical_loss_signal:
        timestamps_signal.append(t)
    if random.random() > optical_loss_idler:
        timestamps_idler.append(t + t_difference)

# jitter
sigma = jitter_fwhm/(2*math.sqrt(2*math.log(2)))
for i in range(len(timestamps_signal)):
    timestamps_signal[i] += math.floor(random.gauss(0, sigma))
    while timestamps_signal[i] < 0:
        timestamps_signal[i] += math.floor(random.gauss(0, sigma))
for i in range(len(timestamps_idler)):
    timestamps_idler[i] += math.floor(random.gauss(0, sigma))
    while timestamps_idler[i] < 0:
        timestamps_idler[i] += math.floor(random.gauss(0, sigma))

# generate dark counts
t_stop = max([timestamps_signal[-1], timestamps_idler[-1]])
for i in range(math.floor(n*dark_count_rate/lambd)):
    timestamps_signal.append(math.floor(random.random()*t_stop))
    timestamps_idler.append(math.floor(random.random()*t_stop))
timestamps_signal.sort()
timestamps_idler.sort()

# deadtime
index = 0
while index < len(timestamps_signal) - 1:
    if timestamps_signal[index + 1] - timestamps_signal[index] < deadtime:
        del timestamps_signal[index + 1]
    else:
        index += 1

index = 0
while index < len(timestamps_idler) - 1:
    if timestamps_idler[index + 1] - timestamps_idler[index] < deadtime:
        del timestamps_idler[index + 1]
    else:
        index += 1

# count coincidences
coincidences = []
if len(timestamps_signal) > len(timestamps_idler):
    list2 = timestamps_signal
    list1 = timestamps_idler
else:
    list1 = timestamps_signal
    list2 = timestamps_idler

# index in list2 of the left bound
left_bound = 0
for x in list1:
    # check interval (x - coincidence_interval, x + coincidence_interval)
    while list2[left_bound] < x - coincidence_interval or left_bound == len(list2):
        left_bound += 1
    # now x - coincidence_interval <= larger[left_bound]
    # ** < or <= **
    if list2[left_bound] < x + coincidence_interval:
        # x and larger[left_bound] are in the same window
        # the timestamp of the coincidence is the time of the later event
        coincidences.append(max(x, list2[left_bound]))

print(len(coincidences)/n)

