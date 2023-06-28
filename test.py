from qiskit import *
from qiskit.quantum_info import Statevector
import math
import random
import numpy as np
from IPython.display import display, Math, Latex

lambd = 100000              # average count rate (100k counts/second)
n = 1000000                 # total number of events (1 mil)
t_difference = 0            # difference between idler and signal
optical_loss = 0.1          # probability of not being detected
dark_count_rate = 10000     # (counts/second)
deadtime = 1000000          # (picoseconds)
hasAfterpulse = true
t_afterpulse = 100000       # average time that an afterpulse takes place after a detection event
pr_afterpulse = 0.1         # probability of afterpulsing

# (picoseconds)
timestamps_signal = []
timestamps_idler = []

# generate pseudo timestamps following an exponential distribution
t = 0
for i in range(n):
    dt = math.floor(random.expovariate(lambd) * 1e12)
    t += dt

    # optical loss
    if random.random() > optical_loss:
        timestamps_signal.append(t)
    if random.random() > optical_loss:
        timestamps_idler.append(t + t_difference)

t_stop = max([timestamps_signal[-1], timestamps_idler[-1]])

# generate dark counts
for i in range(math.floor(n*dark_count_rate/lambd)):
    timestamps_signal.append(random.randrange(t_stop))
    timestamps_idler.append(random.randrange(t_stop))
timestamps_signal.sort()
timestamps_idler.sort()

print(len(timestamps_signal))
print(len(timestamps_idler))


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

print(len(timestamps_signal))
print(len(timestamps_idler))

