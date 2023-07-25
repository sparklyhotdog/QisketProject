import random
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import statsmodels.api as sm

'''
Verifies that the number of generated timestamps in a fixed interval follows a Poisson distribution with a QQ plot.
'''

# lambd = average count rate (100k counts/second)
lambd = 100000

# n = the number of events recorded (1 mil)
n = 1000000

# time interval to integrate counts (1ms = 1e9ps)
interval_width = 1e9

# (picoseconds)
timestamps = []

# generating pseudo timestamps following an exponential distribution
t = 0
for i in range(n):
    t += math.floor(random.expovariate(lambd) * 1e12)
    timestamps.append(t)

# count how many time tags are there within each 1ms interval
counts = np.zeros(math.floor(timestamps[-1]/interval_width) + 1)

# loop through timestamps and put them in the correct time intervals
curr_interval = 0
for i in range(n):
    if timestamps[i] > (curr_interval + 1)*interval_width:
        curr_interval += 1
    counts[curr_interval] += 1

plt.hist(counts, bins=20)
plt.xlabel('Number of occurences in the interval')
plt.ylabel('Frequency')
plt.show()

qdist = stats.poisson(np.mean(counts))

sm.qqplot(counts, dist=qdist, line='45')
plt.title('Poisson Probability Plot')
plt.show()
