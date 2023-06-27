import random
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import statsmodels.api as sm

# lambd = average count rate (counts/second, typically on the order of 10^4/s)
lambd = 100000

# n = the number of events recorded
n = 100000

#time interval to integrate counts, say 1ms (1e9ps)
interval = 1e9

# interval width for the poisson distribution
interval_width = 10

#timestamps in multiples of 1ps, just as the real time taggers are specified
timestamps = []

t = 0
for i in range(n):
    t += math.floor(random.expovariate(lambd) * 1e12)
    timestamps.append(t)


t_start = timestamps[0]
t_stop = timestamps[-1]

counts = np.zeros(math.floor(t_stop / interval) + 1)

for i in range(n):
    index = math.floor((timestamps[i] - t_start) / interval)
    counts[index] +=1

# calculate the number of events occuring in a fixed interval of time
# the poisson distribution is an appropriate model for counts
# num_intervals = math.floor(timestamps[n - 1]/interval_width)
# counts = np.zeros(num_intervals + 1)
# curr_interval = 0

# for i in range(n - 1):
#     # if the time is larger than the upper bound of the current time interval, move on to the next interval
#     if timestamps[i] > (curr_interval + 1)*interval_width:
#         curr_interval += 1
#     # add to the counts
#     counts[curr_interval] += 1

plt.hist(counts)
plt.xlabel('Number of occurences in the interval')
plt.ylabel('Frequency')
plt.show()

test_array = np.array(counts)
test_mu = np.mean(test_array)
qdist = stats.poisson(test_mu)

sm.qqplot(test_array, dist=qdist, line='45')
plt.title('Poisson Probability Plot')
plt.show()
