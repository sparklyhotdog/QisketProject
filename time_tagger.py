import random
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import statsmodels.api as sm

# lambd = average count rate (counts/second, typically on the order of 10^4/s)
lambd = 100000

# n = the number of events recorded
n = 1000000

#time interval to integrate counts, say 1ms (1e9ps)
interval = 1e9

# interval width for the poisson distribution
# interval_width = 10

#timestamps in multiples of 1ps, just as the real time taggers are specified
timestamps = []

#generating pseudo timestamps following an exp distribution
t = 0
for i in range(n):
    t += math.floor(random.expovariate(lambd) * 1e12)
    timestamps.append(t)


t_start = timestamps[0]
t_stop = timestamps[-1]

# Now we want to count how many time tags are there within each 1ms interval
# this count number is recorded by the counts[] array
# the size of counts[] is pre-specified to avoid trouble
counts = np.zeros(math.floor(t_stop / interval) + 1)

# loop through the timestamps to put them into the correct time intervals
for i in range(n):
    index = math.floor((timestamps[i] - t_start) / interval)
    counts[index] +=1


plt.hist(counts, bins=20)
plt.xlabel('Number of occurences in the interval')
plt.ylabel('Frequency')
plt.show()

test_array = np.array(counts)
test_mu = np.mean(test_array)
qdist = stats.poisson(test_mu)

sm.qqplot(test_array, dist=qdist, line='45')
plt.title('Poisson Probability Plot')
plt.show()
