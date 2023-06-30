import matplotlib.pyplot as plt

n = 1000000                     # total number of events (1 mil)
coincidence_interval = 1000    # (picoseconds)

timestamps_signal = []          # (picoseconds)
timestamps_idler = []           # (picoseconds)

# read in the timestamps from the txt files
with open('timestamps_signal') as s:
    for line in s:
        timestamps_signal.append(int(line))

with open('timestamps_idler') as i:
    for line in i:
        timestamps_idler.append(int(line))

# count coincidences
x_val = range(-100000, 100000, 1000)
y_val = []
if len(timestamps_signal) > len(timestamps_idler):
    list2 = timestamps_signal
    list1 = timestamps_idler
else:
    list1 = timestamps_signal
    list2 = timestamps_idler

for delta_t in x_val:
    coincidences = []
    # index in list2 of the left bound
    left_bound = 0
    for x in list1:
        # check interval (x + delta_t - coincidence_interval, x + delta_t + coincidence_interval)
        while list2[left_bound] < (x + delta_t) - coincidence_interval and left_bound < len(list2) - 1:
            left_bound += 1
        # now x + delta_t - coincidence_interval <= larger[left_bound]
        if list2[left_bound] < (x + delta_t) + coincidence_interval:
            # x + delta_t and larger[left_bound] are in the same window
            # the timestamp of the coincidence is the time of the later event
            coincidences.append(max(x + delta_t, list2[left_bound]))
    print(len(coincidences))
    y_val.append(len(coincidences))
plt.plot(x_val, y_val)
plt.xlabel('Time difference (ps)')
plt.ylabel('Counts')
plt.savefig('cross_correlation_plot.png', dpi=1000)
plt.show()
