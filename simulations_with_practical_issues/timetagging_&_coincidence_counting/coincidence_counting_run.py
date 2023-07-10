import matplotlib.pyplot as plt
from alive_progress import alive_bar
import numpy as np

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

range_ps = 200000


Sfloor = np.int64(np.floor(np.array(timestamps_signal) / (range_ps / 2)))
Ifloor = np.int64(np.floor(np.array(timestamps_idler) / (range_ps / 2)))
coinc0 = np.intersect1d(Sfloor, Ifloor, return_indices=True)
coinc1 = np.intersect1d(Sfloor, Ifloor - 1, return_indices=True)
coinc2 = np.intersect1d(Sfloor, Ifloor + 1, return_indices=True)
coinc = np.hstack((coinc0, coinc1, coinc2))

Stime = np.array(timestamps_signal)[coinc[1]]
Itime = np.array(timestamps_idler)[coinc[2]]
dtime = Stime - Itime

# iterate over coincidence_interval, find max of the max(histo)'s
max_counts = 0

for dt in np.arange(0, coincidence_interval, coincidence_interval/10):
    # bins in picoseconds
    bins = np.arange(-range_ps / 2 + dt, range_ps / 2 + coincidence_interval, coincidence_interval)
    [histo, edges] = np.histogram(dtime, bins)
    curr_count = max(histo)
    if curr_count > max_counts:
        max_counts = curr_count
        max_dtime = dtime
        max_bins = bins
    else:
        break
print(max_counts)
plt.hist(max_dtime, max_bins)
plt.xlabel('Time difference (ps)')
plt.ylabel('Counts')
plt.savefig('cross_correlation_plot.png', dpi=1000)
plt.show()

#
# # count coincidences
# x_val = range(-100000, 100000, coincidence_interval)
# y_val = []
# if len(timestamps_signal) > len(timestamps_idler):
#     list2 = timestamps_signal
#     list1 = timestamps_idler
# else:
#     list1 = timestamps_signal
#     list2 = timestamps_idler
#
# with alive_bar(len(x_val), force_tty=True) as bar:
#     for delta_t in x_val:
#         coincidences = []
#         # index in list2 of the left bound
#         left_bound = 0
#         for x in list1:
#             # check interval (x + delta_t - coincidence_interval, x + delta_t + coincidence_interval)
#             while list2[left_bound] < (x + delta_t) - coincidence_interval and left_bound < len(list2) - 1:
#                 left_bound += 1
#             # now x + delta_t - coincidence_interval <= larger[left_bound]
#             if list2[left_bound] < (x + delta_t) + coincidence_interval:
#                 # x + delta_t and larger[left_bound] are in the same window
#                 # the timestamp of the coincidence is the time of the later event
#                 coincidences.append(max(x + delta_t, list2[left_bound]))
#         # print(len(coincidences))
#         bar(1)
#         y_val.append(len(coincidences))
#
# print(max(y_val))
#
# plt.plot(x_val, y_val)
# plt.xlabel('Time difference (ps)')
# plt.ylabel('Counts')
# plt.savefig('cross_correlation_plot.png', dpi=1000)
# plt.show()
