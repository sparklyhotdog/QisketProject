import matplotlib.pyplot as plt
import math
import random


class Simulator:

    def __init__(self,
                 pr,
                 lambd,                     # average count rate (counts/second)
                 total_time,                # (seconds)
                 lag,                       # difference between idler and signal (picoseconds)
                 optical_loss_signal,       # probability of not being detected for the signal photons
                 optical_loss_idler,        # probability of not being detected for the idler photons
                 dark_count_rate,           # (counts/second)
                 deadtime,                  # (picoseconds)
                 jitter_fwhm,               # (picoseconds)
                 coincidence_interval):     # (picoseconds)
        self.pr = pr
        self.lambd = lambd
        self.total_time = total_time
        self.lag = lag
        self.optical_loss_signal = optical_loss_signal
        self.optical_loss_idler = optical_loss_idler
        self.dark_count_rate = dark_count_rate
        self.deadtime = deadtime
        self.jitter_fwhm = jitter_fwhm
        self.coincidence_interval = coincidence_interval
        self.cross_corr_plot_x = range(-50000, 50000, coincidence_interval)
        self.cross_corr_plot_y = []

    def simulate(self):
        n = self.total_time * self.lambd                  # total number of events

        # generate pseudo timestamps following an exponential distribution
        timestamps_signal = []  # (picoseconds)
        timestamps_idler = []  # (picoseconds)
        t = 0
        for i in range(math.floor(n)):
            dt = math.floor(random.expovariate(self.lambd) * 1e12)
            t += dt
            if random.random() < self.pr:
                # optical loss
                if random.random() > self.optical_loss_signal:
                    timestamps_signal.append(t)
                if random.random() > self.optical_loss_idler:
                    timestamps_idler.append(t + self.lag)

        # jitter
        sigma = self.jitter_fwhm / (2 * math.sqrt(2 * math.log(2)))
        for i in range(len(timestamps_signal)):
            timestamps_signal[i] += math.floor(random.gauss(0, sigma))
            while timestamps_signal[i] < 0:
                timestamps_signal[i] += math.floor(random.gauss(0, sigma))
        for i in range(len(timestamps_idler)):
            timestamps_idler[i] += math.floor(random.gauss(0, sigma))
            while timestamps_idler[i] < 0:
                timestamps_idler[i] += math.floor(random.gauss(0, sigma))

        # generate dark counts
        for i in range(math.floor(n * self.dark_count_rate / self.lambd)):
            timestamps_signal.append(math.floor(random.random() * self.total_time))
            timestamps_idler.append(math.floor(random.random() * self.total_time))
        timestamps_signal.sort()
        timestamps_idler.sort()

        # deadtime
        index = 0
        while index < len(timestamps_signal) - 1:
            if timestamps_signal[index + 1] - timestamps_signal[index] < self.deadtime:
                del timestamps_signal[index + 1]
            else:
                index += 1
        index = 0
        while index < len(timestamps_idler) - 1:
            if timestamps_idler[index + 1] - timestamps_idler[index] < self.deadtime:
                del timestamps_idler[index + 1]
            else:
                index += 1

        # count coincidences
        if len(timestamps_signal) > len(timestamps_idler):
            list2 = timestamps_signal
            list1 = timestamps_idler
        else:
            list1 = timestamps_signal
            list2 = timestamps_idler

        for delta_t in self.cross_corr_plot_x:
            coincidences = []
            # index in list2 of the left bound
            left_bound = 0
            for x in list1:
                # check interval (x + delta_t - coincidence_interval, x + delta_t + coincidence_interval)
                while list2[left_bound] < (x + delta_t) - self.coincidence_interval and left_bound < len(list2) - 1:
                    left_bound += 1
                # now x + delta_t - coincidence_interval <= larger[left_bound]
                if list2[left_bound] < (x + delta_t) + self.coincidence_interval:
                    # x + delta_t and larger[left_bound] are in the same window
                    # the timestamp of the coincidence is the time of the later event
                    coincidences.append(max(x + delta_t, list2[left_bound]))
            self.cross_corr_plot_y.append(len(coincidences))
        return max(self.cross_corr_plot_y)

    def plot_cross_corr(self):
        plt.plot(self.cross_corr_plot_x, self.cross_corr_plot_y)
        plt.xlabel('Time difference (ps)')
        plt.ylabel('Counts')
        plt.savefig('cross_correlation_plot.png', dpi=1000)
        plt.show()


if __name__ == "__main__":
    a = Simulator(1, 100000, 10, 100, .1, .1, 1000, 1000000, 100, 1000)
    print(a.simulate())
    a.plot_cross_corr()
