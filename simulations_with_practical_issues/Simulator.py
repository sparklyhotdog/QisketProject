import matplotlib.pyplot as plt
import math
import random
import numpy as np
import yaml


class Simulator:

    def __init__(self, pr, yaml_fn):
        self.yaml_fn = yaml_fn
        y_fn = open(self.yaml_fn, 'r')
        self.dicty = yaml.load(y_fn, Loader=yaml.SafeLoader)
        y_fn.close()

        self.pr = pr
        self.lambd = self.dicty['lambd']
        self.total_time = self.dicty['total_time']
        self.lag = self.dicty['lag']
        self.optical_loss_signal = self.dicty['optical_loss_signal']
        self.optical_loss_idler = self.dicty['optical_loss_idler']
        self.dark_count_rate = self.dicty['dark_count_rate']
        self.deadtime = self.dicty['deadtime']
        self.jitter_fwhm = self.dicty['jitter_fwhm']
        self.coincidence_interval = self.dicty['coincidence_interval']
        self.dtime = []
        self.bins = []

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
        dt_range = 100000      # in picoseconds

        s_floor = np.int64(np.floor(np.array(timestamps_signal) / (dt_range / 2)))
        i_floor = np.int64(np.floor(np.array(timestamps_idler) / (dt_range / 2)))
        coinc0 = np.intersect1d(s_floor, i_floor, return_indices=True)
        coinc1 = np.intersect1d(s_floor, i_floor - 1, return_indices=True)
        coinc2 = np.intersect1d(s_floor, i_floor + 1, return_indices=True)
        coinc = np.hstack((coinc0, coinc1, coinc2))

        s_time = np.array(timestamps_signal)[coinc[1]]
        i_time = np.array(timestamps_idler)[coinc[2]]
        self.dtime = s_time - i_time

        self.bins = np.arange(-dt_range / 2, dt_range / 2 + self.coincidence_interval / 2, self.coincidence_interval)

        return max(np.histogram(self.dtime, self.bins)[0])

    def plot_cross_corr(self):
        plt.hist(self.dtime, self.bins)
        plt.xlabel('Time difference (ps)')
        plt.ylabel('Counts')
        plt.savefig('plots\\cross_correlation_plot.png', dpi=1000)
        plt.show()


if __name__ == "__main__":
    a = Simulator(1, 'config.yaml')
    print(a.simulate())
    a.plot_cross_corr()
