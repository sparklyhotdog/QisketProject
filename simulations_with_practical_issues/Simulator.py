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
        self.timestamps_signal = []
        self.timestamps_idler = []
        self.coinc = np.array

    def run(self):
        n = self.total_time * self.lambd                  # total number of events
        # generate pseudo timestamps following an exponential distribution
        t = 0
        for i in range(math.floor(n)):
            dt = math.floor(random.expovariate(self.lambd) * 1e12)
            t += dt
            if random.random() < self.pr:
                # optical loss
                if random.random() > self.optical_loss_signal:
                    self.timestamps_signal.append(t)
                if random.random() > self.optical_loss_idler:
                    self.timestamps_idler.append(t + self.lag)

        # jitter
        sigma = self.jitter_fwhm / (2 * math.sqrt(2 * math.log(2)))
        for i in range(len(self.timestamps_signal)):
            self.timestamps_signal[i] += math.floor(random.gauss(0, sigma))
            while self.timestamps_signal[i] < 0:
                self.timestamps_signal[i] += math.floor(random.gauss(0, sigma))
        for i in range(len(self.timestamps_idler)):
            self.timestamps_idler[i] += math.floor(random.gauss(0, sigma))
            while self.timestamps_idler[i] < 0:
                self.timestamps_idler[i] += math.floor(random.gauss(0, sigma))

        # generate dark counts
        for i in range(math.floor(n * self.dark_count_rate / self.lambd)):
            self.timestamps_signal.append(math.floor(random.random() * self.total_time))
            self.timestamps_idler.append(math.floor(random.random() * self.total_time))
        self.timestamps_signal.sort()
        self.timestamps_idler.sort()

        # deadtime
        index = 0
        while index < len(self.timestamps_signal) - 1:
            if self.timestamps_signal[index + 1] - self.timestamps_signal[index] < self.deadtime:
                del self.timestamps_signal[index + 1]
            else:
                index += 1
        index = 0
        while index < len(self.timestamps_idler) - 1:
            if self.timestamps_idler[index + 1] - self.timestamps_idler[index] < self.deadtime:
                del self.timestamps_idler[index + 1]
            else:
                index += 1

        # count coincidences

        # make sure that | lag | < 2 * dt_range
        dt_range = 200000      # in picoseconds

        s_floor = np.int64(np.floor(np.array(self.timestamps_signal) / (dt_range / 2)))
        i_floor = np.int64(np.floor(np.array(self.timestamps_idler) / (dt_range / 2)))
        coinc0 = np.intersect1d(s_floor, i_floor, return_indices=True)
        coinc1 = np.intersect1d(s_floor, i_floor - 1, return_indices=True)
        coinc2 = np.intersect1d(s_floor, i_floor + 1, return_indices=True)
        self.coinc = np.hstack((coinc0, coinc1, coinc2))

        return self.coinc.shape[1]

    def plot_cross_corr(self):
        x_val = range(-100000, 100000, self.coincidence_interval)
        y_val = []
        if len(self.timestamps_signal) > len(self.timestamps_idler):
            list2 = self.timestamps_signal
            list1 = self.timestamps_idler
        else:
            list1 = self.timestamps_signal
            list2 = self.timestamps_idler

        for delta_t in x_val:
            coincidences = []
            # index in list2 of the left bound
            left_bound = 0
            for x in list1:
                # check interval (x + delta_t - self.coincidence_interval, x + delta_t + self.coincidence_interval)
                while list2[left_bound] < (x + delta_t) - self.coincidence_interval and left_bound < len(list2) - 1:
                    left_bound += 1
                # now x + delta_t - self.coincidence_interval <= larger[left_bound]
                if list2[left_bound] < (x + delta_t) + self.coincidence_interval:
                    # x + delta_t and larger[left_bound] are in the same window
                    # the timestamp of the coincidence is the time of the later event
                    coincidences.append(max(x + delta_t, list2[left_bound]))
            # print(len(coincidences))
            y_val.append(len(coincidences))
        plt.plot(x_val, y_val)
        plt.xlabel('Time difference (ps)')
        plt.ylabel('Counts')
        plt.savefig('plots\\cross_correlation_plot.png', dpi=1000)
        plt.show()


if __name__ == "__main__":
    a = Simulator(1, 'config.yaml')
    print(a.run())
    a.plot_cross_corr()
