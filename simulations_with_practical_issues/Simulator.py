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
        self.max_counts = 0
        # for the cross correlation plot
        self.histo = np.array
        self.dtime = np.array
        self.bins = np.array

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

        # ___________________________________________________________________
        # count coincidences

        range_ps = 200000

        s_floor = np.int64(np.floor(np.array(self.timestamps_signal) / (range_ps / 2)))
        i_floor = np.int64(np.floor(np.array(self.timestamps_idler) / (range_ps / 2)))
        coinc0 = np.intersect1d(s_floor, i_floor, return_indices=True)
        coinc1 = np.intersect1d(s_floor, i_floor - 1, return_indices=True)
        coinc2 = np.intersect1d(s_floor, i_floor + 1, return_indices=True)
        coinc = np.hstack((coinc0, coinc1, coinc2))

        s_time = np.array(self.timestamps_signal)[coinc[1]]
        i_time = np.array(self.timestamps_idler)[coinc[2]]
        self.dtime = s_time - i_time

        # iterate over coincidence_interval, find max of the max(histo)'s
        num_steps = 2           # the number of dt's checked for the max
        for dt in np.arange(0, self.coincidence_interval, self.coincidence_interval/num_steps):

            bins = np.arange(-range_ps / 2 + dt, range_ps / 2 + self.coincidence_interval, self.coincidence_interval)
            histo = np.histogram(self.dtime, bins)[0]
            curr_count = max(histo)
            if curr_count > self.max_counts:
                self.max_counts = curr_count
                self.bins = bins
                self.histo = histo
            else:
                break

    def get_coincidences(self):
        return self.max_counts

    def get_car(self):
        print(self.histo)
        i = int(np.nonzero(self.histo == self.max_counts)[0])
        accidentals = np.delete(self.histo, [i - 1, i, i + 1])
        return self.max_counts/np.mean(accidentals)

    def plot_cross_corr(self):
        plt.hist(self.dtime, self.bins)
        plt.xlabel('Time difference (ps)')
        plt.ylabel('Counts')
        plt.savefig('plots\\cross_correlation_plot.png', dpi=1000, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    a = Simulator(pr=1, yaml_fn='config.yaml')
    a.run()
    print(a.get_coincidences())
    print(a.get_car())
    a.plot_cross_corr()
