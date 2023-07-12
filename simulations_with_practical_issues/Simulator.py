import matplotlib.pyplot as plt
import math
import random
import numpy as np
import yaml


class Simulator:

    def __init__(self, yaml_fn, pr=1):
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
        self.histo = np.array
        self.dtime = np.array
        self.bins = np.array
        self.accidentals = np.array
        self.coincidences = np.array

    def run(self):
        # generate pseudo timestamps following an exponential distribution
        n = self.total_time * self.lambd                  # total number of events

        t = 0
        for i in range(math.floor(n)):
            dt = math.floor(random.expovariate(self.lambd) * 1e12)
            t += dt
            if random.random() < self.pr:
                # optical loss
                if random.random() > self.optical_loss_signal:
                    self.timestamps_signal.append(t + self.lag)
                if random.random() > self.optical_loss_idler:
                    self.timestamps_idler.append(t)

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

        range_ps = 200000           # checks the time difference for (-range_ps/2, range_ps/2)

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

        # ___________________________________________________________________
        # seperate coincidences and accidentals

        epsilon = 1             # the difference in means allowed amoung the accidentals
        max_i = np.argmax(self.histo)
        i = 0
        prev = np.mean(np.delete(self.histo, range(max_i - i, max_i + i + 1)))
        curr = np.mean(np.delete(self.histo, range(max_i - i - 1, max_i + i + 2)))
        while abs(prev - curr) > epsilon:
            i += 1
            prev = np.mean(np.delete(self.histo, range(max_i - i, max_i + i + 1)))
            curr = np.mean(np.delete(self.histo, range(max_i - i - 1, max_i + i + 2)))

        self.accidentals = np.delete(self.histo, range(max_i - i, max_i + i + 1))
        self.coincidences = np.split(self.histo, [max_i - i, max_i + i + 1])[1]

    def plot_cross_corr(self):
        plt.hist(self.dtime, self.bins)
        plt.xlabel('Time difference (ps)')
        plt.ylabel('Counts')
        plt.savefig('plots\\cross_correlation_plot.png', dpi=1000, bbox_inches='tight')
        plt.show()

    def get_car(self):
        if np.mean(self.accidentals) > 0:
            return self.max_counts/np.mean(self.accidentals)

    def get_coincidences(self):
        return self.max_counts

    def get_coincidences_per_sec(self):
        # total collection time for the signal and idler (in seconds)
        t_signal = (self.timestamps_signal[-1] - self.timestamps_signal[0])/1e12
        t_idler = (self.timestamps_idler[-1] - self.timestamps_idler[0])/1e12
        return 2 * sum(self.coincidences) / (t_signal + t_idler)

    def get_accidentals_per_sec(self):
        # total collection time for the signal and idler (in seconds)
        t_signal = (self.timestamps_signal[-1] - self.timestamps_signal[0]) / 1e12
        t_idler = (self.timestamps_idler[-1] - self.timestamps_idler[0]) / 1e12
        return 2 * sum(self.accidentals) / (t_signal + t_idler)

    def set_dc_rate(self, dc):
        self.dark_count_rate = dc

    def set_loss_signal(self, loss):
        self.optical_loss_signal = loss

    def set_loss_idler(self, loss):
        self.optical_loss_idler = loss

    def set_jitter(self, jitter):
        self.jitter_fwhm = jitter

    def set_deadtime(self, deadtime):
        self.deadtime = deadtime


if __name__ == "__main__":
    a = Simulator('config.yaml')
    a.run()
    print('Coincidences: ' + str(a.get_coincidences()))
    print('Coincidence-to-Accidental Ratio: ' + str(a.get_car()))
    print('Coincidences per second: ' + str(a.get_coincidences_per_sec()))
    print('Accidentals per second: ' + str(a.get_accidentals_per_sec()))
    a.plot_cross_corr()
