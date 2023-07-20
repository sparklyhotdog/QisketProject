import matplotlib.pyplot as plt
import math
import random
import numpy as np
import yaml


def modify_timestamps(in_timestamp_fn, out_timestamp_fn, yaml_fn):

    # load in timestamps
    signal = []
    idler = []
    with open(in_timestamp_fn[0]) as s:
        for line in s:
            signal.append(int(line))

    with open(in_timestamp_fn[1]) as i:
        for line in i:
            idler.append(int(line))

    y_fn = open(yaml_fn, 'r')
    dicty = yaml.load(y_fn, Loader=yaml.SafeLoader)
    y_fn.close()

    loss_signal = dicty['loss_signal']
    loss_idler = dicty['loss_idler']
    dark_count_rate = dicty['dark_count_rate']
    dead_time = dicty['dead_time']
    jitter_fwhm = dicty['jitter_fwhm']

    # optical loss
    for x in signal:
        if random.random() < loss_signal:
            signal.remove(x)
    for x in idler:
        if random.random() < loss_idler:
            idler.remove(x)

    # jitter
    sigma = jitter_fwhm / (2 * math.sqrt(2 * math.log(2)))
    for i in range(len(signal)):
        signal[i] += math.floor(random.gauss(0, sigma))
    for i in range(len(idler)):
        idler[i] += math.floor(random.gauss(0, sigma))

    # dark counts
    time = math.floor((signal[-1] + idler[-1])/2)
    for i in range(math.floor(time * dark_count_rate / 1e12)):
        signal.append(math.floor(random.random() * time))
        idler.append(math.floor(random.random() * time))
    signal.sort()
    idler.sort()

    # dead time
    index = 0
    while index < len(signal) - 1:
        if signal[index + 1] - signal[index] < dead_time:
            del signal[index + 1]
        else:
            index += 1
    index = 0
    while index < len(idler) - 1:
        if idler[index + 1] - idler[index] < dead_time:
            del idler[index + 1]
        else:
            index += 1

    # store new timestamps in text files
    with open(out_timestamp_fn[0], 'w') as s:
        for timestamp in signal:
            print(timestamp, file=s)
    with open(out_timestamp_fn[1], 'w') as i:
        for timestamp in idler:
            print(timestamp, file=i)


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
        self.loss_signal = self.dicty['loss_signal']
        self.loss_idler = self.dicty['loss_idler']
        self.dark_count_rate = self.dicty['dark_count_rate']
        self.dead_time = self.dicty['dead_time']
        self.jitter_fwhm = self.dicty['jitter_fwhm']
        self.coincidence_interval = self.dicty['coincidence_interval']

        self.timestamps_signal = []
        self.timestamps_idler = []

        self.max_counts = 0         # count result from the cross-correlation
        self.car = None             # coincidence-to-accidental ratio
        self.cps = None             # coincidences per second
        self.aps = None             # accidentals per second
        self.histo = None           # unbinned data for the cross-correlation histogram
        self.dtime = None           # binned data for the cross-correlation histogram
        self.bins = None            # bins for the cross-correlation histogram
        self.accidentals = None     # array of the accidental timestamps
        self.coincidences = None    # array of the coincidence timestamps

    def generate_timestamps(self, file_names=None):
        # generate pseudo timestamps following an exponential distribution
        n = self.total_time * self.lambd                  # total number of events

        t = 0
        for i in range(math.floor(n)):
            dt = math.floor(random.expovariate(self.lambd) * 1e12)
            t += dt
            if random.random() < self.pr:
                # optical loss
                if random.random() > self.loss_signal:
                    self.timestamps_signal.append(t + self.lag)
                if random.random() > self.loss_idler:
                    self.timestamps_idler.append(t)

        # jitter
        sigma = self.jitter_fwhm / (2 * math.sqrt(2 * math.log(2)))
        if len(self.timestamps_signal) > 0:
            for i in range(len(self.timestamps_signal)):
                self.timestamps_signal[i] += math.floor(random.gauss(0, sigma))

        if len(self.timestamps_idler) > 0:
            for i in range(len(self.timestamps_idler)):
                self.timestamps_idler[i] += math.floor(random.gauss(0, sigma))

        # generate dark counts
        for i in range(math.floor(n * self.dark_count_rate / self.lambd)):
            self.timestamps_signal.append(math.floor(random.random() * self.total_time * 1e12))
            self.timestamps_idler.append(math.floor(random.random() * self.total_time * 1e12))
        self.timestamps_signal.sort()
        self.timestamps_idler.sort()

        # dead time
        index = 0
        while index < len(self.timestamps_signal) - 1:
            if self.timestamps_signal[index + 1] - self.timestamps_signal[index] < self.dead_time:
                del self.timestamps_signal[index + 1]
            else:
                index += 1
        index = 0
        while index < len(self.timestamps_idler) - 1:
            if self.timestamps_idler[index + 1] - self.timestamps_idler[index] < self.dead_time:
                del self.timestamps_idler[index + 1]
            else:
                index += 1

        # if file names are provided, store timestamps in text files
        if file_names is not None:
            with open(file_names[0], 'w') as s:
                for timestamp in self.timestamps_signal:
                    print(timestamp, file=s)
            with open(file_names[1], 'w') as i:
                for timestamp in self.timestamps_idler:
                    print(timestamp, file=i)

    def cross_corr(self, timestamp_fn=None):

        if timestamp_fn is None:
            # if no timestamp file names are provided, use the class variables
            signal = self.timestamps_signal
            idler = self.timestamps_idler
        else:
            # load timestamps from the files
            signal = []
            idler = []
            with open(timestamp_fn[0]) as s:
                for line in s:
                    signal.append(int(line))

            with open(timestamp_fn[1]) as i:
                for line in i:
                    idler.append(int(line))

        # count coincidences
        if len(signal) > 0 and len(idler) > 0:
            range_ps = 200000           # checks the time difference for (-range_ps/2, range_ps/2)

            s_floor = np.int64(np.floor(np.array(signal) / (range_ps / 2)))
            i_floor = np.int64(np.floor(np.array(idler) / (range_ps / 2)))
            coinc0 = np.intersect1d(s_floor, i_floor, return_indices=True)
            coinc1 = np.intersect1d(s_floor, i_floor - 1, return_indices=True)
            coinc2 = np.intersect1d(s_floor, i_floor + 1, return_indices=True)
            coinc = np.hstack((coinc0, coinc1, coinc2))

            s_time = np.array(signal)[coinc[1]]
            i_time = np.array(idler)[coinc[2]]
            self.dtime = s_time - i_time

            # iterate over coincidence_interval, find max of the max(histo)'s
            num_steps = 2           # the number of dt's checked for the max
            for dt in np.arange(0, self.coincidence_interval, self.coincidence_interval/num_steps):
                bins = np.arange(-range_ps/2 + dt, range_ps/2 + self.coincidence_interval, self.coincidence_interval)
                histo = np.histogram(self.dtime, bins)[0]
                curr_count = max(histo)
                if curr_count > self.max_counts:
                    self.max_counts = curr_count
                    self.bins = bins
                    self.histo = histo
                else:
                    break

            # TODO: fix CAR calculation for edge cases (index out of bounds)
            # ___________________________________________________________________
            # seperate coincidences and accidentals
            epsilon = 10             # the difference in means allowed amoung the accidentals
            max_i = np.argmax(self.histo)
            i = 0
            interval = 5
            prev = np.split(self.histo, [max_i - interval, max_i])[1]
            curr = np.split(self.histo, [max_i - 1 - interval, max_i - 1])[1]
            while abs(np.mean(prev) - np.mean(curr)) > epsilon:
                i += 1
                prev = np.split(self.histo, [max_i - interval - i, max_i - i])[1]
                curr = np.split(self.histo, [max_i - 1 - interval - i, max_i - 1 - i])[1]

            self.accidentals = np.delete(self.histo, range(max_i - i, max_i + i + 1))
            self.coincidences = np.split(self.histo, [max_i - i, max_i + i + 1])[1]

            if np.mean(self.accidentals) > 0:
                self.car = self.max_counts / np.mean(self.accidentals)

            t_signal = (signal[-1] - signal[0]) / 1e12
            t_idler = (idler[-1] - idler[0]) / 1e12
            self.cps = 2 * sum(self.coincidences) / (t_signal + t_idler)
            self.aps = 2 * sum(self.accidentals) / (t_signal + t_idler)

    def plot_cross_corr(self, path=None):
        plt.hist(self.dtime, self.bins)
        plt.xlabel('Time difference (ps)')
        plt.ylabel('Counts')
        # plt.yscale('log')
        plt.ylim(0.5, 10 ** math.ceil(math.log10(self.max_counts)))
        if path is not None:
            plt.savefig(path, dpi=1000, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # modify_timestamps(('timestamps_signal.txt', 'timestamps_idler.txt'),
    #                   ('timestamps_signal1.txt', 'timestamps_idler1.txt'), 'config1.yaml')

    a = Simulator('config.yaml')
    a.generate_timestamps(('timestamps_signal.txt', 'timestamps_idler.txt'))
    a.cross_corr()
    print('Coincidences: ' + str(a.max_counts))
    print('Coincidence-to-Accidental Ratio: ' + str(a.car))
    print('Coincidences per second: ' + str(a.cps))
    print('Accidentals per second: ' + str(a.aps))
    a.plot_cross_corr('plots\\cross_correlation_plot.png')
