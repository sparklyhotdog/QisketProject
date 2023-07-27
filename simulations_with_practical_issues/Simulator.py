import matplotlib.pyplot as plt
import math
import random
import numpy as np
import yaml


def modify_timestamps(in_timestamp_fn, out_timestamp_fn, yaml_fn):
    """
    Modifies a set of timestamps to simulate extending the fiber cable.

    This function takes in a set of timetags stored in text files and simulates extending the fiber cable by adding
    loss, jitter, dark counts/straylight, and dead time. It saves the resulting timetags in two new text files.

    :param (str, str) in_timestamp_fn: file paths of the input signal and idler timestamps
    :param (str, str) out_timestamp_fn: file paths of the output signal and idler timestamps
    :param str yaml_fn: file path for the config file associated with the fiber cable extension
    """

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
    """
    This class simulates counting pairs of entangled photons, factoring in imperfect entanglement states from the photon
    source, optical loss and dispersion in fibers, as well as dark counts and timing jitter in single photon detectors.
    The specifications should be stored in a YAML configuration file.

    It can also simulate a measurement if a probability is provided.
    """

    def __init__(self, yaml_fn, pr=1, range_ps=200000):
        """
        The constructor for the Simulator class.

        :param str yaml_fn: path for the config file
        :param float pr: probability associated with the 00 state
        :param int range_ps: range for time delays checked in the cross-correlation (ps)
        """

        self.yaml_fn = yaml_fn
        y_fn = open(self.yaml_fn, 'r')
        self.dicty = yaml.load(y_fn, Loader=yaml.SafeLoader)
        y_fn.close()

        self.pr = pr
        self.range_ps = range_ps
        self.lambd = self.dicty['lambd']
        self.total_time = self.dicty['total_time']
        self.lag = self.dicty['lag']
        self.loss_signal = self.dicty['loss_signal']
        self.loss_idler = self.dicty['loss_idler']
        self.dark_counts = self.dicty['dark_counts']
        self.ambient_light = self.dicty['ambient_light']
        self.dead_time = self.dicty['dead_time']
        self.jitter_fwhm = self.dicty['jitter_fwhm']
        self.coinc_interval = self.dicty['coinc_interval']

        self.timestamps_signal = []
        self.timestamps_idler = []

        self.max_counts = 0         # count result from the cross-correlation
        self.car = None             # coincidence-to-accidental ratio
        self.cps = None             # coincidences per second
        self.aps = None             # accidentals per second
        self.histo = None           # binned data for the cross-correlation histogram
        self.dtime = None           # unbinned data for the cross-correlation histogram
        self.bins = None            # bins for the cross-correlation histogram

    def generate_timestamps(self, file_names=None):
        """
        Generates timestamps based on the specifications in the config file.

        This method factors in optical loss, jitter, dark counts, and dead time as specified in the config file when
        generating the timestamps, and stores them in self.timestamps_signal and self.timestamps_idler. If a set of
        file names is provided, then timestamps are also saved in a pair of text files.

        :param (str, str) file_names: optional file paths to store the timestamps in
        """

        # generate pseudo timestamps for entangled pairs following an exponential distribution
        t = 0
        for i in range(math.floor(self.total_time * self.lambd)):
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

        # generate ambient light
        for i in range(math.floor(self.ambient_light * self.total_time)):
            # factor in loss
            if random.random() > self.loss_signal:
                self.timestamps_signal.append(math.floor(random.random() * self.total_time * 1e12))
            if random.random() > self.loss_idler:
                self.timestamps_idler.append(math.floor(random.random() * self.total_time * 1e12))

        # generate dark counts
        for i in range(math.floor(self.dark_counts * self.total_time)):
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
        """
        Cross-correlates the timestamp data sets.

        This method performs a cross-correlation between the timestamp data sets, and gives an estimation of the
        coincidence and accidental rate (self.cps and self.aps), as well as the coincidence to accidental ratio
        (self.car). It checks the time delay in the interval (-self.range_ps/2, self.range_ps/2). If no timestamp file
        names are provided, the self.timestamps_signal and self.timestamps_idler class variables are used instead.

        :param (str, str) timestamp_fn: optional file paths of the signal and idler timestamps to be cross-correlated
        """

        if timestamp_fn is None:
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

            s_floor = np.int64(np.floor(np.array(signal) / (self.range_ps / 2)))
            i_floor = np.int64(np.floor(np.array(idler) / (self.range_ps / 2)))
            coinc0 = np.intersect1d(s_floor, i_floor, return_indices=True)
            coinc1 = np.intersect1d(s_floor, i_floor - 1, return_indices=True)
            coinc2 = np.intersect1d(s_floor, i_floor + 1, return_indices=True)
            coinc = np.hstack((coinc0, coinc1, coinc2))

            s_time = np.array(signal)[coinc[1]]
            i_time = np.array(idler)[coinc[2]]
            self.dtime = s_time - i_time

            # iterate over coincidence_interval, find max of the max(histo)'s
            num_steps = 2           # the number of dt's checked for the max
            for dt in np.arange(0, self.coinc_interval, self.coinc_interval / num_steps):
                bins = np.arange(-self.range_ps / 2 + dt, self.range_ps / 2 + self.coinc_interval, self.coinc_interval)
                histo = np.histogram(self.dtime, bins)[0]
                curr_count = max(histo)
                if curr_count > self.max_counts:
                    self.max_counts = curr_count
                    self.bins = bins
                    self.histo = histo
                else:
                    break

            if self.histo is not None:
                # calculates the mean of the interval [acc_1, acc_2) for the accidentals
                acc_1 = 0
                acc_2 = math.floor(0.1 * len(self.histo))
                accidentals = np.split(self.histo, [acc_1, acc_2])[1]

                if np.mean(accidentals) > 0:
                    self.car = self.max_counts / np.mean(accidentals)

                self.cps = self.max_counts / self.total_time
                self.aps = (coinc.shape[1] - self.max_counts) / self.total_time

    def plot_cross_corr(self, path=None):
        """
        Plots the cross-correlation histogram.

        This method displays the cross-correlation histogram. If a path is provided, the figure is also saved there.

        :param str path: optional file path to save the plot in
        """
        plt.hist(self.dtime, self.bins)
        plt.xlabel('Time difference (ps)')
        plt.ylabel('Counts')
        plt.yscale('log')
        plt.ylim(0.5, 10 ** math.ceil(math.log10(self.max_counts) + 0.5))
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
    # a.plot_cross_corr('plots\\cross_correlation_plot.png')
    a.plot_cross_corr()


#
#                ////==   ===\\\
#          _//---__ 0      0   \\
#         //////// //_ {_} \\\ \\\\\
#       ///// / / ///// \\\\\\ \\\ \\\
#         __|  ______       ----\   \\
#        / ___/__-___ \==__    |
#       /_________-- \--__     |
#         |                   |
#         |                   |
#          |                  |
#            |               |
#               |           |
#                  |     |
#                __===|| |=__
