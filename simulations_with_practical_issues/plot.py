from Rotation import Rotation
from Simulator import Simulator
import numpy as np
import math
import matplotlib.pyplot as plt
from alive_progress import alive_bar


states = ['H', 'D', 'V', 'A']


def plot_g2_darkcounts(yaml_fn, dc, colors=None):
    """
    Plots semi-transparent cross-correlation histograms with varying dark count rates overtop of each other, and saves
    the plot with the parameters in the title. If specific colors are not provided, the default Tableau colors are used.

    :param str yaml_fn: file path for the config file
    :param list[int] dc: list of varying dark count rates for the histograms (counts/s)
    :param list[str] colors: optional list of colors for the histograms (counts/s)
    """
    with alive_bar(len(dc), force_tty=True) as bar:

        sim = Simulator(yaml_fn)
        max_max = 0

        for i in range(len(dc)):

            sim.dark_count_rate = dc[i]
            sim.generate_timestamps()
            sim.cross_corr()
            if sim.bins is not None:

                if colors is None:
                    plt.hist(sim.dtime, sim.bins, alpha=0.5, label=dc[i])
                else:
                    plt.hist(sim.dtime, sim.bins, alpha=0.5, label=dc[i], color=colors[i])

                if sim.max_counts > max_max:
                    max_max = sim.max_counts

            bar()

    plt.xlabel('Time difference (ps)')
    plt.ylabel('Counts')
    plt.legend(title='Dark counts per second')
    plt.yscale('log')
    plt.ylim(0.5, 10**math.ceil(math.log10(max_max)))

    title = 'plots\\g2\\g2_vs_darkcounts\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'l=' + str((sim.loss_idler + sim.loss_signal) / 2) + ',' + \
            'dt=' + str(sim.dead_time) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000, bbox_inches='tight')
    plt.show()


def plot_g2_deadtime(yaml_fn, deadtime, colors=None):
    """
    Plots semi-transparent cross-correlation histograms with varying dead times overtop of each other, and saves the
    plot with the parameters in the title. If specific colors are not provided, the default Tableau colors are used.

    :param str yaml_fn: file path for the config file
    :param list[int] deadtime: list of varying dead times for the histograms (ps)
    :param list[str] colors: optional list of colors for the histograms
    """
    with alive_bar(len(deadtime), force_tty=True) as bar:

        max_max = 0

        for i in range(len(deadtime)):

            sim = Simulator(yaml_fn)
            sim.dead_time = deadtime[i]
            sim.generate_timestamps()
            sim.cross_corr()

            if sim.bins is not None:

                if colors is None:
                    plt.hist(sim.dtime, sim.bins, alpha=0.5, label=deadtime[i])
                else:
                    plt.hist(sim.dtime, sim.bins, alpha=0.5, label=deadtime[i], color=colors[i])

            if sim.max_counts > max_max:
                max_max = sim.max_counts

            bar()

    plt.xlabel('Time difference (ps)')
    plt.ylabel('Counts')
    plt.legend(title='Dead time (ps)')
    plt.yscale('log')
    plt.ylim(0.5, 10**math.ceil(math.log10(sim.max_counts)))

    title = 'plots\\g2\\g2_vs_deadtime\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'l=' + str((sim.loss_idler + sim.loss_signal) / 2) + ',' + \
            'dc=' + str(sim.dark_count_rate) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000, bbox_inches='tight')
    plt.show()


def plot_g2_jitter(yaml_fn, jitter, colors=None, fwhm=False):
    """
    Plots semi-transparent cross-correlation histograms with varying jitter full width at half maximums overtop of each
    other, and saves the plot with the parameters in the title. There is an option to include the FWHM calculations and
    annotations on the cross-correlation histograms. If specific colors are not provided, the default Tableau colors are
    used.

    :param str yaml_fn: file path for the config file
    :param list[int] jitter: list of varying jitter FWHMs for the histograms (ps)
    :param list[str] colors: optional list of colors for the histograms
    :param bool fwhm: if the FWHMs will be annotated on the plot
    """
    with alive_bar(len(jitter), force_tty=True) as bar:

        fig, ax = plt.subplots()

        for i in range(len(jitter)):

            sim = Simulator(yaml_fn)
            sim.jitter_fwhm = jitter[i]
            sim.generate_timestamps()
            sim.cross_corr()

            if sim.bins is not None:

                if colors is None:
                    plt.hist(sim.dtime, sim.bins, alpha=0.5, label=jitter[i])
                else:
                    plt.hist(sim.dtime, sim.bins, alpha=0.5, label=jitter[i], color=colors[i])

                if fwhm:
                    half_max = sim.max_counts/2
                    max_i = np.argmax(sim.histo)
                    l_lim_i = max_i
                    r_lim_i = max_i

                    while sim.histo[l_lim_i] > half_max:
                        l_lim_i -= 1
                    while sim.histo[r_lim_i] > half_max:
                        r_lim_i += 1

                    l_lim = sim.bins[l_lim_i + 1]
                    r_lim = sim.bins[r_lim_i]

                    ax.annotate('',
                                xy=(l_lim - 400, half_max), xycoords='data',
                                xytext=(r_lim + 400, half_max), textcoords='data',
                                arrowprops=dict(arrowstyle="<->",
                                                connectionstyle="arc3", color=colors[i], lw=1),
                                )
                    plt.text(r_lim, half_max, str(r_lim - l_lim), color=colors[i])
            bar()

    plt.xlabel('Time difference (ps)')
    plt.ylabel('Counts')
    plt.legend(title='Jitter FWHM')
    plt.xlim(-20000, 20000)

    if fwhm:
        fwhm_str = '_fwhm'
    else:
        fwhm_str = ''

    title = 'plots\\g2\\g2_vs_jitter\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'l=' + str((sim.loss_idler + sim.loss_signal) / 2) + ',' + \
            'dc=' + str(sim.dark_count_rate) + ',' + \
            'dt=' + str(sim.dead_time) + ',' + \
            'ci=' + str(sim.coincidence_interval) + fwhm_str + '.png'

    plt.savefig(title, dpi=1000, bbox_inches='tight')
    plt.show()


def plot_g2_loss(yaml_fn, loss, colors=None):
    """
    Plots semi-transparent cross-correlation histograms with varying optical loss overtop of each other, and saves the
    plot with the parameters in the title. If specific colors are not provided, the default Tableau colors are used.

    :param str yaml_fn: file path for the config file
    :param list[float] loss: list of varying optical loss for the histograms (dB)
    :param list[str] colors: optional list of colors for the histograms
    """
    with alive_bar(len(loss), force_tty=True) as bar:

        max_max = 0

        for i in range(len(loss)):

            sim = Simulator(yaml_fn)
            loss_pr = 1 - 10 ** (-loss[i]/10)
            sim.loss_signal = loss_pr
            sim.loss_idler = loss_pr
            sim.generate_timestamps()
            sim.cross_corr()

            if sim.bins[0] is not None:

                if colors is None:

                    plt.hist(sim.dtime, sim.bins, alpha=0.5, label=loss[i])
                else:
                    plt.hist(sim.dtime, sim.bins, alpha=0.5, label=loss[i], color=colors[i])

            if sim.max_counts > max_max:
                max_max = sim.max_counts

            bar()

    plt.xlabel('Time difference (ps)')
    plt.ylabel('Counts')
    plt.xlim(-10000, 10000)
    plt.legend(title='Optical Loss (dB)')
    plt.yscale('log')
    plt.ylim(0.5, 10**math.ceil(math.log10(max_max)))

    title = 'plots\\g2\\g2_vs_loss\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'dc=' + str(sim.dark_count_rate) + ',' + \
            'dt=' + str(sim.dead_time) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000, bbox_inches='tight')
    plt.show()


def plot_car_darkcounts(yaml_fn, start, stop, num_points):
    """
    Plots the coincidence to accidental ratio (CAR) versus the dark count rate, and saves the plot with the parameters
    in the title.

    :param str yaml_fn: file path for the config file
    :param int start: starting dark count rate (counts/s)
    :param int stop: stopping dark count rate (counts/s)
    :param int num_points: number of points to plot
    """
    x_val = np.linspace(start, stop, num_points)
    y_val = []

    with alive_bar(num_points, force_tty=True) as bar:

        sim = Simulator(yaml_fn)

        for dc in x_val:
            sim.dark_count_rate = dc
            sim.generate_timestamps()
            sim.cross_corr()
            y_val.append(sim.car)
            bar()

    plt.plot(x_val, y_val)
    plt.xlabel('Dark Count Rate (counts/second)')
    plt.ylabel('Coincidence-to-Accidental Rate')

    title = 'plots\\car\\car_vs_darkcounts\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'l=' + str((sim.loss_idler + sim.loss_signal) / 2) + ',' + \
            'dt=' + str(sim.dead_time) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000, bbox_inches='tight')
    plt.show()


def plot_car_deadtime(yaml_fn, start, stop, num_points):
    """
    Plots the coincidence to accidental ratio (CAR) versus the dead time, and saves the plot with the parameters in the
    title.

    :param str yaml_fn: file path for the config file
    :param int start: starting dead time (ps)
    :param int stop: stopping dead time (ps)
    :param int num_points: number of points to plot
    """
    x_val = np.linspace(start, stop, num_points)
    y_val = []

    with alive_bar(num_points, force_tty=True) as bar:

        for deadtime in x_val:
            sim = Simulator(yaml_fn)
            sim.dead_time = deadtime
            sim.generate_timestamps()
            sim.cross_corr()
            y_val.append(sim.car)
            bar()

    plt.plot(x_val, y_val)
    plt.xlabel('Dead Time (ps)')
    plt.ylabel('Coincidence-to-Accidental Rate')

    title = 'plots\\car\\car_vs_deadtime\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'l=' + str((sim.loss_idler + sim.loss_signal) / 2) + ',' + \
            'dc=' + str(sim.dark_count_rate) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000, bbox_inches='tight')
    plt.show()


def plot_car_jitter(yaml_fn, start, stop, num_points):
    """
    Plots the coincidence to accidental ratio (CAR) versus the jitter full width at half maximum, and saves the plot
    with the parameters in the title.

    :param str yaml_fn: file path for the config file
    :param int start: starting jitter fwhm (ps)
    :param int stop: stopping jitter fwhm (ps)
    :param int num_points: number of points to plot
    """
    x_val = np.linspace(start, stop, num_points)
    y_val = []

    with alive_bar(num_points, force_tty=True) as bar:

        for jitter in x_val:
            sim = Simulator(yaml_fn)
            sim.jitter = jitter
            sim.generate_timestamps()
            sim.cross_corr()
            y_val.append(sim.car)
            bar()

    plt.plot(x_val, y_val)
    plt.xlabel('Jitter FWHM (ps)')
    plt.ylabel('Coincidence-to-Accidental Rate')

    title = 'plots\\car\\car_vs_jitter\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'l=' + str((sim.loss_idler + sim.loss_signal) / 2) + ',' + \
            'dc=' + str(sim.dark_count_rate) + ',' + \
            'dt=' + str(sim.dead_time) + ',' + \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000, bbox_inches='tight')
    plt.show()


def plot_car_loss(yaml_fn, start, stop, num_points):
    """
    Plots the coincidence to accidental ratio (CAR) versus the optical loss, and saves the plot with the parameters in
    the title.

    :param str yaml_fn: file path for the config file
    :param float start: starting loss (dB)
    :param float stop: stopping loss (dB)
    :param int num_points: number of points to plot
    """
    x_val = 1 - 10**(-np.linspace(start, stop, num_points)/10)
    y_val = []

    with alive_bar(num_points, force_tty=True) as bar:

        for loss in x_val:
            sim = Simulator(yaml_fn)
            sim.loss_signal = loss
            sim.loss_idler = loss
            sim.generate_timestamps()
            sim.cross_corr()
            y_val.append(sim.car)
            bar()

    plt.plot(np.linspace(start, stop, num_points), y_val)
    plt.xlabel('Optical Loss (dB)')
    plt.ylabel('Coincidence-to-Accidental Rate')

    title = 'plots\\car\\car_vs_loss\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'dc=' + str(sim.dark_count_rate) + ',' + \
            'dt=' + str(sim.dead_time) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000, bbox_inches='tight')
    plt.show()


def plot_visibility_darkcounts(yaml_fn, state, start, stop, num_points, rotations=50):
    """
    Plots the entanglement visibility versus the dark count rate for each of the 4 bases (H, V, D, A), and saves the
    plot with the parameters in the title. The default number of rotation steps is 50.

    :param str yaml_fn: file path for the config file
    :param state: quantum state of the photons
    :type state: (complex, complex, complex, complex)
    :param int start: starting dark count rate (counts/s)
    :param int stop: stopping dark count rate (counts/s)
    :param int num_points: number of points to plot
    :param int rotations: number of rotation steps for the polarizer
    """
    x_val = np.linspace(start, stop, num_points)
    y_val = [[], [], [], []]

    with alive_bar(num_points, force_tty=True) as bar:

        for dc in x_val:
            sim = Rotation(yaml_fn, state, rotations)
            sim.dark_count_rate = dc
            sim.run()
            for i in range(4):
                y_val[i].append(sim.visibility[i])
            bar()

    for i in range(4):
        plt.plot(x_val, y_val[i], label=states[i])
    plt.legend()
    plt.xlabel('Dark Count Rate (counts/second)')
    plt.ylabel('Visibility')
    plt.ylim(-0.01, 1.01)

    title = 'plots\\visibility\\visibility_vs_darkcounts\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'l=' + str((sim.loss_idler + sim.loss_signal) / 2) + ',' + \
            'dt=' + str(sim.dead_time) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000)
    plt.show()


def plot_visibility_deadtime(yaml_fn, state, start, stop, num_points, rotations=50):
    """
    Plots the entanglement visibility versus the dead time for each of the 4 bases (H, V, D, A), and saves the plot with
    the parameters in the title. The default number of rotation steps is 50.

    :param str yaml_fn: file path for the config file
    :param state: quantum state of the photons
    :type state: (complex, complex, complex, complex)
    :param int start: starting dead time (ps)
    :param int stop: stopping dead time (ps)
    :param int num_points: number of points to plot
    :param int rotations: number of rotation steps for the polarizer
    """
    x_val = np.linspace(start, stop, num_points)
    y_val = [[], [], [], []]

    with alive_bar(num_points, force_tty=True) as bar:

        for deadtime in x_val:
            sim = Rotation(yaml_fn, state, rotations)
            sim.dead_time = deadtime
            sim.run()
            for i in range(4):
                y_val[i].append(sim.visibility[i])
            bar()

    for i in range(4):
        plt.plot(x_val, y_val[i], label=states[i])
    plt.legend()
    plt.xlabel('Dead Time (ps)')
    plt.ylabel('Visibility')
    plt.ylim(-0.01, 1.01)

    title = 'plots\\visibility\\visibility_vs_deadtime\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'l=' + str((sim.loss_idler + sim.loss_signal) / 2) + ',' + \
            'dc=' + str(sim.dark_count_rate) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000)
    plt.show()


def plot_visibility_jitter(yaml_fn, state, start, stop, num_points, rotations=50):
    """
    Plots the entanglement visibility versus the jitter full width at half maximum for each of the 4 bases (H, V, D, A),
    and saves the plot with the parameters in the title. The default number of rotation steps is 50.

    :param str yaml_fn: file path for the config file
    :param state: quantum state of the photons
    :type state: (complex, complex, complex, complex)
    :param int start: starting jitter fwhm (ps)
    :param int stop: stopping jitter fwhm (ps)
    :param int num_points: number of points to plot
    :param int rotations: number of rotation steps for the polarizer
    """
    x_val = np.linspace(start, stop, num_points)
    y_val = [[], [], [], []]

    with alive_bar(num_points, force_tty=True) as bar:

        for jitter in x_val:
            sim = Rotation(yaml_fn, state, rotations)
            sim.jitter = jitter
            sim.run()
            for i in range(4):
                y_val[i].append(sim.visibility[i])
            bar()

    for i in range(4):
        plt.plot(x_val, y_val[i], label=states[i])
    plt.legend()
    plt.xlabel('Jitter FWHM (ps)')
    plt.ylabel('Visibility')
    plt.ylim(-0.01, 1.01)

    title = 'plots\\visibility\\visibility_vs_jitter\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'l=' + str((sim.loss_idler + sim.loss_signal) / 2) + ',' + \
            'dc=' + str(sim.dark_count_rate) + ',' + \
            'dt=' + str(sim.dead_time) + ',' + \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000)
    plt.show()


def plot_visibility_loss(yaml_fn, state, start, stop, num_points, rotations=50):
    """
    Plots the entanglement visibility versus the optical loss for each of the 4 bases (H, V, D, A), and saves the plot
    with the parameters in the title. The default number of rotation steps is 50.

    :param str yaml_fn: file path for the config file
    :param state: quantum state of the photons
    :type state: (complex, complex, complex, complex)
    :param float start: starting loss (dB)
    :param float stop: stopping loss (dB)
    :param int num_points: number of points to plot
    :param int rotations: number of rotation steps for the polarizer
    """
    x_val = 1 - 10**(-np.linspace(start, stop, num_points)/10)
    y_val = [[], [], [], []]

    with alive_bar(num_points, force_tty=True) as bar:

        for loss in x_val:
            sim = Rotation(yaml_fn, state, rotations)
            sim.loss_signal = loss
            sim.loss_idler = loss
            sim.run()
            for i in range(4):
                y_val[i].append(sim.visibility[i])
            bar()

    for i in range(4):
        plt.plot(np.linspace(start, stop, num_points), y_val[i], label=states[i])
    plt.legend()
    plt.xlabel('Optical Loss (dB)')
    plt.ylabel('Visibility')
    plt.ylim(-0.01, 1.01)

    title = 'plots\\visibility\\visibility_vs_loss\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'dc=' + str(sim.dark_count_rate) + ',' + \
            'dt=' + str(sim.dead_time) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000)
    plt.show()


def plot_visibility_phasediff(yaml_fn, start, stop, num_points, rotations=50):
    """
    Plots the entanglement visibility versus the phase difference in the entangled state for each of the 4 bases
    (H, V, D, A), and saves the plot with the parameters in the title. The default number of rotation steps is 50.

    :param str yaml_fn: file path for the config file
    :param float start: starting phase difference (radians)
    :param float stop: stopping phase difference (radians)
    :param int num_points: number of points to plot
    :param int rotations: number of rotation steps for the polarizer
    """
    x_val = np.linspace(start, stop, num_points)
    y_val = [[], [], [], []]

    with alive_bar(num_points, force_tty=True) as bar:

        for delta in x_val:
            entangled_state = [1 / math.sqrt(2), 0, 0, np.exp(1j * delta) / math.sqrt(2)]
            sim = Rotation(yaml_fn, entangled_state, rotations)
            sim.run()
            for i in range(4):
                y_val[i].append(sim.visibility[i])
            bar()

    for i in range(4):
        plt.plot(x_val, y_val[i], label=states[i])
    plt.legend()
    plt.xlabel('Phase Difference in the Entangled State')
    plt.ylabel('Visibility')
    plt.ylim(-0.01, 1.01)

    num_ticks = math.floor((stop + 0.001)/math.pi)
    ticks = [0]
    labels = ['0']

    for i in range(1, num_ticks):
        ticks.append(i * math.pi)
        labels.append(str(i) + 'π')

    plt.xticks(ticks, labels)

    title = 'plots\\visibility\\visibility_vs_deadtime\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'l=' + str((sim.loss_idler + sim.loss_signal) / 2) + ',' + \
            'dc=' + str(sim.dark_count_rate) + ',' + \
            'dt=' + str(sim.dead_time) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000)
    plt.show()


if __name__ == '__main__':

    # plot_g2_darkcounts('config.yaml', [0, 100, 1000, 100000], ['C1', 'C8', 'C2', 'C0'])
    # plot_g2_deadtime('config.yaml', [0, 25000, 50000, 75000], ['C1', 'C8', 'C2', 'C0'])
    # plot_g2_jitter('config.yaml', [0, 5000, 10000, 20000, 50000], ['C3', 'C1', 'C2', 'C0', 'C4'], fwhm=True)
    plot_g2_loss('config.yaml', [0, 1, 3, 6, 10], ['C3', 'C1', 'C8', 'C2', 'C0', 'C4'])

    # plot_car_darkcounts('config.yaml', 0, 10000, 256)
    # plot_car_deadtime('config.yaml', 0, 1000000, 64)
    # plot_car_jitter('config.yaml', 0, 8000, 4096)
    # plot_car_loss('config.yaml', 0, 30, 256)

    d = .25
    qubit_state = [1 / math.sqrt(2), 0, 0, np.exp(1j * d) / math.sqrt(2)]

    # plot_visibility_darkcounts('config.yaml', qubit_state, 0, 1000000, 4)
    # plot_visibility_deadtime('config.yaml', qubit_state, 0, 1000000, 8)
    # plot_visibility_jitter('config.yaml', qubit_state, 0, 100000, 8)
    # plot_visibility_loss('config.yaml', qubit_state, 0, 30, 8)
    # plot_visibility_phasediff('config.yaml', 0, 4*math.pi, 64)
