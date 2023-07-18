from Rotation import Rotation
from Simulator import Simulator
import numpy as np
import math
import matplotlib.pyplot as plt
from alive_progress import alive_bar


rotations = 50
states = ['H', 'D', 'V', 'A']


def plot_g2_darkcounts(yaml_fn, dc, colors=None):

    with alive_bar(len(dc), force_tty=True) as bar:
        sim = Simulator(yaml_fn)
        max_max = 0
        for i in range(0, len(dc)):

            sim.set_dc_rate(dc[i])
            sim.run()
            if sim.bins[0] != -1:
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
            'l=' + str((sim.optical_loss_idler + sim.optical_loss_signal) / 2) + ',' + \
            'dt=' + str(sim.deadtime) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000, bbox_inches='tight')
    plt.show()


def plot_g2_deadtime(yaml_fn, deadtime, colors=None):

    with alive_bar(len(deadtime), force_tty=True) as bar:

        max_max = 0

        for i in range(0, len(deadtime)):
            sim = Simulator(yaml_fn)
            sim.set_deadtime(deadtime[i])
            sim.run()
            if sim.bins[0] != -1:
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
            'l=' + str((sim.optical_loss_idler + sim.optical_loss_signal) / 2) + ',' + \
            'dc=' + str(sim.dark_count_rate) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000, bbox_inches='tight')
    plt.show()


def plot_g2_jitter(yaml_fn, jitter, colors=None, fwhm=False):

    with alive_bar(len(jitter), force_tty=True) as bar:
        fig, ax = plt.subplots()

        for i in range(0, len(jitter)):
            sim = Simulator(yaml_fn)

            sim.set_jitter(jitter[i])
            sim.run()
            if sim.bins[0] != -1:
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

    title = 'plots\\g2\\g2_vs_jitter\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'l=' + str((sim.optical_loss_idler + sim.optical_loss_signal) / 2) + ',' + \
            'dc=' + str(sim.dark_count_rate) + ',' + \
            'dt=' + str(sim.deadtime) + ',' + \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000, bbox_inches='tight')
    plt.show()


def plot_g2_loss(yaml_fn, loss, colors=None):

    with alive_bar(len(loss), force_tty=True) as bar:

        max_max = 0
        for i in range(len(loss)):
            sim = Simulator(yaml_fn)
            loss_pr = 10 ** (loss[i]/10)
            sim.set_loss_signal(loss_pr)
            sim.set_loss_idler(loss_pr)
            sim.run()

            if sim.bins[0] != -1:
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
            'dt=' + str(sim.deadtime) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000, bbox_inches='tight')
    plt.show()


def plot_car_darkcounts(yaml_fn, start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = []

    with alive_bar(num_points, force_tty=True) as bar:

        sim = Simulator(yaml_fn)

        for dc in x_val:
            sim.set_dc_rate(dc)
            sim.run()
            car = sim.get_car()
            y_val.append(car)
            bar()

    plt.plot(x_val, y_val)
    plt.xlabel('Dark Count Rate (counts/second)')
    plt.ylabel('Coincidence-to-Accidental Rate')

    title = 'plots\\car\\car_vs_darkcounts\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'l=' + str((sim.optical_loss_idler + sim.optical_loss_signal) / 2) + ',' + \
            'dt=' + str(sim.deadtime) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000, bbox_inches='tight')
    plt.show()


def plot_car_deadtime(yaml_fn, start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = []

    with alive_bar(num_points, force_tty=True) as bar:
        for deadtime in x_val:
            sim = Simulator(yaml_fn)
            sim.set_deadtime(deadtime)
            sim.run()
            car = sim.get_car()
            y_val.append(car)
            bar()

    plt.plot(x_val, y_val)
    plt.xlabel('Dead Time (ps)')
    plt.ylabel('Coincidence-to-Accidental Rate')

    title = 'plots\\car\\car_vs_deadtime\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'l=' + str((sim.optical_loss_idler + sim.optical_loss_signal) / 2) + ',' + \
            'dc=' + str(sim.dark_count_rate) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000, bbox_inches='tight')
    plt.show()


def plot_car_jitter(yaml_fn, start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = []

    with alive_bar(num_points, force_tty=True) as bar:
        for jitter in x_val:
            sim = Simulator(yaml_fn)
            sim.set_jitter(jitter)
            sim.run()
            car = sim.get_car()
            y_val.append(car)
            bar()

    plt.plot(x_val, y_val)
    plt.xlabel('Jitter FWHM (ps)')
    plt.ylabel('Coincidence-to-Accidental Rate')

    title = 'plots\\car\\car_vs_jitter\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'l=' + str((sim.optical_loss_idler + sim.optical_loss_signal) / 2) + ',' + \
            'dc=' + str(sim.dark_count_rate) + ',' + \
            'dt=' + str(sim.deadtime) + ',' + \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000, bbox_inches='tight')
    plt.show()


def plot_car_loss(yaml_fn, start, stop, num_points):

    x_val = 10**(np.linspace(start, stop, num_points)/10)
    y_val = []

    with alive_bar(num_points, force_tty=True) as bar:
        for loss in x_val:
            sim = Simulator(yaml_fn)
            sim.set_loss_signal(loss)
            sim.set_loss_idler(loss)
            sim.run()
            car = sim.get_car()
            y_val.append(car)
            bar()

    plt.plot(np.linspace(start, stop, num_points), y_val)
    plt.xlabel('Optical Loss (dB)')
    plt.ylabel('Coincidence-to-Accidental Rate')

    title = 'plots\\car\\car_vs_loss\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'dc=' + str(sim.dark_count_rate) + ',' + \
            'dt=' + str(sim.deadtime) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000, bbox_inches='tight')
    plt.show()


def plot_visibility_darkcounts(yaml_fn, state, start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = [[], [], [], []]

    with alive_bar(num_points, force_tty=True) as bar:
        for dc in x_val:
            sim = Rotation(yaml_fn, state, rotations)
            sim.set_dc_rate(dc)
            sim.run()
            visibility = sim.get_visibility()
            for i in range(4):
                y_val[i].append(visibility[i])
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
            'l=' + str((sim.optical_loss_idler + sim.optical_loss_signal) / 2) + ',' + \
            'dt=' + str(sim.deadtime) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000)
    plt.show()


def plot_visibility_deadtime(yaml_fn, state, start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = [[], [], [], []]

    with alive_bar(num_points, force_tty=True) as bar:
        for deadtime in x_val:
            sim = Rotation(yaml_fn, state, rotations)
            sim.set_deadtime(deadtime)
            sim.run()
            visibility = sim.get_visibility()
            for i in range(4):
                y_val[i].append(visibility[i])
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
            'l=' + str((sim.optical_loss_idler + sim.optical_loss_signal) / 2) + ',' + \
            'dc=' + str(sim.dark_count_rate) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000)
    plt.show()


def plot_visibility_jitter(yaml_fn, state, start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = [[], [], [], []]

    with alive_bar(num_points, force_tty=True) as bar:
        for jitter in x_val:
            sim = Rotation(yaml_fn, state, rotations)
            sim.set_jitter(jitter)
            sim.run()
            visibility = sim.get_visibility()
            for i in range(4):
                y_val[i].append(visibility[i])
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
            'l=' + str((sim.optical_loss_idler + sim.optical_loss_signal) / 2) + ',' + \
            'dc=' + str(sim.dark_count_rate) + ',' + \
            'dt=' + str(sim.deadtime) + ',' + \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000)
    plt.show()


def plot_visibility_loss(yaml_fn, state, start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = [[], [], [], []]

    with alive_bar(num_points, force_tty=True) as bar:
        for loss in x_val:
            sim = Rotation(yaml_fn, state, rotations)
            loss_pr = 10 ** (loss / 10)
            sim.set_loss_signal(loss_pr)
            sim.set_loss_idler(loss_pr)
            sim.run()
            visibility = sim.get_visibility()
            for i in range(4):
                y_val[i].append(visibility[i])
            bar()

    for i in range(4):
        plt.plot(x_val, y_val[i], label=states[i])
    plt.legend()
    plt.xlabel('Optical Loss (dB)')
    plt.ylabel('Visibility')
    plt.ylim(-0.01, 1.01)

    title = 'plots\\visibility\\visibility_vs_loss\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'dc=' + str(sim.dark_count_rate) + ',' + \
            'dt=' + str(sim.deadtime) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000)
    plt.show()


def plot_visibility_phasediff(yaml_fn, start, stop, num_points):

    x_val = np.linspace(start, stop, num_points)
    y_val = [[], [], [], []]

    with alive_bar(num_points, force_tty=True) as bar:
        for delta in x_val:
            entangled_state = [1 / math.sqrt(2), 0, 0, np.exp(1j * delta) / math.sqrt(2)]
            sim = Rotation(yaml_fn, entangled_state, rotations)
            sim.run()
            visibility = sim.get_visibility()
            for i in range(4):
                y_val[i].append(visibility[i])
            bar()

    for i in range(4):
        plt.plot(x_val, y_val[i], label=states[i])
    plt.legend()
    plt.xlabel('Phase Difference in the Entangled State')
    plt.ylabel('Visibility')
    plt.ylim(-0.01, 1.01)

    title = 'plots\\visibility\\visibility_vs_deadtime\\' + \
            'λ=' + str(sim.lambd) + ',' + \
            '𝜏=' + str(sim.lag) + ',' + \
            str(sim.total_time) + 's,' + \
            'l=' + str((sim.optical_loss_idler + sim.optical_loss_signal) / 2) + ',' + \
            'dc=' + str(sim.dark_count_rate) + ',' + \
            'dt=' + str(sim.deadtime) + ',' + \
            'j=' + str(sim.jitter_fwhm) + ',' \
            'ci=' + str(sim.coincidence_interval) + '.png'

    plt.savefig(title, dpi=1000)
    plt.show()


if __name__ == '__main__':
    d = .25
    qubit_state = [1 / math.sqrt(2), 0, 0, np.exp(1j * d) / math.sqrt(2)]

    # plot_g2_darkcounts('config.yaml', [0, 100, 1000, 100000], ['C1', 'C8', 'C2', 'C0'])
    # plot_g2_deadtime('config.yaml', [0, 25000, 50000, 75000], ['C1', 'C8', 'C2', 'C0'])
    # plot_g2_jitter('config.yaml', [0, 5000, 10000, 20000, 100000], ['C3', 'C1', 'C2', 'C0', 'C4'], fwhm=True)
    # plot_g2_loss('config.yaml', [-10, -6, -3, -1, 0], ['C3', 'C1', 'C8', 'C2', 'C0', 'C4'])

    # plot_car_darkcounts('config.yaml', 0, 10000, 64)
    # plot_car_deadtime('config.yaml', 0, 1000000, 64)
    # plot_car_jitter('config.yaml', 0, 8000, 4096)
    # plot_car_loss('config.yaml', 0, -30, 256)

    # plot_visibility_darkcounts('config.yaml', qubit_state, 0, 1000000, 8)
    # plot_visibility_deadtime('config.yaml', qubit_state, 0, 1000000, 8)
    # plot_visibility_jitter('config.yaml', qubit_state, 0, 100000, 8)
    # plot_visibility_loss('config.yaml', qubit_state, 0, -30, 8)
    # plot_visibility_phasediff('config.yaml', 0, 2*math.pi, 64)
