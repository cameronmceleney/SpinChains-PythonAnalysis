# -*- coding: utf-8 -*-

# -------------------------- Preprocessing Directives -------------------------
import matplotlib
# matplotlib.use('macosx')

# Standard Libraries
import logging as lg
# import os as os
from sys import exit
import seaborn as sns

# 3rd Party packages
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# My packages/Header files
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes

# ----------------------------- Program Information ----------------------------

"""
Description of what foo.py does
"""
PROGRAM_NAME = "foo.py"
"""
Created on (date) by (author)
"""


# ---------------------------- Function Declarations ---------------------------
def square_number():
    """
    User-inputs a number to square

    :return: Squared number
    """

    number_to_square = int(input("Enter a number to square: "))
    squared_number = number_to_square ** number_to_square

    return squared_number


def loggingSetup():
    """
    Minimum Working Example (MWE) for logging. Pre-defined levels are:
        
        Highest               ---->            Lowest
        CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    """
    today_date = datetime.now().strftime("%y%m%d")
    current_time = datetime.now().strftime("%H%M")

    lg.basicConfig(filename=f'./{today_date}-{current_time}.log',
                   filemode='w',
                   level=lg.INFO,
                   format='%(asctime)s | %(module)s::%(funcName)s | %(levelname)s | %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   force=True)


def rc_params_update():
    """Container for program's custom rc params, as well as Seaborn (library) selections."""
    plt.style.use('fivethirtyeight')
    sns.set(context='notebook', font='Kohinoor Devanagari', palette='muted', color_codes=True)
    ##############################################################################
    # Sets global conditions including font sizes, ticks and sheet style
    # Sets various font size. fsize: general text. lsize: legend. tsize: title. ticksize: numbers next to ticks
    medium_size = 14
    small_size = 12
    large_size = 16
    smaller_size = 10
    tiny_size = 10

    # sets the tick direction. Options: 'in', 'out', 'inout'
    t_dir = 'in'
    # sets the tick size(s) and tick width(w) for the major and minor axes of all plots
    t_maj_s = 4
    t_min_s = t_maj_s / 2
    t_maj_w = 0.8
    t_min_w = t_maj_w / 2

    # updates rcParams of the selected style with my preferred options for these plots. Feel free to change
    plt.rcParams.update({'axes.titlesize': medium_size, 'axes.labelsize': smaller_size, 'font.size': small_size,
                         'legend.fontsize': tiny_size,
                         'figure.titlesize': large_size,
                         'xtick.labelsize': tiny_size, 'ytick.labelsize': tiny_size,
                         'axes.edgecolor': 'black', 'axes.linewidth': t_maj_w,
                         "xtick.bottom": True, "ytick.left": True,
                         'xtick.color': 'black', 'ytick.color': 'black', 'ytick.labelcolor': 'black',
                         'text.color': 'black',
                         'xtick.major.size': t_maj_s, 'xtick.major.width': t_maj_w,
                         'xtick.minor.size': t_min_s, 'xtick.minor.width': t_min_w,
                         'ytick.major.size': t_maj_s, 'ytick.major.width': t_maj_w,
                         'ytick.minor.size': t_min_s, 'ytick.minor.width': t_min_w,
                         'xtick.direction': t_dir, 'ytick.direction': t_dir,
                         'axes.spines.top': False, 'axes.spines.bottom': True, 'axes.spines.left': True,
                         'axes.spines.right': False,
                         'savefig.dpi': 1000, "figure.dpi": 1000,
                         'axes.facecolor': 'white', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def tick_setter(ax, x_major, x_minor, y_major, y_minor, is_fft_plot=False):
    if ax is None:
        ax = plt.gca()

    if is_fft_plot:
        ax.xaxis.set(major_locator=ticker.MultipleLocator(x_major), major_formatter=ticker.FormatStrFormatter("%.1f"),
                     minor_locator=ticker.MultipleLocator(x_minor))
        ax.yaxis.set(major_locator=ticker.LogLocator(base=10, numticks=y_major))
        locmin = ticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=y_minor)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        # ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    else:
        ax.xaxis.set(major_locator=ticker.MultipleLocator(x_major), major_formatter=ticker.FormatStrFormatter("%.1f"),
                     minor_locator=ticker.MultipleLocator(x_minor))
        ax.yaxis.set(major_locator=ticker.MaxNLocator(nbins=y_major, prune='lower'),
                     major_formatter=ticker.FormatStrFormatter("%.1f"),
                     minor_locator=ticker.AutoMinorLocator(y_minor))

        formatter = ticker.ScalarFormatter(useMathText=True)
        ax.yaxis.set_major_formatter(formatter)

        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_visible(False)
        ax.yaxis.get_offset_text().set_fontsize(8)
        t = ax.yaxis.get_offset_text()
        t.set_x(-0.045)

    return ax


def compare_dataset_plots():
    spin_site = 3000

    dataset1 = np.loadtxt("D:\\Data\\2022-08-10\\Simulation_Data\\rk2_mx_T0940.csv",
                          delimiter=",", skiprows=11)
    dataset2 = np.loadtxt("D:\\Data\\2022-08-10\\Simulation_Data\\rk2_mx_T1006.csv",
                          delimiter=",", skiprows=11)
    dataset3 = np.loadtxt("D:\\Data\\2022-08-10\\Simulation_Data\\rk2_mx_T1550.csv",
                          delimiter=",", skiprows=11)
    dataset4 = np.loadtxt("D:\\Data\\2022-08-10\\Simulation_Data\\rk2_mx_T1602.csv",
                          delimiter=",", skiprows=11)
    dataset5 = np.loadtxt("D:\\Data\\2022-08-10\\Simulation_Data\\rk2_mx_T0955.csv",
                          delimiter=",", skiprows=11)
    dataset6 = np.loadtxt("D:\\Data\\2022-08-10\\Simulation_Data\\rk2_mx_T0929.csv",
                          delimiter=",", skiprows=11)

    time = dataset1[:, 0] * 1e9
    data_to_plot1 = dataset1[:, spin_site]
    data_to_plot2 = dataset2[:, spin_site]
    data_to_plot3 = dataset3[:, spin_site]
    data_to_plot4 = dataset4[:, spin_site]
    data_to_plot5 = dataset5[:, spin_site]
    data_to_plot6 = dataset6[:, spin_site]

    colour1 = '#64bb6a'
    colour2 = '#fd7f6f'
    colour3 = '#7eb0d5'
    colour4 = '#b2e061'
    colour5 = '#bd7ebe'
    colour6 = '#ffb55a'
    colour7 = '#beb9db'

    fig = plt.figure(figsize=(4, 3))
    ax1 = fig.add_subplot(111)  # 64BB6A, #648ABB, #BB64B5, #BB9664
    ax1.plot(time, abs(data_to_plot1), lw=0.5, marker='o', markersize=0, label="Instant", color=colour1, zorder=1.1)
    ax1.plot(time, abs(data_to_plot2), lw=0.5, marker='o', markersize=0, label="0.05", color=colour2, zorder=1.2)
    ax1.plot(time, abs(data_to_plot3), lw=0.5, marker='o', markersize=0, label="0.1", color=colour3, zorder=1.3)
    ax1.plot(time, abs(data_to_plot4), lw=0.5, marker='o', markersize=0, label="0.25", color=colour4, zorder=1.4)
    ax1.plot(time, abs(data_to_plot5), lw=0.5, marker='o', markersize=0, label="0.5", color=colour5, zorder=1.5)
    ax1.plot(time, abs(data_to_plot6), lw=0.5, marker='o', markersize=0, label="1.0", color=colour6, zorder=1.6)

    ax1.fill_between(time, abs(data_to_plot1), color=colour1, zorder=1.1)
    ax1.fill_between(time, abs(data_to_plot2), color=colour2, zorder=1.2)
    ax1.fill_between(time, abs(data_to_plot3), color=colour3, zorder=1.3)
    ax1.fill_between(time, abs(data_to_plot4), color=colour4, zorder=1.4)
    ax1.fill_between(time, abs(data_to_plot5), color=colour5, zorder=1.5)
    ax1.fill_between(time, abs(data_to_plot6), color=colour6, zorder=1.6)

    # Inset
    ax1_inset = inset_axes(ax1, width=1.55
                           , height=1.55, loc="upper left", bbox_to_anchor=[0.0875, 0.875],
                           bbox_transform=ax1.figure.transFigure)
    mark_inset(ax1, ax1_inset, loc1=3, loc2=4, facecolor="none", edgecolor="black", lw=0.75, alpha=1.0, zorder=1.9)
    # ax1.indicate_inset_zoom(ax1_inset, facecolor='#f9f2e9', edgecolor='black', alpha=1.0, lw=0.75, zorder=1)

    ax1_inset.plot(time, abs(data_to_plot1), lw=0.5, marker='o', markersize=0, label="Instant", color=colour1,
                   zorder=1.1)
    ax1_inset.plot(time, abs(data_to_plot2), lw=0.5, marker='o', markersize=0, label="0.05", color=colour2,
                   zorder=1.2)
    ax1_inset.plot(time, abs(data_to_plot3), lw=0.5, marker='o', markersize=0, label="0.1", color=colour3, zorder=1.3)
    ax1_inset.plot(time, abs(data_to_plot4), lw=0.5, marker='o', markersize=0, label="0.25", color=colour4,
                   zorder=1.4)
    ax1_inset.plot(time, abs(data_to_plot5), lw=0.5, marker='o', markersize=0, label="0.5", color=colour5, zorder=1.5)
    ax1_inset.plot(time, abs(data_to_plot6), lw=0.5, marker='o', markersize=0, label="1.0", color=colour6, zorder=1.6)

    ax1_inset.fill_between(time, abs(data_to_plot1), color=colour1, zorder=1.1)
    ax1_inset.fill_between(time, abs(data_to_plot2), color=colour2, zorder=1.2)
    ax1_inset.fill_between(time, abs(data_to_plot3), color=colour3, zorder=1.3)
    ax1_inset.fill_between(time, abs(data_to_plot4), color=colour4, zorder=1.4)
    ax1_inset.fill_between(time, abs(data_to_plot5), color=colour5, zorder=1.5)
    ax1_inset.fill_between(time, abs(data_to_plot6), color=colour6, zorder=1.6)

    # Main
    ax1.set(xlim=[0.5, 5], ylim=[1e-9, 3.5e-3],
            xlabel="Time [ns]", ylabel="|m$_x$|/M$_S$")

    ax1.xaxis.set(major_locator=ticker.MultipleLocator(1.0), major_formatter=ticker.FormatStrFormatter("%.1f"),
                  minor_locator=ticker.MultipleLocator(0.2))
    # ax1.locator_params(axis='y', nbins=3)
    locmin = ticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.01, numticks=15)
    ax1.yaxis.set_minor_locator(locmin)
    ax1.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax1.set_yticks([1e-4, 1e-3, 2e-3, 3e-3])
    # ax1.yaxis.set(major_locator=ticker.LogLocator(base=10, numticks=15))
    # locmin = ticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=15)
    # ax1.yaxis.set_minor_locator(locmin)
    # ax1.yaxis.set_minor_formatter(ticker.NullFormatter())
    # locmin = ticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=20)
    # ax1.yaxis.set_minor_locator(locmin)
    # ax1.yaxis.set_minor_formatter(ticker.NullFormatter())
    #
    # formatter = ticker.ScalarFormatter(useMathText=True)
    # ax1.yaxis.set_major_formatter(formatter)
    ax1.grid(False)
    ax1_inset.grid(False)

    # Inset
    ax1_inset.set(xlim=[0.5001, 1.4999], ylim=[1e-7, 5e-5],
                  yscale="log")
    # ax1.set_xticks([])
    # ax1.set_yticks([])

    ax1_inset.yaxis.tick_right()
    ax1_inset.tick_params(axis='x', labelsize=8)
    ax1_inset.tick_params(axis='y', labelsize=8)

    ax1_inset.xaxis.set(major_locator=ticker.MultipleLocator(0.25), major_formatter=ticker.FormatStrFormatter("%.1f"),
                        minor_locator=ticker.MultipleLocator(0.05))
    ax1_inset.yaxis.set(major_locator=ticker.LogLocator(base=10, numticks=15))
    locmin = ticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=15)
    ax1_inset.yaxis.set_minor_locator(locmin)
    ax1_inset.yaxis.set_minor_formatter(ticker.NullFormatter())
    #
    # formatter = ticker.ScalarFormatter(useMathText=True)
    # ax1_inset.yaxis.set_major_formatter(formatter)

    for spine in ['top', 'bottom', 'left', 'right']:
        ax1.spines[spine].set_visible(True)
        ax1_inset.spines[spine].set_visible(True)
    for k, spine in ax1.spines.items():  # ax.spines is a dictionary
        spine.set_zorder(10)

    ax1.yaxis.get_offset_text().set_visible(False)
    ax1.text(-0.05, 0.97, r'$\times \mathcal{10}^{{\mathcal{-3}}}$',
             verticalalignment='center', horizontalalignment='center', transform=ax1.transAxes, fontsize=8)

    # ax1.margins(x=0.5, y=0.1e-3)
    ax1_inset.set_axisbelow(False)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax1.tick_params(axis="both", which="both", zorder=10, bottom=True, top=False, left=True, right=False)
    ax1_inset.tick_params(axis="both", which="both", zorder=10, bottom=True, top=False, left=False, right=True)

    ax1.legend(title="t$_g$ [ns]", ncol=1, loc="lower right",
               frameon=True, fancybox=False, facecolor='white', edgecolor='black',
               fontsize=6, title_fontsize=7,
               bbox_to_anchor=(0.94, 0.08), bbox_transform=ax1.figure.transFigure
               ).set_zorder(1.9)

    ax1.set_axisbelow(False)
    ax1_inset.set_axisbelow(False)

    fig.savefig("D:\\Data\\2022-08-10\\Outputs\\comparison_220928_1.png", bbox_inches="tight")


def afm_test():
    type = 7

    hbar_si = 1.05456e-34  # m^2 kg s^-1
    hbar_cgs = 1.0545919e-27  # erg s (cm2 g s-1)
    mu_0 = 1.256637062e-6  # kg m s-2 A-2

    j_2_mev = 6.2415e21
    erg_to_joule = 1e-7
    erg_to_meV = erg_to_joule * j_2_mev
    hz_to_ghz = 1e9

    t_range = 0.8  # T
    h_0_si = np.linspace(-t_range, t_range, 1000)
    h_0_cgs = np.linspace(-t_range * 1e1, t_range * 1e1, 1000)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)

    if type == 1:
        gamma_cgs = 2.8  # GHz / kOe rad
        a = b = 4.873e-8
        c = 3.13e-8
        he_cgs = 530  # kOe
        ha_cgs = 7.87  # kOe

        k_afm = np.linspace(0, 0.5 * 2 * np.pi, 1000)
        gamma_k = np.cos(k_afm * a / 2) * np.cos(k_afm * b / 2) * np.cos(k_afm * c / 2)
        omega_k = np.sqrt(2 * he_cgs * ha_cgs + ha_cgs ** 2 + 0 * he_cgs ** 2 * (1.0 - gamma_k ** 2))

        w_afm_0 = gamma_cgs * (omega_k + h_0_cgs)
        w_afm_0neg = gamma_cgs * (omega_k - h_0_cgs)

        ax.plot(h_0_cgs, w_afm_0, label='$\omega_{pos}$')
        ax.plot(h_0_cgs, w_afm_0neg, label='$\omega_{neg}$')
        # ax.plot(k_afm / (2 * np.pi), w_afm_0, label='0')
        # ax.set(title='Dispersion Relation', xlabel="$k_z$ / 2 $\pi$", ylabel='Frequency [GHz]')

        # ax.set(title='Dispersion Relation', xlabel="$k_z$ / 2 $\pi$", ylabel='Energy (meV) [$\hbar \omega$]', ylim=[0, 8])
        ax.set(xlabel="$H_0$[kOe]", ylabel="$\\frac{\omega}{2 \pi}$ [GHz]")
    elif type == 2:
        a_si = 2e-9  # nm
        gamma_si = 28  # GHz / T raf
        he_si = 53  # T
        ha_si = 0.787  # T

        omega_pos = gamma_si * (np.sqrt(2 * he_si * ha_si + ha_si ** 2) + h_0_si)
        omega_neg = gamma_si * (np.sqrt(2 * he_si * ha_si + ha_si ** 2) - h_0_si)

        ax.plot(h_0_si, omega_pos, label="$\omega_{pos}$")
        ax.plot(h_0_si, omega_neg, label="$\omega_{neg}$")

        ax.set(xlabel="$H_0$[T]", ylabel="$\\frac{\omega}{2 \pi}$ [GHz]")
    elif type == 3:

        a_si = 4.87e-8  # cm
        he_si = 570e3  # Oe
        ha_si = 7.87  # T
        S = 1

        k = np.linspace(0.0, 0.4e-8, 1000)

        omega_ex = 4 * np.abs(he_si) * S / hbar_cgs

        omega = omega_ex * np.sin(k * a_si)
        ax.plot(k, hbar_cgs * omega, label="$\\omega$")

        ax.set(xlabel="$k$ [cm]", ylabel="$Magnon Energy \\hbar \\omega$ [erg]")
    elif type == 4:
        # Comparing dispersion relations of FM and AFM, and plotting in terms of hw/4JS
        a = 2e-9
        k = np.linspace(0, np.pi / a, 1000)
        # ax.plot(k, 2 * 13.25 * 2e-9**2 * k**2)
        ax.plot(k, np.sin(k * a), label='AFM')
        ax.plot(k, 1 - np.cos(k * a), label='FM')
        ax.set(xlabel='$k$ [m]', ylabel='$\\frac{\\hbar \\omega}{4 J S}$')
    elif type == 5:
        # Comparing dispersion relations of FM and AFM in CGS units and plotting in terms of energy
        a = b = 4.873e-8
        c = 3.13e-8
        k = np.linspace(0, np.pi / a, 1000)

        ax.plot(k, np.sin(k * a) * 4 * 570e3, label='AFM')
        ax.plot(k, (1 - np.cos(k * a)) * 4 * 570e3, label='FM')
        ax.set(xlabel='$k$ [cm$^{-1}$]', ylabel='Magnon Energy $\\hbar \\omega$ [erg]')
    elif type == 6:
        # AFM resonance frequencies in CGS units
        gamma = 2.8  # GHz / kOe
        he = 530  # kOe
        ha = 7.87  # kOe
        h0 = np.linspace(-8, 8, 100)
        omega_0 = gamma * (np.sqrt(2 * he * ha + ha ** 2) + h0)
        omega_0n = gamma * (np.sqrt(2 * he * ha + ha ** 2) - h0)
        omega_1 = gamma * (np.sqrt(2 * he * ha + ha ** 2) + 1)
        omega_7 = gamma * (np.sqrt(2 * he * ha + ha ** 2) + ha)

        ax.plot(h0, omega_0, label='$\omega_{pos}$')
        ax.plot(h0, omega_0n, label='$\omega_{neg}$')
    elif type == 7:

        datestamp = "2022-11-11"
        timestamp = "1453"

        omega = np.loadtxt(
            f"/Users/cameronmceleney/CLionProjects/Data/{datestamp}/Simulation_Data/eigenvalues_formatted_eigenvalues_T{timestamp}.csv")
        freqs = np.loadtxt(
            f"/Users/cameronmceleney/CLionProjects/Data/{datestamp}/Simulation_Data/eigenvalues_formatted_eigenvalues_T{timestamp}.csv")
        gamma = 29.2  # GHz / T
        h_0 = 1e-1  # T
        J = 1e-3 * 1.60218E-19  # was in meV then converted to joules
        a = 2e-10  # m

        k = np.sqrt((hbar_si * omega - gamma * hbar_si * h_0) / (2 * J * a ** 2))
        # ax.scatter(k * a / (2 * np.pi), hbar_si * omega * j_2_mev, label="$k$")
        # ax.set(xlabel="k a / $ 2 \\pi$", ylabel="$\\hbar \\omega$ [meV]")
        ax.scatter(np.arange(1, len(freqs) + 1, 1), freqs, s=0.5)
        ax.set(ylabel="Frequency [GHz]", xlabel='Mode Number', xlim=[1800, 2200], ylim=[6200, 8400])

    # ax.legend(title='$H_0$ [T]')
    fig.tight_layout()

    fig.savefig(f"/Users/cameronmceleney/CLionProjects/Data/{datestamp}/Outputs/T{timestamp}_dispersion_{type}",
                bbox_inches="tight")
    exit(0)


def afm_test_si():
    # T: kg s-2 A-1. Units of H: A m-1. Units  of mu0*H: T
    joule_to_meV = 6.242e21

    gamma = 28e9 * 2 * np.pi  # Hz / T
    mu_0 = 1.256637062e-6  # kg m s-2 A-2
    hbar = 1.054571817e-34  # J s
    h_bar_meV = hbar * joule_to_meV
    h_ex = 57 / mu_0  # T
    h_a = 0.82 / mu_0  # T
    h_0 = 0 / mu_0  # T

    k = np.linspace(0, 0.5 * 2 * np.pi, 1000)
    gamma_k = np.cos(k / 2)
    w_k = np.sqrt(2 * h_ex * h_a + h_a ** 2 + (h_ex ** 2) * (1 - gamma_k ** 2))
    omega = gamma * (w_k + h_0)
    print(omega[0] * h_bar_meV)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(k / (2 * np.pi), omega * h_bar_meV, label='new')
    ax.set(xlabel="$k_z$ / 2 $\pi$", ylabel='Energy (meV)')

    ax.legend()

    fig.tight_layout()
    fig.savefig("D:\\Data\\2022-10-17\\Outputs\\test5.png", bbox_inches="tight")


# --------------------------- main() implementation ---------------------------

def main():
    lg.info(f"{PROGRAM_NAME} start")

    # t = np.linspace(0, 5, 2000)
    #
    # y_1 = np.zeros(200)
    # y_2 = 3e-3 * np.sin(2*np.pi*15 * t[200:])
    # y = np.concatenate((y_1, y_2))

    SAMPLE_RATE = int(5e2)  # Number of samples per nanosecond
    DURATION = int(15)  # Nanoseconds

    def generate_sine_wave(freq, sample_rate, duration, delay_num):
        delay = int(sample_rate * delay_num)
        t = np.linspace(0, duration, sample_rate * duration, endpoint=False)
        y_1 = np.zeros(delay)
        y_2 = np.sin((2 * np.pi * freq) * t[delay:])
        y = np.concatenate((y_1, y_2))
        return t, y

    # Generate a 15 GHz sine wave that lasts for 5 seconds
    x1, y1 = generate_sine_wave(15, SAMPLE_RATE, DURATION, 1)
    x2, y2 = generate_sine_wave(15, SAMPLE_RATE, DURATION, 0)
    from scipy.fft import rfft, rfftfreq
    # Number of samples in normalized_tone
    N1 = int(SAMPLE_RATE * DURATION)
    N2 = int(SAMPLE_RATE * DURATION)

    y1f = rfft(y1)
    y2f = rfft(y2)
    x1f = rfftfreq(N1, 1 / SAMPLE_RATE)
    x2f = rfftfreq(N2, 1 / SAMPLE_RATE)

    fig = plt.figure(figsize=(18.73 / 2.54, 4.94 / 2.54))
    ax = fig.add_subplot(111)

    ax.plot(x1f, np.abs(y1f), marker='', lw=1.0, color='#ffb55a', markerfacecolor='black', markeredgecolor='black',
            label="1", zorder=1.2)
    ax.plot(x2f, np.abs(y2f), marker='', lw=1.0, ls='-', color='#64bb6a', markerfacecolor='black',
            markeredgecolor='black',
            label="0", zorder=1.3)

    ax.set(xlim=(5, 25), ylim=(1e0, 1e4),
           xlabel="Frequency [GHz]", ylabel="\n\nAmplitude [arb.]", yscale='log')
    ax.grid(visible=False, axis='both', which='both')
    ax.tick_params(top="on", right="on", which="both")

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
    ax2_inset = inset_axes(ax, width=1.75, height=0.75, loc="upper right", bbox_to_anchor=[0.88, 0.85],
                           bbox_transform=ax.figure.transFigure)
    ax2_inset.plot(x1, y1, lw=0.5, color='#ffb55a', zorder=1)
    ax2_inset.grid(visible=False, axis='both', which='both')
    ax2_inset.yaxis.tick_right()
    ax2_inset.set_xlabel('Time [ns]', fontsize=8)
    ax2_inset.yaxis.set_label_position("right")
    ax2_inset.set_ylabel('Amplitude\n[arb.]', fontsize=8, rotation=-90, labelpad=10)
    ax2_inset.tick_params(axis='both', labelsize=8)
    # ax2_inset.set(xlim=[0, 20], ylim=[-1, 1],
    #              yscale="linear")
    # ax2_inset.set_yticks([])

    ax2_inset.patch.set_color("#f9f2e9")

    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax2_inset.spines[spine].set_visible(True)

    ax.xaxis.set(major_locator=ticker.MultipleLocator(5),
                 minor_locator=ticker.MultipleLocator(1))
    ax2_inset.xaxis.set(major_locator=ticker.MultipleLocator(4),
                        minor_locator=ticker.MultipleLocator(1))
    ax2_inset.xaxis.labelpad = -0.5

    ax.set_axisbelow(False)
    ax2_inset.set_axisbelow(False)

    ax.legend(title="$\Delta t$ [ns]", ncol=1, loc="upper left",
              frameon=False, fancybox=False, facecolor='white', edgecolor='black',
              fontsize=8, title_fontsize=10,
              bbox_to_anchor=(0.0875, 0.85), bbox_transform=ax.figure.transFigure
              ).set_zorder(4)

    fig.savefig("D:\\Data\\2022-08-10\\Outputs\\image4.png", bbox_inches="tight")

    lg.info(f"{PROGRAM_NAME} end")
    exit()


def resizing_system_test(use_gauss):
    if use_gauss:
        # Core parameters
        D = 5.3E-9  # [erg / G * cm]
        gamma = 2.9  # [GHz G**-1] # * 2 * np.pi
        H = 1  # [kG]
        w = 25  # [GHz]
        um = 1e4  # how many [cm] in 1[um]

        # Find wavenumber
        k = np.sqrt(((w / gamma) - H) * 1e3 / D) * (1 / um)
        print(f"k\t\t\t: {k}\t\t[1/um]")

        # Find wavelength
        wavelength = (2 * np.pi) / (k)
        print(f"wavelength\t: {wavelength}\t[um]")

        # Find lattice spacing
        a = 0.002 * (1 / um)  # 0.002 [um] is what we used on the board
        print(f"spacing\t\t: {a} [cm]")

        # Find exchange integral
        J = D / a ** 2
        print(f"J\t\t\t: {J / 1e3}\t\t[kG]")

    else:
        # Core parameters
        D = 5.3E-17 / 2  # [T m**2]
        gamma = 29.3 * 2 * np.pi  # [GHz T**-1] # * 2 * np.pi
        H = 0.1  # [T]
        w = (2 * np.pi) * 42.5  # [GHz]

        # Find wavenumber
        k = np.sqrt(((w / gamma) - H) / D)
        print(f"k\t\t\t: {k * 1e-6}\t\t[1/um]")

        # Find wavelength
        wavelength = (2 * np.pi) / k
        print(f"wavelength\t: {wavelength * 1e6}\t[um]")

        # Find lattice spacing
        a = wavelength * 0.05  # 0.002E-6 is what we used on the board
        print(f"spacing\t\t: {a * 1e6} [um]")

        # Find exchange integral
        J = D / a ** 2
        print(f"J\t\t\t: {J}\t[T]")


def test():
    omega = 20e9
    t = np.linspace(0, 1e-9, 10000)
    drive = np.cos(omega * t)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, drive)
    plt.show()


def propagating_modes():
    g = 29.2
    H = np.linspace(0, 1.0, 1000)
    Ms = 0.1
    fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(6, 4))

    perpendicular = g*np.sqrt(H*(H+4*np.pi*Ms))
    parallel = g*H
    surface = g*(H+2*np.pi*Ms)

    combined = np.vstack((parallel, perpendicular)).T
    plt.plot(H, combined)
    plt.xlabel("Applied Magnetic Field [T]")
    plt.ylabel("Angular Frequency [GHz]")
    plt.title("Bulk Propagating Modes")

    ax1.fill_between(H, parallel, perpendicular, alpha=0.1)
    ax1.grid(False)
    fig.tight_layout()
    plt.show()


def afm():

    h0 = np.arange(-3, 3, 0.1)
    gamma = 28.8
    he = 53
    ha = 0.787
    omega_pos = gamma * np.sqrt(2 * he * ha + ha**2) + h0
    omega_neg = gamma * np.sqrt(2 * he * ha + ha ** 2) - h0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(h0, omega_pos)
    ax.plot(h0, omega_neg)
    fig.tight_layout()
    plt.show()
# ------------------------------ Implementations ------------------------------


def analytic_sine_wave():
    max_simtime = 2e-9
    max_dp = 1000
    n_sites = 200
    spins_array = np.zeros(n_sites)
    y1 = np.zeros(n_sites)
    y2 = np.zeros(n_sites)

    freq1 = 50
    freq2 = 50
    lambda1 = 18e-9
    lambda2 = 18e-9
    phi1 = 0
    phi2 = 0

    offset1 = 0
    offset2 = 0

    w1 = freq1 * 1e9 * 2 * np.pi
    w2 = freq2 * 1e9 * 2 * np.pi
    k1 = (2*np.pi) / lambda1
    k2 = (2*np.pi) / lambda2


    for t in np.linspace(0, max_simtime, max_dp):
        for x in spins_array:

            y1 = np.sin(k1 * (x + phi1) - w1 * t)
            y2 = np.sin(k2 * (x + phi2) - w2 * t)
            y = np.add(y1, y2)



def sine_wave_test():
    plt.rcParams.update({'savefig.dpi': 100, "figure.dpi": 100})

    interactive = True
    # Use for interactive plot. Also change DPI to 40 and allow Pycharm to plot outside of tool window
    if interactive:
        fig = plt.figure(figsize=(6, 6))

    num_dp = 5000  # Number of datapoints (min. 1000 for good quality images)
    max_simtime = 2  # Total simulated time [ns]
    max_distance = 6000  # Total simulated distance [nm]

    a = 2  # lattice constant [nm]

    # Initialise empty arrays to store data
    time_array = np.linspace(0, max_simtime, num_dp)
    x = np.linspace(0, max_distance, num_dp)
    y = np.zeros(num_dp)
    y1 = np.zeros(num_dp)
    y2 = np.zeros(num_dp)

    # Declare parameters
    freq1 = 50  # First driven frequency [GHz]
    freq2 = 54  # Second driven frequency [GHz]
    lambda1 = 18  # Wavelength of first driven wave [nm]
    lambda2 = 17  # Wavelength of second driven wave [nm]
    phi1 = 18 * 10 * a  # phase of first wave [nm]
    phi2 = 0 * a  # phase of second wave [nm]

    # Derive from declared parameters
    w1 = 2 * np.pi * freq1  # First driven angular frequency
    w2 = 2 * np.pi * freq2  # Second driven angular frequency
    k1 = (2*np.pi) / (lambda1 * a)  # wavenumber for first wave
    k2 = (2*np.pi) / (lambda2 * a)  # wavenumber for second wave

    for i, t in enumerate(time_array):
        y1 = np.sin(k1*(x + phi1) - w1*t)
        y2 = np.sin(k2*(x + phi2) - w2*t)
        y = np.add(y1, y2)

    # Plot output graph
    # fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.plot(x, y1, ls='--', color='red', label=f"{freq1} GHz", zorder=1.2)
    ax.plot(x, y2, ls=':', color='blue', label=f"{freq2} GHz", zorder=1.1)
    ax.plot(x, y,  ls='-', color='black', label="Superpos.",zorder=1.3)

    ax.set(title=f"$\phi_1$={phi1/a} & $\phi_2$={phi2/a}\n$\lambda_1$={lambda1} & $\lambda_2$={lambda2} | a={a}",
           xlabel="Position [nm]", ylabel="Amplitude [arb.]", xlim=[0, 3000])
    ax.xaxis.set(major_locator=ticker.MultipleLocator(500),
                   minor_locator=ticker.MultipleLocator(50))
    ax.legend(loc='lower right')
    fig.tight_layout()
    if interactive:
        # For interactive plots
        def mouse_event(event):
            print('x: {} and y: {}'.format(event.xdata, event.ydata))

        fig.canvas.mpl_connect('button_press_event', mouse_event)
        plt.show()
    else:
        plt.show()


if __name__ == '__main__':
    # loggingSetup()
    # rc_params_update()
    # square_number()
    # compare_dataset_plots()
    # test()
    # afm_test()
    sine_wave_test()
    #analytic_sine_wave()
    exit()
    # main()
