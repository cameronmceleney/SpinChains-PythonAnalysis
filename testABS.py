# -*- coding: utf-8 -*-

# -------------------------- Preprocessing Directives -------------------------

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
# Here

# ----------------------------- Program Information ----------------------------

"""
Description of what foo.py does
"""
PROGRAM_NAME = "foo.py"
"""
Created on (date) by (author)
"""


# ---------------------------- Function Declarations ---------------------------

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
    #plt.style.use('fivethirtyeight')
    sns.set(context='notebook', font='Kohinoor Devanagari', palette='muted', color_codes=True)
    ##############################################################################
    # Sets global conditions including font sizes, ticks and sheet style
    # Sets various font size. fsize: general text. lsize: legend. tsize: title. ticksize: numbers next to ticks
    medium_size = 14
    small_size = 12
    large_size = 16
    smaller_size = 10
    tiny_size = 8

    # sets the tick direction. Options: 'in', 'out', 'inout'
    t_dir = 'in'
    # sets the tick size(s) and tick width(w) for the major and minor axes of all plots
    t_maj_s = 4
    t_min_s = t_maj_s / 2
    t_maj_w = 0.8
    t_min_w = t_maj_w / 2

    # updates rcParams of the selected style with my preferred options for these plots. Feel free to change
    plt.rcParams.update({'axes.titlesize': medium_size, 'axes.labelsize': small_size, 'font.size': small_size,
                         'legend.fontsize': tiny_size,
                         'figure.titlesize': large_size,
                         'xtick.labelsize': small_size, 'ytick.labelsize': small_size,
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
                         'savefig.dpi': 1500, "figure.dpi": 1000})


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

    fig = plt.figure(figsize=(4, 3))
    ax1 = fig.add_subplot(111)  # 64BB6A, #648ABB, #BB64B5, #BB9664
    ax1.plot(time, abs(data_to_plot1), lw=0.5, marker='o', markersize=0, label="Instant", color='#fd7f6f', zorder=1)
    ax1.plot(time, abs(data_to_plot2), lw=0.5, marker='o', markersize=0, label="0.05", color='#7eb0d5', zorder=2)
    ax1.plot(time, abs(data_to_plot3), lw=0.5, marker='o', markersize=0, label="0.1", color='#b2e061', zorder=3)
    ax1.plot(time, abs(data_to_plot4), lw=0.5, marker='o', markersize=0, label="0.25", color='#bd7ebe', zorder=4)
    ax1.plot(time, abs(data_to_plot5), lw=0.5, marker='o', markersize=0, label="0.5", color='#ffb55a', zorder=5)
    ax1.plot(time, abs(data_to_plot6), lw=0.5, marker='o', markersize=0, label="1.0", color='#beb9db', zorder=6)

    ax1.fill_between(time, abs(data_to_plot1), color='#fd7f6f', zorder=1)
    ax1.fill_between(time, abs(data_to_plot2), color='#7eb0d5', zorder=2)
    ax1.fill_between(time, abs(data_to_plot3), color='#b2e061', zorder=3)
    ax1.fill_between(time, abs(data_to_plot4), color='#bd7ebe', zorder=4)
    ax1.fill_between(time, abs(data_to_plot5), color='#ffb55a', zorder=5)
    ax1.fill_between(time, abs(data_to_plot6), color='#beb9db', zorder=6)

    # x_maximums1 = []
    # y_maximums1 = []

    # for i in range(2, len(data_to_plot1) - 2):
    #     if data_to_plot1[i - 2] < data_to_plot1[i - 1] and data_to_plot1[i - 1] < data_to_plot1[i] and data_to_plot1[i + 2] < data_to_plot1[i + 1] and \
    #             data_to_plot1[i + 1] < data_to_plot1[i]:
    #         y_maximums1.append(data_to_plot1[i])
    #         x_maximums1.append(time[i])
    # ax1.plot(x_maximums1, y_maximums1, lw=1, zorder=10, label="Instant")
    #
    # x_maximums2 = []
    # y_maximums2 = []
    #
    # for i in range(2, len(data_to_plot2) - 2):
    #    if data_to_plot2[i - 2] < data_to_plot2[i - 1] and data_to_plot2[i - 1] < data_to_plot2[i] and data_to_plot2[i + 2] < data_to_plot2[i + 1] and \
    #            data_to_plot2[i + 1] < data_to_plot2[i]:
    #        y_maximums2.append(data_to_plot2[i])
    #        x_maximums2.append(time[i])
    # ax1.plot(x_maximums2, y_maximums2, lw=1, zorder=2, color='#648ABB')
    # ax1.fill_between(x_maximums2, y_maximums2, color='#648ABB', zorder=2)
    #
    # x_maximums3 = []
    # y_maximums3 = []
    #
    # for i in range(2, len(data_to_plot3) - 2):
    #     if data_to_plot3[i - 2] < data_to_plot3[i - 1] and data_to_plot3[i - 1] < data_to_plot3[i] and data_to_plot3[i + 2] < data_to_plot3[i + 1] and \
    #             data_to_plot3[i + 1] < data_to_plot3[i]:
    #         y_maximums3.append(data_to_plot3[i])
    #         x_maximums3.append(time[i])
    # ax1.plot(x_maximums3, y_maximums3, lw=1, zorder=10, label="0.5 ns")
    #
    # x_maximums4 = []
    # y_maximums4 = []

    # for i in range(2, len(data_to_plot4) - 2):
    #    if data_to_plot4[i - 2] < data_to_plot4[i - 1] and data_to_plot4[i - 1] < data_to_plot4[i] and data_to_plot4[i + 2] < data_to_plot4[i + 1] and \
    #            data_to_plot4[i + 1] < data_to_plot4[i]:
    #        y_maximums4.append(data_to_plot4[i])
    #        x_maximums4.append(time[i])
    # ax1.plot(x_maximums4, y_maximums4, lw=0, zorder=10, label="1.0 ns", color='red')
    #
    # x_maximums5 = []
    # y_maximums5 = []
    #
    # for i in range(2, len(data_to_plot5) - 2):
    #     if data_to_plot5[i - 2] < data_to_plot5[i - 1] and data_to_plot5[i - 1] < data_to_plot5[i] and data_to_plot5[i + 2] < data_to_plot5[i + 1] and \
    #             data_to_plot5[i + 1] < data_to_plot5[i]:
    #         y_maximums5.append(data_to_plot5[i])
    #         x_maximums5.append(time[i])
    # ax1.plot(x_maximums5, y_maximums5, lw=1, zorder=10, label="0.1 ns")
    #
    # x_maximums6 = []
    # y_maximums6 = []
    #
    # for i in range(2, len(data_to_plot6) - 2):
    #    if data_to_plot6[i - 2] < data_to_plot6[i - 1] and data_to_plot6[i - 1] < data_to_plot6[i] and data_to_plot6[i + 2] < data_to_plot6[i + 1] and \
    #            data_to_plot6[i + 1] < data_to_plot6[i]:
    #        y_maximums6.append(data_to_plot6[i])
    #        x_maximums6.append(time[i])
    # ax1.plot(x_maximums6, y_maximums6, lw=1, zorder=6, label="1.0 ns")
    # ax1.fill_between(x_maximums6, y_maximums6, color='#BB9664', zorder=6)

    ax1.set(xlim=[0.5, 5], ylim=[1e-9, 1e-2],
            xlabel="Time [ns]", ylabel="|m$_x$|/M$_S$",
            yscale="log")

    # ax1.xaxis.set(major_locator=ticker.MultipleLocator(1.0), major_formatter=ticker.FormatStrFormatter("%.1f"),
    #             minor_locator=ticker.MultipleLocator(0.2))
    # ax1.yaxis.set(major_locator=ticker.LogLocator(base=10, numticks=3))
    # locmin = ticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=4)
    # ax1.yaxis.set_minor_locator(locmin)
    # ax1.yaxis.set_minor_formatter(ticker.NullFormatter())

    # formatter = ticker.ScalarFormatter(useMathText=True)
    # ax1.yaxis.set_major_formatter(formatter)
    ax1.grid(False)

    for spine in ['top', 'bottom', 'left', 'right']:
        ax1.spines[spine].set_visible(True)
    for k, spine in ax1.spines.items():  # ax.spines is a dictionary
        spine.set_zorder(10)
    ax1.margins(0.1)
    ax1.tick_params(axis="both", which="both", zorder=10, bottom=True, top=True, left=True, right=True)
    ax1.legend(title="t$_g$ [ns]", ncol=2, loc="lower right",
               frameon=True, fancybox=True, facecolor='white', edgecolor='black',
               fontsize=6, title_fontsize=8).set_zorder(20)

    fig.savefig("D:\\Data\\2022-08-10\\Outputs\\comparison.png", bbox_inches="tight")


# --------------------------- main() implementation ---------------------------

def main():
    lg.info(f"{PROGRAM_NAME} start")

    # compare_dataset_plots()

    # t = np.linspace(0, 5, 2000)
    #
    # y_1 = np.zeros(200)
    # y_2 = 3e-3 * np.sin(2*np.pi*15 * t[200:])
    # y = np.concatenate((y_1, y_2))

    SAMPLE_RATE = int(1000)  # Number of samples per nanosecond
    DURATION = int(4)  # Nanoseconds

    def generate_sine_wave(freq, sample_rate, duration):
        delay = int(sample_rate * 2)
        t = np.linspace(0, duration, sample_rate * duration, endpoint=False)
        y_1 = np.zeros(delay)
        y_2 = np.sin((2 * np.pi * freq) * t[delay:])
        y = np.concatenate((y_1, y_2))
        return t, y

    # Generate a 15 GHz sine wave that lasts for 5 seconds
    x, y = generate_sine_wave(15, SAMPLE_RATE, DURATION)
    # plt.plot(x, y)
    # plt.show()

    from scipy.fft import rfft, rfftfreq
    # Number of samples in normalized_tone
    N = int(SAMPLE_RATE * DURATION)

    yf = rfft(y)
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    fig = plt.figure(figsize=(18.73 / 2.54, 4.94 / 2.54))
    ax = fig.add_subplot(111)
    ax.plot(xf, np.abs(yf), marker='', lw=1, color='#64bb6a', markerfacecolor='black', markeredgecolor='black',
            label="Shockwave", zorder=2)
    ax.set(xlim=(0, 30),
           xlabel="Frequency [GHz]", ylabel="\n\nAmplitude [arb.]")
    ax.grid(visible=False, axis='both', which='both')
    ax.tick_params(top="on", right="on", which="both")

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
    ax2_inset = inset_axes(ax, width=2.75, height=0.9, loc="upper left", bbox_to_anchor=[0.1, 0.85],
                           bbox_transform=ax.figure.transFigure)
    ax2_inset.plot(x, y, lw=0.75, color='#64bb6a', zorder=1)
    ax2_inset.grid(visible=False, axis='both', which='both')
    # ax2_inset.yaxis.tick_right()
    ax2_inset.set_xlabel('Time [ns]', fontsize=10)
    ax2_inset.tick_params(axis='both', labelsize=8)
    ax2_inset.set_yticks([])
    ax2_inset.patch.set_color("#f9f2e9")

    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax2_inset.spines[spine].set_visible(True)

    ax.xaxis.set(major_locator=ticker.MultipleLocator(5),
                 minor_locator=ticker.MultipleLocator(1))
    ax2_inset.xaxis.set(major_locator=ticker.MultipleLocator(2),
                        minor_locator=ticker.MultipleLocator(0.5))
    ax.yaxis.set(major_locator=ticker.MaxNLocator(nbins=3, prune='lower'),
                 minor_locator=ticker.AutoMinorLocator(4))

    formatter = ticker.ScalarFormatter(useMathText=True)
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_visible(False)

    fig.savefig("D:\\Data\\2022-08-10\\Outputs\\image2.png", bbox_inches="tight", dpi=1000)

    lg.info(f"{PROGRAM_NAME} end")
    exit()


# ------------------------------ Implementations ------------------------------

if __name__ == '__main__':
    # loggingSetup()
    rc_params_update()
    main()
