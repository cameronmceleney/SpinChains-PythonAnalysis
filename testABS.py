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
    plt.style.use('fivethirtyeight')
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


def compare_dataset_plots():
    spin_site = 3000

    dataset1 = np.loadtxt("D:\\Data\\2022-08-10\\Simulation_Data\\rk2_mx_T0940.csv",
                          delimiter=",", skiprows=11)
    dataset2 = np.loadtxt("D:\\Data\\2022-08-10\\Simulation_Data\\rk2_mx_T1006.csv",
                          delimiter=",", skiprows=11)
    dataset3 = np.loadtxt("D:\\Data\\2022-08-10\\Simulation_Data\\rk2_mx_T0955.csv",
                          delimiter=",", skiprows=11)
    dataset4 = np.loadtxt("D:\\Data\\2022-08-10\\Simulation_Data\\rk2_mx_T0929.csv",
                          delimiter=",", skiprows=11)
    dataset5 = np.loadtxt("D:\\Data\\2022-08-10\\Simulation_Data\\rk2_mx_T1550.csv",
                          delimiter=",", skiprows=11)
    dataset6 = np.loadtxt("D:\\Data\\2022-08-10\\Simulation_Data\\rk2_mx_T1602.csv",
                          delimiter=",", skiprows=11)

    time = dataset1[:, 0] * 1e9
    data_to_plot1 = dataset1[:, spin_site]
    data_to_plot2 = dataset2[:, spin_site]
    data_to_plot3 = dataset3[:, spin_site]
    data_to_plot4 = dataset4[:, spin_site]
    data_to_plot5 = dataset5[:, spin_site]
    data_to_plot6 = dataset6[:, spin_site]

    fig = plt.figure(figsize=(4, 3))
    ax1 = fig.add_subplot(111)  #
    #ax1.plot(time, abs(data_to_plot1), lw=1.00, marker='o', markersize=1, label="Instant", color='#64BB6A', zorder=1)
    #ax1.plot(time, abs(data_to_plot2), lw=0.75, marker='o', markersize=1, label="0.05 ns", color='#648ABB', zorder=2)
    #ax1.plot(time, abs(data_to_plot5), lw=1.00, marker='o', markersize=1, label="0.1 ns", color='#BB64B5', zorder=3)
    #ax1.plot(time, abs(data_to_plot6), lw=1.00, marker='o', markersize=1, label="0.25 ns", color='red', zorder=4)
    #ax1.plot(time, abs(data_to_plot3), lw=1.00, marker='o', markersize=1, label="0.5 ns", color='blue', zorder=5)
    #ax1.plot(time, abs(data_to_plot4), lw=1.00, marker='o', markersize=1, label="1.0 ns", color='#BB9664', zorder=6)
    ax1.set(xlim=[0.5, 5], ylim=[1e-10, 1e-2],
            xlabel="Time [ns]", ylabel="abs(m$_x$/M$_S$)",
            yscale="log")
    ax1.legend(loc="lower right", frameon=False, facecolor=None, edgecolor=None)

    x_maximums1 = []
    y_maximums1 = []

    for i in range(2, len(data_to_plot1) - 2):
        if data_to_plot1[i - 2] < data_to_plot1[i - 1] and data_to_plot1[i - 1] < data_to_plot1[i] and data_to_plot1[i + 2] < data_to_plot1[i + 1] and \
                data_to_plot1[i + 1] < data_to_plot1[i]:
            y_maximums1.append(data_to_plot1[i])
            x_maximums1.append(time[i])
    ax1.plot(x_maximums1, y_maximums1, lw=1, zorder=10)

    x_maximums2 = []
    y_maximums2 = []

    for i in range(2, len(data_to_plot2) - 2):
        if data_to_plot2[i - 2] < data_to_plot2[i - 1] and data_to_plot2[i - 1] < data_to_plot2[i] and data_to_plot2[i + 2] < data_to_plot2[i + 2] and \
                data_to_plot2[i + 1] < data_to_plot2[i]:
            y_maximums2.append(data_to_plot2[i])
            x_maximums2.append(time[i])
    ax1.plot(x_maximums2, y_maximums2, lw=1, zorder=10)

    x_maximums3 = []
    y_maximums3 = []

    for i in range(2, len(data_to_plot3) - 2):
        if data_to_plot3[i - 2] < data_to_plot3[i - 1] and data_to_plot3[i - 1] < data_to_plot3[i] and data_to_plot3[i + 2] < data_to_plot3[i + 1] and \
                data_to_plot3[i + 1] < data_to_plot3[i]:
            y_maximums3.append(data_to_plot3[i])
            x_maximums3.append(time[i])
    ax1.plot(x_maximums3, y_maximums3, lw=1, zorder=10)

    x_maximums4 = []
    y_maximums4 = []

    for i in range(2, len(data_to_plot4) - 2):
        if data_to_plot4[i - 2] < data_to_plot4[i - 1] and data_to_plot4[i - 1] < data_to_plot4[i] and data_to_plot4[i + 2] < data_to_plot4[i + 1] and \
                data_to_plot4[i + 1] < data_to_plot4[i]:
            y_maximums4.append(data_to_plot4[i])
            x_maximums4.append(time[i])
    ax1.plot(x_maximums4, y_maximums4, lw=1, zorder=10)

    x_maximums5 = []
    y_maximums5 = []

    for i in range(2, len(data_to_plot5) - 2):
        if data_to_plot5[i - 2] < data_to_plot5[i - 1] and data_to_plot5[i - 1] < data_to_plot5[i] and data_to_plot5[i + 2] < data_to_plot5[i + 1] and \
                data_to_plot5[i + 1] < data_to_plot5[i]:
            y_maximums5.append(data_to_plot5[i])
            x_maximums5.append(time[i])
    ax1.plot(x_maximums5, y_maximums5, lw=1, zorder=10)

    x_maximums6 = []
    y_maximums6 = []

    for i in range(2, len(data_to_plot6) - 2):
        if data_to_plot6[i - 2] < data_to_plot6[i - 1] and data_to_plot6[i - 1] < data_to_plot6[i] and data_to_plot6[i + 2] < data_to_plot6[i + 1] and \
                data_to_plot6[i + 1] < data_to_plot6[i]:
            y_maximums6.append(data_to_plot6[i])
            x_maximums6.append(time[i])
    ax1.plot(x_maximums6, y_maximums6, lw=1, zorder=10)

    # ax1.xaxis.set(major_locator=ticker.MultipleLocator(2.5),
    #               major_formatter=ticker.FormatStrFormatter("%.1f"),
    #               minor_locator=ticker.MultipleLocator(0.5))
    # ax1.yaxis.set(major_locator=ticker.MaxNLocator(nbins=3, prune='lower'),
    #               major_formatter=ticker.FormatStrFormatter("%.1f"),
    #               minor_locator=ticker.AutoMinorLocator(4))

    # ax1.xaxis.set(major_locator=ticker.MultipleLocator(2.5), major_formatter=ticker.FormatStrFormatter("%.1f"),
    #              minor_locator=ticker.MultipleLocator(0.5))
    # ax1.yaxis.set(major_locator=ticker.LogLocator(base=10, numticks=6),
    #               minor_locator=ticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=12))

    # formatter = ticker.ScalarFormatter(useMathText=True)
    # ax1.yaxis.set_major_formatter(formatter)
    ax1.yaxis.get_offset_text().set_visible(True)
    # ax1.text(-0.05, 0.98, r'$\times \mathcal{10}^{{\mathcal{-3}}}$',
    #         verticalalignment='center', horizontalalignment='center', transform=ax1.transAxes, fontsize=8)

    # ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax1.grid(False)

    fig.savefig("D:\\Data\\2022-08-10\\Outputs\\comparison.png", bbox_inches="tight")


# --------------------------- main() implementation ---------------------------

def main():
    lg.info(f"{PROGRAM_NAME} start")

    compare_dataset_plots()

    # t = np.linspace(0, 5, 2000)
    #
    # y_1 = np.zeros(200)
    # y_2 = 3e-3 * np.sin(2*np.pi*15 * t[200:])
    # y = np.concatenate((y_1, y_2))

    #SAMPLE_RATE = int(1000)  # Number of samples per nanosecond
    #DURATION = int(4)  # Nanoseconds
#
    #def generate_sine_wave(freq, sample_rate, duration):
    #    delay = int(sample_rate * 2)
    #    t = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    #    y_1 = np.zeros(delay)
    #    y_2 = np.sin((2 * np.pi * freq) * t[delay:])
    #    y = np.concatenate((y_1, y_2))
    #    return t, y
#
    ## Generate a 15 GHz sine wave that lasts for 5 seconds
    #x, y = generate_sine_wave(15, SAMPLE_RATE, DURATION)
    ## plt.plot(x, y)
    ## plt.show()
#
    #from scipy.fft import rfft, rfftfreq
    ## Number of samples in normalized_tone
    #N = int(SAMPLE_RATE * DURATION)
#
    #yf = rfft(y)
    #xf = rfftfreq(N, 1 / SAMPLE_RATE)
#
    #fig = plt.figure(figsize=(4, 2))
    #ax = fig.add_subplot(111)
    #ax.plot(xf, np.abs(yf), marker='', lw=1, color='#64bb6a', markerfacecolor='black', markeredgecolor='black',
    #        label="Shockwave", zorder=2)
#
    #ax.set(xlim=(0, 30),
    #       xlabel="Frequency [GHz]", ylabel="Amplitude [arb.]")
    #ax.grid(visible=False, axis='both', which='both')
    #ax.tick_params(top="on", right="on", which="both")
#
    #from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
    #ax2_inset = inset_axes(ax, width=1.2, height=0.5, loc="upper left", bbox_to_anchor=[0.085, 0.875],
    #                       bbox_transform=ax.figure.transFigure)
    #ax2_inset.plot(x, y, lw=0.75, color='#64bb6a')
    #ax2_inset.grid(visible=False, axis='both', which='both')
    #ax2_inset.set_xticks([])
    #ax2_inset.set_yticks([])
    #ax2_inset.patch.set_color("#f9f2e9")
#
    #for spine in ['top', 'bottom', 'left', 'right']:
    #    ax.spines[spine].set_visible(True)
    #    ax2_inset.spines[spine].set_visible(True)
#
#
#
    #ax.xaxis.set(major_locator=ticker.MultipleLocator(5),
    #             minor_locator=ticker.MultipleLocator(1))
    #ax.yaxis.set(major_locator=ticker.MaxNLocator(nbins=3, prune='lower'),
    #             minor_locator=ticker.AutoMinorLocator(4))
#
    #formatter = ticker.ScalarFormatter(useMathText=True)
    #ax.yaxis.set_major_formatter(formatter)
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    #ax.yaxis.get_offset_text().set_visible(False)
#
#
#
    #fig.savefig("D:\\Data\\2022-08-10\\Outputs\\image.png", bbox_inches="tight")

    lg.info(f"{PROGRAM_NAME} end")
    exit()


# ------------------------------ Implementations ------------------------------

if __name__ == '__main__':
    # loggingSetup()
    rc_params_update()
    main()
