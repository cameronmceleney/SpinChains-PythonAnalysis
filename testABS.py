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
    #ax1.indicate_inset_zoom(ax1_inset, facecolor='#f9f2e9', edgecolor='black', alpha=1.0, lw=0.75, zorder=1)

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
    #ax1.locator_params(axis='y', nbins=3)
    locmin = ticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.01, numticks=15)
    ax1.yaxis.set_minor_locator(locmin)
    ax1.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax1.set_yticks([1e-4, 1e-3, 2e-3, 3e-3])
    #ax1.yaxis.set(major_locator=ticker.LogLocator(base=10, numticks=15))
    #locmin = ticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=15)
    #ax1.yaxis.set_minor_locator(locmin)
    #ax1.yaxis.set_minor_formatter(ticker.NullFormatter())
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
    # omega = np.linspace(0, 300, 1000)  # Angular frequency is in rad*GHz
    t_range = 0.5
    h_0 = np.linspace(-t_range, t_range, 1000)
    gamma = 28.8
    h_e = 53  # In tesla
    h_a = 0.787 #  In tesla

    omega_pos = gamma * (np.sqrt(2 * h_e * h_a + h_a**2) + h_0)
    omega_neg = gamma * (np.sqrt(2 * h_e * h_a + h_a**2) - h_0)

    fig = plt.figure(figsize=(4, 2))
    ax = fig.add_subplot(1, 1, 1)

    # # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position('center')
#
    # # Eliminate upper and right axes
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
#
    # # Show ticks in the left and lower axes only
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')

    ax.plot(h_0, omega_pos, label="$\omega_{pos}$")
    ax.plot(h_0, omega_neg, label="$\omega_{neg}$")

    ax.set(xlabel="$H_0$[T]", ylabel="$\omega$ [GHz]")
    ax.legend()

    fig.tight_layout()
    fig.savefig("D:\\Data\\2022-10-03\\Outputs\\test.png", bbox_inches="tight")

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
    ax.plot(x2f, np.abs(y2f), marker='', lw=1.0, ls='-', color='#64bb6a', markerfacecolor='black', markeredgecolor='black',
            label="0", zorder=1.3)

    ax.set(xlim=(5, 25), ylim=(1e0, 1e4),
           xlabel="Frequency [GHz]", ylabel="\n\nAmplitude [arb.]",yscale='log')
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
    #ax2_inset.set(xlim=[0, 20], ylim=[-1, 1],
    #              yscale="linear")
    #ax2_inset.set_yticks([])

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

def test():

    omega = 20e9
    t = np.linspace(0, 1e-9, 10000)
    drive = np.cos(omega * t)
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(t, drive)
    plt.show()
# ------------------------------ Implementations ------------------------------

if __name__ == '__main__':
    # loggingSetup()
    #rc_params_update()
    #square_number()
    #compare_dataset_plots()
    test()
    #afm_test()
    exit()
    #main()
