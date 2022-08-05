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


# --------------------------- main() implementation ---------------------------

def main():
    lg.info(f"{PROGRAM_NAME} start")

    dataset1 = np.loadtxt("D:\\Data\\2022-08-05\\Simulation_Data\\rk2_mx_T1838.csv",
                          delimiter=",", skiprows=11)
    dataset2 = np.loadtxt("D:\\Data\\2022-08-05\\Simulation_Data\\rk2_mx_T1807.csv",
                          delimiter=",", skiprows=11)
    dataset3 = np.loadtxt("D:\\Data\\2022-08-05\\Simulation_Data\\rk2_mx_T1938.csv",
                          delimiter=",", skiprows=11)
    dataset4 = np.loadtxt("D:\\Data\\2022-08-05\\Simulation_Data\\rk2_mx_T1809.csv",
                          delimiter=",", skiprows=11)

    time = dataset1[:, 0] * 1e9
    data_to_plot1 = dataset1[:, 3000]
    data_to_plot2 = dataset2[:, 3000]
    data_to_plot3 = dataset3[:, 3000]
    data_to_plot4 = dataset4[:, 3000]

    fig = plt.figure(figsize=(4, 3))
    ax1 = fig.add_subplot(111)  #
    ax1.plot(time, abs(data_to_plot1), lw=1.0, label="0.05 ps", color='#648ABB', zorder=1)
    ax1.plot(time, abs(data_to_plot2), lw=0.75, label="50 ps", color='#37782c', zorder=2)
    ax1.plot(time, abs(data_to_plot3), lw=1.0, label="500 ps", color='#64bb6a', zorder=3)
    ax1.plot(time, abs(data_to_plot4), lw=1.0, label="1000 ps", color='#9fd983', zorder=4)
    ax1.set(xlim=[0, 5], ylim=[0, 7e-3],
            xlabel="Time [ns]", ylabel="abs(m$_x$/M$_S$)")
    ax1.legend(frameon=False, facecolor=None, edgecolor=None)
    ax1.xaxis.set(major_locator=ticker.MultipleLocator(2.5),
                  major_formatter=ticker.FormatStrFormatter("%.1f"),
                  minor_locator=ticker.MultipleLocator(0.5))
    ax1.yaxis.set(major_locator=ticker.MaxNLocator(nbins=3, prune='lower'),
                  major_formatter=ticker.FormatStrFormatter("%.1f"),
                  minor_locator=ticker.AutoMinorLocator(4))

    formatter = ticker.ScalarFormatter(useMathText=True)
    ax1.yaxis.set_major_formatter(formatter)
    ax1.yaxis.get_offset_text().set_visible(False)
    ax1.text(-0.05, 0.98, r'$\times \mathcal{10}^{{\mathcal{-3}}}$',
             verticalalignment='center', horizontalalignment='center', transform=ax1.transAxes, fontsize=8)

    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax1.grid(False)
    plt.show()
    fig.savefig("D:\\Data\\2022-08-05\\Outputs\\comparison.png", bbox_inches="tight")
    lg.info(f"{PROGRAM_NAME} end")
    exit()


# ------------------------------ Implementations ------------------------------

if __name__ == '__main__':
    loggingSetup()
    rc_params_update()
    main()
