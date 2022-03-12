#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard libraries
import logging as lg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 3rd Party Packages
# Add here

# My packages / Any header files
import system_preparation as sp

"""
    The program 
"""
PROGRAM_NAME = "ShockwavesFFT.py"
"""
    Core Details
    
    Author      : cameronmceleney
    Created on  : 06/03/2022 22:08
    Filename    : main
    IDE         : PyCharm
"""


def rc_params_update():
    """Container for program's custom rc params, as well as Seaborn (library) selections"""
    sns.set(context='notebook', style='dark', font='Kohinoor Devanagari', palette='muted', color_codes=True)
    ##############################################################################
    # Sets global conditions including font sizes, ticks and sheet style
    # Sets various font size. fsize: general text. lsize: legend. tsize: title. ticksize: numbers next to ticks
    fsize = 18
    lsize = 12
    tsize = 24
    ticksize = 14

    # sets the tick direction. Options: 'in', 'out', 'inout'
    t_dir = 'in'
    # sets the tick size(s) and tick width(w) for the major and minor axes of all plots
    t_maj_s = 10
    t_min_s = 5
    t_maj_w = 1.2
    t_min_w = 1

    # updates rcParams of the selected style with my preferred options for these plots. Feel free to change
    plt.rcParams.update({'axes.titlesize': tsize, 'axes.labelsize': fsize, 'font.size': fsize, 'legend.fontsize': lsize,
                         'xtick.labelsize': ticksize, 'ytick.labelsize': ticksize,
                         'axes.edgecolor': 'black', 'axes.linewidth': 1.2,
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
                         'figure.titlesize': 24,
                         'figure.dpi': 300})


def data_analysis(time_stamp=None, file_identifier='LLGTest'):
    """
    Import a dataset in csv format, and plot the signal and the corresponding FFT.

    -----
    Notes
    -----

    Ensure that the first column of the dataset are the timestamps that each measurement was taken at. If this is not
    the case, then replace the variable 'mx_time' with an array of values:

    * mx_time = np.linspace(start_time, end_time, number_of_iterations, endpoint=True)

    :param int time_stamp: The file_ext variable in the C++ code. Set as a function argument to reduce user inputs
    :param str file_identifier: This is the 'filename' variable in the C++ code.

    :return: Nothing.
    """
    rc_params_update()

    if time_stamp is None:
        # time_stamp must be declared if none was provided as a function argument
        time_stamp = str(input("Enter the unique identifier that all filenames will share: "))

    # Tracking how long the data import took is important for monitoring large files.
    lg.info(f"{PROGRAM_NAME} Beginning to import data")
    # Each column of data is the magnetisation amplitudes at a moments of time for a single spin site
    mx_all_data = np.loadtxt(open(f"{sp.generate_dir_tree()[0]}rk2_mx_{file_identifier}{str(time_stamp)}.csv", "rb"),
                             delimiter=",", skiprows=1)
    lg.info(f"{PROGRAM_NAME} Finished importing data")

    # First column of data file is always the real-time at that iteration.
    mx_time = mx_all_data[:, 0]

    shouldContinuePlotting = True

    while shouldContinuePlotting:
        # User will plot data one spin site at a time, as each plot can take an extended amount of time to create

        # target_spin = int(input("Plot which spin (-ve to exit): "))
        target_spin = 1

        if target_spin >= 1:

            generate_site_figure(mx_time, mx_all_data[:, target_spin], target_spin)
            shouldContinuePlotting = False
        else:
            shouldContinuePlotting = False


def generate_site_figure(time_data, y_axis_data, spin_site):
    """
    Plots the given data showing the magnitude against time. Will output the 'signal'.
    """
    plot_set_params = {0: {"title": "Focused View", "xlabel": "Time [ns]", "ylabel": "Amplitude [normalised]", "xlim": (0, 5)},
                       1: {"title": "Full Simulation","xlabel": "Time [ns]", "xlim": (0, 40)},
                       2: {"title": "First Resonant Freq. Region", "xlabel": "Frequency [GHz]", "ylabel": "Amplitude [arb.]", "xlim": (0, 5), "yscale": 'log'},
                       3: {"title": "All Artefacts", "xlabel": "Frequency [GHz]", "xlim": (0, 30), "yscale": 'log'}}

    fig = plt.figure(figsize=(12, 12), constrained_layout=True, )
    fig.suptitle(f"Data from Spin Site #{spin_site}")

    # create 2x1 subfigs
    subfigs = fig.subfigures(nrows=2, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f'Subfigure title {row}')

        if row == 0:
            # create 1x2 subplots per subfig
            axs = subfig.subplots(nrows=1, ncols=2)
            for col, ax in enumerate(axs):
                custom_temporal_plot(time_data / 1e-9, y_axis_data, ax=ax, plt_kwargs=plot_set_params[col])

        else:
            axs = subfig.subplots(nrows=1, ncols=2)
            for col, ax in enumerate(axs):
                col += 2
                custom_fft_plot(y_axis_data, ax=ax, plt_kwargs=plot_set_params[col])

    plt.show()


def custom_temporal_plot(x, y, ax=None, plt_kwargs={}):
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y)
    ax.set(**plt_kwargs)
    return ax


def custom_fft_plot(y, ax=None, plt_kwargs={}):

    frequencies = fft_data[0]
    fourierTransform = fft_data[1]
    natural_freq = fft_data[2]
    drive_freq = fft_data[3]

    if ax is None:
        ax = plt.gca()
    ax.plot(frequencies, abs(fourierTransform), marker='o', lw=1, color='red', markerfacecolor='black',
            markeredgecolor='black')
    ax.set(**plt_kwargs)
    ax.axvline(x=natural_freq, label=f"Natural. {natural_freq:2.2f}")
    ax.axvline(x=drive_freq, label=f"Driving. {drive_freq}", color='green')
    ax.legend(loc=1, bbox_to_anchor=(0.975, 0.975),
              frameon=True, fancybox=True, facecolor='white', edgecolor='white',
              title='Freq. List [GHz]', fontsize=12)

    ax.grid(color='white')
    return ax


def fft_data():
    sim_params = {"stepsize": 2.857e-13,
                  "total_iterations": 141 * 1e3,
                  "gamma": 29.2,
                  "H_static": 0.1,
                  "freq_drive": 3.5,
                  "total_datapoints": 1e7,
                  "hz_to_ghz": 1e-9}

    natural_freq = sim_params['gamma'] * sim_params['H_static']

    # Set values using simulation parameters
    timeInterval = sim_params['stepsize'] * sim_params['total_iterations']
    nSamples = len(y)
    # Could also find this by multiplying the stepsize by the number of iterations between data recordings
    dt = timeInterval / nSamples

    # Compute the FFT
    fourierTransform = np.fft.fft(y)  # Normalize amplitude
    fourierTransform = fourierTransform[range(int(nSamples / 2))]  # Exclude sampling frequency
    frequencies = (np.arange(int(nSamples / 2)) / (dt * nSamples)) * sim_params["hz_to_ghz"]

    return frequencies, fourierTransform, natural_freq, sim_params['freq_drive']


def logging_setup():
    """Initialisation of basic logging information."""
    lg.basicConfig(filename='logfile.log',
                   filemode='w',
                   level=lg.DEBUG,
                   format='%(asctime)s - %(message)s',
                   datefmt='%d/%m/%Y %I:%M:%S %p',
                   force=True)


def main():
    """All functions should be initialised here (excluding core operating features like logging)."""
    lg.info("Program start")

    data_analysis(1522)

    lg.info("Program end")

    exit()


if __name__ == '__main__':
    logging_setup()

    sp.generate_dir_tree()

    main()

    """
    Notes
    
    For the paper, linearFMR = (2 * np.pi * 28.3e9 / (2 * np.pi)) * np.sqrt(172e-6 * (172e-6 + 4 * np.pi * 0.086)) / 1e9
    """
