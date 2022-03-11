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
import globalvariables as gv

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


def plot_data():
    """
    Imports a dataset in csv format, before plotting the signal and the corresponding FFT of the signal. Ensure
    that the first column of the dataset is the timestamps that each measurement was taken at.
    """
    rc_params_update()

    # timeStamp = input("Enter the unique identifier that all filenames will share: ")
    timeStamp = 1522  # Use this line to set a literal in order to minimise user inputs (good for testing)
    FILE_IDENT = 'LLGTest'  # This is the 'filename' variable in the C++ code

    lg.info(f"{PROGRAM_NAME} Begin importing data")
    spin_data_mx = np.loadtxt(open(f"{gv.generate_dir_tree()[0]}rk2_mx_{FILE_IDENT}{str(timeStamp)}.csv", "rb"),
                              delimiter=",", skiprows=1)
    lg.info(f"{PROGRAM_NAME} Finish importing data")

    # Each column of data is the magnetisation amplitudes at moments of time for a single spin
    mx_time = spin_data_mx[:, 0]  # First column of data file is always timestamps
    shouldContinuePlotting = True

    temporal_plot(mx_time, spin_data_mx[:, 1])
    exit(0)

    while shouldContinuePlotting:
        targetSpin = int(input("Plot which spin (-ve to exit): "))

        if targetSpin >= 1:
            temporal_plot(mx_time, spin_data_mx[:, 1])
        else:
            shouldContinuePlotting = False


def temporal_plot(time_data, y_axis_data):
    """
    Plots the given data showing the magnitude against time. Will output the 'signal'.
    """
    dicts = {0: {"title": "Time Domain Data1", "ylabel": "Test"},
             1: {"title": "Time Domain Data2", "xlabel": "Cake", "xlim": (0, 5)},
             2: {"title": "FFT", "xlabel": "Frequency [GHz]", "ylabel": "Amplitude [arb.]", "xlim": (0, 5), "yscale": 'log'},
             3: {"title": "FFT", "xlabel": "Frequency [GHz]", "xlim": (0, 40), "yscale": 'log'}}

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharey='row')

    for i, ax in enumerate(axes.flatten()):
        if i <= 1:
            custom_temporal_plot(time_data / 1e-9, y_axis_data, ax=ax, plt_kwargs=dicts[i])
        else:
            custom_fft_plot(y_axis_data, ax=ax, plt_kwargs=dicts[i])

    plt.suptitle("Data from Spin Site #1")
    plt.tight_layout()
    plt.show()


def custom_temporal_plot(x, y, ax=None, plt_kwargs={}):
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y)
    ax.set(**plt_kwargs)
    return ax

def custom_fft_plot(y, ax=None, plt_kwargs={}):

    stepsize = 2.857e-13
    total_iterations = 141 * 1e3
    gamma = 29.2  # 28.3
    H_static = 0.1
    freq_drive = 30.0  # In GHz
    # total_datapoints = 1e7  # Can be used as the 'number of samples'
    hz_to_ghz = 1e-9  # Multiply by this to convert Hz to GHz

    # Set values using simulation parameters
    timeInterval = stepsize * total_iterations
    nSamples = len(y)
    # Could also find this by multiplying the stepsize by the number of iterations between data recordings
    dt = timeInterval / nSamples

    # Compute the FFT
    fourierTransform = np.fft.fft(y)  # Normalize amplitude
    fourierTransform = fourierTransform[range(int(nSamples / 2))]  # Exclude sampling frequency
    frequencies = (np.arange(int(nSamples / 2)) / (dt * nSamples)) * hz_to_ghz

    if ax is None:
        ax = plt.gca()
    ax.plot(frequencies, abs(fourierTransform), marker='o', lw=1, color='red', markerfacecolor='black', markeredgecolor='black')
    ax.set(**plt_kwargs)
    ax.axvline(x=gamma * H_static, label=f"Natural. {gamma * H_static:2.2f}")
    ax.axvline(x=freq_drive, label=f"Driving. {freq_drive}", color='green')
    ax.legend(loc=1, bbox_to_anchor=(0.975, 0.975),
              frameon=True, fancybox=True, facecolor='white', edgecolor='white',
              title='Freq. List [GHz]', fontsize=12)

    ax.grid(color='white')
    return ax


def fft_plot(amplitude_data):
    """
    Uses simulation parameters to perform a Fast Fourier Transform (FFT) of the given data. This function only performs
    a single FFT on the given dataset, so it can be looped over by multiple function invocations.

    See https://mathematica.stackexchange.com/questions/105439/discrete-fourier-transform-help-on-how-to-convert-x-axis-in-to-the-frequency-wh
    for a Mathematica example.
    """
    stepsize = 2.857e-13
    total_iterations = 141 * 1e3
    gamma = 29.2  # 28.3
    H_static = 0.1
    freq_drive = 30.0  # In GHz
    # total_datapoints = 1e7  # Can be used as the 'number of samples'
    hz_to_ghz = 1e-9  # Multiply by this to convert Hz to GHz

    # Set values using simulation parameters
    timeInterval = stepsize * total_iterations
    nSamples = len(amplitude_data)
    # Could also find this by multiplying the stepsize by the number of iterations between data recordings
    dt = timeInterval / nSamples

    # Compute the FFT
    fourierTransform = np.fft.fft(amplitude_data)  # Normalize amplitude
    fourierTransform = fourierTransform[range(int(nSamples / 2))]  # Exclude sampling frequency
    frequencies = (np.arange(int(nSamples / 2)) / (dt * nSamples)) * hz_to_ghz

    # Plot the FFT and configure graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.plot(frequencies, abs(fourierTransform),
             marker='o', lw=1, color='red', markerfacecolor='black', markeredgecolor='black')

    # axes.vlines(x=29.2*0.2, colors='red', alpha=0.5, label='2.92')
    ax1.axvline(x=gamma * H_static, label=f"Natural. {gamma * H_static:2.2f}")
    ax1.axvline(x=freq_drive, label=f"Driving. {freq_drive}", color='green')

    ax1.set(title=f"FFT",
            xlabel="Frequency [GHz]", ylabel="Amplitude [arb.]",
            xlim=(0, 5),
            yscale='log')

    ax1.legend(loc=1, bbox_to_anchor=(0.975, 0.975),
               frameon=True, fancybox=True, facecolor='white', edgecolor='white',
               title='Freq. List [GHz]', fontsize=12)

    ax1.grid(color='white')

    ax2.plot(frequencies, abs(fourierTransform),
             marker='o', lw=1, color='red', markerfacecolor='black', markeredgecolor='black')

    # axes.vlines(x=29.2*0.2, colors='red', alpha=0.5, label='2.92')
    ax2.axvline(x=gamma * H_static, label=f"Natural. {gamma * H_static:2.2f}")
    ax2.axvline(x=freq_drive, label=f"Driving. {freq_drive}", color='green')
    ax2.axvline(x=(freq_drive * 3), label=f"Triple. {freq_drive * 3}", color='purple')

    ax2.set(title=f"FFT",
            xlabel="Frequency [GHz]", ylabel="Amplitude [arb.]",
            xlim=(0, 40),
            yscale='log')

    ax2.legend(loc=1, bbox_to_anchor=(0.975, 0.975),
               frameon=True, fancybox=True, facecolor='white', edgecolor='white',
               title='Freq. List [GHz]', fontsize=12)

    ax2.grid(color='white')

    fig.tight_layout()
    plt.show()


def logging_setup():
    # Initialisation of basic logging information. 
    lg.basicConfig(filename='logfile.log',
                   filemode='w',
                   level=lg.DEBUG,
                   format='%(asctime)s - %(message)s',
                   datefmt='%d/%m/%Y %I:%M:%S %p',
                   force=True)


def main():
    lg.info("Program start")

    plot_data()

    lg.info("Program end")

    exit()


if __name__ == '__main__':
    logging_setup()

    gv.generate_dir_tree()

    main()

    """
    Notes
    
    For the paper, linearFMR = (2 * np.pi * 28.3e9 / (2 * np.pi)) * np.sqrt(172e-6 * (172e-6 + 4 * np.pi * 0.086)) / 1e9
    """
