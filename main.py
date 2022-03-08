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
    Description of what globalvariables does
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

    timeStamp = input("Enter the unique identifier that all filenames will share: ")
    # timeStamp = 1320  # Use this line to set a literal in order to minimise user inputs (good for testing)
    FILE_IDENT = 'LLGTest'  # This is the 'filename' variable in the C++ code

    lg.info(f"{PROGRAM_NAME} Begin importing data")
    spin_data_mx = np.loadtxt(open(f"{gv.set_file_paths()[0]}rk2_mx_{FILE_IDENT}{str(timeStamp)}.csv", "rb"),
                              delimiter=",", skiprows=1)
    lg.info(f"{PROGRAM_NAME} Finish importing data")

    # Each column of data is the magnetisation amplitudes at moments of time for a single spin
    mx_time = spin_data_mx[:, 0]  # First column of data file is always timestamps
    shouldContinuePlotting = True

    temporal_plot(mx_time, spin_data_mx[:, 1])
    while shouldContinuePlotting:
        targetSpin = int(input("Plot which spin (-ve to exit): "))

        if targetSpin >= 1:
            mx_spin1 = spin_data_mx[:, targetSpin]
            # Invoke plotting functions
            # temporal_plot(mx_time, mx_spin1)
            fft_plot(mx_spin1)
        else:
            shouldContinuePlotting = False


def temporal_plot(time_data, y_axis_data):
    """
    Plots the given data showing the magnitude against time. Will output the 'signal'.
    """
    plt.plot(time_data, (y_axis_data/8.6e-6))  # Plot the single in the time domain
    plt.show()


def fft_plot(amplitude_data):
    """
    Uses simulation parameters to perform a Fast Fourier Transform (FFT) of the given data. This function only performs
    a single FFT on the given dataset, so it can be looped over by multiple function invocations.

    See https://mathematica.stackexchange.com/questions/105439/discrete-fourier-transform-help-on-how-to-convert-x-axis-in-to-the-frequency-wh
    for a Mathematica example.
    """
    stepsize = 4.82e-15 / 5
    total_iterations = 8.3022e6 * 5
    # total_datapoints = 1e7  # Can be used as the 'number of samples'
    hz_to_ghz = 1e-9

    # Set values using simulation parameters
    timeInterval = stepsize * total_iterations
    nSamples = len(amplitude_data)
    # Could also find this by multiplying the stepsize by the number of iterations between data recordings
    dt = timeInterval / nSamples

    # Compute the FFT
    fourierTransform = np.fft.fft(amplitude_data) / nSamples # Normalize amplitude
    fourierTransform = fourierTransform[range(int(nSamples / 2))]  # Exclude sampling frequency
    frequencies = (np.arange(int(nSamples / 2)) / (dt * nSamples)) * hz_to_ghz

    # Plot the FFT and configure graph
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))

    plt.plot(frequencies, abs(fourierTransform),
             marker='o', lw=1, color='red', markerfacecolor='black', markeredgecolor='black')

    # axes.vlines(x=2.92, ymin=1e-6, ymax=1e-3, colors='red', alpha=0.5, label='2.92')

    axes.set(title="FFT to Examine Peak Frequencies",
             xlabel="Frequency [GHz]", ylabel="Amplitude [arb.]",
             xlim=(0, 2), ylim=(1e-10, 1e-7),
             yscale='log')

    axes.legend(loc=1, bbox_to_anchor=(0.975, 0.975),
                frameon=True, fancybox=True, facecolor='white', edgecolor='white',
                title='Resonant\nFreq. [GHz]', fontsize=12)

    axes.grid(color='white')

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

    main()

    """
    Notes
    
    For the paper, linearFMR = (2 * np.pi * 28.3e9 / (2 * np.pi)) * np.sqrt(172e-6 * (172e-6 + 4 * np.pi * 0.086)) / 1e9
    """
