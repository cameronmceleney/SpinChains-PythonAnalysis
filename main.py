# -*- coding: utf-8 -*-

# -------------------------- Preprocessing Directives -------------------------

# Standard Libraries
import logging as lg
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from sys import exit
import globalvariables as gv

# 3rd Party packages
# Add here

# My packages/Header files

# ----------------------------- Program Information ----------------------------

"""
Description of what foo.py does
"""
PROGRAM_NAME = "ShockwavesFFT.py"
"""
Created on (date) by (author)
"""


# ---------------------------- Function Declarations ---------------------------

def fft_example():
    # Copy code from cpp_plot_rk2.py

    # Code from FFT example found online
    sample_rate = 44100  # Hertz
    total_duration = 5  # Seconds
    N = sample_rate * total_duration  # Number of samples in normalised tone

    def generate_sine_wave(select_frequency, select_sample_rate, select_duration):
        x = np.linspace(0, select_duration, select_sample_rate * select_duration, endpoint=False)
        frequencies = x * select_frequency  # 2pi because np.sin takes radians
        y = np.sin((2 * np.pi) * frequencies)
        return x, y

    # Generate a 2 hertz sine wave that lasts for 5 seconds

    _, nice_tone = generate_sine_wave(1000, sample_rate, total_duration)
    _, noise_tone = generate_sine_wave(4000, sample_rate, total_duration)
    noise_tone = noise_tone * 0.3

    mixed_tone = nice_tone + noise_tone

    normalised_tone = np.int16((mixed_tone / mixed_tone.max()) * 32767)
    # write("mysinewave.wav", SAMPLE_RATE, normalised_tone)

    # Calculates the frequencies in the centre of each bin in the output of fft()
    xf = fftpack.rfftfreq(N, 1.0 / sample_rate)
    yf = fftpack.rfft(normalised_tone)  # Calculates the transform itself

    # plt.plot(normalised_tone[:1000])
    plt.plot(xf, np.abs(yf))
    plt.xlim(0, 5000)
    plt.savefig(f"{gv.set_file_paths()[1]}test.png")
    plt.show()


def custom_fft(frequency, maximum_real_time, m_values, time_array):
    """
    Copy code from cpp_plot_rk2.py

                frequency               Frequency of the drivers                    [Hz]
                maximum_real_time       Total duration of the simulation            [s]
                m_values                Tuple containing magnetic moment values     [arb.]

    """

    sampling_rate = 1428571428571428

    # Code from FFT example found online # Hertz
    N = int(1 / 1e-6)  # Number of samples in normalised tone

    norm = np.linalg.norm(m_values)
    normal_magVals = m_values / norm

    # Calculates the frequencies in the centre of each bin in the output of fft()
    xf = fftpack.fftfreq(len(time_array), 1 / (2 * 1e6))
    yf = fftpack.fft(m_values)  # Calculates the transform itself

    fig2 = plt.figure()
    plt.title("FFT (freq)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("M$_x$")
    plt.plot(xf, np.abs(yf))
    # plt.xlim(0, 10e9)
    plt.yscale('log')
    # plt.ylim(1e-6, 1e-4)
    # fig2.savefig("D:\\Data\\03 Mar 22\\RK2 Shockwaves Tests Outputs\\LLGTest1626FFT.png")
    plt.show()


def new_custom_fft():
    # Produces a plot with the correct shape, but the wrong x-axis

    # Code from FFT example found online
    sample_rate = int(42.5e9)  # Hertz
    total_duration = int(1e-15 * 1.75e5 * 4e2)  # Seconds
    N = int(1 / 1e-6)  # Number of samples in normalised tone

    # Generate a 2 hertz sine wave that lasts for 5 seconds
    timeStamp = 1839
    FILE_IDENT = 'LLGTest'
    mx = np.loadtxt(open(f"{gv.set_file_paths()[0]}rk2_mx_{FILE_IDENT}{str(timeStamp)}.csv", "rb"), delimiter=",", skiprows=1)
    mx_time = mx[:, 0]
    mx_spin1 = mx[:, 1]
    # plt.plot(mx_time, mx_spin1) # Plot the single in the time domain
    # plt.xlim(0, 1e-8)

    # Frequency domain representation
    amplitude = mx_spin1
    samplingFrequency = int(42.5e9)
    fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
    fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
    tpCount = len(amplitude)
    values = np.arange(int(tpCount / 2))
    timePeriod = tpCount / samplingFrequency
    frequencies = values / timePeriod
    plt.plot(frequencies, abs(fourierTransform), marker='o', lw=0)
    lim = 1e7
    plt.xlim(0.4e7, 1.4e7)
    plt.yscale('log')
    plt.ylim(1e-6, 1e-3)
    plt.show()
    # Calculates the frequencies in the centre of each bin in the output of fft()
    # xf = fftpack.fftfreq(N, 1.0 / sample_rate)
    # yf = fftpack.fft(mx_spin1)  # Calculates the transform itself

    # plt.plot(normalised_tone[:1000])
    # plt.plot(xf, np.abs(yf))
    # plt.yscale('log')
    # plt.savefig(f"{gv.set_file_paths()[1]}test.png")
    # plt.show()


def new_custom_fft2():

    # Produces a plot with the correct shape, but the wrong x-axis

    timeStamp = 1839
    FILE_IDENT = 'LLGTest'
    mx = np.loadtxt(open(f"{gv.set_file_paths()[0]}rk2_mx_{FILE_IDENT}{str(timeStamp)}.csv", "rb"), delimiter=",", skiprows=1)
    mx_time = mx[:, 0]
    mx_spin1 = mx[:, 1]
    # plt.plot(mx_time, mx_spin1) # Plot the single in the time domain
    # plt.xlim(0, 0.5e-8)
    # plt.show()

    # Frequency domain representation
    timeInterval = 7e-8
    nSamples = len(mx_spin1)
    dt = timeInterval / nSamples

    amplitude = mx_spin1
    samplingFrequency = int(42.5e9)
    fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
    fourierTransform = fourierTransform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
    tpCount = len(amplitude)
    values = np.arange(int(tpCount / 2))
    timePeriod = tpCount / samplingFrequency
    frequencies = values / (dt * nSamples)
    plt.plot(frequencies, abs(fourierTransform), marker='o', lw=0, color='black')
    lim = 1e7
    plt.xlim(1e9, 0.6e10)
    plt.yscale('log')
    plt.ylim(1e-6, 1e-3)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.show()


def logging_setup():
    """
    Minimum Working Example (MWE) for logging. Pre-defined levels are:

        The highest        ---->           The lowest
        CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    """
    lg.basicConfig(filename='logfile.log',
                   filemode='w',
                   level=lg.DEBUG,
                   format='%(asctime)s - %(message)s',
                   datefmt='%d/%m/%Y %I:%M:%S %p',
                   force=True)


def plot_cpp_data(frequency):
    """
    Default values:
        freq = 42.5e9
    """

    FILE_IDENT = 'LLGTest'
    # timeStamp = input("Enter the unique identifier that all filenames will share: ")
    timeStamp = 1839

    lg.info(f"{PROGRAM_NAME} Begin importing data")
    # file_mx = csv.reader(open(f"{gv.set_file_paths()[0]}rk2_mx_{FILE_IDENT}{str(timeStamp)}.csv", "rb"), delimiter=",", skiprows=1)
    mx = np.loadtxt(open(f"{gv.set_file_paths()[0]}rk2_mx_{FILE_IDENT}{str(timeStamp)}.csv", "rb"), delimiter=",", skiprows=1)
    lg.info(f"{PROGRAM_NAME} Finish importing data")

    # Separate here into new function
    """
    Default values are:
        numberOfSpins = 4000
        total_datapoints = 100 (used to be called itermax)
        iterations = 7e5
    """

    stepsize = 1e-15
    iterations = 1.75e5 * 4e2
    maximum_real_time = int(stepsize * iterations)

    total_datapoints = int(1.0 / 1e-6)

    # targetSpin = int(input("Plot which spin: "))
    targetSpin = 1

    mx_time = mx[1:, 0]
    mx_spin = mx[:, targetSpin]

    lg.info(f"{PROGRAM_NAME} Begin FFT plotting")
    lg.info(f"{PROGRAM_NAME} Finish FFT plotting")

    # fig1 = plt.figure()
    # plt.title("Signal (x)")
    # plt.plot(np.arange(0, total_datapoints), mx_spin)
    # fig1.savefig(f"{gv.set_file_paths()[1]}{FILE_IDENT}{str(timeStamp)}.png")
    # plt.show()


# --------------------------- main() implementation ---------------------------

def main():
    lg.info(f"{PROGRAM_NAME} Start")

    new_custom_fft2()

    lg.info(f"{PROGRAM_NAME} End")

    exit()


# ------------------------------ Implementations ------------------------------

if __name__ == '__main__':
    logging_setup()

    main()
