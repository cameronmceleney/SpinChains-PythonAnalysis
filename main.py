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


def custom_fft(frequency, maximum_real_time, m_values):
    """
    Copy code from cpp_plot_rk2.py

                frequency               Frequency of the drivers                    [Hz]
                maximum_real_time       Total duration of the simulation            [s]
                m_values                Tuple containing magnetic moment values     [arb.]

    """

    # Code from FFT example found online # Hertz
    N = int(1 / 1e-6)  # Number of samples in normalised tone

    norm = np.linalg.norm(m_values)
    normal_magVals = m_values / norm

    # Calculates the frequencies in the centre of each bin in the output of fft()
    xf = fftpack.rfftfreq(N, 1 / int(frequency))
    yf = fftpack.rfft(normal_magVals)  # Calculates the transform itself

    fig2 = plt.figure()
    plt.title("FFT (freq)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("M$_x$")
    plt.plot(xf, np.abs(yf))
    plt.xlim(0, 18e7)
    plt.ylim(0, 10)
    fig2.savefig("D:\\Data\\03 Mar 22\\RK2 Shockwaves Tests Outputs\\LLGTest1626FFT.png")
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
    timeStamp = input("Enter the unique identifier that all filenames will share: ")
    # timeStamp = 1358
    filepath_input_data = "D:\\Data\\03 Mar 22\\RK2 Shockwaves Tests Data\\"
    filepath_save_output = "D:\\Data\03 Mar 22\\RK2 Shockwaves Tests Outputs\\"

    lg.info(f"{PROGRAM_NAME} Begin importing data")
    file_mx = open(f"{filepath_input_data}rk2_mx_{FILE_IDENT}{str(timeStamp)}.csv")
    # file_my = open(f"{filepath_input_data}rk2_my_{FILE_IDENT}{str(timeStamp)}.csv")
    # file_mz = open(f"{filepath_input_data}rk2_mz_{FILE_IDENT}{str(timeStamp)}.csv")

    mx = np.loadtxt(file_mx, delimiter=",")
    # my = np.loadtxt(file_my, delimiter=",")
    # mz = np.loadtxt(file_mz, delimiter=",")
    lg.info(f"{PROGRAM_NAME} Finish importing data")

    # Separate here into new function
    """
    Default values are:
        numberOfSpins = 4000
        total_datapoints = 100 (used to be called itermax)
        iterations = 7e5
    """

    stepsize = 1e-17
    iterations = 1.75e5 * 4e2
    maximum_real_time = int(stepsize * iterations)

    total_datapoints = int(1 / 1e-6)

    targetSpin = int(input("Plot which spin: "))
    # sel_iter = int(total_datapoints)

    mx_spin = mx[1:, targetSpin]

    lg.info(f"{PROGRAM_NAME} Begin FFT plotting")
    custom_fft(int(frequency), maximum_real_time, mx_spin)
    lg.info(f"{PROGRAM_NAME} Finish FFT plotting")
    # contains all the names of all subplots to be created
    # m_names = ['Mx', 'My', 'Mz']

    # contains all the labels needed for the plots. 1st row: titles. 2nd row: x-axis labels. 3rd row: y-axis labels
    # plot_labels = [f'Chain Spin ({str(frequency)}) (RK2 - Midpoint) [C++]', 'Iterations', 'Spin Site', 'M Value',
    # 'Value']

    fig1 = plt.figure()
    plt.title("Signal (x)")
    plt.plot(np.arange(0, total_datapoints), mx_spin)
    fig1.savefig(f"{filepath_save_output}{FILE_IDENT}{str(timeStamp)}.png")
    plt.show()


# --------------------------- main() implementation ---------------------------

def main():
    lg.info(f"{PROGRAM_NAME} Start")

    # plot_cpp_data(42.5 * 1e9)
    fft_example()

    lg.info(f"{PROGRAM_NAME} End")

    exit()


# ------------------------------ Implementations ------------------------------

if __name__ == '__main__':
    logging_setup()

    main()
