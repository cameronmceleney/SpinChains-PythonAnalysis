#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard libraries
import logging as lg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Additional libraries
import csv as csv
from os import path

# My packages / Any header files
import system_preparation as sp
import plots_for_rk_methods as plt_rk

"""
    Description of what data_analysis does
"""
PROGRAM_NAME = "data_analysis.py"
"""
    Core Details
    
    Author      : cameronmceleney
    Created on  : 12/03/2022 19:02
    Filename    : data_analysis
    IDE         : PyCharm
"""


def data_analysis(file_descriptor, file_prefix="rk2_mx_", file_identifier="LLGTest"):
    """
    Import a dataset in csv format, plotting the signal and the corresponding FFTs, for a user-defined number of sites.

    -----
    Notes
    -----

    Ensure that the first column of the dataset are the timestamps that each measurement was taken at. If this is not
    the case, then replace the variable 'mx_time' with an array of values:

    * mx_time = np.linspace(start_time, end_time, number_of_iterations, endpoint=True)

    :param str file_prefix: This is the 'file_identity' variable in the C++ code.
    :param str file_identifier: This is the 'filename' variable in the C++ code.
    :param str file_descriptor: The file_ext variable in the C++ code. Set as a function argument to reduce user inputs

    :return: Nothing.
    """
    rc_params_update()

    full_file_name = f"{file_prefix}{file_identifier}{file_descriptor}"
    # data_absolute_path = f"{sp.directory_tree_testing()[0]}{full_file_name}.csv"

    # Tracking how long the data import took is important for monitoring large files.
    lg.info(f"{PROGRAM_NAME} - Invoking functions to import data..")
    # m_all_data, [header_data_params, header_data_sites] = import_data(full_file_name, data_absolute_path)
    mx_data, my_data, eigen_vals_data = import_data(full_file_name, sp.directory_tree_testing()[0],
                                                    only_essentials=False)
    lg.info(f"{PROGRAM_NAME} - All functions that import data are finished!")

    lg.info(f"{PROGRAM_NAME} - Invoking functions to plot data...")
    plt_rk.main2(mx_data, my_data, eigen_vals_data)
    # plt_rk.three_panes(m_all_data, header_data_params, header_data_sites, [0, 1])
    exit()

    # First column of data file is always the real-time at that iteration. Convert to [s] from [ns]
    # mx_time = m_all_data[:, 0] / 1e-9

    shouldContinuePlotting = True
    while shouldContinuePlotting:
        # User will plot data one spin site at a time, as each plot can take an extended amount of time to create

        # target_spin = int(input("Plot which spin (-ve to exit): "))
        target_spin = 1

        if target_spin >= 1:

            # plt_rk.fft_and_signal_four(mx_time, m_all_data[:, target_spin], target_spin)
            shouldContinuePlotting = False
        else:
            shouldContinuePlotting = False


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


def import_data(file_name, input_filepath, only_essentials=True):
    """
    Imports, separates, and returns the simulation data from the simulation headers.

    The header information in each csv file contains every parameter required to re-run the simulation. Importing, and
    using this same data, significantly reduces the number of syntax errors between programs by automating processes.

    :param str file_name: Name of file to be imported. Note! This should not include a prefix like 'eigenvalues'
    :param str input_filepath: The absolute filepath to the dir containing input files.
    :param bool only_essentials: Streamlined option that is used for plotting panes.
    :return: Need to add description.
    """

    if only_essentials:
        lg.info(f"{PROGRAM_NAME} - Importing data points...")
        all_data_without_header = np.loadtxt(open(input_filepath, "rb"), delimiter=",", skiprows=9)
        lg.info(f"{PROGRAM_NAME} - Data points imported!")
        header_data = import_data_headers(input_filepath)
        return all_data_without_header, header_data

    else:
        output_data_names = ["mx_data", "my_data", "eigenvalues_data"]
        output_data = [None, None, None]  # [0]: mx_data. [1]: my_data. [2]: eigenvalues_data.

        # [0]: does_mx_data_exist. [1]: does_my_data_exist. [2]: does_eigenvalues_data_exist
        does_data_exist = [False, False, False]

        # [0]: eigenvectors_mx_filtered_filename. [1]: eigenvectors_my_filtered_filename. [2]: eigenvalues_filename.
        filtered_filenames = [f"mx_formatted_{file_name}.csv", f"my_formatted_{file_name}.csv",
                              f"eigenvalues_formatted_{file_name}.csv"]

        print(f"\nChecking chosen directories for files...")

        for i, name in enumerate(filtered_filenames):

            if path.exists(input_filepath+name):
                output_data[i] = np.loadtxt(input_filepath+name, delimiter=',')
                does_data_exist[i] = True
                print(f"{output_data_names[i]}: found")
            else:
                print(f"{output_data_names[i]}: not found")

        for _, does_exist in enumerate(does_data_exist):

            if not does_exist:
                genFiles = input('Missing files detected. Run import code to generate files? Y/N: ').upper()

                while True:
                    if genFiles == 'Y':

                        eigenvalues_raw = np.loadtxt(f"{input_filepath}eigenvalues_{file_name}.csv", delimiter=",")
                        eigenvectors_raw = np.loadtxt(f"{input_filepath}eigenvectors_{file_name}.csv", delimiter=",")

                        eigenvalues_filtered = np.flipud(eigenvalues_raw[::2])
                        eigenvectors_filtered = np.fliplr(eigenvectors_raw[::2, :])

                        mx_data = eigenvectors_filtered[:, 0::2]
                        my_data = eigenvectors_filtered[:, 1::2]

                        np.savetxt(f"{input_filepath}{filtered_filenames[0]}", mx_data, delimiter=',')
                        np.savetxt(f"{input_filepath}{filtered_filenames[1]}", my_data, delimiter=',')
                        np.savetxt(f"{input_filepath}{filtered_filenames[2]}", eigenvalues_filtered, delimiter=',')

                        print(f"\nFiles successfully generated and save in {input_filepath}!\n")
                        break

                    elif genFiles == 'N':
                        print("\nWill not generate files. Exiting...\n")
                        exit(0)

                    else:
                        while genFiles not in 'YN':
                            genFiles = input("Invalid selection, try again. Run import code to generate missing "
                                             "files? Y/N: ").upper()

        else:
            print("All files successfully found!\n")

        return output_data[0], output_data[1], output_data[2]


def import_data_headers(filename):
    """
    Import the header lines of each csv file to obtain the C++ simulation parameters.

    Each simulation in C++ returns all the key parameters, required to replicate the simulation, as headers in csv
    files. This function imports that data, and creates dictionaries to store it.

    The Python dictionary keys are the same variable names as their C++ counterparts (for consistency). Casting is
    required as data comes from csvreader as strings.

    :param str filename: The filename of the data to be imported. Obtained from data_analysis.data_analysis()

    :return: Returns a tuple. [0] is the dictionary containing all the key simulation parameters. [1] is an array
    containing strings; the names of each spin site.
    """
    lg.info(f"{PROGRAM_NAME} - Importing file headers...")

    with open(filename) as file_header_data:
        csv_reader = csv.reader(file_header_data)
        next(csv_reader)  # 1st line. title_line
        next(csv_reader)  # 2nd line. Blank.
        next(csv_reader)  # 3rd line. Column title for each key simulation parameter. data_names
        data_values = next(csv_reader)  # 4th line. Values associated with column titles from 3rd line.
        next(csv_reader)  # 5th line. Blank.
        next(csv_reader)  # 6th line. Simulation notes. sim_notes
        next(csv_reader)  # 7th line. Describes how to understand column titles from 3rd line. data_names_explained
        next(csv_reader)  # 8th line. Blank.
        simulated_spin_sites = next(csv_reader)  # 9th line. Number for each spin site that was simulated

    # Assignment to dict is done individually to improve readability.
    key_params = dict()
    key_params['biasField'] = float(data_values[0])
    key_params['biasFieldDriving'] = float(data_values[1])
    key_params['biasFieldDrivingScale'] = float(data_values[2])
    key_params['drivingFreq'] = float(data_values[3])
    key_params['drivingRegionLHS'] = int(data_values[4])
    key_params['drivingRegionRHS'] = int(data_values[5])
    key_params['drivingRegionWidth'] = int(data_values[6])
    key_params['maxSimTime'] = float(data_values[7])
    key_params['exchangeMaxVal'] = float(data_values[8])
    key_params['stopIterVal'] = float(data_values[9])
    key_params['exchangeMinVal'] = float(data_values[10])
    key_params['numberOfDataPoints'] = int(data_values[11])
    key_params['numSpins'] = int(data_values[12])
    key_params['stepsize'] = float(data_values[13])

    lg.info(f"{PROGRAM_NAME} - File headers imported!")

    return key_params, simulated_spin_sites
