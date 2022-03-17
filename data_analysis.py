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


def data_analysis(file_descriptor, file_prefix="rk2_mx_", file_identifier="LLGTest", breaking_paper=False):
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
    :param bool breaking_paper: Temporary argument to allow for the user to plot eigenmodes from ranplotter.py (True),
    or signals (cpp_rk2_plot.py) (False).

    :return: Nothing.
    """
    rc_params_update()

    full_file_name = f"{file_prefix}{file_identifier}{file_descriptor}"
    data_absolute_path = f"{sp.directory_tree_testing()[0]}{full_file_name}.csv"

    # Tracking how long the data import took is important for monitoring large files.
    lg.info(f"{PROGRAM_NAME} - Invoking functions to import data..")

    if breaking_paper:
        # Used to plot figures from macedo2021breaking. Often used, so has a fast way to call.
        mx_data, my_data, eigen_vals_data = import_data(full_file_name, sp.directory_tree_testing()[0],
                                                        only_essentials=False)
        lg.info(f"{PROGRAM_NAME} - All functions that import data are finished!")
        lg.info(f"{PROGRAM_NAME} - Invoking functions to plot data...")
        plt_rk.eigenmodes(mx_data, my_data, eigen_vals_data, full_file_name)

    else:
        m_all_data, [header_data_params, header_data_sites] = import_data(full_file_name, data_absolute_path,
                                                                          only_essentials=True)
        lg.info(f"{PROGRAM_NAME} - All functions that import data are finished!")

        lg.info(f"{PROGRAM_NAME} - Invoking functions to plot data...")

        # Use this if you wish to see what ranplotter would normally output
        # plt_rk.three_panes(m_all_data, header_data_params, header_data_sites, [0, 1])
        mx_time = m_all_data[:, 0] / 1e-9

        plt_rk.fft_and_signal_four(mx_time, m_all_data[:, 1], 1, header_data_params)
        exit(0)
        shouldContinuePlotting = True
        while shouldContinuePlotting:
            # User will plot data one spin site at a time, as each plot can take an extended amount of time to create

            # target_spin = int(input("Plot which spin (-ve to exit): "))
            target_spin = 1

            if target_spin >= 1:

                plt_rk.fft_and_signal_four(mx_time, m_all_data[:, target_spin], target_spin)
                shouldContinuePlotting = False
            else:
                shouldContinuePlotting = False

    exit(0)


def rc_params_update():
    """Container for program's custom rc params, as well as Seaborn (library) selections."""
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


def import_data(file_name, input_filepath, only_essentials):
    """
    Imports, separates, formats, and returns the simulation data from the eigenvalue and eigenvector output csv files.

    The data needed can be obtained from the C++ code:

    * For only_essentials=True, use the outputs from 'SpinChainEigenSolver'.
    * For only_essentials=False, use the outputs from 'Numerical_Methods'.

    :param str file_name: Name of file to be imported. Note! This should not include a prefix like 'eigenvalues'
    :param str input_filepath: The absolute filepath to the dir containing input files.
    :param bool only_essentials: Streamlined option that is used for plotting panes.

    :return: Three arrays which can be used to generate all plots in plots_for_rk_methods.py.
    """

    if only_essentials:
        # Outputs the data needed to plot single-image panes
        lg.info(f"{PROGRAM_NAME} - Importing data points...")

        # Loads all input data
        all_data_without_header = np.loadtxt(open(input_filepath, "rb"), delimiter=",", skiprows=9)
        header_data = import_data_headers(input_filepath)

        lg.info(f"{PROGRAM_NAME} - Data points imported!")

        return all_data_without_header, header_data

    else:

        # Containers to store key information about the returned arrays. Iterating through containers was felt to be
        # easier to read than having many lines of variable declarations and initialisations.
        output_data_array_names = ["mx_data", "my_data", "eigenvalues_data"]  # Names of output data arrays found below.
        output_data_arrays = [None, None, None]  # Each array is initialised as none to ensure garbage isn't contained.
        does_data_exist = [False, False, False]  # Tests if each filtered array is in the target directory.
        filtered_filenames = [f"mx_formatted_{file_name}.csv", f"my_formatted_{file_name}.csv",
                              f"eigenvalues_formatted_{file_name}.csv"]  # filtered means being in the needed format

        print(f"\nChecking chosen directories for files...")

        for i, (array_name, file_name) in enumerate(zip(output_data_array_names, filtered_filenames)):

            if path.exists(input_filepath+file_name):
                # Check if each filtered data file (mx, my, eigenvalue) is in the target directory.
                output_data_arrays[i] = np.loadtxt(input_filepath+file_name, delimiter=',')
                does_data_exist[i] = True
                print(f"{array_name}: found")

            else:
                print(f"{array_name}: not found")

        for _, does_exist in enumerate(does_data_exist):
            # Tests existence of each filtered array until either False is returned, or all are present (all True).

            if not does_exist:
                # Generate all filtered files that are needed. Before doing so, allow user to opt-out.
                generate_files_response = input('Run import code to generate missing files? Y/N: ').upper()

                while True:
                    # Loops for as long as user input is accepted. Otherwise, forced them to comply.
                    if generate_files_response == 'Y':

                        # 'Raw' refers to the data produces from the C++ code.
                        eigenvalues_raw = np.loadtxt(f"{input_filepath}eigenvalues_{file_name}.csv", delimiter=",")
                        eigenvectors_raw = np.loadtxt(f"{input_filepath}eigenvectors_{file_name}.csv", delimiter=",")

                        # Filtered refers to the data imported into, and amended by, this Python code.
                        eigenvalues_filtered = np.flipud(eigenvalues_raw[::2])
                        eigenvectors_filtered = np.fliplr(eigenvectors_raw[::2, :])

                        mx_data = eigenvectors_filtered[:, 0::2]
                        my_data = eigenvectors_filtered[:, 1::2]

                        # Use np.savetxt to save the data (2nd parameter) directly to the files (first parameter).
                        np.savetxt(f"{input_filepath}{filtered_filenames[0]}", mx_data, delimiter=',')
                        np.savetxt(f"{input_filepath}{filtered_filenames[1]}", my_data, delimiter=',')
                        np.savetxt(f"{input_filepath}{filtered_filenames[2]}", eigenvalues_filtered, delimiter=',')

                        print(f"\nFiles successfully generated and save in {input_filepath}!\n")
                        break   # Exits while True: loop.

                    elif generate_files_response == 'N':
                        print("\nWill not generate files. Exiting...\n")
                        exit(0)

                    else:
                        while generate_files_response not in 'YN':
                            generate_files_response = input("Invalid selection, try again. Run import code to "
                                                            "generate missing files? Y/N: ").upper()

        else:
            #
            print("All files successfully found!\n")

        return output_data_arrays[0], output_data_arrays[1], output_data_arrays[2]


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
