#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard libraries
import logging as lg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys as sys

# Additional libraries
import csv as csv
from pathlib import Path

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
    data_absolute_path = f"{sp.directory_tree_testing()[0]}{full_file_name}.csv"

    # Tracking how long the data import took is important for monitoring large files.
    lg.info(f"{PROGRAM_NAME} - Invoking functions to import data..")
    # m_all_data, [header_data_params, header_data_sites] = import_data(data_absolute_path)
    lg.info(f"{PROGRAM_NAME} - All functions that import data are finished!")

    mx_data, my_data, eigen_vals_data = import_data(full_file_name, sp.directory_tree_testing()[0], only_essentials=False)

    plt_rk.main2(mx_data, my_data, eigen_vals_data)
    exit()
    lg.info(f"{PROGRAM_NAME} - Invoking functions to plot data...")
    plt_rk.three_panes(m_all_data, header_data_params, header_data_sites, [0, 1])
    exit()

    # First column of data file is always the real-time at that iteration. Convert to [s] from [ns]
    mx_time = m_all_data[:, 0] / 1e-9

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


def import_data(file_name, file_path, only_essentials=True):
    """
    Imports, separates, and returns the simulation data from the simulation headers.

    The header information in each csv file contains every parameter required to re-run the simulation. Importing, and
    using this same data, significantly reduces the number of syntax errors between programs by automating processes.

    :param str file_path: The absolute filepath to the csv file in question. This path should only contain / or \\. An
                          example is C:\\user_name\\path_to_file\\file_name.csv'.
    :return: Two arguments. [0] is a 2D array of all simulation data values. [1] is a tuple (see import_data_headers for
             details).
    """
    if only_essentials:
        lg.info(f"{PROGRAM_NAME} - Importing data points...")
        all_data_without_header = np.loadtxt(open(file_path, "rb"), delimiter=",", skiprows=9)
        lg.info(f"{PROGRAM_NAME} - Data points imported!")

        header_data = import_data_headers(file_path)

        return all_data_without_header, header_data

    else:
        mxvals, myvals, eigenvals = None, None, None

        no_mxvals = False
        no_myvals = False
        no_eigenvals = False

        cpp_eigvals_name = "eigenvalues_" + file_name + ".csv"
        cpp_eigvects_name = "eigenvectors_" + file_name + ".csv"

        cpp_eigvals_path = Path(file_path + cpp_eigvals_name)
        cpp_eigvects_path = Path(file_path + cpp_eigvects_name)

        py_eigvals_name = "np_eigenvalues_" + file_name + ".csv"
        py_mxeigenvects_name = "np_mxeigenvects_" + file_name + ".csv"
        py_myeigenvects_name = "np_myeigenvects_" + file_name + ".csv"

        py_eigvals_path = Path(file_path + py_eigvals_name)
        py_mxeigenvects_path = Path(file_path + py_mxeigenvects_name)
        py_myeigenvects_path = Path(file_path + py_myeigenvects_name)

        print(f"Checking chosen directory [{file_path}] for files...")
        if py_mxeigenvects_path.is_file():
            mxvals = np.loadtxt(open(py_mxeigenvects_path), delimiter=',')
            print(f"mxvals: found ")

        else:
            print("No valid file containing the mxvals was found.")
            no_mxvals = True

        if py_myeigenvects_path.is_file():
            myvals = np.loadtxt(open(py_myeigenvects_path), delimiter=',')
            print("myvals: found")

        else:
            print("No valid file containing the myvals was found.")
            no_myvals = True

        if py_eigvals_path.is_file():
            eigenvals = np.loadtxt(open(py_eigvals_path), delimiter=',')
            print("eigenvals: found")

        else:
            print("No valid file containing the eigenvals was found.")
            no_eigenvals = True

        if no_mxvals | no_myvals | no_eigenvals:

            genFiles = input('Missing files detected. Run full import code to generate files? Y/N: ').upper()

            while True:

                if genFiles == 'Y':

                    # os.chdir(wdirPath)
                    importeigvals = np.loadtxt(open(cpp_eigvals_path), delimiter=",")
                    importedeigvects = np.loadtxt(open(cpp_eigvects_path), delimiter=",")

                    sliceddata_eigvals = np.flipud(importeigvals[::2])
                    sliceddata_eigvects = np.fliplr(importedeigvects[::2, :])

                    mxeigenvects = sliceddata_eigvects[:, 0::2]
                    myeigenvects = sliceddata_eigvects[:, 1::2]

                    np.savetxt(py_eigvals_name, sliceddata_eigvals, delimiter=',')
                    np.savetxt(py_mxeigenvects_name, mxeigenvects, delimiter=',')
                    np.savetxt(py_myeigenvects_name, myeigenvects, delimiter=',')

                    eigenvals = sliceddata_eigvals
                    mxvals = mxeigenvects
                    myvals = myeigenvects

                    print(f"\nFiles successfully generated and save in {file_path}!\n")
                    break

                elif genFiles == 'N':
                    print("Not generating files and exiting.")
                    sys.exit(0)

                else:
                    while genFiles not in 'YN':
                        print('Invalid selection, try again.')
                        genFiles = input(
                            'Missing files detected. Run full import code to generate files? Y/N: ').upper()

        elif no_mxvals == False & no_myvals == False & no_eigenvals == False:
            print("All files successfully found!\n")
        else:
            print("Unknown error with importing files.")
        return mxvals, myvals, eigenvals


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


def import_data_ranplotter(eigens_ext, directoryPath, wdirPath):
    if directoryPath[-1] != "/":
        directoryPath += "/"
    if wdirPath[-1] != "/":
        wdirPath += "/"

    mxvals, myvals, eigenvals = None, None, None

    no_mxvals = False
    no_myvals = False
    no_eigenvals = False

    cpp_eigvals_name = "eigenvalues_" + eigens_ext + ".csv"
    cpp_eigvects_name = "eigenvectors_" + eigens_ext + ".csv"

    cpp_eigvals_path = Path(directoryPath + cpp_eigvals_name)
    cpp_eigvects_path = Path(directoryPath + cpp_eigvects_name)

    py_eigvals_name = "np_eigenvalues_" + eigens_ext + ".csv"
    py_mxeigenvects_name = "np_mxeigenvects_" + eigens_ext + ".csv"
    py_myeigenvects_name = "np_myeigenvects_" + eigens_ext + ".csv"

    py_eigvals_path = Path(directoryPath + py_eigvals_name)
    py_mxeigenvects_path = Path(directoryPath + py_mxeigenvects_name)
    py_myeigenvects_path = Path(directoryPath + py_myeigenvects_name)

    print(f"Checking chosen directory [{directoryPath}] for files...")
    if py_mxeigenvects_path.is_file():
        mxvals = np.loadtxt(open(py_mxeigenvects_path), delimiter=',')
        print(f"mxvals: found ")

    else:
        print("No valid file containing the mxvals was found.")
        no_mxvals = True

    if py_myeigenvects_path.is_file():
        myvals = np.loadtxt(open(py_myeigenvects_path), delimiter=',')
        print("myvals: found")

    else:
        print("No valid file containing the myvals was found.")
        no_myvals = True

    if py_eigvals_path.is_file():
        eigenvals = np.loadtxt(open(py_eigvals_path), delimiter=',')
        print("eigenvals: found")

    else:
        print("No valid file containing the eigenvals was found.")
        no_eigenvals = True

    if no_mxvals | no_myvals | no_eigenvals:

        genFiles = input('Missing files detected. Run full import code to generate files? Y/N: ').upper()

        while True:

            if genFiles == 'Y':

                # os.chdir(wdirPath)
                importeigvals = np.loadtxt(open(cpp_eigvals_path), delimiter=",")
                importedeigvects = np.loadtxt(open(cpp_eigvects_path), delimiter=",")

                sliceddata_eigvals = np.flipud(importeigvals[::2])
                sliceddata_eigvects = np.fliplr(importedeigvects[::2, :])

                mxeigenvects = sliceddata_eigvects[:, 0::2]
                myeigenvects = sliceddata_eigvects[:, 1::2]

                np.savetxt(py_eigvals_name, sliceddata_eigvals, delimiter=',')
                np.savetxt(py_mxeigenvects_name, mxeigenvects, delimiter=',')
                np.savetxt(py_myeigenvects_name, myeigenvects, delimiter=',')

                eigenvals = sliceddata_eigvals
                mxvals = mxeigenvects
                myvals = myeigenvects

                print(f"\nFiles successfully generated and save in {wdirPath}!\n")
                break

            elif genFiles == 'N':
                print("Not generating files and exiting.")
                sys.exit(0)

            else:
                while genFiles not in 'YN':
                    print('Invalid selection, try again.')
                    genFiles = input('Missing files detected. Run full import code to generate files? Y/N: ').upper()

    elif no_mxvals == False & no_myvals == False & no_eigenvals == False:
        print("All files successfully found!\n")
    else:
        print("Unknown error with importing files.")
    return mxvals, myvals, eigenvals