#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard libraries
import logging as lg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Additional libraries
import csv as csv
from os import path, makedirs, listdir, rename
import errno as errno
import shutil as sh
import glob as gl

# My packages / Any header files
import plots_for_rk_methods as plt_rk
import plot_rk_methods_legacy as plt_rk_legacy

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


class PlotEigenmodes:
    def __init__(self, file_descriptor, input_dir_path, output_dir_path, file_prefix="eigenvalues", file_component="",
                 file_identifier="T"):

        # Attribute assignment / Initialisation
        self.parent_dir_path = input_dir_path
        self.output_dir_path = output_dir_path

        self.fd = file_descriptor
        self.fp = file_prefix
        self.fc = file_component
        self.fi = file_identifier

        # Define file names and paths use to find the data
        self.child_dir_name = f"{self.fi}{self.fd}_Eigens"
        self.input_dir_path = f"{self.parent_dir_path}{self.child_dir_name}/"
        self.full_filename = f"{self.fp}{self.fc}_{self.fi}{self.fd}"
        self.full_output_path = f"{self.output_dir_path}{file_identifier}{file_descriptor}"

        # Define all file names use for generating output names
        self.input_filenames = [f"eigenvalues_{self.fi}{self.fd}.csv",
                                f"eigenvectors_{self.fi}{self.fd}.csv"]

        self.input_filename_descriptions = ["eigenvalues_data", "eigenvectors_data"]
        self.formatted_filename_descriptions = ["mx_data", "my_data", "eigenvalues_data"]

        self.formatted_filenames = [f"mx_formatted_{self.fi}{self.fd}.csv",
                                    f"my_formatted_{self.fi}{self.fd}.csv",
                                    f"eigenvalues_formatted_{self.fi}{self.fd}.csv"]

        # Define array to later perform validity checks on the data
        self._arrays_to_output = [np.array([]), np.array([]), np.array([])]  # np.array([]) instead of `None`; explicit
        self.is_input_data_present = [False, False]
        self.is_formatted_data_present = [False, False, False]
        self.is_input_dir_present = False

        # Invoke my customised parameters (for plots)
        rc_params_update()

    def import_eigenmodes(self):
        # Containers to store key information about the returned arrays. Iterating through containers was felt to be
        # easier to read than having many lines of variable declarations and initialisations.

        self._check_directory_tree()

        if all(self.is_formatted_data_present):
            # All directories have been found, so can load into memory
            print(f"\nAll required files found. Loading data into memory...")

            for i, filename in enumerate(self.formatted_filenames):
                file_to_search_for = self.input_dir_path + filename
                self._arrays_to_output[i] = np.loadtxt(file_to_search_for, delimiter=',')

            return

        else:
            print(f"\nSome required files were not found. Need to generate formatted files from datasets...")

            for i, does_exist in enumerate(self.is_formatted_data_present):
                # Tests existence of each filtered array until either False is returned, or all are present (all True).

                try:
                    if does_exist is True:
                        pass
                    elif does_exist is False:
                        self._generate_file_that_is_missing(i)

                except ValueError:
                    lg.info(f"Boolean variable (does_exist) was dtype None.")
                    exit(1)
                finally:
                    pass
                    # self.import_eigenmodes()

            return

    def _check_directory_tree(self, should_show_errors=False):

        if path.exists(self.input_dir_path):
            print(f"Simulation_Data/{self.child_dir_name}: directory found")
            self.is_input_dir_present = True
            self._check_if_files_present(True)

        elif path.exists(f"{self.parent_dir_path}{self.fi}{self.fd}/"):
            # Included to handle legacy files
            rename(f"{self.parent_dir_path}{self.fi}{self.fd}", self.input_dir_path)
            print(f"Simulation_Data/{self.fi}{self.fd}: directory found. Name updated to {self.child_dir_name}")
            self.is_input_dir_present = True
            self._check_if_files_present(True)

        else:
            print(f"Simulation_Data/{self.child_dir_name}: not found")
            try:
                # Try to create each subdirectory (and parent if needed). Always show instances of dirs being created
                makedirs(self.input_dir_path, exist_ok=False)
                print(f"Directory '{self.child_dir_name}' created successfully")
                self.is_input_dir_present = True
                self.import_eigenmodes()

            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

                # Handles exceptions from raise
                if should_show_errors:
                    # User-selected flag in function argument
                    print(f"Directory '{self.child_dir_name}' already exists.")
                    self.is_input_dir_present = True
                    pass

    def _check_if_files_present(self, should_check_all=False):
        print('\n--------------------------------------------------------------------------------')
        print(f"Checking chosen directories for raw input files...")

        if should_check_all is True:
            for i, (file_description, filename) in enumerate(
                    zip(self.input_filename_descriptions, self.input_filenames)):
                # Tests if the required files already exist in the target (input data) directory.
                file_to_search_for = self.input_dir_path + filename

                if path.exists(file_to_search_for):
                    self.is_input_data_present[i] = True
                    print(f"{file_description}: found")
                else:
                    self.is_input_data_present[i] = False
                    print(f"{file_description}: not found")

        print(f"\nChecking chosen directories for formatted input files...")

        for i, (file_description, filename) in enumerate(
                zip(self.formatted_filename_descriptions, self.formatted_filenames)):
            # Tests if the required files already exist in the target (input data) directory.
            file_to_search_for = self.input_dir_path + filename

            if path.exists(file_to_search_for):
                self.is_formatted_data_present[i] = True
                print(f"{file_description}: found")
            else:
                self.is_formatted_data_present[i] = False
                print(f"{file_description}: not found")
        print('--------------------------------------------------------------------------------')

        if all(item is False for item in self.is_formatted_data_present):

            if len(listdir(self.input_dir_path)) == 0:
                # Target directory is empty
                src_folder = self.parent_dir_path
                dst_folder = self.input_dir_path

                # Search files with .txt extension in source directory
                pattern = f"/*{self.fi}{self.fd}.csv"
                files = gl.glob(src_folder + pattern)

                # move the files with txt extension
                for file in files:
                    # extract file name form file path
                    file_name = path.basename(file)
                    sh.move(file, dst_folder + file_name)
                    print('Moved:', file)
            elif all(item is True for item in self.is_input_data_present):
                for i in range(0, 3):
                    self._generate_file_that_is_missing(i)
                    self.import_eigenmodes()

            else:
                print('Unknown error. Exiting.')
                exit(1)

    def _generate_file_that_is_missing(self, index):

        # Instance of missing file has been found, and will need to generate all filtered files that are needed.
        # Before doing so, allow user to opt-out.

        if index in [0, 1]:
            self._generate_missing_eigenvectors()
            return
        elif index in [2]:
            self._generate_missing_eigenvalues()
            return
        else:
            lg.error(f"Index of value {index} was called")
            return

    def _generate_missing_eigenvalues(self):

        lg.info(f"Missing Eigenvalues file found. Attempting to generate new file in correct format...")

        # 'Raw' refers to the data produces from the C++ code.
        print(f"Here: {self.input_dir_path}")
        eigenvalues_raw = np.loadtxt(f"{self.input_dir_path}eigenvalues_{self.fi}{self.fd}.csv",
                                     delimiter=",")

        eigenvalues_filtered = np.sort(eigenvalues_raw[eigenvalues_raw >= 0])

        # Use np.savetxt to save the data (2nd parameter) directly to the files (first parameter).
        np.savetxt(f"{self.input_dir_path}{self.formatted_filenames[2]}", eigenvalues_filtered,
                   delimiter=',')

        lg.info(f"Successfully generated missing (eigenvalues) file, which is saved in {self.input_dir_path}")

    def _generate_missing_eigenvectors(self, should_separate_afm_modes=False):

        lg.info(f"Missing (mx) and/or (my) file(s) found. Attempting to generate new files in correct format...")

        # 'Raw' refers to the data produces from the C++ code.
        eigenvectors_raw = np.loadtxt(f"{self.input_dir_path}eigenvectors_{self.fi}{self.fd}.csv",
                                      delimiter=",")

        # Filtered refers to the data imported into, and amended by, this Python code.
        eigenvectors_filtered = np.fliplr(eigenvectors_raw[::2, :])

        mx_data = eigenvectors_filtered[:, 0::2]
        my_data = eigenvectors_filtered[:, 1::2]

        # Use np.savetxt to save the data (2nd parameter) directly to the files (first parameter).
        np.savetxt(f"{self.input_dir_path}{self.formatted_filenames[0]}", mx_data, delimiter=',')
        np.savetxt(f"{self.input_dir_path}{self.formatted_filenames[1]}", my_data, delimiter=',')

        print(f"{self.formatted_filename_descriptions[0]}: generated\n"
              f"{self.formatted_filename_descriptions[1]}: generated")

        if should_separate_afm_modes:
            # Repeats process for AFM to separate all UP spins from DOWN
            afm_up_mx_data = mx_data[:, 0::2]
            afm_down_mx_data = mx_data[:, 1::2]
            np.savetxt(f"{self.input_dir_path}afm_up_{self.formatted_filenames[0]}", afm_up_mx_data, delimiter=',')
            np.savetxt(f"{self.input_dir_path}afm_down_{self.formatted_filenames[0]}", afm_down_mx_data, delimiter=',')

            print(f"AFM files: generated")

        lg.info(f"Successfully generated missing (mx) and (my) files, which are saved in {self.input_dir_path}")

    def plot_eigenmodes(self):
        lg.info(f"Invoking functions to plot data...")

        """
        Allows the user to plot as many eigenmodes as they would like; one per figure. This function is primarily used
        to replicate Fig. 1 from macedo2021breaking. The use of keywords within this function also allow the user to
        plot the 'generalised fourier coefficients' of a system; mainly used to replicate Figs 4.a & 4.d of the same
        paper.

        :return: A figure (png).

        """
        handle_eigenmodes = plt_rk.Eigenmodes(self._arrays_to_output[0], self._arrays_to_output[1],
                                              self._arrays_to_output[2], f"{self.fi}{self.fd}",
                                              self.input_dir_path, self.full_output_path)

        print('--------------------------------------------------------------------------------')
        print('''
        This will plot the eigenmodes of the selected data. Input the requested 
        modes as single values, or as a space-separated list. Unique [keywords] include:
            *   Exit [EXIT] (Quit the program)
            *   Fourier Coefficients [FRC] (Plot generalised Fourier co-efficients)
            *   Dispersion Relation [DIS] (Plot the dispersion relation)

        Note: None of the keywords are case-sensitive.
              ''')
        print('--------------------------------------------------------------------------------')

        upper_limit_mode = self._arrays_to_output[0].size  # The largest mode which can be plotted for the given data.

        # Test Code!

        previously_plotted_modes = []  # Tracks what mode(s) the user plotted in their last inputs.

        # Take in first user input. Assume input is valid, until an error is raised.
        modes_to_plot = input("Enter mode(s) to plot: ").split()
        has_valid_modes = True

        while True:
            # Plots eigenmodes as long as the user enters a valid input.
            for test_mode in modes_to_plot:

                try:
                    if not 1 <= int(test_mode) <= upper_limit_mode:
                        # Check mode is in the range held by the dataset
                        print(f"That mode does not exist. Please select a mode between 1 & {upper_limit_mode}.")
                        break

                    if set(previously_plotted_modes) & set(modes_to_plot):
                        # Check if mode has already been plotted. Cast to set as they don't allow duplicates.
                        has_valid_modes = False
                        print(
                            f"You have already printed a mode in {previously_plotted_modes}. Please make "
                            f"another choice.")
                        break

                except ValueError:
                    # If the current tested mode is within the range, then it is either a keyword, or invalid.
                    if test_mode.upper() == 'EXIT':
                        exit(0)

                    elif test_mode.upper() == 'FRC':
                        handle_eigenmodes.generalised_fourier_coefficients(use_defaults=False)
                        has_valid_modes = False
                        break

                    elif test_mode.upper() == 'DIS':
                        handle_eigenmodes.plot_dispersion_relation()
                        has_valid_modes = False
                        break

                    has_more_plots = input("Do you want to continue plotting modes? Y/N: ").upper()
                    while True:

                        if has_more_plots == 'Y':
                            has_valid_modes = False  # Prevents plotting of incorrect input, and allows user to retry.
                            break

                        elif has_more_plots == 'N':
                            print("Exiting program...")
                            exit(0)

                        else:
                            while has_more_plots not in 'YN':
                                has_more_plots = input("Do you want to continue plotting modes? Y/N: ").upper()

                if has_valid_modes:
                    handle_eigenmodes.plot_single_eigenmode(int(test_mode), has_endpoints=False)

                else:
                    has_valid_modes = True  # Reset condition
                    continue

            previously_plotted_modes = modes_to_plot  # Reassign the current modes to be the previous attempts.
            modes_to_plot = (input("Enter mode(s) to plot: ")).split()  # Take in the new set of inputs.


class PlotImportedData:

    def __init__(self, file_descriptor, input_dir_path, output_dir_path, file_prefix="rk2", file_component="mx",
                 file_identifier="LLGTest"):
        self.fd = file_descriptor
        self.in_path = input_dir_path
        self.out_path = output_dir_path
        self.fp = file_prefix
        self.fc = file_component
        self.fi = file_identifier

        self.override_method = None
        self.override_function = None
        self.override_site = None
        self.early_exit = False

        rc_params_update()

        self.full_filename = f"{file_prefix}_{file_component}_{file_identifier}{file_descriptor}"  # want 1744

        self.full_output_path = f"{self.out_path}{file_identifier}{file_descriptor}"
        self.input_data_path = f"{self.in_path}{self.full_filename}.csv"

        self.all_imported_data = self.import_data_from_file(self.full_filename, self.input_data_path)

        [self.header_data_params, self.header_data_sites, self.header_sim_flags] = self.import_headers_from_file()

        self.m_time_data = self.all_imported_data[:, 0] / 1e-9  # Convert to from [seconds] to [ns]
        self.m_spin_data = self.all_imported_data[:, 1:]

        self.accepted_keywords = ["3P", "FS", "FT", "EXIT", "PF", "CP"]

    @staticmethod
    def import_data_from_file(filename, input_data_path):
        """
        Outputs the data needed to plot single-image panes.

        Contained in single method to unify processing option. Separated from import_data_headers() (unlike in previous
        files) for when multiple datafiles, with the same header, are imported.
        """
        lg.info(f"Importing data points...")

        # Loads all input data without the header
        try:
            is_file_present_in_dir = path.exists(input_data_path)
            if not is_file_present_in_dir:
                raise FileNotFoundError
        except FileNotFoundError:
            print(f"File {filename} was not found")
            lg.error(f"File {filename} was not found")
            exit(1)
        else:
            lg.info(f"Data points imported!")
            return np.loadtxt(input_data_path, delimiter=",", skiprows=11)

    def import_headers_from_file(self):
        """
        Import the header lines of each csv file to obtain the C++ simulation parameters.

        Each simulation in C++ returns all the key parameters, required to replicate the simulation, as headers in csv
        files. This function imports that data, and creates dictionaries to store it.

        The Python dictionary keys are the same variable names as their C++ counterparts (for consistency). Casting is
        required as data comes from csvreader as strings.

        :return: Returns a tuple. [0] is the dictionary containing all the key simulation parameters. [1] is an array
        containing strings; the names of each spin site.
        """
        lg.info(f"Importing file headers...")

        if self.fi == "LLGTest":
            with open(self.input_data_path) as file_header_data:
                csv_reader = csv.reader(file_header_data)
                next(csv_reader)  # 0th line.
                next(csv_reader)  # 1st line. Blank.
                next(csv_reader)  # 2nd line. Column title for each key simulation parameter. data_names
                data_values = next(csv_reader)  # 3rd line. Values associated with column titles from 4th line.
                next(csv_reader)  # 4th line. Blank.
                next(csv_reader)  # 5th line. Simulation notes.
                next(csv_reader)  # 6th line. Describes how to understand tabular titles.
                next(csv_reader)  # 7th line. Blank.
                list_of_simulated_sites = next(csv_reader)  # 8th line. Array of simulated site numbers.

                data_flags = None
        else:
            with open(self.input_data_path) as file_header_data:
                csv_reader = csv.reader(file_header_data)
                next(csv_reader)  # 0th line.
                next(csv_reader)  # 1st line. Blank.
                data_flags = next(
                    csv_reader)  # 2nd line. Booleans to indicate which modules were used during simulations.
                next(csv_reader)  # 3rd line. Blank.
                next(csv_reader)  # 4th line. Column title for each key simulation parameter. data_names
                data_values = next(csv_reader)  # 5th line. Values associated with column titles from 4th line.
                next(csv_reader)  # 6th line. Blank.
                next(csv_reader)  # 7th line. Simulation notes.
                next(csv_reader)  # 8th line. Describes how to understand tabular titles.
                next(csv_reader)  # 9th line. Blank.
                list_of_simulated_sites = next(csv_reader)  # 10th line. Array of simulated site numbers.

        sim_flags = dict()
        if data_flags is not None:
            # Assignment to dict is done individually to improve readability.
            sim_flags['isLLGUsed'] = str(data_flags[1])
            sim_flags['isShockwaveUsed'] = str(data_flags[3])
            sim_flags['isDriveOnLHS'] = str(data_flags[5])
            sim_flags['methodName'] = str(data_flags[7])
            sim_flags['isDriveStatic'] = str(data_flags[9])

            if 'numericalMethodUsed' not in sim_flags.keys():
                sim_flags['numericalMethodUsed'] = f"{self.fp.upper()} Method"
            else:
                sim_flags['numericalMethodUsed'] = str(data_flags[7])
        elif 'numericalMethodUsed' not in sim_flags.keys():
            sim_flags['numericalMethodUsed'] = f"{self.fp.upper()} Method"
        else:
            sim_flags['numericalMethodUsed'] = f"{self.fp.upper()} Method"

        key_params = dict()

        if self.fi == "LLGTest":
            key_params['staticBiasField'] = float(data_values[0])
            key_params['dynamicBiasField1'] = float(data_values[1])
            key_params['dynamicBiasFieldScaling'] = float(data_values[2])
            key_params['drivingFreq'] = float(data_values[3])
            key_params['drivingRegionLHS'] = int(data_values[4])
            key_params['drivingRegionRHS'] = int(data_values[5])
            key_params['drivingRegionWidth'] = int(data_values[6])
            key_params['maxSimTime'] = float(data_values[7])
            key_params['exchangeMaxVal'] = float(data_values[8])
            key_params['stopIterVal'] = float(data_values[9])
            key_params['exchangeMinVal'] = float(data_values[10])
            key_params['numberOfDataPoints'] = int(data_values[11])
            key_params['totalSpins'] = int(data_values[12])
            key_params['stepsize'] = float(data_values[13])
            key_params['dampedSpins'] = int(data_values[14])
            key_params['gilbertFactor'] = 1e-5  # float(data_values[15])
            sim_flags['isLLGUsed'] = 1  # int(data_values[6])

            key_params['dynamicBiasField2'] = key_params['dynamicBiasField1'] * key_params['dynamicBiasFieldScaling']
            key_params['chainSpins'] = round(key_params['totalSpins'], -3)
            key_params['dampedSpins'] = key_params['totalSpins'] - key_params['chainSpins']
            key_params['gyroMagRatio'] = 2.92E9 * 2 * np.pi

        else:
            key_params['staticBiasField'] = float(data_values[0])
            key_params['dynamicBiasField1'] = float(data_values[1])
            key_params['dynamicBiasFieldScaling'] = float(data_values[2])
            key_params['dynamicBiasField2'] = float(data_values[3])
            key_params['drivingFreq'] = float(data_values[4])
            key_params['drivingRegionLHS'] = int(data_values[5])
            key_params['drivingRegionRHS'] = int(data_values[6])
            key_params['drivingRegionWidth'] = int(data_values[7])
            key_params['maxSimTime'] = float(data_values[8])
            key_params['exchangeMinVal'] = float(data_values[9])
            key_params['exchangeMaxVal'] = float(data_values[10])
            key_params['stopIterVal'] = float(data_values[11])
            key_params['numberOfDataPoints'] = int(data_values[12])
            key_params['chainSpins'] = int(data_values[13])
            key_params['dampedSpins'] = int(data_values[14])
            key_params['totalSpins'] = int(data_values[15])
            key_params['stepsize'] = float(data_values[16])
            key_params['gilbertFactor'] = float(data_values[17])
            key_params['gyroMagRatio'] = float(data_values[18])
            key_params['shockGradientTime'] = float(data_values[19])
            key_params['shockApplyTime'] = float(data_values[20])

        lg.info(f"File headers imported!")

        if "Time [s]" in list_of_simulated_sites:
            list_of_simulated_sites.remove("Time [s]")

        return key_params, list_of_simulated_sites, sim_flags

    def call_methods(self, override_method=None, override_function=None, override_site=None, early_exit=False):

        lg.info(f"Invoking functions to plot data...")
        print('\n--------------------------------------------------------------------------------')

        if early_exit:
            self.early_exit = early_exit
        if override_function is not None:
            self.override_function = override_function
        if override_site is not None:
            self.override_site = override_site

        if override_method is not None:
            # Allows the user to skip input dialogues, and call a specific function from a specific method.
            self.override_method = override_method
            initials_of_method_to_call = self.override_method.upper()

        else:
            print('''
            The plotting functions available are:
    
                *   Three Panes  [3P] (Plot all spin sites varying in time, and compare a selection)
                *   FFT & Signal [FS] (Examine signals from site(s), and the corresponding FFT)
                *   FFT only     [FT] (Interactive plot that outputs (x,y) of mouse click to console)
                *   Paper Figure [PF] (Plot final state of system at all sites)
                *   Contour Plot [CP] (Plot a single site as a 3D map)
    
            The terms within the square brackets are the keys for each function. 
            If you wish to exit the program then type EXIT. Keys are NOT case-sensitive.
                      ''')
            print('--------------------------------------------------------------------------------\n')

            initials_of_method_to_call = input("Which function to use: ").upper()

        if any([self.override_method, self.override_function, self.override_site, self.early_exit]):
            print(f"Override(s) enabled.\nMethod: {self.override_method} | Function: {self.override_function} | "
                  f" Site/Row: {self.override_site} | Early Exit: {self.early_exit}")
            print('--------------------------------------------------------------------------------')

        while True:
            if initials_of_method_to_call in self.accepted_keywords:
                self._data_plotting_selections(initials_of_method_to_call)
                break
            else:
                while initials_of_method_to_call not in self.accepted_keywords:
                    initials_of_method_to_call = input("Invalid option. Select function should to use: ").upper()

        print("Code complete!")
        lg.info(f"Code complete! Exiting.")

    def _data_plotting_selections(self, method_to_call):

        if method_to_call == "3P":
            self._invoke_three_panes()

        elif method_to_call == "FS":
            self._invoke_fs_functions()

        elif method_to_call == "FT":
            self._invoke_fft_functions()

        elif method_to_call == "PF":
            self._invoke_paper_figures()

        elif method_to_call == "CP":
            self._invoke_contour_plot()

        elif method_to_call == "EXIT":
            self._exit_conditions()

    def _invoke_three_panes(self):
        # Use this if you wish to see what ranplotter.py would output
        lg.info(f"Plotting function selected: three panes.")

        sites_to_compare = []
        should_compare_sites = "Y"
        try:
            while should_compare_sites not in "YN":
                should_compare_sites = input("Select sites to compare? Y/N: ").upper()
        except ValueError:
            raise ValueError
        else:
            if should_compare_sites == 'Y':
                sites_to_compare.append(input("Primary sites to compare: ").split())
                sites_to_compare.append(input("Secondary sites to compare: ").split())
                sites_to_compare.append(input("Tertiary sites to compare: ").split())

            elif should_compare_sites == 'N':
                exit(0)

        sites_to_compare = [[int(number_as_string) for number_as_string in str_array] for str_array in sites_to_compare]

        print("Generating plot...")
        plt_rk_legacy.three_panes(self.m_spin_data[:, :], self.header_data_params,
                                  self.full_output_path, sites_to_compare)
        lg.info(f"Plotting 3P complete!")

    def _invoke_fs_functions(self):
        # Use this to see fourier transforms of data

        lg.info(f"Plotting function selected: Fourier Signal.")

        has_more_to_plot = True
        while has_more_to_plot:
            # User will plot one spin site at a time, as plotting can take a long time.
            spins_to_plot = input("Plot which spin (-ve to exit): ").split()

            for target_spin in spins_to_plot:
                target_spin = int(target_spin)

                if target_spin >= 1:
                    print(f"Generating plot for [#{target_spin}]...")
                    lg.info(f"Generating FFT plot for Spin Site [#{target_spin}]")
                    target_spin_in_data = target_spin - 1  # To avoid off-by-one error. First spin date at [:, 0]
                    plt_rk_legacy.fft_and_signal_four(self.m_time_data[:], self.m_spin_data[:, target_spin_in_data],
                                                      target_spin,
                                                      self.header_data_params,
                                                      self.full_output_path)
                    lg.info(f"Finished plotting FFT of Spin Site [#{target_spin}]. Continuing...")
                    # cont_plotting_FFT = False  # Remove this after testing.
                else:
                    print("Exiting FFT plotting.")
                    lg.info(f"Exiting FS based upon user input of [{target_spin}]")
                    has_more_to_plot = False

        lg.info(f"Completed plotting FS!")

    def _invoke_fft_functions(self):
        # Use this to see fourier transforms of data

        lg.info(f"Plotting function selected: Fourier Signal only.")

        # dataset2_full_filename = "rk2_mx_T1614.csv"
        # dataset2_input_data_path = f"D:/Data/2022-11-30/Simulation_Data/{dataset2_full_filename}"
        # dataset2 = np.loadtxt(dataset2_input_data_path, delimiter=",", skiprows=11)
        # dataset2_m_spin_data = dataset2[:, 1:]

        has_more_to_plot = True
        while has_more_to_plot:
            # User will plot one spin site at a time, as plotting can take a long time.
            spins_to_plot = input("Plot which spin (-ve to exit): ").split()

            for target_spin in spins_to_plot:
                target_spin = int(target_spin)

                if target_spin >= 1:
                    print(f"Generating plot for [#{target_spin}]...")
                    lg.info(f"Generating FFT plot for Spin Site [#{target_spin}]")
                    target_spin_in_data = target_spin - 1  # To avoid off-by-one error. First spin date at [:, 0]
                    plt_rk_legacy.fft_only(self.m_spin_data[:, target_spin_in_data], target_spin,
                                           self.header_data_params,
                                           self.full_output_path)
                    # plt_rk.multi_fft_only(self.m_spin_data[:, target_spin_in_data],
                    #                      dataset2_m_spin_data[:, target_spin_in_data], target_spin,
                    #                      self.header_data_params,
                    #                      self.full_output_path)
                    lg.info(f"Finished plotting FFT of Spin Site [#{target_spin}]. Continuing...")
                    # cont_plotting_FFT = False  # Remove this after testing.
                else:
                    print("Exiting FFT plotting.")
                    lg.info(f"Exiting FS based upon user input of [{target_spin}]")
                    has_more_to_plot = False

        lg.info(f"Completed plotting FS!")

    def _invoke_contour_plot(self):
        # Use this if you wish to see what ranplotter.py would output
        lg.info(f"Plotting function selected: contour plot.")
        spin_site = int(input("Plot which site: "))

        mx_name = f"{self.fp}_mx_{self.fi}{self.fd}"
        my_name = f"{self.fp}_my_{self.fi}{self.fd}"
        mz_name = f"{self.fp}_mz_{self.fi}{self.fd}"
        mx_path = f"{self.in_path}{mx_name}.csv"
        my_path = f"{self.in_path}{my_name}.csv"
        mz_path = f"{self.in_path}{mz_name}.csv"

        mx_m_data = self.import_data_from_file(filename=mx_name,
                                               input_data_path=mx_path)
        my_m_data = self.import_data_from_file(filename=my_name,
                                               input_data_path=my_path)
        mz_m_data = self.import_data_from_file(filename=mz_name,
                                               input_data_path=mz_path)
        # plt_rk.create_contour_plot(mx_m_data, my_m_data, mz_m_data, spin_site, self.full_output_path, False)
        plt_rk_legacy.test_3d_plot(mx_m_data, my_m_data, mz_m_data, spin_site)
        lg.info(f"Plotting CP complete!")

    def _invoke_paper_figures(self):
        # Plots final state of system, similar to the Figs. in macedo2021breaking.
        lg.info(f"Plotting function selected: paper figure.")

        paper_fig = plt_rk.PaperFigures(self.m_time_data, self.m_spin_data,
                                        self.header_data_params, self.header_sim_flags, self.header_data_sites,
                                        self.full_output_path)

        if self.override_function is not None:
            pf_selection = self.override_function.upper()

        else:
            pf_selection = str(input("To return type 'BACK'. Which figure (PV [Position]/TV [Time]/HD [Heav. Dis.]"
                                     "/GIF/FFT/RIC) should be created: ")).upper()

        pf_keywords = ["PV", "TV", "HD", "GIF", "FFT", "BACK", "RIC"]  # TODO turn into dict with pf_selection

        while True:
            if pf_selection in pf_keywords:
                break
            else:
                while pf_selection not in pf_keywords:
                    pf_selection = str(input(f"Invalid choice. Select  from [{', '.join(pf_keywords)}]: ")).upper()

        cont_plotting = True

        if pf_selection == pf_keywords[0]:
            while cont_plotting:
                # User will plot one spin site at a time, as plotting can take a long time.
                if self.override_site is not None:
                    rows_to_plot = [self.override_site]
                else:
                    rows_to_plot = (input("Plot which rows of data (-ve to exit): ")).split()

                for row_num in rows_to_plot:

                    try:
                        row_num = int(row_num)

                    except ValueError:
                        if row_num.upper() == pf_keywords[4]:
                            self._invoke_paper_figures()
                        else:
                            print("ValueError. Please enter a valid string.")

                    else:
                        if row_num >= 0:
                            print(f"Generating plot for [#{row_num}]...")

                            lg.info(f"Generating PV plot for row [#{row_num}]")
                            paper_fig.create_position_variation(row_num)
                            lg.info(f"Finished plotting PV of row [#{row_num}]. Continuing...")

                            if self.early_exit:
                                cont_plotting = False

                        else:
                            print("Exiting PF-PV plotting.")
                            lg.info(f"Exiting PF-PV based upon user input of [{row_num}]")
                            cont_plotting = False

        elif pf_selection == pf_keywords[1]:
            while cont_plotting:

                if self.override_site is not None:
                    sites_to_plot = [self.override_site]
                else:
                    sites_to_plot = (input("Plot which site (-ve to exit): ")).split()

                # Loops through user's inputs, ensuring that they are correct.
                for target_site in sites_to_plot:

                    try:
                        target_site = int(target_site)

                    except ValueError:
                        if target_site.upper() == pf_keywords[4]:
                            self._invoke_paper_figures()
                        else:
                            print("ValueError. Please enter a valid string.")

                    else:
                        if target_site >= 0:
                            print(f"Generating plot for [#{target_site}]...")

                            lg.info(f"Generating TV plot for Spin Site [#{target_site}]")
                            paper_fig.plot_site_temporal(target_site, wavepacket_fft=True, visualise_wavepackets=False,
                                                         annotate_precursors_fft=False, annotate_signal=False,
                                                         wavepacket_inset=False, add_key_params=False,
                                                         add_signal_backgrounds=True, publication_details=False,
                                                         interactive_plot=False)
                            # paper_fig.create_time_variation2()
                            # paper_fig.create_time_variation3(interactive_plot=False, use_lower_plot=True,
                            #                                  use_inset_1=True)
                            lg.info(f"Finished plotting TV of Spin Site [#{target_site}]. Continuing...")

                            if self.early_exit:
                                cont_plotting = False

                        else:
                            print("Exiting PF-TV plotting.")
                            lg.info(f"Exiting PF-TV based upon user input of [{target_site}]")
                            cont_plotting = False

        elif pf_selection == pf_keywords[2]:
            paper_fig.create_gif(number_of_frames=0.01)
            print("GIF successfully created!")
            self._invoke_paper_figures()  # Use of override flag here will lead to an infinite loop!

        elif pf_selection == pf_keywords[3]:
            while cont_plotting:
                # User will plot one spin site at a time, as plotting can take a long time.
                if self.override_site is not None:
                    sites_to_plot = [self.override_site]
                else:
                    sites_to_plot = (input("Plot which site (-ve to exit): ")).split()

                for target_site in sites_to_plot:

                    try:
                        target_site = int(target_site)

                    except ValueError:
                        if target_site.upper() == pf_keywords[4]:
                            self._invoke_paper_figures()
                        else:
                            print("ValueError. Please enter a valid string.")

                    else:
                        if target_site >= 1:
                            print(f"Generating plot for [#{target_site}]...")

                            lg.info(f"Generating FFT plot for Spin Site [#{target_site}]")
                            paper_fig.plot_fft(target_site - 1, add_zoomed_region=False)
                            lg.info(f"Finished plotting FFT of Spin Site [#{target_site}]. Continuing...")

                            if self.early_exit:
                                cont_plotting = False

                        else:
                            print("Exiting PF-FFT plotting.")
                            lg.info(f"Exiting PF-FFT based upon user input of [{target_site}]")
                            cont_plotting = False

        elif pf_selection == pf_keywords[4]:
            self.call_methods()

        elif pf_selection == pf_keywords[5]:
            paper_fig.ricardo_paper()

        lg.info(f"Plotting PF complete!")

    @staticmethod
    def _exit_conditions():
        print("Exiting program...")
        lg.info(f"Exiting program from (select_plotter == EXIT)!")
        exit(0)


def rc_params_update():
    """Container for program's custom rc params, as well as Seaborn (library) selections."""
    plt.style.use('fivethirtyeight')
    sns.set(context='notebook', font='Kohinoor Devanagari', palette='muted', color_codes=True)
    ##############################################################################
    # Sets global conditions including font sizes, ticks and sheet style
    # Sets various font size. fsize: general text. lsize: legend. tsize: title. ticksize: numbers next to ticks
    medium_size = 14
    small_size = 12
    large_size = 20
    smaller_size = 10
    # tiny_size = 8

    # sets the tick direction. Options: 'in', 'out', 'inout'
    t_dir = 'in'
    # sets the tick size(s) and tick width(w) for the major and minor axes of all plots
    t_maj_s = 5
    t_min_s = t_maj_s / 2
    t_maj_w = 1
    t_min_w = t_maj_w / 2

    # updates rcParams of the selected style with my preferred options for these plots. Feel free to change
    plt.rcParams.update({'font.family': 'arial', 'font.size': small_size, 'font.weight': 'normal',

                         'figure.titlesize': large_size, 'axes.titlesize': medium_size,
                         'axes.labelsize': small_size, 'legend.fontsize': small_size,

                         'text.color': 'black', 'axes.edgecolor': 'black', 'axes.linewidth': t_maj_w,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',

                         'xtick.major.size': t_maj_s, 'xtick.major.width': t_maj_w,
                         'xtick.minor.size': t_min_s, 'xtick.minor.width': t_min_w,
                         'ytick.major.size': t_maj_s, 'ytick.major.width': t_maj_w,
                         'ytick.minor.size': t_min_s, 'ytick.minor.width': t_min_w,
                         'xtick.labelsize': smaller_size, 'ytick.labelsize': smaller_size,

                         'xtick.color': 'black', 'ytick.color': 'black', 'ytick.labelcolor': 'black',
                         'xtick.direction': t_dir, 'ytick.direction': t_dir,

                         "xtick.bottom": True, "ytick.left": True,
                         'axes.spines.top': True, 'axes.spines.bottom': True, 'axes.spines.left': True,
                         'axes.spines.right': True,
                         'axes.grid': False,

                         'savefig.dpi': 1200, "figure.dpi": 100})
