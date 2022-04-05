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


class PlotEigenmodes:

    def __init__(self, file_descriptor, input_dir_path, output_dir_path, file_prefix="rk2", file_component="mx",
                 file_identifier="500spins"):
        self.fd = file_descriptor
        self.input_dir_path = input_dir_path
        self.output_dir_path = output_dir_path
        self.fp = file_prefix
        self.fc = file_component
        self.fi = file_identifier

        self.full_filename = f"{self.fp}_{self.fc}_{self.fi}{self.fd}"

        rc_params_update()

        # Arrays are inherently mutable, so there is no need to use @property decorator
        self.input_filenames_descriptions = ["mx_data", "my_data", "eigenvalues_data"]
        self._arrays_to_output = []  # Each array is initialised as none to remove garbage.
        self._does_data_exist_in_dir = []  # Tests if each filtered array is in the target directory.
        self.output_filenames = [f"mx_formatted_{self.full_filename}.csv",
                                 f"my_formatted_{self.full_filename}.csv",
                                 f"eigenvalues_formatted_{self.full_filename}.csv"]

    def _import_eigenmodes(self):
        # Containers to store key information about the returned arrays. Iterating through containers was felt to be
        # easier to read than having many lines of variable declarations and initialisations.

        print(f"\nChecking chosen directories for files...")

        for i, (output_file_description, output_file) in enumerate(
                zip(self.input_filenames_descriptions, self.output_filenames)):
            # Tests if the required files already exist in the target (input data) directory.
            file_to_search_for = self.input_dir_path + output_file

            if path.exists(file_to_search_for):
                self._arrays_to_output[i] = np.loadtxt(file_to_search_for, delimiter=',')
                self._does_data_exist_in_dir[i] = True
                print(f"{output_file_description}: found")

            else:
                self._does_data_exist_in_dir[i] = False
                print(f"{output_file_description}: not found")

        for i, does_exist in enumerate(self._does_data_exist_in_dir):
            # Tests existence of each filtered array until either False is returned, or all are present (all True).

            try:
                does_exist is True
            except ValueError:
                try:
                    does_exist is False
                except ValueError:
                    lg.info(f"Boolean variable (does_exist) was dtype None.")
                    exit(1)
                else:
                    self._generate_file_that_is_missing(i)
            else:
                print(f"{self.input_filenames_descriptions[i]} successfully found")

        else:
            print("All files successfully found!\n")

        return self._arrays_to_output[0], self._arrays_to_output[1], self._arrays_to_output[2]

    @staticmethod
    def _generate_file_that_is_missing(index):

        # Instance of missing file has been found, and will need to generate all filtered files that are needed.
        # Before doing so, allow user to opt-out.

        while True:
            generate_file_query = input('Run import code to generate missing files? y/n: ').upper()
            try:
                generate_file_query in "YN"
            except ValueError:
                continue
            else:
                if generate_file_query == 'Y':
                    if index in [0, 1]:
                        print('self._generate_missing_eigenvectors()')
                        return
                    elif index == 2:
                        print('self._generate_missing_eigenvalues()')
                        return
                    else:
                        lg.error(f"Index of value {index} was called")
                        return
                elif generate_file_query == 'N':
                    print("\nWill not generate files. Exiting...\n")
                    lg.info("User chose to not generate missing files. Code exited.")
                    exit(0)

    def _generate_missing_eigenvalues(self):

        lg.info(f"Missing Eigenvalues file found. Attempting to generate new file in correct format...")

        # 'Raw' refers to the data produces from the C++ code.
        eigenvalues_raw = np.loadtxt(f"{self.input_dir_path}eigenvalues_{self.full_filename}.csv",
                                     delimiter=",")

        # Filtered refers to the data imported into, and amended by, this Python code.
        eigenvalues_filtered = np.flipud(eigenvalues_raw[::2])

        # Use np.savetxt to save the data (2nd parameter) directly to the files (first parameter).
        np.savetxt(f"{self.input_dir_path}{self.output_filenames[2]}", eigenvalues_filtered,
                   delimiter=',')

        lg.info(f"Successfully generated missing (eigenvalues) file, which is saved in {self.input_dir_path}")

    def _generate_missing_eigenvectors(self):

        lg.info(f"Missing (mx) and/or (my) file(s) found. Attempting to generate new files in correct format...")

        # 'Raw' refers to the data produces from the C++ code.
        eigenvectors_raw = np.loadtxt(f"{self.input_dir_path}eigenvectors_{self.full_filename}.csv",
                                      delimiter=",")

        # Filtered refers to the data imported into, and amended by, this Python code.
        eigenvectors_filtered = np.fliplr(eigenvectors_raw[::2, :])

        mx_data = eigenvectors_filtered[:, 0::2]
        my_data = eigenvectors_filtered[:, 1::2]

        # Use np.savetxt to save the data (2nd parameter) directly to the files (first parameter).
        np.savetxt(f"{self.input_dir_path}{self.output_filenames[0]}", mx_data, delimiter=',')
        np.savetxt(f"{self.input_dir_path}{self.output_filenames[1]}", my_data, delimiter=',')

        lg.info(f"Successfully generated missing (mx) and (my) files, which are saved in {self.input_dir_path}")

        eigenmodes_data = self._import_eigenmodes()
        [self.mx_data, self.my_data, self.eigenvalues_data] = eigenmodes_data

    def plot_eigenmodes(self):
        lg.info(f"Invoking functions to plot data...")
        plt_rk.eigenmodes(self.mx_data, self.my_data, self.eigenvalues_data, self.full_filename)


class PlotImportedData:

    def __init__(self, file_descriptor, input_dir_path, output_dir_path, file_prefix="rk2", file_component="mx",
                 file_identifier="LLGTest"):
        self.fd = file_descriptor
        self.in_path = input_dir_path
        self.out_path = output_dir_path
        self.fp = file_prefix
        self.fc = file_component
        self.fi = file_identifier

        rc_params_update()

        self.full_filename = f"{file_prefix}_{file_component}_{file_identifier}{file_descriptor}"
        self.full_output_path = f"{self.out_path}{file_identifier}{file_descriptor}"
        self.input_data_path = f"{self.in_path}{self.full_filename}.csv"

        self.all_imported_data = self.import_data_from_file(self.full_filename, self.input_data_path)

        [self.header_data_params, self.header_data_sites] = self.import_headers_from_file()

        self.m_time_data = self.all_imported_data[:, 0] / 1e-9  # Convert to from [seconds] to [ns]
        self.m_spin_data = self.all_imported_data[:, 1:]

        self.accepted_keywords = ["3P", "FS", "EXIT", "PF", "CP"]

    @staticmethod
    def import_data_from_file(filename, input_data_path):
        """
        Outputs the data needed to plot single-image panes.

        Contained in single method to unify processing option. Separated from import_data_headers() (unlike in previous
        files) for when multiple datafiles, with the same header, are imported.
        """
        lg.info(f"Importing data points...")

        all_data_without_header = None
        # Loads all input data
        try:
            is_file_present_in_dir = path.exists(input_data_path)
            if not is_file_present_in_dir:
                raise FileNotFoundError
        except FileNotFoundError:
            print(f"File {filename} was not found")
            lg.error(f"File {filename} was not found")
            exit(1)
        else:
            all_data_without_header = np.loadtxt(input_data_path, delimiter=",", skiprows=9)

        lg.info(f"Data points imported!")

        return all_data_without_header

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

        with open(self.input_data_path) as file_header_data:
            csv_reader = csv.reader(file_header_data)
            next(csv_reader)  # 1st line. title_line
            next(csv_reader)  # 2nd line. Blank.
            next(csv_reader)  # 3rd line. Column title for each key simulation parameter. data_names
            data_values = next(csv_reader)  # 4th line. Values associated with column titles from 3rd line.
            next(csv_reader)  # 5th line. Blank.
            next(csv_reader)  # 6th line. Simulation notes. sim_notes
            next(csv_reader)  # 7th line. Describes how to understand column titles from 3rd line. data_names_explained
            next(csv_reader)  # 8th line. Blank.
            list_of_simulated_sites = next(csv_reader)  # 9th line. Number for each spin site that was simulated

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
        key_params['numGilbert'] = int(data_values[14])

        lg.info(f"File headers imported!")

        if "Time" in list_of_simulated_sites:
            list_of_simulated_sites.remove("Time")

        return key_params, list_of_simulated_sites

    def call_methods(self):

        lg.info(f"Invoking functions to plot data...")

        print('\n---------------------------------------------------------------------------------------')
        print('''
        The plotting functions available are:

            *   Three Panes  [3P] (Plot all spin sites varying in time, and compare a selection)
            *   FFT & Signal [FS] (Examine signals from site(s), and the corresponding FFT)
            *   Paper Figure [PF] (Plot final state of system at all sites)
            *   Contour Plot [CP] (Plot a single site as a 3D map)

        The terms within the square brackets are the keys for each function. 
        If you wish to exit the program then type EXIT. Keys are NOT case-sensitive.
                  ''')
        print('---------------------------------------------------------------------------------------\n')

        initials_of_method_to_call = input("Which function to use: ").upper()

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
        print("Note: To select sites to compare, edit code directly.")
        print("Generating plot...")
        plt_rk.three_panes(self.all_imported_data, self.header_data_params, self.header_data_sites,
                           self.full_output_path,
                           [3, 4, 5])
        lg.info(f"Plotting 3P complete!")

    def _invoke_fft_functions(self):
        # Use this to see fourier transforms of data

        lg.info(f"Plotting function selected: Fourier Signal.")

        has_more_to_plot = True
        while has_more_to_plot:
            # User will plot one spin site at a time, as plotting can take a long time.
            target_spin = int(input("Plot which spin (-ve to exit): "))
            print("Generating plot...")

            if target_spin >= 1:
                plt_rk.fft_and_signal_four(self.m_time_data, self.m_spin_data[:, target_spin], target_spin,
                                           self.header_data_params,
                                           self.full_output_path)
                lg.info(f"Finished plotting spin site #{target_spin} in FS. Continuing...")
                # cont_plotting_FFT = False  # Remove this after testing.
            else:
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
        plt_rk.create_contour_plot(mx_m_data, my_m_data, mz_m_data, spin_site, self.full_output_path, False)
        lg.info(f"Plotting CP complete!")

    def _invoke_paper_figures(self, has_override=False, override_name="PNG"):
        # Plots final state of system, similar to the Figs. in macedo2021breaking.
        lg.info(f"Plotting function selected: paper figure.")

        paper_fig = plt_rk.PaperFigures(self.m_time_data, self.m_spin_data,
                                        self.header_data_params, self.header_data_sites,
                                        self.full_output_path)

        if has_override:
            pf_selection = override_name
        else:
            pf_selection = str(input("Which figure (PNG/SV/GIF) should be created: ")).upper()

        if pf_selection == "PNG":
            paper_fig.create_png()
        elif pf_selection == "SV":
            site_num = int(input("Plot which site: "))
            paper_fig.plot_site_variation(site_num)
        elif pf_selection == "GIF":
            paper_fig.create_gif(number_of_frames=0.01)

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
