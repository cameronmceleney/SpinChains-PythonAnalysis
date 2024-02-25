# -*- coding: utf-8 -*-

# -------------------------- Preprocessing Directives -------------------------

# Full packages
import csv
import logging as log
import numpy as np
import os
import re
from typing import Any

# Specific functions from packages


# My full modules
import plot_rk_methods as plt_rk
import plot_rk_methods_legacy_standalone as plt_rk_legacy_standalone

# Specific functions from my modules
from attribute_mappings_legacy import AttributeMappings
from attribute_defintions import SimulationParametersContainer, SimulationFlagsContainer
from figure_manager import rc_params_update

# ----------------------------- Program Information ----------------------------

"""
Description of what foo.py does
"""
PROGRAM_NAME = "foo.py"
"""
Created on (date) by (author)
"""


def import_headers_from_file(self):
    log.info("Importing file headers...")

    sim_flags = AttributeMappings.dict_with_none(AttributeMappings.sim_flags)
    key_params = AttributeMappings.dict_with_none(AttributeMappings.key_data)

    with open(self.input_data_path) as file_header_data:
        csv_reader = csv.reader(file_header_data)
        next(csv_reader)  # 0th.
        next(csv_reader)  # 1st. Always blank.

        data_flags_titles = next(csv_reader)  # 2nd. Boolean titles which might also include values (older)
        data_flags_values = next(csv_reader)  # 3rd. Might include Boolean values (newer)

        # Count is now in terms of (newer); (older) doesn't have this additional blank line

        if data_flags_values:
            next(csv_reader)  # 4th line. Blank.
        key_sim_param_titles = next(csv_reader)  # 5th. Titles for each key simulation parameter.
        key_sim_param_values = next(csv_reader)  # 6th. Values for each key simulation parameter.
        next(csv_reader)  # 7th. Blank.
        next(csv_reader)  # 8th. Simulation notes.
        next(csv_reader)  # 9th. Description of how to read tabular data
        next(csv_reader)  # 10th. Blank.
        simulated_sites = next(csv_reader)  # 11th line. Titular value (site number) for each simulated site.

    # Iterate over each title in the CSV
    for title, value in zip(key_sim_param_titles, key_sim_param_values):
        for desired_var_name, (_, possible_spellings) in AttributeMappings.key_data.items():
            mapped_title = self.custom_string_mappings(title)
            if mapped_title in possible_spellings:
                key_params[desired_var_name] = self._custom_cast_mappings(value)
                break

    if data_flags_values:
        for title, value in zip(data_flags_titles, data_flags_values):
            for desired_var_name, (_, possible_spellings) in AttributeMappings.sim_flags.items():
                if title in possible_spellings:
                    if len(value) > 1:
                        mapped_value = self.custom_string_mappings(value, False)
                    else:
                        mapped_value = bool(value)
                    sim_flags[desired_var_name] = mapped_value
                    break
    else:
        for i, entry in enumerate(data_flags_titles):
            if i % 2 == 0:
                title, value = data_flags_titles[i], data_flags_titles[i + 1]
                for desired_var_name, (_, possible_spellings) in AttributeMappings.sim_flags.items():
                    if title in possible_spellings:
                        if len(value) > 1:
                            sim_flags[desired_var_name] = self.custom_string_mappings(value, False)
                        else:
                            sim_flags[desired_var_name] = bool(int(value))
                        break
            else:
                continue

    """
    for i, val in enumerate(data_flags_titles):
        if i % 2 == 0:
            mapped_title = self.custom_string_mappings(data_flags_titles[i], False)
            if len(val) > 1:
                mapped_value = self.custom_string_mappings(data_flags_titles[i+1], False)
            else:
                mapped_value = bool(data_flags_titles[i+1])
            sim_flags[mapped_title] = mapped_value
        else:
            continue

    for title, value in zip(key_sim_param_titles, key_sim_param_values):
        mapped_title = self.custom_string_mappings(title)
        mapped_value = self._custom_cast_mappings(value)
        key_params[mapped_title] = mapped_value
    """
    # Cleanup
    if sim_flags['numericalMethodUsed'] is None:
        sim_flags['numericalMethodUsed'] = f"{self.fp.upper()} Method"

    if "Time [s]" in simulated_sites:
        simulated_sites.remove("Time [s]")

    print(key_params)
    print(sim_flags)

    return key_params, simulated_sites, sim_flags


def import_headers_from_file_legacy(fi, input_data_path, fp):
    """
    Import the header lines of each csv file to obtain the C++ simulation parameters.

    THIS IS LEGACY CODE FOR ALL DATA PRODUCED BEFORE 2024-02-20. Each simulation in C++ returns all the key
    parameters, required to replicate the simulation, as headers in csv files. This function imports that data,
    and creates dictionaries to store it.

    The Python dictionary keys are the same variable names as their C++ counterparts (for consistency). Casting is
    required as data comes from csvreader as strings.

    :return: Returns a tuple. [0] is the dictionary containing all the key simulation parameters. [1] is an array
    containing strings; the names of each spin site.
    """
    log.info(f"Importing file headers...")

    if fi == "LLGTest":
        with open(input_data_path) as file_header_data:
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
        with open(input_data_path) as file_header_data:
            csv_reader = csv.reader(file_header_data)
            next(csv_reader)  # 0th line.
            next(csv_reader)  # 1st line. Blank.
            data_flags = next(
                csv_reader)  # 2nd line. Booleans to indicate which modules were used during simulations.
            next(csv_reader)  # 3rd line. Blank.
            data_value_names = next(
                csv_reader)  # 4th line. Column title for each key simulation parameter. data_names
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
            sim_flags['numericalMethodUsed'] = f"{fp.upper()} Method"
        else:
            sim_flags['numericalMethodUsed'] = str(data_flags[7])
    elif 'numericalMethodUsed' not in sim_flags.keys():
        sim_flags['numericalMethodUsed'] = f"{fp.upper()} Method"
    else:
        sim_flags['numericalMethodUsed'] = f"{fp.upper()} Method"

    key_params = dict()

    if fi == "LLGTest":
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

    log.info(f"File headers imported!")

    if "Time [s]" in list_of_simulated_sites:
        list_of_simulated_sites.remove("Time [s]")

    return key_params, list_of_simulated_sites, sim_flags


def _apply_custom_casting_to_value(data_value):
    data_value = data_value.strip()
    if '.' in data_value or 'e' in data_value.lower():
        return float(data_value)
    else:
        return int(data_value)


class PlotImportedData:

    def __init__(self, file_descriptor, input_dir_path, output_dir_path, file_prefix="rk2", file_component="mx",
                 file_identifier="T"):
        self._input_dir_path = input_dir_path
        self._output_dir_path = output_dir_path
        self._input_prefix = file_prefix
        self._input_comp = file_component
        self._input_id = file_identifier
        self._input_desc = file_descriptor

        self.override_method = None
        self.override_function = None
        self.override_site = None
        self.early_exit = False
        self.mass_produce = False

        self._accepted_methods = ["3P", "FS", "FT", "EXIT", "PF", "CP"]

        self._input_filename = f"{self._input_prefix}_{self._input_comp}_{self._input_id}{self._input_desc}"
        self._input_path_full = f"{self._input_dir_path}{self._input_filename}.csv"
        self._output_path_full = f"{self._output_dir_path}{file_identifier}{file_descriptor}"

        self._data_container = self._import_simulation_data()
        self._data_timestamps = self._data_container[:, 0] / 1e-9  # Convert to from [seconds] to [ns]
        self._data_magnetic_moments = self._data_container[:, 1:]

        [self._data_parameters, self._data_sites, self._data_flags] = self._import_simulation_headers()

        rc_params_update()

    def _import_simulation_data(self, filename=None, full_path_to_file=None):
        """
        Outputs the data needed to plot single-image panes.

        Contained in single method to unify processing option. Separated from import_data_headers() (unlike in previous
        files) for when multiple datafiles, with the same header, are imported.
        """
        log.info(f"Importing data points...")

        if filename is None:
            filename = self._input_filename
        if full_path_to_file is None:
            full_path_to_file = self._input_path_full

        # Loads all input data without the header
        try:
            is_file_present_in_dir = os.path.exists(full_path_to_file)
            if not is_file_present_in_dir:
                raise FileNotFoundError
        except FileNotFoundError:
            print(f"File {filename} was not found")
            log.error(f"File {filename} was not found")
            exit(1)
        else:
            log.info(f"Data points imported!")
            return np.loadtxt(full_path_to_file, delimiter=",", skiprows=11)

    def _import_simulation_headers(self):

        parameters = SimulationParametersContainer()
        flags = SimulationFlagsContainer()
        process_data = ProcessData()

        with open(self._input_path_full) as file_header_data:
            csv_reader = csv.reader(file_header_data)
            next(csv_reader)  # Skip the 0th line.
            next(csv_reader)  # Skip the 1st line (always blank).

            # Process simulation flags (2nd and 3rd lines)
            csv_reader, flags = process_data.process_selected_headers(csv_reader, flags, is_sim_flags=True)

            # Additional blank (4th line) in newer format; older format has blank 3rd and key sim params at 4th

            # Count from here is for new format. Process simulation parameters (5th and 6th lines)
            csv_reader, flags = process_data.process_selected_headers(csv_reader, parameters, is_sim_flags=False)

            # Skip lines until simulated sites (11th line)
            for _ in range(4):
                next(csv_reader)  # Skip lines 7th (blank), 8th (simulation notes), 9th (descriptions), 10th (blank)
            simulated_sites = next(csv_reader)  # 11th line

        # Process simulated sites
        if "Time [s]" in simulated_sites:
            simulated_sites.remove("Time [s]")

        return parameters.return_data(), simulated_sites, flags.return_data()

    def call_methods(self, override_method=None, override_function=None, override_site=None,
                     early_exit=False, mass_produce=False):

        log.info(f"Invoking functions to plot data...")
        print('\n--------------------------------------------------------------------------------')

        if early_exit:
            self.early_exit = early_exit
        if override_function is not None:
            self.override_function = override_function
        if override_site is not None:
            self.override_site = override_site

        if mass_produce:
            self.mass_produce = mass_produce

        if override_method is not None:
            # Allows the user to skip input dialogues, and call a specific function from a specific method.
            self.override_method = override_method
            initials_of_method_to_call = self.override_method.upper()

        else:
            # TODO This list of functions is massively out of date!
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
            if self.mass_produce:
                print(f"Producing: {self._input_id}{self._input_desc}")
                log.info(f"Producing: {self._input_id}{self._input_desc}")
            else:
                print(f"Override(s) enabled.\nMethod: {self.override_method} | Function: {self.override_function} | "
                      f" Site/Row: {self.override_site} | Early Exit: {self.early_exit}")
                print('--------------------------------------------------------------------------------')

        while True:
            if initials_of_method_to_call in self._accepted_methods:
                self._invoke_data_plotting_selections(initials_of_method_to_call)
                break
            else:
                while initials_of_method_to_call not in self._accepted_methods:
                    initials_of_method_to_call = input("Invalid option. Select function should to use: ").upper()

        if self.mass_produce:
            print(f"Produced: {self._input_id}{self._input_desc}")
            log.info(f"Produced: {self._input_id}{self._input_desc}")
        else:
            print("Code complete!")
            log.info(f"Code complete! Exiting.")

    def _invoke_data_plotting_selections(self, method_to_call):

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
            self._invoke_exit_conditions()

    def _invoke_three_panes(self):
        # Use this if you wish to see what my old Spyder code would output
        log.info(f"Plotting function selected: three panes.")

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
        plt_rk_legacy_standalone.three_panes(self._data_magnetic_moments[:, :], self._data_parameters,
                                             self._output_path_full, sites_to_compare)
        log.info(f"Plotting 3P complete!")

    def _invoke_fs_functions(self):
        # Use this to see fourier transforms of data

        log.info(f"Plotting function selected: Fourier Signal.")

        has_more_to_plot = True
        while has_more_to_plot:
            # User will plot one spin site at a time, as plotting can take a long time.
            spins_to_plot = input("Plot which spin (-ve to exit): ").split()

            for target_spin in spins_to_plot:
                target_spin = int(target_spin)

                if target_spin >= 1:
                    print(f"Generating plot for [#{target_spin}]...")
                    log.info(f"Generating FFT plot for Spin Site [#{target_spin}]")
                    target_spin_in_data = target_spin - 1  # To avoid off-by-one error. First spin date at [:, 0]
                    plt_rk_legacy_standalone.fft_and_signal_four(self._data_timestamps[:],
                                                                 self._data_magnetic_moments[:, target_spin_in_data],
                                                                 target_spin, self._data_parameters,
                                                                 self._output_path_full)
                    log.info(f"Finished plotting FFT of Spin Site [#{target_spin}]. Continuing...")
                    # cont_plotting_FFT = False  # Remove this after testing.
                else:
                    print("Exiting FFT plotting.")
                    log.info(f"Exiting FS based upon user input of [{target_spin}]")
                    has_more_to_plot = False

        log.info(f"Completed plotting FS!")

    def _invoke_fft_functions(self):
        # Use this to see fourier transforms of data

        log.info(f"Plotting function selected: Fourier Signal only.")

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
                    log.info(f"Generating FFT plot for Spin Site [#{target_spin}]")
                    target_spin_in_data = target_spin - 1  # To avoid off-by-one error. First spin date at [:, 0]
                    plt_rk_legacy_standalone.fft_only(self._data_magnetic_moments[:, target_spin_in_data], target_spin,
                                                      self._data_parameters,
                                                      self._output_path_full)
                    # plt_rk.multi_fft_only(self._data_magnetic_moments[:, target_spin_in_data],
                    #                      dataset2_m_spin_data[:, target_spin_in_data], target_spin,
                    #                      self._data_parameters,
                    #                      self._output_path_full)
                    log.info(f"Finished plotting FFT of Spin Site [#{target_spin}]. Continuing...")
                    # cont_plotting_FFT = False  # Remove this after testing.
                else:
                    print("Exiting FFT plotting.")
                    log.info(f"Exiting FS based upon user input of [{target_spin}]")
                    has_more_to_plot = False

        log.info(f"Completed plotting FS!")

    def _invoke_contour_plot(self):
        # Use this if you wish to see what my old Spyder code would output
        log.info(f"Plotting function selected: contour plot.")
        spin_site = int(input("Plot which site: "))

        mx_name = f"{self._input_prefix}_mx_{self._input_id}{self._input_desc}"
        my_name = f"{self._input_prefix}_my_{self._input_id}{self._input_desc}"
        mz_name = f"{self._input_prefix}_mz_{self._input_id}{self._input_desc}"
        mx_path = f"{self._input_dir_path}{mx_name}.csv"
        my_path = f"{self._input_dir_path}{my_name}.csv"
        mz_path = f"{self._input_dir_path}{mz_name}.csv"

        mx_m_data = self._import_simulation_data(mx_name, mx_path)
        my_m_data = self._import_simulation_data(my_name, my_path)
        mz_m_data = self._import_simulation_data(mz_name, mz_path)
        # plt_rk.create_contour_plot(mx_m_data, my_m_data, mz_m_data, spin_site, self._output_path_full, False)
        plt_rk_legacy_standalone.test_3d_plot(mx_m_data, my_m_data, mz_m_data, spin_site)
        log.info(f"Plotting CP complete!")

    def _invoke_paper_figures(self):
        # Plots final state of system, similar to the Figs. in macedo2021breaking.
        log.info(f"Plotting function selected: paper figure.")

        # if self._input_prefix == "rk2":
        #    paper_fig = plt_rk_legacy.PaperFigures(self._data_timestamps, self._data_magnetic_moments,
        #                                    self._data_parameters, self._data_flags, self._data_sites,
        #                                    self._output_path_full)
        # else:
        paper_fig = plt_rk.PaperFigures(self._data_timestamps, self._data_magnetic_moments,
                                        self._data_parameters, self._data_flags, self._data_sites,
                                        self._output_path_full)

        pf_keywords = {  # Full-name: [Initials, Abbreviation]
            "Spat. Ev.": ["SE", "Spatial Evolution"],
            "Temp. Ev.": ["TE", "Temporal Evolution"],
            "Heav. Dis.": ["HD", "Heaviside-Dispersion"],
            "GIF": ["GIF", "GIF"],
            "FFT": ["FFT", "Fast Fourier Transform"],
            "Prev. Menu": ["BACK", "Previous Menu"],
            "Ric. Paper": ["RIC", "Ricardo's Paper"],
            "Spat. FFT": ["SFFT", "Spatial FFT"]}

        if self.override_function is not None:
            pf_selection = self.override_function.upper()

        else:
            pf_selection = str(input(f"Options:"
                                     f"\n\t- {pf_keywords['Spat. Ev.'][0]}: {pf_keywords['Spat. Ev.'][1]} | "
                                     f"{pf_keywords['Temp. Ev.'][0]}: {pf_keywords['Temp. Ev.'][1]} | "
                                     f"{pf_keywords['Heav. Dis.'][0]}: {pf_keywords['Heav. Dis.'][1]}"
                                     f"\n\t- {pf_keywords['GIF'][0]}: {pf_keywords['GIF'][1]} | "
                                     f"{pf_keywords['FFT'][0]}: {pf_keywords['FFT'][1]} | "
                                     f"{pf_keywords['Ric. Paper'][0]}: {pf_keywords['Ric. Paper'][1]} | "
                                     f"\nTo return type {pf_keywords['Prev. Menu'][0]}: Select an option: ")).upper()

        pf_sel_first_elm = [values[0] for values in pf_keywords.values()]
        while True:
            if pf_selection in [values[0] for values in pf_keywords.values()]:
                break
            else:
                while pf_selection not in pf_keywords.keys():
                    pf_selection = str(input(f"Invalid choice. Select "
                                             f"from [{', '.join(pf_sel_first_elm)}]: ")).upper()

        cont_plotting = True

        if pf_selection == pf_keywords["Spat. Ev."][0]:
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
                        if row_num.upper() == pf_keywords["Prev. Menu"][0]:
                            self._invoke_paper_figures()
                        else:
                            print("ValueError. Please enter a valid string.")

                    else:
                        if row_num >= 0:
                            if not self.mass_produce:
                                print(f"Generating plot for [#{row_num}]...")
                            log.info(f"Generating PV plot for row [#{row_num}]")
                            paper_fig.plot_row_spatial(row_num, fixed_ylim=False, interactive_plot=True)
                            log.info(f"Finished plotting PV of row [#{row_num}]. Continuing...")

                            if self.early_exit:
                                cont_plotting = False

                        else:
                            print("Exiting PF-PV plotting.")
                            log.info(f"Exiting PF-PV based upon user input of [{row_num}]")
                            cont_plotting = False

        elif pf_selection == pf_keywords["Spat. FFT"][0]:
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
                        if row_num.upper() == pf_keywords["Prev. Menu"][0]:
                            self._invoke_paper_figures()
                        else:
                            print("ValueError. Please enter a valid string.")

                    else:
                        if row_num >= 0:
                            if not self.mass_produce:
                                print(f"Generating plot for [#{row_num}]...")
                            log.info(f"Generating PV plot for row [#{row_num}]")
                            paper_fig.plot_row_spatial_ft(row_num, fixed_ylim=False, interactive_plot=True)
                            log.info(f"Finished plotting PV of row [#{row_num}]. Continuing...")

                            if self.early_exit:
                                cont_plotting = False

                        else:
                            print("Exiting PF-PV plotting.")
                            log.info(f"Exiting PF-PV based upon user input of [{row_num}]")
                            cont_plotting = False
        elif pf_selection == pf_keywords["Temp. Ev."][0]:
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
                        if target_site.upper() == pf_keywords["Prev. Menu"][0]:
                            self._invoke_paper_figures()
                        else:
                            print("ValueError. Please enter a valid string.")

                    else:
                        if target_site >= 0:
                            print(f"Generating temporal evolution plot for [#{target_site}]...")

                            log.info(f"Generating PF-TV plot for Spin Site [#{target_site}]")
                            paper_fig.plot_site_temporal(target_site, wavepacket_fft=False, visualise_wavepackets=False,
                                                         annotate_precursors_fft=False, annotate_signal=False,
                                                         wavepacket_inset=False, add_key_params=False,
                                                         add_signal_backgrounds=False, publication_details=False,
                                                         interactive_plot=True)
                            log.info(f"Finished plotting PF-TV of Spin Site [#{target_site}]. Continuing...")

                            if self.early_exit:
                                cont_plotting = False

                        else:
                            print("Exiting PF-TV plotting.")
                            log.info(f"Exiting PF-TV based upon user input of [{target_site}]")
                            cont_plotting = False

        elif pf_selection == pf_keywords["Heav. Dis."][0]:
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
                        if target_site.upper() == pf_keywords["Prev. Menu"][0]:
                            self._invoke_paper_figures()
                        else:
                            print("ValueError. Please enter a valid string.")

                    else:
                        if target_site >= 0:
                            print(f"Generating Heaviside-Dispersion plot for [#{target_site}]...")
                            log.info(f"Generating PF-HD plot for Spin Site [#{target_site}]")
                            # paper_fig.plot_heaviside_and_dispersions(dispersion_relations=True,
                            #                                          use_dual_signal_inset=False,
                            #                                          show_group_velocity_cases=False,
                            #                                          dispersion_inset=False,
                            #                                          use_demag=False, compare_dis=True,
                            #                                          publication_details=False, interactive_plot=True)

                            paper_fig.find_degenerate_modes(find_modes=False, use_demag=False, interactive_plot=True)
                            log.info(f"Finished plotting PF-HD of Spin Site [#{target_site}]. Continuing...")

                            if self.early_exit:
                                cont_plotting = False

                        else:
                            print("Exiting PF-TV plotting.")
                            log.info(f"Exiting PF-TV based upon user input of [{target_site}]")
                            cont_plotting = False

        elif pf_selection == pf_keywords["GIF"][0]:
            paper_fig.create_gif(has_static_ylim=True)
            print("GIF successfully created!")
            if self.early_exit:
                exit(0)
            self._invoke_paper_figures()  # Use of override flag here will lead to an infinite loop!

        elif pf_selection == pf_keywords["FFT"][0]:
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
                        if target_site.upper() == pf_keywords["Prev. Menu"][0]:
                            self._invoke_paper_figures()
                        else:
                            print("ValueError. Please enter a valid string.")

                    else:
                        if target_site >= 1:
                            print(f"Generating plot for [#{target_site}]...")

                            log.info(f"Generating FFT plot for Spin Site [#{target_site}]")
                            paper_fig.plot_fft(target_site - 1, add_zoomed_region=False)
                            log.info(f"Finished plotting FFT of Spin Site [#{target_site}]. Continuing...")

                            if self.early_exit:
                                cont_plotting = False

                        else:
                            print("Exiting PF-FFT plotting.")
                            log.info(f"Exiting PF-FFT based upon user input of [{target_site}]")
                            cont_plotting = False

        elif pf_selection == pf_keywords["Ric. Paper"][0]:
            paper_fig.ricardo_paper()

        elif pf_selection == pf_keywords["Prev. Menu"][0]:
            self.call_methods()

        log.info(f"Plotting PF complete!")

    @staticmethod
    def _invoke_exit_conditions():
        print("Exiting program...")
        log.info(f"Exiting program from (select_plotter == EXIT)!")
        exit(0)

class ProcessData:

    def __init__(self):
        self.csv_reader = None
        self.is_sim_flags = None
        self.header_container = None
        self.header_titles = None
        self.header_values = None

    def process_selected_headers(self, csv_reader,
                                 header_container: SimulationFlagsContainer | SimulationParametersContainer,
                                 is_sim_flags: bool, header_titles=None, header_values=None):

        self._set_internal_attributes(csv_reader, header_container, is_sim_flags, header_titles, header_values)

        if self.header_values:
            # Titles and values in separate lines
            for title, value in zip(self.header_titles, self.header_values):
                self._set_instance_variable(title, value)
        else:
            # Titles and values in one line
            for i in range(0, len(self.header_titles), 2):
                title, value = self.header_titles[i], self.header_titles[i + 1]
                self._set_instance_variable(title, value)

        return_reader = self.csv_reader
        return_container = self.header_container
        self._return_and_reset_internal_attributes()
        return return_reader, return_container

    def _set_internal_attributes(self, csv_reader, header_container, is_sim_flags, header_titles=None,
                                 header_values=None):

        self.csv_reader = csv_reader
        self.is_sim_flags = is_sim_flags
        self.header_container = header_container

        if header_titles is None and header_values is None:
            self.header_titles = next(self.csv_reader)
            self.header_values = next(self.csv_reader)
        else:
            self.header_titles = header_titles
            self.header_values = header_values

    def _set_instance_variable(self, title, value):
        """
        Dynamically set the instance attributes if they match the input data.
        """
        if self.is_sim_flags:
            unpack_container = self.header_container.all_flags.items()
        else:
            unpack_container = self.header_container.all_parameters.items()

        mapped_title = self._apply_custom_mapping_to_string(title)

        for param_name, param_metadata in unpack_container:
            param_names = param_metadata['var_names']

            if mapped_title in param_names:
                # Dynamically set the instance attributes
                setattr(self.header_container, param_name, value)
                break

    def _return_and_reset_internal_attributes(self):

        if self.header_values:
            # Skip the next (blank) line for the newer formats where the titles and values are on separate lines
            next(self.csv_reader)

        self.csv_reader = None
        self.is_sim_flags = None
        self.header_container = None
        self.header_titles = None
        self.header_values = None

    @staticmethod
    def _apply_custom_mapping_to_string(input_string: Any, keep_units=False):
        """Handles the mapping of data names to their corresponding strings as per these rules."""

        def _abbreviations(string, target, replacement):
            if target in string:
                string = string.replace(target, replacement)
            return string

        def _parentheses(string, target, replacement):
            # Updated logic to handle parentheses according to specified rules
            pattern = re.compile(r"\((.*?)\)")
            matches = pattern.findall(string)
            for match in matches:
                formatted_match = f"({match})"
                if match == target:
                    # If match equals target and replacement is not empty, replace it; else remove it
                    if replacement:
                        string = string.replace(formatted_match, replacement)
                    else:
                        string = string.replace(formatted_match, "")
                elif len(match) == 1:
                    # Remove single-character matches and their parentheses
                    string = string.replace(formatted_match, "")

            return string

        def _units(string, target, replacement):
            target_formatted = f"[{target}]"
            if target_formatted in string:
                if keep_units:
                    string = string.replace(target_formatted, "")
                    string += f" _{replacement}"
                else:
                    string = re.sub(r"\[.*?]", "", string)  # r"\[.*?\]"
            return string

        custom_replacements = {
            'Abbreviations': {
                'function': _abbreviations,
                'mappings': {
                    'Frequency': 'Freq',
                    'Gyromagnetic': 'Gyro',
                    'No.': 'Num'
                }
            },
            'Parentheses': {
                'function': _parentheses,
                'mappings': {
                    'lower': 'Lower',
                    'upper': 'Upper',
                    'Shape': 'Shape',
                    'H0': '',
                    'H_D1': '',
                    'H_D2': '',
                    '2Pi*Y': '',
                    'Hz': ''
                }
            },
            'Units': {
                'function': _units,
                'mappings': {
                    'J': 'Joules',
                    'T': 'Tesla',
                    's': 'Seconds',
                    'J/m': 'JoulesPerMetre',
                    'kA/m': 'KiloAmperePerMetre',
                    'Hz': 'Hertz'
                }
            }
        }

        def to_lower_camel_case(s):
            # Split by non-alphanumeric characters and capitalize the first letter of each word except the first one
            s = re.sub(r"\.", "", s)
            parts = re.split(r'\W+', s)
            return parts[0].lower() + ''.join(word.capitalize() for word in parts[1:])

        for category_dict in custom_replacements.values():
            category_rule_func = category_dict['function']
            for find, swap in category_dict['mappings'].items():
                input_string = category_rule_func(input_string, find, swap)

        mapped_and_cased_input = to_lower_camel_case(input_string)

        return mapped_and_cased_input