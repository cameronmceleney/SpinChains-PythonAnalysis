#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Description.

Long description.

Constants:

Examples:
    (Here, place useful implementations of the contents of test_file.py). Note that leading symbol '>>>' includes the
    code in doctests, while '$' does not.)::

        >>>

Todo:

References:
    Style guide: `Google Python Style Guide`_

Notes:
    File version
        0.3.1
    Project
        SpinChains-PythonAnalysis
    Path
        Core Files/data_analysis.py
    Author
        Cameron Aidan McEleney < c.mceleney.1@research.gla.ac.uk >
    Created
        12 Mar 2022
    IDE
        PyCharm

.. _Google Python Style Guide:
   https://google.github.io/styleguide/pyguide.html
"""
__all__ = ['']

# Standard library imports
import csv
from dataclasses import dataclass, field
from errno import EEXIST
import logging as log
from textwrap import dedent
from typing import Any, Optional, Callable

# Third-party imports
from glob import glob
import numpy as np
from numpy.typing import NDArray
import shutil
import os
import re

# Local application imports
from attribute_defintions import SimulationParametersContainer, SimulationFlagsContainer
from figure_manager import rc_params_update
from plot_rk_methods import PaperFigures
from plot_eigenmodes import Eigenmodes
import plot_rk_methods_legacy_standalone as plt_rk_legacy_standalone

# Module-level constants
FILE_NAME_COMPONENTS: dict[str, Optional[str]] = dict.fromkeys(['prefix', 'component', 'identifier', 'descriptor'])
"""Components (i.e. subsections) from the filename holding the dataset are used to name and label output figures."""

FILE_PATHS: dict[str, Optional[str]] = dict.fromkeys(['filename', 'input', 'output'])
"""Absolute paths to the: dataset file, dataset directory, and output directory for saving figures."""


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

    def import_eigenmodes(self) -> None:
        """Populate containers to store key information from the loaded dataset files.

        Determines which formatted files, if any, are missing in the provided directory; generating them if required.
        Iterating through containers was felt to be easier rather than reading many lines of variable declarations
        and initialisations.
        """

        self._check_directory_tree()

        if all(self.is_formatted_data_present):
            print(f"\nAll required files found. Loading data into memory...")

            for i, filename in enumerate(self.formatted_filenames):
                file_to_search_for = self.input_dir_path + filename
                self._arrays_to_output[i] = np.loadtxt(file_to_search_for, delimiter=',')

        else:
            print(f"\nSome required files were not found. Need to generate formatted files from datasets...")

            for i, does_file_exist in enumerate(self.is_formatted_data_present):
                try:
                    if not does_file_exist:
                        self._controller_for_generating_missing_files(i)

                except ValueError:
                    log.exception(f"PlotEigenmodes.import_eigenmodes: Encountered `does_file_exist` as None, indicating"
                                  f"some upstream issue has occurred.")
                    exit(1)

    def plot_eigenmodes(self):
        """Allows the user to plot as many eigenmodes as they would like; one per figure.

        This function is primarily used to replicate Fig. 1 from `macedo2021breaking`. The use of keywords within this
        function also allows the user to plot the 'generalised fourier coefficients' of a system; mainly used to replicate
        Figs 4.a & 4.d of the same paper.
        """
        log.info(f"Invoking functions to plot data...")

        print(f"""
            {'-' * 80}
            Plot the eigenmodes of the selected data. Input the requested modes as 
            single values, or as a space-separated list. Unique [keywords] include:
                    *   Exit [EXIT] (Quit the program)"
                    *   Fourier Coefficients [FRC] (Plot generalised Fourier co-efficients)
                    *   Dispersion Relation [DIS] (Plot the dispersion relation)

            Note: None of the keywords are case-sensitive.
            {'-' * 80}
            """)

        dataset_eigenmodes = Eigenmodes(self._arrays_to_output[0],
                                        self._arrays_to_output[1],
                                        self._arrays_to_output[2],
                                        f"{self.fi}{self.fd}",
                                        self.input_dir_path,
                                        self.full_output_path)

        largest_eigenmode = self._arrays_to_output[0].size  # Largest valid eigenmode in dataset
        previous_plotted_eigenmodes = set()

        should_continue_plotting = True
        while should_continue_plotting:
            requested_tokens = input("Enter eigenmode(s) to plot: ").split()

            # Advanced users might pass a keyword (instead of eigenmodes) to trigger additional plotting behaviour.
            for token in requested_tokens:
                match token.upper():
                    case 'EXIT':
                        should_continue_plotting = False
                        break

                    case 'FRC':
                        dataset_eigenmodes.generalised_fourier_coefficients(use_defaults=False)
                        break

                    case 'DIS':
                        dataset_eigenmodes.plot_dispersion_relation()
                        break

                # Check numeric eigenmodes
                try:
                    eigenmode = int(token)
                    if eigenmode in previous_plotted_eigenmodes: raise SyntaxError
                    if not (1 <= eigenmode <= largest_eigenmode): raise IndexError

                except ValueError:
                    # Serious issue, so reject entire batch.
                    print(f"[{token}] isn't an integer and is not a known keyword.")
                    break

                except SyntaxError:
                    # Reject current duplicate and proceed with batch.
                    print(f"Skipping eigenmode [{token}] as it has already been plotted.")
                    continue

                except IndexError:
                    # Reject current out-of-bounds eigenmode and proceed with batch.
                    print(f"Eigenmode [{token}] is out of dataset's range [1, {largest_eigenmode}].")
                    continue

                else:
                    dataset_eigenmodes.plot_single_eigenmode(eigenmode, has_endpoints=False)
                    previous_plotted_eigenmodes.add(eigenmode)

        print("Exiting program...")
        exit(0)

    def _check_directory_tree(self, should_show_errors=False):

        if os.path.exists(self.input_dir_path):
            print(f"Simulation_Data/{self.child_dir_name}: directory found")
            self.is_input_dir_present = True
            self._check_if_files_present(True)

        elif os.path.exists(f"{self.parent_dir_path}{self.fi}{self.fd}/"):
            # Included to handle legacy files
            os.rename(f"{self.parent_dir_path}{self.fi}{self.fd}", self.input_dir_path)
            print(f"Simulation_Data/{self.fi}{self.fd}: directory found. Name updated to {self.child_dir_name}")
            self.is_input_dir_present = True
            self._check_if_files_present(True)

        else:
            print(f"Simulation_Data/{self.child_dir_name}: not found")
            try:
                # Try to create each subdirectory (and parent if needed). Always show instances of dirs being created
                os.makedirs(self.input_dir_path, exist_ok=False)
                print(f"Directory '{self.child_dir_name}' created successfully")
                self.is_input_dir_present = True
                self.import_eigenmodes()

            except OSError as exc:
                if exc.errno != EEXIST:
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

                if os.path.exists(file_to_search_for):
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

            if os.path.exists(file_to_search_for):
                self.is_formatted_data_present[i] = True
                print(f"{file_description}: found")
            else:
                self.is_formatted_data_present[i] = False
                print(f"{file_description}: not found")
        print('--------------------------------------------------------------------------------')

        if all(item is False for item in self.is_formatted_data_present):

            if len(os.listdir(self.input_dir_path)) == 0:
                # Target directory is empty
                src_folder = self.parent_dir_path
                dst_folder = self.input_dir_path

                # Search files with .txt extension in source directory
                pattern = f"/*{self.fi}{self.fd}.csv"
                files = glob(src_folder + pattern)

                # move the files with txt extension
                for file in files:
                    # extract file name form file path
                    file_name = os.path.basename(file)
                    shutil.move(file, dst_folder + file_name)
                    print('Moved:', file)
            elif all(item is True for item in self.is_input_data_present):
                for i in range(0, 3):
                    self._controller_for_generating_missing_files(i)
                    self.import_eigenmodes()

            else:
                print('Unknown error. Exiting.')
                exit(1)

    def _controller_for_generating_missing_files(self, index: int) -> None:
        """Select private method with which to generate the missing, required, filtered file(s)."""

        match index:
            case 0 | 1:
                self._generate_missing_eigenvectors()

            case 2:
                self._generate_missing_eigenvalues()

            case _:
                log.error(f"PlotEigenmodes._generate_missing_file_controller: Index [{index}] was passed which"
                          f"doesn't correspond to a defined match-case.")
                exit(1)

    def _generate_missing_eigenvalues(self):

        log.info(f"Missing Eigenvalues file found. Attempting to generate new file in correct format...")

        # 'Raw' refers to the data produces from the C++ code.
        print(f"Here: {self.input_dir_path}")
        eigenvalues_raw = np.loadtxt(f"{self.input_dir_path}eigenvalues_{self.fi}{self.fd}.csv",
                                     delimiter=",")

        eigenvalues_filtered = np.sort(eigenvalues_raw[eigenvalues_raw >= 0])

        # Use np.savetxt to save the data (2nd parameter) directly to the files (first parameter).
        np.savetxt(f"{self.input_dir_path}{self.formatted_filenames[2]}", eigenvalues_filtered,
                   delimiter=',')

        log.info(f"Successfully generated missing (eigenvalues) file, which is saved in {self.input_dir_path}")

    def _generate_missing_eigenvectors(self, should_separate_afm_modes=False):

        log.info(f"Missing (mx) and/or (my) file(s) found. Attempting to generate new files in correct format...")

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

        log.info(f"Successfully generated missing (mx) and (my) files, which are saved in {self.input_dir_path}")


class AnalyseData:

    def __init__(self):
        self.data_container = None
        self.data_magnetic_moments = None
        self.data_timestamps = None

        self.header_flags = None
        self.header_parameters = None
        self.header_simulated_sites = None

        self._file_terms: FILE_NAME_COMPONENTS = FILE_NAME_COMPONENTS
        self._file_paths_full: FILE_PATHS = FILE_PATHS

    def import_data(self, file_descriptor, input_dir_path, output_dir_path, file_prefix="rk2", file_component="mx",
                    file_identifier="T", auto_run: bool = True):

        self._file_terms = {"prefix": file_prefix, "component": file_component, "identifier": file_identifier,
                            "descriptor": file_descriptor}

        imported_data = ImportData(self._file_terms, input_dir_path, output_dir_path)

        if auto_run:
            (self.data_container, self.data_timestamps, self.data_magnetic_moments,
             data_headers, self._file_paths_full) = imported_data.default_import()

            self.header_parameters, self.header_simulated_sites, self.header_flags = data_headers
        else:
            print("Data imported. Ready to process.")
            exit(0)

    def process_data(self):

        # Process simulated sites
        if "Time [s]" in self.header_simulated_sites:
            self.header_simulated_sites.remove("Time [s]")

        if ' ' in self.header_simulated_sites:
            self.header_simulated_sites.remove(' ')

        if 'lattice_constant' not in self.header_parameters.keys():
            self.header_parameters['lattice_constant'] = 1e-9

        rc_params_update()

        print("Data processed. Ready to call method.")

    def call_methods(
            self,
            early_exit: Optional[bool] = False,
            loop_function: Optional[bool] = False,
            mass_produce: Optional[bool] = False,
            interactive_mode: Optional[bool] = False,
            *,
            override_method: Optional[str] = None,
            override_function: Optional[str] = None,
            override_site: Optional[int] = None,
            **kwargs
    ) -> None:
        called_method = CallMethods(self._file_terms, self._file_paths_full, self.data_container, self.data_timestamps,
                                    self.header_simulated_sites, self.data_magnetic_moments, self.header_parameters,
                                    self.header_flags, mass_produce)
        kwargs.get('override_method')
        called_method.call_methods(override_method=override_method, override_function=override_function,
                                   override_site=override_site, loop_function=loop_function, early_exit=early_exit,
                                   interactive_mode=interactive_mode)


class ImportData:
    def __init__(self, file_terms, input_dir_path, output_dir_path):

        self._input_dir_path = input_dir_path
        self._output_dir_path = output_dir_path
        self._input_prefix = file_terms["prefix"]
        self._input_comp = file_terms["component"]
        self._input_id = file_terms["identifier"]
        self._input_desc = file_terms["descriptor"]

        self._file_paths_full: FILE_PATHS = FILE_PATHS

        self._data_container = None
        self._data_timestamps = None
        self._data_magnetic_moments = None
        self._data_headers = [None, None, None]

        self._set_internal_attributes()

    def _set_internal_attributes(self):
        filename = f"{self._input_prefix}_{self._input_comp}_{self._input_id}{self._input_desc}"
        self._file_paths_full = {"filename": filename,
                                 "input": f"{self._input_dir_path}{filename}.csv",
                                 "output": f"{self._output_dir_path}{self._input_id}{self._input_desc}"}

    def _import_simulation_data(self, filename=None, full_path_to_file=None):
        """
        Outputs the data needed to plot single-image panes.

        Contained in single method to unify processing option. Separated from import_data_headers() (unlike in previous
        files) for when multiple datafiles, with the same header, are imported.
        """
        log.info(f"Importing data points...")

        if filename is None:
            filename = self._file_paths_full['filename']
        if full_path_to_file is None:
            full_path_to_file = self._file_paths_full['input']

        # Loads all input data without the header
        # Check if the file exists
        if not os.path.exists(full_path_to_file):
            log.error(f"File {filename} was not found")
            print(f"File {filename} was not found")
            exit(1)

        # Loads all input data without the header, handling missing values as NaN
        try:
            data = np.loadtxt(full_path_to_file, delimiter=",", skiprows=11)
            log.info(f"Data points imported successfully!")
            return data
        except ValueError as ve:

            log.error(f"Failed to import data points due to: {ve}")
            print(f"During _import_simulation_data().\n\tValueError: {ve}")
            exit(1)

    def _import_simulation_headers(self):

        parameters = SimulationParametersContainer()
        flags = SimulationFlagsContainer()
        process_data = ProcessData()

        with open(str(self._file_paths_full['input'])) as file_header_data:
            csv_reader = csv.reader(file_header_data)
            next(csv_reader)  # Skip the 0th line.
            next(csv_reader)  # Skip the 1st line (always blank).

            # Process simulation flags (2nd and 3rd lines)
            csv_reader, move_reader, flags = process_data.process_selected_headers(csv_reader, flags, is_sim_flags=True)
            if move_reader:
                next(csv_reader)

            # Additional blank (4th line) in newer format; older format has blank 3rd and key sim params at 4th

            # Count from here is for new format. Process simulation parameters (5th and 6th lines)
            csv_reader, move_reader, parameters = process_data.process_selected_headers(csv_reader, parameters, is_sim_flags=False)

            # Skip lines until simulated sites (11th line)
            for _ in range(4):
                next(csv_reader)  # Skip lines 7th (blank), 8th (simulation notes), 9th (descriptions), 10th (blank)
            simulated_sites = next(csv_reader)  # 11th line

        return parameters.return_data(), simulated_sites, flags.return_data()

    def default_import(self):
        self._data_container = self._import_simulation_data()

        self._data_timestamps = self._data_container[:, 0] / 1e-9  # Convert to from [seconds] to [ns]
        self._data_magnetic_moments = self._data_container[:, 1:]

        self._data_headers = self._import_simulation_headers()

        return (self._data_container, self._data_timestamps, self._data_magnetic_moments, self._data_headers,
                self._file_paths_full)


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

        should_move_reader = False
        if self.header_values:
            # Skip the next (blank) line for the newer formats where the titles and values are on separate lines
            should_move_reader = True

        return_reader = self.csv_reader
        return_container = self.header_container
        self._return_and_reset_internal_attributes()
        return return_reader, should_move_reader, return_container

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
                    'Hz': 'Hertz',
                    'm': 'Metres'
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


@dataclass(frozen=True)
class MethodSpec:
    abbr: str
    handler_name: str
    """Name of the method available on this class instance."""


@dataclass(frozen=True)
class BoundMethodSpec:
    abbr: str
    handler: Callable[..., Any]
    description: str


class CallMethods:
    """Docstring for class."""
    _UNBOUND_METHOD_SPECS: dict[str, MethodSpec] = {
        'three_panes': MethodSpec('3P', 'invoke_three_panes'),
        'fft_spatial': MethodSpec('FS', 'invoke_fs_functions'),
        'fft_temporal': MethodSpec('FT', 'invoke_fft_functions'),
        'paper_figures': MethodSpec('PF', 'invoke_paper_figures'),
        'contour_plots': MethodSpec('CP', 'invoke_contour_plot'),
        'safe_exit': MethodSpec('EXIT', 'invoke_exit_conditions'),
    }
    """Supported means for generating figures."""

    _file_terms: FILE_NAME_COMPONENTS = FILE_NAME_COMPONENTS
    _override_options: dict[str, Optional[int | str]] = {'method': None, 'function': None, 'site': None}

    called_method: str
    early_exit: bool = False
    loop_function: bool = False
    interactive_mode: bool = False

    def __init__(
            self,
            file_terms: FILE_NAME_COMPONENTS,
            file_paths: FILE_PATHS,
            data_container: NDArray[tuple[Any, ...]],
            data_timestamps,
            header_simulated_sites,
            data_magnetic_moments,
            header_parameters,
            header_flags,
            mass_produce: bool = False
    ):
        """Docstring for class instance."""
        self.file_terms = file_terms
        self._file_paths_full = file_paths

        self.data_container = data_container
        self._data_timestamps = data_timestamps
        self._data_magnetic_moments = data_magnetic_moments

        self._header_parameters = header_parameters
        self._header_flags = header_flags
        self._header_simulated_sites = header_simulated_sites

        self.mass_produce: bool = mass_produce

        # Bind each 'handler' to the targeted method instance.
        self.accepted_methods: dict[str, BoundMethodSpec] = {}

        for key, spec in type(self)._UNBOUND_METHOD_SPECS.items():
            fn = getattr(self, '_' + spec.handler_name)
            self.accepted_methods[key] = BoundMethodSpec(
                abbr=spec.abbr,
                handler=fn,
                description=fn.__doc__ or ""
            )

    def _process_selection_on_initialisation(self) -> str | None:
        """Define the initial method used for the selection process.

        Inform the user of how selection is performed if no override was correctly passed.
        """
        user_method_selection = (self._override_options['method']
                                 if self._override_options['method']
                                 else None)

        if not user_method_selection:
            print(dedent(f"""\
                    The plotting functions available are:

                    *   Three Panes  [3P] (Plot all spin sites varying in time, and compare a selection)
                    *   FFT & Signal [FS] (Examine signals from site(s), and the corresponding FFT)
                    *   FFT only     [FT] (Interactive plot that outputs (x,y) of mouse click to console)
                    *   Paper Figure [PF] (Plot final state of system at all sites)
                    *   Contour Plot [CP] (Plot a single site as a 3D map)

                    The terms within the square brackets are the keys for each function. 
                    If you wish to exit the program then type EXIT. Keys are NOT case-sensitive.
                    {'-' * 80}\n"""
                         ))

        if any(v is not None for v in self._override_options.values()) or self.early_exit:
            if self.mass_produce:
                output = f"Producing: {self._file_paths_full['filename']}"
            else:
                output = f"""\
                            Override(s) enabled.
                            Method: {self._override_options['method']} | Function: {self._override_options['function']}
                            Site/Row: {self._override_options['site']} | Early Exit: {self.early_exit}
                            {'-'*80}"""
            print(dedent(output))
            log.info(output)

        return user_method_selection

    def call_methods(
            self,
            *,
            early_exit: bool = False,
            loop_function: bool = False,
            interactive_mode: bool = False,
            override_method: Optional[str] = None,
            override_function: Optional[str] = None,
            override_site: Optional[int] = None
    ) -> None:
        """Call plotting method for valid user input.

        This method is the caller for setting the internal attributes used by the class to manage general plotting
        behaviours: see ``self._set_internal_attributes()``.
        """

        # Dictate exit conditions
        attempts, attempts_max = 0, 4
        continue_calling_methods = True

        # Set internal attributes
        # TODO - Turn these internal attributes of the class instance into a dataclass.
        self.early_exit = early_exit
        self.loop_function = False  # TODO - Has been set to always FALSE as this attribute's implementation fails.
        self.interactive_mode = interactive_mode

        if isinstance(override_method, str):
            self._override_options['method'] = override_method.upper()

        if isinstance(override_function, str):
            self._override_options['function'] = override_function

        if isinstance(override_site, int):
            self._override_options['site'] = override_site

        user_method_selection = self._process_selection_on_initialisation()

        while continue_calling_methods:

            if user_method_selection is None:
                user_method_selection = input("Select function to use: ").upper()

            spec = next(
                (s for s in self.accepted_methods.values() if s.abbr == user_method_selection),
                None
            )

            if spec is None:
                # Invalid abbreviation
                attempts += 1
                valid = ', '.join(s.abbr for s in self.accepted_methods.values())
                print(f"Invalid option. Valid abbreviations are: {valid}")

                # Guards to check invalid attempts
                if attempts > attempts_max:
                    print(f"Maximum attempts [{attempts_max}] exceeded. Exiting...")
                    break

                user_method_selection = None
                continue

            else:
                # Found and called a handler.
                spec.handler()

                if self.early_exit:
                    break

                attempts = 0
                user_method_selection = None

        if self.mass_produce:
            log.info(f"Produced: {self.file_terms['identifier']}{self.file_terms['descriptor']}")
        else:
            log.info("Code complete! Exiting.")

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
        plt_rk_legacy_standalone.three_panes(self._data_magnetic_moments[:, :], self._header_parameters,
                                             self._file_paths_full['output'], sites_to_compare)
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
                                                                 target_spin, self._header_parameters,
                                                                 self._file_paths_full['output'])
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
                                                      self._header_parameters,
                                                      self._file_paths_full['output'])
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

        mag_moment_components = ["mx", "my", "mz"]
        mx_data = None
        my_data = None
        mz_data = None

        def update_file_path(file_path, old_component, new_component):
            # Only works for windows filepaths
            path_parts = file_path.split('\\\\')
            new_file_name = path_parts[-1].replace(old_component, new_component)
            updated_file_path = '\\\\'.join(path_parts[:-1] + [new_file_name])
            return updated_file_path

        local_import = ImportData(self.file_terms, self.file_paths_full['input'], self.file_paths_full['output'])

        for component in mag_moment_components:
            filename = (f"{self.file_terms['prefix']}_{component}_{self.file_terms['identifier']}"
                        f"{self.file_terms['descriptor']}")
            component_path = update_file_path(self.file_paths_full['input'], self.file_terms['component'], component)
            data = local_import._import_simulation_data(filename, component_path)

            match component:
                case "mx":
                    mx_data = data
                case "my":
                    my_data = data
                case "mz":
                    mz_data = data

        # plt_rk.create_contour_plot(mx_m_data, my_m_data, mz_m_data, spin_site, self._output_path_full, False)
        plt_rk_legacy_standalone.test_3d_plot(mx_data, my_data, mz_data, spin_site)
        log.info(f"Plotting CP complete!")

    def _invoke_paper_figures(self):
        # Plots final state of system, similar to the Figs. in macedo2021breaking.
        log.info(f"Plotting function selected: paper figure.")

        # if self._input_prefix == "rk2":
        #    paper_fig = plt_rk_legacy.PaperFigures(self._data_timestamps, self._data_magnetic_moments,
        #                                    self._data_parameters, self._data_flags, self._data_sites,
        #                                    self._output_path_full)
        # else:

        paper_fig = PaperFigures(self._data_timestamps, self._data_magnetic_moments,
                                        self._header_parameters, self._header_flags, self._header_simulated_sites,
                                        self._file_paths_full['output'])

        pf_keywords = {  # Full-name: [Initials, Abbreviation]
            "Spat. Ev.": ["SE", "Spatial Evolution"],
            "Temp. Ev.": ["TE", "Temporal Evolution"],
            "Heav. Dis.": ["HD", "Heaviside-Dispersion"],
            "GIF": ["GIF", "GIF"],
            "FFT": ["FFT", "Fast Fourier Transform"],
            "Prev. Menu": ["BACK", "Previous Menu"],
            "Ric. Paper": ["RIC", "Ricardo's Paper"],
            "Spat. FFT": ["SFFT", "Spatial FFT"]}

        min_site = min([int(site) for site in self._header_simulated_sites])
        max_site = max([int(site) for site in self._header_simulated_sites])
        min_row, max_row = 0, int(len(self._data_timestamps) - 1)

        if self._override_options['function'] is not None:
            pf_selection = self._override_options['function'].upper()

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
                if self._override_options['site'] is not None:
                    rows_to_plot = [self._override_options['site']]
                else:
                    rows_to_plot = (input("Plot which rows of data (-ve to exit): ")).split()

                for row_num in rows_to_plot:

                    try:
                        row_num = int(row_num)

                        if row_num < min_row or row_num > max_row:
                            # Manually raise IndexError if target_site is out of the allowed range
                            raise IndexError

                    except ValueError:
                        if row_num.upper() == pf_keywords["Prev. Menu"][0]:
                            self._invoke_paper_figures()
                        else:
                            print("ValueError. Please enter a valid string.")

                    except IndexError:
                        print(f"IndexError. You chose an invalid row of data: [{row_num}] "
                              f"(options are: {min_row} <= site <= {max_row}).")
                        self._override_options['site'] = None
                        self._invoke_paper_figures()

                    else:
                        if row_num >= 0:
                            if not self.mass_produce:
                                print(f"Generating plot for [#{row_num}]...")
                            log.info(f"Generating PV plot for row [#{row_num}]")
                            paper_fig.plot_row_spatial(row_num, fixed_ylim=False,
                                                       interactive_plot=self.interactive_mode)
                            log.info(f"Finished plotting PV of row [#{row_num}]. Continuing...")

                            if not self.loop_function:
                                cont_plotting = False

                        else:
                            print("Exiting PF-PV plotting.")
                            log.info(f"Exiting PF-PV based upon user input of [{row_num}]")
                            cont_plotting = False

        elif pf_selection == pf_keywords["Spat. FFT"][0]:
            while cont_plotting:
                # User will plot one spin site at a time, as plotting can take a long time.
                if self._override_options['site'] is not None:
                    rows_to_plot = [self._override_options['site']]
                else:
                    rows_to_plot = (input("Plot which rows of data (-ve to exit): ")).split()

                for row_num in rows_to_plot:

                    try:
                        row_num = int(row_num)

                        if row_num < min_row or row_num > max_row:
                            # Manually raise IndexError if target_site is out of the allowed range
                            raise IndexError

                    except ValueError:
                        if row_num.upper() == pf_keywords["Prev. Menu"][0]:
                            self._invoke_paper_figures()
                        else:
                            print("ValueError. Please enter a valid string.")

                    except IndexError:
                        print(f"IndexError. You chose an invalid row of data: [{row_num}] "
                              f"(options are: {min_row} <= site <= {max_row}).")
                        self._override_options['site'] = None
                        self._invoke_paper_figures()

                    else:
                        if row_num >= 0:

                            if not self.mass_produce:
                                print(f"Generating plot for [#{row_num}]...")
                            log.info(f"Generating PV plot for row [#{row_num}]")
                            paper_fig.plot_row_spatial_ft(row_num, fixed_ylim=False,
                                                          interactive_plot=self.interactive_mode)
                            log.info(f"Finished plotting PV of row [#{row_num}]. Continuing...")

                            if self.loop_function:
                                self._override_options['site'] = None
                            else:
                                cont_plotting = False

                        else:
                            print("Exiting PF-PV plotting.")
                            log.info(f"Exiting PF-PV based upon user input of [{row_num}]")
                            cont_plotting = False

        elif pf_selection == pf_keywords["Temp. Ev."][0]:
            while cont_plotting:

                if self._override_options['site'] is not None:
                    sites_to_plot = [self._override_options['site']]
                else:
                    sites_to_plot = (input("Plot which site (-ve to exit): ")).split()

                # Loops through user's inputs, ensuring that they are correct.
                for target_site in sites_to_plot:

                    try:
                        target_site = int(target_site)

                        if target_site < min_site or target_site > max_site:
                            # Manually raise IndexError if target_site is out of the allowed range
                            # raise IndexError
                            pass

                    except ValueError:
                        if target_site.upper() == pf_keywords["Prev. Menu"][0]:
                            self._invoke_paper_figures()
                        else:
                            print("ValueError. Please enter a valid string.")

                    except IndexError:
                        print(f"IndexError. You chose an invalid site [{target_site}] "
                              f"(options are: {min_site} <= site <= {max_site}).")
                        self._override_options['site'] = None
                        self._invoke_paper_figures()

                    else:
                        if target_site >= 0:
                            print(f"Generating temporal evolution plot for [#{target_site}]...")

                            log.info(f"Generating PF-TV plot for Spin Site [#{target_site}]")
                            paper_fig.plot_site_temporal(target_site, wavepacket_fft=False, visualise_wavepackets=False,
                                                         annotate_precursors_fft=False, annotate_signal=False,
                                                         wavepacket_inset=False, add_key_params=False,
                                                         add_signal_backgrounds=False, publication_details=True,
                                                         interactive_plot=self.interactive_mode)
                            log.info(f"Finished plotting PF-TV of Spin Site [#{target_site}]. Continuing...")

                            if not self.loop_function:
                                cont_plotting = False

                        else:
                            print("Exiting PF-TV plotting.")
                            log.info(f"Exiting PF-TV based upon user input of [{target_site}]")
                            cont_plotting = False

        elif pf_selection == pf_keywords["Heav. Dis."][0]:
            while cont_plotting:

                if self._override_options['site'] is not None:
                    sites_to_plot = [self._override_options['site']]
                else:
                    sites_to_plot = (input("Plot which site (-ve to exit): ")).split()

                # Loops through user's inputs, ensuring that they are correct.
                for target_site in sites_to_plot:

                    try:
                        target_site = int(target_site)

                        if target_site < min_site or target_site > max_site:
                            # Manually raise IndexError if target_site is out of the allowed range
                            raise IndexError

                    except ValueError:
                        if target_site.upper() == pf_keywords["Prev. Menu"][0]:
                            self._invoke_paper_figures()
                        else:
                            print("ValueError. Please enter a valid string.")

                    except IndexError:
                        print(f"IndexError. You chose an invalid site [{target_site}] "
                              f"(options are: {min_site} <= site <= {max_site}).")
                        self._override_options['site'] = None
                        self._invoke_paper_figures()

                    else:
                        if target_site >= 0:
                            print(f"Generating Heaviside-Dispersion plot for [#{target_site}]...")
                            log.info(f"Generating PF-HD plot for Spin Site [#{target_site}]")
                            # paper_fig.plot_heaviside_and_dispersions(dispersion_relations=True,
                            #                                          use_dual_signal_inset=False,
                            #                                          show_group_velocity_cases=False,
                            #                                          dispersion_inset=False,
                            #                                          use_demag=False, compare_dis=True,
                            #                                          publication_details=False,
                            #                                          interactive_plot=self.interactive_mode)

                            paper_fig.find_degenerate_modes(find_modes=False, use_demag=True,
                                                            interactive_plot=self.interactive_mode)
                            log.info(f"Finished plotting PF-HD of Spin Site [#{target_site}]. Continuing...")

                            if not self.loop_function:
                                cont_plotting = False

                        else:
                            print("Exiting PF-TV plotting.")
                            log.info(f"Exiting PF-TV based upon user input of [{target_site}]")
                            cont_plotting = False

        elif pf_selection == pf_keywords["GIF"][0]:
            paper_fig.create_gif(has_static_ylim=True)
            print("GIF successfully created!")
            if not self.loop_function:
                exit(0)
            self._invoke_paper_figures()  # Use of override flag here will lead to an infinite loop!

        elif pf_selection == pf_keywords["FFT"][0]:
            while cont_plotting:
                # User will plot one spin site at a time, as plotting can take a long time.
                if self._override_options['site'] is not None:
                    sites_to_plot = [self._override_options['site']]
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

                            if not self.loop_function:
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
