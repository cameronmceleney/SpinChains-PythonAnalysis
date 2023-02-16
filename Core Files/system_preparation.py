#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard libraries
from datetime import datetime
import errno as errno
import logging as lg
import os as os
import re
from sys import platform

"""
    Description of what system_preparation.py does
"""

"""
    Core Details
    
    Author      : cameronmceleney
    Created on  : 04/03/2022 15:52
    Filename    : globalvariables
    IDE         : PyCharm
"""


class SystemSetup:
    """
    Common base class values.

    target_directory_found: If false, then will test to find if the named parent directory (and its
                            children) has already been created. If any dirs in the tree are missing
                            then they will be created.
    """

    def __init__(self):
        """
        Detect the user's operating system to allow for automatic data accessing

        -----
        Notes
        -----

        In addition to accessing data, this class can create a new directory for the current date (by default). It also
        allows the user to create custom names for the parent directory. This function should be run at the start of
        each workday.
        """

        self.has_target_dir_been_found = False

        self.input_data_directory = None
        self.output_data_directory = None
        self.logging_directory = None

        self.input_dir_name = ''
        self.output_dir_name = ''

    def detect_os(self, use_default=True, custom_input_dir_name='', custom_output_dir_name=''):
        """
        Detect the user's operating system.

        :param bool use_default: Select if the target parent directory has a customised name. Default is today's date.
        :param str custom_input_dir_name: User-defined name for the input directory.
        :param str custom_output_dir_name: User-defined name for the output directory.

        :return: [0] is the input data directory; this is the read-from location. [1] is the output data directory.
                 All plots and GIFs should be saved here
         """

        self._set_parameters(use_default, custom_input_dir_name, custom_output_dir_name)

        self._is_dir_name_valid(str_code="input", str_to_test=self.input_dir_name)
        self._is_dir_name_valid(str_code="output", str_to_test=self.output_dir_name)

        if platform == "linux" or platform == "linux2":
            raise SystemError("Detected Linux, which is not yet supported.")

        elif platform == "darwin":
            # OS X. This is the permanent location on my Macbook
            mac_dir_root = "/Users/cameronmceleney/CLionProjects/Data/"

            if not self.has_target_dir_been_found:
                self._create_directory(mac_dir_root, self.input_dir_name)

            self.input_data_directory = f"{mac_dir_root}{self.input_dir_name}/Simulation_Data/"
            self.output_data_directory = f"{mac_dir_root}{self.output_dir_name}/Outputs/"
            self.logging_directory = f"{mac_dir_root}{self.input_dir_name}/Logs/"

        elif platform == "win32" or platform == "win64":
            # Windows. This is the permanent location on my desktop
            windows_dir_root = "D:/Data/"

            if not self.has_target_dir_been_found:
                self._create_directory(windows_dir_root, self.input_dir_name)

            self.input_data_directory = f"{windows_dir_root}{self.input_dir_name}/Simulation_Data/"
            self.output_data_directory = f"{windows_dir_root}{self.output_dir_name}/Outputs/"
            self.logging_directory = f"{windows_dir_root}{self.input_dir_name}/Logs/"

        self._logging_setup()
        lg.info(f"Target (parent) directory is {self.input_dir_name}.")

    def _set_parameters(self, use_default, custom_input_dir_name, custom_output_dir_name):

        if use_default:
            # Guard clause to ensure custom names are implemented
            self.input_dir_name = self._date_of_today()
            self.output_dir_name = self._date_of_today()
            return

        # Custom directory name usage can lead to bugs - syntax must be correct (YYYY-MM-DD)
        if custom_input_dir_name:
            self.input_dir_name = custom_input_dir_name
        else:
            self.input_dir_name = str(input("Enter the name of the input directory: "))

        if custom_output_dir_name:
            self.output_dir_name = custom_output_dir_name
        else:
            self.output_dir_name = str(input("Enter the name of the output directory: "))

    def input_dir(self, should_print=False):

        if self.input_data_directory is None:
            raise ValueError("input_data_directory was None")

        if should_print:
            return print(self.input_data_directory)
        else:
            return self.input_data_directory

    def output_dir(self, should_print=False):

        if self.output_data_directory is None:
            raise ValueError("output_data_directory was None")

        if should_print:
            return print(self.output_data_directory)
        else:
            return self.output_data_directory

    def _logging_dir(self, should_print=False):

        if self.logging_directory is None:
            raise ValueError("logging_directory was None")

        if should_print:
            print(self.logging_directory)
            return
        else:
            return self.logging_directory

    @staticmethod
    def _date_of_today():
        """
        Finds and returns today's date in DD MMM YY format.

        For details on how to change the date format, see the official `documentation <https://strftime.org/>`_.

        :return: Today's date.
        """

        # Old version was ("%d %b %y"). Try for backwards compatibility if current code fails
        date = datetime.today().strftime("%Y-%m-%d")  # Use ISO 8601 for dates

        return date

    def _create_directory(self, root_dir_path, parent_dir_name, should_show_errors=False):
        """
        Create a tree of subdirectories to save simulation data.

        The following notes should be observed:

        * The Python code should be run **before** the C++ simulations to ensure all needed directories exist.

        :param str root_dir_path: Absolute path to location where parent will be created.
        :param str parent_dir_name: Name of parent directory.
        :param bool should_show_errors: Set (True) to show when a directory couldn't be created.

        :return: Nothing.
        """

        # Default behaviour is to use today's date as the dir name; matching the behaviour of C++ code
        parent_dir_path = os.path.join(root_dir_path, parent_dir_name)

        # Create set (to avoid duplicates) of all the directories to be made under the parent_dir
        sub_dirs = {"Simulation_Data", "Outputs", "Logs"}
        path_list = []

        for i, child_dir_name in enumerate(sub_dirs):
            # Create list of paths to all subdirectories to be created.
            path_list.append(os.path.join(parent_dir_path, child_dir_name))

        for i, child_dir_name in enumerate(sub_dirs):
            try:
                # Try to create each subdirectory (and parent if needed). Always show instances of dirs being created
                os.makedirs(path_list[i], exist_ok=False)
                print(f"Directory '{child_dir_name}' created successfully")

            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

                # Handles exceptions from raise
                if should_show_errors:
                    # User-selected flag in function argument
                    print(f"Directory '{child_dir_name}' already exists.")
                    pass

        self.has_target_dir_been_found = True

    def _is_dir_name_valid(self, str_code, str_to_test=''):
        """
        Tests if a given directory exists.

        If a directory does exist, but the directory name is not in the standard format, then the user is given a
        warning before continuing. This method recursively calls itself to check a valid directory name
        """

        # Checks if a given string is in the format ####-##-## where # is a digit from 0-9
        regex_pattern = re.compile('\d{4}-\d{2}-\d{2}')

        if not str_to_test:
            # Testing self.input_dir_name is the primary purpose of this function.
            str_to_test = self.input_dir_name

        if regex_pattern.match(str_to_test) is not None:
            # Directory is named in the correct format
            return True

        elif regex_pattern.match(str_to_test) is None:
            # Unconventional naming format detected

            dir_accepted_answers = ['Y', 'N']

            dir_response = input(f"Your {str_code.upper()} directory ({str_to_test}) is not in the format ####-##-##\n"
                                  "Was this intentional [Y/N]? ").upper()

            while True:
                if dir_response in dir_accepted_answers:
                    if dir_response == 'Y':
                        return True

                    elif dir_response == 'N':
                        # Recursively call until a valid dir name is entered
                        if str_code == "input":
                            self.input_dir_name = input(f"Enter the name of the input directory: ")
                            self._is_dir_name_valid(str_code="input", str_to_test=self.input_dir_name)

                        if str_code == "output":
                            self.output_dir_name = input(f"Enter the name of the output directory: ")
                            self._is_dir_name_valid(str_code="output", str_to_test=self.output_dir_name)
                    break
                else:
                    # Force user to enter Y/N to above question
                    while dir_response not in dir_accepted_answers:
                        dir_response = input("Invalid option. Was this intentional [Y/N]? ").upper()

    def _logging_setup(self):
        """Initialisation of basic logging information."""

        if self.logging_directory is None:
            raise ValueError("Logging directory path was found to be 'None'.")

        today_date = datetime.now().strftime("%y%m%d")
        current_time = datetime.now().strftime("%H%M")

        lg.basicConfig(filename=f'{self.logging_directory}/{today_date}-{current_time}.log',
                       filemode='w',
                       level=lg.INFO,
                       format='%(asctime)s | %(module)s::%(funcName)s | %(levelname)s | %(message)s',
                       datefmt='%Y-%m-%d %H:%M:%S',
                       force=True)
