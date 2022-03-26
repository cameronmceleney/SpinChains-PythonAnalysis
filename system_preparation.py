#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard libraries
from datetime import datetime
import errno as errno
import logging as lg
import os as os
from sys import platform, exit

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


def system_setup(has_directory_been_created=False, has_custom_name=False):
    data, outputs, logging = directory_tree_testing(has_directory_been_created, has_custom_name)
    logging_setup(logging)


def directory_tree_testing(has_directory_been_created=True, has_custom_name=False):
    """Detect the user's operating system to allow for automatic data accessing

    -----
    Notes
    -----

    In addition to accessing data, this class can create a new directory for the current date (by default). It also
    allows the user to create custom names for the parent directory. This function should be run at the start of
    each workday.

    :param bool has_directory_been_created: If false, then will test to find if the named parent directory (and its
                                            children) has already been created. If any dirs in the tree are missing
                                            then they will be created.
    :param bool has_custom_name: Set if a custom name (meaning, not today's date) should be used for the parent dir.

    :return: [0] is the input data directory; this is the read-from location. [1] is the output data directory.
             All plots and GIFs should be saved here"""

    if has_custom_name:
        parent_name = str(input("Enter the name of the parent directory: "))
    else:
        # parent_name = date_of_today()
        parent_name = "24 Mar 22"

    lg.info(f"Target (parent) directory is {parent_name}.")

    if platform == "linux" or platform == "linux2":
        print("Detected Linux. This OS is not yet supported. Exiting")

        exit()

    elif platform == "darwin":
        # OS X
        mac_dir_root = "/Users/cameronmceleney/CLionProjects/Data/"  # Location of my C++ data on Mac

        if not has_directory_been_created:
            create_directory(mac_dir_root, parent_name)

        input_data_directory = f"{mac_dir_root}{parent_name}/Simulation_Data/"
        output_data_directory = f"{mac_dir_root}{parent_name}/Outputs/"
        logging_directory = f"{mac_dir_root}{parent_name}/Logs/"

        return input_data_directory, output_data_directory, logging_directory

    elif platform == "win32" or platform == "win64":
        # Windows
        windows_dir_root = "D:\\Data\\"  # Location of my C++ data on Windows

        if not has_directory_been_created:
            create_directory(windows_dir_root, parent_name)

        input_data_directory = f"{windows_dir_root}{parent_name}\\Simulation_Data\\"
        output_data_directory = f"{windows_dir_root}{parent_name}\\Outputs\\"
        logging_directory = f"{windows_dir_root}{parent_name}\\Logs\\"

        return input_data_directory, output_data_directory, logging_directory


def date_of_today():
    """
    Finds and returns today's date in DD MMM YY format.

    For details on how to change the date format, see the official `documentation <https://strftime.org/>`_.

    :return: Today's date.
    """

    date = datetime.today().strftime("%d %b %y")

    return date


def create_directory(root_dir_path, parent_dir_name, should_show_errors=False):
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
        # Create list of paths to all subdirectories to be created. Done separately as future functions are to be added
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


def logging_setup(logging_directory_path):
    """Initialisation of basic logging information."""
    today_date = datetime.now().strftime("%y%m%d")
    current_time = datetime.now().strftime("%H%M")

    lg.basicConfig(filename=f'{logging_directory_path}/{today_date}-{current_time}.log',
                   filemode='w',
                   level=lg.INFO,
                   format='%(asctime)s | %(module)s::%(funcName)s | %(levelname)s | %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   force=True)
