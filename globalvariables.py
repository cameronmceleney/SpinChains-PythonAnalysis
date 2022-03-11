#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard libraries
from sys import platform, exit
from datetime import datetime
import os as os
import errno as errno

# 3rd Party Packages
# Add here

# My packages / Any header files
# Here

"""
    Description of what globalvariables does
"""

"""
    Core Details
    
    Author      : cameronmceleney
    Created on  : 04/03/2022 15:52
    Filename    : globalvariables
    IDE         : PyCharm
"""


def generate_dir_tree(check_dir_exist=False, set_custom_name=False):
    """Detect the user's operating system in order to set file_paths

    Text

    :param bool set_custom_name: Enable ability to create a custom name for the parent directory
    :return: None."""

    if set_custom_name:
        parent_name = custom_name()
    else:
        parent_name = todays_date()

    if platform == "linux" or platform == "linux2":
        print("Detected Linux. This OS is not yet supported. Exiting")

        exit()

    elif platform == "darwin":
        # OS X
        mac_dir_root = "/Users/cameronmceleney/CLionProjects/Data/"  # Location of my C++ data on Mac

        if check_dir_exist:
            create_directory(mac_dir_root, parent_name)

        input_data_directory = f"{mac_dir_root}{parent_name}/RK2 Shockwaves Tests Data/"
        output_data_directory = f"{mac_dir_root}{parent_name}/RK2 Shockwaves Tests Outputs/"

        return input_data_directory, output_data_directory

    elif platform == "win32" or platform == "win64":
        # Windows
        windows_dir_root = "D:\\Data\\"  # Location of my C++ data on Windows

        if check_dir_exist:
            create_directory(windows_dir_root, parent_name)

        input_data_directory = f"{windows_dir_root}{parent_name}\\RK2 Shockwaves Tests Data\\"
        output_data_directory = f"{windows_dir_root}{parent_name}\\RK2 Shockwaves Tests Outputs\\"

        return input_data_directory, output_data_directory


def custom_name():
    """
    Take input from the user

    :return: A user-defined string"""

    users_custom_name = str(input("Enter the name of the parent directory: "))

    return users_custom_name


def todays_date():
    """
    Finds and returns today's date in DD MMM YY format.

    For details on how to change the date format, see the official `documentation <https://strftime.org/>`_.

    :return: Today's date.
    """

    date = datetime.today().strftime("%d %b %y")

    return date


def create_directory(root_dir_path, parent_dir_name, show_errors=False):
    """
    Create a tree of subdirectories to save simulation data.

    The following notes should be observed:

    * The Python code should be run **before** the C++ simulations to ensure all needed directories exist.

    :param str root_dir_path: Absolute path to location where parent will be created.
    :param str parent_dir_name: Name of parent directory.
    :param bool show_errors: Set (True) to show when a directory couldn't be created.

    :return: Nothing.
    """

    # Default behaviour is to use today's date as the dir name; matching the behaviour of C++ code
    parent_dir_path = os.path.join(root_dir_path, parent_dir_name)

    # Create set (to avoid duplicates) of all the directories to be made under the parent_dir
    sub_directories = {"RK2 Shockwaves Tests Data", "RK2 Shockwaves Tests Outputs"}
    path_list = []

    for i, val in enumerate(sub_directories):
        # Create list of paths to all subdirectories to be created. Done separately as future functions are to be added
        path_list.append(os.path.join(parent_dir_path, val))

    for i, val in enumerate(sub_directories):
        try:
            # Try to create each subdirectory (and parent if needed). Always show instances of dirs being created
            os.makedirs(path_list[i], exist_ok=False)
            print(f"Directory '{val}' created successfully")

        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

            # Handles exceptions from raise
            if show_errors:
                # User-selected flag in function argument
                print(f"Directory '{val}' already exists.")
                pass
