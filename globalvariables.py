#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard libraries
from sys import platform, exit

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
current_date_mac = "09 Mar 22"
current_date_windows = "03 Mar 22"


def set_file_paths():
    # Detect the user's operating system in order to set file_paths

    if platform == "linux" or platform == "linux2":
        # linux
        print("Detected Linux. Exiting")

        exit()

    elif platform == "darwin":
        # OS X
        mac_dir_root = "/Users/cameronmceleney/CLionProjects/Data/"

        input_data_directory = f"{mac_dir_root}{current_date_mac}/RK2 Shockwaves Tests Data/"
        output_data_directory = f"{mac_dir_root}{current_date_mac}/RK2 Shockwaves Tests Outputs/"

        return input_data_directory, output_data_directory

    elif platform == "win32":
        # Windows
        windows_dir_root = "D:\\Data\\"

        input_data_directory = f"{windows_dir_root}{current_date_windows}\\RK2 Shockwaves Tests Data\\"
        output_data_directory = f"{windows_dir_root}{current_date_windows}\\RK2 Shockwaves Tests Outputs\\"

        return input_data_directory, output_data_directory
