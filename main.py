#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard libraries
import logging as lg

# My packages / Any header files
import system_preparation as sp
import data_analysis as das

"""
    This file acts as the main function for the software. To ensure encapsulation is adhered to, all data_analysis is 
    performed is invoked from a separate file, to enable simulation programs to be written and invoked from this main 
    file. 
"""
PROGRAM_NAME = "SpinChains-Python-Analysis main.py"
"""
    Core Details
    
    Author      : cameronmceleney
    Created on  : 06/03/2022 22:08
    Filename    : main
    IDE         : PyCharm
"""


def logging_setup():
    """Initialisation of basic logging information."""
    lg.basicConfig(filename='logfile.log',
                   filemode='w',
                   level=lg.INFO,
                   format='%(asctime)s - %(message)s',
                   datefmt='%d/%m/%Y %I:%M:%S %p',
                   force=True)


def main():
    """All functions should be initialised here (excluding core operating features like logging)."""
    lg.info(f"{PROGRAM_NAME} - Program start...")

    # das.data_analysis(file_prefix="rk2_", file_identifier="500spins", file_descriptor="-nonlin", breaking_paper=True)
    das.data_analysis(file_prefix="rk2_mx_", file_identifier="LLGTest", file_descriptor="2235", breaking_paper=False)
    lg.info(f"{PROGRAM_NAME} - Program end!")

    exit()


if __name__ == '__main__':
    logging_setup()

    sp.directory_tree_testing(has_directory_been_created=False, has_custom_name=False)

    main()

    """
    Notes
    
    For the paper, linearFMR = (2 * np.pi * 28.3e9 / (2 * np.pi)) * np.sqrt(172e-6 * (172e-6 + 4 * np.pi * 0.086)) / 1e9
    """
