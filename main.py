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


def main():
    """All functions should be initialised here (excluding core operating features like logging)."""
    lg.info(f"Program start...")

    # das.data_analysis(file_prefix="rk2_", file_identifier="500spins", file_descriptor="-nonlin", breaking_paper=True)
    # das.data_analysis(file_prefix="rk2_mx_", file_identifier="LLGTest", file_descriptor=filename_base,
    #                  breaking_paper=False)

    system_setup = sp.SystemSetup()
    system_setup.detect_os(has_custom_name=True)

    filename_base = "2313" # str(input("Enter the unique identifier of the file: "))
    Dataset1 = das.PlotImportedData(filename_base, system_setup.input_dir(), system_setup.output_dir(),
                                    file_component='mx')
    Dataset1.call_methods()

    exit()


if __name__ == '__main__':

    main()

    """
    Notes
    
    For the paper, linearFMR = (2 * np.pi * 28.3e9 / (2 * np.pi)) * np.sqrt(172e-6 * (172e-6 + 4 * np.pi * 0.086)) / 1e9
    """
