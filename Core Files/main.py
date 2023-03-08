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
    """
    All functions should be initialised here (excluding core operating features like logging).

    The GitHub token for this project is :https://***REMOVED***@github.com/cameronmceleney
    /SpinChains.git
    """
    lg.info(f"Program start...")

    _should_use_eigens = True

    system_setup = sp.SystemSetup()
    system_setup.detect_os(False, "2023-03-07", "2023-03-07")

    filename_base = "1217"  # str(input("Enter the unique identifier of the file: "))
    if not _should_use_eigens:
        dataset1 = das.PlotImportedData(filename_base, system_setup.input_dir(), system_setup.output_dir(),
                                        file_prefix="rk2", file_component='mx', file_identifier="T")
        dataset1.call_methods()

    elif _should_use_eigens:
        dataset2 = das.PlotEigenmodes(filename_base, system_setup.input_dir(), system_setup.output_dir())
        dataset2.import_eigenmodes()
        dataset2.plot_eigenmodes()  # only this line if raw files don't need imported or converted

    exit(0)


if __name__ == '__main__':
    main()

    """
    Notes
    
    For the paper, linearFMR = (2 * np.pi * 28.3e9 / (2 * np.pi)) * np.sqrt(172e-6 * (172e-6 + 4 * np.pi * 0.086)) / 1e9
    """
