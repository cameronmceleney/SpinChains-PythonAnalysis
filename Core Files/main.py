#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard libraries
import logging as lg

# My packages / Any header files
import system_preparation as sp
import data_analysis as das
import string

"""
    This file acts as the main function for the software. To ensure encapsulation is adhered to, all data_analysis is 
    performed is invoked from a separate file, to enable simulation programs to be written and invoked from this main 
    file. 
    
    The GitHub token for this project is :https://***REMOVED***@github.com/cameronmceleney
    /SpinChains.git
"""
PROGRAM_NAME = "SpinChains-Python-Analysis/Core Files/main.py"
"""
    Core Details
    
    Author      : cameronmceleney
    Created on  : 06/03/2022 22:08
    Filename    : main
    IDE         : PyCharm
"""

if __name__ == '__main__':

    """
    All functions should be initialised here (excluding core operating features like logging).
    """
    lg.info(f"Program start...")

    _should_use_eigens = False
    _mass_produce = False
    filename_base = "1406"  # str(input("Enter the unique identifier of the file: "))

    def generate_filenames():
        suffix = 'a'  # Start with 'a' initially

        while True:
            filename = filename_base + suffix

            # Function logic here - careful to reimport the correct filenames!
            if not _should_use_eigens:
                dataset1 = das.PlotImportedData(filename, system_setup.input_dir(), system_setup.output_dir(),
                                                file_prefix="rk2", file_component='mx', file_identifier="T")
                dataset1.call_methods(override_method="pf", override_function="se", override_site=100, early_exit=True)

            # Increment suffix naturally
            suffix = increment_suffix(suffix)

            # Set `aaa` as an arb. endpoints for now
            if suffix == 'aaa':
                break

    def increment_suffix(suffix):
        alphabet = string.ascii_lowercase

        # Convert suffix to a number
        num = 0
        for i, char in enumerate(reversed(suffix)):
            num += (alphabet.index(char) + 1) * (26 ** i)

        # Increment the number
        num += 1

        # Convert the number back to suffix
        result = ''
        while num > 0:
            num, remainder = divmod(num - 1, 26)
            result = alphabet[remainder] + result

        return result

    system_setup = sp.SystemSetup()
    system_setup.detect_os(True, "2024-01-26", "2024-01-26")

    if not _should_use_eigens:
        if _mass_produce:
            generate_filenames()
        else:
            dataset1 = das.PlotImportedData(filename_base, system_setup.input_dir(), system_setup.output_dir(),
                                            file_prefix="rk2", file_component='mx', file_identifier="T")
            dataset1.call_methods(override_method="pf", override_function="gif", override_site=100, early_exit=True)
    elif _should_use_eigens:
        dataset2 = das.PlotEigenmodes(filename_base, system_setup.input_dir(), system_setup.output_dir())
        dataset2.import_eigenmodes()
        dataset2.plot_eigenmodes()  # only use this line if raw files don't need imported or converted

    exit(0)

"""
Notes
    
For the paper, linearFMR = (2 * np.pi * 28.3e9 / (2 * np.pi)) * np.sqrt(172e-6 * (172e-6 + 4 * np.pi * 0.086)) / 1e9
"""
