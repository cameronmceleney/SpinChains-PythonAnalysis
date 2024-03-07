#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Full packages
import logging as log
import string

# Specific functions from packages


# My full modules
import system_preparation as sp
import data_analysis as das

# Specific functions from my modules


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
    log.info(f"Program start...")

    _should_use_eigens = False
    _mass_produce = False
    _has_numeric_suffix = True
    filename_base = "0150"  # str(input("Enter the unique identifier of the file: "))

    system_setup = sp.SystemSetup()
    system_setup.detect_os(False, "2024-03-06", "2024-03-06")

    def generate_filenames():
        if _has_numeric_suffix:
            suffix = 1
        else:
            suffix = 'a'

        while True:
            if _has_numeric_suffix:
                filename = filename_base + '_' + str(suffix)
            else:
                filename = filename_base + suffix

            # Function logic here - careful to reimport the correct filenames!
            dataset_mass = das.PlotImportedData(filename, system_setup.input_dir(), system_setup.output_dir(),
                                                file_prefix="rk2", file_component='mx', file_identifier="T",)
            dataset_mass.call_methods(override_method="pf", override_function="hd", override_site=100, early_exit=True,
                                      mass_produce=_mass_produce)

            suffix = increment_suffix(suffix, _has_numeric_suffix)

            # Set `aaa` as an arb. endpoints for now
            if suffix == 'aaa' or suffix == 1000:
                break

    def increment_suffix(suffix, has_numeric_suffix):
        if has_numeric_suffix:
            return suffix + 1

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

    if not _should_use_eigens:
        if _mass_produce:
            generate_filenames()
        else:
            dataset1 = das.AnalyseData()
            dataset1.import_data(file_descriptor=filename_base, input_dir_path=system_setup.input_dir(),
                                 output_dir_path=system_setup.output_dir(), file_prefix="rk2", file_component='mx',
                                 file_identifier="T", auto_run=True)
            dataset1.process_data()
            dataset1.call_methods(override_method="pf", override_function="sfft", override_site=20, early_exit=True,
                                  loop_function=True, mass_produce=False)
            exit(0)
    elif _should_use_eigens:
        dataset2 = das.PlotEigenmodes(filename_base, system_setup.input_dir(), system_setup.output_dir())
        dataset2.import_eigenmodes()
        dataset2.plot_eigenmodes()  # only use this line if raw files don't need imported or converted

    exit(0)

"""
Notes
    
For the paper, linearFMR = (2 * np.pi * 28.3e9 / (2 * np.pi)) * np.sqrt(172e-6 * (172e-6 + 4 * np.pi * 0.086)) / 1e9
"""
