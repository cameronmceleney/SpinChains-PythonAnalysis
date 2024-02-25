# -*- coding: utf-8 -*-

# -------------------------- Preprocessing Directives -------------------------

# Full packages
import csv
import logging as log
import numpy as np

# Specific functions from packages

# My full modules

# Specific functions from my modules
from attribute_mappings_legacy import AttributeMappings

# ----------------------------- Program Information ----------------------------

"""
Description of what foo.py does
"""
PROGRAM_NAME = "foo.py"
"""
Created on (date) by (author)
"""


def import_headers_from_file(self):
    log.info("Importing file headers...")

    sim_flags = AttributeMappings.dict_with_none(AttributeMappings.sim_flags)
    key_params = AttributeMappings.dict_with_none(AttributeMappings.key_data)

    with open(self.input_data_path) as file_header_data:
        csv_reader = csv.reader(file_header_data)
        next(csv_reader)  # 0th.
        next(csv_reader)  # 1st. Always blank.

        data_flags_titles = next(csv_reader)  # 2nd. Boolean titles which might also include values (older)
        data_flags_values = next(csv_reader)  # 3rd. Might include Boolean values (newer)

        # Count is now in terms of (newer); (older) doesn't have this additional blank line

        if data_flags_values:
            next(csv_reader)  # 4th line. Blank.
        key_sim_param_titles = next(csv_reader)  # 5th. Titles for each key simulation parameter.
        key_sim_param_values = next(csv_reader)  # 6th. Values for each key simulation parameter.
        next(csv_reader)  # 7th. Blank.
        next(csv_reader)  # 8th. Simulation notes.
        next(csv_reader)  # 9th. Description of how to read tabular data
        next(csv_reader)  # 10th. Blank.
        simulated_sites = next(csv_reader)  # 11th line. Titular value (site number) for each simulated site.

    # Iterate over each title in the CSV
    for title, value in zip(key_sim_param_titles, key_sim_param_values):
        for desired_var_name, (_, possible_spellings) in AttributeMappings.key_data.items():
            mapped_title = self.custom_string_mappings(title)
            if mapped_title in possible_spellings:
                key_params[desired_var_name] = self._custom_cast_mappings(value)
                break

    if data_flags_values:
        for title, value in zip(data_flags_titles, data_flags_values):
            for desired_var_name, (_, possible_spellings) in AttributeMappings.sim_flags.items():
                if title in possible_spellings:
                    if len(value) > 1:
                        mapped_value = self.custom_string_mappings(value, False)
                    else:
                        mapped_value = bool(value)
                    sim_flags[desired_var_name] = mapped_value
                    break
    else:
        for i, entry in enumerate(data_flags_titles):
            if i % 2 == 0:
                title, value = data_flags_titles[i], data_flags_titles[i + 1]
                for desired_var_name, (_, possible_spellings) in AttributeMappings.sim_flags.items():
                    if title in possible_spellings:
                        if len(value) > 1:
                            sim_flags[desired_var_name] = self.custom_string_mappings(value, False)
                        else:
                            sim_flags[desired_var_name] = bool(int(value))
                        break
            else:
                continue

    """
    for i, val in enumerate(data_flags_titles):
        if i % 2 == 0:
            mapped_title = self.custom_string_mappings(data_flags_titles[i], False)
            if len(val) > 1:
                mapped_value = self.custom_string_mappings(data_flags_titles[i+1], False)
            else:
                mapped_value = bool(data_flags_titles[i+1])
            sim_flags[mapped_title] = mapped_value
        else:
            continue

    for title, value in zip(key_sim_param_titles, key_sim_param_values):
        mapped_title = self.custom_string_mappings(title)
        mapped_value = self._custom_cast_mappings(value)
        key_params[mapped_title] = mapped_value
    """
    # Cleanup
    if sim_flags['numericalMethodUsed'] is None:
        sim_flags['numericalMethodUsed'] = f"{self.fp.upper()} Method"

    if "Time [s]" in simulated_sites:
        simulated_sites.remove("Time [s]")

    print(key_params)
    print(sim_flags)

    return key_params, simulated_sites, sim_flags


def import_headers_from_file_legacy(fi, input_data_path, fp):
    """
    Import the header lines of each csv file to obtain the C++ simulation parameters.

    THIS IS LEGACY CODE FOR ALL DATA PRODUCED BEFORE 2024-02-20. Each simulation in C++ returns all the key
    parameters, required to replicate the simulation, as headers in csv files. This function imports that data,
    and creates dictionaries to store it.

    The Python dictionary keys are the same variable names as their C++ counterparts (for consistency). Casting is
    required as data comes from csvreader as strings.

    :return: Returns a tuple. [0] is the dictionary containing all the key simulation parameters. [1] is an array
    containing strings; the names of each spin site.
    """
    log.info(f"Importing file headers...")

    if fi == "LLGTest":
        with open(input_data_path) as file_header_data:
            csv_reader = csv.reader(file_header_data)
            next(csv_reader)  # 0th line.
            next(csv_reader)  # 1st line. Blank.
            next(csv_reader)  # 2nd line. Column title for each key simulation parameter. data_names
            data_values = next(csv_reader)  # 3rd line. Values associated with column titles from 4th line.
            next(csv_reader)  # 4th line. Blank.
            next(csv_reader)  # 5th line. Simulation notes.
            next(csv_reader)  # 6th line. Describes how to understand tabular titles.
            next(csv_reader)  # 7th line. Blank.
            list_of_simulated_sites = next(csv_reader)  # 8th line. Array of simulated site numbers.

            data_flags = None
    else:
        with open(input_data_path) as file_header_data:
            csv_reader = csv.reader(file_header_data)
            next(csv_reader)  # 0th line.
            next(csv_reader)  # 1st line. Blank.
            data_flags = next(
                csv_reader)  # 2nd line. Booleans to indicate which modules were used during simulations.
            next(csv_reader)  # 3rd line. Blank.
            data_value_names = next(
                csv_reader)  # 4th line. Column title for each key simulation parameter. data_names
            data_values = next(csv_reader)  # 5th line. Values associated with column titles from 4th line.
            next(csv_reader)  # 6th line. Blank.
            next(csv_reader)  # 7th line. Simulation notes.
            next(csv_reader)  # 8th line. Describes how to understand tabular titles.
            next(csv_reader)  # 9th line. Blank.
            list_of_simulated_sites = next(csv_reader)  # 10th line. Array of simulated site numbers.

    sim_flags = dict()
    if data_flags is not None:
        # Assignment to dict is done individually to improve readability.
        sim_flags['isLLGUsed'] = str(data_flags[1])
        sim_flags['isShockwaveUsed'] = str(data_flags[3])
        sim_flags['isDriveOnLHS'] = str(data_flags[5])
        sim_flags['methodName'] = str(data_flags[7])
        sim_flags['isDriveStatic'] = str(data_flags[9])

        if 'numericalMethodUsed' not in sim_flags.keys():
            sim_flags['numericalMethodUsed'] = f"{fp.upper()} Method"
        else:
            sim_flags['numericalMethodUsed'] = str(data_flags[7])
    elif 'numericalMethodUsed' not in sim_flags.keys():
        sim_flags['numericalMethodUsed'] = f"{fp.upper()} Method"
    else:
        sim_flags['numericalMethodUsed'] = f"{fp.upper()} Method"

    key_params = dict()

    if fi == "LLGTest":
        key_params['staticBiasField'] = float(data_values[0])
        key_params['dynamicBiasField1'] = float(data_values[1])
        key_params['dynamicBiasFieldScaling'] = float(data_values[2])
        key_params['drivingFreq'] = float(data_values[3])
        key_params['drivingRegionLHS'] = int(data_values[4])
        key_params['drivingRegionRHS'] = int(data_values[5])
        key_params['drivingRegionWidth'] = int(data_values[6])
        key_params['maxSimTime'] = float(data_values[7])
        key_params['exchangeMaxVal'] = float(data_values[8])
        key_params['stopIterVal'] = float(data_values[9])
        key_params['exchangeMinVal'] = float(data_values[10])
        key_params['numberOfDataPoints'] = int(data_values[11])
        key_params['totalSpins'] = int(data_values[12])
        key_params['stepsize'] = float(data_values[13])
        key_params['dampedSpins'] = int(data_values[14])
        key_params['gilbertFactor'] = 1e-5  # float(data_values[15])
        sim_flags['isLLGUsed'] = 1  # int(data_values[6])

        key_params['dynamicBiasField2'] = key_params['dynamicBiasField1'] * key_params['dynamicBiasFieldScaling']
        key_params['chainSpins'] = round(key_params['totalSpins'], -3)
        key_params['dampedSpins'] = key_params['totalSpins'] - key_params['chainSpins']
        key_params['gyroMagRatio'] = 2.92E9 * 2 * np.pi

    else:
        key_params['staticBiasField'] = float(data_values[0])
        key_params['dynamicBiasField1'] = float(data_values[1])
        key_params['dynamicBiasFieldScaling'] = float(data_values[2])
        key_params['dynamicBiasField2'] = float(data_values[3])
        key_params['drivingFreq'] = float(data_values[4])
        key_params['drivingRegionLHS'] = int(data_values[5])
        key_params['drivingRegionRHS'] = int(data_values[6])
        key_params['drivingRegionWidth'] = int(data_values[7])
        key_params['maxSimTime'] = float(data_values[8])
        key_params['exchangeMinVal'] = float(data_values[9])
        key_params['exchangeMaxVal'] = float(data_values[10])
        key_params['stopIterVal'] = float(data_values[11])
        key_params['numberOfDataPoints'] = int(data_values[12])
        key_params['chainSpins'] = int(data_values[13])
        key_params['dampedSpins'] = int(data_values[14])
        key_params['totalSpins'] = int(data_values[15])
        key_params['stepsize'] = float(data_values[16])
        key_params['gilbertFactor'] = float(data_values[17])
        key_params['gyroMagRatio'] = float(data_values[18])
        key_params['shockGradientTime'] = float(data_values[19])
        key_params['shockApplyTime'] = float(data_values[20])

    log.info(f"File headers imported!")

    if "Time [s]" in list_of_simulated_sites:
        list_of_simulated_sites.remove("Time [s]")

    return key_params, list_of_simulated_sites, sim_flags


def _apply_custom_casting_to_value(data_value):
    data_value = data_value.strip()
    if '.' in data_value or 'e' in data_value.lower():
        return float(data_value)
    else:
        return int(data_value)
