# -*- coding: utf-8 -*-

# -------------------------- Preprocessing Directives -------------------------

# Standard Libraries
import logging as lg
# import os as os
from sys import exit

# 3rd Party packages
from datetime import datetime

# import matplotlib.pyplot as plt
# import numpy as np

# My packages/Header files
# Here

# ----------------------------- Program Information ----------------------------

"""
Description of what foo.py does
"""
PROGRAM_NAME = "foo.py"
"""
Created on (date) by (author)
"""


# ---------------------------- Function Declarations ---------------------------
class UnitConversion:
    """
    Will convert INPUT to OUTPUT based upon the selected CONDITIONS.

    :param input_string: Contains all the information from the user required to produce the output.
    :type input_string: str

    :param cgs_to_si: Boolean value, defaults to False.
    :type cgs_to_si: bool
    """

    def __init__(self, input_string, cgs_to_si=False):
        self.unfiltered_string = input_string
        self.cgs_to_si = cgs_to_si

        self.conversion_string = self._split_string()
        self.units_from = self.conversion_string[0]
        self.units_to = self.conversion_string[1]
        self.input_value = self.conversion_string[2]

    def _split_string(self):
        """https://stackoverflow.com/questions/430079/how-to-split-strings-into-text-and-number"""
        input_string = self.unfiltered_string.strip()  # Remove leading and trailing whitespace

        # Identify which delimiter is to be used. Add addition ELIF statement for any further options
        use_delimiter = " "
        if '-' in input_string:
            use_delimiter = "-"

        # Separate string into components
        split_input = input_string.split(use_delimiter)
        if len(split_input) > 2:
            head_units_from = split_input[0]
            head_units_to = split_input[1]
            tail = split_input[2]
        else:
            head_units_from = "Default"
            head_units_to = split_input[0]
            tail = split_input[1]

        # Order of components must stay the same for use throughout code
        try:
            str_head_from = str(head_units_from)
            str_head_to = str(head_units_to)
            float_tail = float(tail)
        except ValueError as ve:
            print(f"ValueError: {ve}.")
            exit(1)
        else:
            return str_head_from, str_head_to, float_tail

    def length(self, text_output=False):
        """
        Converts measurements in length from CGS to S.I. units (or the reverse).

        Further description.

        :return: Converted value
        :rtype: list of floats
        """

        dict_of_lengths = {"Pm": 1e15, "TM": 1e12, "Gm": 1e9, "Mm": 1e6, "km": 1e3, "hm": 1e2, "dam": 1e1,
                           "m": 1.0,
                           "dm": 1e-1, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "nm": 1e-9, "pm": 1e-12, "fm": 1e-15}

        convert_from = "None"
        output_value = None
        try:
            if self.units_from == "DEFAULT":
                if self.cgs_to_si:
                    convert_from = "cm"
                    output_value = self.input_value * dict_of_lengths[convert_from] / dict_of_lengths[self.units_to]
                elif self.cgs_to_si is False:
                    convert_from = "m"
                    output_value = self.input_value * dict_of_lengths[convert_from] / dict_of_lengths[self.units_to]
            else:
                convert_from = self.units_from
                output_value = self.input_value * dict_of_lengths[convert_from] / dict_of_lengths[self.units_to]

        except KeyError as ke:
            print(f"[{ke}] is not a valid length. Remember that the code is case-sensitive.")
            exit(1)

        else:
            if text_output:
                print(f"The mass {self.input_value}[{convert_from}] is {output_value:.5f}[{self.units_to}]")
            else:
                return self.input_value, convert_from, output_value, self.units_to

    def mass(self, text_output=False):
        """
        Converts masses from CGS to S.I. units (or the reverse).

        Further description.

        :return: Converted value
        :rtype: list of floats
        """
        dict_of_masses = {"tonne": 1e6,
                          "kg": 1e3,
                          "g": 1.0, "mg": 1e-3, "ug": 1e-6}

        convert_from = "None"
        output_value = None
        try:
            if self.units_from == "DEFAULT":
                if self.cgs_to_si:
                    convert_from = "g"
                    output_value = self.input_value * dict_of_masses[convert_from] / dict_of_masses[self.units_to]
                elif self.cgs_to_si is False:
                    convert_from = "kg"
                    output_value = self.input_value * dict_of_masses[convert_from] / dict_of_masses[self.units_to]
            else:
                convert_from = self.units_from
                output_value = self.input_value * dict_of_masses[convert_from] / dict_of_masses[self.units_to]

        except KeyError as ke:
            print(f"[{ke}] is not a valid mass. Remember that the code is case-sensitive.")
            exit(1)

        else:
            if text_output:
                print(f"The mass {self.input_value}[{convert_from}] is {output_value:.5f}[{self.units_to}]")
            else:
                return self.input_value, convert_from, output_value, self.units_to

    def time(self, text_output=False):
        """
        Converts measurements in length from CGS to S.I. units (or the reverse).

        Description

        :return: Converted value
        :rtype: list of floats
        """

        dict_of_times = {"D": 86400, "H": 3600, "M": 60,
                         "s": 1.0,
                         "jiffy": 0.01666, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12, "fs": 1e-15}

        convert_from = "None"
        output_value = None
        try:
            if self.units_from == "DEFAULT":
                if self.cgs_to_si:
                    convert_from = "cm"
                    output_value = self.input_value * dict_of_times[convert_from] / dict_of_times[self.units_to]
                elif self.cgs_to_si is False:
                    convert_from = "m"
                    output_value = self.input_value * dict_of_times[convert_from] / dict_of_times[self.units_to]
            else:
                convert_from = self.units_from
                output_value = self.input_value * dict_of_times[convert_from] / dict_of_times[self.units_to]

        except KeyError as ke:
            print(f"[{ke}] is not a valid time. Remember that the code is case-sensitive.")
            exit(1)

        else:
            if text_output:
                print(f"The time {self.input_value}[{convert_from}] is {output_value:.5f}[{self.units_to}]")
            else:
                return self.input_value, convert_from, output_value, self.units_to

    def current(self, text_output=False):
        """
        Converts measurements in electric current from CGS to S.I. units (or the reverse).

        `<link https://www.npl.co.uk/si-units/ampere>`

        :return: Converted value
        :rtype: list of floats
        """

        # 1 ampere = 1.602176634e-19 / 1.602176634 = 9.99e-20; I = Q / t
        dict_of_si_currents = {"MA": 1e9, "GA": 1e6, "kA": 1e3, "hA": 1e2, "daA": 1e1,
                               "A": 1.0,
                               "dA": 1e-1, "cA": 1e-2, "mA": 1e-3, "uA": 1e-6, "nA": 1e-9, "pA": 1e-12, "fA": 1e-15}

        convert_from = "None"
        output_value = None
        try:
            if self.units_from == "DEFAULT":
                if self.cgs_to_si:
                    self._current_cgs()
                elif self.cgs_to_si is False:
                    convert_from = "A"
                    output_value = self.input_value * dict_of_si_currents[convert_from] / dict_of_si_currents[
                        self.units_to]
            else:
                # Can only use default SI units here at the moment; can't mix SI and CGS
                convert_from = self.units_from
                output_value = self.input_value * dict_of_si_currents[convert_from] / dict_of_si_currents[self.units_to]

        except KeyError as ke:
            print(f"[{ke}] is not a valid length. Remember that the code is case-sensitive.")
            exit(1)

        else:
            if text_output:
                print(f"The time {self.input_value}[{convert_from.lower()}] is {output_value:.5f}[{self.units_to}]")
            else:
                return self.input_value, convert_from.lower(), output_value, self.units_to

    def _current_cgs(self):
        """
        This is a work in progress.

        https://www.quora.com/What-is-the-cgs-unit-of-electric-current
        :return:
        """
        c_cgs = 2.99792458e10  # The speed of light in CGS
        return 5


class MagneticFluxDensity(UnitConversion):

    def __init__(self, input_string, cgs_to_si=False):
        super().__init__(input_string, cgs_to_si)

    def compute(self, text_output=False):
        """
        Converts measurements regarding magnetism (in general) from CGS to S.I. units (or the reverse).

        :return: Converted value
        :rtype: list of floats
        """
        # B-field should be in terms of [G], but some people may use [Oe]. In air, 1[G] = 1[Oe]. No support for out
        # with air right now (B[G] = u_r H[Oe]; u_r is the relative permeability).

        try:
            if self.units_from == "DEFAULT":
                if self.cgs_to_si:
                    self._gauss(text_output)
                elif self.cgs_to_si is False:
                    self._si(text_output)
            else:
                if {self.units_to, self.units_from}.issubset({"G", "Oe", "Y"}):
                    return print(f"These units are directly equivalent: "
                                 f"{self.input_value}[{self.units_to}] = {self.input_value}[{self.units_from}]")

                if self.units_to in {"G", "Oe"} or self.units_from in {"G", "Oe"}:
                    self._gauss(text_output)
                elif self.units_to == 'Y' or self.units_from == 'Y':
                    self._gamma(text_output)
                else:
                    self._si(text_output)

        except KeyError as ke:
            print(f"Magnetic Flux Density: [{ke}] is not a valid input. Remember that the code is case-sensitive.")
            raise ke

        else:
            return

    def _gauss(self, text_output):
        """Converts between CGS and SI when Gauss [G] are involved."""
        convert_from = None
        output_value = None

        if self.units_from == "G":
            convert_from = "G"
            gauss_to_tesla = 1e-4 * self.input_value  # Move from [G] to [T], and then put through method
            output_value = self._si(internal_conversion=["T", gauss_to_tesla])

        elif self.units_to == "G":
            convert_from = self.units_from
            input_in_tesla = self._si(internal_conversion=[convert_from, self.input_value])
            output_value = input_in_tesla * 1e4

        if text_output:
            print(f"Magnetic flux density (CGS): {self.input_value}[{convert_from}] is "
                  f"{output_value:.5f}[{self.units_to}]")
        else:
            return self.input_value, convert_from, output_value, self.units_to

    def _gamma(self, text_output):
        """Converts between CGS and SI when Gamma [Y] are involved."""
        convert_from = None
        output_value = None

        if self.units_from == "Y":
            convert_from = "Y"
            gamma_to_tesla = 1e-9 * self.input_value  # Move from [G] to [T], and then put through method
            output_value = self._si(internal_conversion=["T", gamma_to_tesla])

        elif self.units_to == "Y":
            convert_from = self.units_from
            input_in_tesla = self._si(internal_conversion=[convert_from, self.input_value])
            output_value = input_in_tesla * 1e9

        if text_output:
            print(f"Magnetic flux density (CGS): {self.input_value}[{convert_from}] is "
                  f"{output_value:.5f}[{self.units_to}]")
        else:
            return self.input_value, convert_from, output_value, self.units_to

    def _si(self, text_output=False, internal_conversion=None):
        """Converts between two different magnetic flux inputs that are both in SI units"""

        dict_si_magnetic_flux = {"PT": 1e15, "TT": 1e12, "GT": 1e9, "MT": 1e6, "kT": 1e3, "hT": 1e2, "daT": 1e1,
                                 "T": 1.0,
                                 "dT": 1e-1, "cT": 1e-2, "mT": 1e-3, "uT": 1e-6, "nT": 1e-9, "pT": 1e-12, "fT": 1e-15}
        if internal_conversion is None:
            output_value = self.input_value * dict_si_magnetic_flux[self.units_from] / \
                           dict_si_magnetic_flux[self.units_to]

            if text_output:
                print(f"Magnetic flux density (SI-SI): {self.input_value}[{self.units_from}] is {output_value:.5f}"
                      f"[{self.units_to}]")
            else:
                return self.input_value, self.units_from, output_value, self.units_to
        else:
            if self.units_to in ["G", "Y"]:
                output_value = internal_conversion[1] * dict_si_magnetic_flux[internal_conversion[0]] / \
                               dict_si_magnetic_flux["T"]
            else:
                output_value = internal_conversion[1] * dict_si_magnetic_flux[internal_conversion[0]] / \
                               dict_si_magnetic_flux[self.units_to]

            return output_value


class MagneticFlux(UnitConversion):

    def __init__(self, input_string, cgs_to_si=False):
        super().__init__(input_string, cgs_to_si)

    def compute(self, text_output=False):
        """
        Converts measurements regarding magnetism (in general) from CGS to S.I. units (or the reverse).

        :return: Converted value
        :rtype: list of floats
        """
        # There are many different (equivalent) SI units for Magnetic Flux. https://en.wikipedia.org/wiki/Weber_(unit)
        magnetic_flux_units_si = {"Wb", "Ohm*C", "V*s", "H*A", "T*m^2", "J/A", "N*m/A"}

        try:
            if {self.units_from, self.units_to}.issubset(magnetic_flux_units_si):
                # Case where in/out units are equivalent, so there is no need to convert
                if text_output:
                    return print(f"Magnetic flux (equivalent SI): {self.input_value}[{self.units_from}] "
                                 f"= {self.input_value}[{self.units_to}]")
                else:
                    return self.input_value, self.units_from, self.input_value, self.units_to

            if self.units_from == "Mx" or self.units_to == "Mx":
                self._maxwell(text_output)
            else:
                self._si(text_output)
        except KeyError as ke:
            print(f"Magnetic Flux: [{ke}] is not a valid input. Remember that the code is case-sensitive.")
            raise ke

        else:
            return

    def _maxwell(self, text_output):
        """Converts between CGS and SI when Gauss [G] are involved."""
        convert_from = None
        output_value = None

        if self.units_from == "Mx":
            convert_from = "Mx"
            maxwell_to_tesla = 1e-8 * self.input_value  # Move from [Mx] to [Wb], and then put through method
            output_value = self._si(internal_conversion=["Wb", maxwell_to_tesla])

        elif self.units_to == "Mx":
            convert_from = self.units_from
            input_in_tesla = self._si(internal_conversion=[convert_from, self.input_value])
            output_value = input_in_tesla * 1e8

        if text_output:
            print(f"Magnetic flux (CGS): {self.input_value}[{convert_from}] is {output_value:2.2e}[{self.units_to}]")
        else:
            return self.input_value, convert_from, output_value, self.units_to

    def _si(self, text_output=False, internal_conversion=None):
        """Converts between two different magnetic flux inputs that are both in SI units"""

        magnetic_flux_prefixes = {"PWb": 1e15, "TWb": 1e12, "GWb": 1e9, "MWb": 1e6, "kWb": 1e3, "hWb": 1e2, "daWb": 1e1,
                                  "Wb": 1.0,
                                  "dWb": 1e-1, "cWb": 1e-2, "mWb": 1e-3, "uWb": 1e-6, "nWb": 1e-9, "pWb": 1e-12,
                                  "fWb": 1e-15}

        if internal_conversion is None:
            output_value = self.input_value * magnetic_flux_prefixes[self.units_from] / \
                           magnetic_flux_prefixes[self.units_to]

            if text_output:
                print(f"Magnetic flux (SI-SI): {self.input_value}[{self.units_from}] is {output_value:.5f}"
                      f"[{self.units_to}]")
            else:
                return self.input_value, self.units_from, output_value, self.units_to
        else:
            if self.units_to in "Mx":
                output_value = internal_conversion[1] * magnetic_flux_prefixes[internal_conversion[0]] / \
                               magnetic_flux_prefixes["Wb"]
            else:
                output_value = internal_conversion[1] * magnetic_flux_prefixes[internal_conversion[0]] / \
                               magnetic_flux_prefixes[self.units_to]

            return output_value


# ---------------------------- Function Declarations ---------------------------
def logging_setup():
    """
    Minimum Working Example (MWE) for logging. Pre-defined levels are:
        
        Highest               ---->            Lowest
        CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    """
    today_date = datetime.now().strftime("%y%m%d")
    current_time = datetime.now().strftime("%H%M")

    lg.basicConfig(filename=f'./{today_date}-{current_time}.log',
                   filemode='w',
                   level=lg.INFO,
                   format='%(asctime)s | %(module)s::%(funcName)s | %(levelname)s | %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   force=True)


# --------------------------- main() implementation ---------------------------

def main():
    # lg.info(f"{PROGRAM_NAME} start")

    initial_length = input("Enter conversion in format ITOT0000: ")
    uc = UnitConversion(initial_length)
    MagneticFlux(initial_length).compute(True)

    # lg.info(f"{PROGRAM_NAME} end")
    exit()


# ------------------------------ Implementations ------------------------------

if __name__ == '__main__':
    # logging_setup()

    main()
