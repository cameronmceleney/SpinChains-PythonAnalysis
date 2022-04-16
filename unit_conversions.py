# -*- coding: utf-8 -*-

# -------------------------- Preprocessing Directives -------------------------

# Standard Libraries
import logging as lg
# import os as os
from sys import exit

# 3rd Party packages
from datetime import datetime
import re as re

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

    @staticmethod
    def _current_cgs():
        """
        This is a work in progress.

        https://www.quora.com/What-is-the-cgs-unit-of-electric-current
        :return:
        """
        c_cgs = 2.99792458e10  # The speed of light in CGS
        return c_cgs


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
        self.temp_from = None
        self.temp_to = None
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

            if self.units_from != "Wb" and self.units_from in magnetic_flux_units_si:
                self.units_from = "Wb"

            if self.units_to != "Wb" and self.units_to in magnetic_flux_units_si:
                self.units_to = "Wb"

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


class UnitDecomposition:

    def __init__(self, input_string):
        self.unfiltered_string = input_string
        self.filtered_string = None

    def _split_string(self, selected_string=None):
        if selected_string is None:
            selected_string = self.unfiltered_string

        input_string = selected_string.strip()  # Remove leading and trailing whitespace
        use_delimiter = " "
        list_of_inputs = input_string.split(use_delimiter)
        return list_of_inputs

    @staticmethod
    def _check_for_reserved_chars(string_to_test):
        """Ensures that special characters required for mathematics are not present"""
        reserved_chars = {'#', '*', '/', '^', '_'}
        reserved_chars_found = []

        try:
            for char in reserved_chars:
                if char in string_to_test:
                    reserved_chars_found.append(char)
        except ValueError as ve:
            print(f"ValueError: {ve} caused the issue")
            raise
        else:
            if reserved_chars_found:
                raise ValueError(f"Reserved characters {reserved_chars_found} were found. Please edit your expression "
                                 f"to omit these.")

    def _check_if_equation(self, string_to_test=None):
        """Tests if a string is an equation, and breaks it down into components"""
        if string_to_test is None:
            string_to_test = self.unfiltered_string

        if '=' in string_to_test:
            return True
        else:
            return False

    def generate_output(self):

        self.filtered_string = self._split_string()
        if self._check_if_equation():
            self._handle_equations()
        else:
            self._handle_unit_conversion()

    def _handle_equations(self):

        # Separate equation into LHS and RHS
        lhs_equation, rhs_equation = [equations.split(',') for equations in ','.join(self.filtered_string).split('=')]

        # Remove trailing whitespace after separating equation
        lhs_equation = list(filter(None, lhs_equation))
        rhs_equation = list(filter(None, rhs_equation))

        # Tokenise equations to extract individual terms
        lhs_equation = self._tokenise_expression(lhs_equation)
        rhs_equation = self._tokenise_expression(rhs_equation)

        # Generate list (of same length as original) that contains only the prefixes. None indicates either no prefix,
        # or a symbol is present
        lhs_prefixes = self._find_prefixes(lhs_equation)
        rhs_prefixes = self._find_prefixes(rhs_equation)

        # Generate list that contains equation without prefixes. Handle lists separately incase they're different
        # lengths (zip would stop when the shorter list is completed)
        lhs_equation_stripped = []
        for key, value in enumerate(lhs_equation):
            lhs_equation_stripped.append(value.replace(str(lhs_prefixes[key]), ''))

        rhs_equation_stripped = []
        for key, value in enumerate(rhs_equation):
            rhs_equation_stripped.append(value.replace(str(rhs_prefixes[key]), ''))

        # Substitute in the fundamental units to the stripped equation
        lhs_equation_stripped = self._fundamental_units(lhs_equation_stripped)
        rhs_equation_stripped = self._fundamental_units(rhs_equation_stripped)

        # Separate unit from their powers (to allow for combination later)
        lhs_equation_separated_powers = self._find_powers_of_units(lhs_equation_stripped)
        rhs_equation_separated_powers = self._find_powers_of_units(rhs_equation_stripped)

        # Combine powers of the same base unit on each side of equality, then sort powers into order.
        lhs_combined_powers = self._combine_powers_of_units(lhs_equation_separated_powers)
        rhs_combined_powers = self._combine_powers_of_units(rhs_equation_separated_powers)

        # Output final units to user
        lhs_units_output = self._equation_output(lhs_combined_powers, lhs_equation, show_output=True)
        rhs_units_output = self._equation_output(rhs_combined_powers, rhs_equation, show_output=True)

        if lhs_units_output == rhs_units_output:
            print("The terms are the same!")
        else:
            print("The terms are not the same!")

    def _handle_unit_conversion(self):
        return self._find_prefixes(self.filtered_string)

    @staticmethod
    def _find_prefixes(string_to_check):

        si_prefixes = {'Y': [1e24, "Yotta"], 'Z': [1e21, "Zetta"], 'E': [1e18, "Exa"], 'P': [1e15, "Peta"],
                       'T': [1e12, "Tera"], 'G': [1e9, "Giga"], 'M': [1e6, "Mega"], 'k': [1e3, "kilo"],
                       'h': [1e2, "hecto"], 'da': [1e1, "deka"], 'd': [1e-1, "deci"],
                       'c': [1e-2, "centi"], 'm': [1e-3, "milli"], 'u': [1e-6, "mico"], 'n': [1e-9, "nano"],
                       'p': [1e-12, "pico"], 'f': [1e-15, "femto"], 'a': [1e-18, "atto"], 'z': [1e-21, "zepto"],
                       'y': [1e-24, "yocto"]}

        prefixes_found = []
        for term in string_to_check:
            # Check if each term in the expression has any prefixes
            if len(term) <= 1:
                # Ignored individual symbols as they have no prefix
                prefixes_found.append(None)
                continue

            component_symbols = list(term)

            if len(component_symbols) > 2 and component_symbols[0] + component_symbols[1] == "da":
                # Special case; only two letter prefix
                prefixes_found.append("da")
                continue

            if component_symbols[0] in si_prefixes.keys():
                prefixes_found.append(component_symbols[0])
                continue

        return prefixes_found

    @staticmethod
    def _tokenise_expression(string_to_tokenise):
        """
        Tokenizes the expression.

        Link to a resource to explain how to tokenize using Regular Expressions (RegEx) in Python:
        https://stackoverflow.com/questions/43389684/how-can-i-split-a-string-of-a-mathematical-expressions-in-python>.
        """
        if isinstance(string_to_tokenise, list):
            string_to_tokenise = ''.join(string_to_tokenise)

        tokens = []
        tokenizer = re.compile(r"\s*([()+*/-]|\w+)")
        current_pos = 0
        while current_pos < len(string_to_tokenise):
            match = tokenizer.match(string_to_tokenise, current_pos)
            if match is None:
                raise SyntaxError
            tokens.append(match.group(1))
            current_pos = match.end()

        return tokens

    @staticmethod
    def _fundamental_units(list_to_evaluate, print_si=False):
        """
        Converts a symbol to its fundamental unit equivalent.

        Contains a huge list of all physics expressions, and how they are written in fundamental units. For ref. only
        """
        """
        These are the S.I. base units. To avoid symbol mis-identification, base_units dict should never be included in 
        expr_dicts. This is because reserved internal symbols (base_units['A']) are a commonly used symbol (A = area).
        
        A   : electric current              (ampere)        
        cd  : candela                       (luminous intensity) 
        K   : Kelvin                        (temperature)
        kg  : kilogram                      (mass)
        m   : metre                         (length)
        mol : mole                          (amount of substance)
        s   : second                        (time)
        """
        base_units = {'A': "A{1}",
                      'cd': "cd{1}",
                      'K': "K{1}",
                      'kg': "kg{1}",
                      'm': "m{1}",
                      'mol': "mol{1}",
                      's': "s{1}"}

        if print_si:
            for key, value in enumerate(base_units):
                print(f"{key}: {value}")

        # Contain minimal Base Units, and crop up in many other expressions.
        """
        a   : acceleration                  (m{1}*s^{-2})
        A   : area                          (m{2})
        f   : frequency                     (s^{-1})
        I   : current                       (A{1})
        t   : time                          (s^{1})
        v   : velocity                      (m{1}*s^{-1})
        V   : volume                        (m{3})
        """
        core_expr = {'a': base_units['m'] + "*s{-2}",
                     'A': "m{2}",
                     'f': "s{-1}",
                     'I': base_units['A'],
                     'm': base_units['m'],
                     't': base_units['s'],
                     'v': base_units['m'] + "*s{-1}"}

        # Commonly used expressions
        """
        F   : Force                         (Newtons)
        R   : Electrical Resistance         (Ohms)
        V   : Voltage                       (Volt)
        """
        reg_expr = {'F': "kg{1}*" + core_expr['a'],
                    'R': "kg{1}*m{2}*s{−3}*A{−2}",
                    'V': "kg{1}*m{2}*s{−3}*A{−1}"}

        # Rarely used expressions. Complex, compound expressions should go here
        """
        """
        rare_expr = {}

        # All dicts to search for a symbol match. LHS
        expr_dicts = [core_expr, reg_expr, rare_expr]
        list_of_base_units = []

        for _, term in enumerate(list_to_evaluate):
            # Extract each term from the given equation's symbol list
            for dict_to_search in expr_dicts:
                if term in dict_to_search.keys():
                    # If true, substitute the list's symbol for its base unit equivalent
                    list_of_base_units.append(dict_to_search[term])
                    break  # No need to keep searching after a match is found, so can move to next iteration of FOR loop

        return list_of_base_units

    @staticmethod
    def _find_powers_of_units(list_to_separate_powers):

        if isinstance(list_to_separate_powers, list):
            # Convert to string to then use re package.
            list_to_separate_powers = ''.join(list_to_separate_powers)

        # Remove {}* chars which should be the only non-digit, non-alpha chars in the string
        list_to_separate_powers = list(filter(None, re.split('\s*[{}*]+', list_to_separate_powers)))

        # Each even-numbered position (incl. zero) is a letter, and each odd-numbered position is a number. Need to
        # combine into a set of nested lists.

        list_to_return = []
        for key in range(0, len(list_to_separate_powers)):
            list_to_return.append(list_to_separate_powers[key])

        return list_to_return

    @staticmethod
    def _combine_powers_of_units(list_to_combine):

        si_base_units = {'m': 0, 's': 0, "Mole": 0, 'A': 0, 'K': 0, "cd": 0, "kg": 0}

        for _, base_unit in enumerate(si_base_units):
            temp_total = 0
            for x in range(0, len(list_to_combine), 2):
                if str(list_to_combine[x]) == base_unit:
                    element = list_to_combine[x + 1]
                    sign = 1
                    # There are two different unicode hyphens that need to be handled, else a ValueError occurs. Their
                    # codes are '\u8722' and '\u0045'.
                    if element.startswith('−') or element.startswith('-'):
                        sign = -1
                        element = int(element[1:])
                    else:
                        element = int(element)
                    temp_total = temp_total + sign * element
            si_base_units[base_unit] = temp_total

        return si_base_units

    @staticmethod
    def _equation_output(dict_to_output, equation, print_all_components=False, show_output=True):
        """
        Description.

        https://stackoverflow.com/questions/8519599/python-dictionary-to-string-custom-format
        """
        if print_all_components:
            pass
        else:
            for k in list(dict_to_output.keys()):
                if dict_to_output[k] == 0:
                    del dict_to_output[k]

        dict_as_string = ' + '.join([f'{key}^{ {value} }' for key, value in dict_to_output.items()])

        if show_output:
            print(f"{' '.join(equation)} = {dict_as_string}")

        return dict_as_string


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

    # initial_length = input("Enter conversion in format ITOT0000: ")
    # uc = UnitConversion(initial_length)
    # MagneticFluxTest(initial_length).compute(True)

    user_string = "m = I * A"  # input("Enter expression: ")
    UnitDecomposition(user_string).generate_output()
    # test example: dF = um + GT - daH

    # lg.info(f"{PROGRAM_NAME} end")
    exit()


# ------------------------------ Implementations ------------------------------

if __name__ == '__main__':
    # logging_setup()

    main()
