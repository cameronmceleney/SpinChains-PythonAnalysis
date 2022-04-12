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
    """Will convert INPUT to OUTPUT based upon the selected exchange."""

    def __init__(self, input_value):
        self.input = input_value

    @staticmethod
    def _split_string(input_string):
        """https://stackoverflow.com/questions/430079/how-to-split-strings-into-text-and-number"""
        head = (input_string.rstrip("0123456789")).upper()
        tail = int(input_string[len(head):])
        return head, tail

    def length(self, cgs_to_si=True):
        """
        Converts measurements in length from CGS to S.I. units (or the reverse).

        Further description.

        :param cgs_to_si: Boolean value, defaults to False.
        :type cgs_to_si: bool

        :return: Converted value
        :rtype: list of floats
        """
        convert_to = self._split_string(self.input)[0]
        value = self._split_string(self.input)[1]
        if cgs_to_si:
            if convert_to == "M":
                return value, value / 100
            elif convert_to == "CM":
                return value, value * 100


# ---------------------------- Function Declarations ---------------------------

def loggingSetup():
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

    initial_length = "M5"
    uc = UnitConversion(initial_length)
    print(f"The length {uc.length()[0]}m is {uc.length()[1]}cm")

    # lg.info(f"{PROGRAM_NAME} end")
    exit()


# ------------------------------ Implementations ------------------------------

if __name__ == '__main__':
    # loggingSetup()

    main()
