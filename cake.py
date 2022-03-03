# -*- coding: utf-8 -*-

# -------------------------- Preprocessing Directives -------------------------

# Standard Libraries
import logging
import matplotlib.pyplot as plt
import numpy as np
import os as os
from sys import exit

# 3rd Party packages
# Add here

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

def loggingSetup():
    """
    Minimum Working Example (MWE) for logging. Pre-defined levels are:
        
        Highest               ---->            Lowest
        CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    """
    logging.basicConfig(filename='logfile.log',
                        filemode='w',
                        level=logging.DEBUG,
                        format='%(asctime)s - %(messages)s',
                        datefmt='%d/%m/%Y %I:%M:%S %p',
                        force=True)


# --------------------------- main() implementation ---------------------------

def main():
    logging.info(f"{PROGRAM_NAME} start")

    # Enter code here

    logging.info(f"{PROGRAM_NAME} end")
    exit()


# ------------------------------ Implementations ------------------------------

if __name__ == '__main__':
    loggingSetup()

    main()
