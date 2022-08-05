# -*- coding: utf-8 -*-

# -------------------------- Preprocessing Directives -------------------------

# Standard Libraries
import logging as lg
# import os as os
from sys import exit

# 3rd Party packages
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

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
    lg.info(f"{PROGRAM_NAME} start")

    dataset1 = np.loadtxt("D:\\Data\\2022-08-05\\Simulation_Data\\rk2_mx_T1807.csv",
                          delimiter=",", skiprows=11)
    time = dataset1[:, 0]
    data_to_plot=dataset1[:, 3000]

    fig = plt.figure()
    plt.plot(time, abs(data_to_plot), label="50 ns")
    plt.xlabel("Time [ns]")
    plt.ylabel("abs(m$_x$/M$_S$)")
    plt.legend()
    plt.show()
    lg.info(f"{PROGRAM_NAME} end")
    exit()


# ------------------------------ Implementations ------------------------------

if __name__ == '__main__':
    loggingSetup()

    main()
