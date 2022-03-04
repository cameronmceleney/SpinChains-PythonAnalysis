#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard libraries
import logging as lg
import matplotlib.pyplot as plt
import numpy as np
import os as os
import sys as sys

# 3rd Party Packages
# Add here

# My packages / Any header files
# Here

"""
    Description of what globalvariables does
"""

"""
    Core Details
    
    Author      : cameronmceleney
    Created on  : 04/03/2022 15:52
    Filename    : globalvariables
    IDE         : PyCharm
"""


def logging_setup():
    # Initialisation of basic logging information. 
    lg.basicConfig(filename='logfile.log',
                   filemode='w',
                   level=lg.DEBUG,
                   format='%(asctime)s - %(message)s',
                   datefmt='%d/%m/%Y %I:%M:%S %p',
                   force=True)


def set_file_paths():
    # Detect the user's operating system in order to set file_paths
    operating_system = os.name()

    return operating_system

