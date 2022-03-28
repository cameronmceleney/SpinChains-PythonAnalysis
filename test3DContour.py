#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard libraries
import logging as lg
import matplotlib.pyplot as plt
import numpy as np
import os as os
import sys as sys

# 3rd Party Packages
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

# My packages / Any header files
# Here

"""
    Description of what test3DContour does
"""

"""
    Core Details
    
    Author      : cameronmceleney
    Created on  : 28/03/2022 12:53
    Filename    : test3DContour
    IDE         : PyCharm
"""

def compare_plots():
    input_path = "/Users/cameronmceleney/CLionProjects/Data/28 Mar 22/Simulation_Data"
    time = "1421"
    mx_all_data = np.loadtxt(f"{input_path}/rk2_mx_LLGTest{time}.csv", delimiter=",", skiprows=9)
    my_all_data = np.loadtxt(f"{input_path}/rk2_my_LLGTest{time}.csv", delimiter=",", skiprows=9)
    m_time_data = mx_all_data[:, 0]
    mx_m_data = mx_all_data[:, 1:]
    my_m_data = my_all_data[:, 1:]
    spin_site = 400

    plt.title(f"Site: {spin_site}")
    plt.plot(m_time_data * 1e9, mx_m_data[:, spin_site], label=f"Site: {spin_site} mx", zorder=2)
    plt.plot(m_time_data * 1e9, my_m_data[:, spin_site], label=f"Site: {spin_site} my", zorder=1)
    plt.legend()
    plt.show()

def create_contour_Plot():
    input_path = "/Users/cameronmceleney/CLionProjects/Data/28 Mar 22/Simulation_Data"
    time = "1421"
    mx_all_data = np.loadtxt(f"{input_path}/rk2_mx_LLGTest{time}.csv", delimiter=",", skiprows=9)
    my_all_data = np.loadtxt(f"{input_path}/rk2_my_LLGTest{time}.csv", delimiter=",", skiprows=9)
    mz_all_data = np.loadtxt(f"{input_path}/rk2_mz_LLGTest{time}.csv", delimiter=",", skiprows=9)

    m_time_data = mx_all_data[:, 0]
    mx_m_data = mx_all_data[:, 1:]
    my_m_data = my_all_data[:, 1:]
    mz_m_data = my_all_data[:, 1:]

    spin_site = 400

    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    ax.plot3D(mx_m_data[:, spin_site], my_m_data[:, spin_site], mz_m_data[:, spin_site], label=f'Spin Site {spin_site}')
    ax.set_xlabel('m$_x$', fontsize=12)
    ax.set_ylabel('m$_y$', fontsize=12)
    ax.set_zlabel('m$_z$', fontsize=12)
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.legend()
    plt.show()

def test_plot():
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')

    # Creating array points using numpy
    z = np.linspace(0, 15, 1000)
    x = np.sin(z)
    y = np.cos(z)
    ax.plot3D(x, y, z, 'gray')

    plt.show()

def logging_setup():
    # Initialisation of basic logging information. 
    lg.basicConfig(filename='logfile.log',
                   filemode='w',
                   level=lg.DEBUG,
                   format='%(asctime)s - %(message)s',
                   datefmt='%d/%m/%Y %I:%M:%S %p',
                   force=True)


def main():
    lg.info("Program start")

    # test_plot()
    create_contour_Plot()
    # compare_plots()

    lg.info("Program end")

    exit()


if __name__ == '__main__':
    logging_setup()

    main()
