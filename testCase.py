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
    Description of what testCase does
"""

"""
    Core Details
    
    Author      : cameronmceleney
    Created on  : 18/05/2022 14:33
    Filename    : testCase
    IDE         : PyCharm
"""


def RK2Computation():
    static_field = 0.1  # [T]
    applied_field = 3E-3  # [T]
    number_of_spins = 100

    exchange_min = 43.5  # [T]
    exchange_max = 132.0  # [T]
    freq = 42.5 * 1e9  # [Hz]
    ang_freq = 2.0 * np.pi * freq
    gamma = 29.2E9 * (2 * np.pi)  # [Hz / T]

    driving_region_lhs = 1
    driving_region_width = 6
    driving_region_rhs = driving_region_lhs + driving_region_width

    total_time = 0
    stepsize = 1e-15
    halfstep = 1e-15 / 2.0
    max_iterations = 7e5
    number_of_dpoints = 100
    number_of_pairs = number_of_spins - 1

    ########################################################################################################################
    mx_init = 0.0
    my_init = 0.0
    mz_init = 1.0

    mx_start = (np.concatenate((([0.0]), np.zeros(number_of_spins), ([0.0])))).flatten()
    my_start = (np.concatenate((([0.0]), np.zeros(number_of_spins), ([0.0])))).flatten()
    mz_start = (np.concatenate((([0.0]), np.ones(number_of_spins), ([0.0])))).flatten()

    exc_vals = (np.concatenate((([0]), np.linspace(exchange_min, exchange_max, number_of_pairs), ([0])))).flatten()

    ########################################################################################################################
    mx_results = []

    for iteration in range(0, int(max_iterations)):
        total_time += stepsize
        t0 = total_time
        t0_halfstep = total_time + halfstep

        mx_est_mid = np.zeros(number_of_spins + 2)
        my_est_mid = np.zeros(number_of_spins + 2)
        mz_est_mid = np.zeros(number_of_spins + 2)

        for spin in range(1, number_of_spins + 1):
            # RK2 Step 1
            spin_lhs, spin_rhs = spin - 1, spin + 1

            mx1, mx1_lhs, mx1_rhs = mx_start[spin], mx_start[spin_lhs], mx_start[spin_rhs]
            my1, my1_lhs, my1_rhs = my_start[spin], my_start[spin_lhs], my_start[spin_rhs]
            mz1, mz1_lhs, mz1_rhs = mz_start[spin], mz_start[spin_lhs], mz_start[spin_rhs]

            if driving_region_lhs <= spin <= driving_region_rhs:
                h_eff_x1 = exc_vals[spin_lhs] * mx1_lhs + exc_vals[spin] * mx1_rhs + applied_field * np.cos(
                    ang_freq * t0)
            else:
                h_eff_x1 = exc_vals[spin_lhs] * mx1_lhs + exc_vals[spin] * mx1_rhs
            h_eff_y1 = exc_vals[spin_lhs] * my1_lhs + exc_vals[spin] * my1_rhs
            h_eff_z1 = exc_vals[spin_lhs] * mz1_lhs + exc_vals[spin] * mz1_rhs + static_field

            mx_k1 = -1.0 * gamma * (my1 * h_eff_z1 - mz1 * h_eff_y1)
            my_k1 = +1.0 * gamma * (mx1 * h_eff_z1 - mz1 * h_eff_x1)
            mz_k1 = -1.0 * gamma * (mx1 * h_eff_y1 - my1 * h_eff_x1)

            mx_est_mid[spin] = mx1 + mx_k1 * halfstep
            my_est_mid[spin] = my1 + my_k1 * halfstep
            mz_est_mid[spin] = mz1 + mz_k1 * halfstep

        mx_final_val = np.zeros(number_of_spins + 2)
        my_final_val = np.zeros(number_of_spins + 2)
        mz_final_val = np.zeros(number_of_spins + 2)

        for spin in range(1, number_of_spins + 1):
            # RK2 Step 1
            spin_lhs, spin_rhs = spin - 1, spin + 1

            mx2, mx2_lhs, mx2_rhs = mx_est_mid[spin], mx_est_mid[spin_lhs], mx_est_mid[spin_rhs]
            my2, my2_lhs, my2_rhs = my_est_mid[spin], my_est_mid[spin_lhs], my_est_mid[spin_rhs]
            mz2, mz2_lhs, mz2_rhs = mz_est_mid[spin], mz_est_mid[spin_lhs], mz_est_mid[spin_rhs]

            if driving_region_lhs <= spin <= driving_region_rhs:
                h_eff_x2 = exc_vals[spin_lhs] * mx2_lhs + exc_vals[spin] * mx2_rhs + applied_field * np.cos(ang_freq
                                                                                                            * t0_halfstep)
            else:
                h_eff_x2 = exc_vals[spin_lhs] * mx2_lhs + exc_vals[spin] * mx2_rhs
            h_eff_y2 = exc_vals[spin_lhs] * my2_lhs + exc_vals[spin] * my2_rhs
            h_eff_z2 = exc_vals[spin_lhs] * mz2_lhs + exc_vals[spin] * mz2_rhs + static_field

            mx_k2 = -1.0 * gamma * (my2 * h_eff_z2 - mz2 * h_eff_y2)
            my_k2 = +1.0 * gamma * (mx2 * h_eff_z2 - mz2 * h_eff_x2)
            mz_k2 = -1.0 * gamma * (mx2 * h_eff_y2 - my2 * h_eff_x2)

            mx_final_val[spin] = mx2 + mx_k2 * stepsize
            my_final_val[spin] = my2 + my_k2 * stepsize
            mz_final_val[spin] = mz2 + mz_k2 * stepsize

        if iteration % (max_iterations / number_of_dpoints) == 0:
            mx_results.append(mx_final_val)

        mx_start = mx_final_val
        my_start = my_final_val
        mz_start = mz_final_val

    mx_output = np.vstack(mx_results)
    filename = "/Users/cameronmceleney/CLionProjects/Data/2022-05-18/testcase.csv"
    np.savetxt(filename, mx_output, fmt='%.8f')

    plt.plot(np.arange(1, number_of_spins + 1), mx_output[-1, 1:number_of_spins + 1])
    plt.show()

def plotting():
    numspins = 4000
    filename = "/Users/cameronmceleney/CLionProjects/Data/2022-05-18/rk2_mx_"+str(numspins)+".csv"
    data = np.loadtxt(filename, delimiter=',')
    time = data[:, 0]
    mx_spin_data = data[:, 1:]

    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, numspins), mx_spin_data[-1, :])
    # ax.plot(time, data)
    ax.set(xlabel="Spin Sites", ylabel="m$_x$ [arb.]")
    # ax.set(xlabel="Time [s]", ylabel="m$_x$")
    plt.show()

# plotting()

y_ = np.linspace(np.sqrt(1e-4), np.sqrt(1.0), 200)
y_ = y_ ** 2
x_ = range(0, 200, 1)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x_, y_)
plt.show()