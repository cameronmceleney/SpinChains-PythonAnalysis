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
    Description of what rk2test does
"""

"""
    Core Details
    
    Author      : cameronmceleney
    Created on  : 18/05/2022 11:35
    Filename    : rk2test
    IDE         : PyCharm
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs 16 Sep 2021 15:06 

@author: cameronmceleney

This code is a working version of the RK4 code. It includes:
    - includes a nonvarying exchange integral
    - plots the norm of M values
        - (14:49 13 Sep 21) No longer plots the M values by default, although the functionality to do so
        has been left in place. Also plots using subplots2grid instead of addsubplots
    - doesn't compute particularly quickly
    - has no reporting functionality, or csv writer
        - (18:24 09 Sep 21) has reporting functionality but this says "complete" at 99% instead
        of 100%. Will need to fix this in the future but it is servicable for now

THIS IS THE NEWEST VERSION OF THE CODE
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
import time as tm
from matplotlib.animation import FuncAnimation
import csv as csv

##############################################################################
# Initial Conditions
stepsize = 1e-15
halfstep = stepsize / 2.0
number_of_dpoints = 2e1
f = 42.5 * 1E9  # in [Hz]
numspins = 1
max_iterations = 1e2

# sets the frequency (Hz). w=angular frequency. gamma = gyromagnetic ratio (GHz/T)
w = 2.0 * np.pi * f

gamma = 29.2E9 * 2.0 * np.pi  # [GHz/T]
alpha = 1E-4  # Gilbert damping factor
H0 = 0.1  # H0 = bias field [T]
Ms = 1.0  # Saturation Magnetisation [T]

# Takes their units from above declarations. Otherwise, put values in T
mx0 = 0.0
my0 = 0.0
mz0 = Ms

exc_min = 43.5
exc_max = 43.5

filename = "/Users/cameronmceleney/CLionProjects/Data/2022-05-18/testcase.csv"

# initial time
total_time = 0.0
largest_mx_val = 1.0

##############################################################################
# creates arrays for the x,y,z components of the Magnetisation and effective H-field vectors. The mag components
# have an extra column at the start and end for M_p-1 and M_N, which must always be zero
mxStart = (np.full((1, numspins + 2), mx0)).flatten()
myStart = (np.full((1, numspins + 2), my0)).flatten()
mzStart = (np.full((1, numspins + 2), mz0)).flatten()

bias_field_driving = 3E-3

driving_region_lhs = 0
driving_region_rhs = 100
##############################################################################
# Dealing with Jt. This property is independent of what is going on within the spinchain
num_spinpairs = numspins - 1
exc_vals = np.linspace(exc_min, exc_max, num_spinpairs)
exc_vals = (np.insert(exc_vals, [0, num_spinpairs], [0, 0])).flatten()

##############################################################################
# Loop
results = mxStart

for iteration in range(0, int(max_iterations) + 1):

    total_time += stepsize
    t0 = total_time
    t0half = total_time + halfstep
    ########################################
    mx_est_mid = (np.full((1, numspins + 2), 0)).flatten()
    my_est_mid = (np.full((1, numspins + 2), 0)).flatten()
    mz_est_mid = (np.full((1, numspins + 2), 0)).flatten()
    mx_next_val = (np.full((1, numspins + 2), 0)).flatten()
    my_next_val = (np.full((1, numspins + 2), 0)).flatten()
    mz_next_val = (np.full((1, numspins + 2), 0)).flatten()

    ############# RK2 Step 1 ###############
    for spin in range(1, numspins+1):
        # RK2 Step 1
        spinLHS, spinRHS = spin - 1, spin + 1

        mx1, mx1LHS, mx1RHS = mxStart[spin], mxStart[spinLHS], mxStart[spinRHS]
        my1, my1LHS, my1RHS = myStart[spin], myStart[spinLHS], myStart[spinRHS]
        mz1, mz1LHS, mz1RHS = mzStart[spin], mzStart[spinLHS], mzStart[spinRHS]

        heffXK1 = exc_vals[spinLHS] * mx1LHS + exc_vals[spin] * mx1RHS + bias_field_driving * np.cos(w * t0)
        heffYK1 = exc_vals[spinLHS] * my1LHS + exc_vals[spin] * my1RHS
        heffZK1 = exc_vals[spinLHS] * mz1LHS + exc_vals[spin] * mz1RHS + H0

        # mxK1 = gamma * (- (alpha * heffYK1 * mx1 * my1) + heffYK1 * mz1 - heffZK1 * (my1 + alpha * mx1 * mz1) + alpha
        #                 * heffXK1 * my1 ** 2 + mz1 ** 2)
        # myK1 = gamma * (-(heffXK1 * mz1) + heffZK1 * (mx1 - alpha * my1 * mz1) + alpha * (heffYK1 * mx1 ** 2 - heffXK1
        #                                                                                   * mx1 * my1 + heffYK1 * mz1 ** 2))
        # mzK1 = gamma * (heffXK1 * my1 + alpha * heffZK1 * (mx1 ** 2 + my1 ** 2) - alpha * heffXK1 * mx1 * mz1 - heffYK1
        #                 * (mx1 + alpha * my1 * mz1))

        mxK1 = -1.0 * gamma * (my1 * heffZK1 - mz1 * heffYK1)
        myK1 = +1.0 * gamma * (mx1 * heffZK1 - mz1 * heffXK1)
        mzK1 = -1.0 * gamma * (mx1 * heffYK1 - my1 * heffXK1)

        mx_est_mid[spin] = mx1 + mxK1 * halfstep
        my_est_mid[spin] = my1 + myK1 * halfstep
        mz_est_mid[spin] = mz1 + mzK1 * halfstep

        if iteration % 10 == 0:
            print(mxK1, myK1, mzK1)
            print(mx_est_mid, my_est_mid, mz_est_mid)

    ############# RK2 Step 2 ###############
    for spin in range(1, numspins + 1):
        # RK2 Step 2
        spin_lhs, spin_rhs = spin - 1, spin + 1

        mx2, mx2LHS, mx2RHS = mx_est_mid[spin], mx_est_mid[spin_lhs], mx_est_mid[spin_rhs]
        my2, my2LHS, my2RHS = my_est_mid[spin], my_est_mid[spin_lhs], my_est_mid[spin_rhs]
        mz2, mz2LHS, mz2RHS = mz_est_mid[spin], mz_est_mid[spin_lhs], mz_est_mid[spin_rhs]

        heffXK2 = exc_vals[spin_lhs] * mx2LHS + exc_vals[spin] * mx2RHS + bias_field_driving * np.cos(w * t0half)
        heffYK2 = exc_vals[spin_lhs] * my2LHS + exc_vals[spin] * my2RHS
        heffZK2 = exc_vals[spin_lhs] * mz2LHS + exc_vals[spin] * mz2RHS + H0

        # mxK2 = gamma * (- (alpha * heffYK2 * mx2 * my2) + heffYK2 * mz2 - heffZK2 * (my2 + alpha * mx2 * mz2) + alpha
        #                 * heffXK2 * (my2 ** 2 + mz2 ** 2))
        # myK2 = gamma * (-(heffXK2 * mz2) + heffZK2 * (mx2 - alpha * my2 * mz2) + alpha * (heffYK2 * mx2 ** 2 - heffXK2
        #                 * mx2 * my2 + heffYK2 * mz2 ** 2))
        # mzK2 = gamma * (heffXK2 * my2 + alpha * heffZK2 * (mx2 ** 2 + my2 ** 2) - alpha * heffXK2 * mx2 * mz2 - heffYK2
        #                 * (mx2 + alpha * my2 * mz2))
        mxK2 = -1 * gamma * (my2 * heffZK2 - mz2 * heffYK2)
        myK2 = gamma * (mx2 * heffZK2 - mz2 * heffXK2)
        mzK2 = -1 * gamma * (mx2 * heffYK2 - my2 * heffXK2)

        mx_next_val[spin] = mxStart[spin] + mxK2 * stepsize
        my_next_val[spin] = myStart[spin] + myK2 * stepsize
        mz_next_val[spin] = mzStart[spin] + mzK2 * stepsize

        if mx_next_val[spin] + my_next_val[spin] + mz_next_val[spin] > largest_mx_val:
            largest_mx_val = mx_next_val[spin] + my_next_val[spin] + mz_next_val[spin]
    ########################################
    if iteration % (max_iterations / number_of_dpoints) == 0:
        results = np.vstack([results, mx_next_val])

    mxStart = mx_next_val
    myStart = my_next_val
    mzStart = mz_next_val

np.savetxt(filename, results, fmt='%.8f')

########################################################################################################################

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