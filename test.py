#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard libraries
import logging
import matplotlib.pyplot as plt
import numpy as np
from sys import exit

"""
Completes all lab tasks from ENG4099 Lab 1. Written in a slightly funny odd
way in order to match how the problems/code is presented in the labscript
"""

# Declare constants using PEP-9 format
JOULE_TO_EV = 6.242e+18
MASS_E = 9.11e-31  # in atomic units
M_TO_NM = 1e-9
PLANCK_H_BAR = 1.055e-34  # in atomic units
PLANCK_H = 2.0 * np.pi * PLANCK_H_BAR
SPEED_OF_LIGHT = 2.998e+17  # nm/s

BOX_LENGTH = 0.1 * M_TO_NM
x = np.linspace(0.01 * BOX_LENGTH, 1.0 * BOX_LENGTH, 50)
x_nm = x / M_TO_NM
t = np.linspace(0.0, 1.0, 50)

arr_En = [0]
f1 = np.zeros(50)
f2 = np.zeros(50)
f3 = np.zeros(50)


def ComputeEnergyOfLevel(n):
    # Compute energy of electron at an energy level 'n'
    return n ** 2 * (PLANCK_H_BAR ** 2 * np.pi ** 2 / (2.0 * MASS_E * BOX_LENGTH ** 2))


def ComputeWavelength(upperLevel, lowerLevel):
    return PLANCK_H * SPEED_OF_LIGHT / (arr_En[upperLevel] - arr_En[lowerLevel])


def loggingSetup():
    logging.basicConfig(filename='logfile.log',
                        filemode='w',
                        level=logging.DEBUG,
                        format='%(asctime)s - %(message)s',
                        datefmt='%d/%m/%Y %I:%M:%S %p',
                        force=True)
    logging.getLogger('matplotlib.font_manager').disabled = True


def Superposition(w_mn, evalAtTime):
    return (f1 ** 2 + f2 ** 2) / 2 + np.cos(w_mn * evalAtTime) * f1 * np.conj(f2)


def Wavefunction(n, x):
    return np.sqrt(2.0 / BOX_LENGTH) * np.sin(n * np.pi * x / BOX_LENGTH)


def PlotEnergyLevels():
    plt.figure(figsize=(4, 6))

    plt.plot(x_nm, arr_En[1] * (x / x) * JOULE_TO_EV)
    plt.plot(x_nm, arr_En[2] * (x / x) * JOULE_TO_EV)
    plt.plot(x_nm, arr_En[3] * (x / x) * JOULE_TO_EV)
    plt.plot(x_nm, arr_En[4] * (x / x) * JOULE_TO_EV)
    plt.plot(x_nm, arr_En[5] * (x / x) * JOULE_TO_EV)

    plt.legend(['n = 1', 'n = 2', 'n = 3', 'n = 4', 'n = 5'], bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlabel("$x (nm)$", fontsize=14)
    plt.ylabel("$Energy (eV)$", fontsize=14)

    # plt.savefig('En.png')
    plt.show()


def PlotSuperposition():
    # Using the line below leads to slight precision loss for some reason; hence the hardcoding
    # w21 = (arr_En[2] - arr_En[1]) / PLANCK_H_BAR
    w21 = (2.4116611281059233e-17 - 6.029152820264808e-18) / PLANCK_H_BAR

    plt.plot(x_nm, Superposition(w21, 0))
    plt.plot(x_nm, Superposition(w21, np.pi / 2))
    plt.plot(x_nm, Superposition(w21, np.pi / 4))

    plt.legend(['t = 0', 't = $\pi$/2', 't = $\pi/4$'])
    plt.xlabel("$x (nm)$", fontsize=14)
    plt.ylabel("$\Psi(x,t)$", fontsize=14)
    plt.show()


# def animationSuperposition():
#     filenames = []
#     for it in range (100):
#         t=it*0.01
#         plt.plot(x/1e-9,phi(t))
#         plt.xlim(0, L/1e-9)
#         plt.ylim(0, 3.5e10)
#         plt.xlabel("$x (nm)$", fontsize=14)
#         plt.ylabel("$\Psi(x,t)$", fontsize=14)
#         plt.text(0.2,0.15, r'$t=$ {0:10.3f} [au]'.format(t),vfontsize=12)
#         filename='_tmp_'+str(it).zfill(5)+'.png'
#         filenames.append(filename)

def PlotWavefunction():
    print(f"sqrt(2/L) = {np.sqrt(2.0 / BOX_LENGTH)}")
    print(f"2/L = {2.0 / BOX_LENGTH}")

    fig, ax = plt.subplots(3, 2, figsize=(6, 5))

    ax[0, 0].plot(x_nm, f1, 'r')  # row=0, col=0
    ax[0, 0].set_xlabel("$x (nm)$", fontsize=14)
    ax[0, 0].set_ylabel("$\psi_1(x)$", fontsize=14)

    ax[1, 0].plot(x_nm, f2, 'b')  # row=1, col=0
    ax[1, 0].set_xlabel("$x (nm)$", fontsize=14)
    ax[1, 0].set_ylabel("$\psi_2(x)$", fontsize=14)

    ax[2, 0].plot(x_nm, f3, 'g')  # row=2, col=0
    ax[2, 0].set_xlabel("$x (nm)$", fontsize=14)
    ax[2, 0].set_ylabel("$\psi_3(x)$", fontsize=14)

    ax[0, 1].plot(x_nm, f1 * np.conj(f1), 'r')  # row=0, col=1
    ax[0, 1].set_xlabel("$x (nm)$", fontsize=14)
    ax[0, 1].set_ylabel("$|\psi_1(x)|^2$", fontsize=14)

    ax[1, 1].plot(x_nm, f2 * np.conj(f2), 'b')  # row=1, col=1
    ax[1, 1].set_xlabel("$x (nm)$", fontsize=14)
    ax[1, 1].set_ylabel("$|\psi_2(x)|^2$", fontsize=14)

    ax[2, 1].plot(x_nm, f3 * np.conj(f3), 'g')  # row=2, col=1
    ax[2, 1].set_xlabel("$x (nm)$", fontsize=14)
    ax[2, 1].set_ylabel("$|\psi_3(x)|^2$", fontsize=14)
    fig.tight_layout()
    plt.show()


def main():
    logging.info("Program start")

    for levelNumber in range(1, 6):
        #
        arr_En.append(ComputeEnergyOfLevel(levelNumber))
        print(f"E({levelNumber})= {arr_En[levelNumber]} J or {arr_En[levelNumber] * JOULE_TO_EV} eV")

    print(f"The wavelength of a photon emitted for the transition from n = 3 to n = 2 is {ComputeWavelength(3, 2)} nm")
    print(f"The wavelength of a photon emitted for the transition from n = 3 to n = 1 is {ComputeWavelength(3, 1)} nm")
    print(f"The wavelength of a photon emitted for the transition from n = 2 to n = 1 is {ComputeWavelength(2, 1)} nm")

    for i in range(50):
        f1[i] = Wavefunction(1, x[i])
        f2[i] = Wavefunction(2, x[i])
        f3[i] = Wavefunction(3, x[i])

    # PlotEnergyLevels()

    # PlotWavefunction()

    PlotSuperposition()

    logging.info("Program end")
    exit()


if __name__ == '__main__':
    loggingSetup()

    main()







