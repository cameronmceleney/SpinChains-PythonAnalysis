#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Full packages
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

# Specific functions from packages


# My full modules


# Specific functions from my modules


"""
    Need text
"""

"""
    Core Details
    
    Author      : cameronmceleney
    Created on  : 13/03/2022 18:06
    Filename    : plot_eigenmodes.py
    IDE         : PyCharm
"""


class Eigenmodes:
    def __init__(self, mx_data, my_data, eigenvalues_data, filename_ending, input_filepath, output_filepath):

        self.mx_data = mx_data
        self.my_data = my_data
        self.eigenvalues_data = eigenvalues_data
        self.input_filename = filename_ending
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath

    def generalised_fourier_coefficients(self, use_defaults=True, are_eigens_angular_freqs=False):
        """
        Plot coefficients across a range of eigenfrequencies to find frequencies of strong coupling.

        The 'generalised fourier coefficients' indicate the affinity of spins to couple to a particular driving field
        profile. If a non-linear exchange was used, then the rightward and leftward profiles will look different. This
        information can be used to deduce when a system allows for:

            * rightward only propagation.
            * leftward only propagation.
            * propagation in both directions.
            * propagation in neither direction.

        :param bool are_eigens_angular_freqs: Converts eigenvalues from [rad Hz] to [Hz].
        :param bool use_defaults: Use preset parameters to reduce user input, and speed-up running of simulations.

        :return: Single figure plot.
        """
        number_of_spins = self.mx_data[:, 0].size
        early_exit = True

        # use_defaults is a testing flag to speed up the process of running sims.
        if use_defaults:
            step = 5
            lower = 130
            upper = 170
            width_ones = 0.023529411764705882
            width_zeros = 1 - 0.023529411764705882

        else:
            step = int(input("Enter step: "))
            lower = int(input("Enter lower: "))
            upper = int(input("Enter upper: "))
            width_ones = float(input("Enter width of driving region [0, 1]: "))
            width_zeros = 1 - width_ones

        if are_eigens_angular_freqs:
            # Raw data is in units of 2*Pi (angular frequency), so we need to convert back to frequency.
            eigenvalues_angular = np.append([0], self.eigenvalues_data)  # eigenvalues_angular
            eigenvalues = [eigval / (2 * np.pi) for eigval in eigenvalues_angular]
        else:
            # No need for further data processing
            eigenvalues = np.append([0], self.eigenvalues_data)

        x_axis_limits = range(0, number_of_spins, 1)

        # Find widths of each component of the driving regions.
        g_ones = np.ones(int(number_of_spins * width_ones), dtype=int)
        g_zeros = np.zeros(int(number_of_spins * width_zeros), dtype=int)

        # g is the driving field profile along the axis where the drive is applied. My simulations all have the
        # drive along the x-axis, hence the name 'gx'.
        gx_lhs = g_ones + g_zeros
        gx_rhs = g_zeros + g_ones

        fourier_coefficents_lhs = np.empty([])
        fourier_coefficents_rhs = np.empty([])

        for i in range(0, number_of_spins):
            # Select an eigenvector, and take the dot-product to return the coefficient of that particular mode.
            fourier_coefficents_lhs = np.append(fourier_coefficents_lhs, np.dot(gx_lhs, self.mx_data[:, i]))
            fourier_coefficents_rhs = np.append(fourier_coefficents_lhs, np.dot(gx_rhs, self.mx_data[:, i]))

        # Normalise the arrays of coefficients.
        fourier_coefficents_lhs = fourier_coefficents_lhs / np.linalg.norm(fourier_coefficents_lhs)
        fourier_coefficents_rhs = fourier_coefficents_rhs / np.linalg.norm(fourier_coefficents_rhs)

        # Plotting functions. Left here as nothing else will use this functionality.
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.suptitle(r'Overlap Values ($\mathcal{O}_{j}$)'f'for a Uniform System')  # {file_name}
        plt.subplots_adjust(top=0.82)

        # Whichever ax is before the sns.lineplot statements is the one which holds the labels.
        sns.lineplot(x=x_axis_limits, y=np.abs(fourier_coefficents_lhs), lw=3, marker='o', ls='--', label='Left',
                     zorder=1.2)
        sns.lineplot(x=x_axis_limits, y=np.abs(fourier_coefficents_rhs), lw=3, color='r',
                     marker='o', label='Right', zorder=1.1)

        np.savetxt("D:/Data/2023-03-06/Simulation_Data/T1115_Eigens/test.csv",
                   zip(x_axis_limits, np.abs(fourier_coefficents_lhs)))
        if early_exit:
            exit(0)

        # Both y-axes need to match up, so it is clear what eigenmode corresponds to what eigenfrequency.
        ax.set(xlabel=r'Eigenfrequency ( $\frac{\omega_j}{2\pi}$ ) (GHz)', ylabel='Fourier coefficient',
               xlim=[lower, upper], yscale='log', ylim=[1e-4, 1e-2],
               xticks=list(range(lower, upper + 1, step)),
               xticklabels=[float(i) for i in np.round(eigenvalues[lower:upper + 1:step], 3)])

        ax_mode = ax.twiny()  # Create second scale on the upper y-axis of the plot.
        ax_mode.set(xlabel=f'Eigenmode ($A_j$) for m$^x$ components',
                    xlim=ax.get_xlim(),
                    xticks=list(range(int(ax.get_xlim()[0]), int(ax.get_xlim()[1]) + 1, step)))

        ax.legend(loc=1, bbox_to_anchor=(0.975, 0.975),
                  frameon=True, fancybox=True, facecolor='white', edgecolor='white',
                  title='Propagation\n   Direction', fontsize=10)

        ax.grid(visible=True, axis='both', which='both', ls='-', lw=2)

        plt.tight_layout()
        fig.savefig(f"{self.output_filepath}_fourier_coefficents.png", bbox_inches="tight")

    def plot_single_eigenmode(self, eigenmode, has_endpoints=True):
        """
        Plot a single eigenmode with the x- and y-axis magnetic moment components against spin site.

        :param int eigenmode: The eigenmode that is to be plotted.
        :param bool has_endpoints: Allows for fixed nodes to be included on plot. Useful for visualisation purposes.

        :return: Outputs a single figure.

        """
        plt.rcParams.update({'savefig.dpi': 1000, "figure.dpi": 1000})
        print(f'Plotting #{eigenmode}...')
        eigenmode -= 1  # To handle 'off-by-one' error, as first site is at mx_data[0]

        # Select single mode to plot from imported data.
        mx_mode = self.mx_data[:, eigenmode] * -1
        # my_mode = self.my_data[:, eigenmode] * -1

        frequency = self.eigenvalues_data[eigenmode]  # Convert angular (frequency) eigenvalue to frequency [Hz].

        eigenmode += 1  # Return to 'true' count

        # Simulation parameters
        number_of_spins = len(mx_mode)
        driving_width = 0.01

        if has_endpoints:
            # 0-valued reflects the (P-1) and (N+1) end spins that act as fixed nodes for the system.
            mx_mode = np.append(np.append([0], mx_mode), [0])
            # my_mode = np.append(np.append([0], my_mode), [0])
            number_of_spins += 2

        # Generate plot
        fig = plt.figure(figsize=(3.375 * 1.5, 3.375 / 2))
        ax1 = fig.add_subplot(111)

        colour2 = '#5584B9'
        sns.lineplot(x=range(0, len(mx_mode)), y=mx_mode, marker='o', markersize=5,
                     linestyle='', alpha=1, ax=ax1, color=colour2, label='Mx')
        sns.lineplot(x=range(0, len(mx_mode)), y=mx_mode, lw=1.75,
                     linestyle='-', alpha=0.5, ax=ax1, color=colour2)

        ax1.set(xlabel="Distance (nm)", ylabel="Amplitude (arb. units)",
                xlim=(0, number_of_spins))  # ,
        # title = f"Eigenmode #{eigenmode}",
        # xticks=np.arange(0, number_of_spins, np.floor(number_of_spins - 2) / 20))

        ax1.xaxis.set(major_locator=ticker.MultipleLocator(50),
                      minor_locator=ticker.MultipleLocator(10))
        ax1.yaxis.set(major_locator=ticker.MultipleLocator(0.1),
                      minor_locator=ticker.MultipleLocator(0.025))

        ax1.text(0.025, 0.925, "(b)", verticalalignment='center', horizontalalignment='left',
                 transform=ax1.transAxes, fontsize=8)

        # Legend doubles as a legend (showing propagation direction), and the frequency [Hz] of the eigenmode.
        ax1.legend(loc=1, bbox_to_anchor=(0.975, 0.975),
                   frameon=True, fancybox=True, facecolor='white', edgecolor='white',
                   title=f"Frequency (GHz)\n        {frequency: 4.1f}\n     Component",
                   fontsize=8, title_fontsize=8)

        ax1.axvspan(0, number_of_spins * driving_width, color='black', alpha=0.2)

        ax1.grid(visible=True, axis='both', which='both', color='black', ls='--', lw=1, alpha=0.0)

        # ax.set_facecolor('#f4f4f5')
        ax1.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=1.9999)
        ax1.set_axisbelow(False)

        fig.savefig(f"{self.output_filepath}_eigenmode_{eigenmode}.png", bbox_inches="tight")

    def plot_dispersion_relation(self, has_data_file=False):

        fig = plt.figure(figsize=(4.4, 2.0))
        ax = fig.add_subplot(1, 1, 1)

        if has_data_file:
            freqs = np.loadtxt(f"/Users/cameronmceleney/CLionProjects/Data/2023-02-15/Simulation_Data/T1606_Eigens/"
                               f"eigenvalues_formatted_T1606.csv")
            ax.scatter(np.arange(1, len(freqs) + 1, 1), freqs)
            ax.set(xlabel='Mode Number', ylabel="Frequency (GHz)")
            fig.tight_layout()

        else:
            generate_plot = False
            num_datasets = 1

            while generate_plot is False:

                print("----------------------------------------"
                      f"\n\t\t Dataset {num_datasets}", end="\n\n")

                if num_datasets == 1:
                    print("Enter the following parameters ... ")

                exchange_field = float(input("\t\t- exchange stiffness D_b (in T): "))
                external_field = float(input("\t\t- external field H_0 (in T): "))

                gyromag_ratio = float(input("\t\t- gyromagnetic ratio (in GHz / T rad): ")) * 2 * np.pi * 1e9
                lattice_constant = float(input("\t\t- lattice constant (in m): "))
                num_sites = float(input("\t\t- number of sites in the system: "))

                num_sites_array = np.arange(0, num_sites, 1)
                wave_number = (num_sites_array * np.pi) / ((len(num_sites_array) - 1) * lattice_constant)
                frequency = gyromag_ratio * (2 * exchange_field * (1 - np.cos(wave_number * lattice_constant))
                                             + external_field)

                plot_label = input("\nEnter the label for this Dispersion Relation: ")

                ax.plot(wave_number * 1e-9, frequency / 1e12, ls='-', lw=2, label=f"{plot_label}")

                has_another_plot = input("Plot another dataset? [Y/N]: ").upper()

                if has_another_plot == 'Y':
                    num_datasets += 1
                elif has_another_plot == 'N':
                    generate_plot = True

            print("\nGenerating figure...")

            ax.set(xlabel="Wave Number (nm$^{-1}$)", ylabel="Angular Frequency (THz)")
            ax.margins(x=0)
            ax.legend(frameon=False)

            ax.set_axisbelow(False)
            ax.set_facecolor("white")

        fig.savefig(f"{self.output_filepath}_DispersionRelation.png", bbox_inches="tight")

        print("--------------------------------------------------------------------------------")
