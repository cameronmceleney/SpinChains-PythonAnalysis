#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard libraries
import csv as csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# My packages / Any header files
import system_preparation as sp

"""
    Description of what Shockwave Site Comparison does
"""

"""
    Core Details
    
    Author      : cameronmceleney
    Created on  : 13/03/2022 18:06
    Filename    : Shockwave Site Comparison
    IDE         : PyCharm
"""


def import_data_headers(filename):
    """
    Import the header lines of each csv file to obtain the C++ simulation parameters.

    Each simulation in C++ returns all the key parameters, required to replicate the simulation, as headers in csv
    files. This function imports that data, and creates dictionaries to store it.

    The Python dictionary keys are the same variable names as their C++ counterparts (for consistency). Casting is
    required as data comes from csvreader as strings.

    :param str filename: The filename of the data to be imported. Obtained from data_analysis.data_analysis()

    :return: Returns a tuple. [0] is the dictionary containing all the key simulation parameters. [1] is an array
    containing strings; the names of each spin site.
    """
    with open(filename) as file_header_data:
        csv_reader = csv.reader(file_header_data)
        next(csv_reader)  # 1st line. title_line
        next(csv_reader)  # 2nd line. Blank.
        next(csv_reader)  # 3rd line. Column title for each key simulation parameter. data_names
        data_values = next(csv_reader)  # 4th line. Values associated with column titles from 3rd line.
        next(csv_reader)  # 5th line. Blank.
        next(csv_reader)  # 6th line. Simulation notes. sim_notes
        next(csv_reader)  # 7th line. Describes how to understand column titles from 3rd line. data_names_explained
        next(csv_reader)  # 8th line. Blank.
        simulated_spin_sites = next(csv_reader)  # 9th line. Number for each spin site that was simulated

    # Assignment to dict is done individually to improve readability.
    key_params = dict()
    key_params['biasField'] = float(data_values[0])
    key_params['biasFieldDriving'] = float(data_values[1])
    key_params['biasFieldDrivingScale'] = float(data_values[2])
    key_params['drivingFreq'] = float(data_values[3])
    key_params['drivingRegionLHS'] = int(data_values[4])
    key_params['drivingRegionRHS'] = int(data_values[5])
    key_params['drivingRegionWidth'] = int(data_values[6])
    key_params['maxSimTime'] = float(data_values[7])
    key_params['exchangeMaxVal'] = float(data_values[8])
    key_params['stopIterVal'] = float(data_values[9])
    key_params['exchangeMinVal'] = float(data_values[10])
    key_params['numberOfDataPoints'] = int(data_values[11])
    key_params['numSpins'] = int(data_values[12])
    key_params['stepsize'] = float(data_values[13])

    legend_labels = create_plot_labels(simulated_spin_sites,
                                       key_params['drivingRegionLHS'], key_params['drivingRegionRHS'])
    return key_params, legend_labels


def create_plot_labels(simulated_sites, drive_lhs_site, drive_rhs_site):
    """
    Generates arrays for use in plotting.

    :param list simulated_sites: List of sites that were simulated.
    :param int drive_lhs_site:  Site number of the left-hand (lhs) endpoint of the driving region.
    :param int drive_rhs_site:  Site number of the right-hand site (rhs) endpoint of the driving region.

    :return:  List of strings. Each string provides a description of each simulated spin site, including any
              significance of the site.To be used as 'legend labels' in plots.
    """
    for i, site in enumerate(simulated_sites):
        # Imported data will be range of dtypes. Casting here ensures later comparisons can occur.
        simulated_sites[i] = int(site)

    # Save as lists; will be accessed sequentially in plotting functions
    legend_labels = []

    for _, spin_site in enumerate(simulated_sites):
        # Test if each spin site is special, such as being within the driving region.
        if spin_site == drive_lhs_site:
            legend_labels.append(f"LHS DR Spin Site #{spin_site}")

        elif spin_site == drive_rhs_site:
            legend_labels.append(f"RHS DR Spin Site #{spin_site}")

        elif drive_lhs_site < spin_site < drive_rhs_site:
            # Spin site is within the driving region, but not an endpoint
            legend_labels.append(f"DR Spin Site #{spin_site}")

        else:
            legend_labels.append(f"Spin Site #{spin_site}")

    return legend_labels


def plot_graph(filename, amplitude_data, sites_to_compare=None):
    """
    Plots a graph

    :param str filename: Imported data to be plotted. Should only contain the values (no headers, for example).
    :param float64 amplitude_data: The magnitudes of the spin's magnetisation at each moment in time for each spin site.
    :param list[int] sites_to_compare: cake
    """
    key_data, subplot_labels = import_data_headers(filename)
    key_data['maxSimTime'] *= 1e9
    time_values = np.linspace(0, key_data['maxSimTime'], key_data['numberOfDataPoints'] + 1)

    fig = plt.figure(figsize=(12, 12))
    plt.suptitle('Mx Values from Shockwave Code\nRK2[Midpoint]', size=24)

    # Three subplots which are each a different pane
    plot_pane_1 = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=2)  # Top pane for comparison of multiple datasets
    plot_pane_2 = plt.subplot2grid((4, 4), (2, 0), colspan=2, rowspan=2)  # Bottom left pane to show any single dataset
    plot_pane_3 = plt.subplot2grid((4, 4), (2, 2), colspan=2, rowspan=2)  # Bottom right pane to track final spin site

    for ax in [plot_pane_1, plot_pane_2, plot_pane_3]:
        # Convert all x-axis tick labels to scientific notation on all panes
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    spin_sites_to_plot = []
    for i in range(0, len(amplitude_data[0])):
        # The variable spin_sites_to_plot is useful when building the software, as sometimes not all sites are wanted
        spin_sites_to_plot.append(amplitude_data[:, i])

    for site, magnitudes in enumerate(spin_sites_to_plot):

        if sites_to_compare:
            # Assigns each spin site to a pane based upon either user's input (IF), or presets (ELSE)
            if site in sites_to_compare:
                axes = plot_pane_1
                axes.set(title=f"Comparison of User-Selected Sites")  # Sets subplot title
            elif site == (len(spin_sites_to_plot) - 1):
                axes = plot_pane_3
                axes.set(title=f'Final Simulated Site')  # Sets subplot title
            else:
                axes = plot_pane_2
                axes.set(title=f'All Other Sites')  # Sets subplot title
        else:
            if site == 0:
                axes = plot_pane_2
                axes.set(title=f'First Simulated Site')  # Sets subplot title
            elif site == (len(spin_sites_to_plot) - 1):
                axes = plot_pane_3
                axes.set(title=f'Final Simulated Site')  # Sets subplot title
            else:
                axes = plot_pane_1
                axes.set(title=f'All Other Sites')  # Sets subplot title

        axes.xaxis.set(major_locator=ticker.MultipleLocator(key_data['maxSimTime'] * 0.25),
                       minor_locator=ticker.MultipleLocator(key_data['maxSimTime'] * 0.125))

        axes.plot(time_values, magnitudes, ls='-', lw=3, label=subplot_labels[site])
        axes.set(xlabel='Time [ns]', ylabel='Signal [arb.]', xlim=[0, key_data['maxSimTime']])
        axes.legend(loc=1, frameon=True)

        # This IF statement allows the comparison pane (a1) to have different axis limits to the other panes
        if axes == plot_pane_1:
            axes.yaxis.set(major_locator=ticker.MaxNLocator(nbins=5, prune='lower'),
                           minor_locator=ticker.AutoMinorLocator())
        else:
            axes.yaxis.set(major_locator=ticker.MaxNLocator(nbins=5, prune='lower'),
                           minor_locator=ticker.AutoMinorLocator())

    fig.tight_layout()
    fig.savefig(f"{sp.directory_tree_testing()[1]}rk2Shockwave_Test1542.png")
    plt.show()
