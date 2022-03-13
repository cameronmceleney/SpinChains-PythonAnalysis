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
                                       key_params['drivingRegionLHS'], key_params['drivingRegionRHS'])[0]

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


def plot_graph(filename):
    """
    Plots a graph

    :param str filename: Imported data to be plotted. Should only contain the values (no headers for example).

    :return:
    """
    key_data, subplot_labels = import_data_headers(filename)

    ylim_max_subplot_a1 = key_data['biasFieldDriving']
    ylim_max_subplot_others = key_data['biasFieldDriving'] * (1 / key_data['biasFieldDrivingScale']) * 1e-2

    mx_inputfile_np = np.loadtxt(f"{sp.directory_tree_testing()[0]}rk2Shockwave_Test1542.csv", delimiter=",",
                                 skiprows=9)

    # The first two arrays will be compared on pane1, and the final two will each have their own pane
    SitesToPlotContainer = [mx_inputfile_np[:, 0], mx_inputfile_np[:, 1], mx_inputfile_np[:, 2], mx_inputfile_np[:, 3]]
    SubPlotTitles = ['Input Site(s)',
                     'Input Site(s)',
                     'Output Site',
                     'Output Site']

    # contains all the labels needed for the plots. 1st row: titles. 2nd row: x-axis labels. 3rd row: y-axis labels
    PlotAllLabels = ['Mx Values from Shockwave Code\nRK2[Midpoint]',
                     'stopIterVal', 'Time (s)',
                     'M Value', 'Signal (a.u.)']

    fig = plt.figure(figsize=(12, 12))

    plot_pane_1 = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=2)  # Top pane for comparison of multiple datasets
    plot_pane_2 = plt.subplot2grid((4, 4), (2, 0), colspan=2, rowspan=2)  # Bottom left pane to show any single dataset
    plot_pane_3 = plt.subplot2grid((4, 4), (2, 2), colspan=2, rowspan=2)  # Bottom right pane to track final spin site

    # Figure title (here) to allow for individual panes to have their own titles
    plt.suptitle(PlotAllLabels[0], size=24)

    # X-axis will always show either time, or thickness of sample. As all datasets come from the
    # same simulation, this means that all plots will have the same x-axis. Thus this array can
    # be outwith the loop
    time_values = np.linspace(0, key_data['maxSimTime'], key_data['numberOfDataPoints'] + 1)

    for k in range(0, len(SitesToPlotContainer)):

        # Each iteration of the FOR loop handles a different subplot. Design choice to allow
        # for easy scaling. To add a new pane, simply create a new subplot2grid, add dataset to
        # 'SitesToPlotContainer' and then finally add a new statement to this IF block
        axes = None

        if k == 0:
            axes = plot_pane_1
        elif k == 1:
            axes = plot_pane_1
        elif k == 2:
            axes = plot_pane_2
        elif k == 3:
            axes = plot_pane_3

        # Sets subplot title
        axes.set_title(f'{SubPlotTitles[k]}')

        axes.xaxis.set(major_locator=ticker.MultipleLocator(key_data['maxSimTime'] * 0.25),
                       minor_locator=ticker.MultipleLocator(key_data['maxSimTime'] * 0.125 / 1))

        axes.plot(time_values, SitesToPlotContainer[k], ls='-', lw=3, label=subplot_labels[k])
        axes.set(xlabel=PlotAllLabels[2], ylabel=PlotAllLabels[4], xlim=[0, key_data['maxSimTime']])
        axes.legend(loc=1, frameon=True)

        # This IF statement allows the comparison pane (a1) to have different axis limits to the other panes
        if k <= 1:
            axes.set(ylim=[-1 * ylim_max_subplot_a1, ylim_max_subplot_a1])
            axes.yaxis.set(major_locator=ticker.MaxNLocator(nbins=5, prune='lower'),
                           minor_locator=ticker.AutoMinorLocator())
        elif k >= 2:
            axes.set(ylim=[-1 * ylim_max_subplot_others, ylim_max_subplot_others])
            axes.yaxis.set(major_locator=ticker.MaxNLocator(nbins=5, prune='lower'),
                           minor_locator=ticker.AutoMinorLocator())

    # Tightening layout greatly reduces plot size before saving
    fig.tight_layout()

    fig.savefig(f"{sp.directory_tree_testing()[1]}rk2Shockwave_Test1542.png")

    plt.show()
