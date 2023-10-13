# -*- coding: utf-8 -*-

# -------------------------- Preprocessing Directives -------------------------

# Standard Libraries
from sys import exit

# 3rd Party packages
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.ticker as ticker
from typing import Any
from scipy.fft import rfft, rfftfreq

# My packages/Header files
# Here

# ----------------------------- Program Information ----------------------------

"""
This file contains functions that I wrote early on in my PhD. While they have generally been
"""
PROGRAM_NAME = "plot_rk_methods_(legacy).py"
"""
Created on 18:38 by CameronMcEleney
"""


# -------------------------------------- Useful to look at shockwaves. Three panes -------------------------------------
def three_panes(amplitude_data, key_data, filename, sites_to_compare=None):
    """
    Plots a graph

    :param Any amplitude_data: Array of magnitudes of the spin's magnetisation at each moment in time for each spin
                               site.
    :param dict key_data: All key simulation parameters imported from csv file.
    :param list[list[int]] sites_to_compare: Optional. User-defined list of sites to plot.
    :param filename: data.
    """
    key_data['maxSimTime'] *= 1e9

    flat_list = [item for sublist in sites_to_compare for item in sublist]
    subplot_labels = create_plot_labels(flat_list, key_data['drivingRegionLHS'], key_data['drivingRegionRHS'])

    time_values = np.linspace(0, key_data['maxSimTime'], int(key_data['numberOfDataPoints']) + 1)

    fig = plt.figure(figsize=(16, 12))
    plt.suptitle('Comparison of $M_x$ Values At\nDifferent Spin Sites', size=24)

    # Three subplots which are each a different pane
    plot_pane_1 = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=2)  # Top pane for comparison of multiple datasets
    plot_pane_2 = plt.subplot2grid((4, 4), (2, 0), colspan=2, rowspan=2)  # Bottom left pane to show any single dataset
    plot_pane_3 = plt.subplot2grid((4, 4), (2, 2), colspan=2, rowspan=2)  # Bottom right pane to track final spin site

    # for ax in [plot_pane_1, plot_pane_2, plot_pane_3]:
    #     # Convert all x-axis tick labels to scientific notation on all panes
    #     ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    max_value = max([max(sublist) for sublist in sites_to_compare])

    spin_sites_to_plot = []
    for i in range(1, len(amplitude_data[0])):
        # Exclude 0 as the first column of data will always be time values.
        # The variable spin_sites_to_plot is useful when building the software, as sometimes not all sites are wanted.
        spin_sites_to_plot.append(amplitude_data[:, i])

    label_counter = 0
    line_width = 1
    for site in range(1, len(amplitude_data[0, :] + 1)):
        magnitudes = amplitude_data[:, site]

        if site > max_value:
            break

        # Assigns each spin site to a pane based upon either user's input (IF), or presets (ELSE)
        if site in sites_to_compare[0]:
            # dt = np.pi / 100.
            # fs = 1. / dt
            # t = np.arange(0, 8, dt)
            # y = 10. * np.sin(2 * np.pi * 4 * t) + 5. * np.sin(2 * np.pi * 4.25 * t)
            # y = y + np.random.randn(*t.shape)
            # plot_pane_1.psd(y, NFFT=len(t), pad_to=len(t), Fs=fs)
            fs = (key_data['maxSimTime'] / (key_data['numberOfDataPoints'] - 1)) \
                 / 1e-9
            norm = np.linalg.norm(magnitudes)
            normal_magnitudes = magnitudes / norm
            plot_pane_1.psd(normal_magnitudes, NFFT=len(time_values), pad_to=len(time_values), Fs=fs)
            # plot_pane_1.plot(time_values, magnitudes, ls='-', lw=line_width, label=subplot_labels[label_counter])
            label_counter += 1

        elif site in sites_to_compare[1]:
            plot_pane_2.plot(time_values, magnitudes, ls='-', lw=line_width, label=subplot_labels[label_counter])
            label_counter += 1

        elif site in sites_to_compare[2]:
            plot_pane_3.plot(time_values, magnitudes, ls='-', lw=line_width, label=subplot_labels[label_counter])
            label_counter += 1

    for axes in [plot_pane_1, plot_pane_2, plot_pane_3]:
        axes.set(xlabel='Time (ns)', ylabel="m$_x$")  # xlim=[0, key_data['maxSimTime']]
        axes.legend(loc=1, frameon=True)
        # axes.xaxis.set(major_locator=ticker.MultipleLocator(key_data['maxSimTime'] * 0.1),
        #                minor_locator=ticker.MultipleLocator(key_data['maxSimTime'] * 0.05))
        # This IF statement allows the comparison pane (a1) to have different axis limits to the other panes
        if axes == plot_pane_1:
            axes.set(title=f"Primary Sites")
            axes.yaxis.set(major_locator=ticker.MaxNLocator(nbins=5, prune='lower'),
                           minor_locator=ticker.AutoMinorLocator())
        else:
            if axes == plot_pane_2:
                axes.set(title=f'Secondary Sites')
            elif axes == plot_pane_3:
                axes.set(title=f'Tertiary Sites')

            axes.yaxis.set(major_locator=ticker.MaxNLocator(nbins=5, prune='lower'),
                           minor_locator=ticker.AutoMinorLocator())

    fig.tight_layout()
    fig.savefig(f"{filename}_site_comparisons.png")


def create_plot_labels(simulated_sites, drive_lhs_site, drive_rhs_site):
    """
    Generates arrays for use in plotting.

    :param list simulated_sites: List of sites that were simulated.
    :param int drive_lhs_site:  Site number of the left-hand (lhs) endpoint of the driving region.
    :param int drive_rhs_site:  Site number of the right-hand site (rhs) endpoint of the driving region.

    :return:  List of strings. Each string provides a description of each simulated spin site, including any
              significance of the site.To be used as 'legend labels' in plots.
    """
    # for i, site in enumerate(simulated_sites):
    # Imported data will be range of dtypes. Casting here ensures later comparisons can occur.
    simulated_sites = [int(site) for site in simulated_sites]

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


# ------------------------------------------ FFT and Signal Analysis Functions -----------------------------------------
def fft_only(amplitude_data, spin_site, simulation_params, filename):
    plt.rcParams.update({'savefig.dpi': 100, "figure.dpi": 100})

    interactive = True
    # Use for interactive plot. Also change DPI to 40 and allow Pycharm to plot outside of tool window
    if interactive:
        fig = plt.figure(figsize=(9, 9))
    else:
        fig = plt.figure(figsize=(12, 9), constrained_layout=True)

    fig.suptitle(f"Data from Spin Site #{14000}")
    ax = plt.subplot(1, 1, 1)
    num_dp = simulation_params['numberOfDataPoints']
    # frequencies, fourier_transform, natural_frequency, driving_freq = fft_data(amplitude_data, simulation_params)
    # frequencies, fourier_transform = fft_data2(amplitude_data[:])
    # lower1, upper1 = 3363190, 3430357
    # lower2, upper2 = 3228076, 3287423
    # lower3, upper3 = 3118177, 3162992
    # frequencies_blob1, fourier_transform_blob1 = fft_data2(amplitude_data[lower1:upper1],
    #                                                        simulation_params['maxSimTime'],
    #                                                        simulation_params['numberOfDataPoints'])
    # frequencies_blob2, fourier_transform_blob2 = fft_data2(amplitude_data[lower2:upper2],
    #                                                        simulation_params['maxSimTime'],
    #                                                        simulation_params['numberOfDataPoints'])
    # frequencies_blob3, fourier_transform_blob3 = fft_data2(amplitude_data[lower3:upper3],
    #                                                        simulation_params['maxSimTime'],
    #                                                        simulation_params['numberOfDataPoints'])
    frequencies_precursors, fourier_transform_precursors = fft_data2(amplitude_data[12:int(num_dp * 0.25)],
                                                                     simulation_params['maxSimTime'],
                                                                     simulation_params['numberOfDataPoints'])
    frequencies_dsw, fourier_transform_dsw = fft_data2(amplitude_data[int(num_dp * 0.27) + 1:int(num_dp * 0.34)],
                                                       simulation_params['maxSimTime'],
                                                       simulation_params['numberOfDataPoints'])
    frequencies_dsw2, fourier_transform_dsw2 = fft_data2(amplitude_data[int(num_dp * 0.27) + 1:int(num_dp * 0.4)],
                                                         simulation_params['maxSimTime'],
                                                         simulation_params['numberOfDataPoints'])
    frequencies_eq, fourier_transform_eq = fft_data2(amplitude_data[int(num_dp * 0.6) + 1:int(num_dp * 0.95)],
                                                     simulation_params['maxSimTime'],
                                                     simulation_params['numberOfDataPoints'])

    # Plotting. To normalise data, change y-component to (1/N)*abs(fourier_transform) where N is the number of samples.
    # Set marker='o' to see each datapoint, else leave as marker= to hide
    # ax.plot(frequencies, abs(fourier_transform), lw=1, color='white', markerfacecolor='black',
    # markeredgecolor='black', label='all data', alpha=0.0)

    # ax.plot(frequencies_blob1, abs(fourier_transform_blob1), marker='', lw=1, color='#37782c',
    #        markerfacecolor='black', markeredgecolor='black', ls=':', label='Blob 1')
    # ax.plot(frequencies_blob2, abs(fourier_transform_blob2), marker='', lw=1, color='#37782c',
    #        markerfacecolor='black', markeredgecolor='black', ls='--', label='Blob 2')
    # ax.plot(frequencies_blob3, abs(fourier_transform_blob3), marker='', lw=1, color='#37782c',
    #        markerfacecolor='black', markeredgecolor='black', ls='-.', label='Blob 3')
    ax.plot(frequencies_precursors, abs(fourier_transform_precursors), marker='', lw=1, color='#4fc1e8',
            markerfacecolor='black', markeredgecolor='black', label="Pre-Precursors", zorder=1.5)
    ax.plot(frequencies_dsw, abs(fourier_transform_dsw), marker='', lw=1, color='#a0d568',
            markerfacecolor='black', markeredgecolor='black', label="Precursors", zorder=1.3)
    ax.plot(frequencies_dsw2, abs(fourier_transform_dsw2), marker='', lw=1, color='#ffce54',
            markerfacecolor='black', markeredgecolor='black', label="Shockwave Region", zorder=1.2)
    ax.plot(frequencies_eq, abs(fourier_transform_eq), marker='', lw=1, color='#ed5564',
            markerfacecolor='black', markeredgecolor='black', label="Steady State", zorder=1.1)

    ax.set(xlabel="Frequency (GHz)", ylabel="Amplitude (arb. units)", yscale='log', xlim=[0, 16000], ylim=[1e-7, 1e-0])

    ax.legend(loc=0, frameon=True, fancybox=True, facecolor='white', edgecolor='white',
              title=f'Freq. List (GHz)\nDriving - {12500} GHz', fontsize=12)

    # ax.xaxis.set_major_locator(MultipleLocator(5))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    # For the minor ticks, use no labels; default NullFormatter.
    # ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.grid(color='white')
    ax.grid(False)

    if simulation_params['exchangeMinVal'] == simulation_params['exchangeMaxVal']:
        exchangeString = f"Uniform = True ({simulation_params['exchangeMinVal']}) (T)"
    else:
        exchangeString = f"J$_{{min}}$ = {simulation_params['exchangeMinVal']} (T) | J$_{{max}}$ = " \
                         f"{simulation_params['exchangeMaxVal']} (T)"
    text_string = (f"H$_{{0}}$ = {simulation_params['staticBiasField']} (T) | "
                   f"H$_{{D1}}$ = {simulation_params['dynamicBiasField1']: 2.2e} (T) |\n"
                   f"H$_{{D2}}$ = {simulation_params['dynamicBiasField2']: 2.2e}(T) | {exchangeString} | "
                   f"N = {simulation_params['totalSpins']}")

    props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.5, -0.10, text_string, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=props, ha='center', va='center')
    fig.tight_layout()

    if interactive:
        # For interactive plots
        def mouse_event(event: Any):
            print(f'x: {event.xdata} and y: {event.ydata}')

        fig.canvas.mpl_connect('button_press_event', mouse_event)
        fig.tight_layout()  # has to be here
        plt.show()
    else:
        fig.savefig(f"{filename}_ft_only_{spin_site}.png")


def fft_and_signal_four(time_data, amplitude_data, spin_site, simulation_params, filename):
    """
    Plot the magnitudes of the magnetic moment of a spin site against time, as well as the FFTs, over four subplots.

    :param time_data: The time data as an array which must be of the same length as y_axis_data.
    :param amplitude_data: The magnitudes of the spin's magnetisation at each moment in time.
    :param dict simulation_params: Key simulation parameters.
    :param int spin_site: The spin site being plotted.
    :param filename: The name of the file that is being read from.

    :return: A figure containing four sub-plots.
    """
    # Find maximum time in (ns) to the nearest whole (ns), then find how large shaded region should be.
    temporal_xlim = np.round(simulation_params['stopIterVal'] * simulation_params['stepsize'] * 1e9, 1)
    x_scaling = 0.1
    fft_shaded_box_width = 10  # In GHz
    offset = 0  # Zero by default
    t_shaded_xlim = temporal_xlim * x_scaling + offset

    plot_set_params = {0: {"title": "Full Simulation", "xlabel": "Time (ns)", "ylabel": "Amplitude (arb. units)",
                           "xlim": (offset, temporal_xlim)},
                       1: {"title": "Shaded Region", "xlabel": "Time (ns)", "xlim": (offset, t_shaded_xlim)},
                       2: {"title": "Showing All Artefacts", "xlabel": "Frequency (GHz)",
                           "ylabel": "Amplitude (arb. units)",
                           "xlim": (0, 60)},
                       3: {"title": "Shaded Region", "xlabel": "Frequency (GHz)", "xlim": (0, fft_shaded_box_width)}}

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)

    fig.suptitle(f"Data from Spin Site #{spin_site}")

    # create 2 rows of sub-figures
    all_sub_figs = fig.subfigures(nrows=2, ncols=1)

    for row, sub_fig in enumerate(all_sub_figs):
        if row == 0:
            sub_fig.suptitle(f"Temporal Data")

            axes = sub_fig.subplots(nrows=1, ncols=2)

            for i, ax in enumerate(axes, start=row * 2):
                custom_temporal_plot(time_data, amplitude_data, ax=ax, which_subplot=i,
                                     plt_set_kwargs=plot_set_params[i], xlim_scaling=x_scaling, offset=offset)

        else:
            sub_fig.suptitle(f"FFT Data")

            axes = sub_fig.subplots(nrows=1, ncols=2)

            for i, ax in enumerate(axes, start=row * 2):
                custom_fft_plot(amplitude_data, ax=ax, which_subplot=i,
                                plt_set_kwargs=plot_set_params[i], simulation_params=simulation_params,
                                box_width=fft_shaded_box_width)

    fig.savefig(f"{filename}_{spin_site}.png")


def custom_temporal_plot(time_data, amplitude_data, plt_set_kwargs, which_subplot, offset=0, xlim_scaling=0.2, ax=None):
    """
    Custom plotter for each temporal signal within 'fft_and_signal_four'.

    Description.

    :param float time_data: The time data as an array which must be of the same length as y_axis_data.
    :param amplitude_data: The magnitudes of the spin's magnetisation at each moment in time.
    :param dict plt_set_kwargs: **kwargs for the ax.set() function.
    :param int which_subplot: Number of subplot (leftmost is 0). Used to give a subplot any special characteristics.
    :param float offset: Shift x-axis of plots by the given amount.
    :param float xlim_scaling: Allows for the shaded region to have its width varied.
    :param ax: Axes data from plt.subplots(). Used to define subplot and figure behaviour.

    :return: The subplot information required to be plotted in the main figure environment.
    """

    if ax is None:
        ax = plt.gca()

    ax.plot(time_data, amplitude_data)
    ax.set(**plt_set_kwargs)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    if which_subplot == 0:
        # plt_set_kwargs['xlim'][1] is the upper value of the xlim
        ax.axvspan(offset, offset + plt_set_kwargs['xlim'][1] * xlim_scaling, color='#DC143C', alpha=0.2, lw=0)

    ax.grid(False)

    return ax


def custom_fft_plot(amplitude_data, plt_set_kwargs, which_subplot, simulation_params, box_width, ax=None):
    """
    Custom plotter for each FFT within 'fft_and_signal_four'.

    Description.

    :param box_width: Width of box to draw.
    :param amplitude_data: The magnitudes of the spin's magnetisation at each moment in time.
    :param dict plt_set_kwargs: **kwargs for the ax.set() function.
    :param int which_subplot: Number of subplot (leftmost is 0). Used to give a subplot any special characteristics.
    :param dict simulation_params: Key simulation parameters from csv header.
    :param ax: Axes data from plt.subplots(). Used to define subplot and figure behaviour.

    :return: The subplot information required to be plotted in the main figure environment."""
    frequencies, fourier_transform, natural_frequency, driving_freq = fft_data(amplitude_data, simulation_params)

    # driving_freq_hz = simulation_params['drivingFreq'] / 1e9

    if ax is None:
        ax = plt.gca()

    # Plotting. To normalise data, change y-component to (1/N)*abs(fourier_transform) where N is the number of samples.
    # Set marker='o' to see each datapoint, else leave as marker= to hide
    ax.plot(frequencies, abs(fourier_transform),
            marker='', lw=1, color='red', markerfacecolor='black', markeredgecolor='black')
    ax.set(**plt_set_kwargs)

    # ax.axvline(x=driving_freq_hz, label=f"Driving. {driving_freq_hz:2.2f}", color='green')

    if which_subplot == 2:
        ax.axvspan(0, box_width, color='#DC143C', alpha=0.2, lw=0)
        # If at a node, then 3-wave generation may be occurring. This loop plots that location.
        # triple_wave_gen_freq = driving_freq_hz * 3
        # ax.axvline(x=triple_wave_gen_freq, label=f"T.W.G. {triple_wave_gen_freq:2.2f}", color='purple')
    else:
        if simulation_params['exchangeMinVal'] == simulation_params['exchangeMaxVal']:
            exchangeString = f"Uniform Exc. ({simulation_params['exchangeMinVal']} (T))"
        else:
            exchangeString = f"J$_{{min}}$ = {simulation_params['exchangeMinVal']} (T) | J$_{{max}}$ = " \
                             f"{simulation_params['exchangeMaxVal']} (T)"
        text_string = ((f"H$_{{0}}$ = {simulation_params['staticBiasField']} (T) | "
                        f"H$_{{D1}}$ = {simulation_params['dynamicBiasField1']: 2.2e} (T) | "
                        f"H$_{{D2}}$ = {simulation_params['dynamicBiasField2']: 2.2e}(T) \n{exchangeString} | "
                        f"N = {simulation_params['chainSpins']} | ") + r"$\alpha$" +
                       f" = {simulation_params['gilbertFactor']: 2.2e}")

        props = dict(boxstyle='round', facecolor='gainsboro', alpha=1.0)
        # place a text box in upper left in axes coords
        ax.text(0.5, -0.2, text_string, transform=ax.transAxes, fontsize=18,
                verticalalignment='top', bbox=props, ha='center', va='center')
        # By default, plots the natural frequency.
        # ax.axvline(x=natural_frequency, label=f"Natural. {natural_frequency:2.2f}")

    ax.legend(loc=0, frameon=True, fancybox=True, facecolor='white', edgecolor='white',
              title='Freq. List (GHz)', fontsize=12)

    ax.grid(color='white')
    ax.grid(False)

    return ax


def fft_data(amplitude_data, simulation_params):
    """
    Computes the FFT transform of a given signal, and also outputs useful data such as key frequencies.

    :param dict simulation_params: Imported key simulation parameters.
    :param amplitude_data: Magnitudes of magnetic moments for a spin site

    :return: A tuple containing the frequencies [0], FFT [1] of a spin site. Also includes the  natural frequency
    (1st eigenvalue) [2], and driving frequency [3] for the system.
    """
    # Simulation parameters needed for FFT computations that are always the same are saved here.
    # gamma is in [GHz/T] here.
    core_values = {"gamma": simulation_params["gyroMagRatio"] / (2 * np.pi),
                   "hz_to_ghz": 1e-9}

    # Data in file header is in [Hz] by default.
    driving_freq_ghz = simulation_params['drivingFreq'] * core_values["hz_to_ghz"]

    # This is the (first) natural frequency of the system, corresponding to the first eigenvalue. Change as needed to
    # add other markers to the plot(s)
    natural_freq = core_values['gamma'] * simulation_params['staticBiasField']

    # Find bin size by dividing the simulated time into equal segments based upon the number of data-points.
    sample_spacing = (simulation_params["maxSimTime"] / (simulation_params['numberOfDataPoints'] - 1)) / core_values[
        'hz_to_ghz']

    # Compute the FFT
    n = amplitude_data.size
    normalised_data = amplitude_data

    fourier_transform = rfft(normalised_data)
    frequencies = rfftfreq(n, sample_spacing)

    return frequencies, fourier_transform, natural_freq, driving_freq_ghz


def fft_data2(amplitude_data, maxtime, number_dp):
    """
        Computes the FFT transform of a given signal, and also outputs useful data such as key frequencies.

        :param number_dp: Number of datapoints
        :param maxtime: Maximum time simulated.
        :param amplitude_data: Magnitudes of magnetic moments for a spin site

        :return: A tuple containing the frequencies [0], FFT [1] of a spin site. Also includes the  natural frequency
        (1st eigenvalue) [2], and driving frequency [3] for the system.
        """
    # Simulation parameters needed for FFT computations that are always the same are saved here.
    # gamma is in [GHz/T] here.
    core_values = {"gamma": 29.2e9 / (2 * np.pi),
                   "hz_to_ghz": 1e-9}

    # Data in file header is in [Hz] by default.
    # driving_freq_ghz = self.driving_freq # * core_values["hz_to_ghz"]

    # This is the (first) natural frequency of the system, corresponding to the first eigenvalue. Change as needed to
    # add other markers to the plot(s)
    # natural_freq = core_values['gamma'] * self.static_field

    # Find bin size by dividing the simulated time into equal segments based upon the number of data-points.
    sample_spacing = (maxtime / (number_dp - 1)) / core_values[
        'hz_to_ghz']

    # Compute the FFT
    n = amplitude_data.size
    normalised_data = amplitude_data

    fourier_transform = rfft(normalised_data)
    frequencies = rfftfreq(n, sample_spacing)

    return frequencies, fourier_transform  # , natural_freq, driving_freq_ghz
