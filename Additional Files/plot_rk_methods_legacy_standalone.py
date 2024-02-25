# -*- coding: utf-8 -*-

# -------------------------- Preprocessing Directives -------------------------

# Standard Libraries

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
PROGRAM_NAME = "plot_rk_methods_legacy_standalone.py"
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


def create_contour_plot(mx_data, my_data, mz_data, spin_site, output_file, use_tri=False):
    x = mx_data[:, spin_site]
    y = my_data[:, spin_site]
    z = mz_data[:, spin_site]
    time = mx_data[:, 0]

    # 'magma' is also nice
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(time, x, y, label=f'Spin Site {spin_site}', cmap='Blues', edgecolor='none', lw=0.1)
    ax.set_xlabel('time', fontsize=12)
    ax.set_ylabel('m$_x$', fontsize=12)
    ax.set_zlabel('m$_y$', fontsize=12)
    ax = plt.axes(projection='3d')
    if use_tri:
        ax.plot_trisurf(x, y, z, cmap='Blues', lw=0.1, edgecolor='none', label=f'Spin Site {spin_site}')
    else:
        ax.plot3D(x, y, z, label=f'Spin Site {spin_site}')
        ax.legend()

    ax.set_xlabel('m$_x$', fontsize=12)
    ax.set_ylabel('m$_y$', fontsize=12)
    ax.set_zlabel('m$_z$', fontsize=12)

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    fig.savefig(f"{output_file}_contour.png")


def test_3d_plot(mx_data, my_data, mz_data, spin_site):
    x1 = mx_data[:, spin_site]
    y1 = my_data[:, spin_site]
    z1 = mz_data[:, spin_site]

    x2 = mx_data[:, spin_site + 1]
    y2 = my_data[:, spin_site + 1]
    z2 = mz_data[:, spin_site + 1]
    # time = mx_data[:, 0]

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot3D(x1, y1, z1, markersize=1, color='grey', ls='--', alpha=0.8, label='site1', zorder=1.2)
    ax.plot3D(x2, y2, z2, markersize=1, color='black', alpha=0.8, label='site2', zorder=1.1)

    # ax.invert_xaxis()
    ax.set(xlabel='x', ylabel='y', zlabel='z', zlim=[-1, 1])
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"/Users/cameronmceleney/CLionProjects/Data/2022-10-12/Outputs/threeDplot.png")
    plt.show()

def create_time_variation(self, spin_site, colour_precursors=False, annotate_precursors=False,
                          basic_annotations=False, add_zoomed_region=False, add_info_box=False,
                          add_coloured_regions=False, interactive_plot=False):
        """
        LEGACY METHOD FROM PAPERFIGURES. WILL NOT RUN IN THIS MODULE. Plots the magnetisation of a site against time.

        One should ensure that the site being plotted is not inside either of the driving- or damping-regions.

        :param interactive_plot: Allow plot to be interactive
        :param basic_annotations: NEED DOCSTRING
        :param annotate_precursors: Add arrows to denote precursors.
        :param colour_precursors: Draw 1st, 3rd and 5th precursors as separate colours to main figure.
        :param bool add_coloured_regions: Draw coloured boxes onto plot to show driving- and damping-regions.
        :param bool add_info_box: Add text box to base of plot which lists key simulation parameters.
        :param bool add_zoomed_region: Add inset to plot to focus upon precursors.
        :param int spin_site: The number of the spin site to be plotted.

        :return: Saves a .png image to the designated output folder.
        """

        if self.fig is None:
            self.fig = plt.figure(figsize=(4.5, 3.375))
        num_rows = 2
        num_cols = 3
        ax1 = plt.subplot2grid((num_rows, num_cols), (0, 0), rowspan=int(num_rows / 2),
                               colspan=num_cols, fig=self.fig)
        ax2 = plt.subplot2grid((num_rows, num_cols), (int(num_rows / 2), 0),
                               rowspan=num_rows, colspan=num_cols, fig=self.fig)

        ax1.xaxis.labelpad = -1
        ax2.xaxis.labelpad = -1

        ########################################

        ax1_xlim_lower, ax1_xlim_upper = 0.0, 5
        ax1_xlim_range = ax1_xlim_upper - ax1_xlim_lower
        xlim_min, xlim_max = 0, self.max_time  # ns

        ax1_yaxis_base, ax1_yaxis_exponent = 3, '-3'
        ax1_yaxis_order = float('1e' + ax1_yaxis_exponent)

        lower1, upper1 = 0, 2.6
        lower2, upper2 = upper1, 3.76
        lower3, upper3 = upper2, ax1_xlim_upper

        ax1_inset_lower = 0.7

        lower1_blob, upper1_blob = 1.75, upper1  # 0.481, 0.502
        lower2_blob, upper2_blob = 1.3, 1.7  # 0.461, 0.48
        lower3_blob, upper3_blob = 1.05, 1.275  # 0.442, 0.4605

        ax1.set(xlabel=f"Time (ns)", ylabel=r"$\mathrm{m_x}$ (10$^{-3}$)",
                xlim=[ax1_xlim_lower, ax1_xlim_upper],
                ylim=[-ax1_yaxis_base * ax1_yaxis_order * 1.4, ax1_yaxis_base * ax1_yaxis_order])

        ax2.set(xlabel=f"Frequency (GHz)", ylabel=f"Amplitude (arb. units)",
                xlim=[0, 99.999], ylim=[1e-1, 1e3], yscale='log')

        self._tick_setter(ax1, ax1_xlim_range * 0.5, ax1_xlim_range * 0.125, 3, 4, xaxis_num_decimals=1)
        self._tick_setter(ax2, 20, 5, 6, None, is_fft_plot=True)

        line_height = -3.15 * ax1_yaxis_order

        ########################################

        if ax1_xlim_lower > ax1_xlim_upper:
            exit(0)

        def convert_norm(val, a=0, b=1):
            # return int(self.data_points * (2 * ax1_xlim_lower + ( a * (xlim_signal - ax1_xlim_lower) /
            # xlim_max )))  # original
            return int(self.data_points * ((b - a) * ((val - xlim_min) / (xlim_max - xlim_min)) + a))

        lower1_signal, upper1_signal = convert_norm(lower1), convert_norm(upper1)
        lower2_signal, upper2_signal = convert_norm(lower2), convert_norm(upper2)
        lower3_signal, upper3_signal = convert_norm(lower3), convert_norm(upper3)

        lower1_precursor, upper1_precursor = convert_norm(lower1_blob), convert_norm(upper1_blob)
        lower2_precursor, upper2_precursor = convert_norm(lower2_blob), convert_norm(upper2_blob)
        lower3_precursor, upper3_precursor = convert_norm(lower3_blob), convert_norm(upper3_blob)

        ax1.plot(self.time_data[:],
                 self.amplitude_data[:, spin_site], ls='-', lw=0.75,
                 color='#37782c', alpha=0.5,
                 markerfacecolor='black', markeredgecolor='black', zorder=1.01)
        ax1.plot(self.time_data[lower1_signal:upper1_signal],
                 self.amplitude_data[lower1_signal:upper1_signal, spin_site], ls='-', lw=0.75,
                 color='#37782c', label=f"{self.sites_array[spin_site]}",
                 markerfacecolor='black', markeredgecolor='black', zorder=1.1)
        ax1.plot(self.time_data[lower2_signal:upper2_signal],
                 self.amplitude_data[lower2_signal:upper2_signal, spin_site], ls='-', lw=0.75,
                 color='#64bb6a', label=f"{self.sites_array[spin_site]}",
                 markerfacecolor='black', markeredgecolor='black', zorder=1.1)
        ax1.plot(self.time_data[lower3_signal:upper3_signal],
                 self.amplitude_data[lower3_signal:upper3_signal, spin_site], ls='-', lw=0.75,
                 color='#9fd983', label=f"{self.sites_array[spin_site]}",
                 markerfacecolor='black', markeredgecolor='black', zorder=1.1)

        if colour_precursors:
            ax1.plot(self.time_data[lower1_precursor:upper1_precursor],
                     self.amplitude_data[lower1_precursor:upper1_precursor, spin_site], marker='',
                     lw=0.75, color='purple',
                     markerfacecolor='black', markeredgecolor='black', label="Shockwave", zorder=1.2)
            ax1.plot(self.time_data[lower2_precursor:upper2_precursor],
                     self.amplitude_data[lower2_precursor:upper2_precursor, spin_site], marker='',
                     lw=0.75, color='red',
                     markerfacecolor='black', markeredgecolor='black', label="Steady State", zorder=1.2)
            ax1.plot(self.time_data[lower3_precursor:upper3_precursor],
                     self.amplitude_data[lower3_precursor:upper3_precursor, spin_site], marker='',
                     lw=0.75, color='blue',
                     markerfacecolor='black', markeredgecolor='black', label="Steady State", zorder=1.2)

        if basic_annotations:
            text_height = line_height - ax1_yaxis_base * 0.25 * ax1_yaxis_order
            axes_props1 = {"arrowstyle": '|-|, widthA =0.4, widthB=0.4', "color": "#37782c", 'lw': 1.0}
            axes_props2 = {"arrowstyle": '|-|, widthA =0.4, widthB=0.4', "color": "#64bb6a", 'lw': 1.0}
            axes_props3 = {"arrowstyle": '|-|, widthA =0.4, widthB=0.4', "color": "#9fd983", 'lw': 1.0}

            ax1.text(0.95, 0.9, f"(b)",
                     va='center', ha='center', fontsize=self.smaller_size, transform=ax1.transAxes)

            ax2.text(0.05, 0.9, f"(c)",
                     va='center', ha='center', fontsize=self.smaller_size,
                     transform=ax2.transAxes)

            pre_text_lhs, pre_text_rhs = lower1, upper1
            shock_text_lhs, shock_text_rhs = lower2, upper2
            equil_text_lhs, equil_text_rhs = lower3, upper3

            ax1.annotate('', xy=(pre_text_lhs, line_height), xytext=(pre_text_rhs, line_height),
                         va='center', ha='center', arrowprops=axes_props1, fontsize=self.tiny_size)
            ax1.annotate('', xy=(shock_text_lhs, line_height), xytext=(shock_text_rhs, line_height),
                         va='center', ha='center', arrowprops=axes_props2, fontsize=self.tiny_size)
            ax1.annotate('', xy=(equil_text_lhs, line_height), xytext=(equil_text_rhs, line_height),
                         va='center', ha='center', arrowprops=axes_props3, fontsize=self.tiny_size)
            ax1.text((pre_text_lhs + pre_text_rhs) / 2, text_height, 'Precursors', ha='center', va='bottom',
                     fontsize=self.tiny_size)
            ax1.text((shock_text_lhs + shock_text_rhs) / 2, text_height, 'Shockwave', ha='center', va='bottom',
                     fontsize=self.tiny_size)
            ax1.text((equil_text_lhs + equil_text_rhs) / 2, text_height, 'Equilibrium', ha='center', va='bottom',
                     fontsize=self.tiny_size)

        # Use these for paper publication figures
        # ax1.text(-0.03, 1.02, r'$\times \mathcal{10}^{{\mathcal{' + str(int(ax1_yaxis_exponent)) + r'}}}$',
        #         verticalalignment='center',
        #         horizontalalignment='center', transform=ax1.transAxes, fontsize=self.smaller_size)
        # ax1.text(0.04, 0.1, f"(a) 15 GHz", verticalalignment='center', horizontalalignment='left',
        #               transform=ax1.transAxes, fontsize=6)

        # Add zoomed in region if needed.
        if add_zoomed_region:
            # Select datasets to use
            x = self.time_data
            y = self.amplitude_data[:, spin_site]

            # Impose inset onto plot. Use 0.24 for LHS and 0.8 for RHS. 0.7 for T and 0.25 for B
            ax1_inset = inset_axes(ax1, width=2.0, height=0.5, loc="upper left",
                                   bbox_to_anchor=[0.01, 1.14], bbox_transform=ax1.transAxes)
            ax1_inset.plot(x, y, lw=0.75, color='#37782c', zorder=1.1)

            if colour_precursors:
                ax1_inset.plot(x[lower1_precursor:upper1_precursor],
                               y[lower1_precursor:upper1_precursor], marker='',
                               lw=0.75, color='purple',
                               markerfacecolor='black', markeredgecolor='black', label="Shockwave", zorder=1.2)
                ax1_inset.plot(self.time_data[lower2_precursor:upper2_precursor],
                               self.amplitude_data[lower2_precursor:upper2_precursor, spin_site], marker='',
                               lw=0.75, color='red',
                               markerfacecolor='black', markeredgecolor='black', label="Steady State", zorder=1.2)
                ax1_inset.plot(self.time_data[lower3_precursor:upper3_precursor],
                               self.amplitude_data[lower3_precursor:upper3_precursor, spin_site], marker='',
                               lw=0.75, color='blue',
                               markerfacecolor='black', markeredgecolor='black', label="Steady State", zorder=1.2)

            # Select data (of original) to show in inset through changing axis limits
            ylim_in = 2 * ax1_yaxis_order * 1e-1  # float(input("Enter ylim: "))
            ax1_inset.set_xlim(ax1_inset_lower, upper1)
            ax1_inset.set_ylim(-ylim_in, ylim_in)

            arrow_ax1_props = {"arrowstyle": '-|>', "connectionstyle": 'angle3, angleA=0, angleB=60', "color": "black",
                               'lw': 0.8}
            arrow_ax1_props2 = {"arrowstyle": '-|>', "connectionstyle": 'angle3, angleA=0, angleB=120',
                                "color": "black", 'lw': 0.8}

            ax1_inset.annotate('P1', xy=(1.9, -8e-5), xytext=(1.5, -1.3e-4), va='center', ha='center',
                               arrowprops=arrow_ax1_props, fontsize=self.tiny_size)
            ax1_inset.annotate('P2', xy=(1.5, 7e-5), xytext=(1.1, 1.3e-4), va='center', ha='center',
                               arrowprops=arrow_ax1_props2, fontsize=self.tiny_size)
            ax1_inset.annotate('P3', xy=(1.2, -2e-5), xytext=(0.8, -1.3e-4), va='center', ha='center',
                               arrowprops=arrow_ax1_props, fontsize=self.tiny_size)

            # Remove tick labels
            ax1_inset.set_xticks([])
            ax1_inset.set_yticks([])
            ax1_inset.patch.set_color("#f9f2e9")  # #f0a3a9 is equivalent to color 'red' and alpha '0.3'

            # mark_inset(ax1, ax1_inset,loc1=1, loc2=3, facecolor='#f9f2e9', edgecolor='black', alpha=1.0, zorder=1.05)

            # Add box to indicate the region which is being zoomed into on the main figure
            ax1.indicate_inset_zoom(ax1_inset, facecolor='#f9f2e9', edgecolor='black', alpha=1.0, lw=0.75,
                                    zorder=1)

        if add_info_box:
            if self.exchange_min == self.exchange_max:
                exchangeString = f"Uniform Exc. ({self.exchange_min} [T])"
            else:
                exchangeString = f"J$_{{min}}$ = {self.exchange_min} [T] | J$_{{max}}$ = " \
                                 f"{self.exchange_max} [T]"
            text_string = ((f"H$_{{0}}$ = {self.static_field} [T] | H$_{{D1}}$ = {self.driving_field1: 2.2e} [T] | "
                            f"H$_{{D2}}$ = {self.driving_field2: 2.2e}[T] \nf = {self.driving_freq} [GHz] | "
                            f"{exchangeString} | N = {self.chain_spins} | ") + r"$\alpha$" +
                           f" = {self.gilbert_factor: 2.2e}")

            props = dict(boxstyle='round', facecolor='gainsboro', alpha=1.0)

            # place a text box in upper left in axes coords
            ax1.text(0.35, -0.22, text_string, transform=ax1.transAxes, fontsize=6,
                     verticalalignment='top', bbox=props, ha='center', va='center')
            ax1.text(0.85, -0.22, "Time [ns]", fontsize=12, ha='center', va='center',
                     transform=ax1.transAxes)

        if add_coloured_regions:
            rectLHS = mpatches.Rectangle((0, -1 * self.amplitude_data[:, spin_site].max()), 5.75,
                                         2 * self.amplitude_data[:, spin_site].max() + 0.375e-2, alpha=0.05,
                                         facecolor="grey", edgecolor=None, lw=0)
            rectMID = mpatches.Rectangle((5.751, -1 * self.amplitude_data[:, spin_site].max()), 3.249,
                                         2 * self.amplitude_data[:, spin_site].max() + 0.375e-2, alpha=0.25,
                                         facecolor="grey", edgecolor=None, lw=0)
            rectRHS = mpatches.Rectangle((9.0, -1 * self.amplitude_data[:, spin_site].max()), 6,
                                         2 * self.amplitude_data[:, spin_site].max() + 0.375e-2, alpha=0.5,
                                         facecolor="grey", edgecolor=None, lw=0)

            ax1.add_patch(rectLHS)
            ax1.add_patch(rectMID)
            ax1.add_patch(rectRHS)

        frequencies_precursors, fourier_transform_precursors = self._fft_data(
            self.amplitude_data[lower1_signal:upper1_signal, spin_site])
        frequencies_dsw, fourier_transform_dsw = self._fft_data(
            self.amplitude_data[lower2_signal:upper2_signal, spin_site])
        frequencies_eq, fourier_transform_eq = self._fft_data(
            self.amplitude_data[lower3_signal:convert_norm(xlim_max), spin_site])

        ax2.plot(frequencies_precursors, abs(fourier_transform_precursors), marker='', lw=1, color='#37782c',
                 markerfacecolor='black', markeredgecolor='black', label="Precursors", zorder=5)
        ax2.plot(frequencies_dsw, abs(fourier_transform_dsw), marker='', lw=1, color='#64bb6a',
                 markerfacecolor='black', markeredgecolor='black', label="Shockwave", zorder=2)
        ax2.plot(frequencies_eq, abs(fourier_transform_eq), marker='', lw=1, color='#9fd983',
                 markerfacecolor='black', markeredgecolor='black', label="Steady State", zorder=1)

        if annotate_precursors:
            frequencies_blob1, fourier_transform_blob1 = self._fft_data(
                self.amplitude_data[lower1_precursor:upper1_precursor, spin_site])
            frequencies_blob2, fourier_transform_blob2 = self._fft_data(
                self.amplitude_data[lower2_precursor:upper2_precursor, spin_site])
            frequencies_blob3, fourier_transform_blob3 = self._fft_data(
                self.amplitude_data[lower3_precursor:upper3_precursor, spin_site])

            ax2.plot(frequencies_blob1, abs(fourier_transform_blob1), marker='', lw=1, color='#37782c',
                     markerfacecolor='black', markeredgecolor='black', ls=':')
            ax2.plot(frequencies_blob2, abs(fourier_transform_blob2), marker='', lw=1, color='#37782c',
                     markerfacecolor='black', markeredgecolor='black', ls='--')
            ax2.plot(frequencies_blob3, abs(fourier_transform_blob3), marker='', lw=1, color='#37782c',
                     markerfacecolor='black', markeredgecolor='black', ls='-.')

            arrow_ax2_props = {"arrowstyle": '-|>', "connectionstyle": "angle3,angleA=0,angleB=90", "color": "black"}
            ax2.annotate('P1', xy=(26, 1.8e1), xytext=(34.1, 2.02e2), va='center', ha='center',
                         arrowprops=arrow_ax2_props, fontsize=self.smaller_size)
            ax2.annotate('P2', xy=(48.78, 4.34e0), xytext=(56.0, 5.37e1), va='center', ha='center',
                         arrowprops=arrow_ax2_props, fontsize=self.smaller_size)
            ax2.annotate('P3', xy=(78.29, 1.25e0), xytext=(83.9, 5.5), va='center', ha='center',
                         arrowprops=arrow_ax2_props, fontsize=self.smaller_size)

        ax2.legend(ncol=1, loc='upper right', fontsize=self.tiny_size, frameon=False, fancybox=True, facecolor=None,
                   edgecolor=None,
                   bbox_to_anchor=[0.975, 0.95], bbox_transform=ax2.transAxes)

        for ax in [ax1, ax2]:
            ax.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=1.9999)
            ax.set_axisbelow(False)

        self.fig.subplots_adjust(wspace=1, hspace=0.35)

        if interactive_plot:
            # For interactive plots
            def mouse_event(event: Any):
                print(f'x: {event.xdata} and y: {event.ydata}')

            self.fig.canvas.mpl_connect('button_press_event', mouse_event)
            self.fig.tight_layout()  # has to be here
            plt.show()
        else:
            self.fig.savefig(f"{self.output_filepath}_site{spin_site}_tv0.png", bbox_inches="tight")


def create_time_variation3(self, interactive_plot=False, use_inset_1=True, use_lower_plot=False):
    """
    LEGACY METHOD FROM PAPERFIGURES. WILL NOT RUN IN THIS MODULE. Plot the Heaviside Fn and Disp Reln.

    One should ensure that the site being plotted is not inside either of the driving- or damping-regions.

    :return: Saves a .png image to the designated output folder.
    """

    if self._fig is None:
        self._fig = plt.figure(figsize=(4.5, 3.375))

    num_rows = 2
    num_cols = 3
    if use_lower_plot:
        ax1 = plt.subplot2grid((num_rows, num_cols), (0, 0), rowspan=int(num_rows / 2),
                               colspan=num_cols, fig=self._fig)
    else:
        ax1 = plt.subplot2grid((num_rows, num_cols), (0, 0), rowspan=int(num_rows),
                               colspan=num_cols, fig=self._fig)

    SAMPLE_RATE = int(5e2)  # Number of samples per nanosecond

    DURATION = int(2)  # Nanoseconds
    FREQUENCY = int(8)  # GHz

    # Number of samples in normalized_tone

    # def generate_sine_wave(freq, sample_rate, duration, delay_start=0, only_delay=False):
    #     delay = int(sample_rate * delay_start)
    #     t = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    #     y_1 = np.zeros(delay)
    #     y_2 = np.sin((2 * np.pi * freq) * t[delay:])
    #     y_con = np.concatenate((y_1, y_2))
    #
    #     if only_delay:
    #         start = delay
    #         sample_rate = int(sample_rate * delay_start)
    #         end = int(sample_rate * duration)
    #         return t[start:end], y_con[start:end]
    #
    #     return t, y_con

    def generate_sine_wave(freq, sample_rate, duration, delay_start=0.0, only_delay=False):

        if only_delay:
            num_samples = int(sample_rate * (duration - delay_start))
            sample_rate = int(num_samples / (duration - delay_start))

            t = np.linspace(delay_start, duration, num_samples, endpoint=False)
            y = np.sin((2 * np.pi * freq) * t)

            return t, y, sample_rate, num_samples

        else:
            delay = int(sample_rate * delay_start)
            num_samples = int(sample_rate * duration)

            t = np.linspace(0, duration, num_samples, endpoint=False)
            y_1 = np.zeros(delay)
            y_2 = np.sin((2 * np.pi * freq) * t[delay:])
            y_con = np.concatenate((y_1, y_2))

            return t, y_con, sample_rate, num_samples

    # Generate a 15 GHz sine wave that lasts for 5 seconds
    time_instant, signal_instant, sample_rate_instant, num_samples_instant = generate_sine_wave(FREQUENCY,
                                                                                                SAMPLE_RATE,
                                                                                                DURATION, 0, False)
    time_delay, signal_delay, sample_rate_delay, num_samples_delay = generate_sine_wave(FREQUENCY, SAMPLE_RATE,
                                                                                        DURATION, 1.0, False)
    time_delay2, signal_delay2, sample_rate_delay2, num_samples_delay2 = generate_sine_wave(FREQUENCY, SAMPLE_RATE,
                                                                                            DURATION, 1.0, True)

    time_instant_fft, signal_instant_fft = rfftfreq(num_samples_instant, 1 / sample_rate_instant), rfft(
        signal_instant)
    time_delay_fft, signal_delay_fft = rfftfreq(num_samples_delay, 1 / sample_rate_delay), rfft(signal_delay)
    time_delay_fft2, signal_delay_fft2 = rfftfreq(num_samples_delay2, 1 / sample_rate_delay2), rfft(signal_delay2)

    ax1.plot(time_delay_fft, np.abs(signal_delay_fft), marker='', lw=2.0, color='#ffb55a', markerfacecolor='black',
             markeredgecolor='black', label="1", zorder=1.2)
    ax1.plot(time_instant_fft, np.abs(signal_instant_fft), marker='', lw=1.5, ls='--', color='#64bb6a',
             markerfacecolor='black', markeredgecolor='black', label="0", zorder=1.3)
    ax1.plot(time_delay_fft2, np.abs(signal_delay_fft2), marker='', lw=2.0, ls='-.', color='red',
             markerfacecolor='black',
             markeredgecolor='black', label="1", zorder=1.2)

    # ax1.set(xlim=(5.001, 24.999), ylim=(1e-13, 1e4),
    #        xlabel="Frequency (GHz)", ylabel="Amplitude\n(arb. units)", yscale='log')

    ax1.set(xlim=(0.001, 15.999), ylim=(1e0, 1e4),
            xlabel="Frequency (GHz)", ylabel="Amplitude\n(arb. units)", yscale='log')

    # ax1.plot(x1, y1, lw=2, color='#0289F7', zorder=1.2)
    # ax1.plot(x2, y2, lw=2, ls='-', color='#64bb6a', zorder=1.1)
    # ax1.set(xlim=(0, 2), ylim=(-1, 1),
    #         xlabel="Time (ns)", ylabel="Amplitude\n(arb. units)")
    # self._tick_setter(ax1, 1.0, 0.25, 3, 2, is_fft_plot=False, yaxis_num_decimals=1, yscale_type='p')

    ax1.xaxis.labelpad = -2
    ax1.yaxis.labelpad = -0
    self._tick_setter(ax1, 5, 1, 4, 4, is_fft_plot=True)

    ########################################
    if use_inset_1:
        if use_lower_plot:
            ax1_inset = inset_axes(ax1, width=1.3, height=0.72, loc="upper right", bbox_to_anchor=[0.995, 1.175],
                                   bbox_transform=ax1.transAxes)
        else:
            ax1_inset = inset_axes(ax1, width=1.3, height=0.72, loc="upper right", bbox_to_anchor=[0.995, 0.98],
                                   bbox_transform=ax1.transAxes)

        ax1_inset.plot(time_instant, signal_instant, lw=0.5, color='#64bb6a', zorder=1.1)
        ax1_inset.plot(time_delay, signal_delay, lw=0.5, ls='-.', color='#ffb55a', zorder=1.2)
        ax1_inset.plot(time_delay2, signal_delay2, lw=0.5, ls='--', color='red', zorder=1.3)

        ax1_inset.set(xlim=[0, 2], ylim=[-1, 1])
        ax1_inset.set_xlabel('Time (ns)', fontsize=self._fontsizes["tiny"])
        ax1_inset.yaxis.tick_left()
        ax1_inset.yaxis.set_label_position("left")
        ax1_inset.set_ylabel('Amplitude  \n(arb. units)  ', fontsize=self._fontsizes["tiny"], rotation=90,
                             labelpad=20)
        ax1_inset.tick_params(axis='both', labelsize=self._fontsizes["mini"])

        ax1_inset.patch.set_color("#f9f2e9")
        ax1_inset.yaxis.labelpad = 0
        ax1_inset.xaxis.labelpad = -0.5

        self._tick_setter(ax1_inset, 1.0, 0.25, 3, 2, is_fft_plot=False,
                          yaxis_num_decimals=0.1, yscale_type='p')

    ########################################

    if use_lower_plot:
        ax2 = plt.subplot2grid((num_rows, num_cols), (int(num_rows / 2), 0),
                               rowspan=num_rows, colspan=num_cols, fig=self._fig)

        # ax2.scatter(np.arange(1, len(freqs) + 1, 1), freqs, s=0.5)
        external_field, exchange_field = 0.1, 132.5
        gyromag_ratio = 28.8e9 * 2 * np.pi
        lattice_constant = np.sqrt(5.3e-17 / exchange_field)
        system_len = 10e-5  # metres
        max_len = round(system_len / lattice_constant)
        num_spins_array = np.arange(0, max_len, 1)
        wave_number_array = (num_spins_array * np.pi) / ((len(num_spins_array) - 1) * lattice_constant)
        freq_array = gyromag_ratio * (2 * exchange_field * (1 - np.cos(wave_number_array * lattice_constant))
                                      + external_field)

        hz_2_THz = 1e-12
        hz_2_GHz = 1e-9

        ax2.plot(wave_number_array * hz_2_GHz, freq_array * hz_2_THz, color='red', ls='-', label=f'Dataset 1')
        ax2.plot(wave_number_array * hz_2_GHz, gyromag_ratio * (
                external_field + exchange_field * lattice_constant ** 2 * wave_number_array ** 2) * hz_2_THz,
                 color='red', alpha=0.4, ls='-', label=f'Dataset 1')

        # These!!
        # ax2.scatter(np.arccos(1 - ((freqs2 / gamma - h_0) / (2 * h_ex))) / a, freqs2 / 1e12, s=0.5,
        # c='red', label='paper')
        # ax2.plot(k, gamma * (2 * h_ex * (1 - np.cos(k * a)) + h_0) / 1e12, color='red', ls='--', label=f'Kittel')

        ax2.set(xlabel="Wavenumber (nm$^{-1}$)", ylabel='Frequency (THz)', xlim=[0, 2], ylim=[0, 5])  # [0, 15.4]
        self._tick_setter(ax2, 2, 0.5, 3, 2, is_fft_plot=False,
                          xaxis_num_decimals=1, yaxis_num_decimals=0.0, yscale_type='p')

        # ax2.axhline(y=3.8, xmax=1.0, ls='--', lw=1, color='grey', zorder=0.9) # xmax=0.31
        # ax2.axhline(y=10.5, xmax=1.0, ls='--', lw=1, color='grey', zorder=0.9)  # xmax=0.68
        ax2.margins(0)
        ax2.xaxis.labelpad = -2

        # ax2.text(0.997, -0.13, r"$\mathrm{\dfrac{\pi}{a}}$",
        #          verticalalignment='center', horizontalalignment='center', transform=ax2.transAxes,
        #          fontsize=self._fontsizes["smaller"])

        # ax2.text(0.02, 0.88, r"$\mathcal{III}$", verticalalignment='center', horizontalalignment='left',
        #          transform=ax2.transAxes, fontsize=self._fontsizes["smaller"])
        # ax2.text(0.02, 0.5, r"$\mathcal{II}$", verticalalignment='center', horizontalalignment='left',
        #          transform=ax2.transAxes, fontsize=self._fontsizes["smaller"])
        # ax2.text(0.02, 0.12, r"$\mathcal{I}$", verticalalignment='center', horizontalalignment='left',
        #          transform=ax2.transAxes, fontsize=self._fontsizes["smaller"])
        #
        # ax2.text(0.91, 0.82, f"Decreasing", verticalalignment='center', horizontalalignment='center',
        #          transform=ax2.transAxes, fontsize=self._fontsizes["tiny"])
        # ax2.text(0.60, 0.425, f"Linear", verticalalignment='center', horizontalalignment='center',
        #          transform=ax2.transAxes, fontsize=self._fontsizes["tiny"])
        # ax2.text(0.41, 0.12, f"Increasing", verticalalignment='center', horizontalalignment='center',
        #          transform=ax2.transAxes, fontsize=self._fontsizes["tiny"])

        # arrow_ax2_props1 = {"arrowstyle": '-|>', "connectionstyle": "arc3,rad=0.075", "color": "black"}
        # arrow_ax2_props2 = {"arrowstyle": '-|>', "connectionstyle": "arc3,rad=0.0", "color": "black"}
        # arrow_ax2_props3 = {"arrowstyle": '-|>', "connectionstyle": "arc3,rad=-0.075", "color": "black"}
        # ax2.annotate('', xy=(1.665, 2.961), xytext=(1.147, 1.027), va='center', ha='center',
        #              arrowprops=arrow_ax2_props1, fontsize=self._fontsizes["tiny"], transform=ax2.transAxes)
        # ax2.annotate('', xy=(3.058, 9.406), xytext=(2.154, 5.098), va='center', ha='center',
        #              arrowprops=arrow_ax2_props2, fontsize=self._fontsizes["tiny"], transform=ax2.transAxes)
        # ax2.annotate('', xy=(4.155, 13.213), xytext=(3.553, 11.342), va='center', ha='center',
        #              arrowprops=arrow_ax2_props3, fontsize=self._fontsizes["tiny"], transform=ax2.transAxes)

        ########################################
        use_lower_subplot = False
        if use_lower_subplot:
            ax2_inset = inset_axes(ax2, width=1.9, height=0.8, loc="lower right", bbox_to_anchor=[0.99, 0.02],
                                   bbox_transform=ax2.transAxes)
            D_b = 5.3e-17
            a1 = lattice_constant
            a2 = np.sqrt(D_b / 132.5)

            # j_to_meV = 6.24150934190e21
            num_spins_array1 = np.arange(0, 5000, 1)
            num_spins_array2 = np.arange(0, 15811, 1)
            wave_number_array1 = (num_spins_array1 * np.pi) / ((len(num_spins_array1) - 1) * a1)
            wave_number_array2 = (num_spins_array2 * np.pi) / ((len(num_spins_array2) - 1) * a2)

            ax2_inset.plot(wave_number_array1 * hz_2_GHz,
                           (D_b * 2 * gyromag_ratio) * wave_number_array1 ** 2 * hz_2_THz, lw=1.5, ls='--',
                           color='purple',
                           label='$a=0.2$ nm',
                           zorder=1.3)
            ax2_inset.plot(wave_number_array2 * hz_2_GHz,
                           (D_b * 2 * gyromag_ratio) * wave_number_array2 ** 2 * hz_2_THz, lw=1.5, ls='-',
                           label='$a=0.63$ nm',
                           zorder=1.2)

            ax2_inset.set_xlabel('Wavenumber (nm$^{-1}$)', fontsize=self._fontsizes["tiny"])
            ax2_inset.set_xlim(0, 2)
            ax2_inset.set_ylim(0, 10)
            ax2_inset.xaxis.tick_top()
            ax2_inset.xaxis.set_label_position("top")
            ax2_inset.yaxis.set_label_position("left")
            ax2_inset.set_ylabel('Frequency\n(THz)', fontsize=self._fontsizes["tiny"], rotation=90, labelpad=20)
            ax2_inset.tick_params(axis='both', labelsize=self._fontsizes["tiny"])
            ax2.margins(0)

            ax2_inset.patch.set_color("#f9f2e9")
            ax2_inset.yaxis.labelpad = 5
            ax2_inset.xaxis.labelpad = 2.5

            # self._tick_setter(ax2_inset, 2.5, 0.5, 3, 2, is_fft_plot=False)
            ax2_inset.ticklabel_format(axis='y', style='plain')
            ax2_inset.legend(fontsize=self._fontsizes["tiny"], frameon=False)

        ########################################

        # ax1.text(0.025, 0.88, f"(a)", verticalalignment='center', horizontalalignment='left',
        #          transform=ax1.transAxes, fontsize=self._fontsizes["smaller"])

        # ax2.text(0.975, 0.12, f"(b)",
        #         verticalalignment='center', horizontalalignment='right', transform=ax2.transAxes,
        #         fontsize=self._fontsizes["smaller"])

    for ax in [ax1]:
        ax.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=1.99)
        ax.set_axisbelow(False)

    self._fig.subplots_adjust(wspace=1, hspace=0.35)

    if interactive_plot:
        # For interactive plots
        def mouse_event(event: Any):
            print(f'x: {event.xdata} and y: {event.ydata}')

        self._fig.canvas.mpl_connect('button_press_event', mouse_event)
        self._fig.tight_layout()  # has to be here
        plt.show()
    else:
        self._fig.savefig(f"{self.output_filepath}_dispersion_tv3.png", bbox_inches="tight")
