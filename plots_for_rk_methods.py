#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard modules (common)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys as sys

# Third party modules (uncommon)
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as ticker


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


# -------------------------------------- Useful to look at shockwaves. Three panes -------------------------------------
def three_panes(amplitude_data, key_data, list_of_spin_sites, sites_to_compare=None):
    """
    Plots a graph

    :param Any amplitude_data: Array of magnitudes of the spin's magnetisation at each moment in time for each spin
                               site.
    :param dict key_data: All key simulation parameters imported from csv file.
    :param list list_of_spin_sites: Spin sites that were simulated.
    :param list[int] sites_to_compare: Optional. User-defined list of sites to plot.
    """
    key_data['maxSimTime'] *= 1e9

    subplot_labels = create_plot_labels(list_of_spin_sites, key_data['drivingRegionLHS'], key_data['drivingRegionLHS'])

    time_values = np.linspace(0, key_data['maxSimTime'], int(key_data['stopIterVal']) + 1)

    fig = plt.figure(figsize=(12, 12))
    plt.suptitle('Comparison of $M_x$ Values At\nDifferent Spin Sites', size=24)

    # Three subplots which are each a different pane
    plot_pane_1 = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=2)  # Top pane for comparison of multiple datasets
    plot_pane_2 = plt.subplot2grid((4, 4), (2, 0), colspan=2, rowspan=2)  # Bottom left pane to show any single dataset
    plot_pane_3 = plt.subplot2grid((4, 4), (2, 2), colspan=2, rowspan=2)  # Bottom right pane to track final spin site

    for ax in [plot_pane_1, plot_pane_2, plot_pane_3]:
        # Convert all x-axis tick labels to scientific notation on all panes
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    spin_sites_to_plot = []
    for i in range(1, len(amplitude_data[0])):
        # Exclude 0 as the first column of data will always be time values.
        # The variable spin_sites_to_plot is useful when building the software, as sometimes not all sites are wanted.
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


# ------------------------------------------ FFT and Signal Analysis Functions -----------------------------------------
def fft_and_signal_four(time_data, amplitude_data, spin_site, simulation_params, filename):
    """
    Plot the magnitudes of the magnetic moment of a spin site against time, as well as the FFTs, over four subplots.

    :param float time_data: The time data as an array which must be of the same length as y_axis_data.
    :param amplitude_data: The magnitudes of the spin's magnetisation at each moment in time.
    :param dict simulation_params: Key simulation parameters.
    :param int spin_site: The spin site being plotted.

    :return: A figure containing four sub-plots.
    """
    # Find maximum time in [ns] to the nearest whole [ns], then find how large shaded region should be.
    temporal_xlim = np.round(simulation_params['stopIterVal'] * simulation_params['stepsize'] * 1e9, 1)
    x_scaling = 0.2
    t_shaded_xlim = temporal_xlim * x_scaling

    plot_set_params = {0: {"title": "Full Simulation", "xlabel": "Time [ns]", "ylabel": "Amplitude [arb.]",
                           "xlim": (0, temporal_xlim), "ylim": (-1.0, 1.0)},
                       1: {"title": "Shaded Region", "xlabel": "Time [ns]", "xlim": (0, t_shaded_xlim)},
                       2: {"title": "Showing All Artefacts", "xlabel": "Frequency [GHz]", "ylabel": "Amplitude [arb.]",
                           "yscale": 'log', "xlim": (0, 30)},
                       3: {"title": "Shaded Region", "xlabel": "Frequency [GHz]", "yscale": 'log', "xlim": (0, 5)}}

    fig = plt.figure(figsize=(12, 12), constrained_layout=True, )
    fig.suptitle(f"Data from Spin Site #{spin_site}")

    # create 2 rows of sub-figures
    all_sub_figs = fig.subfigures(nrows=2, ncols=1)

    for row, sub_fig in enumerate(all_sub_figs):
        if row == 0:
            sub_fig.suptitle(f"Temporal Data")

            axes = sub_fig.subplots(nrows=1, ncols=2)

            for i, ax in enumerate(axes, start=row * 2):
                custom_temporal_plot(time_data, amplitude_data, ax=ax, which_subplot=i,
                                     plt_set_kwargs=plot_set_params[i], xlim_scaling=x_scaling)

        else:
            sub_fig.suptitle(f"FFT Data")

            axes = sub_fig.subplots(nrows=1, ncols=2)

            for i, ax in enumerate(axes, start=row * 2):
                custom_fft_plot(amplitude_data, ax=ax, which_subplot=i,
                                plt_set_kwargs=plot_set_params[i], simulation_params=simulation_params)

    fig.savefig(f"{filename}_{spin_site}.png")
    plt.show()


def custom_temporal_plot(time_data, amplitude_data, plt_set_kwargs, which_subplot, xlim_scaling=0.2, ax=None):
    """
    Custom plotter for each temporal signal within 'fft_and_signal_four'.

    Description.

    :param float time_data: The time data as an array which must be of the same length as y_axis_data.
    :param amplitude_data: The magnitudes of the spin's magnetisation at each moment in time.
    :param dict plt_set_kwargs: **kwargs for the ax.set() function.
    :param int which_subplot: Number of subplot (leftmost is 0). Used to give a subplot any special characteristics.
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
        ax.axvspan(0, plt_set_kwargs['xlim'][1] * xlim_scaling, color='#DC143C', alpha=0.2, lw=0)

    return ax


def custom_fft_plot(amplitude_data, plt_set_kwargs, which_subplot, simulation_params, ax=None):
    """
    Custom plotter for each FFT within 'fft_and_signal_four'.

    Description.

    :param amplitude_data: The magnitudes of the spin's magnetisation at each moment in time.
    :param dict plt_set_kwargs: **kwargs for the ax.set() function.
    :param int which_subplot: Number of subplot (leftmost is 0). Used to give a subplot any special characteristics.
    :param dict simulation_params: Key simulation parameters from csv header.
    :param ax: Axes data from plt.subplots(). Used to define subplot and figure behaviour.

    :return: The subplot information required to be plotted in the main figure environment."""
    frequencies, fourierTransform, natural_frequency, driving_freq = fft_data(amplitude_data, simulation_params)

    drivingFreq_Hz = simulation_params['drivingFreq'] / 1e9
    if ax is None:
        ax = plt.gca()

    # Must be abs(FFTransform) to make sense!
    ax.plot(frequencies, abs(fourierTransform),
            marker='o', lw=1, color='red', markerfacecolor='black', markeredgecolor='black')
    ax.set(**plt_set_kwargs)

    ax.axvline(x=drivingFreq_Hz, label=f"Driving. {drivingFreq_Hz:2.2f}", color='green')

    if which_subplot == 2:
        ax.axvspan(0, 5, color='#DC143C', alpha=0.2, lw=0)
        # If at a node, then 3-wave generation may be occurring. This loop plots that location.
        triple_wave_gen_freq = drivingFreq_Hz * 3
        ax.axvline(x=triple_wave_gen_freq, label=f"T.W.G. {triple_wave_gen_freq:2.2f}", color='purple')
    else:
        # This should be an eigenfrequency
        ax.axvline(x=natural_frequency, label=f"Natural. {natural_frequency:2.2f}")

    ax.legend(loc=0, frameon=True, fancybox=True, facecolor='white', edgecolor='white',
              title='Freq. List [GHz]', fontsize=12)

    ax.grid(color='white')

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
    core_values = {"gamma": 29.2,
                   "hz_to_ghz": 1e-9}

    # Data in file header is in [Hz] by default.
    driving_Freq_GHz = simulation_params['drivingFreq'] * core_values["hz_to_ghz"]

    # This is the (first) natural frequency of the system, corresponding to the first eigenvalue. Change as needed to
    # add other markers to the plot(s)
    natural_freq = core_values['gamma'] * simulation_params['biasField']

    # Calculate FFT parameters
    timeInterval = simulation_params['stepsize'] * simulation_params['stopIterVal']
    nSamples = simulation_params['numberOfDataPoints']
    dt = timeInterval / nSamples  # Or multiply the stepsize by the number of iterations between data recordings

    # Compute the FFT
    fourierTransform = np.fft.fft(amplitude_data)  # Normalize amplitude after taking FFT
    fourierTransform = fourierTransform[range(int(nSamples / 2))]  # Exclude sampling frequency, and negative values
    frequencies = (np.arange(int(nSamples / 2)) / (dt * nSamples)) * core_values["hz_to_ghz"]

    return frequencies, fourierTransform, natural_freq, driving_Freq_GHz


# --------------------------------------------- Continually plot eigenmodes --------------------------------------------
def eigenmodes(mx_data, my_data, eigenvalues_data, file_name):
    """
    Plot the spin wave modes (eigenmodes) of a given system until a keyword is entered.

    Allows the user to plot as may eigenmodes as they would like; one per figure. This function is primarily used to
    replicate Fig. 1 from macedo2021breaking. The user of keywords within this function also allow the user to plot
    the 'generalised fourier coefficients' of a system. This is mainly used to replicate Figs 4.a & 4.d of the same
    paper.

    :params Any mx_data: The magnetic moments (x-component) at each iteration for all spin sites.
    :params Any my_data: The magnetic moments (y-component) at each iteration for all spin sites.
    :params Any eigenvalues_data: List of eigenvalues for the system.

    :return: Each iteration within this function shows a plot.

    """
    print('---------------------------------------------------------------------------------------')
    print('''
          This will plot the eigenmodes of the selected data. Input the requested 
          modes as single values, or as a space-separated list. Enter any 
          non-int to exit, or type EXIT. If you wish to plot the generalized 
          Fourier coefficients, type FOURIER.

          Note: None of the keywords are case-sensitive.
          ''')
    print('---------------------------------------------------------------------------------------')

    upper_limit_mode = len(mx_data)  # The largest mode which can be plotted for the given data.
    previously_plotted_modes = []  # Tracks what mode(s) the user plotted in their last inputs.

    # Take in first user input. Assume input is valid, until an error is raised.
    modes_to_plot = input("Enter mode(s) to plot: ").split()
    has_valid_modes = True

    while True:
        # Plots eigenmodes as long as the user enters a valid input.
        for test_mode in modes_to_plot:

            try:
                if not 1 <= int(test_mode) <= upper_limit_mode:
                    # Check mode is in the range held by the dataset
                    print(f"That mode does not exist. Please select a mode between 1 & {upper_limit_mode}.")
                    break

                if set(previously_plotted_modes) & set(modes_to_plot):
                    # Check if mode has already been plotted. Cast to set as they don't allow duplicates.
                    has_valid_modes = False
                    print(f"You have already printed a mode in {previously_plotted_modes}. Please make another choice.")
                    break

            except ValueError:
                # If the current tested mode is within the range, then it is either a keyword, or invalid.
                if test_mode.upper() == 'EXIT':
                    sys.exit(0)

                elif test_mode.upper() == 'FOURIER':
                    generalised_fourier_coefficients(mx_data, eigenvalues_data, file_name)
                    has_valid_modes = False
                    break

                BreakQuery = input("Do you want to continue plotting modes? Y/N: ").upper()
                while True:

                    if BreakQuery == 'Y':
                        has_valid_modes = False  # Prevents plotting of incorrect input, and allows user to retry.
                        break

                    elif BreakQuery == 'N':
                        print("Exiting program...")
                        exit(0)

                    else:
                        while BreakQuery not in 'YN':
                            BreakQuery = input("Do you want to continue plotting modes? Y/N: ").upper()

            if has_valid_modes:
                plot_single_eigenmode(int(test_mode), mx_data, my_data, eigenvalues_data)

            else:
                has_valid_modes = True  # Reset condition
                continue

        previously_plotted_modes = modes_to_plot  # Reassign the current modes to be the previous attempts.
        modes_to_plot = (input("Enter mode(s) to plot: ")).split()  # Take in the new set of inputs.


def generalised_fourier_coefficients(amplitude_mx_data, eigenvalues_angular, file_name, use_defaults=True):
    """
    Plot coefficients across a range of eigenfrequencies to find frequencies of strong coupling.

    The 'generalised fourier coefficients' indicate the affinity of spins to couple to a particular driving field
    profile. If a non-linear exchange was used, then the rightward and leftward profiles will look different. This
    information can be used to deduce when a system allows for:

        * rightward only propagation.
        * leftward only propagation.
        * propagation in both directions.
        * propagation in neither direction.

    :param amplitude_mx_data: The magnetic moments (x-component) at each iteration for all spin sites.
    :param eigenvalues_angular:  Eigenvalues, derived from Kittel's equations, of the system.
    :param str file_name: Name of file that data is coming from. Used to name saved plot.
    :param bool use_defaults: Use preset parameters to reduce user input, and speed-up running of simulations.

    :return: Single figure plot.
    """
    number_of_spins = len(amplitude_mx_data[:, 0])

    # use_defaults is a testing flag to speed up the process of running sims.
    if use_defaults:
        step = 10
        lower = 120
        upper = 180
        width_ones = 0.05
        width_zeros = 0.95

    else:
        step, lower, upper = int(input("Enter step, lower & upper: "))
        width_ones = int(input("Enter width of driving region: "))
        width_zeros = 1 - width_ones

    # Raw data is in units of 2*Pi (angular frequency), so we need to convert back to frequency.
    eigenvalues_angular = np.append([0], eigenvalues_angular)
    eigenvalues = [eigval / (2 * np.pi) for eigval in eigenvalues_angular]
    x_axis_limits = range(0, number_of_spins)

    # Find widths of each component of the driving regions.
    g_ones = [1] * int(number_of_spins * width_ones)
    g_zeros = [0] * int(number_of_spins * width_zeros)

    # g is the driving field profile along the axis where the drive is applied. My simulations all have the
    # drive along the x-axis, hence the name 'gx'.
    gx_LHS = g_ones + g_zeros
    gx_RHS = g_zeros + g_ones

    fourier_coefficents_lhs = []
    fourier_coefficents_rhs = []

    for i in range(0, number_of_spins):
        # Select an eigenvector, and take the dot-product to return the coefficient of that particular mode.
        fourier_coefficents_lhs.append(np.dot(gx_LHS, amplitude_mx_data[:, i]))
        fourier_coefficents_rhs.append(np.dot(gx_RHS, amplitude_mx_data[:, i]))

    # Normalise the arrays of coefficients.
    fourier_coefficents_lhs = fourier_coefficents_lhs / np.linalg.norm(fourier_coefficents_lhs)
    fourier_coefficents_lhs = fourier_coefficents_lhs / np.linalg.norm(fourier_coefficents_lhs)

    # Plotting functions. Left here as nothing else will use this functionality.
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle(r'Overlap Values ($\mathcal{O}_{j}$)'f' for {file_name}')
    plt.subplots_adjust(top=0.82)

    # Whichever ax is before the sns.lineplot statements is the one which holds the labels.
    sns.lineplot(x=x_axis_limits, y=np.abs(fourier_coefficents_lhs), lw=3, marker='o', label='Left', zorder=2)
    sns.lineplot(x=x_axis_limits, y=np.abs(fourier_coefficents_rhs), lw=3, color='r',
                 marker='o', label='Right', zorder=1)

    # Both y-axes need to match up, so it is clear what eigenmode corresponds to what eigenfrequency.
    ax.set(xlabel=r'Eigenfrequency ( $\frac{\omega_j}{2\pi}$ ) [GHz]', ylabel='Fourier coefficient [normalised]',
           xlim=[lower, upper], ylim=[0.0, 0.1],
           xticks=list(range(lower, upper + 1, step)),
           xticklabels=[float(i) for i in np.round(eigenvalues[lower:upper + 1:step], 1)])

    ax_mode = ax.twiny()  # Create second scale on the upper y-axis of the plot.
    ax_mode.set(xlabel=f'Eigenmode ($A_j$) for m$^x$ components',
                xlim=ax.get_xlim(),
                xticks=list(range(int(ax.get_xlim()[0]), int(ax.get_xlim()[1]) + 1, step)))

    ax.legend(loc=1, bbox_to_anchor=(0.975, 0.975),
              frameon=True, fancybox=True, facecolor='white', edgecolor='white',
              title='Propagation\n   Direction', fontsize=10)

    ax.grid(lw=2, ls='-')

    plt.show()


def plot_single_eigenmode(eigenmode, mx_data, my_data, eigenvalues_data, has_endpoints=True):
    """
    Plot a single eigenmode with the x- and y-axis magnetic moment components against spin site.

    :param int eigenmode: The eigenmode that is to be plotted.
    :param mx_data: 2D array of amplitudes of the mx components.
    :param my_data: 2D array of amplitudes of the my components.
    :param list eigenvalues_data: 1D array of all eigenvalues in system.
    :param bool has_endpoints: Allows for fixed nodes to be included on plot. Useful for visualisation purposes.

    :return: Outputs a single figure.

    """
    eigenmode += - 1  # To handle 'off-by-one' error, as first site is at mx_data[0]

    # Select single mode to plot from imported data.
    mx_mode = mx_data[:, eigenmode] * -1
    my_mode = my_data[:, eigenmode] * -1

    # Simulation parameters
    number_of_spins = len(mx_mode)
    driving_width = 0.05
    frequency = eigenvalues_data[eigenmode] / (2 * np.pi)  # Convert angular (frequency) eigenvalue to frequency [Hz].

    if has_endpoints:
        # 0-valued reflects the (P-1) and (N+1) end spins that act as fixed nodes for the system.
        mx_mode = np.append(np.append([0], mx_mode), [0])
        my_mode = np.append(np.append([0], my_mode), [0])

    # Generate plot
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    sns.lineplot(x=range(0, len(mx_mode)), y=mx_mode, marker='', ls=':', lw=3, label='Mx', zorder=2)
    sns.lineplot(x=range(0, len(my_mode)), y=my_mode, color='r', ls='-', lw=3, label='My', zorder=1)

    axes.set(title=f"Eigenmode[{eigenmode + 1}]",
             xlabel="Site Number", ylabel="Amplitude [arb.]",
             xlim=(0, number_of_spins), ylim=(-0.05, 0.05),
             xticks=np.arange(0, number_of_spins, np.floor(number_of_spins - 2) / 4))

    # Legend doubles as a legend (showing propagation direction), and the frequency [Hz] of the eigenmode.
    axes.legend(loc=1, bbox_to_anchor=(0.975, 0.975),
                frameon=True, fancybox=True, facecolor='white', edgecolor='white',
                title=f"Frequency [GHz]\n        {frequency:4.2f}\n\n    Propagation\n      Direction",
                fontsize=10)

    axes.axvspan(0, number_of_spins * driving_width, color='black', alpha=0.2)

    axes.grid(color='black', ls='--', alpha=0.1, lw=1)

    plt.show()


# ----------------------------------------------------- Animation -----------------------------------------------------
def animate_plot(key_data, amplitude_data, path_to_save_gif, file_name):
    """
    This function has not been tested, so use at your own risk!

    :param dict key_data: All key simulation parameters imported from csv file.
    :param Any amplitude_data: Magnetic moment values for site that is being animated.
    :param str path_to_save_gif: Absolute path to save location. Must end is a \\ or /, and not include the filename.
    :param str file_name: Filename of the output gif.

    :return: Saves a gif to the target location.
    """
    number_of_images = 100
    step = key_data['stopIterVal'] / number_of_images
    number_of_spins = key_data['numSpins']

    fig = plt.figure()
    ax = plt.axes(xlim=(0, number_of_spins))
    ax.xaxis.set(major_locator=ticker.MultipleLocator(number_of_spins * 0.25),
                 minor_locator=ticker.MultipleLocator(number_of_spins * 0.25 / 4))
    ax.yaxis.set(major_locator=ticker.MaxNLocator(nbins=5, prune='lower'),
                 minor_locator=ticker.AutoMinorLocator())
    plt.suptitle(f"Animation of data from {file_name}")

    line, = ax.plot([], [], linestyle='-', lw=1)
    label = ax.text(0.65, 0.90, '', transform=ax.transAxes)
    ax.set(xlabel="Spin Site", ylabel="Amplitude [arb.]")

    def initialise_animation():
        line.set_data([], [])
        return line,

    def animate_animation(i):
        spinx = np.arange(0, number_of_spins + 1)
        line.set_xdata(spinx)
        line.set_ydata(amplitude_data[i, :])
        label.set_text(f'iteration = {i * step:2.3e}')
        return line

    anim = FuncAnimation(fig, animate_animation, init_func=initialise_animation,
                         frames=key_data['stopIterVal'], interval=200, repeat=True, repeat_delay=400)

    anim.save(f"{path_to_save_gif}{file_name}.gif")
