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
def fft_and_signal_four(time_data, amplitude_data, spin_site):
    """
    Plot the magnitudes of the magnetic moment of a spin site against time, as well as the FFTs, over four subplots.

    :param float time_data: The time data as an array which must be of the same length as y_axis_data.
    :param amplitude_data: The magnitudes of the spin's magnetisation at each moment in time.
    :param int spin_site: The spin site being plotted.

    :return: A figure containing four sub-plots.
    """
    plot_set_params = {0: {"title": "Full Simulation", "xlabel": "Time [ns]", "ylabel": "Amplitude [normalised]",
                           "xlim": (0, 40)},
                       1: {"title": "Shaded Region", "xlabel": "Time [ns]", "xlim": (0, 5)},
                       2: {"title": "Showing All Artefacts", "xlabel": "Frequency [GHz]", "ylabel": "Amplitude [arb.]",
                           "xlim": (0, 30), "yscale": 'log'},
                       3: {"title": "Shaded Region", "xlabel": "Frequency [GHz]", "xlim": (0, 5),
                           "yscale": 'log'}}

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
                                     plt_set_kwargs=plot_set_params[i])

        else:
            sub_fig.suptitle(f"FFT Data")

            axes = sub_fig.subplots(nrows=1, ncols=2)

            for i, ax in enumerate(axes, start=row * 2):
                custom_fft_plot(amplitude_data, ax=ax, which_subplot=i,
                                plt_set_kwargs=plot_set_params[i])

    plt.show()


def custom_temporal_plot(time_data, amplitude_data, plt_set_kwargs, which_subplot, ax=None):
    """Custom plotter for each temporal signal."""
    if ax is None:
        ax = plt.gca()

    ax.plot(time_data, amplitude_data)
    ax.set(**plt_set_kwargs)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    if which_subplot == 0:
        ax.axvspan(0, 5, color='#DC143C', alpha=0.2, lw=0)

    return ax


def custom_fft_plot(amplitude_data, plt_set_kwargs, which_subplot, ax=None):
    """Custom plotter for each FFT."""
    frequencies, FFTransform, natural_frequency, driving_freq = fft_data(amplitude_data)

    if ax is None:
        ax = plt.gca()

    ax.plot(frequencies, abs(FFTransform),
            marker='o', lw=1, color='red', markerfacecolor='black', markeredgecolor='black')
    ax.set(**plt_set_kwargs)

    if which_subplot == 2:
        ax.axvspan(0, 5, color='#DC143C', alpha=0.2, lw=0)
    else:
        ax.axvline(x=natural_frequency, label=f"Natural. {natural_frequency:2.2f}")
        ax.axvline(x=driving_freq, label=f"Driving. {driving_freq}", color='green')

        ax.legend(loc=0, frameon=True, fancybox=True, facecolor='white', edgecolor='white',
                  title='Freq. List [GHz]', fontsize=12)

    ax.grid(color='white')

    return ax


def fft_data(amplitude_data):
    """
    Computes the FFT transform of a given signal, and also outputs useful data such as key frequencies.

    :param amplitude_data: Magnitudes of magnetic moments for a spin site

    :return: A tuple containing the frequencies [0], FFT [1] of a spin site. Also includes the  natural frequency
    (1st eigenvalue) [2], and driving frequency [3] for the system.
    """
    # Data should be added here from the simulation. Reader function will be included in the future.
    sim_params = {"stepsize": 2.857e-13,
                  "total_iterations": 141 * 1e3,
                  "gamma": 29.2,
                  "H_static": 0.1,
                  "freq_drive": 3.5,
                  "total_datapoints": 1e7,
                  "hz_to_ghz": 1e-9}

    # This is the first natural frequency of the system, corresponding to the first eigenvalue.
    natural_freq = sim_params['gamma'] * sim_params['H_static']

    # Calculate FFT parameters
    timeInterval = sim_params['stepsize'] * sim_params['total_iterations']
    nSamples = len(amplitude_data)
    dt = timeInterval / nSamples  # Or multiply the stepsize by the number of iterations between data recordings

    # Compute the FFT
    fourierTransform = np.fft.fft(amplitude_data)  # Normalize amplitude after taking FFT
    fourierTransform = fourierTransform[range(int(nSamples / 2))]  # Exclude sampling frequency, and negative values
    frequencies = (np.arange(int(nSamples / 2)) / (dt * nSamples)) * sim_params["hz_to_ghz"]

    return frequencies, fourierTransform, natural_freq, sim_params['freq_drive']


# --- Continually plots eigenmodes until exit is entered---
def main2(mxvals, myvals, eigenvals):

    maxMode = len(mxvals)
    prev_choice_list = []

    print('---------------------------------------------------------------------------------------')
    print('''
          This will plot the eigenmodes of the selected data. Input the requested 
          modes as single values, or as a space-separated list []. Enter any 
          non-int to exit, or type EXIT. If you wish to plot the generalized 
          Fourier coefficients, type FOURIER.

          NB: None of the keywords are case sensitive
          ''')
    print('---------------------------------------------------------------------------------------')
    userInputlist = input("Enter mode(s) to plot: ").split()
    warn = False
    while True:
        for inputMode in userInputlist:
            try:
                userMode = int(inputMode)
                if not 1 <= userMode <= maxMode:
                    print(f"That mode does not exist. Please select a mode between 1 & {maxMode}.")
                    break
                # for prev_choice in prev_choice_list:
                #     if prev_choice in userInputlist:
                #         warn=True
                #     if warn:
                #         print(f"You just plotted mode {prev_choice}. Please make another choice.")
                #         break
                #     else:
                #         continue
                if set(prev_choice_list) & set(userInputlist):
                    warn = True
                    print(f"You have already printed a mode in {prev_choice_list}. Please make another choice.")
                    break
                else:
                    warn = False

            except ValueError:

                if inputMode.upper() == 'EXIT':
                    sys.exit(0)

                elif inputMode.upper() == 'FOURIER':
                    user_step = int(input("Enter the step size for the plot: "))
                    plot_fourier(user_step, mxvals, myvals, eigenvals)

                BreakQuery = input("Do you want to continue plotting modes? Y/N: ").upper()

                if BreakQuery == 'Y':
                    # userInputlist = input("Enter mode(s) to plot: ")
                    break

                elif BreakQuery == 'N':
                    print("Exiting program.")
                    sys.exit(0)

            if not warn:
                plot_modes(userMode, mxvals, myvals, eigenvals)
                warn = False
            else:
                warn = False
                continue
        prev_choice_list = userInputlist

        userInputlist = (input("Enter mode(s) to plot: ")).split()


def plot_modes(mode, mxvals, myvals, eigenvals):

    mode += - 1

    selected_mode_mx = mxvals[:, mode] * -1
    selected_mode_my = myvals[:, mode] * -1

    eigval = eigenvals[mode]

    selected_mode_mx = np.append(np.append([0], selected_mode_mx), [0])
    selected_mode_my = np.append(np.append([0], selected_mode_my), [0])
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    sns.lineplot(x=range(0, len(selected_mode_mx)), y=selected_mode_mx, marker='', ls=':', lw=3, label='Mx', zorder=2)
    sns.lineplot(x=range(0, len(selected_mode_my)), y=selected_mode_my, color='r', ls='-', lw=3, label='My', zorder=1)

    axes.set(title=f"Eigenmode[{mode + 1}]",
             xlabel="Site Number", ylabel="Amplitude (arb)",
             xlim=(0, len(selected_mode_mx)), ylim=(-0.035, 0.035),
             xticks=np.arange(0, len(selected_mode_mx), np.floor((len(selected_mode_mx) - 2) / 4)))

    legend = axes.legend(loc=1, bbox_to_anchor=(0.975, 0.975),
                         frameon=True, fancybox=True, facecolor='white', edgecolor='white',
                         title='Propagation\n   Direction', fontsize=10)

    axes.text(len(selected_mode_mx) * 0.84, (1 / 3) * axes.get_ylim()[1],
              f"Frequency\n{eigval / (2 * np.pi):4.2f} [GHz]", style='italic', fontsize=12,
              bbox={'facecolor': 'white', 'alpha': 1.0, 'pad': 5, 'edgecolor': 'white'})

    axes.axvspan(0, 1, color='black', alpha=0.2)
    axes.axvspan(len(selected_mode_mx) - 2, len(selected_mode_mx) - 1, color='black', alpha=0.2)

    axes.grid(color='black', ls='--', alpha=0.1, lw=1)
    plt.show()


def plot_fourier(step, mxvals, myvals, eigenval):
    ##############################################################################
    # Sets global conditions including font sizes, ticks and sheet style
    # Sets various font size. fsize: general text. lsize: legend. tsizeL title. ticksize: numbers next to ticks
    fsize = 18
    lsize = 12
    tsize = 24
    ticksize = 14

    # sets the tick direction. Options: 'in', 'out', 'inout'
    t_dir = 'in'
    # sets the the tick size(s) and tick width(w) for the major and minor axes of all plots
    t_maj_s = 10
    t_min_s = 5
    t_maj_w = 1.2
    t_min_w = 1

    # updates rcParams of the selected style with my preferred options for these plots. Feel free to change
    plt.rcParams.update({'axes.titlesize': tsize, 'axes.labelsize': fsize, 'font.size': fsize, 'legend.fontsize': lsize,
                         'xtick.labelsize': ticksize, 'ytick.labelsize': ticksize, 'legend.title_fontsize': lsize + 2,
                         'axes.edgecolor': 'black', 'axes.linewidth': 1,
                         "xtick.bottom": False, "ytick.left": False, 'xtick.top': False, 'ytick.right': False,
                         'xtick.color': 'white', 'ytick.color': 'white', 'ytick.labelcolor': 'black',
                         'xtick.labelcolor': 'black',
                         'text.color': 'black',
                         'xtick.major.size': t_maj_s, 'xtick.major.width': t_maj_w,
                         'xtick.minor.size': t_min_s, 'xtick.minor.width': t_min_w,
                         'ytick.major.size': t_maj_s, 'ytick.major.width': t_maj_w,
                         'ytick.minor.size': t_min_s, 'ytick.minor.width': t_min_w,
                         'xtick.direction': t_dir, 'ytick.direction': t_dir,
                         'axes.spines.top': False, 'axes.spines.bottom': False, 'axes.spines.left': False,
                         'axes.spines.right': False,
                         'figure.dpi': 300})

    ##############################################################################

    nspins = len(mxvals[:, 0])

    eigenval = np.append([0], eigenval)
    eigenvals = [val / (2 * np.pi) for val in eigenval]
    g_ones = [1] * int(nspins * 0.05)
    g_zeros = [0] * int(nspins * 0.95)

    # gx is the driving field profile.
    gx_LHS = g_ones + g_zeros
    gx_RHS = g_zeros + g_ones

    functLHS = []
    functRHS = []
    selected_mode_mx = None
    for n in range(0, nspins):
        # selects an eigenvector
        selected_mode_mx = mxvals[:, n]
        # finds the norm of the selected eigenvector
        norm = np.linalg.norm(selected_mode_mx)
        # calculates the normalised eigenvector
        normalised_selected_mode_mx = selected_mode_mx

        functLHS.append(np.dot(gx_LHS, normalised_selected_mode_mx))
        functRHS.append(np.dot(gx_RHS, normalised_selected_mode_mx))
    # import matplotlib.ticker as ticker
    for i in range(0, len(functLHS)):
        functLHS[i] *= 1 / np.dot(selected_mode_mx, selected_mode_mx)
        functRHS[i] *= 1 / np.dot(selected_mode_mx, selected_mode_mx)
    # [0 $\leq$j$\leq$80]
    fig, axes1 = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle(r'Overlap Values ($\mathcal{O}_{j}$) for 'f'{nspins} spins')
    plt.subplots_adjust(top=0.82)

    # lower = 150
    # upper = 230
    lower = int(input("Enter lower: "))
    upper = int(input("Enter upper: "))
    # r' [j $\in \mathbb{N}$: j$\leq$80]'
    sns.lineplot(x=range(0, nspins), y=np.abs(functLHS), lw=3, marker='o', label='Left', zorder=2)
    sns.lineplot(x=range(0, nspins), y=np.abs(functRHS), lw=3, color='r', marker='o', label='Right', zorder=1)

    axes2 = axes1.twiny()
    #
    axes1.set(xlabel=r'Eigenfrequency ( $\frac{\omega_j}{2\pi}$ ) [GHz]', ylabel='Fourier coefficient',
              xlim=[lower, upper], ylim=[1 / 40, 10 ** 0],
              yscale='log',
              xticks=list(range(lower, upper + 1, step)),
              xticklabels=[float(i) for i in np.round(eigenvals[lower:upper + 1:step], 1)])

    axes2.set(xlabel=f'Eigenmode ($A_j$) for m$^x$ components',
              xlim=axes1.get_xlim(),
              xticks=list(range(int(axes1.get_xlim()[0]), int(axes1.get_xlim()[1]) + 1, step)))

    axes1.legend(loc=1, bbox_to_anchor=(0.975, 0.975),
                 frameon=True, fancybox=True, facecolor='white', edgecolor='white',
                 title='Propagation\n   Direction', fontsize=10)
    axes1.grid(axis='y', which='minor', lw=0)
    axes1.grid(axis='y', which='major', lw=2, ls='-')
    axes1.grid(axis='x', which='both', lw=2, ls='-')
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
