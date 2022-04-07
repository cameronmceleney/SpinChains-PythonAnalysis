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
import matplotlib.patches as mpatches
import gif as gif

# My packages / Any header files

"""
    Contains all the plotting functionality required for my data analysis. The data for each method comes from the file
    data_analysis.py. These plots will only work for data from my RK methods.
"""

"""
    Core Details
    
    Author      : cameronmceleney
    Created on  : 13/03/2022 18:06
    Filename    : plots_for_rk_methods.py
    IDE         : PyCharm
"""


# -------------------------------------- Plot paper figures -------------------------------------
class PaperFigures:
    """
    Generates a single subplot that can either be a PNG or GIF.

    Useful for creating plots for papers, or recreating a paper's work. To change between the png/gif saving options,
    change the invocation in data.analysis.py.
    """

    def __init__(self, time_data, amplitude_data, key_data, array_of_sites, output_filepath):
        self.time_data = time_data
        self.amplitude_data = amplitude_data
        self.sites_array = array_of_sites
        self.output_filepath = output_filepath

        # Individual attributes from key_data that are needed for the class
        self.number_spins = key_data["numSpins"]
        self.driving_freq = key_data['drivingFreq'] / 1e9  # Converts from [s] to [ns].
        self.data_points = key_data['numberOfDataPoints']
        self.max_time = key_data['maxSimTime'] * 1e9
        self.driving_width = key_data['drivingRegionWidth']
        self.numGilbert = key_data['numGilbert']
        self.drLHS = key_data['drivingRegionLHS']

        # Attributes for plots "ylim": [-1 * self.y_axis_limit, self.y_axis_limit]
        self.fig = plt.figure(figsize=(12, 6), dpi=300)
        self.axes = self.fig.add_subplot(111)
        self.y_axis_limit = max(self.amplitude_data[-1, :]) * 1.1  # Add a 10% margin to the y-axis.
        self.kwargs = {"title": f"Mx Values for {self.driving_freq:2.2f} [GHz]",
                       "xlabel": f"Spin Sites", "ylabel": f"m$_x$ [arb.]",
                       "xlim": [0, self.number_spins], "ylim": [-0.005, 0.005]}

    def _draw_figure(self, plot_row=-1, has_single_figure=True):
        """
        Private method to plot the given row of data, and create a single figure.

        If no figure param is passed, then the method will use the class' __init__ attributes.

        :param int plot_row: Given row in dataset to plot.
        :param bool has_single_figure: Flag to ensure that class
        attribute is used for single figure case, to allow for the saving of the figure out with this method.

        :return: No return statement. Method will output a figure to wherever the method was invoked.
        """
        if has_single_figure:
            # For images, may want to further alter plot outside this method. Hence, the use of attribute.
            fig = self.fig
            ax = self.axes
        else:
            # For GIFs
            fig = plt.figure(figsize=(12, 6), dpi=300)  # Each frame requires a new fig to prevent stuttering.
            ax = fig.add_subplot(111)  # Each subplot will be the same so no need to access ax outside of method.

        plt.suptitle("ChainSpin [RK2 - Midpoint]", size=24)
        plt.subplots_adjust(top=0.80)

        ax.plot(np.arange(1, self.number_spins + 1), self.amplitude_data[plot_row, :], ls='-', lw=0.5,
                label=f"{self.time_data[plot_row]:2.2f}")  # Easier to have time-stamp as label than textbox.

        ax.set(**self.kwargs)

        if not has_single_figure:
            left, bottom, width, height = (
                [0, self.number_spins - self.numGilbert],
                ax.get_ylim()[0], self.numGilbert, 2 * ax.get_ylim()[1])

            rectLHS = mpatches.Rectangle((left[0], bottom), width, height,
                                         # fill=False,
                                         alpha=0.1,
                                         facecolor="red")

            rectRHS = mpatches.Rectangle((left[1], bottom), width, height,
                                         # fill=False,
                                         alpha=0.1,
                                         facecolor="red")

            rectDriving = mpatches.Rectangle((self.drLHS, bottom), self.driving_width, height,
                                             # fill=False,
                                             alpha=0.1,
                                             facecolor="blue")

            plt.gca().add_patch(rectLHS)
            plt.gca().add_patch(rectRHS)
            plt.gca().add_patch(rectDriving)

        # Change tick markers as needed.
        ax.xaxis.set(major_locator=ticker.MultipleLocator(self.number_spins * 0.25),
                     minor_locator=ticker.MultipleLocator(self.number_spins * 0.125))
        ax.yaxis.set(major_locator=ticker.MaxNLocator(nbins=5, prune='lower'),
                     minor_locator=ticker.AutoMinorLocator())

        ax.legend(title="Real time [ns]", loc=1,
                  frameon=True, fancybox=True, framealpha=0.5, facecolor='white')

        fig.tight_layout()

    def create_png(self, row_number=-1):
        """
        Generate a PNG for a single row of the given dataset.

        A row corresponds to an instant in time, so this can be particularly useful for investigating the final 'state'
        of a system.

        :param int row_number: Which row of data to be plotted. Defaults to plotting the final row.

        :return: No direct returns. Invoking method will save a .png to the nominated 'Outputs' directory.
        """
        self._draw_figure(plot_row=row_number)
        self.fig.savefig(f"{self.output_filepath}.png")
        plt.show()

    @gif.frame
    def _plot_paper_gif(self, index):
        """
        Private method to save a given row of a data as a frame suitable for use with the git library.

        Require decorator so use method as an inner class instead of creating child class.

        :param int index: The row to be plotted.
        """
        self._draw_figure(index, False)

    def create_gif(self, number_of_frames=0.05):
        """
        Generate a GIF from the imported data.

        Uses the data that is imported in data_analysis.py, and turns each row in to a single figure. Multiple figures
        are then combined to form a GIF. This method does not accept *args or **kwargs, so to make any changes to
        gif.save() one must access this method directly. For more guidance see
        `this article
        <https://towardsdatascience.com/a-simple-way-to-turn-your-plots-into-gifs-in-python-f6ea4435ed3c>`_.

        :param float number_of_frames: How many frames the GIF should have (values between [0.01, 1.0]).

        :return: Will give a .gif file to the 'Outputs' folder of the given folder (selected earlier in the program).
        """

        frames = []

        for index in range(0, int(self.data_points + 1), int(self.data_points * number_of_frames)):
            frame = self._plot_paper_gif(index)
            frames.append(frame)

        gif.save(frames, f"{self.output_filepath}.gif", duration=1, unit='ms')

    def plot_site_variation(self, spin_site):
        """
        Plot the magnetisation of a site against time.

        One should ensure that the site being plotted is not inside either of the driving- or damping-regions.

        :param int spin_site: The number of the spin site to be plotted.

        :return: Saves a .png image to the designated output folder.
        """

        self.axes.plot(self.time_data, self.amplitude_data[:, spin_site], ls='-', lw=1,
                       label=f"{self.sites_array[spin_site]}")  # Easier to have time-stamp as label than textbox.

        self.axes.set(title=f"Mx Values for {self.driving_freq:2.2f} [GHz]",
                      xlabel=f"Time [ns]", ylabel=f"m$_x$ [arb.]",
                      xlim=[0, self.max_time])

        # Change tick markers as needed.
        self.axes.xaxis.set(major_locator=ticker.MultipleLocator(self.max_time * 0.2),
                            minor_locator=ticker.MultipleLocator(self.max_time * 0.1))
        self.axes.yaxis.set(major_locator=ticker.MaxNLocator(nbins=9, prune='lower'),
                            minor_locator=ticker.AutoMinorLocator())

        self.axes.legend(title="Spin Site [#]", loc=1,
                         frameon=True, fancybox=True, framealpha=0.5, facecolor='white')

        self.axes.grid(visible=True, axis='y', which='major')
        self.axes.grid(visible=False, axis='x', which='both')
        self.fig.tight_layout()
        self.fig.savefig(f"{self.output_filepath}_site{spin_site}.png")
        plt.show()

class PaperFigures2:
    """
    Generates a single subplot that can either be a PNG or GIF.

    Useful for creating plots for papers, or recreating a paper's work. To change between the png/gif saving options,
    change the invocation in data.analysis.py.
    """

    def __init__(self, time_data, amplitude_data, amplitude_data2, amplitude_data3, key_data, array_of_sites, output_filepath):
        self.time_data = time_data
        self.amplitude_data = amplitude_data
        self.amplitude_data2 = amplitude_data2
        self.amplitude_data3 = amplitude_data3
        self.sites_array = array_of_sites
        self.output_filepath = output_filepath

        # Individual attributes from key_data that are needed for the class
        self.number_spins = key_data["numSpins"]
        self.driving_freq = key_data['drivingFreq'] / 1e9  # Converts from [s] to [ns].
        self.data_points = key_data['numberOfDataPoints']
        self.max_time = key_data['maxSimTime'] * 1e9
        self.driving_width = key_data['drivingRegionWidth']
        self.numGilbert = key_data['numGilbert']
        self.drLHS = key_data['drivingRegionLHS']

        # Attributes for plots "ylim": [-1 * self.y_axis_limit, self.y_axis_limit]
        self.fig = plt.figure(figsize=(12, 6), dpi=300)
        self.axes = self.fig.add_subplot(111)
        self.y_axis_limit = max(self.amplitude_data[-1, :]) * 1.1  # Add a 10% margin to the y-axis.
        self.kwargs = {"title": f"Mx Values for {self.driving_freq:2.2f} [GHz]",
                       "xlabel": f"Spin Sites", "ylabel": f"m$_x$ [arb.]",
                       "xlim": [0, self.number_spins], "ylim": [-0.01, 0.01]}

    def _draw_figure(self, plot_row=-1, has_single_figure=True):
        """
        Private method to plot the given row of data, and create a single figure.

        If no figure param is passed, then the method will use the class' __init__ attributes.

        :param int plot_row: Given row in dataset to plot.
        :param bool has_single_figure: Flag to ensure that class
        attribute is used for single figure case, to allow for the saving of the figure out with this method.

        :return: No return statement. Method will output a figure to wherever the method was invoked.
        """
        if has_single_figure:
            # For images, may want to further alter plot outside this method. Hence, the use of attribute.
            fig = self.fig
            ax = self.axes
        else:
            # For GIFs
            fig = plt.figure(figsize=(12, 6), dpi=300)  # Each frame requires a new fig to prevent stuttering.
            ax = fig.add_subplot(111)  # Each subplot will be the same so no need to access ax outside of method.

        plt.suptitle("ChainSpin [RK2 - Midpoint]", size=24)
        plt.subplots_adjust(top=0.80)

        ax.plot(np.arange(1, self.number_spins + 1), self.amplitude_data[plot_row, :], ls='-', lw=0.5,
                label=f"Non-linear (x12)", zorder=2)  # Easier to have time-stamp as label than textbox.
        ax.plot(np.arange(1, self.number_spins + 1), self.amplitude_data2[plot_row, :], ls='-', lw=0.5,
                label=f"Linear (x2)", zorder=3)  # Easier to have time-stamp as label than textbox.
        ax.plot(np.arange(1, self.number_spins + 1), self.amplitude_data3[plot_row, :], ls='-', lw=0.5,
                label=f"Non-linear (x15)", zorder=1)  # Easier to have time-stamp as label than textbox.

        ax.set(**self.kwargs)

        if not has_single_figure:
            left, bottom, width, height = (
                [0, self.number_spins - self.numGilbert],
                ax.get_ylim()[0], self.numGilbert, 2 * ax.get_ylim()[1])

            rectLHS = mpatches.Rectangle((left[0], bottom), width, height,
                                         # fill=False,
                                         alpha=0.1,
                                         facecolor="red")

            rectRHS = mpatches.Rectangle((left[1], bottom), width, height,
                                         # fill=False,
                                         alpha=0.1,
                                         facecolor="red")

            rectDriving = mpatches.Rectangle((self.drLHS, bottom), self.driving_width, height,
                                             # fill=False,
                                             alpha=0.1,
                                             facecolor="blue")

            plt.gca().add_patch(rectLHS)
            plt.gca().add_patch(rectRHS)
            plt.gca().add_patch(rectDriving)

        # Change tick markers as needed.
        ax.xaxis.set(major_locator=ticker.MultipleLocator(self.number_spins * 0.25),
                     minor_locator=ticker.MultipleLocator(self.number_spins * 0.125))
        ax.yaxis.set(major_locator=ticker.MaxNLocator(nbins=5, prune='lower'),
                     minor_locator=ticker.AutoMinorLocator())

        ax.legend(title=f"Real time [ns]\n{self.time_data[plot_row]:2.2f}\n Scaling compared to\ninitial drive", loc=1,
                  frameon=True, fancybox=True, framealpha=0.5, facecolor='white')

        fig.tight_layout()

    def create_png(self, row_number=-1):
        """
        Generate a PNG for a single row of the given dataset.

        A row corresponds to an instant in time, so this can be particularly useful for investigating the final 'state'
        of a system.

        :param int row_number: Which row of data to be plotted. Defaults to plotting the final row.

        :return: No direct returns. Invoking method will save a .png to the nominated 'Outputs' directory.
        """
        self._draw_figure(plot_row=row_number)
        self.fig.savefig(f"{self.output_filepath}.png")
        plt.show()

    @gif.frame
    def _plot_paper_gif(self, index):
        """
        Private method to save a given row of a data as a frame suitable for use with the git library.

        Require decorator so use method as an inner class instead of creating child class.

        :param int index: The row to be plotted.
        """
        self._draw_figure(index, False)

    def create_gif(self, number_of_frames=0.05):
        """
        Generate a GIF from the imported data.

        Uses the data that is imported in data_analysis.py, and turns each row in to a single figure. Multiple figures
        are then combined to form a GIF. This method does not accept *args or **kwargs, so to make any changes to
        gif.save() one must access this method directly. For more guidance see
        `this article
        <https://towardsdatascience.com/a-simple-way-to-turn-your-plots-into-gifs-in-python-f6ea4435ed3c>`_.

        :param float number_of_frames: How many frames the GIF should have (values between [0.01, 1.0]).

        :return: Will give a .gif file to the 'Outputs' folder of the given folder (selected earlier in the program).
        """

        frames = []

        for index in range(0, int(self.data_points + 1), int(self.data_points * number_of_frames)):
            frame = self._plot_paper_gif(index)
            frames.append(frame)

        gif.save(frames, f"{self.output_filepath}.gif", duration=1, unit='ms')

    def plot_site_variation(self, spin_site):
        """
        Plot the magnetisation of a site against time.

        One should ensure that the site being plotted is not inside either of the driving- or damping-regions.

        :param int spin_site: The number of the spin site to be plotted.

        :return: Saves a .png image to the designated output folder.
        """

        self.axes.plot(self.time_data, self.amplitude_data[:, spin_site], ls='-', lw=1,
                       label=f"{self.sites_array[spin_site]}")  # Easier to have time-stamp as label than textbox.

        self.axes.set(title=f"Mx Values for {self.driving_freq:2.2f} [GHz]",
                      xlabel=f"Time [ns]", ylabel=f"m$_x$ [arb.]",
                      xlim=[0, self.max_time])

        # Change tick markers as needed.
        self.axes.xaxis.set(major_locator=ticker.MultipleLocator(self.max_time * 0.2),
                            minor_locator=ticker.MultipleLocator(self.max_time * 0.1))
        self.axes.yaxis.set(major_locator=ticker.MaxNLocator(nbins=9, prune='lower'),
                            minor_locator=ticker.AutoMinorLocator())

        self.axes.legend(title="Spin Site [#]", loc=1,
                         frameon=True, fancybox=True, framealpha=0.5, facecolor='white')

        self.axes.grid(visible=True, axis='y', which='major')
        self.axes.grid(visible=False, axis='x', which='both')
        self.fig.tight_layout()
        self.fig.savefig(f"{self.output_filepath}_site{spin_site}.png")
        plt.show()

# -------------------------------------- Useful to look at shockwaves. Three panes -------------------------------------
def three_panes(amplitude_data, key_data, list_of_spin_sites, filename, sites_to_compare=None):
    """
    Plots a graph

    :param Any amplitude_data: Array of magnitudes of the spin's magnetisation at each moment in time for each spin
                               site.
    :param dict key_data: All key simulation parameters imported from csv file.
    :param list list_of_spin_sites: Spin sites that were simulated.
    :param list[int] sites_to_compare: Optional. User-defined list of sites to plot.
    :param filename: data.
    """
    key_data['maxSimTime'] *= 1e9

    subplot_labels = create_plot_labels(list_of_spin_sites, key_data['drivingRegionLHS'], key_data['drivingRegionRHS'])

    time_values = np.linspace(0, key_data['maxSimTime'], int(key_data['numberOfDataPoints']) + 1)

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
        axes.set(xlabel='Time [ns]', ylabel="m$_x$", xlim=[0, key_data['maxSimTime']])
        axes.legend(loc=1, frameon=True)

        # This IF statement allows the comparison pane (a1) to have different axis limits to the other panes
        if axes == plot_pane_1:
            axes.yaxis.set(major_locator=ticker.MaxNLocator(nbins=5, prune='lower'),
                           minor_locator=ticker.AutoMinorLocator())
        else:
            axes.yaxis.set(major_locator=ticker.MaxNLocator(nbins=5, prune='lower'),
                           minor_locator=ticker.AutoMinorLocator())

    fig.tight_layout()
    fig.savefig(f"{filename}.png")
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
    :param filename: The name of the file that is being read from.

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
    frequencies, fourier_transform, natural_frequency, driving_freq = fft_data(amplitude_data, simulation_params)

    driving_freq_hz = simulation_params['drivingFreq'] / 1e9
    if ax is None:
        ax = plt.gca()

    # Must be abs(FFTransform) to make sense!
    ax.plot(frequencies, abs(fourier_transform),
            marker='o', lw=1, color='red', markerfacecolor='black', markeredgecolor='black')
    ax.set(**plt_set_kwargs)

    ax.axvline(x=driving_freq_hz, label=f"Driving. {driving_freq_hz:2.2f}", color='green')

    if which_subplot == 2:
        ax.axvspan(0, 5, color='#DC143C', alpha=0.2, lw=0)
        # If at a node, then 3-wave generation may be occurring. This loop plots that location.
        triple_wave_gen_freq = driving_freq_hz * 3
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
    driving_freq_ghz = simulation_params['drivingFreq'] * core_values["hz_to_ghz"]

    # This is the (first) natural frequency of the system, corresponding to the first eigenvalue. Change as needed to
    # add other markers to the plot(s)
    natural_freq = core_values['gamma'] * simulation_params['biasField']

    # Calculate FFT parameters
    time_interval = simulation_params['stepsize'] * simulation_params['stopIterVal']
    n_samples = simulation_params['numberOfDataPoints']
    dt = time_interval / n_samples  # Or multiply the stepsize by the number of iterations between data recordings

    # Compute the FFT
    fourier_transform = np.fft.fft(amplitude_data)  # Normalize amplitude after taking FFT
    fourier_transform = fourier_transform[range(int(n_samples / 2))]  # Exclude sampling frequency, and negative values
    frequencies = (np.arange(int(n_samples / 2)) / (dt * n_samples)) * core_values["hz_to_ghz"]

    return frequencies, fourier_transform, natural_freq, driving_freq_ghz


def create_contour_plot(mx_data, my_data, mz_data, spin_site, output_file, use_tri=False):
    x = mx_data[:, spin_site]
    y = my_data[:, spin_site]
    z = mz_data[:, spin_site]

    # 'magma' is also nice
    fig = plt.figure(figsize=(12, 12))
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
    plt.show()
    fig.savefig(f"{output_file}_contour.png")


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

                has_more_plots = input("Do you want to continue plotting modes? Y/N: ").upper()
                while True:

                    if has_more_plots == 'Y':
                        has_valid_modes = False  # Prevents plotting of incorrect input, and allows user to retry.
                        break

                    elif has_more_plots == 'N':
                        print("Exiting program...")
                        exit(0)

                    else:
                        while has_more_plots not in 'YN':
                            has_more_plots = input("Do you want to continue plotting modes? Y/N: ").upper()

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
    gx_lhs = g_ones + g_zeros
    gx_rhs = g_zeros + g_ones

    fourier_coefficents_lhs = []
    fourier_coefficents_rhs = []

    for i in range(0, number_of_spins):
        # Select an eigenvector, and take the dot-product to return the coefficient of that particular mode.
        fourier_coefficents_lhs.append(np.dot(gx_lhs, amplitude_mx_data[:, i]))
        fourier_coefficents_rhs.append(np.dot(gx_rhs, amplitude_mx_data[:, i]))

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
