#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib as mpl
# For interactive plots on Mac
# matplotlib.use('macosx')

# Standard modules (common)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Third party modules (uncommon)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import gif as gif
from scipy.fft import rfft, rfftfreq
from typing import Any

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

    def __init__(self, time_data, amplitude_data, key_data, sim_flags, array_of_sites, output_filepath):
        self.time_data = time_data
        self.amplitude_data = amplitude_data
        self.sites_array = array_of_sites
        self.output_filepath = output_filepath

        self.nm_method = sim_flags['numericalMethodUsed']

        # Individual attributes from key_data that are needed for the class
        self.static_field = key_data['staticBiasField']
        self.driving_field1 = key_data['dynamicBiasField1']
        self.driving_field2 = key_data['dynamicBiasField2']
        self.driving_freq = key_data['drivingFreq'] / 1e9  # Converts from [s] to (ns).
        self.drLHS = key_data['drivingRegionLHS']
        self.drRHS = key_data['drivingRegionRHS']
        self.driving_width = key_data['drivingRegionWidth']
        self.max_time = key_data['maxSimTime'] * 1e9
        self.stop_iteration_value = key_data['stopIterVal']
        self.exchange_min = key_data['exchangeMinVal']
        self.exchange_max = key_data['exchangeMaxVal']
        self.data_points = key_data['numberOfDataPoints']
        self.chain_spins = key_data['chainSpins']
        self.dampedSpins = key_data['dampedSpins']
        self.number_spins = key_data['totalSpins']
        self.stepsize = key_data['stepsize'] * 1e9
        self.gilbert_factor = key_data['gilbertFactor']
        self.gyro_mag_ratio = key_data['gyroMagRatio']

        # Attributes for plots
        self.fig = None
        self.axes = None
        self.y_axis_limit = max(self.amplitude_data[-1, :]) * 1.1  # Add a 10% margin to the y-axis.
        self.kwargs = {"xlabel": f"Site Number [$N_i$]", "ylabel": f"m$_x$ / M$_S$",
                       "xlim": [0, self.number_spins], "ylim": [-1 * self.y_axis_limit, self.y_axis_limit]}

        self.large_size = 20
        self.medium_size = 14
        self.small_size = 11
        self.smaller_size = 10
        self.tiny_size = 8
        self.mini_size = 7

    def _draw_figure(self, plot_row=-1, has_single_figure=True, draw_regions_of_interest=True):
        """
        Private method to plot the given row of data, and create a single figure.

        If no figure param is passed, then the method will use the class' __init__ attributes.

        :param int plot_row: Given row in dataset to plot.
        :param bool has_single_figure: Flag to ensure that class
        attribute is used for single figure case, to allow for the saving of the figure out with this method.

        :return: No return statement. Method will output a figure to wherever the method was invoked.
        """
        if self.fig is None:
            if has_single_figure:
                # For single images, may want to further alter plot outside this method.
                self.fig = plt.figure(figsize=(4.4, 2.0))  # Strange dimensions are to give a 4x2 inch image
                self.axes = self.fig.add_subplot(111)
            else:
                # For GIFs. Each frame requires a new fig to prevent stuttering. Each subplot will be the same
                # so no need to access ax outside of method.
                cm = 1 / 2.54
                self.fig = plt.figure(figsize=(11.12 * cm * 2, 6.15 * cm * 2))
                self.axes = self.fig.add_subplot(111)
        else:
            self.axes.clear()  # Use this if looping through a single PaperFigures object for multiple create_png inputs

        self.axes.set_aspect("auto")

        # Easier to have time-stamp as label than textbox.
        self.axes.plot(np.arange(0, self.number_spins), self.amplitude_data[plot_row, :], ls='-', lw=2 * 0.75,
                       label=f"{self.time_data[plot_row]: 2.2f} (ns)", color='#64bb6a')

        self.axes.set(**self.kwargs)

        # self.axes.text(-0.04, 0.96, r'$\times \mathcal{10}^{{\mathcal{-3}}}$', verticalalignment='center',
        # horizontalalignment='center', transform=self.axes.transAxes, fontsize=6)
        # self.axes.text(0.88, 0.88, f"(c) {self.time_data[plot_row]:2.3f} ns",
        #               verticalalignment='center', horizontalalignment='center', transform=self.axes.transAxes,
        #               fontsize=6)

        self.axes.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=6)

        if draw_regions_of_interest:
            left, bottom, width, height = (
                [0, (self.number_spins - self.dampedSpins), (self.drLHS + self.dampedSpins)],
                self.axes.get_ylim()[0] * 2,
                (self.dampedSpins, self.driving_width),
                4 * self.axes.get_ylim()[1])

            rectangle_lhs = mpatches.Rectangle((left[0], bottom), width[0], height,
                                               alpha=0.5, facecolor="grey", edgecolor=None, lw=0)

            rectangle_rhs = mpatches.Rectangle((left[1], bottom), width[0], height,
                                               alpha=0.5, facecolor="grey", edgecolor=None, lw=0)

            rectangle_driving_region = mpatches.Rectangle((left[2], bottom), width[1], height,
                                                          alpha=0.25, facecolor="grey", edgecolor=None, lw=0)

            plt.gca().add_patch(rectangle_lhs)
            plt.gca().add_patch(rectangle_rhs)
            plt.gca().add_patch(rectangle_driving_region)

        # Change tick markers as needed.
        self._tick_setter(self.axes, self.number_spins / 4, self.number_spins / 8, 3, 4)

        class ScalarFormatterClass(ticker.ScalarFormatter):
            def _set_format(self):
                self.format = "%1.1f"

        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0, 0))
        self.axes.xaxis.set_major_formatter(yScalarFormatter)
        self.axes.yaxis.set_major_formatter(yScalarFormatter)

        self.fig.tight_layout()

    def _draw_figure1(self, plot_row=-1, has_single_figure=True, draw_regions_of_interest=True):
        """
        Private method to plot the given row of data, and create a single figure.

        If no figure param is passed, then the method will use the class' __init__ attributes.

        :param int plot_row: Given row in dataset to plot.
        :param bool has_single_figure: Flag to ensure that class
        attribute is used for single figure case, to allow for the saving of the figure out with this method.

        :return: No return statement. Method will output a figure to wherever the method was invoked.
        """
        if has_single_figure:
            # For single images, may want to further alter plot outside this method.
            self.fig = plt.figure(figsize=(4.4, 2.0))  # Strange dimensions are to give a 4x2 inch image
            self.axes = self.fig.add_subplot(111)
        else:
            # For GIFs. Each frame requires a new fig to prevent stuttering. Each subplot will be the same so no need
            # to access ax outside of method.
            cm = 1 / 2.54
            self.fig = plt.figure(figsize=(11.12 * cm * 2, 6.15 * cm * 2))
            self.axes = self.fig.add_subplot(111)

        self.axes.set_aspect("auto")
        self.axes.axis('off')

        # Easier to have time-stamp as label than textbox.
        self.axes.plot(np.arange(0, self.number_spins), self.amplitude_data[plot_row, :], ls='-', lw=2 * 0.75,
                       label=f"{self.time_data[plot_row]: 2.2f} (ns)", color='#64bb6a')

        self.axes.set(**self.kwargs)

        # self.axes.text(-0.04, 0.96, r'$\times \mathcal{10}^{{\mathcal{-3}}}$', verticalalignment='center',
        # horizontalalignment='center', transform=self.axes.transAxes, fontsize=6)
        # self.axes.text(0.88, 0.88, f"(c) {self.time_data[plot_row]:2.3f} ns",
        #               verticalalignment='center', horizontalalignment='center', transform=self.axes.transAxes,
        #               fontsize=6)

        self.axes.xaxis.labelpad = -1.5
        self.axes.yaxis.labelpad = -5
        self.axes.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=6)

        if draw_regions_of_interest:
            left, bottom, width, height = (
                [0, (self.number_spins - self.dampedSpins), (self.drLHS + self.dampedSpins)],
                self.axes.get_ylim()[0] * 2,
                (self.dampedSpins, self.driving_width),
                4 * self.axes.get_ylim()[1])

            rectangle_lhs = mpatches.Rectangle((left[0], bottom), width[0], height,
                                               alpha=0.5, facecolor="grey", edgecolor=None, lw=0)

            rectangle_rhs = mpatches.Rectangle((left[1], bottom), width[0], height,
                                               alpha=0.5, facecolor="grey", edgecolor=None, lw=0)

            rectangle_driving_region = mpatches.Rectangle((left[2], bottom), width[1], height,
                                                          alpha=0.25, facecolor="grey", edgecolor=None, lw=0)

            plt.gca().add_patch(rectangle_lhs)
            plt.gca().add_patch(rectangle_rhs)
            plt.gca().add_patch(rectangle_driving_region)

        # Change tick markers as needed.
        xlim_major_ticks = 500
        self._tick_setter(self.axes, xlim_major_ticks, 25, 3, 4)

        self.axes.vlines(x=[500, 5300], ymin=-0.5, ymax=0.5,
                         colors='black', lw=10,
                         label='vline_multiple - full height')

        self.axes.text(300, 3.2e-3, 'A', fontsize=24)
        self.axes.text(5400, 3.2e-3, 'B', fontsize=24)

        class ScalarFormatterClass(ticker.ScalarFormatter):
            def _set_format(self):
                self.format = "%1.1f"

        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0, 0))
        self.axes.xaxis.set_major_formatter(yScalarFormatter)
        self.axes.yaxis.set_major_formatter(yScalarFormatter)

        self.fig.tight_layout()

    def create_position_variation(self, row_number=-1, should_add_data=False):
        """
        Generate a PNG for a single row of the given dataset.

        A row corresponds to an instant in time, so this can be particularly useful for investigating the final 'state'
        of a system.

        :param should_add_data:
        :param int row_number: Which row of data to be plotted. Defaults to plotting the final row.

        :return: No direct returns. Invoking method will save a .png to the nominated 'Outputs' directory.
        """
        self._draw_figure(row_number)

        self.axes.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        if should_add_data:
            # Add text to figure with simulation parameters
            if self.exchange_min == self.exchange_max:
                exchangeString = f"Uniform Exc.: {self.exchange_min} (T)"
            else:
                exchangeString = f"J$_{{min}}$ = {self.exchange_min} (T) | J$_{{max}}$ = " \
                                 f"{self.exchange_max} (T)"
            data_string = (f"H$_{{0}}$ = {self.static_field} (T) | N = {self.chain_spins} | " + r"$\alpha$" +
                           f" = {self.gilbert_factor: 2.2e}\nH$_{{D1}}$ = {self.driving_field1: 2.2e} (T) | "
                           f"H$_{{D2}}$ = {self.driving_field2: 2.2e} (T) \n{exchangeString}")

            props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.5)
            # Place text box in upper left in axes coords
            self.axes.text(0.05, 1.2, data_string, transform=self.axes.transAxes, fontsize=12,
                           verticalalignment='top', bbox=props, ha='center', va='center')

        # Add spines to all plots (to override any rcParams elsewhere in the code
        for spine in ['top', 'bottom', 'left', 'right']:
            self.axes.spines[spine].set_visible(True)

        self.axes.grid(visible=True, axis='both', which='both')

        self.fig.savefig(f"{self.output_filepath}_row{row_number}.png", bbox_inches="tight")

    @gif.frame
    def _plot_paper_gif(self, index):
        """
        Private method to save a given row of a data as a frame suitable for use with the git library.

        Require decorator so use method as an inner class instead of creating child class.

        :param int index: The row to be plotted.
        """
        self._draw_figure(index, False, False)

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

        plt.rcParams.update({'savefig.dpi': 200, "figure.dpi": 200})

        for index in range(0, int(self.data_points + 1), int(self.data_points * number_of_frames)):
            frame = self._plot_paper_gif(index)
            frames.append(frame)

        gif.save(frames, f"{self.output_filepath}.gif", duration=0.5)

    def create_time_variation(self, spin_site, colour_precursors=False, annotate_precursors=False,
                              basic_annotations=False, add_zoomed_region=False, add_info_box=False,
                              add_coloured_regions=False, interactive_plot=False):
        """
        Plot the magnetisation of a site against time.

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

            # Add spines to all plots (to override any rcParams elsewhere in the code
            for spine in ['top', 'bottom', 'left', 'right']:
                ax1_inset.spines[spine].set_visible(True)
                ax1.spines[spine].set_visible(True)

            # mark_inset(ax1, ax1_inset,loc1=1, loc2=3, facecolor='#f9f2e9', edgecolor='black', alpha=1.0, zorder=1.05)

            # Add box to indicate the region which is being zoomed into on the main figure
            ax1.indicate_inset_zoom(ax1_inset, facecolor='#f9f2e9', edgecolor='black', alpha=1.0, lw=0.75,
                                    zorder=1)
        elif add_zoomed_region is False:
            for spine in ['top', 'bottom', 'left', 'right']:
                ax1.spines[spine].set_visible(True)

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
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            # ax.set_facecolor('#f4f4f5')
            ax.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=1.9999)

            # Add spines to all plots (to override any rcParams elsewhere in the code
            for spine in ['top', 'bottom', 'left', 'right']:
                ax.spines[spine].set_visible(True)

            ax.set_axisbelow(False)
            ax.set_facecolor('white')

        self.fig.subplots_adjust(wspace=1, hspace=0.35)

        if interactive_plot:
            # For interactive plots
            def mouse_event(event: Any):
                print(f'x: {event.xdata} and y: {event.ydata}')

            self.fig.canvas.mpl_connect('button_press_event', mouse_event)
            self.fig.tight_layout()  # has to be here
            plt.show()
        else:
            self.fig.savefig(f"{self.output_filepath}_site{spin_site}.pdf", bbox_inches="tight")

    def create_time_variation1(self, spin_site, colour_precursors=False, annotate_precursors=False,
                               basic_annotations=False, add_zoomed_region=False, add_info_box=False,
                               add_coloured_regions=False, interactive_plot=False):
        """
        Plot the magnetisation of a site against time.

        One should ensure that the site being plotted is not inside either of the driving- or damping-regions.

        :param interactive_plot: NEED DOCSTRING
        :param basic_annotations: NEED DOCSTRING
        :param annotate_precursors: Add arrows to denote precursors.
        :param colour_precursors: Draw 1st, 3rd and 5th precursors as separate colours to main figure.
        :param bool add_coloured_regions: Draw coloured boxes onto plot to show driving- and damping-regions.
        :param bool add_info_box: Add text box to base of plot which lists key simulation parameters.
        :param bool add_zoomed_region: Add inset to plot to focus upon precursors.
        :param int spin_site: The number of the spin site to be plotted.

        :return: Saves a .png image to the designated output folder.
        """
        # Data at 2023-03-06/rrk2_mx_T1118

        if self.fig is None:
            self.fig = plt.figure(figsize=(4.5, 3.375))
        num_rows = 2
        num_cols = 3
        ax1 = plt.subplot2grid((num_rows, num_cols), (0, 0), rowspan=int(num_rows / 2),
                               colspan=num_cols, fig=self.fig)
        ax2 = plt.subplot2grid((num_rows, num_cols), (int(num_rows / 2), 0), rowspan=num_rows,
                               colspan=num_cols, fig=self.fig)

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
            return int(self.data_points * ((b - a) * ((val - xlim_min) / (xlim_max - xlim_min)) + a))

        lower1_signal, upper1_signal = convert_norm(lower1), convert_norm(upper1)
        lower2_signal, upper2_signal = convert_norm(lower2), convert_norm(upper2)
        lower3_signal, upper3_signal = convert_norm(lower3), convert_norm(upper3)

        lower1_precursor, upper1_precursor = convert_norm(lower1_blob), convert_norm(upper1_blob)
        lower2_precursor, upper2_precursor = convert_norm(lower2_blob), convert_norm(upper2_blob)
        lower3_precursor, upper3_precursor = convert_norm(lower3_blob), convert_norm(upper3_blob)

        color_gen = "#73B741"  # gr #73B741 dg #8C8E8D" # dg "#80BE53"
        color_gen1 = "#F77D6A"
        color_precursors = "#CD331B"
        color_shockwave = "#B896B0"  # cy 3EB8A1
        color_equilib = "#3775B2"  # B79549
        # 37782c, 64bb6a, 9fd983
        ax1.plot(self.time_data[:],
                 self.amplitude_data[:, spin_site], ls='-', lw=0.75,
                 color=f'{color_gen}', alpha=0.5,
                 markerfacecolor='black', markeredgecolor='black', zorder=1.01)
        ax1.plot(self.time_data[lower1_signal:upper1_signal],
                 self.amplitude_data[lower1_signal:upper1_signal, spin_site], ls='-', lw=0.75,
                 color=f'{color_gen}', label=f"{self.sites_array[spin_site]}",
                 markerfacecolor='black', markeredgecolor='black', zorder=1.1)
        ax1.plot(self.time_data[lower2_signal:upper2_signal],
                 self.amplitude_data[lower2_signal:upper2_signal, spin_site], ls='-', lw=0.75,
                 color=f'{color_gen}', label=f"{self.sites_array[spin_site]}",
                 markerfacecolor='black', markeredgecolor='black', zorder=1.1)
        ax1.plot(self.time_data[lower3_signal:upper3_signal],
                 self.amplitude_data[lower3_signal:upper3_signal, spin_site], ls='-', lw=0.75,
                 color=f'{color_gen}', label=f"{self.sites_array[spin_site]}",
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
            axes_props1 = {"arrowstyle": '|-|, widthA =0.4, widthB=0.4', "color": f"{color_precursors}", 'lw': 1.0}
            axes_props2 = {"arrowstyle": '|-|, widthA =0.4, widthB=0.4', "color": f"{color_shockwave}", 'lw': 1.0}
            axes_props3 = {"arrowstyle": '|-|, widthA =0.4, widthB=0.4', "color": f"{color_equilib}", 'lw': 1.0}

            # ax1.text(0.95, 0.9, f"(b)",
            #         va='center', ha='center', fontsize=self.smaller_size, transform=ax1.transAxes)
            #
            # ax2.text(0.05, 0.9, f"(c)",
            #         va='center', ha='center', fontsize=self.smaller_size,
            #         transform=ax2.transAxes)

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
                                   bbox_to_anchor=[0.01, 1.22], bbox_transform=ax1.transAxes)
            ax1_inset.plot(x, y, lw=0.75, color=f'{color_gen}', zorder=1.1)

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

            arrow_ax1_props = {"arrowstyle": '-|>', "connectionstyle": 'angle3, angleA=0, angleB=40', "color": "black",
                               'lw': 0.8}
            arrow_ax1_props2 = {"arrowstyle": '-|>', "connectionstyle": 'angle3, angleA=0, angleB=140',
                                "color": "black", 'lw': 0.8}

            ax1_inset.annotate('P1', xy=(1.85, -6e-5), xytext=(1.5, -1.3e-4), va='center', ha='center',
                               arrowprops=arrow_ax1_props, fontsize=self.tiny_size)
            ax1_inset.annotate('P2', xy=(1.45, 6e-5), xytext=(1.1, 1.3e-4), va='center', ha='center',
                               arrowprops=arrow_ax1_props2, fontsize=self.tiny_size)
            ax1_inset.annotate('P3', xy=(1.15, -3e-5), xytext=(0.8, -1.3e-4), va='center', ha='center',
                               arrowprops=arrow_ax1_props, fontsize=self.tiny_size)

            # Remove tick labels
            ax1_inset.set_xticks([])
            ax1_inset.set_yticks([])
            ax1_inset.patch.set_color("#f9f2e9")  # #f0a3a9 is equivalent to color 'red' and alpha '0.3'

            # Add spines to all plots (to override any rcParams elsewhere in the code
            for spine in ['top', 'bottom', 'left', 'right']:
                ax1_inset.spines[spine].set_visible(True)
                ax1.spines[spine].set_visible(True)

            # mark_inset(ax1, ax1_inset,loc1=1, loc2=3, facecolor='#f9f2e9', edgecolor='black', alpha=1.0, zorder=1.05)

            # Add box to indicate the region which is being zoomed into on the main figure
            # ax1.indicate_inset_zoom(ax1_inset, facecolor='#f9f2e9', edgecolor='black', alpha=1.0, lw=0.75,
            #                        zorder=1)
            rect = mpl.patches.Rectangle((0.7, -6e-4), 1.91, 1.2e-3, lw=1, edgecolor='black',
                                         facecolor='#f9f2e9')
            ax1.add_patch(rect)
        elif add_zoomed_region is False:
            for spine in ['top', 'bottom', 'left', 'right']:
                ax1.spines[spine].set_visible(True)

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

        ax2.plot(frequencies_precursors, abs(fourier_transform_precursors), marker='', lw=1,
                 color=f"{color_precursors}", markerfacecolor='black', markeredgecolor='black',
                 label="Precursors", zorder=1.5)
        ax2.plot(frequencies_dsw, abs(fourier_transform_dsw), marker='', lw=1, color=f'{color_shockwave}',
                 markerfacecolor='black', markeredgecolor='black', label="Shockwave", zorder=1.2)
        ax2.plot(frequencies_eq, abs(fourier_transform_eq), marker='', lw=1, color=f'{color_equilib}',
                 markerfacecolor='black', markeredgecolor='black', label="Steady State", zorder=1.1)

        if annotate_precursors:
            frequencies_blob1, fourier_transform_blob1 = self._fft_data(
                self.amplitude_data[lower1_precursor:upper1_precursor, spin_site])
            frequencies_blob2, fourier_transform_blob2 = self._fft_data(
                self.amplitude_data[lower2_precursor:upper2_precursor, spin_site])
            frequencies_blob3, fourier_transform_blob3 = self._fft_data(
                self.amplitude_data[lower3_precursor:upper3_precursor, spin_site])

            ax2.plot(frequencies_blob1, abs(fourier_transform_blob1), marker='', lw=1, color=f'{color_gen1}',
                     markerfacecolor='black', markeredgecolor='black', ls=':', zorder=1.9)
            ax2.plot(frequencies_blob2, abs(fourier_transform_blob2), marker='', lw=1, color=f'{color_gen1}',
                     markerfacecolor='black', markeredgecolor='black', ls='--', zorder=1.9)
            ax2.plot(frequencies_blob3, abs(fourier_transform_blob3), marker='', lw=1, color=f'{color_gen1}',
                     markerfacecolor='black', markeredgecolor='black', ls='-.', zorder=1.9)

            arrow_ax2_props = {"arrowstyle": '-|>', "connectionstyle": "angle3,angleA=0,angleB=90", "color": "black"}
            ax2.annotate('P1', xy=(26, 1.8e1), xytext=(34.1, 2.02e2), va='center', ha='center',
                         arrowprops=arrow_ax2_props, fontsize=self.smaller_size)
            ax2.annotate('P2', xy=(48.78, 4.34e0), xytext=(56.0, 5.37e1), va='center', ha='center',
                         arrowprops=arrow_ax2_props, fontsize=self.smaller_size)
            ax2.annotate('P3', xy=(78.29, 1.25e0), xytext=(83.9, 7.5), va='center', ha='center',
                         arrowprops=arrow_ax2_props, fontsize=self.smaller_size)

        ax2.legend(ncol=1, loc='upper right', fontsize=self.tiny_size, frameon=False, fancybox=True, facecolor=None,
                   edgecolor=None,
                   bbox_to_anchor=[0.99, 0.975], bbox_transform=ax2.transAxes)

        for ax in [ax1, ax2]:
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            # ax.set_facecolor('#f4f4f5')
            ax.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=1.9999)

            # Add spines to all plots (to override any rcParams elsewhere in the code
            for spine in ['top', 'bottom', 'left', 'right']:
                ax.spines[spine].set_visible(True)

            ax.set_axisbelow(False)
            ax.set_facecolor('white')

        self.fig.subplots_adjust(wspace=1, hspace=0.35)

        if interactive_plot:
            # For interactive plots
            def mouse_event(event: Any):
                print(f'x: {event.xdata} and y: {event.ydata}')

            self.fig.canvas.mpl_connect('button_press_event', mouse_event)
            self.fig.tight_layout()  # has to be here
            plt.show()
        else:
            self.fig.savefig(f"{self.output_filepath}_site{spin_site}2.eps", bbox_inches="tight")

    def create_time_variation2(self, use_inset=False, interactive_plot=False):
        """
        Plot the magnetisation of a site against time.

        One should ensure that the site being plotted is not inside either of the driving- or damping-regions.

        :return: Saves a .png image to the designated output folder.
        """

        if self.fig is None:
            self.fig = plt.figure(figsize=(4.5, 3.375))
        num_rows = 2
        num_cols = 3

        ax1 = plt.subplot2grid((num_rows, num_cols), (0, 0), rowspan=int(num_rows / 2),
                               colspan=num_cols, fig=self.fig)

        SAMPLE_RATE = int(5e2)  # Number of samples per nanosecond
        DURATION = int(40)  # Nanoseconds

        def generate_sine_wave(freq, sample_rate, duration, delay_num):
            delay = int(sample_rate * delay_num)
            t = np.linspace(0, duration, sample_rate * duration, endpoint=False)
            y_1 = np.zeros(delay)
            y_2 = np.sin((2 * np.pi * freq) * t[delay:])
            y_con = np.concatenate((y_1, y_2))
            return t, y_con

        # Generate a 15 GHz sine wave that lasts for 5 seconds
        x1, y1 = generate_sine_wave(8, SAMPLE_RATE, DURATION, 1)
        x2, y2 = generate_sine_wave(8, SAMPLE_RATE, DURATION, 0)
        from scipy.fft import rfft, rfftfreq
        # Number of samples in normalized_tone
        n1 = n2 = int(SAMPLE_RATE * DURATION)

        y1f, y2f = rfft(y1), rfft(y2)
        x1f, x2f = rfftfreq(n1, 1 / SAMPLE_RATE), rfftfreq(n2, 1 / SAMPLE_RATE)

        ax1.plot(x1f, np.abs(y1f), marker='', lw=1, color='#ffb55a', markerfacecolor='black', markeredgecolor='black',
                 label="1", zorder=1.2)  # 0289F7
        ax1.plot(x2f, np.abs(y2f), marker='', lw=1, ls='-', color='#64bb6a', markerfacecolor='black',
                 markeredgecolor='black', label="0", zorder=1.3)

        ax1.set(xlim=(0.001, 15.999), ylim=(1e0, 1e5),
                xlabel="Frequency (GHz)", ylabel="Amplitude (arb. units)", yscale='log')

        ax1.xaxis.labelpad = -2
        ax1.yaxis.labelpad = -0
        self._tick_setter(ax1, 4, 1, 4, 4, is_fft_plot=True)

        ########################################
        ax1_inset = inset_axes(ax1, width=1.3, height=0.36, loc="upper right", bbox_to_anchor=[0.995, 0.805],
                               bbox_transform=ax1.transAxes)

        ax1_inset2 = inset_axes(ax1, width=1.3, height=0.36, loc="upper right", bbox_to_anchor=[0.995, 1.185],
                                bbox_transform=ax1.transAxes)

        ax1_inset.plot(x1, y1, lw=1, color='#ffb55a', zorder=1.2)
        ax1_inset2.plot(x2, y2, lw=1., ls='-', color='#64bb6a', zorder=1.1)

        for ax in [ax1_inset, ax1_inset2]:
            ax.set(xlim=[0, 2], ylim=[-1, 1])
            ax.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=1.99)

            if ax == ax1_inset:
                ax.set_xlabel('Time (ns)', fontsize=self.tiny_size)
                ax.yaxis.tick_left()
                ax.yaxis.set_label_position("left")
                ax.set_ylabel('Amplitude  \n(arb. units)  ', fontsize=self.tiny_size, rotation=90, labelpad=20)
                ax.yaxis.set_label_coords(-.2, 1.15)
                ax.xaxis.labelpad = -1

            if ax == ax1_inset2:
                ax.tick_params(axis='x', which='both', labelbottom=False)
            ax.tick_params(axis='both', labelsize=self.mini_size)

            # ax1_inset.patch.set_color("#f9f2e9")

            for spine in ["top", "bottom", "left", "right"]:
                ax.spines[spine].set_visible(True)

            self._tick_setter(ax, 1.0, 0.5, 1, 0.5, yaxis_multi_loc=True, is_fft_plot=False, yaxis_num_decimals=1,
                              yscale_type='p')

        ########################################

        ax2 = plt.subplot2grid((num_rows, num_cols), (int(num_rows / 2), 0),
                               rowspan=num_rows, colspan=num_cols, fig=self.fig)

        # ax2.scatter(np.arange(1, len(freqs) + 1, 1), freqs, s=0.5)
        external_field, exchange_field = 0.1, 132.5
        gyromag_ratio = 28.8e9
        lattice_constant = np.sqrt(5.3e-17 / exchange_field)
        system_len = 10e-5  # metres
        max_len = round(system_len / lattice_constant)
        num_spins_array = np.arange(0, max_len, 1)
        wave_number_array = (num_spins_array * np.pi) / ((len(num_spins_array) - 1) * lattice_constant)
        freq_array = gyromag_ratio * (2 * exchange_field * (1 - np.cos(wave_number_array * lattice_constant))
                                      + external_field)

        hz_2_THz = 1e-12
        hz_2_GHz = 1e-9

        ax2.plot(wave_number_array * hz_2_GHz, freq_array * hz_2_THz, color='red', lw=1., ls='-', label=f'Dataset 1')
        ax2.plot(wave_number_array * hz_2_GHz, gyromag_ratio * (
                external_field + exchange_field * lattice_constant ** 2 * wave_number_array ** 2) * hz_2_THz,
                 color='red', lw=1., alpha=0.4, ls='--', label=f'Dataset 1')

        # These!!
        # ax2.scatter(np.arccos(1 - ((freqs2 / gamma - h_0) / (2 * h_ex))) / a, freqs2 / 1e12,
        #             s=0.5, c='red', label='paper')
        # ax2.plot(k, gamma * (2 * h_ex * (1 - np.cos(k * a)) + h_0) / 1e12, color='red', ls='--', label=f'Kittel')

        ax2.set(xlabel="Wavenumber (nm$^{-1}$)", ylabel='Frequency (THz)', ylim=[0, 15.4])
        self._tick_setter(ax2, 2, 0.5, 3, 2, is_fft_plot=False, xaxis_num_decimals=1, yaxis_num_decimals=0,
                          yscale_type='p')
        ax2.grid(False)

        ax2.axhline(y=3.8, xmax=1.0, ls='--', lw=1, color='grey', zorder=0.9)  # xmax=0.31
        ax2.axhline(y=10.5, xmax=1.0, ls='--', lw=1, color='grey', zorder=0.9)  # xmax=0.68
        ax2.margins(0)
        ax2.xaxis.labelpad = -2

        ax2.text(0.997, -0.13, r"$\mathrm{\dfrac{\pi}{a}}$",
                 verticalalignment='center', horizontalalignment='center', transform=ax2.transAxes,
                 fontsize=self.smaller_size)

        ax2.text(0.02, 0.88, r"$\mathcal{III}$", verticalalignment='center', horizontalalignment='left',
                 transform=ax2.transAxes, fontsize=self.smaller_size)
        ax2.text(0.02, 0.5, r"$\mathcal{II}$", verticalalignment='center', horizontalalignment='left',
                 transform=ax2.transAxes, fontsize=self.smaller_size)
        ax2.text(0.02, 0.12, r"$\mathcal{I}$", verticalalignment='center', horizontalalignment='left',
                 transform=ax2.transAxes, fontsize=self.smaller_size)

        ax2.text(0.91, 0.82, f"Decreasing", verticalalignment='center', horizontalalignment='center',
                 transform=ax2.transAxes, fontsize=self.tiny_size)
        ax2.text(0.60, 0.425, f"Constant", verticalalignment='center', horizontalalignment='center',
                 transform=ax2.transAxes, fontsize=self.tiny_size)
        ax2.text(0.41, 0.12, f"Increasing", verticalalignment='center', horizontalalignment='center',
                 transform=ax2.transAxes, fontsize=self.tiny_size)

        arrow_ax2_props1 = {"arrowstyle": '-|>', "connectionstyle": "arc3,rad=0.075", "color": "black"}
        arrow_ax2_props2 = {"arrowstyle": '-|>', "connectionstyle": "arc3,rad=0.0", "color": "black"}
        arrow_ax2_props3 = {"arrowstyle": '-|>', "connectionstyle": "arc3,rad=-0.075", "color": "black"}
        ax2.annotate('', xy=(1.665, 2.961), xytext=(1.147, 1.027), va='center', ha='center',
                     arrowprops=arrow_ax2_props1, fontsize=self.tiny_size, transform=ax2.transAxes)
        ax2.annotate('', xy=(3.058, 9.406), xytext=(2.154, 5.098), va='center', ha='center',
                     arrowprops=arrow_ax2_props2, fontsize=self.tiny_size, transform=ax2.transAxes)
        ax2.annotate('', xy=(4.155, 13.213), xytext=(3.553, 11.342), va='center', ha='center',
                     arrowprops=arrow_ax2_props3, fontsize=self.tiny_size, transform=ax2.transAxes)

        ########################################
        if use_inset:
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

            ax2_inset.set_xlabel('Wavenumber (nm$^{-1}$)', fontsize=self.tiny_size)
            ax2_inset.set_xlim(0, 2)
            ax2_inset.set_ylim(0, 10)
            ax2_inset.xaxis.tick_top()
            ax2_inset.xaxis.set_label_position("top")
            ax2_inset.yaxis.set_label_position("left")
            ax2_inset.set_ylabel('Frequency\n(THz)', fontsize=self.tiny_size, rotation=90, labelpad=20)
            ax2_inset.tick_params(axis='both', labelsize=self.tiny_size)
            ax2.margins(0)

            ax2_inset.patch.set_color("#f9f2e9")
            ax2_inset.yaxis.labelpad = 5
            ax2_inset.xaxis.labelpad = 2.5

            # self._tick_setter(ax2_inset, 2.5, 0.5, 3, 2, is_fft_plot=False)
            ax2_inset.ticklabel_format(axis='y', style='plain')
            ax2_inset.legend(fontsize=self.tiny_size, frameon=False)

        ########################################

        ax1.text(0.025, 0.88, f"(a)", verticalalignment='center', horizontalalignment='left',
                 transform=ax1.transAxes, fontsize=self.smaller_size)

        ax2.text(0.975, 0.12, f"(b)", verticalalignment='center', horizontalalignment='right',
                 transform=ax2.transAxes, fontsize=self.smaller_size)

        for ax in [ax1, ax2]:

            for spine in ["top", "bottom", "left", "right"]:
                ax.spines[spine].set_visible(True)

            ax.grid(False)
            ax.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=1.99)
            ax.set_axisbelow(False)

            if ax == ax1 or ax == ax2:
                ax.set_facecolor("white")

        self.fig.subplots_adjust(wspace=1, hspace=0.35)

        if interactive_plot:
            # For interactive plots
            def mouse_event(event: Any):
                print(f'x: {event.xdata} and y: {event.ydata}')

            self.fig.canvas.mpl_connect('button_press_event', mouse_event)
            self.fig.tight_layout()  # has to be here
            plt.show()
        else:
            self.fig.savefig(f"{self.output_filepath}_dispersion.png", bbox_inches="tight")

    def create_time_variation3(self, interactive_plot=False, use_inset_1=True, use_lower_plot=False):
        """
        Plot the magnetisation of a site against time.

        One should ensure that the site being plotted is not inside either of the driving- or damping-regions.

        :return: Saves a .png image to the designated output folder.
        """

        if self.fig is None:
            self.fig = plt.figure(figsize=(4.5, 3.375))

        num_rows = 2
        num_cols = 3
        if use_lower_plot:
            ax1 = plt.subplot2grid((num_rows, num_cols), (0, 0), rowspan=int(num_rows / 2),
                                   colspan=num_cols, fig=self.fig)
        else:
            ax1 = plt.subplot2grid((num_rows, num_cols), (0, 0), rowspan=int(num_rows),
                                   colspan=num_cols, fig=self.fig)

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
            ax1_inset.set_xlabel('Time (ns)', fontsize=self.tiny_size)
            ax1_inset.yaxis.tick_left()
            ax1_inset.yaxis.set_label_position("left")
            ax1_inset.set_ylabel('Amplitude  \n(arb. units)  ', fontsize=self.tiny_size, rotation=90, labelpad=20)
            ax1_inset.tick_params(axis='both', labelsize=self.mini_size)

            ax1_inset.patch.set_color("#f9f2e9")
            ax1_inset.yaxis.labelpad = 0
            ax1_inset.xaxis.labelpad = -0.5
            ax1_inset.grid(False)

            self._tick_setter(ax1_inset, 1.0, 0.25, 3, 2, is_fft_plot=False, yaxis_num_decimals=1, yscale_type='p')

        ########################################

        if use_lower_plot:
            ax2 = plt.subplot2grid((num_rows, num_cols), (int(num_rows / 2), 0),
                                   rowspan=num_rows, colspan=num_cols, fig=self.fig)

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
            self._tick_setter(ax2, 2, 0.5, 3, 2, is_fft_plot=False, xaxis_num_decimals=1, yaxis_num_decimals=0,
                              yscale_type='p')
            ax2.grid(False)

            # ax2.axhline(y=3.8, xmax=1.0, ls='--', lw=1, color='grey', zorder=0.9) # xmax=0.31
            # ax2.axhline(y=10.5, xmax=1.0, ls='--', lw=1, color='grey', zorder=0.9)  # xmax=0.68
            ax2.margins(0)
            ax2.xaxis.labelpad = -2

            # ax2.text(0.997, -0.13, r"$\mathrm{\dfrac{\pi}{a}}$",
            #          verticalalignment='center', horizontalalignment='center', transform=ax2.transAxes,
            #          fontsize=self.smaller_size)

            # ax2.text(0.02, 0.88, r"$\mathcal{III}$", verticalalignment='center', horizontalalignment='left',
            #          transform=ax2.transAxes, fontsize=self.smaller_size)
            # ax2.text(0.02, 0.5, r"$\mathcal{II}$", verticalalignment='center', horizontalalignment='left',
            #          transform=ax2.transAxes, fontsize=self.smaller_size)
            # ax2.text(0.02, 0.12, r"$\mathcal{I}$", verticalalignment='center', horizontalalignment='left',
            #          transform=ax2.transAxes, fontsize=self.smaller_size)
            #
            # ax2.text(0.91, 0.82, f"Decreasing", verticalalignment='center', horizontalalignment='center',
            #          transform=ax2.transAxes, fontsize=self.tiny_size)
            # ax2.text(0.60, 0.425, f"Linear", verticalalignment='center', horizontalalignment='center',
            #          transform=ax2.transAxes, fontsize=self.tiny_size)
            # ax2.text(0.41, 0.12, f"Increasing", verticalalignment='center', horizontalalignment='center',
            #          transform=ax2.transAxes, fontsize=self.tiny_size)

            # arrow_ax2_props1 = {"arrowstyle": '-|>', "connectionstyle": "arc3,rad=0.075", "color": "black"}
            # arrow_ax2_props2 = {"arrowstyle": '-|>', "connectionstyle": "arc3,rad=0.0", "color": "black"}
            # arrow_ax2_props3 = {"arrowstyle": '-|>', "connectionstyle": "arc3,rad=-0.075", "color": "black"}
            # ax2.annotate('', xy=(1.665, 2.961), xytext=(1.147, 1.027), va='center', ha='center',
            #              arrowprops=arrow_ax2_props1, fontsize=self.tiny_size, transform=ax2.transAxes)
            # ax2.annotate('', xy=(3.058, 9.406), xytext=(2.154, 5.098), va='center', ha='center',
            #              arrowprops=arrow_ax2_props2, fontsize=self.tiny_size, transform=ax2.transAxes)
            # ax2.annotate('', xy=(4.155, 13.213), xytext=(3.553, 11.342), va='center', ha='center',
            #              arrowprops=arrow_ax2_props3, fontsize=self.tiny_size, transform=ax2.transAxes)

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

                ax2_inset.set_xlabel('Wavenumber (nm$^{-1}$)', fontsize=self.tiny_size)
                ax2_inset.set_xlim(0, 2)
                ax2_inset.set_ylim(0, 10)
                ax2_inset.xaxis.tick_top()
                ax2_inset.xaxis.set_label_position("top")
                ax2_inset.yaxis.set_label_position("left")
                ax2_inset.set_ylabel('Frequency\n(THz)', fontsize=self.tiny_size, rotation=90, labelpad=20)
                ax2_inset.tick_params(axis='both', labelsize=self.tiny_size)
                ax2.margins(0)

                ax2_inset.patch.set_color("#f9f2e9")
                ax2_inset.yaxis.labelpad = 5
                ax2_inset.xaxis.labelpad = 2.5

                # self._tick_setter(ax2_inset, 2.5, 0.5, 3, 2, is_fft_plot=False)
                ax2_inset.ticklabel_format(axis='y', style='plain')
                ax2_inset.legend(fontsize=self.tiny_size, frameon=False)

            ########################################

            # ax1.text(0.025, 0.88, f"(a)", verticalalignment='center', horizontalalignment='left',
            #          transform=ax1.transAxes, fontsize=self.smaller_size)

            # ax2.text(0.975, 0.12, f"(b)",
            #         verticalalignment='center', horizontalalignment='right', transform=ax2.transAxes,
            #         fontsize=self.smaller_size)

        for ax in [ax1]:

            for spine in ["top", "bottom", "left", "right"]:
                ax.spines[spine].set_visible(True)

            ax.grid(False)
            ax.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=1.99)
            ax.set_axisbelow(False)
            ax.set_facecolor("white")

        self.fig.subplots_adjust(wspace=1, hspace=0.35)

        if interactive_plot:
            # For interactive plots
            def mouse_event(event: Any):
                print(f'x: {event.xdata} and y: {event.ydata}')

            self.fig.canvas.mpl_connect('button_press_event', mouse_event)
            self.fig.tight_layout()  # has to be here
            plt.show()
        else:
            self.fig.savefig(f"{self.output_filepath}_dispersion.png", bbox_inches="tight")

    def plot_fft(self, spin_site, add_zoomed_region=False):
        """
        Plot the magnitudes of the magnetic moment of a spin site against time, as well as the FFTs, over four subplots.

        :param add_zoomed_region: Add inset to plot for showing Heaviside function.
        :param int spin_site: The spin site being plotted.

        :return: A figure containing four sub-plots.
        """
        # Find maximum time in (ns) to the nearest whole (ns), then find how large shaded region should be.

        plot_set_params = {0: {"xlabel": "Time (ns)", "ylabel": "m$_x$ / M$_S$"},
                           1: {"xlabel": "Frequency (GHz)", "ylabel": "Amplitude (a.u.)",
                               "xlim": (0, 60), "ylim": (1e-4, 1e1)},
                           2: {"xlabel": "Frequency (GHz)"}}

        # Signal that varies in time #37782c

        # FFT stuff

        lower1_wave, upper1_wave = 12, 3345
        lower2_wave, upper2_wave = 3345, 5079
        lower3_wave, upper3_wave = 5079, 10000
        lower1_blob, upper1_blob = 2930, 3320
        lower2_blob, upper2_blob = 2350, 2570
        lower3_blob, upper3_blob = 1980, 2130
        frequencies_blob1, fourier_transform_blob1 = self._fft_data(
            self.amplitude_data[lower1_blob:upper1_blob, spin_site])
        frequencies_blob2, fourier_transform_blob2 = self._fft_data(
            self.amplitude_data[lower2_blob:upper2_blob, spin_site])
        frequencies_blob3, fourier_transform_blob3 = self._fft_data(
            self.amplitude_data[lower3_blob:upper3_blob, spin_site])
        frequencies_precursors, fourier_transform_precursors = self._fft_data(
            self.amplitude_data[lower1_wave:upper1_wave, spin_site])
        frequencies_dsw, fourier_transform_dsw = self._fft_data(
            self.amplitude_data[lower2_wave:upper2_wave, spin_site])
        frequencies_eq, fourier_transform_eq = self._fft_data(
            self.amplitude_data[lower3_wave:upper3_wave, spin_site])

        # FFT for blobs
        fig = plt.figure()
        # ax2 = plt.subplot2grid((4, 8), (0, 0), rowspan=4, colspan=8)
        ax2 = fig.add_subplot(111)
        ax2.plot(frequencies_blob1, abs(fourier_transform_blob1), marker='', lw=1, color='#37782c',
                 markerfacecolor='black', markeredgecolor='black', ls=':')
        ax2.plot(frequencies_blob2, abs(fourier_transform_blob2), marker='', lw=1, color='#37782c',
                 markerfacecolor='black', markeredgecolor='black', ls='--')
        ax2.plot(frequencies_blob3, abs(fourier_transform_blob3), marker='', lw=1, color='#37782c',
                 markerfacecolor='black', markeredgecolor='black', ls='-.')
        ax2.plot(frequencies_precursors, abs(fourier_transform_precursors), marker='', lw=1, color='#37782c',
                 markerfacecolor='black', markeredgecolor='black', label="Precursors", zorder=5)
        ax2.plot(frequencies_dsw, abs(fourier_transform_dsw), marker='', lw=1, color='#64bb6a',
                 markerfacecolor='black', markeredgecolor='black', label="Shockwave", zorder=2)
        ax2.plot(frequencies_eq, abs(fourier_transform_eq), marker='', lw=1, color='#9fd983',
                 markerfacecolor='black', markeredgecolor='black', label="Steady State", zorder=1)

        ax2.set(**plot_set_params[1], yscale='log')
        arrow_ax2_props = {"arrowstyle": '-|>', "connectionstyle": "angle3,angleA=0,angleB=90", "color": "black"}
        ax2.annotate('P1', xy=(24.22, 0.029), xytext=(26.31, 0.231), va='center', ha='center',
                     arrowprops=arrow_ax2_props, fontsize=8)
        ax2.annotate('P2', xy=(36.48, 0.0096), xytext=(39.91, 0.13), va='center', ha='center',
                     arrowprops=arrow_ax2_props, fontsize=8)
        ax2.annotate('P3', xy=(52.00, 0.0045), xytext=(56.25, 0.075), va='center', ha='center',
                     arrowprops=arrow_ax2_props, fontsize=8)
        ax2.legend(ncol=1, fontsize=6, frameon=False, fancybox=True, facecolor=None, edgecolor=None,
                   bbox_to_anchor=[0.7, 0.65])
        self._tick_setter(ax2, 20, 5, 3, 4, is_fft_plot=True)

        for ax in [ax2]:
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            # ax.set_facecolor('#f4f4f5')
            ax.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=6)

        # Add zoomed in region if needed.
        if add_zoomed_region:
            # Select datasets to use
            x = self.time_data[:]
            y = self.amplitude_data[:, spin_site]

            # Impose inset onto plot. Treat as a separate subplot. Use 0.24 for LHS and 0.8 for RHS. 0.7 for TR and
            ax2_inset = inset_axes(ax2, width=2.4, height=0.625, loc="lower left", bbox_to_anchor=[0.14, 0.625],
                                   bbox_transform=ax2.figure.transFigure)
            ax2_inset.plot(x, y, lw=0.75, color='#37782c')

            # Select data (of original) to show in inset through changing axis limits
            ax2_inset.set_xlim(1.25, 2.5)
            ax2_inset.set_ylim(-0.2e-3, 0.2e-3)

            # Remove tick labels
            ax2_inset.set_xticks([])
            ax2_inset.set_yticks([])
            ax2_inset.patch.set_color("#f9f2e9")  # #f0a3a9 is equivalent to color 'red' and alpha '0.3' fbe3e5
            # ax2_inset.patch.set_alpha(0.3)
            # ax2_inset.set(facecolor='red', alpha=0.3)
            # mark_inset(self.axes, ax2_inset,loc1=1, loc2=2, facecolor="red", edgecolor=None, alpha=0.3)

            # Add box to indicate the region which is being zoomed into on the main figure
            ax2.indicate_inset_zoom(ax2_inset, facecolor='#f9f2e9', edgecolor='black', alpha=1.0, lw=0.75, zorder=1)
            arrow_inset_props = {"arrowstyle": '-|>', "connectionstyle": "angle3,angleA=0,angleB=90", "color": "black"}
            ax2_inset.annotate('P1', xy=(2.228, -1.5e-4), xytext=(1.954, -1.5e-4), va='center', ha='center',
                               arrowprops=arrow_inset_props, fontsize=6)
            ax2_inset.annotate('P2', xy=(1.8, -8.48e-5), xytext=(1.407, -1.5e-4), va='center', ha='center',
                               arrowprops=arrow_inset_props, fontsize=6)
            ax2_inset.annotate('P3', xy=(1.65, 6e-5), xytext=(1.407, 1.5e-4), va='center', ha='center',
                               arrowprops=arrow_inset_props, fontsize=6)

        # Add spines to all plots (to override any rcParams elsewhere in the code
        for spine in ['top', 'bottom', 'left', 'right']:
            # ax1_inset.spines[spine].set_visible(True)
            # ax1.spines[spine].set_visible(True)
            ax2.spines[spine].set_visible(True)

        ax2.set_axisbelow(False)
        fig.savefig(f"{self.output_filepath}_site{spin_site}_fft.png", bbox_inches="tight")

        early_exit = True
        if early_exit:
            exit(0)

        ax = plt.subplot2grid((4, 8), (2, 0), rowspan=2, colspan=8)
        SAMPLE_RATE = int(5e2)  # Number of samples per nanosecond
        DURATION = int(15)  # Nanoseconds

        def generate_sine_wave(freq, sample_rate, duration, delay_num):
            delay = int(sample_rate * delay_num)
            t = np.linspace(0, duration, sample_rate * duration, endpoint=False)
            y_1 = np.zeros(delay)
            y_2 = np.sin((2 * np.pi * freq) * t[delay:])
            y_con = np.concatenate((y_1, y_2))
            return t, y_con

        # Generate a 15 GHz sine wave that lasts for 5 seconds
        x1, y1 = generate_sine_wave(15, SAMPLE_RATE, DURATION, 1)
        x2, y2 = generate_sine_wave(15, SAMPLE_RATE, DURATION, 0)
        from scipy.fft import rfft, rfftfreq
        # Number of samples in normalized_tone
        n1 = int(SAMPLE_RATE * DURATION)
        n2 = int(SAMPLE_RATE * DURATION)

        y1f = rfft(y1)
        y2f = rfft(y2)
        x1f = rfftfreq(n1, 1 / SAMPLE_RATE)
        x2f = rfftfreq(n2, 1 / SAMPLE_RATE)

        ax.plot(x1f, np.abs(y1f), marker='', lw=1.0, color='#ffb55a', markerfacecolor='black', markeredgecolor='black',
                label="1", zorder=1.2)
        ax.plot(x2f, np.abs(y2f), marker='', lw=1.0, ls='-', color='#64bb6a', markerfacecolor='black',
                markeredgecolor='black',
                label="0", zorder=1.3)

        ax.set(xlim=(5, 25), ylim=(1e0, 1e4),
               xlabel="Frequency (GHz)", yscale='log')
        ax.grid(visible=False, axis='both', which='both')
        ax.tick_params(top="on", right="on", which="both")

        ax.set_ylabel("Amplitude (arb. units)\n", x=-10, y=1)

        ax.text(0.04, 0.88, f"(b)",
                verticalalignment='center', horizontalalignment='left', transform=ax.transAxes, fontsize=8)
        ax2.text(0.04, 0.88, f"(a)",
                 verticalalignment='center', horizontalalignment='left', transform=ax2.transAxes, fontsize=8)

        ax2_inset = inset_axes(ax, width=1.8, height=0.7, loc="upper right", bbox_to_anchor=[0.88, 0.47],
                               bbox_transform=ax.figure.transFigure)
        ax2_inset.plot(x1, y1, lw=0.5, color='#ffb55a', zorder=1.2)
        ax2_inset.plot(x2, y2, lw=0.5, ls='--', color='#64bb6a', zorder=1.1)
        ax2_inset.grid(visible=False, axis='both', which='both')
        ax2_inset.yaxis.tick_right()
        ax2_inset.set_xlabel('Time (ns)', fontsize=8)
        ax2_inset.set(xlim=[0, 2])
        ax2_inset.yaxis.set_label_position("right")
        ax2_inset.set_ylabel('Amplitude (arb. units)', fontsize=8, rotation=-90, labelpad=10)
        ax2_inset.tick_params(axis='both', labelsize=8)

        ax2_inset.patch.set_color("#f9f2e9")

        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_visible(True)
            ax2_inset.spines[spine].set_visible(True)

        ax.xaxis.set(major_locator=ticker.MultipleLocator(5),
                     minor_locator=ticker.MultipleLocator(1))
        ax2_inset.xaxis.set(major_locator=ticker.MultipleLocator(1),
                            minor_locator=ticker.MultipleLocator(0.2))
        ax2_inset.xaxis.labelpad = -0.5

        ax.set_axisbelow(False)
        ax2_inset.set_axisbelow(False)

        # ax.legend(title="$\Delta t$ (ns)", ncol=1, loc="upper left",
        #          frameon=False, fancybox=False, facecolor='white', edgecolor='black',
        #          fontsize=8, title_fontsize=10,
        #          bbox_to_anchor=(0.14, 0.475), bbox_transform=ax.figure.transFigure
        #          ).set_zorder(4)
        fig.subplots_adjust(wspace=0.1, hspace=-0.3)
        fig.tight_layout()
        # def mouse_event(event):
        #    print('x: {} and y: {}'.format(event.xdata, event.ydata))
        #
        # cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
        # plt.show()
        fig.savefig(f"{self.output_filepath}_site{spin_site}_fft.png", bbox_inches="tight")

    def _fft_data(self, amplitude_data):
        """
        Computes the FFT transform of a given signal, and also outputs useful data such as key frequencies.

        :param amplitude_data: Magnitudes of magnetic moments for a spin site

        :return: A tuple containing the frequencies [0], FFT [1] of a spin site. Also includes the  natural frequency
        (1st eigenvalue) [2], and driving frequency [3] for the system.
        """

        # Find bin size by dividing the simulated time into equal segments based upon the number of data-points.
        sample_spacing = (self.max_time / (self.data_points - 1))

        # Compute the FFT
        n = amplitude_data.size
        normalised_data = amplitude_data

        fourier_transform = rfft(normalised_data)
        frequencies = rfftfreq(n, sample_spacing)

        return frequencies, fourier_transform

    def _tick_setter(self, ax, x_major, x_minor, y_major, y_minor, yaxis_multi_loc=False, is_fft_plot=False,
                     xaxis_num_decimals=1, yaxis_num_decimals=1, yscale_type='f'):

        if ax is None:
            ax = plt.gca()

        if is_fft_plot:
            ax.xaxis.set(major_locator=ticker.MultipleLocator(x_major),
                         major_formatter=ticker.FormatStrFormatter("%.1f"),
                         minor_locator=ticker.MultipleLocator(x_minor))
            ax.yaxis.set(major_locator=ticker.LogLocator(base=10, numticks=y_major))
            locmin = ticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=y_minor)
            ax.yaxis.set_minor_locator(locmin)
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())

            # ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        else:
            ax.xaxis.set(major_locator=ticker.MultipleLocator(x_major),
                         major_formatter=ticker.FormatStrFormatter(f"%.{xaxis_num_decimals}f"),
                         minor_locator=ticker.MultipleLocator(x_minor))
            ax.yaxis.set(major_locator=ticker.MaxNLocator(nbins=y_major, prune='lower'),
                         major_formatter=ticker.FormatStrFormatter(f"%.{yaxis_num_decimals}f"),
                         minor_locator=ticker.AutoMinorLocator(y_minor))

            if yaxis_multi_loc:
                ax.yaxis.set(major_locator=ticker.MultipleLocator(y_major),
                             major_formatter=ticker.FormatStrFormatter(f"%.{yaxis_num_decimals}f"),
                             minor_locator=ticker.MultipleLocator(y_minor))

            class ScalarFormatterClass(ticker.ScalarFormatter):
                def _set_format(self):
                    self.format = f"%.{yaxis_num_decimals}f"

            yScalarFormatter = ScalarFormatterClass(useMathText=True)
            yScalarFormatter.set_powerlimits((0, 0))
            # ax.xaxis.set_major_formatter(yScalarFormatter)
            ax.yaxis.set_major_formatter(yScalarFormatter)

            if yscale_type == 'p':
                ax.ticklabel_format(axis='y', style='plain')

            # ax.yaxis.labelpad = -3

            ax.yaxis.get_offset_text().set_visible(False)
            ax.yaxis.get_offset_text().set_fontsize(8)
            t = ax.yaxis.get_offset_text()
            t.set_x(-0.045)

        return ax

    def ricardo_paper(self, interactive_plot=False):
        """
        Plot the magnetisation of a site against time.

        One should ensure that the site being plotted is not inside either of the driving- or damping-regions.

        :return: Saves a .png image to the designated output folder.
        """

        ########################################
        for val in [4, 5, 6]:
            # filename = f"D:/Data/2023-03-08/Simulation_Data/Ricardo Data/dataFig{val}.csv"
            filename = (f'/Users/cameronmceleney/CLionProjects/Data/2023-03-08/Simulation_Data/'
                        f'Ricardo Data/dataFig{val}.csv')
            dataset = np.loadtxt(filename, skiprows=1, delimiter=',', dtype='float')

            fig = plt.figure(figsize=(3.375, 3.375 / 2))
            ax = fig.add_subplot(111)

            xaxis_data = dataset[:, 0]
            yaxis_data1 = dataset[:, 1]

            yaxis_data2 = None
            colour1, colour2 = None, None
            label1, label2 = None, None
            leg_loc, leg_pos = None, None

            ax_lw = 2.25
            ax_s = 10

            if val == 4:
                ax.set(xlabel=r"Time Increment", ylabel=r"$\Delta d_{cores}$ (nm)",
                       xlim=[0, 7999.9], ylim=[-0.001, 60])

                ax.yaxis.labelpad = 0

                # ax.text(0.925, 0.1, '(a)', va='center', ha='center',
                #         transform=ax.transAxes, fontsize=self.smaller_size)

                self._tick_setter(ax, 2e3, 1e3, 4, 2,
                                  xaxis_num_decimals=0, yaxis_num_decimals=0, yscale_type='p')
                colour1 = '#EA653A'  # orange

            if val in [5, 6]:
                yaxis_data2 = dataset[:, 2]

                ax_s = 24

                colour1 = '#3A9846'  # green
                colour2 = '#5584B9'  # blue

                label1, label2 = 'Bloch', 'Néel'

            ax.plot(xaxis_data, yaxis_data1, ls='-', lw=ax_lw, color=colour1, alpha=0.5,
                    zorder=1.01)
            ax.scatter(xaxis_data, yaxis_data1, color=colour1, marker='o', s=ax_s, fc=colour1, ec='None',
                       label=label1, zorder=1.01)

            if val in [5, 6]:
                if val == 5:
                    leg_loc = "upper right"
                    leg_pos = (0.96, 0.95)

                    ax.set(xlabel=r"Angular Frequency (1)", ylabel="Avg. Velocity (10$^{-3}$)",
                           xlim=[0.095, 0.355])

                    # ax.text(0.075, 0.1, '(b)', va='center', ha='center', transform=ax.transAxes,
                    #         fontsize=self.smaller_size)

                    self._tick_setter(ax, 0.1, 0.05, 3, 2,
                                      xaxis_num_decimals=1, yaxis_num_decimals=0, yscale_type='')

                    # ax.text(-0.02, 1.05, r'$\times \mathcal{10}^{{\mathcal{' + str(int(-3)) + r'}}}$',
                    #        verticalalignment='center',
                    #        horizontalalignment='center', transform=ax.transAxes, fontsize=self.smaller_size)

                if val == 6:
                    leg_loc = "upper left"
                    leg_pos = (0.02, 0.95)

                    ax.set(xlabel=r"Pumping Field (1)", ylabel="Avg. Velocity (10$^{-3}$)",
                           xlim=[0.1425, 0.3075], ylim=[0.00185, 0.0081])

                    self._tick_setter(ax, 0.1, 0.025, 4, 2,
                                      xaxis_num_decimals=1, yaxis_num_decimals=0, yscale_type='')
                    # ax.text(0.925, 0.1, '(c)', va='center', ha='center', transform=ax.transAxes,
                    #         fontsize=self.smaller_size)

                    # ax.text(-0.02, 1.05, r'$\times \mathcal{10}^{{\mathcal{' + str(int(-3)) + r'}}}$',
                    #        verticalalignment='center',
                    #        horizontalalignment='right', transform=ax.transAxes, fontsize=self.smaller_size)

                ax.plot(xaxis_data, yaxis_data2, ls='-', lw=ax_lw, color=colour2, alpha=0.5,
                        zorder=1.01)
                ax.scatter(xaxis_data, yaxis_data2, color=colour2, marker='o', s=ax_s, fc=colour2, ec='None',
                           label=label2, zorder=1.01)

                ax.legend(ncol=1, loc=leg_loc, handletextpad=-0.25,
                          frameon=False, fancybox=False, facecolor='None', edgecolor='black',
                          fontsize=self.tiny_size, bbox_to_anchor=leg_pos, bbox_transform=ax.transAxes).set_zorder(4)

            for axis in [ax]:
                axis.xaxis.grid(False)
                axis.yaxis.grid(False)
                # ax.set_facecolor('#f4f4f5')
                ax.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=1.9999)

                # Add spines to all plots (to override any rcParams elsewhere in the code
                for spine in ['top', 'bottom', 'left', 'right']:
                    axis.spines[spine].set_visible(True)

                axis.set_axisbelow(False)
                axis.set_facecolor('white')

            if interactive_plot:
                # For interactive plots
                def mouse_event(event: Any):
                    print(f'x: {event.xdata} and y: {event.ydata}')

                self.fig.canvas.mpl_connect('button_press_event', mouse_event)
                self.fig.tight_layout()  # has to be here
                plt.show()
            else:
                fig.savefig(f"{self.output_filepath}_data{val}.png", bbox_inches="tight")


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

        ax.grid(lw=2, ls='-')

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
        axes = fig.add_subplot(111)

        colour2 = '#5584B9'
        sns.lineplot(x=range(0, len(mx_mode)), y=mx_mode, marker='o', markersize=5,
                     linestyle='', alpha=1, ax=axes, color=colour2, label='Mx')
        sns.lineplot(x=range(0, len(mx_mode)), y=mx_mode, lw=1.75,
                     linestyle='-', alpha=0.5, ax=axes, color=colour2)

        axes.set(xlabel="Distance (nm)", ylabel="Amplitude (arb. units)",
                 xlim=(0, number_of_spins))  # ,
        # title = f"Eigenmode #{eigenmode}",
        # xticks=np.arange(0, number_of_spins, np.floor(number_of_spins - 2) / 20))

        axes.xaxis.set(major_locator=ticker.MultipleLocator(50),
                       minor_locator=ticker.MultipleLocator(10))
        axes.yaxis.set(major_locator=ticker.MultipleLocator(0.1),
                       minor_locator=ticker.MultipleLocator(0.025))

        axes.text(0.025, 0.925, "(b)", verticalalignment='center', horizontalalignment='left',
                  transform=axes.transAxes, fontsize=8)

        # Legend doubles as a legend (showing propagation direction), and the frequency [Hz] of the eigenmode.
        axes.legend(loc=1, bbox_to_anchor=(0.975, 0.975),
                    frameon=True, fancybox=True, facecolor='white', edgecolor='white',
                    title=f"Frequency (GHz)\n        {frequency: 4.1f}\n     Component",
                    fontsize=8, title_fontsize=8)

        axes.axvspan(0, number_of_spins * driving_width, color='black', alpha=0.2)

        axes.grid(color='black', ls='--', alpha=0.0, lw=1)

        for axis in [axes]:
            axis.xaxis.grid(False)
            axis.yaxis.grid(False)
            # ax.set_facecolor('#f4f4f5')
            axis.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=1.9999)

            # Add spines to all plots (to override any rcParams elsewhere in the code
            for spine in ['top', 'bottom', 'left', 'right']:
                axis.spines[spine].set_visible(True)

            axis.set_axisbelow(False)
            axis.set_facecolor('white')

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
            ax.grid(False)
            ax.legend(frameon=False)

            for spine in ["top", "bottom", "left", "right"]:
                ax.spines[spine].set_visible(True)

            ax.set_axisbelow(False)
            ax.set_facecolor("white")

        fig.savefig(f"{self.output_filepath}_DispersionRelation.png", bbox_inches="tight")

        print("--------------------------------------------------------------------------------")


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
