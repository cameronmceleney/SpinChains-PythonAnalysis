#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For interactive plots on Mac
# import matplotlib
# matplotlib.use('macosx')

# Standard modules (common)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys as sys

# Third party modules (uncommon)
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.transforms as mtrans
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import gif as gif
from scipy.fft import rfft, rfftfreq

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

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
        self.stop_itervals = key_data['stopIterVal']
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
        self.fig = plt.figure(figsize=(3.5, 1.3))
        self.axes = self.fig.add_subplot(111)
        self.y_axis_limit = max(self.amplitude_data[-1, :]) * 1.1  # Add a 10% margin to the y-axis.
        self.kwargs = {"xlabel": f"Site Number [$N_i$]", "ylabel": f"m$_x$ / M$_S$",
                       "xlim": [0, self.number_spins], "ylim": [-1 * self.y_axis_limit, self.y_axis_limit]}

    def _draw_figure(self, plot_row=-1, has_single_figure=True, draw_regions_of_interest=True):
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

        # Easier to have time-stamp as label than textbox.
        self.axes.plot(np.arange(0, self.number_spins), self.amplitude_data[plot_row, :], ls='-', lw=0.75,
                       label=f"{self.time_data[plot_row]:2.2f} (ns)", color='#64bb6a')

        self.axes.set(**self.kwargs)

        # self.axes.text(-0.04, 0.96, r'$\times \mathcal{10}^{{\mathcal{-3}}}$', verticalalignment='center',
        # horizontalalignment='center', transform=self.axes.transAxes, fontsize=6)
        self.axes.text(0.88, 0.88, f"(c) {self.time_data[plot_row]:2.3f} ns",
                       verticalalignment='center', horizontalalignment='center', transform=self.axes.transAxes,
                       fontsize=6)

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
        self.axes.clear()  # Use this if looping through a single PaperFigures object for multiple create_png inputs
        self._draw_figure(row_number)

        self.axes.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        if should_add_data:
            # Add text to figure with simulation parameters
            if self.exchange_min == self.exchange_max:
                exchangeString = f"Uniform Exc. : {self.exchange_min} (T)"
            else:
                exchangeString = f"J$_{{min}}$ = {self.exchange_min} (T) | J$_{{max}}$ = " \
                                 f"{self.exchange_max} (T)"
            data_string = f"H$_{{0}}$ = {self.static_field} (T) | N = {self.chain_spins} | " + r"$\alpha$" \
                                                                                               f" = {self.gilbert_factor: 2.2e}\n" \
                                                                                               f"H$_{{D1}}$ = {self.driving_field1:2.2e} (T) | " \
                                                                                               f"H$_{{D2}}$ = {self.driving_field2:2.2e} (T) \n{exchangeString}"

            props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.5)
            # Place text box in upper left in axes coords
            self.axes.text(0.05, 1.2, data_string, transform=self.axes.transAxes, fontsize=12,
                           verticalalignment='top', bbox=props, ha='center', va='center')

        # Add spines to all plots (to override any rcParams elsewhere in the code
        for spine in ['top', 'bottom', 'left', 'right']:
            self.axes.spines[spine].set_visible(True)

        self.axes.grid(visible=False, axis='both', which='both')

        self.fig.savefig(f"{self.output_filepath}_row{row_number}.png", bbox_inches="tight")

    @gif.frame
    def _plot_paper_gif(self, index):
        """
        Private method to save a given row of a data as a frame suitable for use with the git library.

        Require decorator so use method as an inner class instead of creating child class.

        :param int index: The row to be plotted.
        """
        self._draw_figure(index, False, True)

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
                              basic_annotations=False,
                              add_zoomed_region=False, add_info_box=False, add_coloured_regions=False):
        """
        Plot the magnetisation of a site against time.

        One should ensure that the site being plotted is not inside either of the driving- or damping-regions.

        :param annotate_precursors: Add arrows to denote precursors.
        :param colour_precursors: Draw 1st, 3rd and 5th precursors as separate colours to main figure.
        :param bool add_coloured_regions: Draw coloured boxes onto plot to show driving- and damping-regions.
        :param bool add_info_box: Add text box to base of plot which lists key simulation parameters.
        :param bool add_zoomed_region: Add inset to plot to focus upon precursors.
        :param int spin_site: The number of the spin site to be plotted.

        :return: Saves a .png image to the designated output folder.
        """
        small_font = 6 * 1.25
        mid_font = 8 * 1.25
        self.axes.clear()
        self.axes.set_aspect('auto')
        fig = plt.figure()
        num_rows = 2
        num_cols = 2
        ax1 = plt.subplot2grid((num_rows, num_cols), (0, 0), rowspan=int(num_rows / 2), colspan=num_cols)
        ax2 = plt.subplot2grid((num_rows, num_cols), (int(num_rows / 2), 0), rowspan=num_rows, colspan=num_cols)

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

        ax1.set(xlabel=f"Time (ns)", ylabel=f"Magnetisation (T)",
                xlim=[ax1_xlim_lower, ax1_xlim_upper],
                ylim=[-ax1_yaxis_base * ax1_yaxis_order * 1.4, ax1_yaxis_base * ax1_yaxis_order])

        ax2.set(xlabel=f"Frequency (GHz)", ylabel=f"Amplitude (arb. units)",
                xlim=[0, 100], ylim=[1e-1, 1e3], yscale='log')

        self._tick_setter(ax1, ax1_xlim_range * 0.5, ax1_xlim_range * 0.125, 3, 4, xaxis_num_decimals=2)
        self._tick_setter(ax2, 20, 5, 6, None, is_fft_plot=True)

        line_height = -3.15 * ax1_yaxis_order

        ########################################

        if ax1_xlim_lower > ax1_xlim_upper: exit(0)

        def convert_norm(val, a=0, b=1):
            # return int(self.data_points * (2 * ax1_xlim_lower + ( a * (xlim_signal - ax1_xlim_lower) / xlim_max )))  # original
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
            text_height = line_height - ax1_yaxis_base * 0.2 * ax1_yaxis_order
            axes_props1 = {"arrowstyle": '|-|, widthA =0.3, widthB=0.3', "color": "#37782c", 'lw': 0.8}
            axes_props2 = {"arrowstyle": '|-|, widthA =0.3, widthB=0.3', "color": "#64bb6a", 'lw': 0.8}
            axes_props3 = {"arrowstyle": '|-|, widthA =0.3, widthB=0.3', "color": "#9fd983", 'lw': 0.8}

            ax1.text(0.95, 0.9, f"(b)",
                     va='center', ha='center', fontsize=mid_font, transform=ax1.transAxes)

            ax2.text(0.05, 0.9, f"(c)",
                     va='center', ha='center', fontsize=mid_font,
                     transform=ax2.transAxes)

            pre_text_lhs, pre_text_rhs = lower1, upper1
            shock_text_lhs, shock_text_rhs = (lower2), (upper2)
            equib_text_lhs, equib_text_rhs = (lower3), (upper3)

            ax1.annotate('', xy=(pre_text_lhs, line_height), xytext=(pre_text_rhs, line_height),
                         va='center', ha='center', arrowprops=axes_props1, fontsize=small_font)
            ax1.annotate('', xy=(shock_text_lhs, line_height), xytext=(shock_text_rhs, line_height),
                         va='center', ha='center', arrowprops=axes_props2, fontsize=small_font)
            ax1.annotate('', xy=(equib_text_lhs, line_height), xytext=(equib_text_rhs, line_height),
                         va='center', ha='center', arrowprops=axes_props3, fontsize=small_font)
            ax1.text((pre_text_lhs + pre_text_rhs) / 2, text_height, 'Precursors', ha='center', va='bottom',
                     fontsize=small_font)
            ax1.text((shock_text_lhs + shock_text_rhs) / 2, text_height, 'Shockwave', ha='center', va='bottom',
                     fontsize=small_font)
            ax1.text((equib_text_lhs + equib_text_rhs) / 2, text_height, 'Equilibrium', ha='center', va='bottom',
                     fontsize=small_font)

        # Use these for paper publication figures
        ax1.text(-0.03, 1.02, r'$\times \mathcal{10}^{{\mathcal{' + str(int(ax1_yaxis_exponent)) + r'}}}$',
                 verticalalignment='center',
                 horizontalalignment='center', transform=ax1.transAxes, fontsize=mid_font)
        # ax1.text(0.04, 0.1, f"(a) 15 GHz", verticalalignment='center', horizontalalignment='left',
        #               transform=ax1.transAxes, fontsize=6)

        # Add zoomed in region if needed.
        if add_zoomed_region:
            # Select datasets to use
            x = self.time_data
            y = self.amplitude_data[:, spin_site]

            # Impose inset onto plot. Use 0.24 for LHS and 0.8 for RHS. 0.7 for T and 0.25 for B
            ax1_inset = inset_axes(ax1, width=2.3, height=0.7, loc="upper left",
                                   bbox_to_anchor=[0.01, 1.125], bbox_transform=ax1.transAxes)
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
                               arrowprops=arrow_ax1_props, fontsize=small_font)
            ax1_inset.annotate('P2', xy=(1.5, 7e-5), xytext=(1.1, 1.3e-4), va='center', ha='center',
                               arrowprops=arrow_ax1_props2, fontsize=small_font)
            ax1_inset.annotate('P3', xy=(1.2, -2e-5), xytext=(0.8, -1.3e-4), va='center', ha='center',
                               arrowprops=arrow_ax1_props, fontsize=small_font)

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
            textstr = f"H$_{{0}}$ = {self.static_field} [T] | " \
                      f"H$_{{D1}}$ = {self.driving_field1:2.2e} [T] | " \
                      f"H$_{{D2}}$ = {self.driving_field2:2.2e}[T] \n" \
                      f"f = {self.driving_freq} [GHz] |" \
                      f"{exchangeString} | N = {self.chain_spins} | " + r"$\alpha$" + \
                      f" = {self.gilbert_factor: 2.2e}"

            props = dict(boxstyle='round', facecolor='gainsboro', alpha=1.0)

            # place a text box in upper left in axes coords
            ax1.text(0.35, -0.22, textstr, transform=ax1.transAxes, fontsize=6,
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
            ax2.annotate('P1', xy=(26, 1.8e1), xytext=(30.1, 7.4e1), va='center', ha='center',
                         arrowprops=arrow_ax2_props, fontsize=mid_font)
            ax2.annotate('P2', xy=(48.78, 4.34e0), xytext=(53.25, 2.1e1), va='center', ha='center',
                         arrowprops=arrow_ax2_props, fontsize=mid_font)
            ax2.annotate('P3', xy=(78.29, 1.25e0), xytext=(81.78, 6.66e0), va='center', ha='center',
                         arrowprops=arrow_ax2_props, fontsize=mid_font)

        ax2.legend(ncol=1, fontsize=small_font, frameon=False, fancybox=True, facecolor=None, edgecolor=None,
                   bbox_to_anchor=[0.78, 0.60], bbox_transform=ax2.transAxes)

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

        fig.subplots_adjust(wspace=1, hspace=0.3)

        if False:
            # For interactive plots
            def mouse_event(event):
                print('x: {} and y: {}'.format(event.xdata, event.ydata))

            fig.canvas.mpl_connect('button_press_event', mouse_event)
            fig.tight_layout()  # has to be here
            plt.show()
        else:
            fig.savefig(f"{self.output_filepath}_site{spin_site}.pdf", bbox_inches="tight")

    def create_time_variation2(self, spin_site, colour_precursors=False, annotate_precursors=False,
                               basic_annotations=False,
                               add_zoomed_region=False, add_info_box=False, add_coloured_regions=False):
        """
        Plot the magnetisation of a site against time.

        One should ensure that the site being plotted is not inside either of the driving- or damping-regions.

        :param annotate_precursors: Add arrows to denote precursors.
        :param colour_precursors: Draw 1st, 3rd and 5th precursors as separate colours to main figure.
        :param bool add_coloured_regions: Draw coloured boxes onto plot to show driving- and damping-regions.
        :param bool add_info_box: Add text box to base of plot which lists key simulation parameters.
        :param bool add_zoomed_region: Add inset to plot to focus upon precursors.
        :param int spin_site: The number of the spin site to be plotted.

        :return: Saves a .png image to the designated output folder.
        """
        small_font = 6 * 1.25
        mid_font = 8 * 1.25
        self.axes.clear()
        self.axes.set_aspect('auto')
        fig = plt.figure()
        num_rows = 2
        num_cols = 2

        ax1 = plt.subplot2grid((num_rows, num_cols), (0, 0), rowspan=int(num_rows / 2), colspan=num_cols)

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
        n1 = n2 = int(SAMPLE_RATE * DURATION)

        y1f, y2f = rfft(y1), rfft(y2)
        x1f, x2f = rfftfreq(n1, 1 / SAMPLE_RATE), rfftfreq(n2, 1 / SAMPLE_RATE)

        ax1.plot(x1f, np.abs(y1f), marker='', lw=1.0, color='#ffb55a', markerfacecolor='black', markeredgecolor='black',
                 label="1", zorder=1.2)
        ax1.plot(x2f, np.abs(y2f), marker='', lw=1.0, ls='-', color='#64bb6a', markerfacecolor='black',
                 markeredgecolor='black',
                 label="0", zorder=1.3)

        ax1.set(xlim=(5, 25), ylim=(1e0, 1e4),
                xlabel="Frequency (GHz)", ylabel="Amplitude (arb. units)", yscale='log')

        self._tick_setter(ax1, 5, 1, 3, 4, is_fft_plot=True)

        ########################################

        ax1_inset = inset_axes(ax1, width=1.65, height=0.725, loc="upper right", bbox_to_anchor=[0.99, 0.98],
                               bbox_transform=ax1.transAxes)
        ax1_inset.plot(x1, y1, lw=0.5, color='#ffb55a', zorder=1.2)
        ax1_inset.plot(x2, y2, lw=0.5, ls='--', color='#64bb6a', zorder=1.1)

        ax1_inset.set(xlim=[0, 2])
        ax1_inset.set_xlabel('Time (ns)', fontsize=mid_font)
        ax1_inset.yaxis.tick_left()
        ax1_inset.yaxis.set_label_position("left")
        ax1_inset.set_ylabel('Amplitude\n(arb. units)', fontsize=mid_font, rotation=90, labelpad=20)
        ax1_inset.tick_params(axis='both', labelsize=mid_font)

        ax1_inset.patch.set_color("#f9f2e9")
        ax1_inset.yaxis.labelpad = 0
        ax1_inset.xaxis.labelpad = -0.5

        self._tick_setter(ax1_inset, 1.0, 0.25, 3, 2, is_fft_plot=False, yaxis_num_decimals=1, yscale_type='p')

        ########################################

        ax2 = plt.subplot2grid((num_rows, num_cols), (int(num_rows / 2), 0), rowspan=num_rows, colspan=num_cols)

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

        ax2.plot(wave_number_array * hz_2_GHz, freq_array * hz_2_THz,color='red', ls='-', label=f'Dataset 1')
        ax2.plot(wave_number_array * hz_2_GHz, gyromag_ratio * (external_field + exchange_field * lattice_constant**2 * wave_number_array**2) * hz_2_THz, color='red', alpha=0.25, ls='-', label=f'Dataset 1')

        # These!!
        # ax2.scatter(np.arccos(1 - ((freqs2 / gamma - h_0) / (2 * h_ex))) / a, freqs2 / 1e12, s=0.5, c='red', label='paper')
        # ax2.plot(k, gamma * (2 * h_ex * (1 - np.cos(k * a)) + h_0) / 1e12, color='red', ls='--', label=f'Kittel')

        ax2.set(xlabel="Wavenumber (nm$^{-1}$)", ylabel='Frequency (THz)', ylim=[0, 15.4])
        self._tick_setter(ax2, 2, 0.5, 3, 4, is_fft_plot=False, xaxis_num_decimals=2, yaxis_num_decimals=0, yscale_type='p')
        ax2.grid(False)

        ax2.axhline(y=3.8, xmax=1.0, ls='--', lw=1, color='grey', zorder=0.9)  # xmax=0.31
        ax2.axhline(y=10.5, xmax=1.0, ls='--', lw=1, color='grey', zorder=0.9)  # xmax=0.68
        ax2.margins(0)

        ax2.text(0.995, -0.1, r"$\mathrm{\dfrac{\pi}{a}}$",
                 verticalalignment='center', horizontalalignment='left', transform=ax2.transAxes, fontsize=mid_font)

        ax2.text(0.02, 0.88, r"$\mathcal{III}$",
                 verticalalignment='center', horizontalalignment='left', transform=ax2.transAxes, fontsize=mid_font)
        ax2.text(0.02, 0.5, r"$\mathcal{II}$",
                 verticalalignment='center', horizontalalignment='left', transform=ax2.transAxes, fontsize=mid_font)
        ax2.text(0.02, 0.12, r"$\mathcal{I}$",
                 verticalalignment='center', horizontalalignment='left', transform=ax2.transAxes, fontsize=mid_font)

        ax2.text(0.875, 0.82, f"-ve slope",
                 verticalalignment='center', horizontalalignment='center', transform=ax2.transAxes, fontsize=mid_font)
        ax2.text(0.625, 0.45, f"linear\nslope",
                 verticalalignment='center', horizontalalignment='center', transform=ax2.transAxes, fontsize=mid_font)
        ax2.text(0.375, 0.12, f"+ve slope",
                 verticalalignment='center', horizontalalignment='center', transform=ax2.transAxes, fontsize=mid_font)

        arrow_ax2_props1 = {"arrowstyle": '-|>', "connectionstyle": "arc3,rad=0.0", "color": "black"}
        arrow_ax2_props2 = {"arrowstyle": '-|>', "connectionstyle": "arc3,rad=0.1", "color": "black"}
        arrow_ax2_props3 = {"arrowstyle": '-|>', "connectionstyle": "arc3,rad=-0.1", "color": "black"}
        # ax2.annotate('', xy=(2400, 9000), xytext=(2000, 6500), va='center', ha='center',
        #             arrowprops=arrow_ax2_props1, fontsize=mid_font, transform=ax2.transAxes)
        # ax2.annotate('', xy=(1150, 2300), xytext=(750, 400), va='center', ha='center',
        #             arrowprops=arrow_ax2_props2, fontsize=mid_font, transform=ax2.transAxes)
        # ax2.annotate('', xy=(3250, 14750), xytext=(2850, 13000), va='center', ha='center',
        #             arrowprops=arrow_ax2_props3, fontsize=mid_font, transform=ax2.transAxes)

        ########################################
        if False:
            ax2_inset = inset_axes(ax2, width=1.9, height=0.8, loc="lower right", bbox_to_anchor=[0.99, 0.02],
                                   bbox_transform=ax2.transAxes)
            D_b = 5.3e-17
            a1 = lattice_constant
            a2 = np.sqrt(D_b / 132.5)

            j_to_meV = 6.24150934190e21
            num_spins_array1 = np.arange(0, 5000, 1)
            num_spins_array2 = np.arange(0, 15811, 1)
            wave_number_array1 = (num_spins_array1 * np.pi) / ((len(num_spins_array1) - 1) * a1)
            wave_number_array2 = (num_spins_array2 * np.pi) / ((len(num_spins_array2) - 1) * a2)

            ax2_inset.plot(wave_number_array1 * hz_2_GHz,
                           (D_b * 2 * gyromag_ratio) * wave_number_array1 ** 2 * hz_2_THz, lw=1.5, ls='--', color='purple',
                           label='$a=0.2$ nm',
                           zorder=1.3)
            ax2_inset.plot(wave_number_array2 * hz_2_GHz,
                           (D_b * 2 * gyromag_ratio) * wave_number_array2 ** 2 * hz_2_THz, lw=1.5, ls='-',
                           label='$a=0.63$ nm',
                           zorder=1.2)

            ax2_inset.set_xlabel('Wavenumber (nm$^{-1}$)', fontsize=mid_font)
            ax2_inset.set_xlim(0, 2)
            ax2_inset.set_ylim(0, 10)
            ax2_inset.xaxis.tick_top()
            ax2_inset.xaxis.set_label_position("top")
            ax2_inset.yaxis.set_label_position("left")
            ax2_inset.set_ylabel('Frequency\n(THz)', fontsize=mid_font, rotation=90, labelpad=20)
            ax2_inset.tick_params(axis='both', labelsize=mid_font)
            ax2.margins(0)

            ax2_inset.patch.set_color("#f9f2e9")
            ax2_inset.yaxis.labelpad = 5
            ax2_inset.xaxis.labelpad = 2.5

            # self._tick_setter(ax2_inset, 2.5, 0.5, 3, 2, is_fft_plot=False)
            ax2_inset.ticklabel_format(axis='y', style='plain')
            ax2_inset.legend(fontsize=small_font, frameon=False)

        ########################################

        ax1.text(0.025, 0.88, f"(a)",
                 verticalalignment='center', horizontalalignment='left', transform=ax1.transAxes, fontsize=mid_font)

        ax2.text(0.975, 0.88, f"(b)",
                 verticalalignment='center', horizontalalignment='right', transform=ax2.transAxes, fontsize=mid_font)

        for ax in [ax1, ax2, ax1_inset]:

            for spine in ["top", "bottom", "left", "right"]:
                ax.spines[spine].set_visible(True)

            ax.grid(False)
            ax.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=1.99)
            ax.set_axisbelow(False)

            if ax == ax1 or ax == ax2:
                ax.set_facecolor("white")

        fig.subplots_adjust(wspace=1, hspace=0.3)

        if False:
            # For interactive plots
            def mouse_event(event):
                print('x: {} and y: {}'.format(event.xdata, event.ydata))

            fig.canvas.mpl_connect('button_press_event', mouse_event)
            fig.tight_layout()  # has to be here
            plt.show()
        else:
            fig.savefig(f"{self.output_filepath}_dispersion.png", bbox_inches="tight")

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

    def _tick_setter(self, ax, x_major, x_minor, y_major, y_minor, is_fft_plot=False,
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
        number_of_spins = len(self.mx_data[:, 0])

        # use_defaults is a testing flag to speed up the process of running sims.
        if use_defaults:
            step = 20
            lower = 0
            upper = 240
            width_ones = 0.05
            width_zeros = 0.95

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
            fourier_coefficents_lhs.append(np.dot(gx_lhs, self.mx_data[:, i]))
            fourier_coefficents_rhs.append(np.dot(gx_rhs, self.mx_data[:, i]))

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

        # Both y-axes need to match up, so it is clear what eigenmode corresponds to what eigenfrequency.
        ax.set(xlabel=r'Eigenfrequency ( $\frac{\omega_j}{2\pi}$ ) (GHz)', ylabel='Fourier coefficient',
               xlim=[lower, upper], yscale='log', ylim=[1e-3, 1e-1],
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
        plt.rcParams.update({'savefig.dpi': 300, "figure.dpi": 300})
        print(f'Plotting #{eigenmode}...')
        eigenmode -= 1  # To handle 'off-by-one' error, as first site is at mx_data[0]

        # Select single mode to plot from imported data.
        mx_mode = self.mx_data[:, eigenmode] * -1
        my_mode = self.my_data[:, eigenmode] * -1

        frequency = self.eigenvalues_data[eigenmode]  # Convert angular (frequency) eigenvalue to frequency [Hz].

        eigenmode += 1  # Return to 'true' count

        # Simulation parameters
        number_of_spins = len(mx_mode) + 2
        driving_width = 0.05

        if has_endpoints:
            # 0-valued reflects the (P-1) and (N+1) end spins that act as fixed nodes for the system.
            mx_mode = np.append(np.append([0], mx_mode), [0])
            my_mode = np.append(np.append([0], my_mode), [0])

        # Generate plot
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))

        sns.lineplot(x=range(0, len(mx_mode)), y=mx_mode, marker='o', markersize='3', ls=':', lw=3, label='Mx',
                     zorder=1.2)
        sns.lineplot(x=range(0, len(my_mode)), y=my_mode, color='r', ls='-', lw=3, label='My', zorder=1.1)

        axes.set(title=f"Eigenmode #{eigenmode}",
                 xlabel="Site Number", ylabel="Amplitude (arb. units)",
                 xlim=(0, number_of_spins))  # ,
        # xticks=np.arange(0, number_of_spins, np.floor(number_of_spins - 2) / 20))
        axes.xaxis.set(major_locator=ticker.MultipleLocator(10),
                       minor_locator=ticker.MultipleLocator(1))

        # Legend doubles as a legend (showing propagation direction), and the frequency [Hz] of the eigenmode.
        axes.legend(loc=1, bbox_to_anchor=(0.975, 0.975),
                    frameon=True, fancybox=True, facecolor='white', edgecolor='white',
                    title=f"Frequency (GHz)\n        {frequency:4.2f}\n\n    Component",
                    fontsize=10)

        axes.axvspan(0, number_of_spins * driving_width, color='black', alpha=0.2)

        axes.grid(color='black', ls='--', alpha=0.1, lw=1)

        plt.tight_layout()
        fig.savefig(f"{self.output_filepath}_eigenmode_{eigenmode}.png")

    def plot_dispersion_relation(self, has_data_file=False):

        fig = plt.figure(figsize=(4.4, 2.0))
        ax = fig.add_subplot(1, 1, 1)

        if has_data_file:
            freqs = np.loadtxt(
                f"/Users/cameronmceleney/CLionProjects/Data/2023-02-15/Simulation_Data/T1606_Eigens/eigenvalues_formatted_T1606.csv")
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


# -------------------------------------- Useful to look at shockwaves. Three panes -------------------------------------
def three_panes(amplitude_data, key_data, list_of_spin_sites, filename, sites_to_compare=None):
    """
    Plots a graph

    :param Any amplitude_data: Array of magnitudes of the spin's magnetisation at each moment in time for each spin
                               site.
    :param dict key_data: All key simulation parameters imported from csv file.
    :param list list_of_spin_sites: Spin sites that were simulated.
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
    textstr = f"H$_{{0}}$ = {simulation_params['staticBiasField']} (T) | " \
              f"H$_{{D1}}$ = {simulation_params['dynamicBiasField1']:2.2e} (T) | " \
              f"\nH$_{{D2}}$ = {simulation_params['dynamicBiasField2']:2.2e}(T)" \
              f" | {exchangeString} | N = {simulation_params['totalSpins']}"

    props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.5, -0.10, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=props, ha='center', va='center')
    fig.tight_layout()
    if interactive:
        # For interactive plots
        def mouse_event(event):
            print('x: {} and y: {}'.format(event.xdata, event.ydata))

        fig.canvas.mpl_connect('button_press_event', mouse_event)
        plt.show()
    else:
        fig.savefig(f"{filename}_ft_only_{spin_site}.png")


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

    driving_freq_hz = simulation_params['drivingFreq'] / 1e9
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
        textstr = f"H$_{{0}}$ = {simulation_params['staticBiasField']} (T) | " \
                  f"H$_{{D1}}$ = {simulation_params['dynamicBiasField1']:2.2e} (T) | " \
                  f"H$_{{D2}}$ = {simulation_params['dynamicBiasField2']:2.2e}(T) \n" \
                  f"{exchangeString} | N = {simulation_params['chainSpins']} | " + r"$\alpha$" + \
                  f" = {simulation_params['gilbertFactor']: 2.2e}"

        props = dict(boxstyle='round', facecolor='gainsboro', alpha=1.0)
        # place a text box in upper left in axes coords
        ax.text(0.5, -0.2, textstr, transform=ax.transAxes, fontsize=18,
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
    time = mx_data[:, 0]

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
