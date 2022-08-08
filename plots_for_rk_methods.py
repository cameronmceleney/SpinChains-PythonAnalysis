#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard modules (common)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys as sys
import types as typ

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

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

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
        self.driving_freq = key_data['drivingFreq'] / 1e9  # Converts from [s] to [ns].
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

        # Attributes for plots "ylim": [-1 * self.y_axis_limit, self.y_axis_limit]
        cm = 1 / 2.54
        self.fig = plt.figure(figsize=(4, 2))
        self.axes = self.fig.add_subplot(111)
        self.y_axis_limit = 6.4e-3  # max(self.amplitude_data[-1, :]) * 1.1  # Add a 10% margin to the y-axis.
        self.kwargs = {"xlabel": f"Spin Sites", "ylabel": f"m$_x$ / M$_S$",
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
            # For images, may want to further alter plot outside this method. Hence, the use of attribute.
            cm = 1 / 2.54
            self.fig = plt.figure(figsize=(11.12 * cm, 6.15 * cm))  # Strange dimensions are to give a 4x2 inch image
            self.axes = self.fig.add_subplot(111)
        else:
            # For GIFs
            cm = 1 / 2.54
            self.fig = plt.figure(figsize=(11.12 * cm * 2, 6.15 * cm * 2),
                                  dpi=450)  # Each frame requires a new fig to prevent stuttering.
            self.axes = self.fig.add_subplot(
                111)  # Each subplot will be the same so no need to access ax outside of method.

        self.axes.set_aspect("auto")
        #  plt.suptitle(f"{self.nm_method}")
        #  plt.subplots_adjust(top=0.80)

        self.axes.plot(np.arange(1, self.number_spins + 1), self.amplitude_data[plot_row, :], ls='-', lw=0.75,
                       label=f"{self.time_data[plot_row]:2.2f} [ns]",
                       color='#64bb6a')  # Easier to have time-stamp as label than textbox.

        self.axes.set(**self.kwargs)

        if draw_regions_of_interest:
            left, bottom, width, height = (
                [0, self.number_spins - self.dampedSpins, self.drLHS + self.dampedSpins],
                self.axes.get_ylim()[0] * 2,
                (self.dampedSpins, self.driving_width),
                4 * self.axes.get_ylim()[1])

            rectLHS = mpatches.Rectangle((left[0], bottom), width[0], height,
                                         # fill=False,
                                         alpha=0.6,
                                         facecolor="#BB64B5",
                                         edgecolor=None,
                                         lw=0)

            rectRHS = mpatches.Rectangle((left[1], bottom), width[0], height,
                                         # fill=False,
                                         alpha=0.6,
                                         facecolor="#BB64B5",
                                         edgecolor=None,
                                         lw=0)

            rectDriving = mpatches.Rectangle((left[2], bottom), width[1], height,
                                             # fill=False,
                                             alpha=0.3,
                                             facecolor="#BB64B5",
                                             edgecolor=None,
                                             lw=0)

            plt.gca().add_patch(rectLHS)
            plt.gca().add_patch(rectRHS)
            plt.gca().add_patch(rectDriving)

        # Change tick markers as needed.
        self._tick_setter(self.axes, 2500, 500, 3, 4)

        self.axes.legend(loc=1, ncol=1, fontsize=6,
                         frameon=False, fancybox=True, facecolor=None, edgecolor=None)

        self.fig.tight_layout()

    def create_png(self, row_number=-1):
        """
        Generate a PNG for a single row of the given dataset.

        A row corresponds to an instant in time, so this can be particularly useful for investigating the final 'state'
        of a system.

        :param int row_number: Which row of data to be plotted. Defaults to plotting the final row.

        :return: No direct returns. Invoking method will save a .png to the nominated 'Outputs' directory.
        """
        self.axes.clear()  # Use this if looping through a single PaperFigures object for multiple create_png inputs
        self._draw_figure(plot_row=row_number)

        self.axes.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        # Add text to figure with simulation parameters
        # if self.exchange_min == self.exchange_max:
        #    exchangeString = f"Uniform Exc. : {self.exchange_min} [T]"
        # else:
        #    exchangeString = f"J$_{{min}}$ = {self.exchange_min} [T] | J$_{{max}}$ = " \
        #                     f"{self.exchange_max} [T]"
        # textstr = f"H$_{{0}}$ = {self.static_field} [T] | N = {self.chain_spins} | " + r"$\alpha$" \
        #                                                                               f" = {self.gilbert_factor: 2.2e}\n" \
        #                                                                               f"H$_{{D1}}$ = {self.driving_field1:2.2e} [T] | H$_{{D2}}$ = {self.driving_field2:2.2e} [T] \n" \
        #                                                                               f"{exchangeString}"
        #
        # props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.5)
        ## Place text box in upper left in axes coords
        # self.axes.text(0.05, 1.2, textstr, transform=self.axes.transAxes, fontsize=12,
        #               verticalalignment='top', bbox=props, ha='center', va='center')

        # Figure outputs

        # Add spines to all plots (to override any rcParams elsewhere in the code
        for spine in ['top', 'bottom', 'left', 'right']:
            self.axes.spines[spine].set_visible(True)

        self.axes.grid(visible=False, axis='both', which='both')

        self.axes.text(-0.045, 0.97, r'$\times \mathcal{10}^{{\mathcal{-3}}}$', horizontalalignment='center',
                       verticalalignment='center', transform=self.axes.transAxes, fontsize=6)

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

        for index in range(0, int(self.data_points + 1), int(self.data_points * number_of_frames)):
            frame = self._plot_paper_gif(index)
            frames.append(frame)

        gif.save(frames, f"{self.output_filepath}.gif", duration=1, unit='ms')

    def plot_site_variation(self, spin_site, add_zoomed_region=True, add_info_box=True, add_colored_regions=True):
        """
        Plot the magnetisation of a site against time.

        One should ensure that the site being plotted is not inside either of the driving- or damping-regions.

        :param int spin_site: The number of the spin site to be plotted.

        :return: Saves a .png image to the designated output folder.
        """

        self.axes.clear()
        self.axes.set_aspect('auto')
        # self.axes.plot(self.time_data, self.amplitude_data[:, spin_site], ls='-', lw=0.5,
        #               label=f"{self.sites_array[spin_site]}", color="#64bb6a")  # Easier to have time-stamp as label than textbox.
        self.axes.plot(self.time_data[12:1072], self.amplitude_data[12:1072, spin_site], marker='', lw=1,
                       color='#37782c',
                       markerfacecolor='black', markeredgecolor='black', label="Precursors", zorder=5)
        self.axes.plot(self.time_data[1072:1727], self.amplitude_data[1072:1727, spin_site], marker='', lw=1,
                       color='#64bb6a',
                       markerfacecolor='black', markeredgecolor='black', label="Shockwave", zorder=2)
        self.axes.plot(self.time_data[1727:], self.amplitude_data[1727:, spin_site], marker='', lw=1, color='#9fd983',
                       markerfacecolor='black', markeredgecolor='black', label="Steady State", zorder=1)

        self.axes.text(-0.045, 0.97, r'$\times \mathcal{10}^{{\mathcal{-3}}}$', horizontalalignment='center',
                       verticalalignment='center', transform=self.axes.transAxes, fontsize=6)

        ymax_plot = self.amplitude_data[:, spin_site].max()
        xlim_in = 4.0  # int(input("Enter xlim: "))
        # title = f"Mx Values for {self.driving_freq:2.2f} [GHz]",
        self.axes.set(xlabel=f"Time [ns]", ylabel=f"m$_x$ / M$_S$",
                      xlim=[0, xlim_in],
                      ylim=[-1 * self.amplitude_data[:, spin_site].max(),
                            ymax_plot])

        self._tick_setter(self.axes, 2.5, 0.5, 3, 4)

        # Change tick markers as needed.
        # self.axes.xaxis.set(major_locator=ticker.MultipleLocator(2.5),
        #                     minor_locator=ticker.MultipleLocator(0.5))
        # self.axes.yaxis.set(major_locator=ticker.MaxNLocator(nbins=3, prune='lower'),
        #                     minor_locator=ticker.AutoMinorLocator(4),
        #                     major_formatter=ticker.FormatStrFormatter("%.1f"))
        #
        # formatter = ticker.ScalarFormatter(useMathText=True)
        # self.axes.yaxis.set_major_formatter(formatter)
        # self.axes.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        # # self.axes.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        # # formatter = ticker.ScalarFormatter(useMathText=True)
        # # self.axes.yaxis.set_major_formatter(formatter)
        # # self.axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1f'))
        # # self.axes.yaxis.get_offset_text().set_fontsize(6)
        # self.axes.yaxis.get_offset_text().set_visible(False)

        # self.axes.legend(title="Spin Site [#]", loc=1,frameon=True, fancybox=True, framealpha=0.5, facecolor='white')

        # Add zoomed in region if needed.
        if add_zoomed_region:
            # Select datasets to use
            x = self.time_data
            y = self.amplitude_data[:, spin_site]

            # Impose inset onto plot. Treat as a separate subplot. Use 0.24 for LHS and 0.8 for RHS. 0.7 for T and 0.25 for B
            ax2_inset = inset_axes(self.axes, width=1.2, height=0.6, loc="center", bbox_to_anchor=[0.2525, 0.255],
                                   bbox_transform=self.axes.figure.transFigure)
            ax2_inset.plot(x, y, lw=0.75, color='#37782c')

            # Select data (of original) to show in inset through changing axis limits
            ylim_in = 4e-4  # float(input("Enter ylim: "))
            ax2_inset.set_xlim(0.6, 1.52)
            ax2_inset.set_ylim(-ylim_in, ylim_in)

            # Remove tick labels
            ax2_inset.set_xticks([])
            ax2_inset.set_yticks([])
            ax2_inset.patch.set_color("#f9f2e9")  # #f0a3a9 is equivalent to color 'red' and alpha '0.3'
            # ax2_inset.patch.set_alpha(0.3)

            # ax2_inset.set(facecolor='red', alpha=0.3)

            # Add spines to all plots (to override any rcParams elsewhere in the code
            for spine in ['top', 'bottom', 'left', 'right']:
                ax2_inset.spines[spine].set_visible(True)
                self.axes.spines[spine].set_visible(True)

            # mark_inset(self.axes, ax2_inset,loc1=1, loc2=2, facecolor="red", edgecolor=None, alpha=0.3)

            # Add box to indicate the region which is being zoomed into on the main figure
            self.axes.indicate_inset_zoom(ax2_inset, facecolor='#f9f2e9', edgecolor='black', alpha=1.0, lw=0.75,
                                          zorder=1)

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
            self.axes.text(0.35, -0.22, textstr, transform=self.axes.transAxes, fontsize=6,
                           verticalalignment='top', bbox=props, ha='center', va='center')
            self.axes.text(0.85, -0.22, "Time [ns]", fontsize=12, ha='center', va='center',
                           transform=self.axes.transAxes)

        if add_colored_regions:
            rectLHS = mpatches.Rectangle((0, -1 * self.amplitude_data[:, spin_site].max()), 5.75,
                                         2 * self.amplitude_data[:, spin_site].max() + + 0.375e-2,
                                         # fill=False,
                                         alpha=0.05,
                                         facecolor="grey",
                                         edgecolor=None,
                                         lw=0)
            rectMID = mpatches.Rectangle((5.751, -1 * self.amplitude_data[:, spin_site].max()), 3.249,
                                         2 * self.amplitude_data[:, spin_site].max() + 0.375e-2,
                                         # fill=False,
                                         alpha=0.25,
                                         facecolor="grey",
                                         edgecolor=None,
                                         lw=0)
            rectRHS = mpatches.Rectangle((9.0, -1 * self.amplitude_data[:, spin_site].max()), 6,
                                         2 * self.amplitude_data[:, spin_site].max() + + 0.375e-2,
                                         # fill=False,
                                         alpha=0.5,
                                         facecolor="grey",
                                         edgecolor=None,
                                         lw=0)

            self.axes.add_patch(rectLHS)
            self.axes.add_patch(rectMID)
            self.axes.add_patch(rectRHS)
        self.axes.grid(visible=False, axis='y', which='both')
        self.axes.grid(visible=False, axis='x', which='both')
        # self.fig.tight_layout()

        self.fig.savefig(f"{self.output_filepath}_site{spin_site}.png", bbox_inches="tight")

    def plot_fft(self, spin_site, add_zoomed_region=True):
        """
        Plot the magnitudes of the magnetic moment of a spin site against time, as well as the FFTs, over four subplots.

        :param int spin_site: The spin site being plotted.

        :return: A figure containing four sub-plots.
        """
        # Find maximum time in [ns] to the nearest whole [ns], then find how large shaded region should be.

        plot_set_params = {0: {"xlabel": "Time [ns]", "ylabel": "m$_x$ / M$_S$", "xlim": (0, 5.5)},
                           1: {"xlabel": "Frequency [GHz]", "ylabel": "Amplitude [arb.]",
                               "xlim": (0, 50), "ylim": (1E-4, 1E1)},
                           2: {"xlabel": "Frequency [GHz]", "xlim": (0, 10)}}
        fig = plt.figure()

        blob1_lower, blob1_uppper = 1250, 1325
        blob2_lower, blob2_upper = 1455, 1555
        blob3_lower, blob3_upper = 1727, 1941
        precursor_lower, precursor_upper = 12, 1941
        shock_lower, shock_upper = 1941, 2888
        equil_lower, equil_upper = 2888, 9999

        # Signal that varies in time
        ax1 = plt.subplot2grid((4, 8), (0, 0), rowspan=2, colspan=8)
        ax1.plot(self.time_data[precursor_lower:precursor_upper],
                 self.amplitude_data[precursor_lower:precursor_upper, spin_site],
                 color='#37782c')  # precursor
        ax1.plot(self.time_data[shock_lower:shock_upper],
                 self.amplitude_data[shock_lower:shock_upper, spin_site],
                 color='#64bb6a')  # shockwave
        ax1.plot(self.time_data[equil_lower:equil_upper],
                 self.amplitude_data[equil_lower:equil_upper, spin_site], color='#9fd983')  # equilibrium

        ax1.set(**plot_set_params[0])
        ax1.set_facecolor('#f4f4f5')
        self._tick_setter(ax1, 2.5, 0.5, 3, 4)

        # FFT stuff
        frequencies_blob1, fourier_transform_blob1 = self._fft_data(
            self.amplitude_data[blob1_lower:blob1_uppper, spin_site])
        frequencies_blob2, fourier_transform_blob2 = self._fft_data(
            self.amplitude_data[blob2_lower:blob2_upper, spin_site])
        frequencies_blob3, fourier_transform_blob3 = self._fft_data(
            self.amplitude_data[blob3_lower:blob3_upper, spin_site])
        frequencies_precursors, fourier_transform_precursors = self._fft_data(
            self.amplitude_data[12:precursor_upper, spin_site])
        frequencies_dsw, fourier_transform_dsw = self._fft_data(self.amplitude_data[shock_lower:shock_upper, spin_site])
        frequencies_eq, fourier_transform_eq = self._fft_data(self.amplitude_data[equil_lower:equil_upper, spin_site])

        # FFT for blobs #37782c
        ax2 = plt.subplot2grid((4, 8), (2, 0), rowspan=2, colspan=8)
        ax2.plot(frequencies_blob1, abs(fourier_transform_blob1), marker='', lw=1, color='#37782c',
                 markerfacecolor='black', markeredgecolor='black', ls=':')
        ax2.plot(frequencies_blob2, abs(fourier_transform_blob2), marker='', lw=1, color='#37782c',
                 markerfacecolor='black', markeredgecolor='black', ls='--')
        ax2.plot(frequencies_blob3, abs(fourier_transform_blob3), marker='', lw=1, color='#37782c',
                 markerfacecolor='black', markeredgecolor='black', ls='-.')
        ax2.plot(frequencies_precursors, abs(fourier_transform_precursors), marker='', lw=0.75, color='#37782c',
                 markerfacecolor='black', markeredgecolor='black', label="Precursors", zorder=5)
        ax2.plot(frequencies_dsw, abs(fourier_transform_dsw), marker='', lw=0.75, color='#64bb6a',
                 markerfacecolor='black', markeredgecolor='black', label="Shockwave", zorder=2)
        ax2.plot(frequencies_eq, abs(fourier_transform_eq), marker='', lw=0.75, color='#9fd983',
                 markerfacecolor='black', markeredgecolor='black', label="Steady State", zorder=1)
        ax2.set(**plot_set_params[1], yscale='log')
        arrow_ax2_props = {"arrowstyle": '-|>', "connectionstyle": "angle3,angleA=0,angleB=90", "color": "black"}
        ax2.annotate('P1', xy=(20.23, 0.0401), xytext=(22.94, 0.143), va='center', ha='center',
                     arrowprops=arrow_ax2_props, fontsize=8)
        ax2.annotate('P2', xy=(28.87, 0.0145), xytext=(31.07, 0.0704), va='center', ha='center',
                     arrowprops=arrow_ax2_props, fontsize=8)
        ax2.annotate('P3', xy=(38.18, 0.0073), xytext=(40.73, 0.0474), va='center', ha='center',
                     arrowprops=arrow_ax2_props, fontsize=8)
        ax2.legend(ncol=1, fontsize=6, frameon=False, fancybox=True, facecolor=None, edgecolor=None,
                   bbox_to_anchor=[0.82, 0.68])
        self._tick_setter(ax2, 10, 2.5, 4, 6, is_fft_plot=True)

        for ax in [ax1, ax2]:
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.set_facecolor('#f4f4f5')
            ax.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=6)

        # Add zoomed in region if needed.
        if add_zoomed_region:
            # Select datasets to use
            x = self.time_data[:]
            y = self.amplitude_data[:, spin_site]

            # Impose inset onto plot. Treat as a separate subplot. Use 0.24 for LHS and 0.8 for RHS. 0.7 for TR and
            ax1_inset = inset_axes(ax1, width=2.2, height=0.55, loc="center", bbox_to_anchor=[0.32, 0.6875],
                                   bbox_transform=ax1.figure.transFigure)
            ax1_inset.plot(x, y, lw=0.75, color='#37782c')

            # Select data (of original) to show in inset through changing axis limits
            ax1_inset.set_xlim(1.5, 2.7)
            ax1_inset.set_ylim(-0.6e-3, 0.6e-3)

            # Remove tick labels
            ax1_inset.set_xticks([])
            ax1_inset.set_yticks([])
            ax1_inset.patch.set_color("#f9f2e9")  # #f0a3a9 is equivalent to color 'red' and alpha '0.3' fbe3e5
            # ax2_inset.patch.set_alpha(0.3)
            # ax2_inset.set(facecolor='red', alpha=0.3)
            # mark_inset(self.axes, ax2_inset,loc1=1, loc2=2, facecolor="red", edgecolor=None, alpha=0.3)

            # Add box to indicate the region which is being zoomed into on the main figure
            ax1.indicate_inset_zoom(ax1_inset, facecolor='#f9f2e9', edgecolor='black', alpha=1.0, lw=0.75, zorder=1)
            arrow_inset_props = {"arrowstyle": '-|>', "connectionstyle": "angle3,angleA=0,angleB=90", "color": "black"}
            ax1_inset.annotate('P1', xy=(2.476, -4.84E-4), xytext=(2.2, -4.5e-4), va='center', ha='center',
                               arrowprops=arrow_inset_props, fontsize=6)
            ax1_inset.annotate('P2', xy=(2.113, 2.5E-4), xytext=(1.6, 4.5e-4), va='center', ha='center',
                               arrowprops=arrow_inset_props, fontsize=6)
            ax1_inset.annotate('P3', xy=(1.822, -2E-4), xytext=(1.6, -4.5e-4), va='center', ha='center',
                               arrowprops=arrow_inset_props, fontsize=6)

        # Add spines to all plots (to override any rcParams elsewhere in the code
        for spine in ['top', 'bottom', 'left', 'right']:
            ax1_inset.spines[spine].set_visible(True)
            ax1.spines[spine].set_visible(True)
            ax2.spines[spine].set_visible(True)

        fig.subplots_adjust(wspace=1.0, hspace=0.5)
        fig.tight_layout()

        #def mouse_event(event):
        #    print('x: {} and y: {}'.format(event.xdata, event.ydata))
        #
        #cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
        #
        #plt.show()
        fig.savefig(f"{self.output_filepath}_site{spin_site}_fft.png", bbox_inches="tight")

    def _fft_data(self, amplitude_data):
        """
        Computes the FFT transform of a given signal, and also outputs useful data such as key frequencies.

        :param dict simulation_params: Imported key simulation parameters.
        :param amplitude_data: Magnitudes of magnetic moments for a spin site

        :return: A tuple containing the frequencies [0], FFT [1] of a spin site. Also includes the  natural frequency
        (1st eigenvalue) [2], and driving frequency [3] for the system.
        """
        # Simulation parameters needed for FFT computations that are always the same are saved here.
        # gamma is in [GHz/T] here.
        core_values = {"gamma": self.gyro_mag_ratio / (2 * np.pi),
                       "hz_to_ghz": 1e-9}

        # Data in file header is in [Hz] by default.
        # driving_freq_ghz = self.driving_freq # * core_values["hz_to_ghz"]

        # This is the (first) natural frequency of the system, corresponding to the first eigenvalue. Change as needed to
        # add other markers to the plot(s)
        # natural_freq = core_values['gamma'] * self.static_field

        # Find bin size by dividing the simulated time into equal segments based upon the number of data-points.
        sample_spacing = (self.max_time / (self.data_points - 1))

        # Compute the FFT
        n = amplitude_data.size
        normalised_data = amplitude_data

        fourier_transform = rfft(normalised_data)
        frequencies = rfftfreq(n, sample_spacing)

        return frequencies, fourier_transform  # , natural_freq, driving_freq_ghz

    def _tick_setter(self, ax, x_major, x_minor, y_major, y_minor, set_power=-3, is_fft_plot=False):

        class OOMFormatter(ticker.ScalarFormatter):
            def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
                self.oom = order
                self.fformat = fformat
                ticker.ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)

            def _set_order_of_magnitude(self):
                self.orderOfMagnitude = self.oom

            def _set_format(self, vmin=None, vmax=None):
                self.format = self.fformat
                if self._useMathText:
                    self.format = r'$\mathdefault{%s}$' % self.format

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
                         major_formatter=ticker.FormatStrFormatter("%.1f"),
                         minor_locator=ticker.MultipleLocator(x_minor))
            ax.yaxis.set(major_locator=ticker.MaxNLocator(nbins=y_major, prune='lower'),
                         minor_locator=ticker.AutoMinorLocator(y_minor))

            ax.yaxis.set_major_formatter(OOMFormatter(set_power, "%1.1f"))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(set_power, set_power))
            ax.yaxis.get_offset_text().set_visible(False)

            # formatter = ticker.ScalarFormatter(useMathText=True)
            # ax.yaxis.set_major_formatter(formatter)
            # ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
            # ax.yaxis.set(major_formatter=ticker.FormatStrFormatter("%1.1f"),
            #              minor_formatter=ticker.FormatStrFormatter("%1.1f"))
            # ax.yaxis.get_offset_text().set_fontsize(6)
            # t = ax.yaxis.get_offset_text()
            # t.set_x(-0.045)

        return ax


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
        axes.set(xlabel='Time [ns]', ylabel="m$_x$")  # xlim=[0, key_data['maxSimTime']]
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
    interactive = False
    # Use for interactive plot. Also change DPI to 40 and allow Pycharm to plot outside of tool window
    if interactive:
        fig = plt.figure(figsize=(9, 9))
    else:
        fig = plt.figure(figsize=(12, 9), constrained_layout=True)

    fig.suptitle(f"Data from Spin Site #{spin_site}")
    ax = plt.subplot(1, 1, 1)

    frequencies, fourier_transform, natural_frequency, driving_freq = fft_data(amplitude_data, simulation_params)

    # Plotting. To normalise data, change y-component to (1/N)*abs(fourier_transform) where N is the number of samples.
    # Set marker='o' to see each datapoint, else leave as marker='' to hide
    ax.plot(frequencies, abs(fourier_transform),
            marker='', lw=2, color='red', markerfacecolor='black', markeredgecolor='black')
    ax.set(xlabel="Frequency [GHz]", ylabel="Amplitude [arb.]",
           xlim=[0, 60])

    ax.legend(loc=0, frameon=True, fancybox=True, facecolor='white', edgecolor='white',
              title=f'Freq. List [GHz]\nDriving - {driving_freq}', fontsize=12)

    # ax.xaxis.set_major_locator(MultipleLocator(5))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    # For the minor ticks, use no labels; default NullFormatter.
    # ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.grid(color='white')
    ax.grid(False)

    if simulation_params['exchangeMinVal'] == simulation_params['exchangeMaxVal']:
        exchangeString = f"Uniform = True ({simulation_params['exchangeMinVal']}) [T]"
    else:
        exchangeString = f"J$_{{min}}$ = {simulation_params['exchangeMinVal']} [T] | J$_{{max}}$ = " \
                         f"{simulation_params['exchangeMaxVal']} [T]"
    textstr = f"H$_{{0}}$ = {simulation_params['staticBiasField']} [T] | " \
              f"H$_{{D1}}$ = {simulation_params['dynamicBiasField1']:2.2e} [T] | " \
              f"H$_{{D2}}$ = {simulation_params['dynamicBiasField2']:2.2e}[T]" \
              f" | {exchangeString} | N = {simulation_params['totalSpins']}"

    props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.5, -0.15, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=props, ha='center', va='center')

    if interactive:
        # For interactive plots
        def mouse_event(event):
            print('x: {} and y: {}'.format(event.xdata, event.ydata))

        cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
        plt.show()
    else:
        fig.savefig(f"{filename}_{spin_site}.png")


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
    x_scaling = 0.1
    fft_shaded_box_width = 10  # In GHz
    offset = 0  # Zero by default
    t_shaded_xlim = temporal_xlim * x_scaling + offset

    plot_set_params = {0: {"title": "Full Simulation", "xlabel": "Time [ns]", "ylabel": "Amplitude [arb.]",
                           "xlim": (offset, temporal_xlim)},
                       1: {"title": "Shaded Region", "xlabel": "Time [ns]", "xlim": (offset, t_shaded_xlim)},
                       2: {"title": "Showing All Artefacts", "xlabel": "Frequency [GHz]", "ylabel": "Amplitude [arb.]",
                           "xlim": (0, 60)},
                       3: {"title": "Shaded Region", "xlabel": "Frequency [GHz]", "xlim": (0, fft_shaded_box_width)}}

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
    # Set marker='o' to see each datapoint, else leave as marker='' to hide
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
        a = 5
        if simulation_params['exchangeMinVal'] == simulation_params['exchangeMaxVal']:
            exchangeString = f"Uniform Exc. ({simulation_params['exchangeMinVal']} [T])"
        else:
            exchangeString = f"J$_{{min}}$ = {simulation_params['exchangeMinVal']} [T] | J$_{{max}}$ = " \
                             f"{simulation_params['exchangeMaxVal']} [T]"
        textstr = f"H$_{{0}}$ = {simulation_params['staticBiasField']} [T] | " \
                  f"H$_{{D1}}$ = {simulation_params['dynamicBiasField1']:2.2e} [T] | " \
                  f"H$_{{D2}}$ = {simulation_params['dynamicBiasField2']:2.2e}[T] \n" \
                  f"{exchangeString} | N = {simulation_params['chainSpins']} | " + r"$\alpha$" + \
                  f" = {simulation_params['gilbertFactor']: 2.2e}"

        props = dict(boxstyle='round', facecolor='gainsboro', alpha=1.0)
        # place a text box in upper left in axes coords
        ax.text(0.5, -0.2, textstr, transform=ax.transAxes, fontsize=18,
                verticalalignment='top', bbox=props, ha='center', va='center')
        # By default, plots the natural frequency.
        # ax.axvline(x=natural_frequency, label=f"Natural. {natural_frequency:2.2f}")

    ax.legend(loc=0, frameon=True, fancybox=True, facecolor='white', edgecolor='white',
              title='Freq. List [GHz]', fontsize=12)

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
    sample_spacing = (simulation_params["maxSimTime"] / (simulation_params["numberOfDataPoints"] - 1)) / core_values[
        'hz_to_ghz']

    # Compute the FFT
    n = amplitude_data.size
    normalised_data = amplitude_data

    fourier_transform = rfft(normalised_data)
    frequencies = rfftfreq(n, sample_spacing)

    return frequencies, fourier_transform, natural_freq, driving_freq_ghz


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
    # ax = plt.axes(projection='3d')
    # if use_tri:
    #     ax.plot_trisurf(x, y, z, cmap='Blues', lw=0.1, edgecolor='none', label=f'Spin Site {spin_site}')
    # else:
    #     ax.plot3D(x, y, z, label=f'Spin Site {spin_site}')
    #     ax.legend()
    #
    # ax.set_xlabel('m$_x$', fontsize=12)
    # ax.set_ylabel('m$_y$', fontsize=12)
    # ax.set_zlabel('m$_z$', fontsize=12)
    #
    # ax.xaxis.set_rotate_label(False)
    # ax.yaxis.set_rotate_label(False)
    # ax.zaxis.set_rotate_label(False)
    fig.savefig(f"{output_file}_contour.png")


# --------------------------------------------- Continually plot eigenmodes --------------------------------------------
def eigenmodes(mx_data, my_data, eigenvalues_data, file_name):
    """
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
                    generalised_fourier_coefficients(mx_data, eigenvalues_data, file_name, False)
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
        step = int(input("Enter step: "))
        lower = int(input("Enter lower: "))
        upper = int(input("Enter upper: "))
        width_ones = float(input("Enter width of driving region: "))
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
    fourier_coefficents_rhs = fourier_coefficents_rhs / np.linalg.norm(fourier_coefficents_rhs)

    # Plotting functions. Left here as nothing else will use this functionality.
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle(r'Overlap Values ($\mathcal{O}_{j}$)'f'for a Non-Uniform System')  # {file_name}
    plt.subplots_adjust(top=0.82)

    # Whichever ax is before the sns.lineplot statements is the one which holds the labels.
    sns.lineplot(x=x_axis_limits, y=np.abs(fourier_coefficents_lhs) * 1e1, lw=3, marker='o', label='Left', zorder=2)
    sns.lineplot(x=x_axis_limits, y=np.abs(fourier_coefficents_rhs) * 1e1, lw=3, color='r',
                 marker='o', label='Right', zorder=1)

    # Both y-axes need to match up, so it is clear what eigenmode corresponds to what eigenfrequency.
    ax.set(xlabel=r'Eigenfrequency ( $\frac{\omega_j}{2\pi}$ ) [GHz]', ylabel='Fourier coefficient',
           xlim=[lower, upper], yscale='log', ylim=[1e-2, 1e-0],
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

    plt.tight_layout()
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
    number_of_spins = len(mx_mode) + 2
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
                title=f"Frequency [GHz]\n        {frequency:4.2f}\n\n    Component",
                fontsize=10)

    axes.axvspan(0, number_of_spins * driving_width, color='black', alpha=0.2)

    axes.grid(color='black', ls='--', alpha=0.1, lw=1)

    plt.tight_layout()
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
    number_of_spins = key_data['chainSpins']

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
