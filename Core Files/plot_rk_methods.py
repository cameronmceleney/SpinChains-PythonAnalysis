#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv

import matplotlib as mpl
from sys import platform as sys_platform

if sys_platform == 'darwin':
    mpl.use('macosx')
elif sys_platform in ['win32', 'win64']:
    mpl.use('TkAgg')

# Full packages
import decimal as dec
import imageio as imio
import math as math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mplcursors
import mpl_toolkits.axes_grid1
import mpl_toolkits.axes_grid1.inset_locator
import numpy as np
import os as os
import scipy as sp

# Specific functions from packages
from typing import TypedDict, Any, Dict, List, Optional, Union

# My full modules


# Specific functions from my modules
from attribute_defintions import SimulationParametersContainer, SimulationFlagsContainer
from figure_manager import FigureManager, colour_schemes

"""
    Contains all the plotting functionality required for my data analysis. The data for each method comes from the file
    data_analysis.py. These plots will only work for data from my RK methods.
"""

"""
    Core Details
    
    Author      : cameronmceleney
    Created on  : 13/03/2022 18:06
    Filename    : plot_rk_methods.py
    IDE         : PyCharm
"""


class PaperFigures(SimulationFlagsContainer, SimulationParametersContainer):
    """
    Generates a single subplot that can either be a PNG or GIF.

    Useful for creating plots for papers, or recreating a paper's work. To change between the png/gif saving options,
    change the invocation in data.analysis.py.
    """
    cm_to_inch = 1 / 2.54
    hz_to_Ghz = 1e-9

    class PlotScheme(TypedDict):
        signal_xlim: List[int]
        signal_rescale: List[int]
        rescale_extras: List[float | int | str]
        ax1_xlim: List[float]
        ax1_ylim: List[float]
        ax2_xlim: List[float]
        ax2_ylim: List[float]
        ax3_xlim: List[float]
        ax3_ylim: List[int]
        signal1_xlim: List[int]
        signal2_xlim: List[int]
        signal3_xlim: List[int]
        ax1_label: str
        ax2_label: str
        ax3_label: str
        ax1_line_height: float

    def __init__(self, time_data, amplitude_data, params_dict, flags_dict, array_of_sites, output_filepath,
                 params_container=None, flags_container=None):
        super().__init__()
        # Data and paths read-in from data_analysis.py
        self.time_data = time_data
        self.amplitude_data = amplitude_data
        self.sites_array = array_of_sites
        self.output_filepath = output_filepath

        if params_container is not None:
            self.update_with_container(params_container)
        else:
            self.update_with_dict(params_dict)

        if flags_container is not None:
            self.update_with_container(flags_container)
        else:
            self.update_with_dict(flags_dict)

        if self.lattice_constant() < 0:
            self.lattice_constant.update(1e-9)

        if self.exchange_dmi_constant() == 0.625:
            # TODO. Change this. Note that, for now, the halving of the DMI is taken care of by the C++ code
            self.exchange_dmi_constant *= 2
        # Attributes for plots
        self._fig = None
        self._axes = None
        self._yaxis_lim = 1.1  # Add a 10% margin to the y-axis.
        self._yaxis_lim_fix = 8e-3

        self.track_zorder = [[], []]

        # Text sizes for class to override rcParams
        self._fontsizes = {"large": 20, "medium": 14, "small": 11, "smaller": 10, "tiny": 8, "mini": 7}

    def _draw_figure(self, row_index: int = -1, has_single_figure: bool = True, publish_plot: bool = False,
                     draw_regions_of_interest: bool = False, static_ylim=False, axes=None,
                     interactive_plot: bool = False, real_distance: bool = True) -> None:
        """
        Private method to plot the given row of data, and create a single figure.

        Figure attributes are controlled from __init__ to ensure consistency.

        :param static_ylim:
        :param row_index: Plot given row from dataset; most commonly plotted should be the default.
        :param has_single_figure: If `False`, change figure dimensions for GIFs
        :param draw_regions_of_interest: Draw coloured boxes onto plot to show driving- and damping-regions.
        :param publish_plot: Flag to add figure number and LaTeX annotations for publication.

        :return: Method updates `self._fig` and `self.axis` within the class.
        """
        if axes is not None:
            self._axes = axes

        if self._fig is None:
            figsize = (4.4, 2.2) if has_single_figure else (4.4, 4.4)
            self._fig = plt.figure(figsize=figsize)
            self._axes = self._fig.add_subplot(111)
            if not has_single_figure:
                # Adjust for GIFs
                plt.rcParams.update({'savefig.dpi': 200, "figure.dpi": 200})

        # Adjust y-axis limit based on static_ylim and amplitude data
        if static_ylim:
            self._yaxis_lim = self._yaxis_lim_fix
        else:
            self._yaxis_lim = max(self.amplitude_data[row_index, :]) * (1.1 if self._fig is None else self._yaxis_lim)

        self._axes.clear()
        self._axes.set_aspect("auto")

        # Easier to have time-stamp as label than textbox.
        self._axes.plot(np.arange(0, self.num_sites_total()), self.amplitude_data[row_index, :], ls='-',
                        lw=2 * 0.75, color='#64bb6a', label=f"Signal", zorder=1.1)

        self._axes.set(xlabel=f"Position, $n_i$ (site index)",
                       xlim=[0.0, self.num_sites_total()],
                       ylim=[-self._yaxis_lim, self._yaxis_lim])

        _, y_major_labels, _ = self._choose_scaling(subplot_to_scale=self._axes)
        self._axes.set(ylabel=f"m$_x$ (a.u. " + y_major_labels[1] + ")")


        if publish_plot:
            self._axes.text(-0.04, 0.96, r'$\times \mathcal{10}^{{\mathcal{-3}}}$', va='center',
                            ha='center', transform=self._axes.transAxes, fontsize=6)
            self._axes.text(0.88, 0.88, f"(c) {self.time_data[row_index]:2.3f} ns",
                            va='center', ha='center', transform=self._axes.transAxes,
                            fontsize=6)

        if draw_regions_of_interest:
            left, bottom, width, height = (
                [0, (self.num_sites_total() - self.num_sites_abc()),
                 (self.driving_region_lhs() + self.num_sites_abc())],
                self._axes.get_ylim()[0] * 2,
                (self.num_sites_abc(), self.driving_region_width()),
                4 * self._axes.get_ylim()[1])

            rectangle_lhs = mpatches.Rectangle((left[0], bottom), width[0], height, lw=0,
                                               alpha=0.5, facecolor="grey", edgecolor=None)

            rectangle_rhs = mpatches.Rectangle((left[1], bottom), width[0], height, lw=0,
                                               alpha=0.5, facecolor="grey", edgecolor=None)

            rectangle_driving_region = mpatches.Rectangle((left[2], bottom), width[1], height, lw=0,
                                                          alpha=0.75, facecolor="grey", edgecolor=None)

            plt.gca().add_patch(rectangle_lhs)
            plt.gca().add_patch(rectangle_rhs)
            plt.gca().add_patch(rectangle_driving_region)

        self._fig.tight_layout()

    def plot_row_spatial(self, row_index: int = -1, should_annotate_parameters: bool = False,
                         fixed_ylim: bool = False, interactive_plot: bool = False) -> None:
        """
        Plot a row of data to show spatial evolution.

        A row corresponds to an instant in time, so this can be particularly useful for investigating the final state
        of a system. Also, can be used to show the evolution of the whole system if multiple images are generated.

        :param fixed_ylim:
        :param interactive_plot:
        :param row_index: Plot given row from dataset; most commonly plotted should be the default.
        :param should_annotate_parameters: Add simulation parameters to plot. Useful when presenting work in meetings.

        :return: Saves a .png to the nominated 'Outputs' directory.
        """
        self._draw_figure(row_index, draw_regions_of_interest=False, static_ylim=fixed_ylim)

        self._tick_setter(self._axes, 1000, 200, 3, 4,
                          yaxis_num_decimals=1.1, show_sci_notation=False)

        _, y_major_labels, _ = self._choose_scaling(value=abs(self._axes.get_ylim()[1]))

        self._axes.set(ylabel=f"m$_x$ (a.u. " + y_major_labels[1] + ")")

        self._plot_cleanup(self._axes)

        if should_annotate_parameters:
            if self.exchange_heisenberg_min() == self.exchange_heisenberg_max():
                exchange_string = f"Uniform Exc.: {self.exchange_heisenberg_min()} (T)"
            else:
                exchange_string = f"J$_{{min}}$ = {self.exchange_heisenberg_min()} (T) | J$_{{max}}$ = " \
                                  f"{self.exchange_heisenberg_min()} (T)"

            parameters_textbody = (f"H$_{{0}}$ = {self.bias_zeeman_static()} (T) | N = {self.num_sites_chain()} | " + r"$\alpha$" +
                                   f" = {self.gilbert_chain(): 2.2e}\nH$_{{D1}}$ = {self.bias_zeeman_oscillating_1(): 2.2e} (T) | "
                                   f"H$_{{D2}}$ = {self.bias_zeeman_oscillating_2(): 2.2e} (T) \n{exchange_string}")

            parameters_text_props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.5)
            self._axes.text(0.05, 1.2, parameters_textbody, transform=self._axes.transAxes, fontsize=12,
                            ha='center', va='center', bbox=parameters_text_props)

        self._fig.savefig(f"{self.output_filepath}_row{row_index}.png", bbox_inches="tight")

        if interactive_plot:
            # Initialize variables
            num_clicks = 0
            total_diff_wl = 0
            last_wl = None

            def reset_clicks():
                nonlocal num_clicks, total_diff_wl, last_wl
                num_clicks = 0
                total_diff_wl = 0
                last_wl = None
                print("Click data has been reset.")

            # For interactive plots
            def mouse_event(event: Any):
                nonlocal num_clicks, total_diff_wl, last_wl

                if event.button == 3:  # Assuming right click is the reset trigger
                    reset_clicks()
                    return

                # Increment clicks count
                num_clicks += 1

                # Calculate difference and average only from the second click
                if last_wl is not None:
                    diff_wl = abs(event.xdata - last_wl)
                    total_diff_wl += diff_wl
                    # Ensure at least one difference has been calculated before averaging
                    if num_clicks > 1:
                        avg_diff_wl = total_diff_wl / (num_clicks - 1)
                        print(
                            f'Click #{num_clicks}: x: {event.xdata:.1f}, Avg. \u03BB: {avg_diff_wl:.1f}, '
                            f'Avg. k: {(2 * np.pi / avg_diff_wl):.3e} | y: {event.ydata:.3e}')
                    else:
                        print(f'Click #{num_clicks}: x: {event.xdata}, y: {event.ydata}')
                else:
                    print(f'Click #{num_clicks}: x: {event.xdata}, y: {event.ydata}')

                # Update last_wl with the current xdata for the next click
                last_wl = event.xdata

            self._fig.canvas.mpl_connect('button_press_event', mouse_event)
            self._fig.tight_layout()
            plt.show()

        plt.close(self._fig)

    def _process_signal(self, ax: Optional[Union[plt.Axes, List[plt.Axes]]] = None, row_index: int = -1,
                        signal_index: int = 1, scalar_map: mpl.cm.ScalarMappable = None,
                        plot_scheme: PlotScheme = None,
                        colour_scheme_settings: List[int | bool] = (1, False), negative_wavevector: bool = False):

        signal_xlims = plot_scheme[f'signal{signal_index}_xlim']
        system_xlims = plot_scheme['signal_xlim']
        system_xlim_rescale = plot_scheme['signal_rescale']
        signal_xlim_min, signal_xlim_max = signal_xlims[0], signal_xlims[1]
        if signal_xlim_min == signal_xlim_max:
            # Escape case
            return

        ########################################
        if ax is None:
            ax = [plt.gca()]
        elif isinstance(ax, plt.Axes):
            ax = [ax]
        if plot_scheme is None:
            exit("No plot scheme provided for signal processing.")

        ########################################
        # Access colour scheme for time evolution and plot
        colour_scheme = colour_schemes[colour_scheme_settings[0]]
        if colour_scheme_settings[1]:
            signal_colour = colour_scheme['ax1_colour_matte']
        else:
            signal_colour = colour_scheme[f'signal{signal_index}_colour']

        if scalar_map is None:
            norm = mpl.colors.Normalize(vmin=0, vmax=1)  # Assuming you want to normalize from 0 to 1
            cmap = 'hot'
            scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

        ########################################
        # Internal constants
        hz_pi_to_norm = 2 * np.pi  # Convert from 1/k to k

        ########################################
        # Take FFT of the signal
        wavevectors, fourier_transform = self._fft_data(self.amplitude_data[row_index,
                                                        signal_xlim_min:signal_xlim_max],
                                                        spatial_spacing=self.lattice_constant(), fft_window='hamming')

        ########################################
        # Obtain frequencies: default is 'angular' when γ in [rad * s^-1 * T^-1]
        # Need to flip negative wavevectors to ensure all values can be shown on single side of plot
        wavevectors *= -hz_pi_to_norm if negative_wavevector else hz_pi_to_norm

        if (self.has_dmi and self.is_dmi_only_within_map and self.has_dmi_map
                and (signal_xlims[0] < self.driving_region_lhs() or signal_xlims[1] > self.driving_region_rhs())):
            # If the system has a valid DMI map which this region is out with then set the DMI to zero
            region_dmi = 0.0
        else:
            region_dmi = self.exchange_dmi_constant()

        # hz_pi_to_norm required for `frequencies` to be linear and not angular when γ in [rad * s^-1 * T^-1]
        frequencies = ((self.gyro_mag() / hz_pi_to_norm) *
                       (self.exchange_heisenberg_max() * (self.lattice_constant() ** 2) * (wavevectors ** 2)
                        + self.bias_zeeman_static()
                        + region_dmi * self.lattice_constant() * wavevectors))

        wavevectors *= -1 * self.hz_to_Ghz if negative_wavevector else self.hz_to_Ghz
        frequencies *= self.hz_to_Ghz

        # Find the index of the element with the minimum absolute difference
        absolute_diff = np.abs(wavevectors - plot_scheme['ax2_xlim'][1])
        closest_index = np.argmin(absolute_diff)

        # Find the maximum of vec2 up to the count
        max_fft_value = np.max(fourier_transform[:closest_index])
        if len(self.track_zorder[0]) == 0:
            self.track_zorder[0].append(max_fft_value)
            self.track_zorder[1].append(1.3)
            zorder_to_use = self.track_zorder[1][0]
        else:
            if max_fft_value < np.min(self.track_zorder[0]):
                # New value is smaller, so push to front of plot
                update_zorder = min(self.track_zorder[1]) + 0.01
                zorder_to_use = update_zorder
                self.track_zorder[0].append(max_fft_value)
                self.track_zorder[1].append(update_zorder)
            elif max_fft_value > np.max(self.track_zorder[0]):
                # New value is larger, so push to back of plot
                update_zorder = max(self.track_zorder[1]) - 0.01
                zorder_to_use = update_zorder
                self.track_zorder[0].append(max_fft_value)
                self.track_zorder[1].append(update_zorder)
            else:
                zorder_to_use = self.track_zorder[1][0]

        ########################################
        ax[0].plot(np.arange(signal_xlim_min, signal_xlim_max),
                   self.amplitude_data[row_index, signal_xlim_min:signal_xlim_max],
                   ls='-', lw=1.5, color=signal_colour, label=f"Segment {signal_index}",
                   markerfacecolor='black', markeredgecolor='black', zorder=zorder_to_use)

        ax[1].plot(wavevectors, fourier_transform,
                   lw=1.5, color=signal_colour, marker='', markerfacecolor='black', markeredgecolor='black',
                   label=f"Segment {signal_index}", zorder=zorder_to_use)

        scatter_x = -wavevectors if negative_wavevector else wavevectors
        ax[2].scatter(scatter_x, frequencies,
                      c=scalar_map.to_rgba(fourier_transform), s=12, marker='.',
                      label=f"Segment {signal_index}", zorder=zorder_to_use)

    def _plot_cleanup(self, axis=None):
        if axis is None:
            axis = [plt.gca()]
        elif isinstance(axis, plt.Axes):
            axis = [axis]

        for ax in axis:
            ax.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=1.99)
            ax.grid(visible=False, axis='both', which='both')

            for tick_pos in [1, -2]:
                tick_label_last = ax.get_xticklabels()[tick_pos]
                tick_label_last.set_visible(False)

                tick_label_last.set_visible(False)
                xtick_position, ytick_position = tick_label_last.get_position()
                tick_text = tick_label_last.get_text()
                if len(axis) != 3:
                    y_offset = -0.045
                else:
                    y_offset = 0.125 if ax == axis[2] else -0.045
                if tick_pos == 1:
                    ax.text(xtick_position, ytick_position + y_offset, str(tick_text), ha='left', va='top',
                            fontsize=self._fontsizes["smaller"], transform=ax.get_xaxis_transform())
                else:
                    ax.text(xtick_position, ytick_position + y_offset, str(tick_text), ha='right', va='top',
                            fontsize=self._fontsizes["smaller"], transform=ax.get_xaxis_transform())

            ax.set_axisbelow(False)  # Must be last manipulation of subplots

    def plot_row_spatial_ft(self, row_index: int = -1,
                            fixed_ylim: bool = False, interactive_plot: bool = False) -> None:
        """
        Plot a row of data to show spatial evolution.

        A row corresponds to an instant in time, so this can be particularly useful for investigating the final state
        of a system. Also, can be used to show the evolution of the whole system if multiple images are generated.

        :param fixed_ylim:
        :param interactive_plot:
        :param row_index: Plot given row from dataset; most commonly plotted should be the default.

        :return: Saves a .png to the nominated 'Outputs' directory.
        """
        self._fig = plt.figure(figsize=(4.5, 6))
        plt.rcParams.update({'savefig.dpi': 1200})
        num_rows, num_cols = 3, 3
        ax_subplots = []

        for i in range(0, 3):
            ax_subplots.append(plt.subplot2grid((num_rows, num_cols), (i, 0), rowspan=1,
                                                colspan=num_cols, fig=self._fig))

        self._fig.subplots_adjust(wspace=1, hspace=0.4, bottom=0.2)

        ########################################
        # Nested Dict to enable many cases (different plots and papers)
        plot_schemes: Dict[int, PaperFigures.PlotScheme] = {
            0: {
                'signal_xlim': [0, self.num_sites_total()],  # Example value, replace 100 with self._total_num_spins()
                'signal_rescale': [-int(self.num_sites_total() / 2), int(self.num_sites_total() / 2)],
                'rescale_extras': [self.lattice_constant() if self.lattice_constant.dtype is not None else 1, 1e-6,
                                   'um'],
                'ax2_xlim': [0.0, 0.25],
                'ax2_ylim': [1e-3, 1e1],
                'ax3_xlim': [-0.15, 0.15],
                'ax3_ylim': [0, 20],
                'signal1_xlim': [int(self.num_sites_abc + 1),
                                 int(self.driving_region_lhs - 1)],
                'signal2_xlim': [int(self.driving_region_rhs + 1),
                                 int(self.num_sites_total - self.num_sites_abc + 1)],
                'signal3_xlim': [0, 0],
                'ax1_label': '(a)',
                'ax2_label': '(b)',
                'ax3_label': '(c)',
                'ax1_line_height': 3.15e-3
            }
        }

        if self.is_dmi_only_within_map() and self.has_dmi_map():
            plot_schemes[0]['signal1_xlim'] = [int(self.num_sites_abc + 1 - self.dmi_region_offset()),
                                               int(self.driving_region_lhs - 1)]
            plot_schemes[0]['signal2_xlim'] = [int(self.driving_region_rhs + 1 + self.dmi_region_offset()),
                                               int(self.num_sites_total - self.num_sites_abc + 1)]

        ########################################
        # Accessing the selected colour scheme and create a colourbar
        select_plot_scheme = plot_schemes[0]

        #for term in ['signal1_xlim', 'signal2_xlim', 'signal_xlim']:
        #    select_plot_scheme[term] *= self.lattice_constant()
        #    print(select_plot_scheme[term])

        # Create a ScalarMappable for the color mapping
        norm = mpl.colors.Normalize(vmin=0, vmax=1)  # Assuming you want to normalize from 0 to 1
        cmap = 'magma_r'
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

        # Adding a colourbar to ax3 using the ScalarMappable
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax_subplots[2])
        cax3 = divider.append_axes("bottom", size="7.5%", pad=0.0)

        # Set the ticks at the top and bottom using normalized values
        ax3_cbar = self._fig.colorbar(sm, ax=ax_subplots[2], cax=cax3, location='bottom', orientation='horizontal',
                                      shrink=1.0)
        ax3_cbar.set_label('Intensity (a.u.)', loc='center', labelpad=-5)
        ax3_cbar.ax.tick_params(axis='x', top=False, bottom=True, pad=3.5)
        ax3_cbar.set_ticks(ticks=[norm.vmin + 0.03, norm.vmax - 0.035], labels=['Min', 'Max'])
        #ax3_cbar.set_ticks(ticks=[norm.vmin + 0.03, norm.vmax - 0.03], labels=[str(norm.vmin), str(norm.vmax)])

        ########################################
        # ax_subplots[1] is handled in the _draw_figure method
        ax_subplots[0].set(yscale='linear')
        ax_subplots[1].set(xlabel=f"Wavevector, $k$ (nm"r"$^{-1}$)", ylabel=f"Intensity (a.u.)", yscale='log',
                           xlim=select_plot_scheme['ax2_xlim'], ylim=select_plot_scheme['ax2_ylim'])
        ax_subplots[2].set(ylabel=f"Frequency, $f$ (GHz)", yscale='linear',
                           xlim=select_plot_scheme['ax3_xlim'], ylim=select_plot_scheme['ax3_ylim'])
        ax_subplots[2].tick_params(pad=2, labeltop=True, labelbottom=False, labelsize=self._fontsizes["smaller"])
        ax_subplots[2].invert_yaxis()
        ########################################
        # Process each signal
        self._draw_figure(row_index, axes=ax_subplots[0], draw_regions_of_interest=False, static_ylim=fixed_ylim)

        self._process_signal(ax_subplots, row_index, 1, sm, select_plot_scheme, [2, False], True)
        self._process_signal(ax_subplots, row_index, 2, sm, select_plot_scheme, [2, False])
        self._process_signal(ax_subplots, row_index, 3, sm, select_plot_scheme, [2, False])

        ########################################
        self._tick_setter(ax_subplots[0], int(self.num_sites_total()/4), int(self.num_sites_total()/8), 3, 4,
                          yaxis_num_decimals=1.1, show_sci_notation=False, xaxis_rescale=self.lattice_constant())

        self._tick_setter(ax_subplots[1], select_plot_scheme['ax2_xlim'][1] / 2, select_plot_scheme['ax2_xlim'][1] / 8,
                          4, None, xaxis_num_decimals=1.2, is_fft_plot=True)

        self._tick_setter(ax_subplots[2], x_major=select_plot_scheme['ax3_xlim'][1] / 2,
                          x_minor=select_plot_scheme['ax3_xlim'][1] / 4,
                          y_major=select_plot_scheme['ax3_ylim'][1] / 2, y_minor=select_plot_scheme['ax3_ylim'][1] / 4,
                          yaxis_multi_loc=True, xaxis_num_decimals=.2, yaxis_num_decimals=2.1, yscale_type='plain')
        ########################################
        # Post-processing actions
        ax_subplots[0].set(ylim=[-1e-2, 1e-2])
        _, y_major_labels, _ = self._choose_scaling(subplot_to_scale=ax_subplots[0], row_index=row_index)
        ax_subplots[0].set(ylabel=f"m$_x$ (a.u. " + y_major_labels[1] + ")")

        ax_subplots[0].legend(ncol=1, loc='best', fontsize=self._fontsizes["tiny"], frameon=False, fancybox=True,
                              facecolor=None, edgecolor=None)
        ax_subplots[1].legend(ncol=1, loc='best', fontsize=self._fontsizes["tiny"], frameon=False, fancybox=True,
                              facecolor=None, edgecolor=None)
        # ax3.legend(ncol=1, loc='upper center', fontsize=self._fontsizes["small"], frameon=False, fancybox=True,
        #           facecolor=None, edgecolor=None)
        plt.tight_layout(w_pad=0.2, h_pad=0.25)
        self._plot_cleanup(ax_subplots)

        ########################################
        self._fig.savefig(f"{self.output_filepath}_row{row_index}_ft.png", bbox_inches="tight")
        output_text_to_file = True
        if output_text_to_file:
            try:
                # Open the file in append mode
                with (open(f"D:\\Data\\2024-04-01\\Outputs\\T1317_details.txt", 'a') as file):
                    # Write the data to the file
                    data_to_write = (f"{self.output_filepath}"
                                     f","
                                     f"{self.amplitude_data[row_index, int(self.driving_region_lhs() - np.round(1e-6/1e-9))]}"
                                     f","
                                     f"{self.amplitude_data[row_index, int(self.driving_region_rhs() + np.round(1e-6/1e-9))]}")
                    file.write(data_to_write + '\n')  # Assuming 'data' is a string with two columns separated by a comma
                    file.close()
            except IOError:
                print("Error: Unable to append data to the file.")
            finally:
                print(f"Data written to file:\n"
                      f"\t- Before: {self.amplitude_data[row_index, int(self.driving_region_lhs() - np.round(1e-6 / 1e-9))]}\n"
                      f"\t- After : {self.amplitude_data[row_index, int(self.driving_region_rhs() + np.round(1e-6 / 1e-9))]}"
                      )

        # All additional functionality should be after here
        if interactive_plot:
            figure_manager = FigureManager(self._fig, ax_subplots, self._fig.get_size_inches()[0],
                                           self._fig.get_size_inches()[1], self._fig.dpi, self.driving_freq(),
                                           select_plot_scheme)
            figure_manager.connect_events()
            figure_manager.wait_for_close()

        plt.close(self._fig)

    def _plot_paper_gif(self, row_index: int, has_static_ylim: bool = False) -> plt.Figure:
        """
        Private method to save a given row of a data as a frame suitable for use with the git library.

        Requires decorator so use method as an inner class instead of creating child class.

        :param row_index: The row to be plotted.

        :return: Method indirectly updates `self._fig` and `self.axis` by calling self._draw_figure().
        """
        self._draw_figure(row_index, False, draw_regions_of_interest=False, publish_plot=False,
                          static_ylim=has_static_ylim)

        self._tick_setter(self._axes, int(self.num_sites_total()/4), int(self.num_sites_total()/8), 3, 4,
                          yaxis_num_decimals=1.1, show_sci_notation=False, xaxis_rescale=self.lattice_constant())

        self._axes.text(1.0, -0.095, f"{self.time_data[row_index]: .2f} ns", va='center',
                        ha='right', transform=self._axes.transAxes, fontsize=self._fontsizes["small"])

        self._fig.tight_layout()

        return self._fig

    def create_gif(self, number_of_frames: float = 0.01,
                   frames_per_second: float = 10, has_static_ylim: bool = False) -> None:
        frame_filenames = []

        for index in range(0, int(self.num_dp_per_site + 1), int(self.num_dp_per_site * number_of_frames)):
            frame = self._plot_paper_gif(index, has_static_ylim=has_static_ylim)
            frame_filename = f"{self.output_filepath}_{index}.png"
            frame.savefig(frame_filename)
            frame_filenames.append(frame_filename)
            plt.close(frame)  # Close the figure to free memory

        with imio.get_writer(f"{self.output_filepath}.gif", mode='I', fps=frames_per_second, loop=0) as writer:
            for filename in frame_filenames:
                image = imio.v3.imread(filename)
                writer.append_data(image)

        # Clean up by removing the individual frame files
        for filename in frame_filenames:
            os.remove(filename)

    def plot_site_temporal(self, site_index: int, wavepacket_fft: bool = False,
                           visualise_wavepackets: bool = False, annotate_precursors_fft: bool = False,
                           annotate_signal: bool = False, wavepacket_inset: bool = False,
                           add_key_params: bool = False, add_signal_backgrounds: bool = False,
                           publication_details: bool = False, interactive_plot: bool = False) -> None:
        """
        Two pane figure where upper pane shows the time evolution of the magnetisation of a site, and the lower pane
        shows the FFT of the precursors, shockwave, equilibrium region, and wavepacket(s) (optional).

        One should ensure that the site being plotted is not inside either of the driving- or damping-regions. Example
        data at 2023-03-06/rk2_mx_T1118 .

        Throughout this method, wavepacket1 will always refer to the wavepacket in the
        precursor region CLOSEST to the shockwave edge, and increasing numbers refers to further away (and closer to the
        signal start point). The variables say 'equil` while the plot labels say 'steady state' - this is make the
        variables easier to read. Can update plot annotations if 'Equilibrium' is a better fix.

        :param site_index: Number of site being plotted.
        :param wavepacket_fft: Take FFT of each wavepacket and plot.
        :param visualise_wavepackets: Redraw each wavepacket with high-contrast colours on time evolution pane.
        :param annotate_precursors_fft: Add arrows/labels to denote wavepackets (P1 wavepacket closest to shock-edge).
        :param annotate_signal: Draw regions (with labels) onto time evolution showing precursor/shock/equil regions.
        :param wavepacket_inset: On time evolution pane, zoom in on precursor region to show the wavepackets as inset.
        :param add_key_params: Add text box between panes which lists key simulation parameters.
        :param add_signal_backgrounds: Shade behind signal regions to improve clarity; backup to `annotate_signals`. 
        :param publication_details: Add figure reference label and scientific notation to y-axis. Needs edited each run
        :param interactive_plot: If `True` mouse-clicks to print x/y coords to terminal, else saves image.

        :return: Saves a .png image to the designated output folder.
        """
        #

        # Setup figure environment
        if self._fig is None:
            self._fig = plt.figure(figsize=(4.5, 3.375))

        num_rows = 2
        num_cols = 3

        ax1 = plt.subplot2grid((num_rows, num_cols), (0, 0), rowspan=int(num_rows / 2),
                               colspan=num_cols, fig=self._fig)
        ax2 = plt.subplot2grid((num_rows, num_cols), (int(num_rows / 2), 0),
                               rowspan=num_rows, colspan=num_cols, fig=self._fig)

        self._fig.subplots_adjust(wspace=1, hspace=0.35)

        ax1.xaxis.labelpad = -1
        ax2.xaxis.labelpad = -1

        # ax1_yaxis_base, ax1_yaxis_exponent = 3, '-3'
        # ax1_yaxis_order = float('1e' + ax1_yaxis_exponent)

        # line_height = -3.15 * ax1_yaxis_order  # Assists formatting when drawing precursor lines on plots

        ########################################
        # Set colour scheme

        # Accessing the selected colour scheme
        select_colour_scheme = 2
        is_colour_matte = False
        selected_scheme = colour_schemes[select_colour_scheme]

        ########################################
        # All times in nanoseconds (ns)
        plot_schemes = {  # D:\Data\2023-04-19\Outputs
            0: {  # mceleney2023dispersive Fig. 1b-c [2022-08-29/T1337_site3000]
                'signal_xlim': (0.0, self.sim_time_max),
                'ax1_xlim': [0.0, 5.0],
                'ax1_ylim': [-4.0625e-3, 3.125e-3],
                'ax1_inset_xlim': [0.7, 2.6],
                'ax1_inset_ylim': [-2e-4, 2e-4],
                'ax1_inset_width': 1.95,
                'ax1_inset_height': 0.775,
                'ax1_inset_bbox': [0.08, 0.975],
                'ax2_xlim': [0.0, 99.9999],
                'ax2_ylim': [1e-1, 1e3],
                'precursor_xlim': (0, 2.6),  # 12:3356
                'signal_onset_xlim': (2.6, 3.79),  # 3445:5079
                'equilib_xlim': (3.8, 5.0),  # 5079::
                'ax1_label': '(b)',
                'ax2_label': '(c)',
                'ax1_line_height': 3.15e-3
            },
            1: {  # Jiahui T0941/T1107_site3
                'signal_xlim': (0.0, self.sim_time_max),
                'ax1_xlim': [0.0, 1.50 - 0.00001],
                'ax1_ylim': [self.amplitude_data[:, site_index].min(), self.amplitude_data[:, site_index].max()],
                'ax1_inset_xlim': [0.01, 0.02],
                'ax1_inset_ylim': [-2e-4, 2e-4],
                'ax1_inset_width': 1.95,
                'ax1_inset_height': 0.775,
                'ax1_inset_bbox': [0.08, 0.975],
                'ax2_xlim': [0.0001, 599.9999],
                'ax2_ylim': [1e-2, 1e1],
                'precursor_xlim': (0.0, 0.99),  # 0.0, 0.75
                'signal_onset_xlim': (0.0, 0.01),  # 0.75, 1.23
                'equilib_xlim': (0.99, 1.5),  # 1.23, 1.5
                'ax1_label': '(a)',
                'ax2_label': '(b)',
                'ax1_line_height': int(self.amplitude_data[:, site_index].min() * 0.9)
            },
            2: {  # Jiahui T0941/T1107_site1
                'signal_xlim': (0.0, self.sim_time_max),
                'ax1_xlim': [0.0, 1.50 - 0.00001],
                'ax1_ylim': [self.amplitude_data[:, site_index].min(), self.amplitude_data[:, site_index].max()],
                'ax1_inset_xlim': [0.01, 0.02],
                'ax1_inset_ylim': [-2e-4, 2e-4],
                'ax1_inset_width': 1.95,
                'ax1_inset_height': 0.775,
                'ax1_inset_bbox': [0.08, 0.975],
                'ax2_xlim': [0.0001, 119.9999],
                'ax2_ylim': [1e-3, 1e1],  # A            B            C            D           E
                'precursor_xlim': (0.0, 0.42),  # (0.00, 0.54) (0.00, 0.42) (0.00, 0.42) (0.00, 0.65) (0.00, 0.42)
                'signal_onset_xlim': (0.42, 0.65),  # (0.00, 0.01) (0.42, 0.54) (0.42, 0.65) (0.65, 1.20) (0.42, 1.20)
                'equilib_xlim': (0.65, 1.5),  # (0.54, 1.50) (0.54, 1.50) (0.65, 1.50) (1.20, 1.50) (1.20, 1.50)
                'ax1_label': '(a)',
                'ax2_label': '(b)',
                'ax1_line_height': int(self.amplitude_data[:, site_index].min() * 0.9)
            },
            3: {  # Test for me
                'signal_xlim': (0.0, self.sim_time_max),
                'ax1_xlim': [0.0, self.sim_time_max - 0.00001],
                'ax1_ylim': [self.amplitude_data[:, site_index].min(), self.amplitude_data[:, site_index].max()],
                'ax1_inset_xlim': [0.01, 0.02],
                'ax1_inset_ylim': [-2e-4, 2e-4],
                'ax1_inset_width': 1.95,
                'ax1_inset_height': 0.775,
                'ax1_inset_bbox': [0.08, 0.975],
                'ax2_xlim': [0.0001, 99.9999],
                'ax2_ylim': [1e-2, 1e1],  # A            B            C            D           E
                'precursor_xlim': (0.3, 1.2),  # (0.00, 0.54) (0.00, 0.42) (0.00, 0.42) (0.00, 0.65) (0.00, 0.42)
                'signal_onset_xlim': (0.0, 0.3),  # (0.00, 0.01) (0.42, 0.54) (0.42, 0.65) (0.65, 1.20) (0.42, 1.20)
                'equilib_xlim': (0.00, 0.00),  # (0.54, 1.50) (0.54, 1.50) (0.65, 1.50) (1.20, 1.50) (1.20, 1.50)
                'ax1_label': '(a)',
                'ax2_label': '(b)',
                'ax1_line_height': int(self.amplitude_data[:, site_index].min() * 0.9)
            }
        }

        select_plot_scheme = plot_schemes[3]
        signal_xlim_min, signal_xlim_max = select_plot_scheme['signal_xlim']
        ax1_xlim_lower, ax1_xlim_upper = select_plot_scheme['ax1_xlim']

        precursors_xlim_min_raw, precursors_xlim_max_raw = select_plot_scheme['precursor_xlim']
        shock_xlim_min_raw, shock_xlim_max_raw = select_plot_scheme['signal_onset_xlim']
        equil_xlim_min_raw, equil_xlim_max_raw = select_plot_scheme['equilib_xlim']

        # TODO Need to find all the values and turn this section into a dictionary
        wavepacket_schemes = {
            0: {  # mceleney2023dispersive Fig. 1b-c [2022-08-29/T1337_site3000]
                'wp1_xlim': (1.75, precursors_xlim_max_raw),
                'wp2_xlim': (1.3, 1.7),
                'wp3_xlim': (1.05, 1.275)
            },
            1: {  # mceleney2023dispersive Fig. 3a-b [2022-08-08/T1400_site3000]
                'wp1_xlim': (0.481, 0.502),
                'wp2_xlim': (0.461, 0.480),
                'wp3_xlim': (0.442, 0.4605)
            },
            2: {  # mceleney2023dispersive Fig. 3a-b [2022-08-08/T1400_site3000]
                'wp1_xlim': (0.01, 0.02),
                'wp2_xlim': (0.02, 0.03),
                'wp3_xlim': (0.03, 0.04)
            }
            # ... add more wavepacket schemes as needed
        }
        select_wp_vals = wavepacket_schemes[2]
        wavepacket1_xlim_min_raw, wavepacket1_xlim_max_raw = select_wp_vals['wp1_xlim']
        wavepacket2_xlim_min_raw, wavepacket2_xlim_max_raw = select_wp_vals['wp2_xlim']
        wavepacket3_xlim_min_raw, wavepacket3_xlim_max_raw = select_wp_vals['wp3_xlim']

        # If element [0-2] are changed, must also update calls for `converted_values` below
        data_names = ['Signal 1', 'Signal 2', 'Signal 3', 'wavepacket1', 'wavepacket2', 'wavepacket3']
        wavepacket_labels = ['P1', 'P2', 'P3']  # Maybe make into a dict also having x/y coords contained here
        raw_values = {
            'Signal 1': (precursors_xlim_min_raw, precursors_xlim_max_raw),
            'Signal 2': (shock_xlim_min_raw, shock_xlim_max_raw),
            'Signal 3': (equil_xlim_min_raw, equil_xlim_max_raw),
            'wavepacket1': (wavepacket1_xlim_min_raw, wavepacket1_xlim_max_raw),
            'wavepacket2': (wavepacket2_xlim_min_raw, wavepacket2_xlim_max_raw),
            'wavepacket3': (wavepacket3_xlim_min_raw, wavepacket3_xlim_max_raw)
        }

        ########################################

        ax1.set(xlabel=f"Time (ns)", ylabel=r"$\mathrm{m_x}$", xlim=select_plot_scheme['ax1_xlim'],
                ylim=select_plot_scheme['ax1_ylim'])  # "$($10^{-4}$)"

        ax2.set(xlabel=f"Frequency (GHz)", ylabel=f"Amplitude (arb. units)", yscale='log',
                xlim=select_plot_scheme['ax2_xlim'], ylim=select_plot_scheme['ax2_ylim'], )

        self._tick_setter(ax1, 0.5, 0.25, 3, 4, xaxis_num_decimals=1.1,
                          show_sci_notation=True)
        ax2_xlim_round = round(select_plot_scheme['ax2_xlim'][1], 0)
        self._tick_setter(ax2, int(ax2_xlim_round / 5), int(ax2_xlim_round / 10), 3, None,
                          xaxis_num_decimals=1.2, is_fft_plot=True)

        ########################################
        if ax1_xlim_lower > ax1_xlim_upper:
            exit(0)

        def convert_norm(val, a=0, b=1):
            # Magic. Don't touch! Normalises precursor region so that both wavepackets and feature can be defined using
            # their own x-axis limits.
            return int(self.num_dp_per_site * ((b - a) * ((val - signal_xlim_min)
                                                           / (signal_xlim_max - signal_xlim_min)) + a))

        converted_values = {name: (convert_norm(raw_values[name][0]), convert_norm(raw_values[name][1])) for name in
                            data_names}

        precursors_xlim_min, precursors_xlim_max = converted_values['Signal 1']
        shock_xlim_min, shock_xlim_max = converted_values['Signal 2']
        equil_xlim_min, equil_xlim_max = converted_values['Signal 3']
        wavepacket1_xlim_min, wavepacket1_xlim_max = converted_values['wavepacket1']
        wavepacket2_xlim_min, wavepacket2_xlim_max = converted_values['wavepacket2']
        wavepacket3_xlim_min, wavepacket3_xlim_max = converted_values['wavepacket3']

        ########################################
        # Access colour scheme for time evolution and plot
        if is_colour_matte:
            ax1_colour_matte = precursor_colour = shock_colour = equil_colour = selected_scheme['ax1_colour_matte']
        else:
            ax1_colour_matte = selected_scheme['ax1_colour_matte']
            precursor_colour = selected_scheme['precursor_colour']
            shock_colour = selected_scheme['shock_colour']
            equil_colour = selected_scheme['equil_colour']

        ax1.plot(self.time_data[:], self.amplitude_data[:, site_index],
                 ls='-', lw=0.75, color=f'{ax1_colour_matte}', alpha=0.5,
                 markerfacecolor='black', markeredgecolor='black', zorder=1.01)
        if not precursors_xlim_min == precursors_xlim_max:
            ax1.plot(self.time_data[precursors_xlim_min:precursors_xlim_max],
                     self.amplitude_data[precursors_xlim_min:precursors_xlim_max, site_index],
                     ls='-', lw=0.75, color=f'{precursor_colour}', label=f"{self.sites_array[site_index]}",
                     markerfacecolor='black', markeredgecolor='black', zorder=1.1)
        if not shock_xlim_min == shock_xlim_max:
            ax1.plot(self.time_data[shock_xlim_min:shock_xlim_max],
                     self.amplitude_data[shock_xlim_min:shock_xlim_max, site_index],
                     ls='-', lw=0.75, color=f'{shock_colour}', label=f"{self.sites_array[site_index]}",
                     markerfacecolor='black', markeredgecolor='black', zorder=1.1)
        if not equil_xlim_min == equil_xlim_max:
            ax1.plot(self.time_data[equil_xlim_min:equil_xlim_max],
                     self.amplitude_data[equil_xlim_min:equil_xlim_max, site_index],
                     ls='-', lw=0.75, color=f'{equil_colour}', label=f"{self.sites_array[site_index]}",
                     markerfacecolor='black', markeredgecolor='black', zorder=1.1)

        ########################################
        # Access colour scheme again for FFT of time evolution
        # ax2_colour_matte = selected_scheme['ax2_colour_matte']
        precursor_colour = selected_scheme['precursor_colour']
        shock_colour = selected_scheme['shock_colour']
        equil_colour = selected_scheme['equil_colour']

        if not precursors_xlim_min == precursors_xlim_max:
            frequencies_precursors, fourier_transform_precursors = (
                self._fft_data(self.amplitude_data[precursors_xlim_min:precursors_xlim_max, site_index]))
            ax2.plot(frequencies_precursors, abs(fourier_transform_precursors),
                     lw=1, color=f"{precursor_colour}", marker='', markerfacecolor='black', markeredgecolor='black',
                     label=data_names[0], zorder=1.5)
        if not shock_xlim_min == shock_xlim_max:
            frequencies_dsw, fourier_transform_dsw = (
                self._fft_data(self.amplitude_data[shock_xlim_min:shock_xlim_max, site_index]))
            ax2.plot(frequencies_dsw, abs(fourier_transform_dsw),
                     lw=1, color=f'{shock_colour}', marker='', markerfacecolor='black', markeredgecolor='black',
                     label=data_names[1], zorder=1.2)
        if not equil_xlim_min == equil_xlim_max:
            frequencies_eq, fourier_transform_eq = (
                self._fft_data(self.amplitude_data[equil_xlim_min:convert_norm(signal_xlim_max), site_index]))
            ax2.plot(frequencies_eq, abs(fourier_transform_eq),
                     lw=1, color=f'{equil_colour}', marker='', markerfacecolor='black', markeredgecolor='black',
                     label=data_names[2], zorder=1.1)

        # for i, j, k in zip(abs(fourier_transform_precursors), abs(fourier_transform_dsw),
        #                    abs(fourier_transform_eq)):
        #     if i < 1:
        #         print(f'Small value PRE found: {i}')
        #     if j < 1:
        #         print(f'Small value DSW found: {j}')
        #     if k < 1:
        #         print(f'Small value EQ found: {k}')

        ax2.legend(ncol=1, loc='upper right', fontsize=self._fontsizes["tiny"], frameon=False, fancybox=True,
                   facecolor=None, edgecolor=None, bbox_to_anchor=[0.99, 0.975], bbox_transform=ax2.transAxes)

        ########################################
        if wavepacket_fft:
            wavepacket1_freqs, wavepacket1_fft = self._fft_data(
                self.amplitude_data[wavepacket1_xlim_min:wavepacket1_xlim_max, site_index])
            wavepacket2_freqs, wavepacket2_fft = self._fft_data(
                self.amplitude_data[wavepacket2_xlim_min:wavepacket2_xlim_max, site_index])
            wavepacket3_freqs, wavepacket3_fft = self._fft_data(
                self.amplitude_data[wavepacket3_xlim_min:wavepacket3_xlim_max, site_index])

            ax2.plot(wavepacket1_freqs, abs(wavepacket1_fft), marker='', lw=1, color=f'{precursor_colour}',
                     markerfacecolor='black', markeredgecolor='black', ls=':', zorder=1.9)
            ax2.plot(wavepacket2_freqs, abs(wavepacket2_fft), marker='', lw=1, color=f'{shock_colour}',
                     markerfacecolor='black', markeredgecolor='black', ls='--', zorder=1.9)
            ax2.plot(wavepacket3_freqs, abs(wavepacket3_fft), marker='', lw=1, color=f'{equil_colour}',
                     markerfacecolor='black', markeredgecolor='black', ls='-.', zorder=1.9)

        if annotate_precursors_fft:
            arrow_ax2_props = {"arrowstyle": '-|>', "connectionstyle": "angle3,angleA=0,angleB=90", "color": "black"}

            ax2.annotate(wavepacket_labels[0], xy=(26, 1.8e1), xytext=(34.1, 2.02e2), va='center', ha='center',
                         arrowprops=arrow_ax2_props, fontsize=self._fontsizes["smaller"])
            ax2.annotate(wavepacket_labels[1], xy=(48.78, 4.34e0), xytext=(56.0, 5.37e1), va='center', ha='center',
                         arrowprops=arrow_ax2_props, fontsize=self._fontsizes["smaller"])
            ax2.annotate(wavepacket_labels[2], xy=(78.29, 1.25e0), xytext=(83.9, 7.5), va='center', ha='center',
                         arrowprops=arrow_ax2_props, fontsize=self._fontsizes["smaller"])

        if visualise_wavepackets:
            ax1.plot(self.time_data[wavepacket1_xlim_min:wavepacket1_xlim_max],
                     self.amplitude_data[wavepacket1_xlim_min:wavepacket1_xlim_max, site_index],
                     lw=0.75, color=colour_schemes[0]["wavepacket1"], marker='', markerfacecolor='black',
                     markeredgecolor='black',
                     label="Shockwave", zorder=1.2)
            ax1.plot(self.time_data[wavepacket2_xlim_min:wavepacket2_xlim_max],
                     self.amplitude_data[wavepacket2_xlim_min:wavepacket2_xlim_max, site_index],
                     lw=0.75, color=colour_schemes[0]["wavepacket2"], marker='', markerfacecolor='black',
                     markeredgecolor='black',
                     label="Steady State", zorder=1.2)
            ax1.plot(self.time_data[wavepacket3_xlim_min:wavepacket3_xlim_max],
                     self.amplitude_data[wavepacket3_xlim_min:wavepacket3_xlim_max, site_index],
                     lw=0.75, color=colour_schemes[0]["wavepacket3"], marker='', markerfacecolor='black',
                     markeredgecolor='black',
                     label="Steady State", zorder=1.2)

        if annotate_signal:
            # Leave these alone!
            label_height = select_plot_scheme['ax1_line_height'] - 3 * 0.25 * -3
            precursor_label_props = {"arrowstyle": '|-|, widthA =0.4, widthB=0.4', "color": f"{precursor_colour}",
                                     'lw': 1.0}
            shock_label_props = {"arrowstyle": '|-|, widthA =0.4, widthB=0.4', "color": f"{shock_colour}", 'lw': 1.0}
            equil_label_props = {"arrowstyle": '|-|, widthA =0.4, widthB=0.4', "color": f"{equil_colour}", 'lw': 1.0}

            precursor_label_lhs, precursor_label_rhs = precursors_xlim_min_raw, precursors_xlim_max_raw
            shock_label_lhs, shock_label_rhs = shock_xlim_min_raw, shock_xlim_max_raw
            equil_label_lhs, equil_label_rhs = equil_xlim_min_raw, equil_xlim_max_raw

            ax1.annotate('', xy=(precursor_label_lhs, select_plot_scheme['ax1_line_height']),
                         xytext=(precursor_label_rhs, select_plot_scheme['ax1_line_height']),
                         va='center', ha='center', arrowprops=precursor_label_props, fontsize=self._fontsizes["tiny"])
            ax1.annotate('', xy=(shock_label_lhs, select_plot_scheme['ax1_line_height']),
                         xytext=(shock_label_rhs, select_plot_scheme['ax1_line_height']),
                         va='center', ha='center', arrowprops=shock_label_props, fontsize=self._fontsizes["tiny"])
            ax1.annotate('', xy=(equil_label_lhs, select_plot_scheme['ax1_line_height']),
                         xytext=(equil_label_rhs, select_plot_scheme['ax1_line_height']),
                         va='center', ha='center', arrowprops=equil_label_props, fontsize=self._fontsizes["tiny"])

            ax1.text((precursor_label_lhs + precursor_label_rhs) / 2, label_height, data_names[0], ha='center',
                     va='bottom',
                     fontsize=self._fontsizes["tiny"])
            ax1.text((shock_label_lhs + shock_label_rhs) / 2, label_height, data_names[1], ha='center', va='bottom',
                     fontsize=self._fontsizes["tiny"])
            ax1.text((equil_label_lhs + equil_label_rhs) / 2, label_height, data_names[2], ha='center', va='bottom',
                     fontsize=self._fontsizes["tiny"])

        if wavepacket_inset:
            # Add zoomed in region if needed.

            # Select datasets to use
            time_dataset = self.time_data
            amplitude_dataset = self.amplitude_data[:, site_index]
            # Impose inset onto plot. Use 0.24 for LHS and 0.8 for RHS. 0.7 for T and 0.25 for B
            ax1_inset = mpl_toolkits.axes_grid1.inset_locator.inset_axes(ax1,
                                                                         width=select_plot_scheme['ax1_inset_width'],
                                                                         height=select_plot_scheme['ax1_inset_height'],
                                                                         loc="lower left",
                                                                         bbox_to_anchor=select_plot_scheme[
                                                                             'ax1_inset_bbox'],
                                                                         bbox_transform=ax1.transAxes)
            ax1_inset.plot(time_dataset, amplitude_dataset, lw=0.75, color=f'{ax1_colour_matte}', zorder=1.1)

            if visualise_wavepackets:
                ax1_inset.plot(time_dataset[wavepacket1_xlim_min:wavepacket1_xlim_max],
                               amplitude_dataset[wavepacket1_xlim_min:wavepacket1_xlim_max], marker='',
                               lw=0.75, color=colour_schemes[0]["wavepacket1"],
                               markerfacecolor='black', markeredgecolor='black', label="Shockwave", zorder=1.2)
                ax1_inset.plot(self.time_data[wavepacket2_xlim_min:wavepacket2_xlim_max],
                               self.amplitude_data[wavepacket2_xlim_min:wavepacket2_xlim_max, site_index], marker='',
                               lw=0.75, color=colour_schemes[0]["wavepacket2"],
                               markerfacecolor='black', markeredgecolor='black', label="Steady State", zorder=1.2)
                ax1_inset.plot(self.time_data[wavepacket3_xlim_min:wavepacket3_xlim_max],
                               self.amplitude_data[wavepacket3_xlim_min:wavepacket3_xlim_max, site_index], marker='',
                               lw=0.75, color=colour_schemes[0]["wavepacket3"],
                               markerfacecolor='black', markeredgecolor='black', label="Steady State", zorder=1.2)

            # Select data (of original) to show in inset through changing axis limits
            # ylim_in = 2 * ax1_yaxis_order * 1e-1
            ax1_inset.set(xlim=select_plot_scheme['ax1_inset_xlim'], ylim=select_plot_scheme['ax1_inset_ylim'], )

            arrow_lower_props = {"arrowstyle": '-|>', "connectionstyle": 'angle3, angleA=0, angleB=40',
                                 "color": "black",
                                 'lw': 0.8}
            arrow_upper_props = {"arrowstyle": '-|>', "connectionstyle": 'angle3, angleA=0, angleB=140',
                                 "color": "black", 'lw': 0.8}

            ax1_inset.annotate(wavepacket_labels[0], xy=(1.85, -6e-5), xytext=(1.5, -1.3e-4), va='center', ha='center',
                               arrowprops=arrow_lower_props, fontsize=self._fontsizes["tiny"])
            ax1_inset.annotate(wavepacket_labels[1], xy=(1.45, 6e-5), xytext=(1.1, 1.3e-4), va='center', ha='center',
                               arrowprops=arrow_upper_props, fontsize=self._fontsizes["tiny"])
            ax1_inset.annotate(wavepacket_labels[2], xy=(1.15, -3e-5), xytext=(0.8, -1.3e-4), va='center', ha='center',
                               arrowprops=arrow_lower_props, fontsize=self._fontsizes["tiny"])

            # Override rcParams for inset
            ax1_inset.set_xticks([])
            ax1_inset.set_yticks([])
            ax1_inset.patch.set_color("#f9f2e9")  # #f0a3a9 is equivalent to color 'red' and alpha '0.3'

            rect = mpl.patches.Rectangle((0.7, -6e-4), 1.91, 1.2e-3, lw=1, edgecolor='black',
                                         facecolor='#f9f2e9')
            ax1.add_patch(rect)

            # Legacy Code. Kept for future reuse
            # mark_inset(ax1, ax1_inset,loc1=1, loc2=3, facecolor='#f9f2e9', edgecolor='black', alpha=1.0, zorder=1.05)
            # Add box to indicate the region which is being zoomed into on the main figure
            # ax1.indicate_inset_zoom(ax1_inset, facecolor='#f9f2e9', edgecolor='black', alpha=1.0, lw=0.75,
            #                        zorder=1)

        if add_key_params:
            if self.exchange_heisenberg_min == self.exchange_heisenberg_max:
                exchange_string = f"Uniform Exc. ({self.exchange_heisenberg_min} [T])"
            else:
                exchange_string = f"J$_{{min}}$ = {self.exchange_heisenberg_min} [T] | J$_{{max}}$ = " \
                                  f"{self.exchange_heisenberg_max} [T]"
            info_box_full_text = (
                    (f"H$_{{0}}$ = {self.bias_zeeman_static} [T] | H$_{{D1}}$ = {self.bias_zeeman_oscillating_1: 2.2e} [T] | "
                     f"H$_{{D2}}$ = {self.bias_zeeman_oscillating_2: 2.2e}[T] \nf = {self.driving_freq} [GHz] | "
                     f"{exchange_string} | N = {self.num_sites_chain} | ") + r"$\alpha$" +
                    f" = {self.gilbert_chain: 2.2e}")

            info_box_props = dict(boxstyle='round', facecolor='gainsboro', alpha=1.0)

            # Move xlabel on ax1 to make space for info box
            ax1.set(xlabel='')
            ax1.text(0.35, -0.24, info_box_full_text, transform=ax1.transAxes, fontsize=6,
                     verticalalignment='top', bbox=info_box_props, ha='center', va='center')
            ax1.text(0.85, -0.2, "Time [ns]", fontsize=12, ha='center', va='center',
                     transform=ax1.transAxes)

        if add_signal_backgrounds:
            extend_height = 0.375e-2  # Makes shaded region extend past the top/bottom of each region
            precursor_background = mpatches.Rectangle((0, -1 * self.amplitude_data[:, site_index].max()),
                                                      (precursors_xlim_max_raw - precursors_xlim_min_raw),
                                                      2 * self.amplitude_data[:, site_index].max() + extend_height,
                                                      alpha=0.3,
                                                      facecolor=colour_schemes[0]["wavepacket4"], edgecolor=None,
                                                      lw=0)
            shock_background = mpatches.Rectangle((shock_xlim_min_raw, -1 * self.amplitude_data[:, site_index].max()),
                                                  (shock_xlim_max_raw - shock_xlim_min_raw),
                                                  2 * self.amplitude_data[:, site_index].max() + extend_height,
                                                  alpha=0.15,
                                                  facecolor=colour_schemes[0]["wavepacket5"], edgecolor=None, lw=0)
            equil_background = mpatches.Rectangle((equil_xlim_min_raw, -1 * self.amplitude_data[:, site_index].max()),
                                                  (equil_xlim_max_raw - equil_xlim_min_raw),
                                                  2 * self.amplitude_data[:, site_index].max() + extend_height,
                                                  alpha=0.3,
                                                  facecolor=colour_schemes[0]["wavepacket6"], edgecolor=None, lw=0)

            ax1.add_patch(precursor_background)
            ax1.add_patch(shock_background)
            ax1.add_patch(equil_background)

        if publication_details:
            # Add scientific notation (annotated) above y-axis
            # ax1.text(-0.03, 1.02, r'$\times \mathcal{10}^{{\mathcal{' + str(int(ax1_yaxis_exponent)) + r'}}}$',
            #          va='center', ha='center', transform=ax1.transAxes, fontsize=self._fontsizes["smaller"])

            # Add figure reference lettering
            ax1.text(0.95, 0.9, f"{select_plot_scheme['ax1_label']}", va='center', ha='center',
                     fontsize=self._fontsizes["smaller"],
                     transform=ax1.transAxes)
            ax2.text(0.05, 0.9, f"{select_plot_scheme['ax2_label']}", va='center', ha='center',
                     fontsize=self._fontsizes["smaller"],
                     transform=ax2.transAxes)

        for ax in self._fig.axes:
            ax.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=1.9999)
            ax.set_axisbelow(False)  # Must be last manipulation of subplots

        if interactive_plot:
            # For interactive plots
            def mouse_event(event: Any):
                print(f'x: {event.xdata} and y: {event.ydata}')

            self._fig.canvas.mpl_connect('button_press_event', mouse_event)
            plt.show()

        self._fig.savefig(f"{self.output_filepath}_site{site_index}.png", bbox_inches="tight")

    def plot_heaviside_and_dispersions(self, dispersion_relations: bool = True, use_dual_signal_inset: bool = False,
                                       show_group_velocity_cases: bool = False, dispersion_inset: bool = False,
                                       use_demag: bool = False, compare_dis: bool = False,
                                       publication_details: bool = False, interactive_plot: bool = False) -> None:
        """
        Two pane figure where upper pane shows the FFT of Quasi-Heaviside Step Function(s), and the lower pane
        shows dispersion relations of our datasets.

        Filler text. TODO
        
        :param compare_dis: 
        :param use_demag: 
        :param dispersion_inset: Show inset in lower pane which compared Dk^2 dispersion relations
        :param dispersion_relations: Plot lower pane of fig if true
        :param use_dual_signal_inset: Show signals for quasi-Heaviside in separate insets
        :param show_group_velocity_cases: Annotate to show information on type of dispersion (normal/anomalous)
        :param publication_details: Add figure reference lettering
        :param interactive_plot: If `True` mouse-clicks to print x/y coords to terminal, else saves image.

        :return: Saves a .png image to the designated output folder.
        """

        if self._fig is None:
            self._fig = plt.figure(figsize=(4.5, 3.375))
        self._fig.subplots_adjust(wspace=1, hspace=0.35)

        num_rows, num_cols = 2, 3

        if dispersion_relations:
            ax1 = plt.subplot2grid((num_rows, num_cols), (0, 0), rowspan=int(num_rows / 2),
                                   colspan=num_cols, fig=self._fig)
        else:
            ax1 = plt.subplot2grid((num_rows, num_cols), (0, 0), rowspan=int(num_rows),
                                   colspan=num_cols, fig=self._fig)

        SAMPLE_RATE = int(5e2)  # Number of samples per nanosecond
        DURATION = int(40)  # Nanoseconds
        FREQUENCY = int(8)  # GHz

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
        (time_instant, signal_instant,
         sample_rate_instant, num_samples_instant) = generate_sine_wave(FREQUENCY, SAMPLE_RATE, DURATION,
                                                                        0.0, False)
        (time_delay, signal_delay,
         sample_rate_delay, num_samples_delay) = generate_sine_wave(FREQUENCY, SAMPLE_RATE, DURATION,
                                                                    1.0, False)

        time_instant_fft = sp.fftpack.rfftfreq(num_samples_instant, 1 / sample_rate_instant)
        signal_instant_fft = sp.fftpack.rfft(signal_instant)

        time_delay_fft = sp.fftpack.rfftfreq(num_samples_delay, 1 / sample_rate_delay)
        signal_delay_fft = sp.fftpack.rfft(signal_delay)

        ax1.plot(time_delay_fft, np.abs(signal_delay_fft), marker='', lw=2.0, color='#ffb55a',
                 markerfacecolor='black', markeredgecolor='black', label="1", zorder=1.2)
        ax1.plot(time_instant_fft, np.abs(signal_instant_fft), marker='', lw=1.5, ls='--', color='#64bb6a',
                 markerfacecolor='black', markeredgecolor='black', label="0", zorder=1.3)

        ax1.set(xlim=(0.001, 15.999), ylim=(1e0, 1e4), xlabel="Frequency (GHz)", ylabel="Amplitude\n(arb. units)",
                yscale='log')

        ax1.xaxis.labelpad, ax1.yaxis.labelpad = -2.0, 0
        self._tick_setter(ax1, 4, 1, 4, 4, is_fft_plot=True)

        ########################################
        if use_dual_signal_inset and dispersion_relations:
            ax1_inset_delayed = mpl_toolkits.axes_grid1.inset_locator.inset_axes(ax1, width=1.3, height=0.36,
                                                                                 loc="upper right",
                                                                                 bbox_to_anchor=[0.995, 0.805],
                                                                                 bbox_transform=ax1.transAxes)

            ax1_inset_instant = mpl_toolkits.axes_grid1.inset_locator.inset_axes(ax1, width=1.3, height=0.36,
                                                                                 loc="upper right",
                                                                                 bbox_to_anchor=[0.995, 1.185],
                                                                                 bbox_transform=ax1.transAxes)

            ax1_inset_delayed.plot(time_delay, signal_delay, lw=1, color='#ffb55a', zorder=1.2)
            ax1_inset_instant.plot(time_instant, signal_instant, lw=1., ls='-', color='#64bb6a', zorder=1.1)

            for ax in [ax1_inset_delayed, ax1_inset_instant]:
                ax.set(xlim=[0, 2], ylim=[-1, 1])
                ax.tick_params(axis="both", which="both", labelsize=self._fontsizes["mini"],
                               bottom=True, top=True, left=True, right=True, zorder=1.99)

                self._tick_setter(ax, 1.0, 0.5, 1, 0.5, yaxis_multi_loc=True,
                                  is_fft_plot=False, yaxis_num_decimals=1.1, yscale_type='p')

                if ax == ax1_inset_delayed:
                    ax.set_xlabel('Time (ns)', fontsize=self._fontsizes["tiny"])
                    ax.set_ylabel('Amplitude  \n(arb. units)  ', fontsize=self._fontsizes["tiny"], rotation=90,
                                  labelpad=20)
                    ax.yaxis.tick_left()
                    ax.yaxis.set_label_position("left")

                    ax.yaxis.set_label_coords(-.2, 1.15)
                    ax.xaxis.labelpad = -1

                if ax == ax1_inset_instant:
                    ax.tick_params(axis='x', which='both', labelbottom=False)

        elif not use_dual_signal_inset and False:
            if dispersion_relations:
                ax1_inset = mpl_toolkits.axes_grid1.inset_locator.inset_axes(ax1, width=1.3, height=0.72,
                                                                             loc="upper right",
                                                                             bbox_to_anchor=[0.995, 1.175],
                                                                             bbox_transform=ax1.transAxes)

            else:
                ax1_inset = mpl_toolkits.axes_grid1.inset_locator.inset_axes(ax1, width=1.3, height=0.72,
                                                                             loc="upper right",
                                                                             bbox_to_anchor=[0.995, 0.98],
                                                                             bbox_transform=ax1.transAxes)

            ax1_inset.plot(time_instant, signal_instant, lw=0.5, color='#64bb6a', zorder=1.1)
            ax1_inset.plot(time_delay, signal_delay, lw=0.5, ls='-.', color='#ffb55a', zorder=1.2)

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
                              yaxis_num_decimals=1.0, yscale_type='p')
        ########################################
        if dispersion_relations:
            # Key values and computations that are common to both systems
            hz_2_GHz, hz_2_THz = 1e-9, 1e-12
            mu0 = 1.256e-6  # m kg s^-2 A^-2

            # Key values and compute wavenumber plus frequency for Moon
            external_field_moon = 0.1  # exchange_field = [8.125, 32.5]  # [T]
            gyromag_ratio_moon = 28.01e9  # 28.8e9
            lattice_constant_moon = 2e-9  # np.sqrt(5.3e-17 / exchange_field)
            system_len_moon = 8e-6  # metres
            sat_mag_moon = 800e3  # A/m
            exc_stiff_moon = 1.3e-11  # J/m
            demag_mag_moon = sat_mag_moon
            dmi_vals_moon = [0, 1.5e-3, 1.5e-3]  # J/m^2
            p_vals_moon = [0, -1, 1]

            # Key values and computations of values for our system
            external_field, exchange_field = 0.1, 4.16  # 0.1, 132.5  # [T]
            gyromag_ratio = 28.01e9  # 28.8e9
            lattice_constant = 2e-9  # np.sqrt(5.3e-17 / exchange_field)
            system_len = 8e-6  # metres
            dmi_val_const = 1.94
            dmi_vals = [0, -dmi_val_const, dmi_val_const]  # J/m^2

            max_len = round(system_len / lattice_constant)
            num_spins_array = np.arange(-max_len, max_len, 1)
            wave_number_array = (num_spins_array * np.pi) / ((len(num_spins_array) - 1) * lattice_constant)

            freq_array = gyromag_ratio * (2 * exchange_field * (1 - np.cos(wave_number_array * lattice_constant))
                                          + external_field)
            ########################################
            # Plot dispersion relations
            ax2 = plt.subplot2grid((num_rows, num_cols), (int(num_rows / 2), 0),
                                   rowspan=num_rows, colspan=num_cols, fig=self._fig)

            ax2.plot(wave_number_array * hz_2_GHz, freq_array * hz_2_THz, color='red', lw=1., ls='-',
                     label=f'Our System')
            ax2.plot(wave_number_array * hz_2_GHz, gyromag_ratio * (
                    external_field + exchange_field * lattice_constant ** 2 * wave_number_array ** 2) * hz_2_THz,
                     color='red', lw=1., alpha=0.4, ls='--', label=f'Dk2 dataset')

            # These!!
            # ax2.scatter(np.arccos(1 - ((freqs2 / gamma - h_0) / (2 * h_ex))) / a, freqs2 / 1e12,
            #             s=0.5, c='red', label='paper')
            # ax2.plot(k, gamma * (2 * h_ex * (1 - np.cos(k * a)) + h_0) / 1e12, color='red', ls='--', label=f'Kittel')

            ax2.set(xlabel="Wavenumber (nm$^{-1}$)", ylabel='Frequency (THz)', ylim=[0, 15.4])
            self._tick_setter(ax2, 2, 0.5, 3, 2, is_fft_plot=False,
                              xaxis_num_decimals=.1, yaxis_num_decimals=2.1, yscale_type='p')

            ax2.margins(0)
            ax2.xaxis.labelpad = -2

            if compare_dis:
                self._fig.suptitle('Comparison of my derivation with Moon\'s')

                if not use_demag:
                    demag_mag_moon = 0

                ax1.clear()
                for dmi_val in dmi_vals:
                    """ Don't delete yet! Need to check the maths                    
                    freq_array = gyromag_ratio * (4 * (exc_stiff / sat_mag) / lattice_constant**2
                                                 * (1 - np.cos(wave_number_array * lattice_constant))
                                                 + external_field
                                                 + (dmi_val/sat_mag) * wave_number_array)
                    freq_array = gyromag_ratio * (2 * (exc_stiff / sat_mag) * wave_number_array**2
                                                  + external_field
                                                  + (2 * dmi_val/sat_mag) * wave_number_array)
                    ax1.plot(wave_number_array * hz_2_GHz, freq_array_dk2 * hz_2_GHz, lw=1., alpha=0.4, ls='--',
                            label=r'$(Dk^2)$'f'D = {dmi_val}'r'$(mJ/m^2$)')

                    These!!
                    ax2.scatter(np.arccos(1 - ((freqs2 / gamma - h_0) / (2 * h_ex))) / a, freqs2 / 1e12,
                                s=0.5, c='red', label='paper')
                    ax2.plot(k, gamma * (2 * h_ex * (1 - np.cos(k * a)) + h_0) / 1e12, 
                             color='red', ls='--', label=f'Kittel')
                    """
                    freq_array = gyromag_ratio * (2 * exchange_field * lattice_constant ** 2 * wave_number_array ** 2
                                                  + external_field
                                                  + dmi_val * lattice_constant * wave_number_array)

                    ax1.plot(wave_number_array * hz_2_GHz, freq_array * hz_2_GHz,
                             lw=1., ls='-', label=f'D = {dmi_val}')
                    ax1.set(xlabel="Wavevector (nm$^{-1}$)", ylabel='Frequency (GHz)',
                            xlim=[-0.25, 0.25], ylim=[0, 40])
                    self._tick_setter(ax1, 0.1, 0.05, 3, 2, is_fft_plot=False,
                                      xaxis_num_decimals=.1, yaxis_num_decimals=2.0, yscale_type='plain')
                    ax1.margins(0)
                    ax1.xaxis.labelpad = -2
                    ax1.legend(title='Mine\n'r'$(J/m^2$)', title_fontsize=self._fontsizes["smaller"],
                               fontsize=self._fontsizes["tiny"], frameon=True, fancybox=True)

                ax2.clear()
                for p_val, dmi_val in zip(p_vals_moon, dmi_vals_moon):
                    max_len_moon = round(system_len_moon / lattice_constant_moon)
                    num_spins_array_moon = np.arange(-max_len_moon, max_len_moon, 1)
                    wave_number_array_moon = (num_spins_array_moon * np.pi) / (
                            (len(num_spins_array_moon) - 1) * lattice_constant_moon)

                    h0 = external_field_moon / mu0
                    j_star = ((2 * exc_stiff_moon) / (mu0 * sat_mag_moon))
                    h0_plus_jk = h0 + j_star * wave_number_array_moon ** 2
                    d_star = ((2 * dmi_val) / (mu0 * sat_mag_moon))

                    freq_array_moon = gyromag_ratio_moon * mu0 * (np.sqrt(h0_plus_jk * (h0_plus_jk + demag_mag_moon))
                                                                  + p_val * d_star * wave_number_array_moon)

                    ax2.plot(wave_number_array_moon * hz_2_GHz, freq_array_moon * hz_2_GHz,
                             lw=1., ls='-', label=f'D = {p_val * dmi_val}')
                    """ Don't delete yet! Need to check the maths                   
                    ax1.plot(wave_number_array * hz_2_GHz, freq_array_dk2 * hz_2_GHz, lw=1., alpha=0.4, ls='--',
                            label=r'$(Dk^2)$'f'D = {dmi_val}'r'$(mJ/m^2$)')

                    These!!
                    ax2.scatter(np.arccos(1 - ((freqs2 / gamma - h_0) / (2 * h_ex))) / a, freqs2 / 1e12,
                                s=0.5, c='red', label='paper')
                    ax2.plot(k, gamma * (2 * h_ex * (1 - np.cos(k * a)) + h_0) / 1e12, 
                             color='red', ls='--', label=f'Kittel')
                    """

                    ax2.set(xlabel="Wavevector (nm$^{-1}$)", ylabel='Frequency (GHz)', xlim=[-0.15, 0.15], ylim=[0, 20])
                    self._tick_setter(ax2, 0.1, 0.05, 3, 2, is_fft_plot=False,
                                      xaxis_num_decimals=.1, yaxis_num_decimals=2.0, yscale_type='plain')

                    ax2.margins(0)
                    ax2.xaxis.labelpad = -2

                    ax2.legend(title='Theirs\n'r'$(J/m^2$)', title_fontsize=self._fontsizes["smaller"],
                               fontsize=self._fontsizes["tiny"], frameon=True, fancybox=True)

            if show_group_velocity_cases:
                # Change xaxis limit to show in terms of lattice constant
                ax2.text(0.997, -0.13, r"$\mathrm{\dfrac{\pi}{a}}$",
                         verticalalignment='center', horizontalalignment='center', transform=ax2.transAxes,
                         fontsize=self._fontsizes["smaller"])

                ########################################
                # Horizon lines and labels for each region (paper's notation)
                ax2.axhline(y=3.8, xmax=1.0, ls='--', lw=1, color='grey', zorder=0.9)  # xmax=0.31
                ax2.axhline(y=10.5, xmax=1.0, ls='--', lw=1, color='grey', zorder=0.9)  # xmax=0.68

                ax2.text(0.02, 0.88, r"$\mathcal{III}$", verticalalignment='center', horizontalalignment='left',
                         transform=ax2.transAxes, fontsize=self._fontsizes["smaller"])
                ax2.text(0.02, 0.5, r"$\mathcal{II}$", verticalalignment='center', horizontalalignment='left',
                         transform=ax2.transAxes, fontsize=self._fontsizes["smaller"])
                ax2.text(0.02, 0.12, r"$\mathcal{I}$", verticalalignment='center', horizontalalignment='left',
                         transform=ax2.transAxes, fontsize=self._fontsizes["smaller"])

                # Use arrow to show rate of change of group velocity (normal (+ve)/no (nil)/anomalous (-ve) dispersion
                ax2.text(0.91, 0.82, f"Decreasing", verticalalignment='center', horizontalalignment='center',
                         transform=ax2.transAxes, fontsize=self._fontsizes["tiny"])
                ax2.text(0.60, 0.425, f"Constant", verticalalignment='center', horizontalalignment='center',
                         transform=ax2.transAxes, fontsize=self._fontsizes["tiny"])
                ax2.text(0.41, 0.12, f"Increasing", verticalalignment='center', horizontalalignment='center',
                         transform=ax2.transAxes, fontsize=self._fontsizes["tiny"])

                ########################################
                # Annotate group velocity derivative rate of change
                arrow_props_normal_disp = {"arrowstyle": '-|>', "connectionstyle": "arc3,rad=0.075", "color": "black"}
                arrow_props_no_disp = {"arrowstyle": '-|>', "connectionstyle": "arc3,rad=0.0", "color": "black"}
                arrow_props_anom_disp = {"arrowstyle": '-|>', "connectionstyle": "arc3,rad=-0.075", "color": "black"}

                ax2.annotate('', xy=(1.665, 2.961), xytext=(1.147, 1.027), va='center', ha='center',
                             arrowprops=arrow_props_normal_disp, fontsize=self._fontsizes["tiny"],
                             transform=ax2.transAxes)
                ax2.annotate('', xy=(3.058, 9.406), xytext=(2.154, 5.098), va='center', ha='center',
                             arrowprops=arrow_props_no_disp, fontsize=self._fontsizes["tiny"], transform=ax2.transAxes)
                ax2.annotate('', xy=(4.155, 13.213), xytext=(3.553, 11.342), va='center', ha='center',
                             arrowprops=arrow_props_anom_disp, fontsize=self._fontsizes["tiny"],
                             transform=ax2.transAxes)

            if dispersion_inset:
                # Key Parameters
                # j_to_meV = 6.24150934190e21
                D_b = 5.3e-17  # exchange field created by exchange energy of spin wave
                a1, a2 = lattice_constant, np.sqrt(D_b / 132.5)

                # Wave number calculation
                num_spins_array1 = np.arange(0, 5000, 1)
                num_spins_array2 = np.arange(0, 15811, 1)
                wave_number_array1 = (num_spins_array1 * np.pi) / ((len(num_spins_array1) - 1) * a1)
                wave_number_array2 = (num_spins_array2 * np.pi) / ((len(num_spins_array2) - 1) * a2)

                ########################################
                # Construct inset and plot dispersion relation(s) for D*k^2 cases for both datasets
                ax2_inset_disp_rels = mpl_toolkits.axes_grid1.inset_locator.inset_axes(ax2, width=1.25, height=0.5,
                                                                                       loc="lower right",
                                                                                       bbox_to_anchor=[0.9875, 0.02],
                                                                                       bbox_transform=ax2.transAxes)

                ax2_inset_disp_rels.plot(wave_number_array1 * hz_2_GHz,
                                         (D_b * 2 * gyromag_ratio) * wave_number_array1 ** 2 * hz_2_THz, lw=1.5,
                                         ls='--', color='purple', label='$a=0.2$ nm', zorder=1.3)
                ax2_inset_disp_rels.plot(wave_number_array2 * hz_2_GHz,
                                         (D_b * 2 * gyromag_ratio) * wave_number_array2 ** 2 * hz_2_THz, lw=1.5, ls='-',
                                         label='$a=0.63$ nm', zorder=1.2)

                ########################################
                # Set figure formatting
                if not show_group_velocity_cases:
                    ax2_inset_disp_rels.set_xlabel('Wavenumber (nm$^{-1}$)', fontsize=self._fontsizes["tiny"])
                    ax2_inset_disp_rels.set_ylabel('Freq (THz)', fontsize=self._fontsizes["tiny"], rotation=90,
                                                   labelpad=20)

                ax2_inset_disp_rels.set(xlim=[0, 2], ylim=[0, 10])
                ax2_inset_disp_rels.xaxis.tick_top()
                ax2_inset_disp_rels.xaxis.set_label_position("top")
                ax2_inset_disp_rels.yaxis.set_label_position("left")
                ax2_inset_disp_rels.tick_params(axis='both', labelsize=self._fontsizes["tiny"])
                ax2.margins(0)

                ax2_inset_disp_rels.patch.set_color("#f9f2e9")
                ax2_inset_disp_rels.xaxis.labelpad, ax2_inset_disp_rels.yaxis.labelpad = 2.5, 5

                self._tick_setter(ax2_inset_disp_rels, 1, 0.5, 3, 3,
                                  xaxis_num_decimals=0, yaxis_num_decimals=1.0,
                                  is_fft_plot=False, yscale_type='p')

                ax2_inset_disp_rels.legend(fontsize=self._fontsizes["mini"], frameon=False)

            ########################################
            if publication_details:
                ax1.text(0.025, 0.88, f"(a)", verticalalignment='center', horizontalalignment='left',
                         transform=ax1.transAxes, fontsize=self._fontsizes["smaller"])

                ax2.text(0.975, 0.12, f"(b)", verticalalignment='center', horizontalalignment='right',
                         transform=ax2.transAxes, fontsize=self._fontsizes["smaller"])

        for ax in self._fig.axes:
            ax.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=1.99)
            ax.set_facecolor("white")
            ax.set_axisbelow(False)

        if interactive_plot:
            # For interactive plots
            def mouse_event(event: Any):
                if event.xdata is not None and event.ydata is not None:
                    print(f'x: {event.xdata:f} and y: {event.ydata:f}')

            self._fig.canvas.mpl_connect('button_press_event', mouse_event)

            cursor = mplcursors.cursor(hover=True)
            cursor.connect("add", lambda sel: sel.annotation.set_text(
                f'x={sel.target[0]:.4f}, y={sel.target[1]:.4f}'))

            # Hide the arrow in the callback
            @cursor.connect("add")
            def on_add(sel):
                sel.annotation.get_bbox_patch().set(fc="white")  # Change background color
                sel.annotation.arrow_patch.set_alpha(0)  # Make the arrow invisible

            self._fig.tight_layout()  # has to be here
            plt.show()

        else:
            self._fig.savefig(f"{self.output_filepath}_dispersion_tv2.png", bbox_inches="tight")

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
            ax.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=6)

        # Add zoomed in region if needed.
        if add_zoomed_region:
            # Select datasets to use
            x = self.time_data[:]
            y = self.amplitude_data[:, spin_site]

            # Impose inset onto plot. Treat as a separate subplot. Use 0.24 for LHS and 0.8 for RHS. 0.7 for TR and
            ax2_inset = mpl_toolkits.axes_grid1.inset_locator.inset_axes(ax2, width=2.4, height=0.625, loc="lower left",
                                                                         bbox_to_anchor=[0.14, 0.625],
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
            # mark_inset(self._axes, ax2_inset,loc1=1, loc2=2, facecolor="red", edgecolor=None, alpha=0.3)

            # Add box to indicate the region which is being zoomed into on the main figure
            ax2.indicate_inset_zoom(ax2_inset, facecolor='#f9f2e9', edgecolor='black', alpha=1.0, lw=0.75, zorder=1)
            arrow_inset_props = {"arrowstyle": '-|>', "connectionstyle": "angle3,angleA=0,angleB=90", "color": "black"}
            ax2_inset.annotate('P1', xy=(2.228, -1.5e-4), xytext=(1.954, -1.5e-4), va='center', ha='center',
                               arrowprops=arrow_inset_props, fontsize=6)
            ax2_inset.annotate('P2', xy=(1.8, -8.48e-5), xytext=(1.407, -1.5e-4), va='center', ha='center',
                               arrowprops=arrow_inset_props, fontsize=6)
            ax2_inset.annotate('P3', xy=(1.65, 6e-5), xytext=(1.407, 1.5e-4), va='center', ha='center',
                               arrowprops=arrow_inset_props, fontsize=6)

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
        ax.tick_params(top="on", right="on", which="both")

        ax.set_ylabel("Amplitude (arb. units)\n", x=-10, y=1)

        ax.text(0.04, 0.88, f"(b)",
                verticalalignment='center', horizontalalignment='left', transform=ax.transAxes, fontsize=8)
        ax2.text(0.04, 0.88, f"(a)",
                 verticalalignment='center', horizontalalignment='left', transform=ax2.transAxes, fontsize=8)

        ax2_inset = mpl_toolkits.axes_grid1.inset_locator.inset_axes(ax, width=1.8, height=0.7, loc="upper right",
                                                                     bbox_to_anchor=[0.88, 0.47],
                                                                     bbox_transform=ax.figure.transFigure)
        ax2_inset.plot(x1, y1, lw=0.5, color='#ffb55a', zorder=1.2)
        ax2_inset.plot(x2, y2, lw=0.5, ls='--', color='#64bb6a', zorder=1.1)
        ax2_inset.yaxis.tick_right()
        ax2_inset.set_xlabel('Time (ns)', fontsize=8)
        ax2_inset.set(xlim=[0, 2])
        ax2_inset.yaxis.set_label_position("right")
        ax2_inset.set_ylabel('Amplitude (arb. units)', fontsize=8, rotation=-90, labelpad=10)
        ax2_inset.tick_params(axis='both', labelsize=8)

        ax2_inset.patch.set_color("#f9f2e9")

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

    def _fft_data(self, input_data, spatial_spacing: bool = None, fft_window: str = False):
        """
        Computes the discrete Fourier transform (DFT) of a given 1-D signal using FFT algorithms.

        Regarding the returned `dft_samples`, if the `input_data` contains:
            - spatial information then dft_samples will represent 'wavevectors', and be in units of inverse-length.
            - temporal information then dft_samples will represent 'frequencies', and be in units of Hz.

        Regarding the returned `discrete_fourier_transform`, it will contain the magnitudes of the DFT results. It will
        be the absolute value of the DFT results; caution if the phase information is needed for subsequent analysis.

        :param input_data: Magnitudes of a magnetic moment components for a given axis.
        :param spatial_spacing: The lattice constant for the system for spatial data.
        :param fft_window: Apply a window to the data before taking the FFT; default is a Hann window.

        :return: A tuple containing the sample frequencies [0] and DFT results [1].
        """

        # Pad the data to greatly improve efficiency of the FFT computation
        n_total = input_data.size
        n_total_padded = sp.fft.next_fast_len(n_total)

        # Find the bin size
        if spatial_spacing is None:
            # For temporal: divide the simulated time into equally sized segments based upon the number of data-points.
            sample_spacing = (self.sim_time_max() / (self.num_dp_per_site() - 1))
        else:
            # For spatial: division of lattice is already known with the lattice constant
            sample_spacing = spatial_spacing

        # Data for DFT (create copy to avoid modifying original data)
        data_to_process = input_data

        # Select a window for the FFT.
        if fft_window is not None:
            if isinstance(fft_window, str):
                # For valid windows, see https://docs.scipy.org/doc/scipy/reference/signal.windows.html
                window_func = getattr(sp.signal.windows, fft_window, "hann")
                window = window_func(n_total)
            elif callable(fft_window):
                # Incase the user wants a custom window function
                window = sp.signal.fft_window(n_total)
            else:
                # Default case
                window = sp.signal.windows.hann(n_total)

            # *= doesn't work here for some reason without breaking subplots
            data_to_process = data_to_process * window

        # Perform the FFT
        discrete_fourier_transform = sp.fft.fft(data_to_process, n_total_padded)

        # Samples from the DFT.
        dft_samples = sp.fft.fftfreq(n_total_padded, sample_spacing)

        # Always skip the DC component at y[0] as I don't need the signal's mean value
        if n_total_padded % 2 == 0:
            # For N even, the elements `y = [1, N / 2 -1]` contain the positive-frequency terms with the final
            # element `y = N / 2` containing the Nyquist frequency
            positive_wavevector_indices = slice(1, n_total_padded // 2)
        else:
            # For N odd, the elements `y = [1, (N - 1) / 2]` contain the positive-frequency terms
            positive_wavevector_indices = slice(1, (n_total_padded + 1) // 2)

        # Only want positive wavevectors and their corresponding fourier transform. Must take absolute value of DFT.
        discrete_fourier_transform = np.abs(discrete_fourier_transform[positive_wavevector_indices])
        dft_samples = dft_samples[positive_wavevector_indices]

        return dft_samples, discrete_fourier_transform


    def _choose_scaling(self, value=None, subplot_to_scale=None, row_index=None, presets=None):
        if value is None and subplot_to_scale is None:
            exit(1)

        if presets is None:
            presets = {
                'nano': [1e-9, r'$\mathrm{nm}$'],
                'micro': [1e-6, r'$\mathrm{{\mu} m}$'],
                'milli': [1e-3, r'$\mathrm{mm}$']
                # Add as needed
            }

        if subplot_to_scale is not None:
            value = subplot_to_scale.get_ylim()[1]
            magnitude_value = int(np.floor(np.log10(value)))
            # Convert uppermost y-tick label to a float, and compared against ylim (upper). If the uppermost tick is
            # greater than ylim (upper) it means an automatic scientific notation conversion (10e-2 -> 1e01)
            # occurred and needs to be undone.
            if float(subplot_to_scale.get_yticklabels()[-2].get_text()) * 10 ** magnitude_value > value:
                magnitude_value -= 1
        else:
            magnitude_value = int(np.floor(np.log10(value)))

        closest_preset_name, (closest_preset_value, closest_preset_tag) = min(presets.items(),
                                                                              key=lambda x: abs(
                                                                                  magnitude_value - np.log10(x[1][0])))

        # Generate labels and values for closest preset and raw value
        closest_preset_exp = int(np.log10(closest_preset_value))
        # This gives us the order, so the +1 is required so we can plot across all values in this order
        # e.g. if value_exp = -3 (i.e. 1e-3 order)

        closest_preset_labels = [
            r'$\times \mathcal{10}^{' + f'{closest_preset_exp}' + '}$',
            r'$\mathcal{10}^{' + f'{closest_preset_exp}' + '}$',
            closest_preset_tag
        ]

        value_labels = [
            r'$\times \mathcal{10}^{' + f'{magnitude_value}' + '}$',
            r'$\mathcal{10}^{' + f'{magnitude_value}' + '}$'
        ]

        return closest_preset_labels, value_labels, closest_preset_value

    def _tick_setter(self, ax, x_major, x_minor, y_major, y_minor, yaxis_multi_loc=False, is_fft_plot=False,
                     xaxis_num_decimals=.1, yaxis_num_decimals=.1, yscale_type='sci', format_xaxis=False,
                     show_sci_notation=False, xaxis_rescale=None):

        if ax is None:
            ax = plt.gca()

        if is_fft_plot:
            ax.xaxis.set(major_locator=ticker.MultipleLocator(x_major),
                         major_formatter=ticker.FormatStrFormatter(f"%{xaxis_num_decimals}f"),
                         minor_locator=ticker.MultipleLocator(x_minor))
            ax.yaxis.set(major_locator=ticker.LogLocator(base=10, numticks=y_major))
            log_minor_locator = ticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=y_minor)
            ax.yaxis.set_minor_locator(log_minor_locator)
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())

            # ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        else:
            if xaxis_rescale is not None:
                x_major *= xaxis_rescale
                x_minor *= xaxis_rescale
                x_scaled_labels, x_major_labels, x_major_scaled = self._choose_scaling(value=x_major)
                rescaled_dif = xaxis_rescale / x_major_scaled

                shift = -self.num_sites_total()/2
                # Container for all new xdata across lines for determining xlims
                all_new_xdata = np.array([])

                # Iterate through all line objects in the axes
                for line in ax.get_lines():
                    xdata, ydata = line.get_data()
                    # Apply shift and scale transformation
                    new_xdata = (xdata + shift) * rescaled_dif
                    line.set_data(new_xdata, ydata)
                    all_new_xdata = np.concatenate((all_new_xdata, new_xdata))

                # Ensure all_new_xdata is sorted for xlim calculation
                all_new_xdata_sorted = np.sort(all_new_xdata)

                # Set xlims to exclude the first and last value
                ax.set_xlim([all_new_xdata_sorted[0], all_new_xdata_sorted[-1] + rescaled_dif])

                ax.set(xlabel=r'Position, $d$ (' + x_scaled_labels[2] + ')')
                x_major /= x_major_scaled
                x_minor /= x_major_scaled

            ax.xaxis.set(major_locator=ticker.MultipleLocator(x_major),
                         major_formatter=ticker.FormatStrFormatter(f"%{xaxis_num_decimals}f"),
                         minor_locator=ticker.MultipleLocator(x_minor))
            ax.yaxis.set(major_locator=ticker.MaxNLocator(nbins=y_major, prune='lower'),
                         major_formatter=ticker.FormatStrFormatter(f"%{yaxis_num_decimals}f"),
                         minor_locator=ticker.AutoMinorLocator(y_minor))

            if yaxis_multi_loc:
                ax.yaxis.set(major_locator=ticker.MultipleLocator(y_major),
                             major_formatter=ticker.FormatStrFormatter(f"%{yaxis_num_decimals}f"),
                             minor_locator=ticker.MultipleLocator(y_minor))

            class ScalarFormatterClass(ticker.ScalarFormatter):
                def _set_format(self):
                    self.format = f"%{yaxis_num_decimals}f"

            yScalarFormatter = ScalarFormatterClass(useMathText=True)
            yScalarFormatter.set_powerlimits((0, 0))
            if format_xaxis:
                ax.xaxis.set_major_formatter(yScalarFormatter)
            ax.yaxis.set_major_formatter(yScalarFormatter)

            if yscale_type == 'sci':
                ax.ticklabel_format(axis='y', style='sci')
            elif yscale_type == 'plain':
                ax.ticklabel_format(axis='y', style='plain')

            # ax.yaxis.labelpad = -3
            if show_sci_notation:
                ax.yaxis.get_offset_text().set_visible(True)
                ax.yaxis.get_offset_text().set_fontsize(8)
                t = ax.yaxis.get_offset_text()
                t.set_x(-0.045)
            else:
                ax.yaxis.get_offset_text().set_visible(False)

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
                #         transform=ax.transAxes, fontsize=self._fontsizes["smaller"])

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
                    #         fontsize=self._fontsizes["smaller"])

                    self._tick_setter(ax, 0.1, 0.05, 3, 2,
                                      xaxis_num_decimals=1, yaxis_num_decimals=0, yscale_type='')

                    # ax.text(-0.02, 1.05, r'$\times \mathcal{10}^{{\mathcal{' + str(int(-3)) + r'}}}$',
                    #        va='center',
                    #        ha='center', transform=ax.transAxes, fontsize=self._fontsizes["smaller"])

                if val == 6:
                    leg_loc = "upper left"
                    leg_pos = (0.02, 0.95)

                    ax.set(xlabel=r"Pumping Field (1)", ylabel="Avg. Velocity (10$^{-3}$)",
                           xlim=[0.1425, 0.3075], ylim=[0.00185, 0.0081])

                    self._tick_setter(ax, 0.1, 0.025, 4, 2,
                                      xaxis_num_decimals=1, yaxis_num_decimals=0, yscale_type='')
                    # ax.text(0.925, 0.1, '(c)', va='center', ha='center', transform=ax.transAxes,
                    #         fontsize=self._fontsizes["smaller"])

                    # ax.text(-0.02, 1.05, r'$\times \mathcal{10}^{{\mathcal{' + str(int(-3)) + r'}}}$',
                    #        va='center',
                    #        ha='right', transform=ax.transAxes, fontsize=self._fontsizes["smaller"])

                ax.plot(xaxis_data, yaxis_data2, ls='-', lw=ax_lw, color=colour2, alpha=0.5,
                        zorder=1.01)
                ax.scatter(xaxis_data, yaxis_data2, color=colour2, marker='o', s=ax_s, fc=colour2, ec='None',
                           label=label2, zorder=1.01)

                ax.legend(ncol=1, loc=leg_loc, handletextpad=-0.25,
                          frameon=False, fancybox=False, facecolor='None', edgecolor='black',
                          fontsize=self._fontsizes["tiny"], bbox_to_anchor=leg_pos,
                          bbox_transform=ax.transAxes).set_zorder(4)

            for axis in [ax]:
                # ax.set_facecolor('#f4f4f5')
                ax.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=1.9999)

                axis.set_axisbelow(False)
                axis.set_facecolor('white')

            if interactive_plot:
                # For interactive plots
                def mouse_event(event: Any):
                    print(f'x: {event.xdata} and y: {event.ydata}')

                self._fig.canvas.mpl_connect('button_press_event', mouse_event)
                self._fig.tight_layout()  # has to be here
                plt.show()
            else:
                fig.savefig(f"{self.output_filepath}_data{val}.png", bbox_inches="tight")

    def find_degenerate_modes(self, use_demag: bool = False, publication_details: bool = False,
                              find_modes: bool = False, interactive_plot: bool = False) -> None:
        """
        Two pane figure where upper pane shows the FFT of Quasi-Heaviside Step Function(s), and the lower pane
        shows dispersion relations of our datasets.

        Filler text. TODO

        :param find_modes:
        :param use_demag:
        :param publication_details: Add figure reference lettering
        :param interactive_plot: If `True` mouse-clicks to print x/y coords to terminal, else saves image.

        :return: Saves a .png image to the designated output folder.
        """
        if self._fig is None:
            self._fig = plt.figure(figsize=(8, 6))  # (figsize=(4.5, 3.375))
        self._fig.subplots_adjust(wspace=1, hspace=0.35)

        num_rows, num_cols = 1, 3

        def round_to_sig_figs(x, sig_figs):
            if x == 0:
                return 0
            else:
                return round(x, sig_figs - int(math.floor(math.log10(abs(x)))) - 1)

        ax1 = plt.subplot2grid((num_rows, num_cols), (0, 0), rowspan=int(num_rows / 1),
                               colspan=num_cols, fig=self._fig)
        # ax2 = plt.subplot2grid((num_rows, num_cols), (0, 0),
        #                       rowspan=num_rows, colspan=num_cols, fig=self._fig)
        ########################################
        # Key values and computations that are common to both systems
        hz_2_GHz, hz_2_THz, m_2_nm = 1e-9, 1e-12, 1e9
        mu0 = 1.25663706212e-6  # m kg s^-2 A^-2

        # Key values and compute wavenumber plus frequency for Moon
        external_field_moon = 0.2  # exchange_field = [8.125, 32.5]  # [T]
        gyromag_ratio_moon = 29.2e9  # 28.8e9
        lattice_constant_moon = 1e-9  # 1e-9 np.sqrt(5.3e-17 / exchange_field)
        system_len_moon = 9.2e-6  # metres 4e-6
        sat_mag_moon = 800e3  # A/m
        exc_stiff_moon = 1.6 * 1.3e-11  # J/m
        demag_mag_moon = sat_mag_moon
        dmi_val_const_moon = 0.4e-3  # 1.0e-3
        dmi_vals_moon = [0, dmi_val_const_moon, dmi_val_const_moon]  # J/m^2
        p_vals_moon = [0, -1, 1]

        # Key values and computations of values for our system
        external_field = external_field_moon
        exchange_field = (2 * exc_stiff_moon) / (sat_mag_moon * lattice_constant_moon ** 2)
        gyromag_ratio = gyromag_ratio_moon
        lattice_constant = lattice_constant_moon  # np.sqrt(5.3e-17 / exchange_field)
        system_len = system_len_moon  # metres
        dmi_val_const = (2 * dmi_val_const_moon) / (sat_mag_moon * lattice_constant_moon)  # 1.9416259130841862  # 2.5
        dmi_vals = [0, -dmi_val_const, dmi_val_const]  # J/m^2

        dec.getcontext().prec = 30

        ########################################

        if not use_demag:
            demag_mag_moon = 0

        if find_modes:

            # System length setup
            max_len = round(system_len / lattice_constant)
            half_max_length = int(max_len / 2)
            n_lower, n_upper = 0, half_max_length
            num_spins_array = np.arange(-int(max_len / 2), int(max_len / 2) + 1, 1)

            # Output controls
            should_print_only_matches = True
            should_print_only_half_ints = False
            should_highlight_all_matches = True
            should_highlight_half_ints = True

            # TODO. Add filter for for closest half-int match
            filter_for_closest_matching_wavelength = False
            filter_for_closest_matching_frequency = True

            print_cutoff_freq = [12, 20]  # must be in GHz

            use_original_wavenumbers = True
            use_relative_atol = True

            if should_print_only_half_ints:
                should_print_only_matches = False

            # Precision of output
            wv_rnd = 5
            fq_rnd = 4

            # Error tolerances
            wv_tol = 10 * 10 ** -wv_rnd
            freq_atol = 10 * 10 ** -fq_rnd
            freq_rtol = 5e-3
            half_int_atol = 10e-2

            # Calculate all wavevectors in system
            wavevector_array = (2 * num_spins_array * np.pi) / system_len

            # Calculate all frequencies in system assuming that there is no demagnetisation
            freq_array = gyromag_ratio * (
                    round_to_sig_figs(exchange_field * lattice_constant ** 2, 3) * wavevector_array ** 2
                    + external_field
                    + (dmi_val_const * lattice_constant * wavevector_array))

            # Convert frequencies to [GHz] with rounding
            freq_array = abs(np.round(freq_array * hz_2_GHz, fq_rnd))

            # Find minima (frequency). Only need to check all frequencies that are greater than this ONCE
            min_freq_index = int(np.where(np.isclose(freq_array, min(freq_array), atol=0))[0])

            print(
                f"Lattice constant [nm]: {lattice_constant * 1e9} | DMI constant [T]: +/- {dmi_val_const} | "
                f"Exchange field [T]: {exchange_field} | External field [T]: {external_field} | "
                f"Gyromagnetic ratio [GHz/T]: {gyromag_ratio * 1e-9} | System length [um]: {system_len * 1e6}")
            print(
                f'wv_rnd: {wv_rnd} | fq_rnd: {fq_rnd} | wv_tol: {wv_tol} | freq_atol: {freq_atol} | '
                f'half_int_atol: {half_int_atol}')

            # Calculate all wavelengths
            wavelengths_array = np.zeros_like(wavevector_array, dtype=float)

            # Set wavelength to infinity where wave number is zero
            zero_wave_indices = wavevector_array == 0
            wavelengths_array[zero_wave_indices] = np.inf

            # Perform division where wave number is non-zero and convert to [nm]
            non_zero_wave_indices = ~zero_wave_indices
            wavelengths_array[non_zero_wave_indices] = ((2 * np.pi) / wavevector_array[
                non_zero_wave_indices])

            # Convert to nm
            wavelengths_array = np.round(wavelengths_array * m_2_nm, 3)  # Sometimes might want abs() so that all wavelengths are +ve (for readability)

            # Convert wave numbers to [1/nm] with rounding
            wavevector_array = np.round(wavevector_array * 1e-9, wv_rnd)

            if min_freq_index >= half_max_length:
                wavevectors_from_n = wavevector_array[:min_freq_index+1]
            else:
                wavevectors_from_n = wavevector_array[min_freq_index:]

            # Initialize containers
            matches_container = []

            def is_wavelength_half(n1, n2, atol=1e-2, convert_to_wavelength=False):
                results = []
                scaling_factors = []

                if isinstance(n1, list):
                    temp = n1
                    n1 = n2
                    n2 = temp

                if n1 == 0 or np.isinf(n1):
                    return results, scaling_factors
                else:
                    if convert_to_wavelength:
                        # Convert from wavenumber [m] to wavelength [nm]
                        wavelength1 = abs(((2 * np.pi) / n1) * m_2_nm)
                    else:
                        wavelength1 = abs(n1)

                # Determine larger and smaller numbers
                for i in n2:
                    if i == 0 or np.isinf(i):
                        results.append(None)
                        continue
                    else:
                        if convert_to_wavelength:
                            wavelength2 = abs((2 * np.pi) / i)
                        else:
                            wavelength2 = abs(i)
                    larger, smaller = max(wavelength1, wavelength2), min(wavelength1, wavelength2)

                    sf_rnd = 3
                    sf_base = np.round((larger / smaller % 1), sf_rnd)

                    # Calculate scaling
                    if sf_base == 0.0:
                        sf_div = np.round((larger / smaller), sf_rnd)
                        if sf_div == 1.0:
                            scaling_factors.append(0.0)
                        elif sf_div > 1.0:
                            scaling_factors.append(1.0)
                    else:
                        scaling_factors.append(sf_base)

                    # Calculate tolerance range
                    half_smaller = smaller / 2
                    tolerance = half_smaller * atol
                    lower_bound = half_smaller - tolerance
                    upper_bound = half_smaller + tolerance

                    # Check if modulus is within the tolerance range
                    modulus = larger % smaller
                    if lower_bound <= modulus <= upper_bound:
                        results.append(True)
                    else:
                        results.append(False)

                return results, scaling_factors

            def find_matches_with_adaptive_tolerance(frequency_value, frequencies, relative_tolerance):
                # Function to find matching frequencies with adaptive tolerance
                tolerance = abs(frequency_value) * relative_tolerance
                matches = np.flatnonzero(np.abs(frequencies - frequency_value) <= tolerance)
                return matches

            for wavevector_n in wavevectors_from_n:

                # Step 2: Find the index of the closest match in wave_number_array
                closest_match_index = np.where(np.isclose(wavevector_array, wavevector_n, atol=wv_tol))[0]

                # Step 3, 4, 5: For each match, find frequency and check for other occurrences
                for match_index in closest_match_index:
                    match_frequency = freq_array[match_index]

                    if wavevector_array[match_index] == 0 or np.isinf(wavevector_array[match_index]):
                        continue

                    match_wavelength = wavelengths_array[match_index]

                    # Find all other occurrences of this frequency, and then their indices
                    if use_relative_atol:
                        matched_freq_indices = find_matches_with_adaptive_tolerance(match_frequency, freq_array,
                                                                                    freq_rtol)
                    else:
                        matched_freq_indices = np.where(np.isclose(freq_array, match_frequency, atol=freq_atol))[0]

                    other_occurrences_indices = [i for i in matched_freq_indices if i != match_index]
                    other_occurrences_frequencies = freq_array[other_occurrences_indices]

                    # Find the corresponding wavevectors for the other occurrences of the given frequency
                    other_occurrences_wavevectors = wavevector_array[other_occurrences_indices]

                    # Calculate all wavelengths
                    if use_original_wavenumbers:
                        other_occurrences_wavelengths = wavelengths_array[other_occurrences_indices]
                    else:
                        # Set wavelength to infinity where wave number is zero
                        other_occurrences_wavelengths = np.zeros_like(other_occurrences_wavevectors, dtype=float)
                        other_zero_wave_indices = other_occurrences_wavevectors == 0
                        other_occurrences_wavelengths[other_zero_wave_indices] = np.inf

                        # Perform division where wave number is non-zero
                        other_non_zero_wave_indices = ~other_zero_wave_indices
                        other_occurrences_wavelengths[other_non_zero_wave_indices] = ((2 * np.pi) /
                                                                                      other_occurrences_wavevectors[
                                                                                          other_non_zero_wave_indices])
                    # TODO. Add code to prune other results so that every f/k pair (positive) only has one other f/k pair (negative)

                    # Check if we have any matches
                    (other_occurrences_half_ints,
                     other_occurrences_scaling) = is_wavelength_half(match_wavelength, other_occurrences_wavelengths,
                                                                     atol=half_int_atol)

                    if filter_for_closest_matching_wavelength:
                        base_case = [1, 0]
                        base_idx = [-1, -1]

                        for i, val in enumerate(other_occurrences_scaling):
                            if val < base_case[0]:
                                base_case[0] = val
                                base_idx[0] = i
                            elif val > base_case[1]:
                                base_case[1] = val
                                base_idx[1] = i

                        use_idx = None
                        if base_case[1] >= 1.0:
                            use_idx = base_idx[1]
                        elif base_case[0] == 0.0 and base_case[1] == 1.0:
                            print('errrrrr what happened here?')
                        else:
                            use_idx = base_idx[0]

                        other_occurrences_scaling = [other_occurrences_scaling[use_idx]]
                        other_occurrences_half_ints = [not other_occurrences_half_ints[use_idx]]
                        other_occurrences_frequencies = [other_occurrences_frequencies[use_idx]]
                        other_occurrences_wavelengths = [other_occurrences_wavelengths[use_idx]]
                        other_occurrences_wavevectors = [other_occurrences_wavevectors[use_idx]]
                        other_occurrences_indices = [other_occurrences_indices[use_idx]]
                    elif filter_for_closest_matching_frequency:
                        pass

                    # Recording the information
                    matches_container.append({
                        'match_index': match_index,
                        'user_wavevector': wavevector_n,
                        'closest_wavevector': wavevector_n,
                        'match_frequency': match_frequency,
                        'match_wavelength': abs(match_wavelength),

                        'other_occurrences_indices': other_occurrences_indices,
                        'other_occurrences_wavevectors': other_occurrences_wavevectors,
                        'other_occurrences_frequencies': other_occurrences_frequencies,
                        'other_occurrences_wavelengths': [abs(x) for x in other_occurrences_wavelengths],
                        'other_occurrences_half_ints': other_occurrences_half_ints,
                        'other_occurrences_scaling': other_occurrences_scaling
                    })

            # Step 6: Sort the container by frequency and then by wavevector
            matches_container.sort(key=lambda x: (x['match_frequency'], x['closest_wavevector']))

            # for entry in frequency_container:
            #     print((entry['match_frequency'], entry['other_occurrences_frequencies']))
            # exit(0)

            select_colour_scheme = colour_schemes[3]

            line_counter = 0
            for entry in matches_container:
                # Note that I can't run simulations for wavelengths smaller than 1nm so there's no point being
                # more precise than this
                match_index = entry['match_index']
                match_frequency = entry['match_frequency']
                match_wavevector = entry['closest_wavevector']
                match_wavelength = entry['match_wavelength']

                if print_cutoff_freq[0] is not None and match_frequency < print_cutoff_freq[0]:
                    continue

                if print_cutoff_freq[1] is not None and match_frequency > print_cutoff_freq[1]:
                    exit(0)

                if should_print_only_half_ints and not any(entry['other_occurrences_half_ints']):
                    continue

                color = select_colour_scheme['ENDC']
                if entry['other_occurrences_indices']:
                    # Current case has a match. Rarest case first
                    if should_highlight_half_ints and any(entry['other_occurrences_half_ints']):
                        color = select_colour_scheme['PURPLE']

                    elif should_highlight_all_matches:
                        color = select_colour_scheme['BLUE']
                if entry['other_occurrences_indices'] or not should_print_only_matches:
                    # Print the match information
                    print(f"{color}"
                          f"\u03C9/2\u03C0: {match_frequency:.{fq_rnd}f} [GHz] | "
                          f"kn: {match_wavevector:.{wv_rnd}f} [1/nm] | "
                          f"in: {match_index}, "
                          f"λn: {match_wavelength:.{2}f} [nm]"
                          f"\t", end="")

                    # Iterate over other occurrences should the exist
                    if entry['other_occurrences_indices']:
                        for enum_index, (
                                other_index, other_wavevector, other_wavelength, other_freq, other_scaling,
                                other_half_int) in enumerate(
                            zip(entry['other_occurrences_indices'],
                                entry['other_occurrences_wavevectors'],
                                entry['other_occurrences_wavelengths'],
                                entry['other_occurrences_frequencies'],
                                entry['other_occurrences_scaling'],
                                entry['other_occurrences_half_ints'])):

                            if should_highlight_half_ints and other_half_int:
                                if filter_for_closest_matching_wavelength and not (other_scaling < 0.05 or other_scaling > 0.95):
                                    color = select_colour_scheme['BLUE']
                                else:
                                    color = select_colour_scheme['PURPLE']
                            elif should_highlight_all_matches:
                                color = select_colour_scheme['BLUE']

                            print(
                                f"{color}| {other_freq:.{fq_rnd}f} [GHz],  "
                                f"i{enum_index + 1}: {other_index}, "
                                f"\u03BB{enum_index + 1}: {other_wavelength:.{2}f} [nm], "
                                f"k{enum_index + 1}: {other_wavevector:.{wv_rnd}f} [1/nm],"
                                f"\t\u03BE{enum_index + 1}: {other_scaling:.3f}"
                                f" ", end="")

                    print("|", end="\n")
                    line_counter += 1

                # Check if 10 lines have been printed
                if line_counter >= 10:
                    print("------------------------------------------------")
                    line_counter = 0

            print("--------------------------------------------------------------------------------")
            exit(0)

        else:
            print(f"Lattice constant [nm]: {lattice_constant * 1e9} | DMI constant [T]: +/- {dmi_val_const} |"
                  f" Exchange field [T]: {exchange_field} | External field [T]: {external_field} |"
                  f" Gyromagnetic ratio [GHz/T]: {gyromag_ratio * 1e-9} | System length [um]: {system_len * 1e6} |"
                  f" Sites: {round(system_len / lattice_constant)}")

            # Plot dispersion relations
            self._fig.suptitle('Dispersion Relation')
            for dmi_val in dmi_vals:
                max_len = round(system_len / lattice_constant)
                num_spins_array = np.arange(-int(max_len / 2), int(max_len / 2) + 1, 1)
                wave_number_array = (2 * num_spins_array * np.pi) / system_len
                # old: wave_number_array = (2*num_spins_array*np.pi)/ ((len(num_spins_array) - 1) * lattice_constant)

                freq_array = gyromag_ratio * (
                        round_to_sig_figs(exchange_field * lattice_constant ** 2, 3) * wave_number_array ** 2
                        + external_field
                        + (dmi_val * lattice_constant * wave_number_array))
                # + (((2 * dmi_val) / (sat_mag_moon)) * wave_number_array))

                ax1.plot(wave_number_array * hz_2_GHz, freq_array * hz_2_GHz, lw=0., ls='-',
                         label=f'D = {dmi_val}', marker='o', markersize=1.5)

                ax1.set(xlabel="Wavevector (nm$^{-1}$)",
                        ylabel='Frequency (GHz)', xlim=[-0.4, 0.4], ylim=[8, 30])
                self._tick_setter(ax1, 0.1, 0.05, 3, 2, is_fft_plot=False,
                                  xaxis_num_decimals=.1, yaxis_num_decimals=2.0, yscale_type='plain')

                ax1.margins(0)
                ax1.xaxis.labelpad = -2
                ax1.legend(title=f'Mine - D [T]\n(H_ex = {exchange_field:2.3f}[T])',
                           title_fontsize=self._fontsizes["smaller"],
                           fontsize=self._fontsizes["tiny"], frameon=True, fancybox=True)

                # file_name = 'D:/Data/2024-02-14/disp_data1.csv'
            #
            # # Writing to the file
            # with open(file_name, 'w', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #
            #     # Iterate over the arrays and write
            #     for i in range(len(wave_number_array)):
            #         writer.writerow([wave_number_array[i] * hz_2_GHz, freq_array[i] * hz_2_GHz])

            """
            for p_val, dmi_val in zip(p_vals_moon, dmi_vals_moon):
                max_len_moon = round(system_len_moon / lattice_constant_moon)
                num_spins_array_moon = np.arange(-int(max_len_moon / 2), int(max_len_moon / 2) + 1, 1)
                wave_number_array_moon = (2 * num_spins_array_moon * np.pi) / (
                        (len(num_spins_array_moon) - 1) * lattice_constant_moon)

                # Remove all mu0 on denominator due to precision error when included
                h0 = external_field_moon
                j_star = (2 * exc_stiff_moon) / sat_mag_moon
                h0_plus_jk = h0 + j_star * (wave_number_array_moon ** 2)

                d_star = (2 * dmi_val) / sat_mag_moon

                # Removed mu0 multiplying whole expression to be consistent with other removals of mu0
                freq_array_moon = gyromag_ratio_moon * (np.sqrt(h0_plus_jk * (h0_plus_jk + demag_mag_moon))
                                                        + p_val * d_star * wave_number_array_moon)

                ax2.plot(wave_number_array_moon * hz_2_GHz, freq_array_moon * hz_2_GHz, lw=0, ls='-',
                         label=f'D = {p_val * dmi_val:.3e}', marker='o', markersize=1.5)
                ax2.set(xlabel="Wavevector (nm$^{-1}$)",
                        ylabel='Frequency (GHz)', xlim=[-0.5, 0.5], ylim=[0, 60])
                self._tick_setter(ax2, 0.1, 0.05, 3, 2, is_fft_plot=False,
                                  xaxis_num_decimals=.1, yaxis_num_decimals=2.0, yscale_type='plain')

                ax2.margins(0)
                ax2.xaxis.labelpad = -2

                ax2.legend(title='Theirs - D [J/m2]', title_fontsize=self._fontsizes["smaller"],
                           fontsize=self._fontsizes["tiny"], frameon=True, fancybox=True)
            """
        ########################################
        if publication_details:
            ax1.text(0.025, 0.88, f"(a)", verticalalignment='center', horizontalalignment='left',
                     transform=ax1.transAxes, fontsize=self._fontsizes["smaller"])

            # ax2.text(0.975, 0.12, f"(b)", verticalalignment='center', horizontalalignment='right',
            #         transform=ax2.transAxes, fontsize=self._fontsizes["smaller"])

        for ax in self._fig.axes:
            ax.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=1.99)
            ax.set_facecolor("white")
            ax.set_axisbelow(False)

        if interactive_plot:
            # For interactive plots
            def mouse_event(event: Any):
                if event.xdata is not None and event.ydata is not None:
                    print(f'k: {event.xdata:f} (\u03BB: {2 * np.pi / event.xdata:f}) and f: {event.ydata:f}')

            self._fig.canvas.mpl_connect('button_press_event', mouse_event)

            cursor = mplcursors.cursor(hover=True)
            cursor.connect("add", lambda sel: sel.annotation.set_text(f'(k: {sel.target[0]:.6f}, f: {sel.target[1]:.6f})'))

            # Hide the arrow in the callback
            @cursor.connect("add")
            def on_add(sel):
                sel.annotation.get_bbox_patch().set(fc="white")  # Change background color
                sel.annotation.arrow_patch.set_alpha(0)  # Make the arrow invisible

            self._fig.tight_layout()  # has to be here
            self._fig.savefig(f"{self.output_filepath}_dispersion.png", bbox_inches="tight")
            plt.show()
