#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib as mpl
# For interactive plots on Mac
# matplotlib.use('macosx')

# Standard modules (common)
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import imageio as imio
import os as os
import seaborn as sns

# Third party modules (uncommon)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import gif as gif
from scipy.fft import rfft, rfftfreq, fft, fftfreq
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

        # Data and paths read-in from data_analysis.py
        self.time_data = time_data
        self.amplitude_data = amplitude_data
        self.sites_array = array_of_sites
        self.output_filepath = output_filepath

        # Individual attributes from key_data that are needed for the class
        self._nm_method = sim_flags['numericalMethodUsed']
        self._static_field = key_data['staticBiasField']
        self._driving_field1 = key_data['dynamicBiasField1']
        self._driving_field2 = key_data['dynamicBiasField2']
        self._driving_freq = key_data['drivingFreq'] / 1e9  # Converts from [s] to (ns).
        self._driving_region_lhs = key_data['drivingRegionLHS']
        self._driving_region_rhs = key_data['drivingRegionRHS']
        self._driving_width = key_data['drivingRegionWidth']
        self._max_time = key_data['maxSimTime'] * 1e9
        self._stop_iteration_value = key_data['stopIterVal']
        self._exchange_min = key_data['exchangeMinVal']
        self._exchange_max = key_data['exchangeMaxVal']
        self._num_data_points = key_data['numberOfDataPoints']
        self._chain_spins = key_data['chainSpins']
        self._num_damped_spins = key_data['dampedSpins']
        self._total_num_spins = key_data['totalSpins']
        self._stepsize = key_data['stepsize'] * 1e9
        self._gilbert_factor = key_data['gilbertFactor']
        self._gyro_mag_ratio = key_data['gyroMagRatio']

        # Attributes for plots
        self._fig = None
        self._axes = None
        self._yaxis_lim = 1.1  # Add a 10% margin to the y-axis.
        self._fig_kwargs = {"xlabel": f"Site Number [$N_i$]", "ylabel": f"m$_x$",
                            "xlim": [0.25 * self._total_num_spins, 0.75 * self._total_num_spins],
                            "ylim": [-1 * self._yaxis_lim, self._yaxis_lim]}

        # Text sizes for class to override rcParams
        self._fontsizes = {"large": 20, "medium": 14, "small": 11, "smaller": 10, "tiny": 8, "mini": 7}

    def _draw_figure(self, row_index: int = -1, has_single_figure: bool = True, publish_plot: bool = False,
                     draw_regions_of_interest: bool = False, static_ylim=True) -> None:
        """
        Private method to plot the given row of data, and create a single figure.

        Figure attributes are controlled from __init__ to ensure consistentcy.

        :param static_ylim:
        :param row_index: Plot given row from dataset; most commonly plotted should be the default.
        :param has_single_figure: If `False`, change figure dimensions for GIFs
        :param draw_regions_of_interest: Draw coloured boxes onto plot to show driving- and damping-regions.
        :param publish_plot: Flag to add figure number and LaTeX annotations for publication.

        :return: Method updates `self._fig` and `self.axis` within the class.
        """
        if self._fig is None:
            if has_single_figure:
                self._fig = plt.figure(figsize=(4.4, 4.4))
                self._axes = self._fig.add_subplot(111)
                self._yaxis_lim *= max(self.amplitude_data[row_index, :])
            else:
                # For GIFs only. Each frame requires a new fig to prevent stuttering.
                cm = 1 / 2.54
                self._fig = plt.figure(figsize=(4.4, 2.0))
                # self._fig = plt.figure(figsize=(11.12 * cm * 2, 6.15 * cm * 2))  # Sizes are from PRL guidelines
                self._axes = self._fig.add_subplot(111)
                if static_ylim:
                    self._yaxis_lim *= max(self.amplitude_data[-1, :])
                else:
                    self._yaxis_lim *= max(self.amplitude_data[row_index, :])
                plt.rcParams.update({'savefig.dpi': 200, "figure.dpi": 200})  # Prevent excessive image sizes

            # Any method-wide updates
            self._axes.clear()
            self._fig_kwargs["ylim"] = [-1 * self._yaxis_lim, self._yaxis_lim]
        else:
            self._axes.clear()  # Only triggered if existing plot is present
            if static_ylim:
                self._yaxis_lim = 1.1 * max(self.amplitude_data[-1, :])
            else:
                self._yaxis_lim = 1.1 * max(self.amplitude_data[row_index, :])
            self._fig_kwargs["ylim"] = [-1 * self._yaxis_lim, self._yaxis_lim]

        self._axes.set_aspect("auto")

        # Easier to have time-stamp as label than textbox.
        self._axes.plot(np.arange(0, self._total_num_spins), self.amplitude_data[row_index, :], ls='-',
                        lw=2 * 0.75, color='#64bb6a', label=f"{self.time_data[row_index]: 2.2f} (ns)")

        self._axes.set(**self._fig_kwargs)
        # self._axes.legend(fontsize=self._fontsizes["tiny"], frameon=False, fancybox=False,
        #            facecolor=None, edgecolor=None)

        # Keep tick manipulations in this block to ease debugging
        self._axes.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=6)
        self._tick_setter(self._axes, self._total_num_spins / 4, self._total_num_spins / 8, 3, 4,
                          yaxis_num_decimals=1.1, show_sci_notation=True)

        if publish_plot:
            self._axes.text(-0.04, 0.96, r'$\times \mathcal{10}^{{\mathcal{-3}}}$', va='center',
                            ha='center', transform=self._axes.transAxes, fontsize=6)
            self._axes.text(0.88, 0.88, f"(c) {self.time_data[row_index]:2.3f} ns",
                            va='center', ha='center', transform=self._axes.transAxes,
                            fontsize=6)

        if draw_regions_of_interest:
            left, bottom, width, height = (
                [0, (self._total_num_spins - self._num_damped_spins),
                 (self._driving_region_lhs + self._num_damped_spins)],
                self._axes.get_ylim()[0] * 2,
                (self._num_damped_spins, self._driving_width),
                4 * self._axes.get_ylim()[1])

            rectangle_lhs = mpatches.Rectangle((left[0], bottom), width[0], height, lw=0,
                                               alpha=0.5, facecolor="grey", edgecolor=None)

            rectangle_rhs = mpatches.Rectangle((left[1], bottom), width[0], height, lw=0,
                                               alpha=0.5, facecolor="grey", edgecolor=None)

            rectangle_driving_region = mpatches.Rectangle((left[2], bottom), width[1], height, lw=0,
                                                          alpha=0.25, facecolor="grey", edgecolor=None)

            plt.gca().add_patch(rectangle_lhs)
            plt.gca().add_patch(rectangle_rhs)
            plt.gca().add_patch(rectangle_driving_region)

        self._axes.grid(visible=False, axis='both', which='both')
        self._fig.tight_layout()

    def plot_row_spatial(self, row_index: int = -1, should_annotate_parameters: bool = False,
                         interactive_plot=False) -> None:
        """
        Plot a row of data to show spatial evolution.

        A row corresponds to an instant in time, so this can be particularly useful for investigating the final state
        of a system. Also, can be used to show the evolution of the whole system if multiple images are generated.

        :param interactive_plot:
        :param row_index: Plot given row from dataset; most commonly plotted should be the default.
        :param should_annotate_parameters: Add simulation parameters to plot. Useful when presenting work in meetings.

        :return: Saves a .png to the nominated 'Outputs' directory.
        """
        self._draw_figure(row_index)

        self._axes.grid(visible=True, axis='both', which='both')

        if should_annotate_parameters:
            if self._exchange_min == self._exchange_max:
                exchange_string = f"Uniform Exc.: {self._exchange_min} (T)"
            else:
                exchange_string = f"J$_{{min}}$ = {self._exchange_min} (T) | J$_{{max}}$ = " \
                                  f"{self._exchange_max} (T)"

            parameters_textbody = (f"H$_{{0}}$ = {self._static_field} (T) | N = {self._chain_spins} | " + r"$\alpha$" +
                                   f" = {self._gilbert_factor: 2.2e}\nH$_{{D1}}$ = {self._driving_field1: 2.2e} (T) | "
                                   f"H$_{{D2}}$ = {self._driving_field2: 2.2e} (T) \n{exchange_string}")

            parameters_text_props = dict(boxstyle='round', facecolor='gainsboro', alpha=0.5)
            self._axes.text(0.05, 1.2, parameters_textbody, transform=self._axes.transAxes, fontsize=12,
                            ha='center', va='center', bbox=parameters_text_props)

        if interactive_plot:
            # For interactive plots
            def mouse_event(event: Any):
                print(f'x: {event.xdata} and y: {event.ydata}')

            self._fig.canvas.mpl_connect('button_press_event', mouse_event)
            self._fig.tight_layout()  # Must be directly before plt.show()
            plt.show()
        else:
            self._fig.savefig(f"{self.output_filepath}_row{row_index}.png", bbox_inches="tight")

    def _plot_paper_gif(self, row_index: int, has_static_ylim: bool = False) -> plt.Figure:
        """
        Private method to save a given row of a data as a frame suitable for use with the git library.

        Requires decorator so use method as an inner class instead of creating child class.

        :param row_index: The row to be plotted.

        :return: Method indirectly updates `self._fig` and `self.axis` by calling self._draw_figure().
        """
        self._draw_figure(row_index, False, draw_regions_of_interest=False, publish_plot=False,
                          static_ylim=has_static_ylim)

        return self._fig

    def create_gif(self, number_of_frames: float = 0.01,
                   frames_per_second: float = 10, has_static_ylim: bool = False) -> None:
        frame_filenames = []

        for index in range(0, int(self._num_data_points + 1), int(self._num_data_points * number_of_frames)):
            frame = self._plot_paper_gif(index, has_static_ylim=has_static_ylim)
            frame_filename = f"{self.output_filepath}_{index}.png"
            frame.savefig(frame_filename)
            frame_filenames.append(frame_filename)
            plt.close(frame)  # Close the figure to free memory

        with imio.get_writer(f"{self.output_filepath}.gif", mode='I', fps=frames_per_second, loop=0) as writer:
            for filename in frame_filenames:
                image = imio.imread(filename)
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
        :param visualise_wavepackets: Redraw each wavepacket with high-constrast colours on time evolution pane.
        :param annotate_precursors_fft: Add arrows/labels to denote wavepackets (P1 wavepacket closest to shock-edge).
        :param annotate_signal: Draw regions (with labels) onto time evolution showing precursor/shock/equil regions.
        :param wavepacket_inset: On time evolution pane, zoom in on precursor region to show the wavepackets as inset.
        :param add_key_params: Add text box between panes which lists key simulation parameters.
        :param add_signal_backgrounds: Shade behind signal regions to improve clarity; backup to `annotate_signals`. 
        :param publication_details: Add figure reference label and scientific notation to y-axis. Needs edited each run
        :param interactive_plot: If `True` mouseclicks to print x/y coords to terminal, else saves image.

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
        colour_schemes = {
            0: {  # Controls wavepacket visualisation. From https://coolors.co/palettes/popular/3%20colors
                "wavepacket1": "#26547c",
                "wavepacket2": "#ef476f",
                "wavepacket3": "#ffd166",
                "wavepacket4": "#edae49",
                "wavepacket5": "#d1495b",
                "wavepacket6": "#00798c",
            },
            1: {  # Taken from time_variation (depreciated function)
                'ax1_colour_matte': "#37782C",
                'ax2_colour_matte': "#37782C",
                'precursor_colour': "#37782C",
                'shock_colour': "#64bb6a",
                'equil_colour': "#9fd983"
            },
            2: {  # Taken from time_variation1
                'ax1_colour_matte': "#73B741",  # gr #73B741 dg #8C8E8D" # dg "#80BE53"
                'ax2_colour_matte': "#F77D6A",
                'precursor_colour': "#CD331B",
                'shock_colour': "#B896B0",  # cy 3EB8A1
                'equil_colour': "#377582"  # B79549
                # 37782C, #64BB6A, #9FD983
            }
            # ... add more colour schemes as needed
        }

        # Accessing the selected colour scheme
        select_colour_scheme = 2
        is_colour_matte = False
        selected_scheme = colour_schemes[select_colour_scheme]

        ########################################
        # All times in nanosecnds (ns)
        plot_schemes = {  # D:\Data\2023-04-19\Outputs
            0: {  # mceleney2023dipsersive Fig. 1b-c [2022-08-29/T1337_site3000]
                'signal_xlim': (0.0, self._max_time),
                'ax1_xlim': [0.0, 5.0],
                'ax1_ylim': [-4.0625e-3, 3.125e-3],
                'ax1_inset_xlim': [0.7, 2.6],
                'ax1_inset_ylim': [-2e-4, 2e-4],
                'ax1_inset_width': 1.95,
                'ax1_inset_height': 0.775,
                'ax1_inset_bbox': [0.08, 0.975],
                'ax2_xlim': [0.0, 99.9999],
                'ax2_ylim': [1e-1, 1e3],
                'precusor_xlim': (0, 2.6),  # 12:3356
                'signal_onset_xlim': (2.6, 3.79),  # 3445:5079
                'equilib_xlim': (3.8, 5.0),  # 5079::
                'ax1_label': '(b)',
                'ax2_label': '(c)',
                'ax1_line_height': 3.15e-3
            },
            1: {  # Jiahui T0941/T1107_site3
                'signal_xlim': (0.0, self._max_time),
                'ax1_xlim': [0.0, 1.50 - 0.00001],
                'ax1_ylim': [self.amplitude_data[:, site_index].min(), self.amplitude_data[:, site_index].max()],
                'ax1_inset_xlim': [0.01, 0.02],
                'ax1_inset_ylim': [-2e-4, 2e-4],
                'ax1_inset_width': 1.95,
                'ax1_inset_height': 0.775,
                'ax1_inset_bbox': [0.08, 0.975],
                'ax2_xlim': [0.0001, 599.9999],
                'ax2_ylim': [1e-2, 1e1],
                'precusor_xlim': (0.0, 0.99),  # 0.0, 0.75
                'signal_onset_xlim': (0.0, 0.01),  # 0.75, 1.23
                'equilib_xlim': (0.99, 1.5),  # 1.23, 1.5
                'ax1_label': '(a)',
                'ax2_label': '(b)',
                'ax1_line_height': int(self.amplitude_data[:, site_index].min() * 0.9)
            },
            2: {  # Jiahui T0941/T1107_site1
                'signal_xlim': (0.0, self._max_time),
                'ax1_xlim': [0.0, 1.50 - 0.00001],
                'ax1_ylim': [self.amplitude_data[:, site_index].min(), self.amplitude_data[:, site_index].max()],
                'ax1_inset_xlim': [0.01, 0.02],
                'ax1_inset_ylim': [-2e-4, 2e-4],
                'ax1_inset_width': 1.95,
                'ax1_inset_height': 0.775,
                'ax1_inset_bbox': [0.08, 0.975],
                'ax2_xlim': [0.0001, 119.9999],
                'ax2_ylim': [1e-3, 1e1],  # A            B            C            D           E
                'precusor_xlim': (0.0, 0.42),  # (0.00, 0.54) (0.00, 0.42) (0.00, 0.42) (0.00, 0.65) (0.00, 0.42)
                'signal_onset_xlim': (0.42, 0.65),  # (0.00, 0.01) (0.42, 0.54) (0.42, 0.65) (0.65, 1.20) (0.42, 1.20)
                'equilib_xlim': (0.65, 1.5),  # (0.54, 1.50) (0.54, 1.50) (0.65, 1.50) (1.20, 1.50) (1.20, 1.50)
                'ax1_label': '(a)',
                'ax2_label': '(b)',
                'ax1_line_height': int(self.amplitude_data[:, site_index].min() * 0.9)
            }
        }

        select_plot_scheme = plot_schemes[2]
        signal_xlim_min, signal_xlim_max = select_plot_scheme['signal_xlim']
        ax1_xlim_lower, ax1_xlim_upper = select_plot_scheme['ax1_xlim']
        ax1_xlim_range = ax1_xlim_upper - ax1_xlim_lower

        precursors_xlim_min_raw, precursors_xlim_max_raw = select_plot_scheme['precusor_xlim']
        shock_xlim_min_raw, shock_xlim_max_raw = select_plot_scheme['signal_onset_xlim']
        equil_xlim_min_raw, equil_xlim_max_raw = select_plot_scheme['equilib_xlim']

        # TODO Need to find all the values and turn this section into a dictionary
        wavepacket_schemes = {
            0: {  # mceleney2023dipsersive Fig. 1b-c [2022-08-29/T1337_site3000]
                'wp1_xlim': (1.75, precursors_xlim_max_raw),
                'wp2_xlim': (1.3, 1.7),
                'wp3_xlim': (1.05, 1.275)
            },
            1: {  # mceleney2023dipsersive Fig. 3a-b [2022-08-08/T1400_site3000]
                'wp1_xlim': (0.481, 0.502),
                'wp2_xlim': (0.461, 0.480),
                'wp3_xlim': (0.442, 0.4605)
            },
            2: {  # mceleney2023dipsersive Fig. 3a-b [2022-08-08/T1400_site3000]
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
        data_names = ['Precursors', 'Shockwave', 'Steady State', 'wavepacket1', 'wavepacket2', 'wavepacket3']
        wavepacket_labels = ['P1', 'P2', 'P3']  # Maybe make into a dict also having x/y coords contained here
        raw_values = {
            'Precursors': (precursors_xlim_min_raw, precursors_xlim_max_raw),
            'Shockwave': (shock_xlim_min_raw, shock_xlim_max_raw),
            'Steady State': (equil_xlim_min_raw, equil_xlim_max_raw),
            'wavepacket1': (wavepacket1_xlim_min_raw, wavepacket1_xlim_max_raw),
            'wavepacket2': (wavepacket2_xlim_min_raw, wavepacket2_xlim_max_raw),
            'wavepacket3': (wavepacket3_xlim_min_raw, wavepacket3_xlim_max_raw)
        }

        ########################################

        ax1.set(xlabel=f"Time (ns)", ylabel=r"$\mathrm{m_x}$($10^{-4}$)", xlim=select_plot_scheme['ax1_xlim'],
                ylim=select_plot_scheme['ax1_ylim'])

        ax2.set(xlabel=f"Frequency (GHz)", ylabel=f"Amplitude (arb. units)", xlim=select_plot_scheme['ax2_xlim'],
                ylim=select_plot_scheme['ax2_ylim'], yscale='log')

        self._tick_setter(ax1, 0.5, 0.25, 3, 4, xaxis_num_decimals=1,
                          show_sci_notation=False)
        self._tick_setter(ax2, 40, 10, 3, None, is_fft_plot=True)

        ########################################
        if ax1_xlim_lower > ax1_xlim_upper:
            exit(0)

        def convert_norm(val, a=0, b=1):
            # Magic. Don't touch! Normalises precursor region so that both wavepackets and feature can be defined using
            # their own x-axis limits.
            return int(self._num_data_points * ((b - a) * ((val - signal_xlim_min)
                                                           / (signal_xlim_max - signal_xlim_min)) + a))

        converted_values = {name: (convert_norm(raw_values[name][0]), convert_norm(raw_values[name][1])) for name in
                            data_names}

        precursors_xlim_min, precursors_xlim_max = converted_values['Precursors']
        shock_xlim_min, shock_xlim_max = converted_values['Shockwave']
        equil_xlim_min, equil_xlim_max = converted_values['Steady State']
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
        ax1.plot(self.time_data[precursors_xlim_min:precursors_xlim_max],
                 self.amplitude_data[precursors_xlim_min:precursors_xlim_max, site_index],
                 ls='-', lw=0.75, color=f'{precursor_colour}', label=f"{self.sites_array[site_index]}",
                 markerfacecolor='black', markeredgecolor='black', zorder=1.1)
        ax1.plot(self.time_data[shock_xlim_min:shock_xlim_max],
                 self.amplitude_data[shock_xlim_min:shock_xlim_max, site_index],
                 ls='-', lw=0.75, color=f'{shock_colour}', label=f"{self.sites_array[site_index]}",
                 markerfacecolor='black', markeredgecolor='black', zorder=1.1)
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

        frequencies_precursors, fourier_transform_precursors = (
            self._fft_data(self.amplitude_data[precursors_xlim_min:precursors_xlim_max, site_index]))
        frequencies_dsw, fourier_transform_dsw = (
            self._fft_data(self.amplitude_data[shock_xlim_min:shock_xlim_max, site_index]))
        frequencies_eq, fourier_transform_eq = (
            self._fft_data(self.amplitude_data[equil_xlim_min:convert_norm(signal_xlim_max), site_index]))

        for i, j, k in zip(abs(fourier_transform_precursors), abs(fourier_transform_dsw),
                           abs(fourier_transform_eq)):
            if i < 1:
                print(f'Small value PRE found: {i}')
            if j < 1:
                print(f'Small value DSW found: {j}')
            if k < 1:
                print(f'Small value EQ found: {k}')
        ax2.plot(frequencies_precursors, abs(fourier_transform_precursors),
                 lw=1, color=f"{precursor_colour}", marker='', markerfacecolor='black', markeredgecolor='black',
                 label=data_names[0], zorder=1.5)
        ax2.plot(frequencies_dsw, abs(fourier_transform_dsw),
                 lw=1, color=f'{shock_colour}', marker='', markerfacecolor='black', markeredgecolor='black',
                 label=data_names[1], zorder=1.2)
        ax2.plot(frequencies_eq, abs(fourier_transform_eq),
                 lw=1, color=f'{equil_colour}', marker='', markerfacecolor='black', markeredgecolor='black',
                 label=data_names[2], zorder=1.1)

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
            ax1_inset = inset_axes(ax1, width=select_plot_scheme['ax1_inset_width'],
                                   height=select_plot_scheme['ax1_inset_height'], loc="lower left",
                                   bbox_to_anchor=select_plot_scheme['ax1_inset_bbox'], bbox_transform=ax1.transAxes)
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
            if self._exchange_min == self._exchange_max:
                exchange_string = f"Uniform Exc. ({self._exchange_min} [T])"
            else:
                exchange_string = f"J$_{{min}}$ = {self._exchange_min} [T] | J$_{{max}}$ = " \
                                  f"{self._exchange_max} [T]"
            info_box_full_text = (
                    (f"H$_{{0}}$ = {self._static_field} [T] | H$_{{D1}}$ = {self._driving_field1: 2.2e} [T] | "
                     f"H$_{{D2}}$ = {self._driving_field2: 2.2e}[T] \nf = {self._driving_freq} [GHz] | "
                     f"{exchange_string} | N = {self._chain_spins} | ") + r"$\alpha$" +
                    f" = {self._gilbert_factor: 2.2e}")

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
                                                      facecolor=colour_schemes[0]["wavepacket4"], edgecolor=None, lw=0)
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
            self._fig.tight_layout()  # Must be directly before plt.show()
            plt.show()
        else:
            self._fig.savefig(f"{self.output_filepath}_site{site_index}.png", bbox_inches="tight")

    def plot_heaviside_and_dispersions(self, dispersion_relations: bool = True, use_dual_signal_inset: bool = False,
                                       show_group_velocity_cases: bool = False, dispersion_inset: bool = False,
                                       use_demag: bool = False, compare_dis: bool = False,
                                       publication_details: bool = False, interactive_plot: bool = False) -> None:
        """
        Two pane figure where upper pane shows the FFT of Quasi-Heaviside Step Function(s), and the lower pane
        shows dispersion relations of our datasets.

        Filler text. TODO
        
        :param dispersion_inset: Show inset in lower pane which compared Dk^2 dispersion relations
        :param dispersion_relations: Plot lower pane of fig if true
        :param use_dual_signal_inset: Show signals for quasi-Heaviside in separate insets
        :param show_group_velocity_cases: Annotate to show information on type of dispersion (normal/anomalous)
        :param publication_details: Add figure reference lettering
        :param interactive_plot: If `True` mouseclicks to print x/y coords to terminal, else saves image.

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

        time_instant_fft, signal_instant_fft = rfftfreq(num_samples_instant, 1 / sample_rate_instant), rfft(
            signal_instant)
        time_delay_fft, signal_delay_fft = rfftfreq(num_samples_delay, 1 / sample_rate_delay), rfft(signal_delay)

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
            ax1_inset_delayed = inset_axes(ax1, width=1.3, height=0.36, loc="upper right",
                                           bbox_to_anchor=[0.995, 0.805], bbox_transform=ax1.transAxes)

            ax1_inset_instant = inset_axes(ax1, width=1.3, height=0.36, loc="upper right",
                                           bbox_to_anchor=[0.995, 1.185], bbox_transform=ax1.transAxes)

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
                ax1_inset = inset_axes(ax1, width=1.3, height=0.72, loc="upper right", bbox_to_anchor=[0.995, 1.175],
                                       bbox_transform=ax1.transAxes)
            else:
                ax1_inset = inset_axes(ax1, width=1.3, height=0.72, loc="upper right", bbox_to_anchor=[0.995, 0.98],
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
                    # freq_array = gyromag_ratio * (4 * (exc_stiff / sat_mag) / lattice_constant**2
                    #                              * (1 - np.cos(wave_number_array * lattice_constant))
                    #                              + external_field
                    #                              + (dmi_val/sat_mag) * wave_number_array)
                    # freq_array = gyromag_ratio * (2 * (exc_stiff / sat_mag) * wave_number_array**2
                    #                               + external_field
                    #                               + (2 * dmi_val/sat_mag) * wave_number_array)
                    freq_array = gyromag_ratio * (2 * exchange_field * (lattice_constant) ** 2 * wave_number_array ** 2
                                                  + external_field
                                                  + dmi_val * lattice_constant * wave_number_array)

                    ax1.plot(wave_number_array * hz_2_GHz, freq_array * hz_2_GHz, lw=1., ls='-',
                             label=f'D = {dmi_val}')
                    # ax1.plot(wave_number_array * hz_2_GHz, freq_array_dk2 * hz_2_GHz, lw=1., alpha=0.4, ls='--',
                    #         label=r'$(Dk^2)$'f'D = {dmi_val}'r'$(mJ/m^2$)')

                    # These!!
                    # ax2.scatter(np.arccos(1 - ((freqs2 / gamma - h_0) / (2 * h_ex))) / a, freqs2 / 1e12,
                    #             s=0.5, c='red', label='paper')
                    # ax2.plot(k, gamma * (2 * h_ex * (1 - np.cos(k * a)) + h_0) / 1e12, color='red', ls='--', label=f'Kittel')

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
                    d_star = ((2 * dmi_val) / (mu0 * sat_mag_moon))

                    freq_array_moon = gyromag_ratio_moon * mu0 * (np.sqrt((h0 + j_star * wave_number_array_moon ** 2)
                                                                          * (
                                                                                  h0 + demag_mag_moon + j_star * wave_number_array_moon ** 2))
                                                                  + p_val * d_star * wave_number_array_moon)

                    ax2.plot(wave_number_array_moon * hz_2_GHz, freq_array_moon * hz_2_GHz, lw=1., ls='-',
                             label=f'D = {p_val * dmi_val}')
                    # ax1.plot(wave_number_array * hz_2_GHz, freq_array_dk2 * hz_2_GHz, lw=1., alpha=0.4, ls='--',
                    #         label=r'$(Dk^2)$'f'D = {dmi_val}'r'$(mJ/m^2$)')

                    # These!!
                    # ax2.scatter(np.arccos(1 - ((freqs2 / gamma - h_0) / (2 * h_ex))) / a, freqs2 / 1e12,
                    #             s=0.5, c='red', label='paper')
                    # ax2.plot(k, gamma * (2 * h_ex * (1 - np.cos(k * a)) + h_0) / 1e12, color='red', ls='--', label=f'Kittel')

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
                ax2_inset_disp_rels = inset_axes(ax2, width=1.25, height=0.5, loc="lower right",
                                                 bbox_to_anchor=[0.9875, 0.02], bbox_transform=ax2.transAxes)

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

        ax2_inset = inset_axes(ax, width=1.8, height=0.7, loc="upper right", bbox_to_anchor=[0.88, 0.47],
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

    def _fft_data(self, amplitude_data):
        """
        Computes the FFT transform of a given signal, and also outputs useful data such as key frequencies.

        :param amplitude_data: Magnitudes of magnetic moments for a spin site

        :return: A tuple containing the frequencies [0], FFT [1] of a spin site. Also includes the  natural frequency
        (1st eigenvalue) [2], and driving frequency [3] for the system.
        """

        # Find bin size by dividing the simulated time into equal segments based upon the number of data-points.
        sample_spacing = (self._max_time / (self._num_data_points - 1))

        # Compute the FFT
        n = amplitude_data.size
        normalised_data = amplitude_data

        fourier_transform = rfft(normalised_data)
        frequencies = rfftfreq(n, sample_spacing)

        return frequencies, fourier_transform

    def _tick_setter(self, ax, x_major, x_minor, y_major, y_minor, yaxis_multi_loc=False, is_fft_plot=False,
                     xaxis_num_decimals=.1, yaxis_num_decimals=.1, yscale_type='sci', format_xaxis=False,
                     show_sci_notation=False):

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
                    #        verticalalignment='center',
                    #        horizontalalignment='center', transform=ax.transAxes, fontsize=self._fontsizes["smaller"])

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
                    #        verticalalignment='center',
                    #        horizontalalignment='right', transform=ax.transAxes, fontsize=self._fontsizes["smaller"])

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

        :param publication_details: Add figure reference lettering
        :param interactive_plot: If `True` mouseclicks to print x/y coords to terminal, else saves image.

        :return: Saves a .png image to the designated output folder.
        """
        if self._fig is None:
            self._fig = plt.figure(figsize=(4.5, 3.375))
        self._fig.subplots_adjust(wspace=1, hspace=0.35)

        num_rows, num_cols = 2, 3

        ax1 = plt.subplot2grid((num_rows, num_cols), (0, 0), rowspan=int(num_rows / 2),
                               colspan=num_cols, fig=self._fig)
        ax2 = plt.subplot2grid((num_rows, num_cols), (int(num_rows / 2), 0),
                               rowspan=num_rows, colspan=num_cols, fig=self._fig)
        ########################################
        # Key values and computations that are common to both systems
        hz_2_GHz, hz_2_THz, m_2_nm = 1e-9, 1e-12, 1e9
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

        ########################################

        if not use_demag:
            demag_mag_moon = 0

        if find_modes:

            # System length setup
            max_len = round(system_len / lattice_constant)
            half_max_length = int(max_len / 2)
            n_lower, n_upper = 0, half_max_length

            num_spins_array = np.arange(-half_max_length, half_max_length + 1, 1)
            total_sys_pairs = (max_len - 1) * lattice_constant

            # Output controls
            should_print_only_matches = True
            should_print_only_half_ints = True
            should_highlight_all_matches = True
            should_highlight_half_ints = True
            use_original_wavenumbers = True

            if should_print_only_half_ints:
                should_print_only_matches = False

            # Precision of output
            wv_rnd = 6
            fq_rnd = 3  # kz resolution for rounding

            # Error tolerances
            wv_tol = 10 ** -(wv_rnd - 1)
            freq_atol = 12 * 10 ** -(fq_rnd - 1)
            half_int_atol = 5e-1

            # Calculate all wavevectors in system
            wave_number_array = (2 * num_spins_array * np.pi) / total_sys_pairs

            # Calculate all frequencies in system assuming that there is no demagnetisation
            freq_array = gyromag_ratio * (2 * exchange_field * lattice_constant ** 2 * wave_number_array ** 2
                                          + external_field
                                          + dmi_val_const * lattice_constant * wave_number_array)
            # Calculate all wavelengths
            wavelengths_array = np.zeros_like(wave_number_array, dtype=float)

            # Set wavelength to infinity where wave number is zero
            zero_wave_indices = wave_number_array == 0
            wavelengths_array[zero_wave_indices] = np.inf

            # Perform division where wave number is non-zero and convert to [nm]
            non_zero_wave_indices = ~zero_wave_indices
            wavelengths_array[non_zero_wave_indices] = ((2 * np.pi) / wave_number_array[non_zero_wave_indices]) * m_2_nm

            # Convert wave numbers to [1/nm] with rounding
            wave_number_array = np.round(wave_number_array * 1e-9, wv_rnd)
            wavevectors_from_n = wave_number_array[half_max_length + n_lower:half_max_length + n_upper + 1]

            # Convert frequencies to [GHz] with rounding
            freq_array = abs(np.round(freq_array * hz_2_GHz, fq_rnd))  # all in GHz now

            # Initialize containers
            frequency_container = []
            positive_wave_numbers = wave_number_array[wave_number_array >= 0]

            #l1 = wavelengths_array[2001:2040]
            #l2 = np.flip(wavelengths_array[1813:1852])
            #print(l1)
            #print(l2)
            #print(l1/l2 % 1)
            #f1 = freq_array[2001:2040]
            #f2 = np.flip(freq_array[1813:1852])
            #print(abs(f1))
            #print(abs(f2))
            #exit(0)

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
                    half_smaller = smaller / 2

                    # Calculate modulus
                    modulus = larger % smaller

                    # Calculate scaling
                    scaling_factors.append(larger / smaller % 1)

                    # Calculate tolerance range
                    tolerance = half_smaller * atol
                    lower_bound = half_smaller - tolerance
                    upper_bound = half_smaller + tolerance

                    # Check if modulus is within the tolerance range
                    if lower_bound <= modulus <= upper_bound:
                        results.append(True)
                    else:
                        results.append(False)

                return results, scaling_factors

            def is_wavelength_whole(n1, n2, atol=1e-2, convert_to_wavelength=False):
                if convert_to_wavelength:
                    # Convert from wavenumber [m] to wavelength [nm]
                    wavelength1 = abs(((2 * np.pi) / n1) * m_2_nm)
                    wavelength2 = abs(((2 * np.pi) / n2) * m_2_nm)
                else:
                    wavelength1 = abs(n1)
                    wavelength2 = abs(n2)

                # Determine larger and smaller numbers
                larger, smaller = max(wavelength1, wavelength2), min(wavelength1, wavelength2)

                # Calculate modulus
                modulus = larger / smaller

                # Test rounding

                # Check if modulus is within the tolerance range
                # return lower_bound <= modulus <= upper_bound

            for wavevector_n in wavevectors_from_n:
                if use_original_wavenumbers:
                    # Will always exactly match so no need to test
                    closest_match_wavevector = wavevector_n
                else:
                    closest_match_wavevector = min(positive_wave_numbers, key=lambda x: abs(x - wavevector_n))

                # Step 2: Find the index of the closest match in wave_number_array
                closest_match_index = np.where(np.isclose(wave_number_array, closest_match_wavevector, atol=wv_tol))[0]

                # Step 3, 4, 5: For each match, find frequency and check for other occurrences
                for match_index in closest_match_index:
                    # For each match, find the corresponding frequency
                    match_frequency = freq_array[match_index]
                    if match_frequency < external_field * gyromag_ratio * hz_2_GHz:
                        continue
                    match_wavelength = wavelengths_array[match_index]

                    # Find all other occurrences of this frequency, and then their indices
                    matched_freq_indices = np.where(np.isclose(freq_array, match_frequency, atol=freq_atol))[0]
                    other_occurrences_indices = [i for i in matched_freq_indices if i != match_index]
                    other_occurrences_frequencies = freq_array[other_occurrences_indices]  # mainly for debugging

                    # Find the corresponding wavevectors for the other occurrences of the given frequency
                    other_occurrences_wavevectors = wave_number_array[other_occurrences_indices]

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

                    # Check if we have any matches
                    other_occurrences_half_ints, other_occurrences_scaling = is_wavelength_half(match_wavelength, other_occurrences_wavelengths,
                                                                     atol=half_int_atol)
                    # Debugging
                    #if other_occurrences_indices:
                    #    print(f"Freq: {match_frequency} at {match_index} for {match_wavelength} | "
                    #          f"Matches: {freq_array[other_occurrences_indices]} at {other_occurrences_indices} "
                    #          f"for {other_occurrences_wavelengths} ")
                    #    print(wavevector_n, closest_match, other_occurrences_wavevectors,
                    #          match_index, other_occurrences_indices,
                    #          match_frequency, freq_array[other_occurrences_indices],
                    #          match_wavelength, other_occurrences_wavelengths)

                    # Recording the information
                    frequency_container.append({
                        'match_index': match_index,
                        'user_wavevector': wavevector_n,
                        'closest_wavevector': closest_match_wavevector,
                        'match_frequency': match_frequency,
                        'match_wavelength': match_wavelength,

                        'other_occurrences_indices': other_occurrences_indices,
                        'other_occurrences_wavevectors': other_occurrences_wavevectors,
                        'other_occurrences_frequencies': other_occurrences_frequencies,
                        'other_occurrences_wavelengths': other_occurrences_wavelengths,
                        'other_occurrences_half_ints': other_occurrences_half_ints,
                        'other_occurrences_scaling': other_occurrences_scaling
                    })

            # Step 6: Sort the container by frequency and then by wavevector
            frequency_container.sort(key=lambda x: (x['match_frequency'], x['closest_wavevector']))

            # for entry in frequency_container:
            #     print((entry['match_frequency'], entry['other_occurrences_frequencies']))
            # exit(0)

            class bcolours:
                PURPLE = '\033[95m'
                BLUE = '\033[94m'
                GREEN = '\033[92m'
                ORANGE = '\033[93m'
                RED = '\033[91m'
                ENDC = '\033[0m'  # Black

                def disable(self):
                    self.HEADER = ''
                    self.OKBLUE = ''
                    self.OKGREEN = ''
                    self.WARNING = ''
                    self.FAIL = ''
                    self.ENDC = ''  # Black

            line_counter = 0
            for entry in frequency_container:
                # Note that I can't run simulations for wavelengths smaller than 1nm so there's no point being
                # more precise than this
                match_index = entry['match_index']
                match_frequency = entry['match_frequency']
                match_wavevector = entry['closest_wavevector']
                match_wavelength = entry['match_wavelength']

                if should_print_only_half_ints and not any(entry['other_occurrences_half_ints']):
                    continue

                color = bcolours.ENDC
                if entry['other_occurrences_indices']:
                    # Current case has a match. Rarest case first
                    if should_highlight_half_ints and any(entry['other_occurrences_half_ints']):
                        color = bcolours.PURPLE

                    elif should_highlight_all_matches:
                        color = bcolours.BLUE
                if entry['other_occurrences_indices'] or not should_print_only_matches:
                    # Print the match information
                    print(f"{color}"
                          f"\u03C9/2\u03C0: {match_frequency:.{fq_rnd}f} [GHz] | "
                          f"kn: {match_wavevector:.{wv_rnd}f} [1/nm] | "
                          f"in: {match_index}, "
                          f"λn: {match_wavelength:.{0}f} [nm]"
                          f"\t", end="")

                    # Iterate over other occurrences should the exist
                    if entry['other_occurrences_indices']:
                        for enum_index, (other_index, other_wavevector, other_wavelength, other_scaling, other_half_int) in enumerate(
                                zip(entry['other_occurrences_indices'],
                                    entry['other_occurrences_wavevectors'],
                                    entry['other_occurrences_wavelengths'],
                                    entry['other_occurrences_scaling'],
                                    entry['other_occurrences_half_ints'])):

                            if should_highlight_half_ints and other_half_int:
                                color = bcolours.PURPLE
                            elif should_highlight_all_matches:
                                color = bcolours.BLUE

                            print(
                                f"{color}| i{enum_index+1}: {other_index}, "
                                f"\u03BB{enum_index+1}: {other_wavelength:.{0}f} [nm], "
                                f"k{enum_index+1}: {other_wavevector:.{wv_rnd}f} [1/nm],"
                                f"\t\u03BE{enum_index+1}: {other_scaling:.3f}"
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
            # Plot dispersion relations
            self._fig.suptitle('Comparison of my derivation with Moon\'s')
            for dmi_val in dmi_vals:
                max_len = round(system_len / lattice_constant)
                num_spins_array = np.arange(-int(max_len / 2), int(max_len / 2) + 1, 1)
                wave_number_array = (num_spins_array * np.pi) / ((len(num_spins_array) - 1) * lattice_constant)

                freq_array = gyromag_ratio * (2 * exchange_field * (lattice_constant) ** 2 * wave_number_array ** 2
                                              + external_field
                                              + dmi_val * lattice_constant * wave_number_array)

                ax1.plot(wave_number_array * hz_2_GHz, freq_array * hz_2_GHz, lw=1., ls='-',
                         label=f'D = {dmi_val}')

                ax1.set(xlabel="Wavevector (nm$^{-1}$)",
                        ylabel='Frequency (GHz)', xlim=[-0.15, 0.15], ylim=[0, 20])
                self._tick_setter(ax1, 0.1, 0.05, 3, 2, is_fft_plot=False,
                                  xaxis_num_decimals=.1, yaxis_num_decimals=2.0, yscale_type='plain')

                ax1.margins(0)
                ax1.xaxis.labelpad = -2
                ax1.legend(title='Mine\n'r'$(J/m^2$)', title_fontsize=self._fontsizes["smaller"],
                           fontsize=self._fontsizes["tiny"], frameon=True, fancybox=True)

            for p_val, dmi_val in zip(p_vals_moon, dmi_vals_moon):
                max_len_moon = round(system_len_moon / lattice_constant_moon)
                num_spins_array_moon = np.arange(-max_len_moon, max_len_moon, 1)
                wave_number_array_moon = (num_spins_array_moon * np.pi) / (
                        (len(num_spins_array_moon) - 1) * lattice_constant_moon)

                h0 = external_field_moon / mu0
                j_star = ((2 * exc_stiff_moon) / (mu0 * sat_mag_moon))
                d_star = ((2 * dmi_val) / (mu0 * sat_mag_moon))

                freq_array_moon = gyromag_ratio_moon * mu0 * (np.sqrt((h0 + j_star * wave_number_array_moon ** 2)
                                                                      * (
                                                                              h0 + demag_mag_moon + j_star * wave_number_array_moon ** 2))
                                                              + p_val * d_star * wave_number_array_moon)

                ax2.plot(wave_number_array_moon * hz_2_GHz, freq_array_moon * hz_2_GHz, lw=1., ls='-',
                         label=f'D = {p_val * dmi_val}')
                ax2.set(xlabel="Wavevector (nm$^{-1}$)",
                        ylabel='Frequency (GHz)', xlim=[-0.15, 0.15], ylim=[0, 20])
                self._tick_setter(ax2, 0.1, 0.05, 3, 2, is_fft_plot=False,
                                  xaxis_num_decimals=.1, yaxis_num_decimals=2.0, yscale_type='plain')

                ax2.margins(0)
                ax2.xaxis.labelpad = -2

                ax2.legend(title='Theirs\n'r'$(J/m^2$)', title_fontsize=self._fontsizes["smaller"],
                           fontsize=self._fontsizes["tiny"], frameon=True, fancybox=True)

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

        ax.grid(visible=True, axis='both', which='both', ls='-', lw=2)

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
        ax1 = fig.add_subplot(111)

        colour2 = '#5584B9'
        sns.lineplot(x=range(0, len(mx_mode)), y=mx_mode, marker='o', markersize=5,
                     linestyle='', alpha=1, ax=ax1, color=colour2, label='Mx')
        sns.lineplot(x=range(0, len(mx_mode)), y=mx_mode, lw=1.75,
                     linestyle='-', alpha=0.5, ax=ax1, color=colour2)

        ax1.set(xlabel="Distance (nm)", ylabel="Amplitude (arb. units)",
                xlim=(0, number_of_spins))  # ,
        # title = f"Eigenmode #{eigenmode}",
        # xticks=np.arange(0, number_of_spins, np.floor(number_of_spins - 2) / 20))

        ax1.xaxis.set(major_locator=ticker.MultipleLocator(50),
                      minor_locator=ticker.MultipleLocator(10))
        ax1.yaxis.set(major_locator=ticker.MultipleLocator(0.1),
                      minor_locator=ticker.MultipleLocator(0.025))

        ax1.text(0.025, 0.925, "(b)", verticalalignment='center', horizontalalignment='left',
                 transform=ax1.transAxes, fontsize=8)

        # Legend doubles as a legend (showing propagation direction), and the frequency [Hz] of the eigenmode.
        ax1.legend(loc=1, bbox_to_anchor=(0.975, 0.975),
                   frameon=True, fancybox=True, facecolor='white', edgecolor='white',
                   title=f"Frequency (GHz)\n        {frequency: 4.1f}\n     Component",
                   fontsize=8, title_fontsize=8)

        ax1.axvspan(0, number_of_spins * driving_width, color='black', alpha=0.2)

        ax1.grid(visible=True, axis='both', which='both', color='black', ls='--', lw=1, alpha=0.0)

        # ax.set_facecolor('#f4f4f5')
        ax1.tick_params(axis="both", which="both", bottom=True, top=True, left=True, right=True, zorder=1.9999)
        ax1.set_axisbelow(False)

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
            ax.legend(frameon=False)

            ax.set_axisbelow(False)
            ax.set_facecolor("white")

        fig.savefig(f"{self.output_filepath}_DispersionRelation.png", bbox_inches="tight")

        print("--------------------------------------------------------------------------------")
