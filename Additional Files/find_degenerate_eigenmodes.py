# -*- coding: utf-8 -*-

# -------------------------- Preprocessing Directives -------------------------

# Standard Libraries
import logging as lg
# import os as os
from sys import exit

# 3rd Party packages
from datetime import datetime

# import matplotlib.pyplot as plt
# import numpy as np

# My packages/Header files
# Here

# ----------------------------- Program Information ----------------------------

"""
Description of what foo.py does
"""
PROGRAM_NAME = "foo.py"
"""
Created on (date) by (author)
"""


# ---------------------------- Function Declarations ---------------------------

class find_degenerate_modes:
    """
    Generates a single subplot that can either be a PNG or GIF.

    Useful for creating plots for papers, or recreating a paper's work. To change between the png/gif saving options,
    change the invocation in data.analysis.py.
    """

    def __init__(self):

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
                self._yaxis_lim *= max(self.amplitude_data[row_index, :])
                plt.rcParams.update({'savefig.dpi': 200, "figure.dpi": 200})  # Prevent excessive image sizes

            # Any method-wide updates
            self._axes.clear()
            self._fig_kwargs["ylim"] = [-1 * self._yaxis_lim, self._yaxis_lim]
        else:
            self._axes.clear()  # Only triggered if existing plot is present
            if not static_ylim:
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

def loggingSetup():
    """
    Minimum Working Example (MWE) for logging. Pre-defined levels are:
        
        Highest               ---->            Lowest
        CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    """
    today_date = datetime.now().strftime("%y%m%d")
    current_time = datetime.now().strftime("%H%M")

    lg.basicConfig(filename=f'./{today_date}-{current_time}.log',
                   filemode='w',
                   level=lg.INFO,
                   format='%(asctime)s | %(module)s::%(funcName)s | %(levelname)s | %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   force=True)


# --------------------------- main() implementation ---------------------------

def main():
    lg.info(f"{PROGRAM_NAME} start")

    # Enter code here

    lg.info(f"{PROGRAM_NAME} end")
    exit()


# ------------------------------ Implementations ------------------------------

if __name__ == '__main__':
    loggingSetup()

    main()
