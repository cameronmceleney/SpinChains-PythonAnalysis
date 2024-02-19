#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Full packages
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.widgets
import numpy as np
import seaborn as sns

# Specific functions from packages
from datetime import datetime, timedelta
from sys import platform as sys_platform
from tabulate import tabulate
from typing import Any, List

# My full modules


# Specific functions from my modules


"""
    Need to add
"""

"""
    Core Details

    Author      : cameronmceleney
    Created on  : 14/02/2024 12:00
    Filename    : figure_manager.py
    IDE         : PyCharm
"""


class Debouncer:
    def __init__(self, delay_ms=500):
        """
        Initialize the debouncer with a specified delay in milliseconds.
        """
        self.delay = timedelta(milliseconds=delay_ms)
        self.last_time = None

    def should_process(self):
        """
        Determine if the current event should be processed.
        """
        now = datetime.now()
        if self.last_time is None or now - self.last_time >= self.delay:
            self.last_time = now
            return True
        return False

    def should_not_process(self):
        """
        Determine if the current event should not be processed, essentially the inverse of should_process.
        """
        return not self.should_process()


class FigureManager:
    def __init__(self, fig, ax, width, height, dpi, driving_freq):
        self.fig = fig
        self.subplots = ax
        self.figure_manager = plt.get_current_fig_manager()

        # If frequently crashing due to being unresponsive, increase the delay
        self.debouncer = Debouncer(delay_ms=100)
        self.driving_freq = driving_freq

        self.width = width
        self.height = height
        self.dpi = dpi
        self.state = False
        self.button = None

        self.mouse_inputs = {'MB1': {'xdata': None, 'ydata': None, 'inaxes': None, 'key': None, 'button': None}}
        self.modifiers = {'ctrl', 'cmd', 'shift'}
        self.should_draw = {'elements': {'dot': False, 'vline': False, 'hline': False}}
        self.file_empty = True

        self.select_subplot = 0
        self.drawn_lines = []
        self.drawn_dots = []
        self.cid1 = None
        self.cid2 = None
        self.cid3 = None

        self.keymappings = {}
        self.are_keymaps_strings = False

        self.engage_left_click = True

        self.num_clicks = 0
        self.total_diff_wl = 0
        self.last_wl = None
        self.print_mode = 0  # Introduce a flag to toggle between print statements

        self.axes_list = []  # List to store all subplot axes in the figure
        self._populate_axes_list()  # Populate the list with subplot axes
        # self._create_widgets()
        self.default_keybindings()

    def wait_for_close(self):
        plt.show()

    def connect_events(self):
        """Connects the key press event with additional parameters."""
        self.cid1 = self.figure_manager.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.cid2 = self.figure_manager.canvas.mpl_connect('key_press_event', self.on_letter_press)
        self.cid3 = self.figure_manager.canvas.mpl_connect('key_press_event', self.on_number_press)

    def default_keybindings(self):
        """
        Add custom keymappings including to pre-bound keys in matplotlib.

        One can either edit rcParams directly (like below) or create a matplotlibrc file, and place a copy in
        "$HOME/.matplotlib/matplotlibrc". The original file (DO NOT EDIT) is located at
        "sites-packages/matplotlib/mpl-data/" (relative to your Python installation directory).
        """
        mpl_default_keymaps_remap = {
            'fullscreen': ('Toggle fullscreen mode', ['ctrl+f', 'cmd+f']),
            'home': ('Reset original view', ['r', 'home']),
            'back': ('Back to previous view', ['left']),
            'forward': ('Forward to next view', ['right']),
            'pan': ('Pan axes with left mouse, zoom with right', ['up']),
            'zoom': ('Zoom to rectangle', ['down']),
            'save': ('Save the figure', ['ctrl+s']),
            'help': ('Show help', ['f1']),
            'quit': ('Close the current figure', ['Q']),
            'quit_all': ('Close all figures', ['ctrl+Q', 'cmd+Q']),
            'grid': ('Toggle grid', ['G']),
            'grid_minor': ('Toggle minor grid', ['']),
            'yscale': ('Toggle scaling of y-axes (log/linear)', ['ctrl+l', 'cmd+l']),
            'xscale': ('Toggle scaling of x-axes (log/linear)', ['ctrl+k', 'cmd+k']),
            'copy': ('Copy the current selection to the clipboard', ['ctrl+c', 'cmd+c'])
        }

        # Update plt.rcParams and prepare new default keymappings
        for key, (description, keybindings) in mpl_default_keymaps_remap.items():
            plt.rcParams[f'keymap.{key}'] = keybindings
            self.keymappings[key] = [description, keybindings, ['default']]

        custom_keymaps = {
            'clear dots': ('Clear all drawn dots', ['c']),
            'clear lines': ('Clear all drawn lines', ['C']),
            'draw dots': ('Toggle (on/off) drawing dots', ['d']),
            'draw lines': ('Toggle (parallel/perp./off) drawing lines', ['D']),
            'export new': ('Export the current frequency and wavevector', ['e']),
            'export existing': ('Export the current wavevector', ['E']),
            'export clear': ('Clear the last export', ['ctrl+e', 'cmd+e']),
            'resize figure': ('Resize to original figure size', ['f']),
            'user help': ('List core keymappings', ['H']),
            'full help': ('List all keymappings', ['ctrl+h', 'cmd+h']),
            'use mouse': ('Toggle mouse interaction (WIP)', ['m']),
            'reset clicks': ('Reset click data', ['R']),
            'toggle state': ('Toggle state of button (WIP)', ['t']),
            'convert x-value': ('Convert last x-axis value to λ [nm]', ['w'])
        }

        # Merge custom keymappings and set category to custom
        for key, (desc, keys) in custom_keymaps.items():
            self.keymappings[key] = [desc, keys, ['custom']]

    def on_mouse_click(self, event: Any) -> None:
        """Handles mouse click events to store the y-coordinate."""
        if not self.engage_left_click:
            return

        if self.debouncer.should_not_process():
            return

        if event.button == 1:
            # Left click
            self.mouse_inputs['MB1'].update({"xdata": event.xdata,
                                             "ydata": event.ydata,
                                             "inaxes": event.inaxes,
                                             "key": event.key,
                                             "button": event.button})
            if event.inaxes:
                self._console_output()
                self._drawing_element_interface()

        elif event.button == 2:
            # Middle mouse
            self.print_mode = (self.print_mode + 1) % 3
            print(f'----------------------------------------'
                  f'\nPrint mode changed to {self.print_mode}.')

        elif event.button == 3:
            pass

    def on_letter_press(self, event: Any) -> None:

        if not any(event.key.startswith(mod + '+') for mod in self.modifiers):
            if self.debouncer.should_not_process():
                return

        if event.key in self.keymappings['clear dots'][1] + self.keymappings['clear lines'][1]:
            self.clear_element_interface(event)

        elif event.key in self.keymappings['draw dots'][1] + self.keymappings['draw lines'][1]:
            self._drawing_element_selector(event.key)

        elif (event.key in self.keymappings['export new'][1] + self.keymappings['export existing'][1]
              + self.keymappings['export clear'][1]):
            self._export_data(event.key, 'D:/Data/2024-02-14/output_data3.csv')

        elif event.key in self.keymappings['resize figure'][1]:
            self._reset_figure_size()

        elif event.key in self.keymappings['user help'][1] + self.keymappings['full help'][1]:
            self._show_help(event.key)

        elif event.key in self.keymappings['use mouse'][1]:
            self.engage_mouse_button_1()

        elif event.key in self.keymappings['reset clicks'][1]:
            self._reset_clicks()

        elif event.key in self.keymappings['toggle state'][1]:  # Example: Toggle state with the 't' key
            self._toggle_state()

        elif event.key in self.keymappings['convert x-value'][1]:
            self._handle_conversions()

    def on_number_press(self, event: Any) -> None:
        """Selects a subplot based on the key pressed."""
        if not any(event.key.startswith(mod + '+') for mod in self.modifiers):
            if self.debouncer.should_not_process():
                return

        try:
            event_key = int(event.key)
            if event_key in self.axes_list:
                # Update the current subplot based on the key press if within range
                self.select_subplot = self.axes_list[event_key]
                print('--------------------')
                print(f"Selected subplot {event_key}")
        except ValueError:
            pass
        else:
            pass

    def _export_data(self, event_key: str, file_path: str = 'event_data.csv') -> None:
        # Proceed with original function's logic
        if self.mouse_inputs['MB1']['xdata'] is None:
            print("No data to export.")
            return
        elif self.mouse_inputs['MB1']['inaxes'] is not self.subplots[1] or self.mouse_inputs['MB1']['inaxes'] is not self.subplots[2]:
            return

        header_row = ['Frequency [GHz]', 'Wavevector (LHS) [1/nm]', 'Wavevector (RHS) [1/nm]']
        if self.file_empty:
            try:
                # Attempt to open the file in 'r+' mode to read and write without truncating existing content
                with open(file_path, 'r+', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    first_row = next(reader, None)

                    # If the file is not empty but the first row is not the header, seek to start and write the header
                    if not first_row:
                        csvfile.seek(0)
                        writer = csv.writer(csvfile)
                        writer.writerow(header_row)
                        print("Inserted default values into empty file.")
                        csvfile.flush()  # Ensure the header is written before proceeding
                        csvfile.seek(0)  # Reset file pointer to start for subsequent operations
                        self.file_empty = False

                    elif first_row != header_row:
                        # Handle the case if the first row is not the header
                        csvfile.seek(0)  # Reset file pointer to start for subsequent operations
                        self.file_empty = False

                    #if event_key in self.keymappings['export existing'][1] + self.keymappings['export clear'][1]:
                    #    print("Cannot work on empty file.")
                    #    return

            except FileNotFoundError:
                # If the file doesn't exist, create it and write the header
                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(header_row)
                    print("Created the file and inserted default values.")
                self.file_empty = False
            else:
                with open(file_path, 'r+', newline='') as csvfile:
                    # Read the file to check if the frequency is already present
                    reader = list(csv.reader(csvfile))
                    if event_key in self.keymappings['export existing'][1] or event_key in \
                            self.keymappings['export clear'][1]:
                        if not reader:
                            print("Cannot work on empty file.")
                            return

        with open(file_path, 'r+', newline='') as csvfile:
            # Read the file to check if the frequency is already present
            reader = list(csv.reader(csvfile))
            last_row = reader[-1] if reader else None

            last_value = last_row[-1] if last_row else None

            if last_value == str(self.mouse_inputs['MB1']['xdata']) and event_key not in self.keymappings['export clear'][1]:
                print("Skipped writing duplicate data.")
                return

            freq_already_present = str(self.driving_freq) in last_row if last_row else False

        if event_key in self.keymappings['export new'][1]:
            # Append a new row with the current frequency and wavevector
            if freq_already_present:
                print("Freq. already present in current line. Append instead.")
            else:
                with open(file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([self.driving_freq, self.mouse_inputs['MB1']['xdata']])
                print(f"Export. FREQ: {self.driving_freq} | K: {self.mouse_inputs['MB1']['xdata']: .3f}")

        elif event_key in self.keymappings['export existing'][1]:
            # Append the current wavevector to the last row
            with open(file_path, 'r+', newline='') as csvfile:
                reader = list(csv.reader(csvfile))
                if reader:
                    reader[-1].append(self.mouse_inputs['MB1']['xdata'])
                    csvfile.seek(0)
                    csvfile.truncate()
                    writer = csv.writer(csvfile)
                    writer.writerows(reader)
                    print(f'Append. K: {self.mouse_inputs["MB1"]["xdata"]: .3f}')

        elif event_key in self.keymappings['export clear'][1]:
            # Read the file, remove the last value in the last row, and rewrite the file
            with open(file_path, 'r+', newline='') as csvfile:
                if header_row[2] in reader[-1]:
                    print(f"Can't clear the header row.")
                    return
                else:
                    if len(reader[-1]) > 1:
                        removed = reader[-1].pop()
                        csvfile.seek(0)
                        csvfile.truncate()
                        writer = csv.writer(csvfile)
                        writer.writerows(reader)
                        print(f"Removed: {removed}")

                    elif str(self.driving_freq) in reader[-1][0]:
                        print(f"Can't clear: only freq. exported.")
                        return
        else:
            print("Error: Failed to export.")

    def _populate_axes_list(self):
        """Populates the list with all subplot axes in the figure."""
        self.axes_list = list(range(0, len(plt.gcf().axes) + 1))

    def _show_help(self, event_key: str):
        """Prints the keymappings to the console."""
        table_data = []

        print(self.keymappings['full help'])

        for key, value in self.keymappings.items():
            if event_key in self.keymappings['user help'][1] and 'custom' in value[2]:
                table_data.append([key, value[0], ', '.join(value[1])])

            elif event_key in self.keymappings['full help'][1]:
                table_data.append([key, value[0], ', '.join(value[1])])

        print(tabulate(table_data, headers=["Event", "Effect", "Keymap"], tablefmt="fancy_outline"))

    def _reset_figure_size(self):
        """Resets the figure size based on the key press event."""
        """Resets the figure size based on the key press event."""
        if self.figure_manager is not None:
            # Attempt to retrieve the window object using other methods or attributes
            width = self.width * self.dpi
            height = self.height * self.dpi

            width, height = (int(width * 1.75), int(height * 1.75)) if sys_platform in ['win32', 'win64'] else (
                int(width), int(height))

            self.figure_manager.resize(int(width), int(height))
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        else:
            print("Error: Figure manager is not initialized.")

    def _drawing_element_selector(self, event_key: str):

        should_draw = self.should_draw['elements']
        print(f'Draw ', end='')

        if event_key in self.keymappings['draw dots'][1]:
            should_draw['dot'] = not should_draw['dot']
            if should_draw['dot']:
                should_draw.update({'vline': False, 'hline': False})
            print(f'dots: {'enabled' if should_draw['dot'] else 'disabled'}.')

        elif event_key in self.keymappings['draw lines'][1]:
            if not should_draw['hline'] and not should_draw['vline']:
                should_draw['hline'] = True
                print(f'hline: enabled')
            elif should_draw['hline']:
                should_draw['hline'] = False
                should_draw['vline'] = True
                print(f'vline: enabled')
            elif should_draw['vline']:
                should_draw['hline'] = should_draw['vline'] = False
                print(f'lines: disabled.')

            # Disable dot drawing if any line drawing is enabled
            if should_draw['hline'] or should_draw['vline']:
                should_draw['dot'] = False

    def _drawing_element_interface(self, line_lims: List[float | int] = (0, 1)) -> None:
        """Interface to gather user input to draw elements (lines/dots) onto the figure."""

        new_element = None
        if self.should_draw['elements']['dot']:
            new_element = self.mouse_inputs['MB1']['inaxes'].plot(self.mouse_inputs['MB1']['xdata'],
                                                                  self.mouse_inputs['MB1']['ydata'],
                                                                  markersize=4, marker='x', color='black', zorder=1.31)
            self.drawn_dots.extend(new_element)

        elif self.should_draw['elements']['hline']:
            new_element = self.mouse_inputs['MB1']['inaxes'].axhline(y=self.mouse_inputs['MB1']['ydata'],
                                                                     xmin=line_lims[0], xmax=line_lims[1],
                                                                     ls='-', lw=1.0, color='black',
                                                                     alpha=0.5, zorder=1.3)

        elif self.should_draw['elements']['vline']:
            new_element = self.mouse_inputs['MB1']['inaxes'].axvline(x=self.mouse_inputs['MB1']['xdata'],
                                                                     ymin=line_lims[0], ymax=line_lims[1],
                                                                     ls='-', lw=1.0, color='black',
                                                                     alpha=0.5, zorder=1.3)

        if self.should_draw['elements']['vline'] or self.should_draw['elements']['hline']:
            self.drawn_lines.append(new_element)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def clear_element_interface(self, event: Any) -> None:
        """Interface to gather user input to clear drawn elements."""
        if event.key in self.keymappings['clear dots'][1]:
            for dot in self.drawn_dots:
                dot.remove()
            self.drawn_dots.clear()
        elif event.key in self.keymappings['clear lines'][1]:
            for line in self.drawn_lines:
                line.remove()
            self.drawn_lines.clear()
        else:
            # Leaving space for future functionality
            pass
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _reset_clicks(self):
        if self.num_clicks != 0:
            self.num_clicks = 0
            self.total_diff_wl = 0
            self.last_wl = None
            print("Click data has been reset.")

    def _handle_conversions(self):
        if self.mouse_inputs['MB1']['xdata'] is not None and self.mouse_inputs['MB1']['ydata'] is not None:
            print(f'λ: {(2 * np.pi) / self.mouse_inputs['MB1']['xdata']:.3f} [nm]')
        else:
            pass

    def _create_widgets(self):
        button_ax = plt.axes((0.1, 0.05, 0.1, 0.075))  # left, bottom, width, height
        self.button = mpl.widgets.Button(button_ax, 'Click Me')

    def _toggle_state(self):
        self.state = not self.state
        print(f"State is now: {self.state}")
        # Update the button label or appearance based on the state
        self.button.label.set_text('On' if self.state else 'Off')
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def engage_mouse_button_1(self):
        self.engage_left_click = not self.engage_left_click

        if self.engage_left_click:
            self.cid1 = self.figure_manager.canvas.mpl_connect('button_press_event', self.cid1)
        else:
            self.figure_manager.canvas.mpl_disconnect(self.cid1)

        print(f'Mouse had been: {"enabled" if self.engage_left_click else "disabled"}.')

    def _console_output(self, ) -> None:
        """Handles console outputs based on the mouse event."""

        if self.print_mode == 1:
            # Override each default by pressing num key after mouse is hovering over plot
            if self.select_subplot == 1 or self.mouse_inputs['MB1']['inaxes'] == self.subplots[0]:
                print(f'n: {round(self.mouse_inputs['MB1']['xdata'], 0):.0f} [site] |'
                      f' m_x: {self.mouse_inputs['MB1']['ydata']:.3e} [a.u.]')

            elif self.select_subplot == 2 or self.mouse_inputs['MB1']['inaxes'] == self.subplots[1]:
                print(f'k: {self.mouse_inputs['MB1']['xdata']:.5f} [nm] | '
                      f'Amplitude: {self.mouse_inputs['MB1']['ydata']:.3e} [a.u.]')

            elif self.select_subplot == 3 or self.mouse_inputs['MB1']['inaxes'] == self.subplots[2]:
                print(f'k: {self.mouse_inputs['MB1']['xdata']:.5f} [nm] | '
                      f'f: {self.mouse_inputs['MB1']['ydata']:.5f} [GHz]')

            else:
                print(f'x: {self.mouse_inputs['MB1']['xdata']} | '
                      f'y: {self.mouse_inputs['MB1']['ydata']}')

        elif self.print_mode == 2 and self.mouse_inputs['MB1']['inaxes'] == self.subplots[0]:
            # Only used to find average wavelength
            self.num_clicks += 1
            print(f'#{self.num_clicks} | n: {round(self.mouse_inputs['MB1']['xdata'], 0):.0f} [site] | '
                  f'm_x: {self.mouse_inputs['MB1']['ydata']:.3e} [a.u.]', end='')

            if self.last_wl is not None:
                diff_wl = abs(self.mouse_inputs['MB1']['xdata'] - self.last_wl)
                self.total_diff_wl += diff_wl

                if self.num_clicks > 1:
                    avg_diff_wl = self.total_diff_wl / (self.num_clicks - 1)
                    print(
                        f' | Avg. λ: {avg_diff_wl:.1f}, Avg. k: {(2 * np.pi / avg_diff_wl):.3e}', end='')
            print(end='\n')
            self.last_wl = self.mouse_inputs['MB1']['xdata']


def rc_params_update():
    """Container for program's custom rc params, as well as Seaborn (library) selections."""
    plt.style.use('fivethirtyeight')
    sns.set(context='notebook', font='Kohinoor Devanagari', palette='muted', color_codes=True)
    ##############################################################################
    # Sets global conditions including font sizes, ticks and sheet style
    # Sets various font size. fsize: general text. lsize: legend. tsize: title. ticksize: numbers next to ticks
    medium_size = 14
    small_size = 12
    large_size = 20
    smaller_size = 10
    # tiny_size = 8

    # sets the tick direction. Options: 'in', 'out', 'inout'
    t_dir = 'in'
    # sets the tick size(s) and tick width(w) for the major and minor axes of all plots
    t_maj_s = 5
    t_min_s = t_maj_s / 2
    t_maj_w = 1
    t_min_w = t_maj_w / 2

    # updates rcParams of the selected style with my preferred options for these plots. Feel free to change
    plt.rcParams.update({'font.family': 'arial', 'font.size': small_size, 'font.weight': 'normal',

                         'figure.titlesize': large_size, 'axes.titlesize': medium_size,
                         'axes.labelsize': small_size, 'legend.fontsize': small_size,

                         'text.color': 'black', 'axes.edgecolor': 'black', 'axes.linewidth': t_maj_w,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',

                         'xtick.major.size': t_maj_s, 'xtick.major.width': t_maj_w,
                         'xtick.minor.size': t_min_s, 'xtick.minor.width': t_min_w,
                         'ytick.major.size': t_maj_s, 'ytick.major.width': t_maj_w,
                         'ytick.minor.size': t_min_s, 'ytick.minor.width': t_min_w,
                         'xtick.labelsize': smaller_size, 'ytick.labelsize': smaller_size,

                         'xtick.color': 'black', 'ytick.color': 'black', 'ytick.labelcolor': 'black',
                         'xtick.direction': t_dir, 'ytick.direction': t_dir,

                         "xtick.bottom": True, "ytick.left": True,
                         'axes.spines.top': True, 'axes.spines.bottom': True, 'axes.spines.left': True,
                         'axes.spines.right': True,
                         'axes.grid': False,

                         'savefig.dpi': 1200, "figure.dpi": 100})


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
        'signal1_colour': "#37782C",
        'signal2_colour': "#64bb6a",
        'signal3_colour': "#9fd983"
    },
    2: {  # Taken from time_variation1
        'ax1_colour_matte': "#73B741",  # gr #73B741 dg #8C8E8D" # dg "#80BE53"
        'ax2_colour_matte': "#F77D6A",
        'signal1_colour': "#CD331B",
        'signal2_colour': "#B896B0",  # cy 3EB8A1
        'signal3_colour': "#377582"  # B79549
        # 37782C, #64BB6A, #9FD983
    },
    3: {  # Default Blender colours (previously called `bcolors` by me)
        'PURPLE': '\033[95m',
        'BLUE': '\033[94m',
        'GREEN': '\033[92m',
        'ORANGE': '\033[93m',
        'RED': '\033[91m',
        'ENDC': '\033[0m'  # Black
    }
    # ... add more colour schemes as needed
}
