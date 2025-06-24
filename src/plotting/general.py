#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""(One liner introducing this file general.py)

(
Leading paragraphs explaining file in more detail.
)

Attributes:
    (Here, place any module-scope constants users will import.)
    
Constants:
    MODULE_LEVEL_CONSTANT1 (int): A module-level constant.

Examples:
    (Here, place useful implementations of the contents of general.py). Note that leading symbol '>>>' includes the 
    code in doctests, while '$' does not.)::
        
        >>> bar = 1
        >>> foo = bar + 1

(
Trailing paragraphs summarising final details.
)

Todo:
    * (Optional section for module-wide tasks).
    * (Use format: 'YYMMDD/task_identifier - one-liner task description'
    
References:
    Style guide: `Google Python Style Guide`_

Notes:
    File version
        0.1.0
    Project
        SpinChains-PythonAnalysis
    Path
        src/plotting/general.py
    Author
        Cameron Aidan McEleney < c.mceleney.1@research.gla.ac.uk >
    Created
        18 Jun 2025
    IDE
        PyCharm
        
.. _Google Python Style Guide:
   https://google.github.io/styleguide/pyguide.html
"""
import typing
# from __future__ import foo

# Standard library imports
from abc import ABC, abstractmethod
from collections import Callable, namedtuple, Sequence
from dataclasses import dataclass, field
from itertools import cycle
from textwrap import dedent
from typing import Literal, Optional, TypeVar, TypedDict

# Third-party imports
import mpl_toolkits
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pint
from fontTools.t1Lib import font_dictionary_keys
from matplotlib import ticker
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from pint import UnitRegistry
import scipy as sp
from scipy import constants as C_

# Local application imports
from attribute_defintions import SimulationFlagsContainer, SimulationParametersContainer
from src.plotting.schemes import PlotScheme, DefaultSchemes
from figure_manager import colour_schemes as ColourSchemes

# Module-level constants
UREG_: UnitRegistry = UnitRegistry()
Q_ = UREG_.Quantity

__all__ = ['']

FontSizes = dict(
    large=20,
    medium=14,
    small=11,
    smaller=10,
    tiny=8,
    mini=7
)

CONTAINER = TypeVar(
    'CONTAINER',
    SimulationParametersContainer,
    SimulationFlagsContainer
)


class DataPipeLine:
    def __init__(
            self,
            time: NDArray,
            amplitude: NDArray,
            sites: NDArray,
            simulation_parameters: dict | SimulationParametersContainer,
            simulation_flags: dict | SimulationFlagsContainer
    ):
        """"""
        self.times: NDArray = time
        self.amplitudes: NDArray = amplitude
        self.site_indices: NDArray = sites

        # Instantiate empty containers; previously relied on user to provide these
        self.params: SimulationParametersContainer = SimulationParametersContainer()
        self.flags: SimulationFlagsContainer = SimulationFlagsContainer()

        # Update containers. TODO. Fixes required to actual container code (see report)
        self._update_container(self.params, simulation_parameters)
        self._update_container(self.flags, simulation_flags)
        self._correct_sim_presets()

    def _update_container(self, container: CONTAINER, props: dict | CONTAINER) -> None:
        """Update class instance containers used to generate figures from user simulation properties."""

        if isinstance(props, dict):
            container.update_with_dict(props)
            return

        if isinstance(props, type(container)):
            container.update_with_container(props)
            return

        raise NotImplementedError(f"Cannot update {type(container).__name__} from {type(props).__name__}")

    def _correct_sim_presets(self) -> None:
        """"""
        if self.params.lattice_constant() < 0:
            # Condition is met only when params_container doesn't contain a valid lattice constant.
            # This is true only for my old simulations due to their header layout
            self.params.lattice_constant.update(Q_('1nm').to_base_units().m)

        if self.params.exchange_dmi_constant() == 0.625:
            # Old batch of simulations used this precise value; multiplication required to fix scaling issue
            # from bad maths.
            self.params.exchange_dmi_constant *= 2

    def get(self, key):
        return getattr(self, key)


@dataclass
class FigureOptions:
    is_single: bool = False
    highlight_regions: bool = False
    is_interactive: bool = False
    is_for_publication: bool = False
    autosave: bool = True


class Formatter(ABC):
    def __init__(self, *, data: DataPipeLine, opts: FigureOptions):
        self._index = None
        self.data = data
        self.opts = opts

        self._default_layouts = DefaultSchemes(self.data.params, self.data.flags)

    def __call__(self, fig=None, ax=None):
        """Automatically produce a plot using the abstracted methods defined in Formatter.

        Order of execution:
        1. `make_plot()` - generates the figure and populates with data using `plt.plot().
        2. `format_figure()` - formats the figure and individual axes (subplots).
        3. `format_axis()` - common changes to all subplots of the figure.
        """
        if fig is not None and ax is not None:
            self.make_plot(fig)

        if fig is not None:
            self.format_figure(fig)

        if ax is not None:
            self.format_axis(ax)

    @property
    def index(self) -> int:
        return self._index

    @index.setter
    def index(self, val: Optional[int] = None):
        try:
            x = self.data.times[val]
        except IndexError:
            raise IndexError(f"Index {val} is not within {self.data.times.shape}")

        self._index = (-1
                       if val is None
                       else val)

    @abstractmethod
    def make_plot(self, fig: Figure, index: Optional[int] = None) -> None:
        """Primary commands to produce a plot.

        The contents of this method should be limited to the minimum, high-level requirements of generating plots
        This might include
        - creating a figure and axes,
        - plotting data on the axes,
        - saving the final figure.

        Each derived class should implement this method, and abstract away details such as:
            - setting different axis labels to several subplots -> `format_figure()`
            - setting the same axis label to all subplots -> `format_axis()`
            - formatting of the output file -> `format_output()`
            - customised method to draw shapes onto a single subplot -> create private method in concrete class.
        """
        ...

    def format_axes(self, axes: list[Axes]) -> Optional[list[Axes]]:
        """Define changes affect all axes within a figure.

        .
        """
        raise NotImplementedError('Derived must implement.')

    def format_figure(self, fig: Figure) -> Optional[Figure]:
        """
        Define changes affecting the entire figure or key properties of individual subplots

        Key properties might include: axis labels, titles, ... .
        """
        raise NotImplementedError('Derived must implement.')

    def format_output(self, output_path: str):
        raise NotImplementedError('Derived must implement.')

    @staticmethod
    def label_subplots_for_publication(
            frame_index: int,
            fig: Figure,
            position: Literal['left', 'right', 'split'],
    ) -> Figure:

        first_ax = fig.axes[0]
        first_ax.text(x=-0.04,
                      y=0.96,
                      s=r'$\times \mathcal{10}^{{\mathcal{-3}}}$',
                      ha='center',
                      va='center',
                      transform=fig.axes[0].transAxes)

        label_positions: list[tuple[float, float]] = []

        if position in ('left', 'split'):
            label_positions.append((0.12, 0.88))
        if position in ('right', 'split'):
            label_positions.append((0.88, 0.88))

        for i, (ax, (x, y)) in enumerate(zip(fig.axes, cycle(label_positions))):
            letter = chr(ord('a') + (i % 26))
            ax.text(x=x,
                    y=y,
                    s=f"({letter}) {frame_index: 2.3f} ns",
                    fontsize=6,
                    va='center',
                    ha='center',
                    transform=ax.transAxes)

        return fig

    @staticmethod
    def cleanup_subplots(axes: Optional[Axes] = None):
        axes = axes or plt.gcf().axes

        for ax in axes:
            ax.tick_params(axis='both',
                           which='both',
                           top=True,
                           right=True,
                           bottom=True,
                           left=True,
                           zorder=1.99)
            ax.grid(axis='both',
                    which='both',
                    visible=False)

            for tick_pos in (1, -2):
                last_tick_label = ax.get_xticklabels()[tick_pos]
                last_tick_label.set_visible(False)

                xtick_pos, ytick_pos = last_tick_label.get_position()
                ytick_pos += (-0.045
                              # If not central subplot in stack of three.
                              if not (len(axes) == 3 and ax == axes[2])
                              else 0.125)

                ax.text(x=xtick_pos,
                        y=ytick_pos,
                        s=str(last_tick_label.get_text()),
                        ha='left' if tick_pos == 1 else 'right',
                        va='top',
                        fontsize=FontSizes.get('smaller'),
                        transform=ax.get_xaxis_transform())


class BaseSpatial(Formatter):
    """Plots amplitude vs. time."""

    def __init__(self, data, opts, index: Optional[int] = None):
        super().__init__(data=data, opts=opts)
        self.selected_time: int = 0
        self.set_indices(index)

    def make_plot(self, fig: Figure, *, index: Optional[int] = None) -> None:

        ax = fig.axes[0]

        if index is not None:
            self.set_indices(index)

        time = self.data.get('time')
        amplitudes = self.data.get('amplitude')[self.index, :]

        ax.plot(time,
                amplitudes,
                ls='-',
                lw=0.75,
                color='#64bb6a',
                zorder=1.1,
                label="Signal"
                )

    def format_axis(self, ax):
        _, y_major_labels, _ = self._._choose_scaling(subplot_to_scale=ax)

        ax.set(xlabel="Site index, n$_{i}$",
               ylabel=f"m$_x$ (a.u. + {y_major_labels[1]} )",
               xlim=[0.0, self.data.params.num_sites_total()],
               ylim=[-self._._yaxis_lim, self._._yaxis_lim])

        if self.opts.highlight_regions:
            self._highlight_key_regions(ax)

        return ax

    def format_figure(self, fig, **kwargs):
        self.label_subplots_for_publication(self.selected_time, fig, kwargs.get('position'))

    def set_indices(self, idx) -> None:
        self.index = idx
        self.selected_time = self.data.get('times')[self.index]

    def _highlight_key_regions(self, ax: Axes):
        fields = ('lhs', 'driven', 'rhs')

        shape_position = namedtuple(
            typename='shape_positions',
            field_names=('lhs', 'driven', 'rhs'),
        )

        _bottom = ax.get_ylim()[0] * 2
        anchors = shape_position(lhs=(0, _bottom),
                                 driven=(self.data.params.driving_region_lhs() + self.data.params.num_sites_abc(),
                                         _bottom),
                                 rhs=(self.data.params.num_sites_total() - self.data.params.num_sites_abc(), _bottom))

        widths = shape_position(lhs=self.data.params.num_sites_abc(),
                                driven=self.data.params.driving_region_width(),
                                rhs=self.data.params.num_sites_abc())

        _height = 4 * ax.get_ylim()[1]
        heights = shape_position(_height, _height, _height)

        for key in fields:
            rect = Rectangle(xy=getattr(anchors, key),
                             width=getattr(widths, key),
                             height=getattr(heights, key),
                             lw=0,
                             alpha=0.75 if key == 'rhs' else 0.5,
                             facecolor='grey',
                             edgecolor=None)

            plt.gca().add_patch(rect)


class SpatialInstance(BaseSpatial):
    Flags = namedtuple(
        typename='Flags',
        field_names=('annotate_text',)
    )

    def __init__(self, data, opts, index: int = -1, *, has_annotated_text: bool = False):
        super().__init__(data=data, opts=opts, index=index)
        self.flags = self.Flags(annotate_text=has_annotated_text,)

    def make_plot(self, fig, index=None) -> None:
        super().make_plot(fig, index=index)
        ax = fig.gca()

        # TODO. Implement TickSetter and call here.
        y_major_labels = NotImplementedError()  # TODO. Implement ChooseScaling and call here.
        self.cleanup_subplots(ax)

        if self.flags.annotate_text:
            self._add_annotations(ax)

    def _add_annotations(self, ax: Axes) -> plt.Text:
        """"""
        p = self.data.params  # Alias due to frequent container accessing

        exchange_string = (
            f"Uniform Exc.: {p.exchange_heisenberg_min()} (T)"
            if p.exchange_heisenberg_min() == p.exchange_heisenberg_max()
            else f"J$_{{min}}$ = {p.exchange_heisenberg_min()} (T) |"
                 f"J$_{{max}}$ = {p.exchange_heisenberg_min()} (T)"
        )

        message = dedent(rf"""\
            H$_{{0}}$ = {p.bias_zeeman_static()} (T)
            | N = {p.num_sites_chain()}
            | $\alpha$ = {p.gilbert_chain(): 2.2e}
            H$_{{D1}}$ = {p.bias_zeeman_oscillating_1(): 2.2e} (T)
            | H$_{{D2}}$ = {p.bias_zeeman_oscillating_2(): 2.2e} (T)
            {exchange_string}
            """)

        return ax.text(x=0.05,
                       y=1.2,
                       s=message,
                       fontsize=FontSizes['small'],
                       ha='center',
                       va='center',
                       bbox=dict(boxstyle='round', facecolor='gainsboro', alpha=0.5),
                       transform=ax.transAxes)


@dataclass
class DatasetIndicesForFFTs:
    """Container for indices used to compute FFTs.

    Helpful when only wishing to plot particular regions of the dataset.
    """
    index_pairs: dict[str, tuple[int, int]] = field(default_factory=dict)
    plot_settings: dict[str, dict[str, typing.Any]] = field(default_factory=dict)

    def __post_init__(self):
        self._get_default_regions()


    def add_region(self, name: str, start: Sequence | int, end: Optional[int] = None) -> None:
        """.

        .
        """
        if isinstance(start, Sequence) and len(start) == 2:
            start, end = start
        else:
            raise ValueError(f"Expected a sequence of two elements or two separate integers for {name!r}, "
                             f"got {start!r} and {end!r}")

        self.index_pairs[name] = (start, end)

    def _get_default_regions(self, *, is_setup: bool = False) -> dict[str, tuple[int, int]]:
        """Add default regions to the dataset indices for FFTs.

        These values come from old code.
        """

        defaults, colours, styles = {}, {}, {}

        # Label wave (regions with indices smallest -> largest to match output plot legend labels).
        defaults['precursors'] = (12, 3345)
        colours['precursors'] = '#37782c'

        defaults['shockwave'] = (defaults['wave1'][1], 5079)
        colours['shockwave'] = '#64bb6a'

        defaults['steady state'] = (defaults['wave2'][1], 10000)
        colours['steady state'] = '#9fd983'

        for name in defaults.keys():
            styles[name] = '-'

        # Label precursors with indices largest -> smallest to match output plot legend labels.
        defaults['precursor_1'] = (2930, 3320)
        styles['precursor1'] = ':'

        defaults['precursor_2'] = (2350, 2570)
        styles['precursor2'] = '--'

        defaults['precursor_3'] = (1980, 2130)
        styles['precursor3'] = '-.'

        for name in ('precursor1', 'precursor2', 'precursor3'):
            colours[name] = '#37782c'

        # Initial setup
        if is_setup:
            self.index_pairs.update(defaults)

            for name, colour in defaults.keys():
                self.plot_settings[name] = dict(colour=colours[name], style=styles[name])

        return defaults


class FFTFormatter(Formatter):

    def __init__(self, site_index, data, opts):
        super().__init__(data=data, opts=opts)
        self.site_index = site_index

        self.fft_indices = DatasetIndicesForFFTs()

    def make_plot(self, fig: Figure, index: Optional[int] = None, signal_inset: bool = False) -> None:
        """.

        .
        """

        if index is None:
            raise ValueError('`index` must be provided to `FFTFormatter.make_plot()`')

        # TODO. Implement method that controls Figure creation in base-class make_plot()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 8))
        fig.subplots_adjust(wspace=0.1, hspace=-0.3)

        ax1_inset = ax1.inset_axes(bounds=(0.88, 0.47, 1.8, 0.7),
                                   loc='upper right',
                                   bbox_transform=ax1.figure.transFigure)

        # Plot upper subplot including inset
        duration, sample_rate = int(1.5e1), int(5e2)
        for delay in (0, 1):
            t, y = self.generate_sine_wave(15, sample_rate, duration, delay)

            freqs = sp.fft.rfftfreq(int(sample_rate * duration), 1 / sample_rate)
            dft = np.abs(sp.fft.rfft(y))

            colour = '#64bb6a' if delay == 0 else '#ffb55a'
            zorder = 1.3 if delay == 0 else 1.2

            ax1.plot(freqs,
                     dft,
                     lw=1.0,
                     color=colour,

                     marker='',
                     markerfacecolor='black',
                     markeredgecolor='black',

                     label=f'{delay}',
                     zorder=zorder)

            ax1_inset.plot(t, y, lw=0.5, color=colour, zorder=zorder)

        # Plot lower subplot containing FFT results
        for name, lims in self.fft_indices.index_pairs.items():
            dft_samples, dft = self.get_fft(self.data.amplitudes[lims[0]:lims[1], self.site_index], self.data.params)

            ax2.plot(dft_samples,
                     dft,
                     ls=self.fft_indices.plot_settings[name]['style'],
                     lw=1,
                     color=self.fft_indices.plot_settings[name]['colour'],

                     marker='',
                     markerfacecolor='black',
                     markeredgecolor='black',

                     label=name if 'precursor_' not in name else None)

        if signal_inset:
            self._add_zoomed_signal(ax1)
            self._add_annotations(ax1, name='upper')

        self._add_annotations(ax2, 'lower')

    def format_figure(self, fig: Figure) -> Optional[Figure]:
        """."""

        (ax1, ax2) = fig.axes
        inset = plt.axes()

        # Upper plot containing time-domain signal
        ax1.set(xlim=(5, 25),
                ylim=(1e0, 1e4))

        ax1.set_ylabel(f'{ax1.get_ylabel()}\n',
                       x=-10,
                       y=1)

        # TODO. Implement TickSetter here
        ax1.xaxis.set(major_locator=ticker.MultipleLocator(5),
                      minor_locator=ticker.MultipleLocator(1))

        # Upper plot inset
        inset.set(xlabel=r'Time, $t$ (ns)',
                  ylabel='Amplitude (a.u.)',
                  fontsize=FontSizes['small'],
                  xlim=(0, 2))

        inset.xaxis.labelpad = -0.5
        inset.yaxis.set_label_position('right')

        inset.yaxis.tick_right()
        inset.tick_params(axis='both', labelsize=FontSizes['small'])

        inset.patch.set_color("#f9f2e9")

        inset.xaxis.set(major_locator=ticker.MultipleLocator(1),
                        minor_locator=ticker.MultipleLocator(0.2))

        # Lower plot containing FFT results
        ax2.set(xlim=(0, 60),
                ylim=(1e-4, 1e1),
                yscale='log')

        # TODO. Implement TickSetter here
        # self._tick_setter(ax, 20, 5, 3, 4, is_fft_plot=True)

        ax2.legend(ncol=1,
                   fontsize=FontSizes['small'],

                   frameon=False,
                   fancybox=True,
                   facecolor=None,
                   edgecolor=None,

                   bbox_to_anchor=(0.7, 0.65),
                   axisbelow=False)

        self.cleanup_subplots(ax2)

    def format_axes(self, axes: list[Axes]) -> Optional[list[Axes]]:
        """."""

        for ax in axes:
            ax.set(xlabel=r'Frequency, $f$ (GHz)',
                   ylabel='Amplitude (a.u.)')

            ax.set_axisbelow(False)

    @staticmethod
    def _add_annotations(ax: Axes, name: Literal['lower', 'upper']) -> None:
        """.

        .
        """
        arrow_props = {
            'arrowstyle': '-|>',
            'connectionstyle': 'angle3,angleA=0,angleB=90',
            'color': 'black'
        }

        if name == 'lower':
            annotations = {
                'P1': dict(pos_arrow=(24.22, 0.029),
                           pos_text=(26.31, 0.231)),
                'P2': dict(pos_arrow=(36.48, 0.0096),
                           pos_text=(39.91, 0.13)),
                'P3': dict(pos_arrow=(52.00, 0.0045),
                           pos_text=(56.25, 0.075))
            }
        elif name == 'upper':
            annotations = {
                'P1': dict(pos_arrow=(2.228, -1.5e-4),
                           pos_text=(1.954, -1.5e-4)),
                'P2': dict(pos_arrow=(1.8, -8.48e-5),
                           pos_text=(1.407, -1.5e-4)),
                'P3': dict(pos_arrow=(1.65, 6e-5),
                           pos_text=(1.407, 1.5e-4))
            }
        else:
            raise AttributeError(f'Unknown subplot axis: {name}')

        for name, props in annotations.items():
            ax.annotate(text=name,
                        xy=props['pos_arrow'],
                        xytext=props['pos_text'],
                        arrowprops=arrow_props,
                        va='center',
                        ha='center',
                        fontsize=FontSizes['small'])

    def _add_zoomed_signal(self, ax: Axes) -> None:
        """.

        .
        """
        inset = ax.inset_axes(bounds=(0.14, 0.625, 2.4, 0.625),
                              loc="lower left",
                              xlim=(1.25, 2.5),
                              ylim=(-0.2e-3, 0.2e-3),
                              xticklabels=[],
                              yticklabels=[],
                              # alpha=0.3,
                              # facecolor='red',
                              bbox_transform=ax.figure.transFigure)

        inset.plot(self.data.times[:],
                   self.data.amplitudes[:, self.site_index],  # TODO Change to `self.index` from base class
                   color='#37782c')

        zoom_colour = '#f9f2e9'
        inset.patch.set_color(zoom_colour)

        ax.indicate_inset_zoom(inset,
                               lw=0.75,
                               alpha=0.9,
                               facecolor=zoom_colour,
                               edgecolor='black',
                               zorder=1.0)

    def get_fft(
            self,
            inputs: NDArray,
            params: SimulationParametersContainer,
            *,
            spacing: Optional[float] = None,
            fft_window: Optional[str | Callable] = None
    ) -> tuple:
        """Compute the discrete Fourier transform (DFT) of a 1-D signal.

        Args:
            inputs: Input signal to be transformed.
            params: ... .
            spacing: ... .
            fft_window: Window function to apply before computing the FFT.
                Defaults to False.

        Returns:
            DFT `samples` and `transforms` as a tuple.

            If ``inputs`` contain spatial information, then the returned `samples` represent the wavevectors :math:`k`
            in units of :math:`m^{-1}`. Otherwise, the returned `samples` represent the frequencies :math:`f` in units
            of :math:`Hz`. The returned `transforms` are the Fourier coefficients corresponding to the `samples`.

        :
        """

        data = inputs.copy()  # Copy to avoid modifying the original data
        dp_total = data.size
        dp_total_with_padding = sp.fft.next_fast_len(dp_total)

        # Determining the sample spacing provides the `bin width/size` for the FFT.
        bin_width = (spacing
                     if spacing is not None
                     else params.sim_time_max() / (params.num_dp_per_site() - 1))

        # Strangely, using `data *= window` here leads to future subplots breaking.
        data = data * self._get_fft_window(fft_window)

        dft = sp.fft.fft(x=data, n=dp_total_with_padding)
        dft_samples = sp.fft.fftfreq(n=dp_total_with_padding, d=bin_width)

        # Only want positive samples (e.g. wavevectors, frequencies) and their corresponding transforms.
        # For total indices (N) the Nyquist frequency is at N/2.
        # Always skip y[0] component as it's the DC component (zero frequency) of the signal's mean value.
        pos_indices = (slice(1, dp_total_with_padding // 2)  # `y = [1, N / 2 - 1]`
                       if dp_total_with_padding % 2 == 0
                       else slice(1, (dp_total_with_padding + 1) // 2))  # `y = [1, (N - 1) / 2]`

        return dft_samples[pos_indices], np.abs(dft[pos_indices])

    @staticmethod
    def _get_fft_window(
            datapoints: int,
            fft_window: str | Callable | None,
    ) -> sp.signal.windows | int:
        """Helper method.

        For valid windows, see `scipy.signal.windows`_ documentation.

        .. _scipy.signal.windows:
            https://docs.scipy.org/doc/scipy/reference/signal.windows.html

        """

        if fft_window is None:
            # Special case (for unit-tests and debugging) signifying no windowing.
            return 1

        if isinstance(fft_window, str):
            func = getattr(sp.signals.windows, fft_window.lower(), 'hann')
            return func(datapoints)

        if callable(fft_window) and hasattr(sp.signal, 'fft_window'):
            return sp.signal.fft_window(datapoints)

        raise TypeError("fft_window must be a valid string or callable, got {fft_window!r}")

    @staticmethod
    def generate_sine_wave(
            frequency: float,
            sample_rate: int,
            duration: int,
            delay_start: Optional[int] = None
    ) -> tuple[NDArray, NDArray]:
        """Generate a sine wave signal.

        Args:
            frequency: Frequency of the sine wave in Hz.
            sample_rate: Number of samples per second.
            duration: Duration of the signal in seconds.
            delay_start: Delay before the signal starts in seconds.

        Returns:
            Tuple containing time samples and corresponding sine wave values.
        """
        delay = int(sample_rate * delay_start if delay_start is not None else 1)

        t = np.linspace(0, duration, sample_rate * duration, endpoint=False)

        y_1 = np.zeros(delay)
        y_2 = np.sin(t[delay:] * UREG_.convert(frequency, 'cycle / s', 'Hz'))
        y_con = np.concatenate((y_1, y_2))

        return t, y_con


@dataclass
class TrackLinesLayers:
    peak: float
    zorder: float = 1.3


class SpatialFFT(BaseSpatial):
    def __init__(self, data, opts, index: int = -1):
        super().__init__(data=data, opts=opts, index=index)

        # Old code used default_layout `basic2` to strike a balance between clearly showing FFT of
        # fundamental eigenmodes, and their (possibly appearing) higher harmonics.
        self.layout_scheme: PlotScheme = self._default_layouts.basic2
        self._z_orders: dict[str, TrackLinesLayers] = {}

    def make_plot(self, fig, index=None) -> None:
        super().make_plot(fig, index=index)
        orig_ax = fig.axes[0]

        gs = fig.add_gridspec(3, 3)

        fig.axes.clear()
        fig.add_subplot(gs[0, :], label='base')

        orig_ax.set_subplotspec(gs[0, :])
        fig.axes.append(orig_ax)

        axs = []
        for row in (1, 2):
            ax = fig.add_subplot(gs[row, :])
            axs.append(ax)

        if fig.axes != [orig_ax] + axs:
            raise ValueError()

        if self.opts.is_interactive:
            # TODO. Link `FigureManager` in here
            figure_manager = None

        plt.close(fig)

    def _process_signal(
            self,
            fig: Figure,
            ax_name: str = 'ax',
            row_index: int = -1,
            signal_index: int = 1,
            *,
            has_default_colours: bool = True,
            colour_scheme_name: int = 0,
            is_wavevector_negative: bool = False
    ) -> None:

        if self.layout_scheme.axes[ax_name].xlim.lower == self.layout_scheme.axes[ax_name].xlim.upper:
            # Special case to signify this subplot shouldn't be post-processed.
            return

        # TODO. Create new _fft_data method
        cyclic_k, fourier_transform = (), ()

        angular_k = Q_(cyclic_k, 'cycle').to('').magnitude
        if is_wavevector_negative:
            angular_k *= -1

        # Constant terms
        linear_freq = (
                (self.data.params.exchange_heisenberg_max
                 * self.data.params.lattice_constant() ** 2  # Required for correctly scaled units
                 * angular_k ** 2)
                + self.data.params.bias_zeeman_static()
        )

        if self.data.flags.has_dmi_map:
            linear_freq += (
                    self.data.params.lattice_constant()  # Required for correctly scaled units
                    * self.data.params.exchange_dmi_constant()
                    * angular_k
            )

        # Convert all units to GHz for better readability on plots
        linear_freq *= UREG_.convert(self.data.params.gyro_mag(),
                                     "1 / (T s)",
                                     "cycle GHz / T")

        # Now return to original signs
        angular_k = Q_(cyclic_k, '1/m').to('1/nm').magnitude
        if is_wavevector_negative:
            angular_k *= -1

        self._calculate_z_order(ax_name, angular_k, fourier_transform)

        # Plotting time
        if self.layout_scheme.axes[ax_name].xlim.lower > self.layout_scheme.axes[ax_name].xlim.upper:
            raise ValueError(f"Condition would lead to FFT returning invalid 'n_padded_total < 0'."
                             f"Either reverse `xlim.lower` and `xlim.upper` or set new values.")

        if (not has_default_colours
                and colour_scheme_name not in ColourSchemes.keys()):
            raise AttributeError(f"Invalid colour scheme {colour_scheme_name} provided for signal processing.")

        signal_colour = (ColourSchemes[colour_scheme_name]['ax1_colour_matte']
                         if has_default_colours
                         else ColourSchemes[colour_scheme_name][f'signal{signal_index}_colour'])

        xlims = self.layout_scheme.signals[ax_name].xlim
        data_pairing = [
            (np.arange(xlims.lower, xlims.upper),
             self.data.amplitudes[row_index, xlims.lower:xlims.upper]),
            (angular_k, fourier_transform),
        ]

        # Onto the plotting
        axes = fig.axes

        for i, (x, y) in enumerate(data_pairing):
            axes[i].plot(
                x, y,
                ls='-',
                lw=1.5,
                color=signal_colour,
                label=f"Segment {signal_index}",
                zorder=self._z_orders[ax_name].zorder,

                marker='' if i != 0 else None,
                markerfacecolor='black',
                markeredgecolor='black',
            )

        if signal_index == 4:
            # Add dispersion relation to imshow
            axes[2].plot(self.data.times,
                         self.data.amplitudes[self.data.params.num_sites_abc():self.data.params.num_sites_total()
                                              - self.data.params.num_sites_abc()])

    def _calculate_z_order(
            self,
            ax_name: str,
            wavenumbers,
            fourier_transform: NDArray,
            use_default: bool = True
    ):
        """.


        """
        cutoff = self.layout_scheme['ax2'].xlim.upper
        closest_idx = np.argmin(np.abs(wavenumbers - cutoff))
        peak = float(np.max(fourier_transform[closest_idx:]))

        if use_default or not self._z_orders:
            zo = TrackLinesLayers(peak=peak)
            if not self._z_orders:
                self._z_orders[ax_name] = zo
            return zo

        existing = list(self._z_orders.values())
        min_ = min(existing, key=lambda z: z.peak)
        max_ = max(existing, key=lambda z: z.peak)

        if peak < min_.peak:
            new_z = min_.zorder + 0.01
        elif peak > max_.peak:
            new_z = max_.zorder - 0.01
        else:
            new_z = self._z_orders['ax1'].zorder

        z_o = TrackLinesLayers(peak=peak, zorder=new_z)
        self._z_orders[ax_name] = z_o

        return z_o


    def format_figure(self, fig, **kwargs):
        fig.subplots_adjust(wspace=1,
                            hspace=0.4,
                            bottom=0.2)

        # Will be three in total for this layout design.
        ax1, ax2, ax3, _ = fig.axes

        # Upper subplot
        ax1.set(yscale='linear')

        # Middle subplot
        ax2.set(xlabel=r"Wavevector, $k$ (nm$^{-1}$)",
                ylabel="Intensity (a.u.)",
                xlim=self.layout_scheme.axes['ax2'].xlim,
                ylim=self.layout_scheme.axes['ax2'].ylim,
                yscale='log')

        # Bottom subplot
        ax3.set(ylabel=r"Frequency, $f$ (GHz)",
                xlim=self.layout_scheme.axes['ax3'].xlim,
                ylim=self.layout_scheme.axes['ax3'].ylim,
                yscale='linear')

        ax3.tick_params(pad=2,
                        labeltop=True,
                        labelbottom=False,
                        labelsize=FontSizes['smaller'])

        ax3.invert_yaxis()

        self._create_colourbar(fig, ax3)

    def _create_colourbar(self, fig: Figure, ax: Optional[Axes] = None) -> None:
        """Create in-place axis with colorbar.

        If `ax == None`, method adds colourbar to final (bottom) subplot.
        """

        try:
            ax = fig.get_axes()
        except IndexError:
            raise IndexError(f"Figure {fig} does not contain any axes.")
        else:
            ax = ax[-1]

        norm = mpl.colors.Normalize(vmin=0, vmax=1)

        # Create a ScalarMappable for the colour mapping
        scalar_map = mpl.cm.ScalarMappable(cmap='magma_r',
                                           norm=norm)

        # Add a colourbar to the subplot using this ScalarMappable
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        cax3 = divider.append_axes("bottom",
                                   size="7.5%",
                                   pad=0.0)

        # Set the ticks at the top and bottom using normalized values
        ax3_cbar = fig.colorbar(scalar_map,
                                ax=ax,
                                cax=cax3,
                                location='bottom',
                                orientation='horizontal',
                                shrink=1.0)

        # Customise colourbar
        ax3_cbar.ax.tick_params(axis='x',
                                top=False, bottom=True,
                                pad=3.5)

        ax3_cbar.set_label('Intensity (a.u.)',
                           loc='center',
                           labelpad=-5)

        ax3_cbar.set_ticks(ticks=[norm.vmin + 0.03, norm.vmax - 0.035],
                           labels=['Min', 'Max'],
                           # labels=[str(norm.vmin), str(norm.vmax)]
                           )

    def format_output(self, path_to_output_file: str):
        """"""

        try:
            open(path_to_output_file)
        except FileNotFoundError:
            print("Unable to append data to file. Continuing...")

        data = [path_to_output_file]

        def calculate_distance(val: int | float, from_left: bool):
            sign = -1 if from_left else 1
            # TODO. Survey others to decide if PINT is more readable when entering S.I. units than normal literals
            return val + sign * round(Q_('1um').to('m').m / Q_('1nm').to('m').m)

        data.append(str(self.data.amplitudes[self.index,
                                             calculate_distance(self.data.params.driving_region_lhs(),
                                                                True)]))

        data.append(str(self.data.amplitudes[self.index,
                                             calculate_distance(self.data.params.driving_region_rhs(),
                                                                False)]))

        with open(path_to_output_file, 'a') as file_:
            file_.write(",".join(data) + "\n")
            file_.close()

        print(f"Data written to file:\n"
              f"\t- Before: {data[1]}"
              f"\t- After: {data[2]}")


class FigureBuilder:
    def __init__(
            self,
            data: DataPipeLine,
            formatters: list[Formatter],
            handlers: Optional[list] = None):
        self.data = data
        self.formatters = (formatters or None)
        self.handlers = handlers or []

        self.fig = Figure()

        self._post_init()

    def _post_init(self):
        """Custom, unique changes."""

        # Important! This offset must be manually controlled; arises due to errors in certain sim. datasets.
        self.data.params.driving_region_lhs += 300
        self.data.params.driving_region_rhs += 300

    def build(self) -> Figure:

        try:
            if self.formatters is None:
                raise TypeError
        except TypeError:
            raise TypeError('Must provide at least one formatter to build a figure.')

        ax = self.fig.add_subplot(111)

        for fmt in self.formatters:
            fmt(self.fig, ax)

        for handler_ in self.handlers:
            handler_(self.fig)

        return self.fig


class PaperFigures:
    def __init__(
            self,
            *,
            time: NDArray,
            amplitude: NDArray,
            params: dict | SimulationParametersContainer,
            flags: dict | SimulationFlagsContainer,
            site_indices: NDArray,
            output_path: str,
            opts: FigureOptions = FigureOptions()
    ):
        self.data = DataPipeLine(time, amplitude, site_indices, params, flags, )
        self.output = output_path
        self.opts = opts

        self.fmts: list[Formatter] = []
        self.handlers: list = []

    def make_spatial_instance(
            self,
            index: int,
            take_fft: bool = False,
            *,
            has_annotated_text: bool = False,
            has_single_figure: bool = True
    ):
        builder = FigureBuilder(self.data, self.fmts, self.handlers)

        # Handle figure
        # builder.fig = Figure()

        if take_fft:
            builder.fig.set_size_inches(4.5, 6.0)
            builder.formatters.append(SpatialFFT(self.data,
                                      self.opts,
                                      index))
        else:
            builder.fig.set_size_inches(4.4, 2.2) if has_single_figure else (4.4, 4.4)
            builder.formatters.append(SpatialInstance(self.data,
                                      self.opts,
                                      index,
                                      has_annotated_text=has_annotated_text))

        builder.handlers.extend([
            # ClickHandler()
        ])

        fig = builder.build()

        if self.opts.autosave:
            fig.savefig(self.output,
                        dpi=1200 if take_fft else 300)

        # fig.builder.handlers()

        return fig
