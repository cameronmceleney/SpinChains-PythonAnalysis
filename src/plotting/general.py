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

# from __future__ import foo

# Standard library imports
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from itertools import cycle
from textwrap import dedent
from typing import Literal, Optional, TypeVar

# Third-party imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from pint import UnitRegistry

# Local application imports
from attribute_defintions import SimulationFlagsContainer, SimulationParametersContainer
from src.plotting.utils import PlotScheme

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

    def __call__(self, fig=None, ax=None):
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
        ...

    def format_axis(self, ax: Axes) -> Optional[Axes]:
        raise NotImplementedError('Derived must implement.')

    def format_figure(self, fig: Figure) -> Optional[Figure]:
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


class SpatialFFT(BaseSpatial):
    def __init__(self, data, opts, index: int = -1):
        super().__init__(data=data, opts=opts, index=index)

        # Important! This offset must be manually controlled; arises due to errors in certain sim. datasets.
        self.data.params.driving_region_lhs += 300
        self.data.params.driving_region_rhs += 300

    def make_plot(self, fig, index=None) -> None:
        super().make_plot(fig, index=index)
        ax = fig.get_axes()

        num_rows, num_cols = 3, 3
        for i in range(0, 3):
            subplot = plt.subplot2grid(fig=fig,
                                       shape=(num_rows, num_cols),
                                       loc=(i, 0),
                                       rowspan=1,
                                       colspan=num_cols,)
            ax.append(subplot)

        if self.opts.is_interactive:
            # TODO. Link `FigureManager` in here
            figure_manager = None

        plt.close(fig)

    def format_figure(self, fig, **kwargs):
        fig.subplots_adjust(wspace=1,
                            hspace=0.4,
                            bottom=0.2)

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
            time,
            amplitude,
            params,
            flags,
            site_indices,
            output_path,
            opts: FigureOptions = FigureOptions()
    ):
        self.data = DataPipeLine(time, amplitude, params, flags, site_indices)
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
