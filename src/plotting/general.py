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
from functools import cached_property
from itertools import cycle
from typing import Any, Literal

# Third-party imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import numpy as np
from pint import UnitRegistry

# Local application imports
from attribute_defintions import SimulationFlagsContainer, SimulationParametersContainer

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


class DataPipeLine(
    SimulationFlagsContainer,
    SimulationParametersContainer
):
    def __init__(
            self,
            time,
            amplitude,
            sim_parms: dict | SimulationParametersContainer,
            sim_flags: dict | SimulationFlagsContainer,
            sites
    ):
        """"""
        self.time = time
        self.amplitude = amplitude
        self._update_containers(sim_parms, sim_flags)
        self._sites = sites

    def correct_sim_presets(self):
        if self.lattice_constant() < 0:
            # Condition is met only when params_container doesn't contain a valid lattice constant.
            # This is true only for my old simulations due to their header layout
            self.lattice_constant.update(Q_('1nm').to_base_units().m)

        if self.exchange_dmi_constant() == 0.625:
            # Old batch of simulations used this precise value; multiplication required to fix scaling issue
            # from bad maths.
            self.exchange_dmi_constant *= 2

    def _update_params(self, data: dict | SimulationParametersContainer) -> None:
        """Load additional simulation parameters into base container."""
        if isinstance(data, dict):
            self.update_with_dict(data)
            return

        if isinstance(data, SimulationParametersContainer):
            self.update_with_container(data)
            return

        raise NotImplementedError(f"Cannot update params using an object of type {type(data)}")

    def _update_flags(self, data: dict | SimulationFlagsContainer) -> None:
        """Load additional simulation flags into base container."""
        if isinstance(data, dict):
            self.update_with_dict(data)
            return

        if isinstance(data, SimulationParametersContainer):
            self.update_with_container(data)
            return

        raise NotImplementedError(f"Cannot update params using an object of type {type(data)}")

    def _update_containers(self, sim_params, sim_flags):
        self._update_params(sim_params)
        self._update_flags(sim_flags)

    def get(self, key):
        return getattr(self, key)


@dataclass
class FigureOptions:
    is_single: bool = False
    highlight_regions: bool = False
    is_interactive: bool = False
    is_for_publication: bool = False


class Formatter(ABC):
    def __init__(self, opts: FigureOptions):
        self.opts = opts

    @abstractmethod
    def format(self, fig: Figure, ax: Axes, data_pipeline: DataPipeLine):
        ...

    def __call__(self, fig, ax, data):
        return self.format(fig, ax, data)


class AxisFormatter(Formatter):
    def format(self, fig, ax, data):
        ax.set(xlabel="time",
               ylabel="amplitude")
        ax.grid = False


class BaseSpatial(Formatter):
    """Plots amplitude vs. time."""

    def __init__(self, opts: FigureOptions, index: int = None):
        super().__init__(opts)
        self.index = index
        self.frame_index = None

    def format(self, fig: Figure, ax: Axes, data_pipeline: DataPipeLine):

        time_data = data_pipeline.get('time')
        self.frame_index = time_data[self.index]
        amplitude_data = data_pipeline.get('amplitude')

        amplitudes_at_time = (amplitude_data[self.index, :]
                              if self.index is not None
                              else amplitude_data[-1, :])

        ax.plot(time_data,
                amplitudes_at_time,
                ls='-',
                lw=0.75,
                color='#64bb6a',
                zorder=1.1,
                label="Signal"
                )

        _, y_major_labels, _ = self._._choose_scaling(subplot_to_scale=self._.axes)

        ax.set(xlabel="Site index, n$_{i}$",
               ylabel=f"m$_x$ (a.u. + {y_major_labels[1]} )",
               xlim=[0.0, self._.num_sites_total()],
               ylim=[-self._._yaxis_lim, self._._yaxis_lim])

        if self.opts.highlight_regions:
            self._highlight_key_regions(fig, ax)

    def format_subplots_for_publication(
            self,
            fig: Figure,
            position: Literal['left', 'right', 'split'],
    ):

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
                    s=f"({letter}) {self.frame_index: 2.3f} ns",
                    fontsize=6,
                    va='center',
                    ha='center',
                    transform=ax.transAxes)

    def _highlight_key_regions(self, fig: Figure, ax: Axes):
        fields = ('lhs', 'driven', 'rhs')

        shape_position = namedtuple(
            typename='shape_positions',
            field_names=('lhs', 'driven', 'rhs'),
        )

        _bottom = ax.get_ylim()[0] * 2
        anchors = shape_position(lhs=(0, _bottom),
                                 driven=(self._.driving_region_lhs() + self._.num_sites_abc(), _bottom),
                                 rhs=(self._.num_sites_total() - self._.num_sites_abc(), _bottom))

        widths = shape_position(lhs=self._.num_sites_abc(),
                                driven=self._.driving_region_width(),
                                rhs=self._.num_sites_abc())

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


class FigureBuilder:
    def __init__(self, data: DataPipeLine, formatters: list[Formatter]):
        self.data = data
        self.formatters = formatters

    def build(self) -> Figure:
        fig = Figure()
        ax = fig.add_subplot(111)

        for fmt in self.formatters:
            fmt(fig, ax, self.data)

        return fig


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
        self.opts = opts

        self.fmts: list[Formatter] = [AxisFormatter(opts)]
        self.output = output_path

    def make_spatial_plot(
            self,
            frame_index: int,
            save: bool = False
    ):

        fmt = BaseSpatial(
            self.opts,
            frame_index
        )

        builder = FigureBuilder(self.data, [fmt])
        fig = builder.build()

        if save:
            fig.savefig(self.output, dpi=300)

        return fig