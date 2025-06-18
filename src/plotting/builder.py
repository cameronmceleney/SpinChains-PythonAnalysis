#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""(One liner introducing this file builder.py)

(
Leading paragraphs explaining file in more detail.
)

Attributes:
    (Here, place any module-scope constants users will import.)
    
Constants:
    MODULE_LEVEL_CONSTANT1 (int): A module-level constant.

Examples:
    (Here, place useful implementations of the contents of builder.py). Note that leading symbol '>>>' includes the 
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
        src/plotting/builder.py
    Author
        Cameron Aidan McEleney < c.mceleney.1@research.gla.ac.uk >
    Created
        16 Jun 2025
    IDE
        PyCharm
        
.. _Google Python Style Guide:
   https://google.github.io/styleguide/pyguide.html
"""

__all__ = ['PaperFigures']

import typing
# Standard library imports
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Optional

# Third-party imports
from matplotlib import (pyplot as plt,
                        ticker as mpl_tick)
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray
from pint import UnitRegistry
from scipy import constants

# Local application imports
from attribute_defintions import SimulationFlagsContainer, SimulationParametersContainer
from src.plotting.spatial import ClickHandler, SpatialPlot

# Module-level constants
UREG_: UnitRegistry = UnitRegistry()
Q_ = UREG_.Quantity


@dataclass
class PlotOptions:
    is_single_figure: bool = False
    has_highlighted_regions: bool = False
    is_interactive: bool = False

    FontSizes = namedtuple(
        'FontSizes',
        ('large', 'medium', 'small', 'smaller', 'tiny', 'mini'),
        defaults=(20, 14, 11, 10, 8, 7)
    )

class _BuilderHelper(PlotOptions):
    fig = None
    axis = None

    def set_figure(self, figure: Figure):
        self.fig = figure

    def set_axis(self, axes: Axes):
        self.axis = axes

    def reset_axes(self):
        self.axis.clear()
        self.axis.set_aspect('auto')


class PlotFormatter(_BuilderHelper):
    """"""
    def __init__(
            self,
            **plot_options,
    ):
        """"""
        super().__init__(
            plot_options
        )
        self.builder = None

    def __call__(self):
        raise NotImplementedError("Derived must override.")

    def assign_builder(self, builder: Any):
        self.builder = builder

    def axes_from_kwargs(self, source: Any, has_data: bool = True):
        try:
            source.get('axes')
        except AttributeError:
            pass
        else:
            if isinstance(source['axes'], plt.Axes):
                self.builder.axes = source.get('axes')

        if has_data:
            self.reset_axes()



class DataFormatter(
    SimulationFlagsContainer,
    SimulationParametersContainer
):
    def __init__(self):
        super().__init__()
        self.correct_sim_presets()

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

    def update_containers(self, sim_params, sim_flags):
        self._update_params(sim_params)
        self._update_flags(sim_flags)


class FigureBuilder(PlotFormatter):
    pass


class PaperFigures(
    SimulationFlagsContainer,
    SimulationParametersContainer
):

    def __init__(
            self,
            time_data: NDArray[tuple[Any, ...], np.float64],
            amplitude_data: NDArray[tuple[Any, ...], np.float64],
            params_dict,
            flags_dict,
            site_indices: list | NDArray,
            output_filepath: str,
            params_container: Optional[SimulationParametersContainer] = None,
            flags_container: Optional[SimulationFlagsContainer] = None,
            *,
            is_plot_interactive: bool = False,
            **kwargs
    ):
        super().__init__()

        # Data/paths read-in from `data_analysis.py`
        self.data_time = time_data
        self.data_amplitude = amplitude_data
        self.sites = site_indices
        self.output_path = output_filepath

        # Load simulation material parameters/physical constants
        if params_container is None:
            self.update_with_dict(params_dict)
        else:
            self.update_with_container(params_container)

        # Load simulation control-flags
        if flags_container is None:
            self.update_with_dict(flags_dict)
        else:
            self.update_with_container(flags_container)

        # Plotting
        self.is_interactive = is_plot_interactive
        self.fig: Figure
        self.axes: Axes
        self._yaxis_lim = 1.3
        self._yaxis_lim_fix = 8e-3
        self.track_zorder = [[], []]

        _FontSizes = namedtuple(
            'FontSizes',
            ('large', 'medium', 'small', 'smaller', 'tiny', 'mini'),
            defaults=(20, 14, 11, 10, 8, 7)
        )
        self._font_sizes: _FontSizes = _FontSizes()

        # Helper objects for plotting
        self.spatial = SpatialPlot(self)

        self._post_init()

    def _post_init(self):
        if self.lattice_constant() < 0:
            # Condition is met only when params_container doesn't contain a valid lattice constant.
            # This is true only for my old simulations due to their header layout
            self.lattice_constant.update(Q_('1nm').to_base_units().m)

        if self.exchange_dmi_constant() == 0.625:
            # Old batch of simulations used this precise value; multiplication required to fix scaling issue
            # from bad maths.
            self.exchange_dmi_constant *= 2

        _FontSizes = namedtuple(
            'FontSizes',
            ('large', 'medium', 'small', 'smaller', 'tiny', 'mini'),
            defaults=(20, 14, 11, 10, 8, 7)
        )


