#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""(One liner introducing this file schemes.py)

(
Leading paragraphs explaining file in more detail.
)

Attributes:
    (Here, place any module-scope constants users will import.)
    
Constants:
    MODULE_LEVEL_CONSTANT1 (int): A module-level constant.

Examples:
    (Here, place useful implementations of the contents of schemes.py). Note that leading symbol '>>>' includes the
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
        src/plotting/schemes.py
    Author
        Cameron Aidan McEleney < c.mceleney.1@research.gla.ac.uk >
    Created
        20 Jun 2025
    IDE
        PyCharm
        
.. _Google Python Style Guide:
   https://google.github.io/styleguide/pyguide.html
"""

# from __future__ import foo

# Standard library imports
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Optional, TypeAlias

# Third-party imports
from pint import UnitRegistry

# Local application imports
from attribute_defintions import (SimulationParametersContainer as SimParams,
                                  SimulationFlagsContainer as SimFlags)
from src.utils import PairHelper

# Module-level constants
UREG: UnitRegistry = UnitRegistry(auto_reduce_dimensions=True)
Q_: UnitRegistry.Quantity = UREG.Quantity

__all__ = ['PlotScheme', 'DefaultSchemes']


class AxLims:
    """Value pair often used for axis limits in the range :math:`[x_\text{lower}, x_\text{upper}]`.

    This class is a private helper for other utilities in this module.

    Using a tuple FORCES the user to consider BOTH limit when making updates.
    """

    __slots__ = ('lower', 'upper')

    def __init__(self, lower: Optional[Real] = None, upper: Optional[Real] = None):
        """TODO docstring."""
        self.lower = lower
        self.upper = upper

    @classmethod
    def from_raw(
            cls,
            raw: Real | Sequence[Optional[Real]],
            *,
            name: Optional[str] = None,
            allow_none: bool = True,
            cast: Callable[[Real], ...] = float
    ) -> "AxLims":
        """TODO docstring."""
        low, high = PairHelper.ensure_pair(raw,
                                           name=name if name is not None else cls.__name__,
                                           allow_none=allow_none,
                                           cast=cast)

        return cls(low, high)

    def __repr__(self):
        return f"{self.__class__.__name__}(lower={self.lower}, upper={self.upper})"

    def as_tuple(self) -> tuple[Real, Real]:
        return self.lower, self.upper


AXIS_TYPE: TypeAlias = 'AxLims' | Sequence[Optional[Real]]


@dataclass
class SubplotConfig:
    """Key parameters typically rescaled/edited during processes when generating subplots."""
    xlim: AXIS_TYPE
    ylim: AXIS_TYPE
    label: str
    line_height: Optional[float] = None

    # Catch-all for arguments to be passed on to AxLims
    ax_lims_kwargs: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        opts = self.ax_lims_kwargs.copy()  # Copying leads to their automatic deletion after __post_init__

        if not isinstance(self.xlim, AxLims):
            self.xlim = AxLims.from_raw(self.xlim, name=self.__class__.__name__, **opts)
        if not isinstance(self.ylim, AxLims):
            self.ylim = AxLims.from_raw(self.ylim, name=self.__class__.__name__, **opts)


@dataclass
class SignalConfig:
    """Key parameters specific to the main signal required to be tracked during generation of subplots."""
    xlim: AXIS_TYPE
    rescale: AXIS_TYPE
    extras: Optional[Sequence[Real | str]] = None

    # Catch-all for arguments to be passed on to AxLims
    ax_lims_kwargs: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        opts = self.ax_lims_kwargs.copy()  # Copying leads to their automatic deletion after __post_init__
        if not isinstance(self.xlim, AxLims):
            self.xlim = AxLims.from_raw(self.rescale, name=self.__class__.__name__, **opts)
        if not isinstance(self.rescale, AxLims):
            self.rescale = AxLims.from_raw(self.rescale, name=self.__class__.__name__, **opts)


@dataclass
class PlotScheme:

    #: maps names like `ax1`, `ax2` to `SubplotConfig`
    axes: dict[str, SubplotConfig] = field(default_factory=dict)

    #: maps names like `signal1`, `signal2` to `SignalConfig`
    signals: dict[str, SignalConfig] = field(default_factory=dict)

    def create_axis(
            self,
            name: str,
            *,
            xlim: AXIS_TYPE,
            ylim: AXIS_TYPE,
            label: str,
            line_height: Optional[float] = None,
            **ax_lims_kwargs

    ) -> SubplotConfig:
        """"""

        subplot = SubplotConfig(xlim, ylim, label, line_height, ax_lims_kwargs)
        self.add_axis(name, subplot)
        return subplot

    def create_signal(
            self,
            name,
            *,
            xlim: AXIS_TYPE,
            rescale: AXIS_TYPE = None,
            extras: Optional[Sequence[Real | str]] = None,
            **ax_lims_kwargs
    ) -> SignalConfig:
        """"""

        signal = SignalConfig(xlim, rescale, extras, ax_lims_kwargs)
        self.add_signal(name, signal)
        return signal

    def add_axis(self, name: str, cfg: SubplotConfig) -> None:
        self.axes[name] = cfg

    def add_signal(self, name: str, cfg: SignalConfig) -> None:
        self.signals[name] = cfg

    def __getitem__(self, key: str) -> SubplotConfig | SignalConfig:
        if key in self.axes:
            return self.axes[key]
        if key in self.signals:
            return self.signals[key]
        raise KeyError(f"No axis or signal named [{key}].")

    @property
    def valid_subplots(self) -> list[str]:
        return list(self.axes)

    @property
    def valid_signals(self) -> list[str]:
        return list(self.signals)


class DefaultSchemes:
    """TODO docstring."""

    def __init__(self, params: SimParams, flags: SimFlags):
        """TODO docstring."""
        
        self._params = params
        self._flags = flags
        
        self.basic1: PlotScheme = self.make_basic1(self._params, self._flags)
        self.basic2: PlotScheme = self.make_basic2(self._params, self._flags)

    @staticmethod
    def _process_dmi_flags(params: SimParams, flags: SimFlags):
        """These changes come directly from my old code.

        Found they lead to more insightful plots by re-centering the plots when DMI is enabled
        in simulations.
        """

        # Base cases
        signal_ax1_xlim = (params.num_sites_abc, params.driving_region_lhs - 1)
        signal_ax2_xlim = (params.driving_region_rhs + 1,
                           params.num_sites_total - params.num_sites_abc + 1)

        # Check all DMI flags before altering plot ranges to prevent later IndexErrors!
        if (isinstance(flags.is_dmi_only_within_map(), bool) and flags.is_dmi_only_within_map()
                and flags.has_dmi_map()
                and flags.has_dmi()):
            signal_ax1_xlim = (params.num_sites_abc + 1 - params.dmi_region_offset(),
                               params.driving_region_lhs - 1)

            signal_ax2_xlim = (params.driving_region_rhs + 1 + params.dmi_region_offset(),
                               params.num_sites_total - params.num_sites_abc + 1)

        return signal_ax1_xlim, signal_ax2_xlim

    def make_basic1(self, params: SimParams, flags: SimFlags) -> PlotScheme:
        """TODO docstring."""

        scheme = PlotScheme()

        # Main signal
        default_lattice = UREG('1 um')
        scheme.create_signal(name='signal',
                             xlim=(0, params.num_sites_total()),
                             rescale=(-0.5 * params.num_sites_total,
                                      0.5 * params.num_sites_total()),
                             extras=[(params.lattice_constant()
                                      if not (params.lattice_constant.dtype is None)
                                      else 1),
                                     default_lattice.m,
                                     default_lattice.u.__format__('~')],
                             ax_lims_kwargs=dict(cast=int))

        # Additional signals for specific subplots, indicated by 'name'
        signal_ax1_xlim, signal_ax2_xlim = self._process_dmi_flags(params, flags)

        scheme.create_signal(name='ax1',
                             xlim=signal_ax1_xlim,
                             ax_lims_kwargs=dict(cast=int))

        scheme.create_signal(name='ax2',
                             xlim=signal_ax2_xlim,
                             ax_lims_kwargs=dict(cast=int))

        # Axes for subplots
        line_height = 3.15e-3
        scheme.create_axis(name='ax1',
                           xlim=[None, None],
                           ylim=[None, None],
                           label='a',
                           line_height=line_height)

        scheme.create_axis(name='ax2',
                           xlim=(0.0, 0.25),
                           ylim=(1e-4, 1.0),
                           label='b',
                           line_height=line_height)

        scheme.create_axis(name='ax3',
                           xlim=(-0.25, 0.25),
                           ylim=(0, 40),
                           label='c',
                           line_height=line_height)

        return scheme

    def make_basic2(self, params: SimParams, flags: SimFlags) -> PlotScheme:
        """TODO docstring."""

        scheme = PlotScheme()

        # Main signal
        default_lattice = UREG('1 um')
        scheme.create_signal(name='signal',
                             xlim=(0, params.num_sites_total()),
                             rescale=(0.0, params.num_sites_total()),
                             extras=[(params.lattice_constant()
                                      if not (params.lattice_constant.dtype is None)
                                      else 1),
                                     default_lattice.m,
                                     default_lattice.u.__format__('~')])

        # Additional signals for specific subplots, indicated by 'name'
        signal_ax1_xlim, signal_ax2_xlim = self._process_dmi_flags(params, flags)

        scheme.create_signal(name='ax1',
                             xlim=signal_ax1_xlim,
                             ax_lims_kwargs=dict(cast=int))

        scheme.create_signal(name='ax2',
                             xlim=signal_ax2_xlim,
                             ax_lims_kwargs=dict(cast=int))

        # Axes for subplots
        line_height = 3.15e-3
        scheme.create_axis(name='ax1',
                           xlim=[None, None],
                           ylim=[None, None],
                           label='a',
                           line_height=line_height)

        scheme.create_axis(name='ax2',
                           xlim=(0.0, 0.40),
                           ylim=(1e-5, 1.0),
                           label='b',
                           line_height=line_height)

        scheme.create_axis(name='ax3',
                           xlim=(-0.12, 0.12),
                           ylim=(0, 40),
                           label='c',
                           line_height=line_height)

        return scheme
        
    def get(self, key):
        return getattr(self, key)
