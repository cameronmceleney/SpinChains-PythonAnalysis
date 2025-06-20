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
from typing import Optional, TypeAlias

# Third-party imports
from pint import UnitRegistry

# Local application imports
from attribute_defintions import SimulationParametersContainer as SimParams
from src.utils import PairHelper

# Module-level constants
UREG: UnitRegistry = UnitRegistry(auto_reduce_dimensions=True)
Q_: UnitRegistry.Quantity = UREG.Quantity

__all__ = ['']


class AxLims:
    """Value pair often used for axis limits in the range :math:`[x_\text{lower}, x_\text{upper}]`.

    This class is a private helper for other utilities in this module.

    Using a tuple FORCES the user to consider BOTH limit when making updates.
    """

    __slots__ = ('lower', 'upper')

    def __init__(
            self,
            lower: Real | Sequence[Optional[Real]] | None,
            upper: Optional[Real] = None,
            *,
            allow_none: bool = False,
            cast: Callable[[Real], ...] = float
    ):
        seq = (lower
               if upper is None and isinstance(lower, Sequence)
               else (lower, upper))

        low, high = PairHelper.ensure_pair(seq,
                                           name=self.__class__.__name__,
                                           allow_none=allow_none,
                                           cast=cast)
        self.lower = low
        self.upper = high

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

    def __post_init__(self):
        if not isinstance(self.xlim, AxLims):
            self.xlim = AxLims(self.xlim)
        if not isinstance(self.ylim, AxLims):
            self.ylim = AxLims(self.ylim)


@dataclass
class SignalConfig:
    """Key parameters specific to the main signal required to be tracked during generation of subplots."""
    xlim: AXIS_TYPE
    rescale: AXIS_TYPE
    extras: Optional[Sequence[Real | str]] = None

    def __post_init__(self):
        if not isinstance(self.xlim, AxLims):
            self.xlim = AxLims(self.xlim)
        if not isinstance(self.rescale, AxLims):
            self.rescale = AxLims(self.rescale)


@dataclass
class PlotScheme:

    #: maps names like `ax1`, `ax2` to `SubplotConfig`
    axes: dict[str, SubplotConfig] = field(default_factory=dict)

    #: maps names like `signal1`, `signal2` to `SignalConfig`
    signals: dict[str, SignalConfig] = field(default_factory=dict)

    def create_subplot(
            self,
            name: str,
            *,
            xlim: AXIS_TYPE,
            ylim: AXIS_TYPE,
            label: str,
            line_height: Optional[float] = None
    ) -> SubplotConfig:
        """"""

        subplot = SubplotConfig(xlim, ylim, label, line_height)
        self.append_subplot_config(name, subplot)
        return subplot

    def create_signal(
            self,
            name,
            *,
            xlim: AXIS_TYPE,
            rescale: AXIS_TYPE = None,
            extras: Optional[Sequence[Real | str]] = None
    ) -> SignalConfig:
        """"""

        signal = SignalConfig(xlim, rescale, extras)
        self.append_signal_config(name, signal)
        return signal

    def append_subplot_config(self, name: str, cfg: SubplotConfig) -> None:
        self.axes[name] = cfg

    def append_signal_config(self, name: str, cfg: SignalConfig) -> None:
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

    def __init__(self, params: SimParams):
        """TODO docstring."""

        self._basic1: PlotScheme = self._make_basic1(params)

    def _make_basic1(self, params: SimParams) -> PlotScheme:
        """TODO docstring."""

        scheme = PlotScheme()

        # Main signal
        default_lattice = UREG('1 um')
        scheme.create_signal(
            name='signal',
            xlim=(0, params.num_sites_total()),
            rescale=(0.0, 0.5 * params.num_sites_total()),
            extras=[(params.lattice_constant()
                     if not (params.lattice_constant.dtype is None)
                     else 1),
                    default_lattice.m,
                    default_lattice.u.__format__('~')]
        )

        # Additional Signals
        scheme.create_signal(
            name='ax1',
            xlim=()
        )


        line_height = 3.15e-3
        self._basic1.create_subplot(
            name='ax1',
            xlim=[None, None],
            ylim=[None, None],
            label='a',
            line_height=line_height
        )
        self._basic1.create_subplot(
            name='ax2',
            xlim=(0.0, 0.25),
            ylim=(1e-4, 1.0),
            label='b',
            line_height=line_height
        )

        self._basic1.create_subplot(
            name='ax3',
            xlim=(-0.25, 0.25),
            ylim=(0, 40),
            label='c',
            line_height=line_height
        )

        return scheme

    def get(self, key):
        return getattr(self, key)
