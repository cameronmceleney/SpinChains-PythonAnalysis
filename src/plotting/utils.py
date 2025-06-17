#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""(One liner introducing this file utils.py)

(
Leading paragraphs explaining file in more detail.
)

Attributes:
    (Here, place any module-scope constants users will import.)
    
Constants:
    MODULE_LEVEL_CONSTANT1 (int): A module-level constant.

Examples:
    (Here, place useful implementations of the contents of utils.py). Note that leading symbol '>>>' includes the 
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
        src/plotting/utils.py
    Author
        Cameron Aidan McEleney < c.mceleney.1@research.gla.ac.uk >
    Created
        16 Jun 2025
    IDE
        PyCharm
        
.. _Google Python Style Guide:
   https://google.github.io/styleguide/pyguide.html
"""

# from __future__ import foo

__all__ = ['SignalConfig', 'SubplotConfig']

# Standard library imports
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Literal,NamedTuple, Optional

# Third-party imports
from matplotlib.axes import Axes
from matplotlib import (pyplot as plt,
                        ticker as mtick)
from matplotlib.ticker import (AutoMinorLocator, FormatStrFormatter, LogLocator,
                               MaxNLocator, MultipleLocator, NullFormatter)
import numpy as np

# Local application imports

# Module-level constants


class AxLims(NamedTuple):
    """Value pair often used for axis limits in the range :math:`[x_\text{lower}, x_\text{upper}]`.

    This class is a private helper for other utilities in this module.
    """
    lower: float
    upper: float


@dataclass(frozen=True)
class SubplotConfig:
    """Key parameters typically rescaled/edited during processes when generating subplots."""
    xlim: AxLims
    ylim: AxLims
    label: str
    line_height: Optional[float] = None


@dataclass
class SignalConfig:
    """Key parameters specific to the main signal required to be tracked during generation of subplots."""
    xlim: AxLims
    rescale: Optional[list[float]] = None
    extras: Optional[list[float | int | str]] = None


@dataclass
class PlotScheme:

    #: maps names like `ax1`, `ax2` to `SubplotConfig`
    axes: dict[str, SubplotConfig] = field(default_factory=dict)

    #: maps names like `signal1`, `signal2` to `SignalConfig`
    signals: dict[str, SignalConfig] = field(default_factory=dict)

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
    def valid_axes(self) -> list[str]:
        return list(self.axes)

    @property
    def valid_signals(self) -> list[str]:
        return list(self.signals)


@dataclass
class AxisLocator:
    major: float
    minor: float

    def rescale(self, factor: float, locator: Literal['major', 'minor', 'both']) -> None:
        if locator in ('major', 'both'):
            self.major *= factor
        if locator in ('minor', 'both'):
            self.minor *= factor


class TickCustomisation:
    def __init__(self, owner):
        self._ = owner

    def set_ticks(
            self,
            plot: Optional[Axes],
            xaxis_locators: AxisLocator | tuple[float, float],
            yaxis_locators: AxisLocator | tuple[float, float],
            is_fft: bool = False,
            *,
            sig_figs: tuple[float, float] = (.1, .1),
            scale_type=Literal['plain', 'sci'],
            has_multi_locators_y_axis: bool = False,
            has_scientific_notation: bool = False,
            has_formated_x_axis: bool = False,
            has_shifted_x_axis: bool = False,
            x_axis_scaling_factor=None
    ) -> Axes:
        """"""

        # Wrapping for shorter variable names
        xaxis = (xaxis_locators
                 if isinstance(xaxis_locators, AxisLocator)
                 else AxisLocator(*xaxis_locators))

        yaxis = (xaxis_locators
                 if isinstance(xaxis_locators, AxisLocator)
                 else AxisLocator(*xaxis_locators))

        ax = plt.gca() if plot is None else plot

        if is_fft:
            ax.xaxis.set(major_locator=MultipleLocator(xaxis.major),
                         minor_locator=MultipleLocator(xaxis.minor),
                         major_formatter=FormatStrFormatter(f"%{sig_figs[0]}f"))

            _yaxis_base = 10
            ax.yaxis.set(major_locator=LogLocator(base=_yaxis_base,
                                                  numticks=int(yaxis.major)),
                         minor_locator=LogLocator(base=_yaxis_base,
                                                  subs=0.1 * np.arange(1, _yaxis_base),
                                                  numticks=int(yaxis.minor)),
                         minor_formatter=NullFormatter())

            return ax

        # General cases don't involve plotting Fast Fourier Transform (FFT) results
        ax.xaxis.set(major_locator=MultipleLocator(xaxis.major),
                     minor_locator=MultipleLocator(xaxis.minor),
                     major_formatter=FormatStrFormatter(f"%{sig_figs[0]}f"))

        if has_multi_locators_y_axis:
            ax.yaxis.set(major_locator=MultipleLocator(yaxis.major),
                         minor_locator=MultipleLocator(yaxis.minor),
                         major_formatter=FormatStrFormatter(f"%{sig_figs[1]}f"))
        else:
            ax.yaxis.set(major_locator=MaxNLocator(nbins=yaxis.major, prune='lower'),
                         minor_locator=AutoMinorLocator(yaxis.minor),
                         major_formatter=FormatStrFormatter(f"&{sig_figs[1]}f")
                         )

        if x_axis_scaling_factor is not None:
            scaled_labels, major_labels, scaled_major = self._choose_scaling(value=xaxis.major / x_axis_scaling_factor)
            rescaling_ratio = x_axis_scaling_factor / scaled_major

            shift = -self._.num_sites_total() / 2 if has_shifted_x_axis else 0

            xdata = np.array([])

            # Iterate through all line objects in axes
            for line in ax.get_lines():
                line_x, line_y = line.get_data()

                # Apply shift and scale transformation
                line_x = rescaling_ratio * (line_x + shift)
                line.set_data(line_x, line_y)
                xdata = np.concatenate((xdata, line_x))

            # Exclude first/last values for stylistic reasons on plots
            ax.set(xlim=(xdata[0], xdata[-1] + rescaling_ratio),
                   xlabel=f'Length, $L$ ({major_labels[2]})')


        return ax



