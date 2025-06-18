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
SELECT_AXIS = Literal['x', 'y', 'both']


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

    @staticmethod
    def _ensure_axis_locator(val):
        if isinstance(val, AxisLocator):
            return val
        elif isinstance(val, (tuple, list)) and len(val) == 2:
            return AxisLocator(*val)
        else:
            raise TypeError(f"Invalid axis locator: {val}")

    @staticmethod
    def _apply_fft_locators(
            ax: Axes,
            x_locs: AxisLocator | tuple[float, float],
            y_locs: AxisLocator | tuple[float, float],
            sig_figs: tuple[float, float]
    ) -> Axes:
        # Setup specific axes ticks, leaving other changes to the FFT plotting method.
        ax.xaxis.set(major_locator=MultipleLocator(x_locs.major),
                     minor_locator=MultipleLocator(x_locs.minor),
                     major_formatter=FormatStrFormatter(f"%{sig_figs[0]}f"))

        _yaxis_base = 10
        ax.yaxis.set(major_locator=LogLocator(base=_yaxis_base,
                                              numticks=int(y_locs.major)),
                     minor_locator=LogLocator(base=_yaxis_base,
                                              subs=0.1 * np.arange(1, _yaxis_base),
                                              numticks=int(y_locs.minor)),
                     minor_formatter=NullFormatter())

        return ax

    def _apply_axis_scaling_and_shift(
            self,
            ax: Axes,
            x_locs: AxisLocator | tuple[float, float],
            shift_axis: Optional[SELECT_AXIS],
            x_axis_scaling_factor: Any
    ) -> None:
        # Shift axes before updating ticks to prevent difficult logic errors arising.
        scaled_labels, major_labels, scaled_major = self._choose_scaling(value=x_locs.major / x_axis_scaling_factor)
        rescaling_ratio = x_axis_scaling_factor / scaled_major

        shift = (-0.5 * self._.num_sites_total()
                 if shift_axis in ('x', 'both')
                 else 0)

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

    def set_ticks(
            self,
            plot: Optional[Axes],
            xaxis_locators: AxisLocator | tuple[float, float],
            yaxis_locators: AxisLocator | tuple[float, float],
            is_fft: bool = False,
            *,
            sig_figs: tuple[float, float] = (.1, .1),
            scale_type: Literal['plain', 'sci'] = 'sci',
            has_multi_locators_y_axis: bool = False,
            has_scientific_notation: bool = False,
            scalar_format_axis: SELECT_AXIS = 'y',
            shift_axis: Optional[SELECT_AXIS] = None,
            x_axis_scaling_factor=None
    ) -> Axes:
        """"""
        # Wrapping for shorter variable names
        x_locs = self._ensure_axis_locator(xaxis_locators)
        y_locs = self._ensure_axis_locator(yaxis_locators)
        ax = plot or plt.gca()

        # Special case
        if is_fft:
            return self._apply_fft_locators(ax, x_locs, y_locs, sig_figs)

        # General cases don't involve plotting Fast Fourier Transform (FFT) results
        if x_axis_scaling_factor is not None:
            self._apply_axis_scaling_and_shift(ax, x_locs, shift_axis, x_axis_scaling_factor)

        # Handle general locators
        # Initial setup, x-axis
        ax.xaxis.set(major_locator=MultipleLocator(x_locs.major),
                     minor_locator=MultipleLocator(x_locs.minor))

        # Initial setup, y-axis
        if has_multi_locators_y_axis:
            ax.yaxis.set(major_locator=MultipleLocator(y_locs.major),
                         minor_locator=MultipleLocator(y_locs.minor))
        else:
            ax.yaxis.set(major_locator=MaxNLocator(nbins=int(y_locs.major), prune='lower'),
                         minor_locator=AutoMinorLocator(int(y_locs.minor)))

        # Additional custom major-formatter options.
        ax.ticklabel_format(axis=scalar_format_axis,
                            scilimits=(0, 0),
                            useMathText=True)

        ax.xaxis.set_major_formatter(FormatStrFormatter(f"%{sig_figs[0]}f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter(f"%{sig_figs[1]}f"))

        # Keep offset, but turn invisible, to not reposition figure elements.
        if has_scientific_notation:
            ax.yaxis.get_offset_text().set(x=-0.045, fontsize=8, visible=True)
        else:
            ax.yaxis.get_offset_text().set(visible=False)

        return ax

    def _choose_scaling(self, value=None, subplot_to_scale=None, row_index=None, presets=None):
        """
        TODO. Check if following commented code is a suitable replacement for this method.
        """
        fmt = mtick.EngFormatter(unit='Hz', places=1, sep=" ")
        # ax.yaxis.set_major_formatter(fmt)
        # return fmt
        if value is None and subplot_to_scale is None:
            exit(1)

        if presets is None:
            presets = {
                'nano': [1e-9, r'$\mathrm{nm}$'],
                'micro': [1e-6, r'$\mathrm{{\mu} m}$'],
                'milli': [1e-3, r'$\mathrm{mm}$']
                # Add as needed
            }

        if subplot_to_scale is not None:
            value = subplot_to_scale.get_ylim()[1]
            magnitude_value = int(np.floor(np.log10(value)))
            # Convert uppermost y-tick label to a float, and compared against ylim (upper). If the uppermost tick is
            # greater than ylim (upper) it means an automatic scientific notation conversion (10e-2 -> 1e01)
            # occurred and needs to be undone.
            if float(subplot_to_scale.get_yticklabels()[-2].get_text()) * 10 ** magnitude_value > value:
                magnitude_value -= 1
        else:
            magnitude_value = int(np.floor(np.log10(value)))

        closest_preset_name, (closest_preset_value, closest_preset_tag) = min(presets.items(),
                                                                              key=lambda x: abs(
                                                                                  magnitude_value - np.log10(x[1][0])))

        # Generate labels and values for closest preset and raw value
        closest_preset_exp = int(np.log10(closest_preset_value))
        # This gives us the order, so the +1 is required so we can plot across all values in this order
        # e.g. if value_exp = -3 (i.e. 1e-3 order)

        closest_preset_labels = [
            r'$\times \mathcal{10}^{' + f'{closest_preset_exp}' + '}$',
            r'$\mathcal{10}^{' + f'{closest_preset_exp}' + '}$',
            closest_preset_tag
        ]

        value_labels = [
            r'$\times \mathcal{10}^{' + f'{magnitude_value}' + '}$',
            r'$\mathcal{10}^{' + f'{magnitude_value}' + '}$'
        ]

        return closest_preset_labels, value_labels, closest_preset_value
