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
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

# Third-party imports

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



