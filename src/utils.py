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
        src/utils.py
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
from numbers import Real
from typing import Optional, TypeVar

# Third-party imports

# Local application imports

# Module-level constants
R = TypeVar('R')

__all__ = ['PairHelper']


class PairHelper:

    @staticmethod
    def ensure_pair(
            seq: Sequence[Optional[Real]],
            *,
            name: str = 'pair',
            allow_none: bool = False,
            cast: Callable[[Real], ...] = float
    ) -> tuple[Optional[Real], Optional[Real]]:
        """Turn input sequences into length-2 tuples.

        Use cases such as checking `schemes.AxLims` work as `AxLims` is a NamedTuple, which is a Sequence.
        """
        try:
            a, b = seq
        except ValueError:
            raise ValueError(f"{name!r} must be a two-element sequence, but got {seq!r}")

        # lower/upper come from arguments of `schemes.AxLims`
        for val, side in ((a, 'lower'), (b, 'upper')):
            if allow_none and val is None:
                continue
            if not isinstance(val, Real):
                raise TypeError(f"{name!r} {side} must be type {Real!r}, got {val!r}")

        return (None if a is None else cast(a),
                None if b is None else cast(b))
