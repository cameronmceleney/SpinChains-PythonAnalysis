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
        src/calculations/utils.py
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

__all__ = ['Vector3', 'MagneticSystemConstants']

# Standard library imports
from collections import deque
from dataclasses import dataclass
from functools import cached_property
from typing import Deque, TypeAlias

# Third-party imports
import numpy as np
from scipy.constants import mu_0

# Local application imports


# Module-level constants
VECTOR3_FLOATS: TypeAlias = tuple[float, float, float]
"""A module-level constant with in-line docstring."""


@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float

    def as_tuple(self) -> VECTOR3_FLOATS:
        return self.x, self.y, self.z

    def rotate(self, step: int = 1) -> "Vector3":
        """Cyclic vector's components by `step`.

        Internally uses `collections.deque` for the rotation.
        """

        seq: Deque[float] = deque(self.as_tuple())
        seq.rotate(step)
        return Vector3(*seq)


@dataclass
class MagneticSystemConstants:
    saturation_magnetisation: float
    exchange_stiffness: float
    gyromagnetic_ratio: float

    has_demagnetisation: bool
    zeeman_field_static: float = None
    dmi_micromagnetic: float = None
    aniso_axis: tuple[float, float, float] = None
    uniaxial_anisotropy_K1: float = None
    uniaxial_anisotropy_K2: float = None

    p: int = 1

    @property
    def has_dmi(self):
        return self.dmi_micromagnetic is not None

    @property
    def has_uniaxial_anisotropy(self):
        return bool(self.uniaxial_anisotropy_K1 or self.uniaxial_anisotropy_K2)

    @cached_property
    def gamma(self):
        return self.gyromagnetic_ratio * 2 * np.pi

    @cached_property
    def moon_exchange_energy(self):
        return (2 * self.exchange_stiffness) / (mu_0 * self.saturation_magnetisation)

    @cached_property
    def moon_dmi_energy(self):
        return (2 * self.dmi_micromagnetic) / (mu_0 * self.saturation_magnetisation)
