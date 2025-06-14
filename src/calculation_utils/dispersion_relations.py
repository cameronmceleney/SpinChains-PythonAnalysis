#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""(One liner introducing this file dispersion_relations.py)

(
Leading paragraphs explaining file in more detail.
)

Attributes:
    (Here, place any module-scope constants users will import.)
    
Constants:
    MODULE_LEVEL_CONSTANT1 (int): A module-level constant.

Examples:
    (Here, place useful implementations of the contents of dispersion_relations.py). Note that leading symbol '>>>' includes the 
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
        src/calculation_utils/dispersion_relations.py
    Author
        Cameron Aidan McEleney < c.mceleney.1@research.gla.ac.uk >
    Created
        14 Jun 2025
    IDE
        PyCharm
        
.. _Google Python Style Guide:
   https://google.github.io/styleguide/pyguide.html
"""

# from __future__ import foo

__all__ = [""]

import typing
# Standard library imports
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar, Deque, Literal, Optional, Sequence

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import constants

# Local application imports
# (e.g. from .helpers import foo

# Module-level constants
VECTOR3_FLOATS: typing.TypeAlias = tuple[float, float, float]
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
class DemagnetisationFactorCalculator:

    # Class-level constants
    _OPTIONS: ClassVar = Literal['uniform_prism', 'wedge']
    _FIELD_AXES: ClassVar = Literal['x', 'y', 'z']

    # User inputs
    dims: Sequence[float] | Vector3 = None
    field_axis: _FIELD_AXES = field(default='z', kw_only=True)

    # Processed in __post_init__
    factors: Vector3 = field(init=False, default=None)

    def __post_init__(self):

        if not isinstance(self.dims, (Vector3, list, tuple)):
            raise TypeError('`dims` must be either a Vector3, list, or tuple.')

        # Pad list/tuple as required so Vector3 can be assigned.
        if isinstance(self.dims, (list, tuple)):
            temp_dims = list(self.dims)
            if not (1 <= len(temp_dims) <= 3):
                raise ValueError('`dims` when passed as list or tuple, `dims` must have 1-3 elements.')
            temp_dims += [0.0] * (3 - len(temp_dims))
            self.dims = Vector3(*temp_dims)

    def _rotation_mapping(self, axis: str) -> Vector3:
        """Currently assumes that field is aligned along `z`.

        Will add other configurations in the future."""
        if isinstance(self.field_axis, str):
            rotation_map = {'x': 2, 'y': 1, 'z': 0}
        else:
            raise KeyError
        return self.dims.rotate(rotation_map[axis])

    def calculate_factors(self, equation: _OPTIONS) -> None:
        """"""
        if equation not in self._OPTIONS:
            raise ValueError(f"Unknown system '{equation}'")

        match equation:
            case 'uniform_prism':
                self.factors = Vector3(
                    self._calculate_uniform_prism(self._rotation_mapping('x')),
                    self._calculate_uniform_prism(self._rotation_mapping('y')),
                    self._calculate_uniform_prism(self._rotation_mapping('z')))

    @staticmethod
    def _calculate_uniform_prism(base: Vector3) -> float:
        """Calculate the demagnetisation factors for a prism experiencing a uniform external static Zeeman field.

        The equation used by this function is taken directly from this `magpar website`_. It assumes that the external
        field is applied perpendicular to driving field, and thus the propagation direction of the induced spin-waves.
        Distances should be given in nanometres.

        Args:
            base: details.
        .. _magpar website:
            http://www.magpar.net/static/magpar-0.9rc2/doc/html/demagcalc.html
        """
        # Stick with a/b/c so it's easier to compare code against source website.
        a, b, c = base.as_tuple()

        r = a ** 2 + b ** 2 + c ** 2
        sqrt_r = np.sqrt(r)

        factor = ((b ** 2 - c ** 2) / (2 * b * c)) * np.log((sqrt_r - a) / (sqrt_r + a))

        factor += ((a ** 2 - c ** 2) / (2 * a * c)) * np.log((sqrt_r - b) / (sqrt_r + b))

        sqrt_plane = np.sqrt(a ** 2 + b ** 2)
        factor += (b / (2 * c)) * np.log((sqrt_plane + a)
                                         / (sqrt_plane - a))

        factor += (a / (2 * c)) * np.log((sqrt_plane + b)
                                         / (sqrt_plane - b))

        sqrt_plane = np.sqrt(b ** 2 + c ** 2)
        factor += (c / (2 * a)) * np.log((sqrt_plane - b)
                                         / (sqrt_plane + b))

        sqrt_plane = np.sqrt(a ** 2 + c ** 2)
        factor += (c / (2 * b)) * np.log((sqrt_plane - a)
                                         / (sqrt_plane + a))

        factor += 2 * np.arctan((a * b) / (c * sqrt_r))

        factor += (a ** 3 + b ** 3 - 2 * c ** 3) / (3 * a * b * c)

        factor += ((a ** 2 + b ** 2 - 2 * c ** 2) / (3 * a * b * c)) * sqrt_r

        factor += (c / (a * b)) * (np.sqrt(a ** 2 + c ** 2) + np.sqrt(b ** 2 + c ** 2))

        factor -= ((np.power((a ** 2 + b ** 2), 3 / 2) + np.power((b ** 2 + c ** 2), 3 / 2) + np.power(
            (c ** 2 + a ** 2), 3 / 2)) / (3 * a * b * c))

        factor /= np.pi

        return factor


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
        return (2 * self.exchange_stiffness) / (constants.mu_0 * self.saturation_magnetisation)

    @cached_property
    def moon_dmi_energy(self):
        return (2 * self.dmi_micromagnetic) / (constants.mu_0 * self.saturation_magnetisation)


class CalculateDispersionRelation:

    def __init__(
            self,
            wavevectors: NDArray,
            sys_dims: Sequence[float] | Vector3,
            *,
            lattice_constant: float,
            sys_consts: Optional[MagneticSystemConstants] = None,
            **sys_consts_kwargs
    ):
        """"""
        self._wavevectors = wavevectors
        self._dims = Vector3(*sys_dims)
        self._lattice_constant = lattice_constant

        if sys_consts and sys_consts_kwargs:
            raise ValueError("Pass either sys_consts or its individual fields, but not both.")
        if sys_consts is not None:
            self.sys_consts = sys_consts
        else:
            self.sys_consts = MagneticSystemConstants(**sys_consts_kwargs)

        self._demag_factors_calc = DemagnetisationFactorCalculator(self._dims)

    def generalised_with_ua(self) -> NDArray:
        """Generalised dispersion relation with uniaxial anisotropy.

        Requires all magnetic field terms to be in units of :math:`A m^{-1}`.

        Returns:
            `NDArray` of linear frequencies in Hz.
        """
        # This function requires the field components to be in [A m^{-1}]
        H0 = self.sys_consts.zeeman_field_static / constants.mu_0

        demag = self._demag_factors_calc
        demag.calculate_factors('uniform_prism')

        # Read this line as `H_0 + J^{star} * k^{2}`
        const_terms = H0 + self.sys_consts.moon_exchange_energy * (self._wavevectors ** 2)

        if self.sys_consts.has_uniaxial_anisotropy:
            const_terms += (
                    (2 * aniso_axis[2] ** 2) / (self.sys_consts.saturation_magnetisation * constants.mu_0)
                    * (self.sys_consts.uniaxial_anisotropy_K1
                       + 2 * self.sys_consts.uniaxial_anisotropy_K2 * aniso_axis[2] ** 2)
            )

        if self.sys_consts.has_demagnetisation:
            inner_root = const_terms + self.sys_consts.saturation_magnetisation * (demag.factors.x - demag.factors.z)
            inner_root *= const_terms + self.sys_consts.saturation_magnetisation * (demag.factors.y - demag.factors.z)
            angular_freqs = np.sqrt(inner_root)
        else:
            angular_freqs = const_terms

        if self.sys_consts.has_dmi:
            # Negative dmi_energy term is to match the orientation convention to my C++ code.
            angular_freqs += -1 * self.sys_consts.moon_dmi_energy * self._wavevectors

        return angular_freqs * self.sys_consts.gamma * constants.mu_0
