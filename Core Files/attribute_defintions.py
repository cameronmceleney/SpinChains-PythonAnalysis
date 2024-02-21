# -*- coding: utf-8 -*-

# -------------------------- Preprocessing Directives -------------------------

# Full packages


# Specific functions from packages
from typing import Dict, List, Optional, Tuple, Type

# My full modules


# Specific functions from my modules


# ----------------------------- Program Information ----------------------------

"""
Description of what foo.py does
"""
PROGRAM_NAME = "attribute_definitions.py"
"""
Created on 2024-02-21 by cameronmceleney
"""


class TypedVariable:
    def __init__(self, name, var_type, default=None, metadata=None):
        self.name = name
        self.type = var_type
        self.value = default
        self.metadata = metadata or []

    def __get__(self, instance, owner):
        return self.value

    def __set__(self, instance, value):
        if not isinstance(value, self.type) and value is not None:
            raise TypeError(f"Expected type {self.type.__name__}, got {type(value).__name__}")
        self.value = value

    def __delete__(self, instance):
        self.value = None

    def __getitem__(self, key):
        if key == 'name':
            return self.name
        elif key == 'type':
            return self.type
        elif key == 'value':
            return self.value
        elif key == 'metadata':
            return self.metadata
        else:
            raise KeyError(f"Key '{key}' not found")


class VariablesMeta(type):
    def __new__(mcs, name, bases, attrs):
        # Collect TypedVariable instances into a class-level dictionary
        typed_variables = {k: v for k, v in attrs.items() if isinstance(v, TypedVariable)}
        attrs['_typed_variables'] = typed_variables

        __typed_vars2 = {
            k: (v.type, v.value, v.metadata) for k, v in attrs.items() if isinstance(v, TypedVariable)
        }
        attrs['_variables'] = __typed_vars2

        # Optionally, directly set TypedVariable values as class attributes for autocomplete/type hinting
        for var_name, typed_var in typed_variables.items():
            attrs[var_name] = typed_var.value

        return super().__new__(mcs, name, bases, attrs)

    def __getitem__(cls, key: str):
        # Access to TypedVariable instances via class indexing
        if isinstance(key, str):
            if key == '_variables':
                return cls._variables
            else:
                return cls._variables[key]
        else:
            raise TypeError("Unsupported key type")


class VariablesContainer(metaclass=VariablesMeta):
    """
    This class is a container for TypedVariable instances, which can be accessed via class indexing.

    Example usage to access the whole Dict:
     -         typed_var_dict = VariablesContainer._typed_variables
               print(typed_var_dict)  # Shows all TypedVariable instances
               print(VariablesContainer._typed_variables['staticZeemanStrength'].metadata)
    """
    # _typed_variables: Dict[str, TypedVariable] = {}
    # _variables: Dict[str, Tuple[str, Optional[Type], List[str]]] = {}
    staticZeemanStrength = TypedVariable('staticZeemanStrength', float, None, ['staticBiasField', 'Static Bias Field'])
    oscillatingZeemanStrength1 = TypedVariable('oscillatingZeemanStrength1', float, None,
                                               ['dynamicBiasField', 'Dynamic Bias Field'])
    shockwaveScaling = TypedVariable('shockwaveScaling', float, None,
                                     ['dynamicBiasFieldScaleFactor', 'Dynamic Bias Field Scale Factor'])
    oscillatingZeemanStrength2 = TypedVariable('oscillatingZeemanStrength2', float, None,
                                               ['secondDynamicBiasField', 'Second Dynamic Bias Field'])
    drivingFreq = TypedVariable('drivingFreq', float, None, ['drivingFreq', 'drivingFrequency'])
    drivingRegionLhs = TypedVariable('drivingRegionLhs', int, None,
                                     ['drivingRegionStartSite', 'Driving Region Start Site'])
    drivingRegionRhs = TypedVariable('drivingRegionRhs', int, None, ['drivingRegionEndSite', 'Driving Region End Site'])
    drivingRegionWidth = TypedVariable('drivingRegionWidth', int, None, ['drivingRegionWidth', 'Driving Region Width'])
    maxSimTime = TypedVariable('maxSimTime', float, None, ['maxSimTime', 'Max. Sim. Time'])
    heisenbergExchangeMin = TypedVariable('heisenbergExchangeMin', float, None, ['minExchangeVal', 'Min. Exchange Val'])
    heisenbergExchangeMax = TypedVariable('heisenbergExchangeMax', float, None, ['maxExchangeVal', 'Max. Exchange Val'])
    iterationEnd = TypedVariable('iterationEnd', float, None, ['maxIterations', 'Max. Iterations'])
    numberOfDataPoints = TypedVariable('numberOfDataPoints', float, None, ['numDatapoints', 'No. DataPoints'])
    numSpinsInChain = TypedVariable('numSpinsInChain', int, None, ['numSpinsInChain', 'No. Spins in Chain'])
    numSpinsInABC = TypedVariable('numSpinsInABC', int, None, ['numDampedSpinsPerSide', 'No. Damped Spins (per side)'])
    systemTotalSpins = TypedVariable('systemTotalSpins', int, None, ['numTotalSpins', 'No. Total Spins'])
    stepsize = TypedVariable('stepsize', float, None, ['Stepsize'])
    gilbertDamping = TypedVariable('gilbertDamping', float, None, ['gilbertDampingFactor', 'Gilbert Damping Factor'])
    gyroMagConst = TypedVariable('gyroMagConst', float, None, ['gyroRatio', 'Gyromagnetic Ratio'])
    shockwaveGradientTime = TypedVariable('shockwaveGradientTime', float, None,
                                          ['shockwaveGradientTime', 'Shockwave Gradient Time'])
    shockwaveApplicationTime = TypedVariable('shockwaveApplicationTime', float, None,
                                             ['shockwaveApplicationTime', 'Shockwave Application Time'])
    gilbertABCOuter = TypedVariable('gilbertABCOuter', float, None, ['abcDampingLower', 'ABC Damping (lower)'])
    gilbertABCInner = TypedVariable('gilbertABCInner', float, None, ['abcDampingUpper', 'ABC Damping (upper)'])
    dmiConstant = TypedVariable('dmiConstant', float, None, ['dmiConstant', 'DMI Constant'])
    satMag = TypedVariable('satMag', float, None, ['saturationMagnetisation', 'Saturation Magnetisation'])
    exchangeStiffness = TypedVariable('exchangeStiffness', float, None, ['exchangeStiffness', 'Exchange Stiffness'])
    anisotropyField = TypedVariable('anisotropyField', float, None,
                                    ['anisotropyShapeField', 'Anisotropy (Shape) Field'])
    latticeConstant = TypedVariable('latticeConstant', float, None, ['latticeConstant', 'Lattice Constant'])

    # Define other variables similarly...

class AttributeMappings:
    key_data: Dict[str, Tuple[Type, List[str]]] = {
        'staticZeemanStrength': (float, ['staticBiasField', 'Static Bias Field']),
        'oscillatingZeemanStrength1': (float, ['dynamicBiasField', 'Dynamic Bias Field']),
        'shockwaveScaling': (float, ['dynamicBiasFieldScaleFactor', 'Dynamic Bias Field Scale Factor']),
        'oscillatingZeemanStrength2': (float, ['secondDynamicBiasField', 'Second Dynamic Bias Field']),
        'drivingFreq': (float, ['drivingFreq', 'drivingFrequency']),
        'drivingRegionLhs': (int, ['drivingRegionStartSite', 'Driving Region Start Site']),
        'drivingRegionRhs': (int, ['drivingRegionEndSite', 'Driving Region End Site']),
        'drivingRegionWidth': (int, ['drivingRegionWidth', 'Driving Region Width']),
        'maxSimTime': (float, ['maxSimTime', 'Max. Sim. Time']),
        'heisenbergExchangeMin': (float, ['minExchangeVal', 'Min. Exchange Val']),
        'heisenbergExchangeMax': (float, ['maxExchangeVal', 'Max. Exchange Val']),
        'iterationEnd': (float, ['maxIterations', 'Max. Iterations']),
        'numberOfDataPoints': (float, ['numDatapoints', 'No. DataPoints']),
        'numSpinsInChain': (int, ['numSpinsInChain', 'No. Spins in Chain']),
        'numSpinsInABC': (int, ['numDampedSpinsPerSide', 'No. Damped Spins (per side)']),
        'systemTotalSpins': (int, ['numTotalSpins', 'No. Total Spins']),
        'stepsize': (float, ['Stepsize']),
        'gilbertDamping': (float, ['gilbertDampingFactor', 'Gilbert Damping Factor']),
        'gyroMagConst': (float, ['gyroRatio', 'Gyromagnetic Ratio']),
        'shockwaveGradientTime': (float, ['shockwaveGradientTime', 'Shockwave Gradient Time']),
        'shockwaveApplicationTime': (float, ['shockwaveApplicationTime', 'Shockwave Application Time']),
        'gilbertABCOuter': (float, ['abcDampingLower', 'ABC Damping (lower)']),
        'gilbertABCInner': (float, ['abcDampingUpper', 'ABC Damping (upper)']),
        'dmiConstant': (float, ['dmiConstant', 'DMI Constant']),
        'satMag': (float, ['saturationMagnetisation', 'Saturation Magnetisation']),
        'exchangeStiffness': (float, ['exchangeStiffness', 'Exchange Stiffness']),
        'anisotropyField': (float, ['anisotropyShapeField', 'Anisotropy (Shape) Field']),
        'latticeConstant': (float, ['latticeConstant', 'Lattice Constant'])
    }

    sim_flags: Dict[str, Tuple[Type, List[str]]] = {
        'hasLLG': (bool, ['shouldUseLLG', 'usingMagdynamics', 'Using magDynamics']),
        'usingShockwave': (bool, ['hasShockwave', 'usingShockwave', 'Using Shockwave']),
        'driveFromLhs': (bool, ['shouldDriveLHS', 'driveFromLhs', 'Drive from LHS']),
        'numericalMethodUsed': (str, ['numericalMethod', 'numericalMethodUsed', 'Numerical Method Used']),
        'hasStaticDrive': (bool, ['isOscillatingZeemanStatic', 'hasStaticDrive', 'Has Static Drive']),
        'hasDipolar': (bool, ['hasDipolar', 'Has Dipolar']),
        'hasDmi': (bool, ['hasDMI', 'hasDmi', 'Has DMI']),
        'hasStt': (bool, ['hasSTT', 'hasStt', 'Has STT']),
        'hasZeeman': (bool, ['hasStaticZeeman', 'hasZeeman', 'Has Zeeman']),
        'hasDemagIntense': (bool, ['hasDemagIntense', 'Has Demag Intense']),
        'hasDemagFft': (bool, ['hasDemagFFT', 'hasDemagFft', 'Has Demag FFT']),
        'hasShapeAnisotropy': (bool, ['hasShapeAnisotropy', 'Has Shape Anisotropy'])
    }

    @staticmethod
    def dict_with_none(attributes: Dict[str, Tuple[Type, List[str]]]) -> Dict[str, None]:
        """
        This function takes a dictionary mapping attribute names to their types
        and returns a new dictionary with the same keys, but all values initialized to None.

        The purpose is to prepare a structure for dynamic attribute assignment,
        where type hints are used for static analysis rather than enforcing types at runtime.
        """
        return {key: None for key in attributes.keys()}


def apply_type_hints(cls):
    # Decorator implementation that uses AttributeMappings from the same file
    for attr, (attr_type, _) in AttributeMappings.key_data.items():
        cls.__annotations__[attr] = attr_type
    for attr, (attr_type, _) in AttributeMappings.sim_flags.items():
        cls.__annotations__[attr] = attr_type
    return cls
