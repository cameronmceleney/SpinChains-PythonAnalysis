# -*- coding: utf-8 -*-

# -------------------------- Preprocessing Directives -------------------------

# Full packages
import operator

# Specific functions from packages
from typing import Dict, List, Tuple, Type, Any, TypeVar

# My full modules


# Specific functions from my modules


# ----------------------------- Program Information ----------------------------

"""
    This file contains the old mappings I used (before 19 Feb 24) for importing simulation data from CSV files,
    and then parsing the data into a format that can be used for analysis.
"""
PROGRAM_NAME = "attribute_mappings_legacy.py"
"""
Created on 2024-02-24 by cameronmceleney
"""
T = TypeVar('T')  # Generic type variable for SimulationVariable


class TypedVariable:
    def __init__(self, name, var_type, value=None, metadata=None):
        self.name = name
        self.value_dtype = var_type
        self.value = value
        self.default = None
        self.metadata = metadata or []

    def __get__(self, instance, owner):
        if instance is None:
            return self
        # Return a callable that still allows access to this TypedVariable's attributes
        return TypedVariableInstance(self, instance)

    def __set__(self, instance, value):
        if not isinstance(value, self.value_dtype) and value is not None:
            raise TypeError(f"Expected type {self.value_dtype.__name__}, got {type(value).__name__}")
        instance.__dict__[self.name] = value


class TypedVariableInstance:
    def __init__(self, typed_variable: TypedVariable, instance: Any):
        self._typed_variable = typed_variable
        self._instance = instance

    def __call__(self):
        # This method allows the instance to be called like a function to retrieve its value.
        # Retrieve the value from the instance's dictionary or use the default if not set.
        return self._instance.__dict__.get(self._typed_variable.name, self._typed_variable.default)

    def get_metadata(self) -> List[str]:
        # Return the metadata associated with the TypedVariable.
        return self._typed_variable.metadata

    def get_name(self):
        # Return the metadata associated with the TypedVariable.
        return self._typed_variable.name

    def get_type(self):
        # Return the type information of the TypedVariable.
        return self._typed_variable.value_dtype

    def __getattr__(self, item):
        # This method is called when an attribute lookup has not found the attribute in the usual places.
        if hasattr(self._typed_variable, item):
            return getattr(self._typed_variable, item)

        # Delegate arithmetic and other operations to the value directly.
        try:
            if item in dir(operator) and callable(getattr(operator, item)):
                def operation_wrapper(*args, **kwargs):
                    operation = getattr(operator, item)
                    return operation(self.get_value(), *args, **kwargs)

                return operation_wrapper
        except AttributeError:
            pass

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def __repr__(self):
        # Provide a string representation of the instance's current value for easier debugging and logging.
        return repr(self.__call__())


class VariablesMeta(type):
    def __new__(mcs, name, bases, attrs):
        # Initialize _typed_variables and _variables for the new class
        typed_variables = {}
        _variables = {}

        # Inherit _typed_variables and _variables from bases
        for base in bases:
            if hasattr(base, '_typed_variables'):
                # noinspection PyProtectedMember
                typed_variables.update(base._typed_variables)
            if hasattr(base, '_variables'):
                # noinspection PyProtectedMember
                _variables.update(base._variables)

        # Update with current class's TypedVariable instances
        for k, v in attrs.items():
            if isinstance(v, TypedVariable):
                typed_variables[k] = v
                _variables[k] = (v.name, v.value_dtype, v.default, v.metadata)

        attrs['_typed_variables'] = typed_variables
        attrs['_variables'] = _variables

        # Set annotations for type hinting
        annotations = attrs.get('__annotations__', {})
        for var_name, typed_var in typed_variables.items():
            annotations[var_name] = typed_var.value_dtype
        attrs['__annotations__'] = annotations

        return super().__new__(mcs, name, bases, attrs)

    def __getitem__(cls, key: str):
        if key in cls.__dict__.get('_variables', {}):
            return cls.__dict__['_variables'][key]
        elif key in cls.__dict__.get('_typed_variables', {}):
            return cls.__dict__['_typed_variables'][key]
        raise KeyError(f"{key} is not available in {cls.__name__}")


class VariablesContainer(metaclass=VariablesMeta):
    # Define variables here as class attributes
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

    @classmethod
    def get_metadata(cls, name):
        var = getattr(cls, name, None)
        if var and isinstance(var, TypedVariable):
            return var.metadata
        raise AttributeError(f"{name} not found")


class AttributeMeta(type):
    def __new__(mcs, name, bases, attrs):

        if '__annotations__' not in attrs:
            attrs['__annotations__'] = {}

        for attr, (attr_type, _) in AttributeMappings.key_data.items():
            attrs['__annotations__'][attr] = attr_type
            attrs[attr] = None  # Set default value to None

        for attr, (attr_type, _) in AttributeMappings.sim_flags.items():
            attrs['__annotations__'][attr] = attr_type
            attrs[attr] = None  # Set default value to None

        return super().__new__(mcs, name, bases, attrs)

    def __init__(cls, name, bases, attrs):
        # perform any additional initialization here...
        super().__init__(name, bases, attrs)


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
