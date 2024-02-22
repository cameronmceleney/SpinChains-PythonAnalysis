# -*- coding: utf-8 -*-

# -------------------------- Preprocessing Directives -------------------------

# Full packages
import operator

# Specific functions from packages
from typing import Dict, List, Tuple, Type, Any, TypeVar, Generic, Callable

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
T = TypeVar('T')  # Generic type variable for SimulationVariable


class SimulationVariable(Generic[T]):
    def __init__(self, name: str, var_type: Type[T], value: T = None, metadata: List[str] = None):
        """
        This class is used to define a parameter for a simulation. It is used to enforce type checking and
        provide metadata for the parameter.

        :param name:
        :param var_type:
        :param value:
        :param metadata:
        """
        self.name: str = name
        self.value_dtype: Type[T] = var_type
        self.value: T = value
        self.default: T = None
        self.metadata: List[str] = metadata if metadata is not None else []

    def __get__(self, instance: Any, owner: Type) -> Any:
        """
        This method is called when the attribute is accessed from an instance or class.

        :param instance:
        :param owner:
        :return: either SimulationVariableInstance[T] or SimulationVariable[T]
        """
        if instance is None:
            return self
        return SimulationVariableInstance(self, instance)

    def __set__(self, instance: Any, value: T):
        """
        This method is called when the attribute is set on an instance.

        :param instance:
        :param value:
        :return:
        """
        if not self.validate_type(value):
            raise TypeError(f"Expected type {self.value_dtype.__name__}, got {type(value).__name__}")
        instance.__dict__[self.name] = value

    def validate_type(self, value: Any) -> bool:
        """
        This method is used to directly validate the type of the value being set.

        :param value:
        :return:
        """
        if self.value_dtype in (int, float) and isinstance(value, (int, float)):
            return True  # Allow int for float types or vice versa
        return isinstance(value, self.value_dtype)

    def add_metadata(self, new_metadata: str):
        """
        This method is used to add metadata to the parameter.
        :param new_metadata:
        :return:
        """
        if new_metadata not in self.metadata:
            self.metadata.append(new_metadata)

    def update_metadata(self, new_metadata: List[str]) -> None:
        """
        Enables updating metadata during the code (rare case).

        :param new_metadata:
        :return:
        """
        self.metadata = new_metadata


class SimulationVariableInstance(Generic[T]):
    """
    This class is used to provide a callable instance of a SimulationVariable. It is used to enforce type checking and
    provide metadata for the parameter.

    :return:
    """

    def __init__(self, typed_variable: SimulationVariable[T], instance: Any):
        """

        :param typed_variable:
        :param instance:
        """
        self._typed_variable: SimulationVariable[T] = typed_variable
        self._instance: Any = instance

    def __call__(self) -> T:
        """
        This method is called when the instance is called like a function.

        :return:
        """
        return self._instance.__dict__.get(self._typed_variable.name, self._typed_variable.default)

    def get_metadata(self) -> List[str]:
        """
        This method is used to retrieve the metadata associated with the SimulationVariable.

        :return:
        """
        return self._typed_variable.metadata

    def get_name(self) -> str:
        return self._typed_variable.name

    def get_dtype(self) -> Type[T]:
        return self._typed_variable.value_dtype

    def __getattr__(self, item: str) -> Callable:
        if hasattr(self._typed_variable, item):
            return getattr(self._typed_variable, item)

        value = self.__call__()
        if hasattr(value, item):
            return getattr(value, item)

        if item in dir(operator) and callable(getattr(operator, item)):
            def operation_wrapper(*args, **kwargs):
                operation = getattr(operator, item)
                return operation(self(), *args, **kwargs)

            return operation_wrapper

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def __repr__(self) -> str:
        return repr(self.__call__())


class SimulationParameterMeta(type):
    def __new__(cls, name: str, bases: tuple, attrs: dict):
        _typed_variables = {}
        _variables = {}
        container_type = attrs.get('container_type', None)

        for base in bases:
            # Direct access is necessary for functionality, but I dislike the compile complaining about it
            if hasattr(base, '_typed_variables'):
                # _typed_variables.update(base._typed_variables)  # Direct access
                base_typed_variables = getattr(base, '_typed_variables', {})
                _typed_variables.update(base_typed_variables)
            if hasattr(base, '_variables'):
                # _variables.update(base._variables)  # Direct access
                base_variables = getattr(base, '_variables', {})
                _variables.update(base_variables)

        for k, v in attrs.items():
            if isinstance(v, SimulationVariable):
                _typed_variables[k] = v
                _variables[k] = (v.name, v.value_dtype, v.default, v.metadata)

        attrs['_typed_variables'] = _typed_variables
        attrs['_variables'] = _variables

        # Automatically generate key_params or sim_flags based on container type
        if container_type == 'parameters':
            attrs['key_params'] = {k: v for k, v in _variables.items() if v[3] and 'keyParam' in v[3]}
        elif container_type == 'flags':
            attrs['sim_flags'] = {k: v for k, v in _variables.items() if v[3] and 'simFlag' in v[3]}

        annotations = attrs.get('__annotations__', {})
        for var_name, typed_var in _typed_variables.items():
            annotations[var_name] = typed_var.value_dtype
        attrs['__annotations__'] = annotations

        return super().__new__(cls, name, bases, attrs)

    def __getitem__(cls, key: str):
        if key in cls.__dict__.get('_variables', {}):
            return cls.__dict__['_variables'][key]
        elif key in cls.__dict__.get('_typed_variables', {}):
            return cls.__dict__['_typed_variables'][key]
        raise KeyError(f"{key} is not available in {cls.__name__}")


class SimulationParametersContainer(metaclass=SimulationParameterMeta):
    container_type = 'parameters'

    bias_zeeman_static = SimulationVariable('staticZeemanStrength', float, None,
                                            ['staticBiasField', 'Static Bias Field'])
    bias_zeeman_oscillating_1 = SimulationVariable('oscillatingZeemanStrength1', float, None,
                                                   ['dynamicBiasField', 'Dynamic Bias Field'])
    shockwave_scaling = SimulationVariable('shockwaveScaling', float, None,
                                           ['dynamicBiasFieldScaleFactor', 'Dynamic Bias Field Scale Factor'])
    bias_zeeman_oscillating_2 = SimulationVariable('oscillatingZeemanStrength2', float, None,
                                                   ['secondDynamicBiasField', 'Second Dynamic Bias Field'])

    driving_freq = SimulationVariable('drivingFreq', float, None, ['drivingFreq', 'drivingFrequency'])
    driving_region_lhs = SimulationVariable('drivingRegionLhs', int, None,
                                            ['drivingRegionStartSite', 'Driving Region Start Site'])
    driving_region_rhs = SimulationVariable('drivingRegionRhs', int, None,
                                            ['drivingRegionEndSite', 'Driving Region End Site'])
    driving_region_width = SimulationVariable('drivingRegionWidth', int, None,
                                              ['drivingRegionWidth', 'Driving Region Width'])

    sim_time_max = SimulationVariable('maxSimTime', float, None, ['maxSimTime', 'Max. Sim. Time'])
    exchange_heisenberg_min = SimulationVariable('heisenbergExchangeMin', float, None,
                                                 ['minExchangeVal', 'Min. Exchange Val'])
    exchange_heisenberg_max = SimulationVariable('heisenbergExchangeMax', float, None,
                                                 ['maxExchangeVal', 'Max. Exchange Val'])
    iteration_total = SimulationVariable('iterationEnd', float, None, ['maxIterations', 'Max. Iterations'])

    num_dp_per_site = SimulationVariable('numberOfDataPoints', float, None,
                                         ['numDatapoints', 'No. DataPoints', 'numDataPointsPerSite'])
    num_sites_chain = SimulationVariable('numSpinsInChain', int, None, ['numSpinsInChain', 'No. Spins in Chain'])
    num_sites_abc = SimulationVariable('numSpinsInABC', int, None,
                                       ['numDampedSpinsPerSide', 'No. Damped Spins (per side)'])
    num_sites_total = SimulationVariable('systemTotalSpins', int, None, ['numTotalSpins', 'No. Total Spins'])

    stepsize = SimulationVariable('stepsize', float, None, ['Stepsize'])
    gilbert_chain = SimulationVariable('gilbertDampingFactor', float, None,
                                       ['gilbertDampingFactor', 'Gilbert Damping Factor'])
    gyro_mag = SimulationVariable('gyroMagConst', float, None, ['gyroRatio', 'Gyromagnetic Ratio'])
    shockwave_time_gradient = SimulationVariable('shockwaveGradientTime', float, None,
                                                 ['shockwaveGradientTime', 'Shockwave Gradient Time'])

    shockwave_time_application = SimulationVariable('shockwaveApplicationTime', float, None,
                                                    ['shockwaveApplicationTime', 'Shockwave Application Time'])
    gilbert_abc_outer = SimulationVariable('gilbertABCOuter', float, None, ['abcDampingLower', 'ABC Damping (lower)'])
    gilbert_abc_inner = SimulationVariable('gilbertABCInner', float, None, ['abcDampingUpper', 'ABC Damping (upper)'])
    exchange_dmi_constant = SimulationVariable('dmiConstant', float, None, ['dmiConstant', 'DMI Constant'])

    sat_mag = SimulationVariable('satMag', float, None, ['saturationMagnetisation', 'Saturation Magnetisation'])
    exchange_stiffness = SimulationVariable('exchangeStiffness', float, None,
                                            ['exchangeStiffness', 'Exchange Stiffness'])
    anisotropy_field = SimulationVariable('anisotropyField', float, None,
                                          ['anisotropyShapeField', 'Anisotropy (Shape) Field'])
    lattice_constant = SimulationVariable('latticeConstant', float, None, ['latticeConstant', 'Lattice Constant'])

    @classmethod
    def get_metadata(cls, name):
        var = getattr(cls, name, None)
        if var and isinstance(var, SimulationVariable):
            return var.metadata
        raise AttributeError(f"{name} not found")


class SimulationFlagsContainer(metaclass=SimulationParameterMeta):
    container_type = 'flags'

    numerical_method = SimulationVariable('numericalMethodUsed', str, None,
                                          ['numericalMethod', 'numericalMethodUsed', 'Numerical Method Used'])

    has_llg = SimulationVariable('hasLLG', bool, None,
                                 ['shouldUseLLG', 'shouldUseLlg', 'hasLLG', 'hasLlg'])
    has_sllg = SimulationVariable('hasSLLG', bool, None, ['shouldUseSLLG', 'shouldUseSllg'])

    has_shockwave = SimulationVariable('hasShockwave', bool, None, ['hasShockwave'])
    has_dipolar = SimulationVariable('hasDipolar', bool, None, ['hasDipolar'])
    has_dmi = SimulationVariable('hasDMI', bool, None, ['hasDMI', 'hasDmi'])
    has_stt = SimulationVariable('hasSTT', bool, None, ['hasSTT', 'hasStt'])
    has_bias_zeeman_static = SimulationVariable('hasStaticZeeman', bool, None, ['hasStaticZeeman'])
    has_demag_1d_thin_film = SimulationVariable('hasDemag1DThinFilm', bool, None, ['hasDemag1DThinFilm'])
    has_demag_intense = SimulationVariable('hasDemagIntense', bool, None, ['hasDemagIntense'])
    has_demag_fft = SimulationVariable('hasDemagFFT', bool, None, ['hasDemagFFT'])
    has_anisotropy_shape = SimulationVariable('hasShapeAnisotropy', bool, None, ['hasShapeAnisotropy'])

    has_multiple_layers = SimulationVariable('hasMultipleLayers', bool, None, ['hasMultipleLayers'])
    has_single_exchange_region = SimulationVariable('hasSingleExchangeRegion', bool, None, ['hasSingleExchangeRegion'])

    is_drive_discrete_sites = SimulationVariable('hasDrivenDiscreteSites', bool, None, ['shouldDriveDiscreteSites'])
    is_drive_custom_position = SimulationVariable('hasCustomDrivePosition', bool, None, ['hasCustomDrivePosition'])
    is_drive_layers_all = SimulationVariable('hasDrivenAllLayers', bool, None, ['shouldDriveAllLayers'])
    is_drive_ends = SimulationVariable('hasDrivenBothSides', bool, None, ['shouldDriveBothSides'])
    is_drive_centre = SimulationVariable('hasDrivenCentre', bool, None, ['shouldDriveCentre'])
    is_drive_lhs = SimulationVariable('driveFromLhs', bool, None, ['shouldDriveLHS'])
    is_drive_rhs = SimulationVariable('hasDrivenRHS', bool, None, ['shouldDriveRHS'])

    @classmethod
    def get_metadata(cls, name):
        var = getattr(cls, name, None)
        if var and isinstance(var, SimulationVariable):
            return var.metadata
        raise AttributeError(f"{name} not found")


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
