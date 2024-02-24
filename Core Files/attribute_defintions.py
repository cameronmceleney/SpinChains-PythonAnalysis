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
    def __init__(self, key: str, var_dtype: Type[T], var_names: List[str] = None):
        """
        This class is used to define a parameter for a simulation. It is used to enforce type checking and
        provide metadata for the parameter.

        :param key: Variable's name which is used as a Dict key
        :param var_dtype:
        :param var_names: Spellings of the parameter in the simulation program (first name should be current name)
        """
        self._key: str = key
        self._value_dtype: Type[T] = var_dtype
        self._val_default: T = None
        self._var_names: List[str] = var_names if var_names is not None else []

        # Variable's name which should match the name in the simulation program
        self._name: str = self._var_names[0]

    def __get__(self, instance: Any, owner: Type) -> Any:
        """
        This method is called when the attribute is accessed from an instance or class.

        :param instance: The object itself.
        :param owner: The object's type.
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
        converted_value = self.convert_type(value)
        if not self.validate_type(converted_value):
            raise TypeError(f"Expected type {self._value_dtype.__name__}, got {type(converted_value).__name__}")
        instance.__dict__[self._key] = converted_value  # Modify instance's dictionary

    def validate_type(self, value: Any) -> bool:
        """
        This method is used to directly validate the type of the value being set.

        :param value:
        :return:
        """
        if self._value_dtype in (int, float) and isinstance(value, (int, float)):
            return True  # Allow int for float types or vice versa
        return isinstance(value, self._value_dtype)

    def convert_type_test(self, value: Any) -> T:

        if self._value_dtype is None:
            return value
        if self._value_dtype is bool:
            return bool(value) if isinstance(value, str) and value.lower() in ['true', '1', 'yes'] else False
        elif self._value_dtype in [int, float]:
            try:
                return self._value_dtype(value)
            except ValueError:
                return self._value_dtype(int(value) if value.isdigit() else float(value))
        return value  # For str and other types

    def convert_type(self, value: Any) -> T:
        if self._value_dtype is None:
            return value
        elif self._value_dtype is bool:
            true_values = ['true', '1', 'yes']
            false_values = ['false', '0', 'no']
            if isinstance(value, str):
                lowered_value = value.lower()
                if lowered_value in true_values:
                    return True
                elif lowered_value in false_values:
                    return False
                else:
                    raise ValueError(f"Unrecognized boolean string: '{value}'")
            else:
                return bool(value)
        elif self._value_dtype in [int, float]:
            try:
                return self._value_dtype(value)
            except ValueError:
                return self._value_dtype(int(value) if value.isdigit() else float(value))
        return value  # For str and other types


class SimulationVariableInstance(Generic[T]):
    """
    This class is used to provide a callable instance of a SimulationVariable. It is used to enforce type checking and
    provide metadata for the parameter.

    :return:
    """

    def __init__(self, sim_var: SimulationVariable[T], instance: Any):
        """

        :param sim_var:
        :param instance:
        """
        self._sim_var: SimulationVariable[T] = sim_var
        self._instance: Any = instance

    def __call__(self) -> T:
        """
        This method is called when the instance is called like a function, and returns its value.

        :return:
        """
        return self._instance.__dict__.get(self._sim_var._key, self._sim_var._val_default)

    def __getattr__(self, item: str) -> Callable:
        if hasattr(self._sim_var, item):
            return getattr(self._sim_var, item)

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

    @property
    def var_names(self) -> List[str]:
        return self._sim_var._var_names

    @property
    def name(self) -> str:
        return self._sim_var._name

    @property
    def key(self) -> str:
        return self._sim_var._key

    @property
    def dtype(self) -> Type[T]:
        return self._sim_var._value_dtype

    @property
    def default_value(self) -> T:
        return self._sim_var._val_default

    @property
    def param_metadata(self):
        """Access the class-level metadata specific to a given variable."""
        return self._instance.__class__.__name__, (self._sim_var.__class__.__name__, self._sim_var.__dict__)

    @property
    def instance_metadata(self):
        """OLD. Access all parameters in the current instance.

        Can be used from any parameter. USE CONTAINER.RETURN_DATA() INSTEAD."""
        return self._instance.__dict__

    @property
    def specific_metadata(self):
        """Return specific metadata for a variable."""
        return {self.key: self.__call__()}

    @property
    def metadata(self) -> Dict[str, T]:
        """Access the key metadata for a specific parameter in the given instance."""
        return {self._sim_var._key: self._instance.__dict__.get(self._sim_var._key, self._sim_var._val_default)}

    @property
    def object(self):
        return self._instance

    # Implementing the multiplication special method
    def __mul__(self, other):
        # Ensure the current instance's value is not None
        self_value = self()
        if self_value is None:
            raise ValueError(f"Cannot perform multiplication: '{self._sim_var._key}' value is None")

        # Handle multiplication with other instances or numeric types
        if isinstance(other, SimulationVariableInstance):
            other_value = other()
            # Check if the other instance's value is None
            if other_value is None:
                raise ValueError(f"Cannot perform multiplication with '{other._sim_var._key}': value is None")
        elif isinstance(other, (int, float)):
            other_value = other
        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{type(self_value)}' and '{type(other)}'")

        # Perform the multiplication if both values are valid
        return self_value * other_value

    # General approach for other arithmetic operations, e.g., addition
    def __add__(self, other):
        if isinstance(other, (int, float, SimulationVariableInstance)):
            other_value = other() if isinstance(other, SimulationVariableInstance) else other
            return self() + other_value
        raise TypeError(f"Unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")

    # Implement the reverse arithmetic operations to handle cases where the instance is on the right
    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    # Example of implementing subtraction
    def __sub__(self, other):
        if isinstance(other, (int, float, SimulationVariableInstance)):
            other_value = other() if isinstance(other, SimulationVariableInstance) else other
            return self() - other_value
        raise TypeError(f"Unsupported operand type(s) for -: '{type(self)}' and '{type(other)}'")

    def __truediv__(self, other):
        if isinstance(other, (int, float, SimulationVariableInstance)):
            other_value = other() if isinstance(other, SimulationVariableInstance) else other
            if other_value == 0:
                raise ZeroDivisionError("division by zero")
            return self() / other_value
        raise TypeError(f"Unsupported operand type(s) for /: '{type(self)}' and '{type(other)}'")

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, SimulationVariableInstance)):
            self_value = self()
            if self_value == 0:
                raise ZeroDivisionError("division by zero")
            other_value = other() if isinstance(other, SimulationVariableInstance) else other
            return other_value / self_value
        raise TypeError(f"Unsupported operand type(s) for /: '{type(other)}' and '{type(self)}'")

    def __pow__(self, other, modulo=None):
        if isinstance(other, (int, float, SimulationVariableInstance)):
            other_value = other() if isinstance(other, SimulationVariableInstance) else other
            # Handle modulo if provided for the pow() function
            if modulo is not None:
                return pow(self(), other_value, modulo)
            return self() ** other_value
        raise TypeError(f"Unsupported operand type(s) for ** or pow(): '{type(self)}' and '{type(other)}'")

    def __rpow__(self, other):
        if isinstance(other, (int, float, SimulationVariableInstance)):
            other_value = other() if isinstance(other, SimulationVariableInstance) else other
            return other_value ** self()
        raise TypeError(f"Unsupported operand type(s) for ** or pow(): '{type(other)}' and '{type(self)}'")

    def __rsub__(self, other):
        if isinstance(other, (int, float, SimulationVariableInstance)):
            other_value = other() if isinstance(other, SimulationVariableInstance) else other
            return other_value - self()
        raise TypeError(f"Unsupported operand type(s) for -: '{type(other)}' and '{type(self)}'")

    def __int__(self):
        # Attempt to return an integer representation of the instance's value
        value = self()
        try:
            return int(value)
        except ValueError as e:
            raise TypeError(f"Cannot convert {self.__class__.__name__} value to int: {value}") from e

    # You might also consider implementing other conversion methods as needed, such as:
    def __float__(self):
        value = self()
        try:
            return float(value)
        except ValueError as e:
            raise TypeError(f"Cannot convert {self.__class__.__name__} value to float: {value}") from e



class SimulationVariableContainerMeta(type):
    def __new__(cls, name: str, bases: tuple, attrs: dict):

        container_type = attrs.get('container_type', None)

        _container_var_objs: Dict[Any, SimulationVariable[T]] = {}  # test
        _container_var_dicts:  Dict[Any, Dict[str, str | Type[T] | list[str]]] = {}
        sim_vars = {}  # Unified dictionary fpr easy mapping to container type-specific attributes

        for base in bases:
            # Direct access is necessary for functionality, but I dislike the compiler complaining about it
            if hasattr(base, '_container_var_objs'):
                _container_var_objs.update(base._container_var_objs)
            if hasattr(base, '_container_var_dicts'):
                _container_var_dicts.update(base._container_var_dicts)
                # Below two lines are how to do this without direct access
                # base_sim_params_typed = getattr(base, '_container_var_objs', {})
                # _container_var_objs.update(base_sim_params_typed)

        for key, val in attrs.items():
            if isinstance(val, SimulationVariable):
                _container_var_objs[key] = val
                _container_var_dicts[key] = {'key': val._key, 'dtype': val._value_dtype,
                                  'name': val._name,  'var_names': val._var_names}
                sim_vars[key] = _container_var_dicts[key]

        # Set annotations for type hinting
        _container_annotations = attrs.get('__annotations__', {})
        for var_name, typed_var in _container_var_objs.items():
            _container_annotations[var_name] = typed_var._value_dtype

        # Assign the collected dictionaries to class attributes
        attrs['_container_var_objs'] = _container_var_objs
        attrs['_container_var_dicts'] = _container_var_dicts
        attrs['__annotations__'] = _container_annotations

        # Depending on the container type, assign the simulation variables to the appropriate attribute
        if container_type == 'parameters':
            attrs['all_parameters'] = sim_vars
        elif container_type == 'flags':
            attrs['all_flags'] = sim_vars

        def return_data(self, exclude_none_type: bool = True):
            """Output all the data in a given container as a Dict"""
            data = {}
            for var_key, sim_var in self._container_var_objs.items():
                value = getattr(self, var_key)()
                if exclude_none_type and value is None:
                    continue
                data[var_key] = value
            return data

        def update_variables(self, data_dict):
            for k, v in data_dict.items():
                if k in self._container_var_objs:
                    sim_var = self._container_var_objs[k]
                    if not sim_var.validate_type(v):
                        raise TypeError(
                            f"Incorrect type for {k}. Expected {sim_var._value_dtype.__name__}, got {type(v).__name__}.")
                    setattr(self, k, v)

        def update_with_dict(self, data_dict, use_paired_variables=True):
            """Update container attributes from a dictionary with type checking and apply special pairings correctly."""
            special_pairings = getattr(self.__class__, '_special_pairings', {})

            # Update all variables as per data_dict without yet applying pairings
            self.update_variables(data_dict)

            # Apply special pairings if enabled
            if use_paired_variables:
                for pair_descriptor, (key1, key2, was_paired) in special_pairings.items():
                    if key1 in data_dict or key2 in data_dict:
                        key1_val = getattr(self, key1, None)()
                        key2_val = getattr(self, key2, None)()

                        # Determine which value to update based on which keys are present in the input dict
                        if not was_paired:
                            if key1 in data_dict and key2 not in data_dict:
                                # Invert key1's value for key2
                                new_val_for_key2 = not key1_val if isinstance(key1_val, bool) else key1_val
                                setattr(self, key2, new_val_for_key2)
                            elif key2 in data_dict and key1 not in data_dict:
                                # Invert key2's value for key1
                                new_val_for_key1 = not key2_val if isinstance(key2_val, bool) else key2_val
                                setattr(self, key1, new_val_for_key1)

                            # Mark this pairing as processed
                            special_pairings[pair_descriptor][2] = True
#

        def update_with_container(self, other_container, use_paired_variables=True):
            """Update container attributes from another container instance, with an option to apply special pairings."""
            data_dict = other_container.return_data()
            self.update_with_dict(data_dict, apply_special_pairings=use_paired_variables)

        # Inject the return_data method into the class
        attrs['return_data'] = return_data
        attrs['update_with_dict'] = update_with_dict
        attrs['update_variables'] = update_variables
        attrs['update_with_container'] = update_with_container

        return super().__new__(cls, name, bases, attrs)

    def __getitem__(cls, key: str):
        if key in cls.__dict__.get('_container_var_dicts', {}):
            return cls.__dict__['_container_var_dicts'][key]
        elif key in cls._container_var_dicts:
            return getattr(cls, key)
        elif key in cls.__dict__.get('_container_var_objs', {}):
            return cls.__dict__['_container_var_objs'][key]
        raise KeyError(f"{key} is not available in {cls.__name__}")


class SimulationParametersContainer(metaclass=SimulationVariableContainerMeta):
    container_type = 'parameters'

    bias_zeeman_static = SimulationVariable('bias_zeeman_static', float,
                                            ['staticZeemanStrength',
                                             'staticBiasField', 'Static Bias Field'])

    bias_zeeman_oscillating_1 = SimulationVariable('bias_zeeman_oscillating_1', float,
                                                   ['oscillatingZeemanStrength',
                                                    'dynamicBiasField', 'Dynamic Bias Field'])

    shockwave_scaling = SimulationVariable('shockwave_scaling', float,
                                           ['shockwaveScaling',
                                            'dynamicBiasFieldScaleFactor', 'Dynamic Bias Field Scale Factor'])

    bias_zeeman_oscillating_2 = SimulationVariable('bias_zeeman_oscillating_2', float,
                                                   ['shockwaveMax',
                                                    'oscillatingZeemanStrength2', 'secondDynamicBiasField',
                                                    'Second Dynamic Bias Field'])

    driving_freq = SimulationVariable('driving_freq', float,
                                      ['drivingFreq',
                                       'drivingFrequency', 'Driving Frequency'])

    driving_region_lhs = SimulationVariable('driving_region_lhs', int,
                                            ['drivingRegionLhs',
                                             'drivingRegionStartSite', 'Driving Region Start Site'])

    driving_region_rhs = SimulationVariable('driving_region_rhs', int,
                                            ['drivingRegionRhs',
                                             'drivingRegionEndSite', 'Driving Region End Site'])

    driving_region_width = SimulationVariable('driving_region_width', int,
                                              ['drivingRegionWidth',
                                               'Driving Region Width'])

    sim_time_max = SimulationVariable('sim_time_max', float,
                                      ['maxSimTime',
                                       'Max. Sim. Time'])

    exchange_heisenberg_min = SimulationVariable('exchange_heisenberg_min', float, ['exchangeEnergyMin',
                                                                                    'exchangeHeisenbergMin',
                                                                                    'minExchangeVal',
                                                                                    'Min. Exchange Val'])

    exchange_heisenberg_max = SimulationVariable('exchange_heisenberg_max', float,
                                                 ['heisenbergExchangeMax',
                                                  'exchangeEnergyMax', 'maxExchangeVal', 'Max. Exchange Val'])

    iteration_total = SimulationVariable('iteration_total', float, ['iterationEnd', 'maxIterations', 'Max. Iterations'])

    num_dp_per_site = SimulationVariable('num_dp_per_site', float,
                                         ['numberOfDataPoints',
                                          'numDatapoints', 'No. DataPoints', 'numDataPointsPerSite'])

    num_sites_chain = SimulationVariable('num_sites_chain', int, ['numSpinsInChain',
                                                                  'numSpinsInChain', 'No. Spins in Chain'])

    num_sites_abc = SimulationVariable('num_sites_abc', int,
                                       ['numSpinsInABC', 'numDampedSpinsPerSide', 'No. Damped Spins'])

    num_sites_total = SimulationVariable('num_sites_total', int, ['systemTotalSpins',
                                                                  'numTotalSpins', 'No. Total Spins'])

    stepsize = SimulationVariable('stepsize', float, ['stepsize', 'Stepsize'])

    gilbert_chain = SimulationVariable('gilbert_chain', float,
                                       ['gilbertDampingFactor', 'Gilbert Damping Factor'])

    gyro_mag = SimulationVariable('gyro_mag', float, ['gyroMagConst', 'gyroRatio', 'Gyromagnetic Ratio'])

    shockwave_time_gradient = SimulationVariable('shockwave_time_gradient', float,
                                                 ['shockwaveGradientTime', 'Shockwave Gradient Time'])

    shockwave_time_application = SimulationVariable('shockwave_time_application', float,
                                                    ['shockwaveApplicationTime', 'Shockwave Application Time'])

    gilbert_abc_outer = SimulationVariable('gilbert_abc_outer', float,
                                           ['gilbertABCOuter', 'abcDampingLower', 'Lower ABC Damping'])

    gilbert_abc_inner = SimulationVariable('gilbert_abc_inner', float,
                                           ['gilbertABCInner', 'abcDampingUpper', 'Upper ABC Damping'])

    exchange_dmi_constant = SimulationVariable('exchange_dmi_constant', float,
                                               ['dmiConstant', 'DMI Constant', 'Dmi Constant'])

    sat_mag = SimulationVariable('sat_mag', float, ['satMag', 'saturationMagnetisation', 'Saturation Magnetisation'])

    exchange_stiffness = SimulationVariable('exchange_stiffness', float,
                                            ['exchangeStiffness', 'Exchange Stiffness'])

    anisotropy_field = SimulationVariable('anisotropy_field', float,
                                          ['anisotropyField', 'anisotropyShapeField', 'Anisotropy Shape Field'])

    lattice_constant = SimulationVariable('lattice_constant', float, ['latticeConstant', 'Lattice Constant'])

    def __getitem__(self, key: str):
        # Access class attributes directly
        if key in self._container_var_dicts:
            return key, self._container_var_dicts[key]
        raise KeyError(f"{key} is not available in this container.")

    def return_data(self, *args, **kwargs):
        """ Real implementation is injected by the metaclass"""
        pass

    def update_with_dict(self, *args, **kwargs):
        """Update container attributes from a dictionary with type checking."""
        pass

    def update_with_container(self, *args, **kwargs):
        """Update container attributes from another container instance."""
        pass


class SimulationFlagsContainer(metaclass=SimulationVariableContainerMeta):
    container_type = 'flags'

    # If one is set to True, the other should be set to False and vice versa
    _special_pairings = {
        'drive_side': ['is_drive_lhs', 'is_drive_rhs', False],
        'sim_eqn': ['has_llg', 'has_sllg', False],
    }

    numerical_method = SimulationVariable('numerical_method', str,
                                          ['numericalMethodUsed', 'numericalMethod', 'Numerical Method Used'])

    has_llg = SimulationVariable('has_llg', bool,
                                 ['shouldUseLLG',
                                  'shouldUseLlg', 'hasLLG', 'hasLlg', 'usingMagdynamics', 'Using magDynamics'])

    has_sllg = SimulationVariable('has_sllg', bool, ['hasSLLG', 'shouldUseSLLG', 'shouldUseSllg'])

    has_shockwave = SimulationVariable('has_shockwave', bool, ['hasShockwave', 'usingShockwave', 'Using Shockwave'])

    has_dipolar = SimulationVariable('has_dipolar', bool, ['hasDipolar', 'Has Dipolar'])

    has_dmi = SimulationVariable('has_dmi', bool, ['hasDMI', 'hasDmi', 'Has DMI'])

    has_stt = SimulationVariable('has_stt', bool, ['hasSTT', 'hasStt', 'Has STT'])

    has_bias_zeeman_static = SimulationVariable('has_bias_zeeman_static', bool,
                                                ['hasStaticZeeman', 'hasZeeman', 'Has Zeeman'])

    has_bias_zeeman_oscillating_static = SimulationVariable('has_bias_zeeman_oscillating_static', bool,
                                                            ['isOscillatingZeemanStatic', 'hasStaticDrive',
                                                             'Has Static Drive'])

    has_demag_1d_thin_film = SimulationVariable('has_demag_1d_thin_film', bool, ['hasDemag1DThinFilm'])

    has_demag_intense = SimulationVariable('has_demag_intense', bool, ['hasDemagIntense', 'Has Demag Intense'])

    has_demag_fft = SimulationVariable('has_demag_fft', bool, ['hasDemagFFT', 'hasDemagFft', 'Has Demag FFT'])

    has_anisotropy_shape = SimulationVariable('has_anisotropy_shape', bool,
                                              ['hasShapeAnisotropy', 'Has Shape Anisotropy'])

    has_multiple_layers = SimulationVariable('has_multiple_layers', bool, ['hasMultipleLayers'])

    has_single_exchange_region = SimulationVariable('has_single_exchange_region', bool, ['hasSingleExchangeRegion'])

    is_drive_discrete_sites = SimulationVariable('is_drive_discrete_sites', bool,
                                                 ['hasDrivenDiscreteSites', 'shouldDriveDiscreteSites'])

    is_drive_custom_position = SimulationVariable('is_drive_custom_position', bool, ['hasCustomDrivePosition'])

    is_drive_layers_all = SimulationVariable('is_drive_layers_all', bool,
                                             ['hasDrivenAllLayers', 'shouldDriveAllLayers'])
    is_drive_ends = SimulationVariable('is_drive_ends', bool, ['hasDrivenBothSides', 'shouldDriveBothSides'])

    is_drive_centre = SimulationVariable('is_drive_centre', bool, ['hasDrivenCentre', 'shouldDriveCentre'])

    is_drive_lhs = SimulationVariable('is_drive_lhs', bool,
                                      ['driveFromLhs', 'shouldDriveLHS', 'driveFromLhs', 'Drive from LHS'])

    is_drive_rhs = SimulationVariable('is_drive_rhs', bool, ['hasDrivenRHS', 'shouldDriveRHS'])

    def __getitem__(self, key: str):
        # Access class attributes directly
        if key in self._container_var_dicts:
            return self._container_var_dicts[key]
        raise KeyError(f"{key} is not available in this container.")

    def update_with_dict(self, *args, **kwargs):
        """Update container attributes from a dictionary with type checking."""
        pass

    def update_with_container(self, *args, **kwargs):
        """Update container attributes from another container instance."""
        pass


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

    def __init__(self, name, bases, attrs):
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
