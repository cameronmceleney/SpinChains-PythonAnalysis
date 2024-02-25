# -*- coding: utf-8 -*-

# -------------------------- Preprocessing Directives -------------------------

# Full packages
import operator

# Specific functions from packages
from typing import Dict, List, Type, Any, TypeVar, Generic, Callable

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

    def convert_type(self, value: Any) -> T:
        if value is None:
            return value

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

        Note that `type(value) == <class 'attribute_defintions.SimulationVariableInstance'>` while the return of
        `__call__` is going to whatever the `_var_dtype` of the given `_key` is.
        """
        value = self._instance.__dict__.get(self._sim_var._key, self._sim_var._val_default)
        #print('here', value, self._sim_var._key, self._sim_var._value_dtype)
        return self._sim_var.convert_type(value)

    def __getattr__(self, item: str) -> Callable:
        if hasattr(self._sim_var, item):
            return getattr(self._sim_var, item)

        value = self.__call__()

        if hasattr(value, item):
            return getattr(value, item)

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

    def mul_test(self, other):
        self_value = self()
        if self_value is None:
            raise ValueError(f"Attempted to multiply a 'None' value for '{self._sim_var._key}'.")

        if isinstance(other, SimulationVariableInstance):
            other_value = other()
            if other_value is None:
                raise ValueError(f"Attempted to multiply by a 'None' value from '{other._sim_var._key}'.")
        elif isinstance(other, (int, float)):
            other_value = other
        elif isinstance(other, list):
            # Define a specific operation for lists, if applicable
            # Example: return a list where each element is multiplied by self_value
            return [self_value * elem for elem in other]
        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{type(self_value)}' and '{type(other)}'.")

        return self_value * other_value

    def __perform_operation(self, other, operation):
        self_value = self()
        if self_value is None:
            raise ValueError(f"Attempted operation with a 'None' value for '{self._sim_var._key}'.")

        if isinstance(other, SimulationVariableInstance):
            other_value = other()
            if other_value is None:
                raise ValueError(f"Attempted operation with a 'None' value from '{other._sim_var._key}'.")
        elif isinstance(other, (int, float)):
            other_value = other
        elif isinstance(other, list) and operation == operator.mul:  # Specific case for multiplication
            return [self_value * elem for elem in other]
        else:
            raise TypeError(
                f"Unsupported operand type(s) for {operation.__name__}: '{type(self_value)}' and '{type(other)}'.")

        return operation(self_value, other_value)

    def __mul__(self, other):
        return self.__perform_operation(other, operator.mul)

    def __add__(self, other):
        return self.__perform_operation(other, operator.add)

    def __sub__(self, other):
        return self.__perform_operation(other, operator.sub)

    def __truediv__(self, other):
        return self.__perform_operation(other, operator.truediv)

    def __pow__(self, other, modulo=None):
        self_value = self()
        other_value = other() if isinstance(other, SimulationVariableInstance) else other
        if modulo is None:
            return pow(self_value, other_value)
        else:
            return pow(self_value, other_value, modulo)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rpow__(self, other):
        return self.__perform_operation(other, operator.pow)

    def __rtruediv__(self, other):
        return self.__perform_operation(other, operator.truediv)

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
                            f"Incorrect type for {k}. Expected {sim_var._value_dtype.__name__}, "
                            f"got {type(v).__name__}.")
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

    def return_data(self, *args, **kwargs):
        """ Real implementation is injected by the metaclass"""
        pass

    def update_with_dict(self, *args, **kwargs):
        """Update container attributes from a dictionary with type checking."""
        pass

    def update_with_container(self, *args, **kwargs):
        """Update container attributes from another container instance."""
        pass
