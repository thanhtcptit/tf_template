# -*- coding: utf-8 -*-
import os
import copy
import json
import inspect
import logging

from overrides import overrides
from collections import defaultdict
from typing import Any, Dict, List, TypeVar, Type, Union, cast, Tuple, Set
from collections import MutableMapping, OrderedDict

from src.utils.file_utils import cached_path

# _jsonnet doesn't work on Windows, so we have to use fakes.
try:
    from _jsonnet import evaluate_file, evaluate_snippet
except ImportError:
    def evaluate_file(filename: str, **_kwargs) -> str:
        logger.warning(f"_jsonnet not loaded, treating {filename} as json")
        with open(filename, 'r') as evaluation_file:
            return evaluation_file.read()

    def evaluate_snippet(_filename: str, expr: str, **_kwargs) -> str:
        logger.warning(f"_jsonnet not loaded, treating snippet as json")
        return expr

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

T = TypeVar('T')

# If a function parameter has no default value specified,
# this is what the inspect module returns.
_NO_DEFAULT = inspect.Parameter.empty  # pylint: disable=invalid-name


class ConfigurationError(Exception):
    def __init__(self, message):
        super(ConfigurationError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


def unflatten(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a "flattened" dict with compound keys, e.g.
        {"a.b": 0}
    unflatten it:
        {"a": {"b": 0}}
    """
    unflat: Dict[str, Any] = {}

    for compound_key, value in flat_dict.items():
        curr_dict = unflat
        parts = compound_key.split(".")
        for key in parts[:-1]:
            curr_value = curr_dict.get(key)
            if key not in curr_dict:
                curr_dict[key] = {}
                curr_dict = curr_dict[key]
            elif isinstance(curr_value, dict):
                curr_dict = curr_value
            else:
                raise ConfigurationError("flattened dictionary is invalid")
        if not isinstance(curr_dict, dict) or parts[-1] in curr_dict:
            raise ConfigurationError("flattened dictionary is invalid")
        else:
            curr_dict[parts[-1]] = value

    return unflat


def with_fallback(preferred: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dicts, preferring values from `preferred`.
    """
    preferred_keys = set(preferred.keys())
    fallback_keys = set(fallback.keys())
    common_keys = preferred_keys & fallback_keys

    merged: Dict[str, Any] = {}

    for key in preferred_keys - fallback_keys:
        merged[key] = copy.deepcopy(preferred[key])
    for key in fallback_keys - preferred_keys:
        merged[key] = copy.deepcopy(fallback[key])

    for key in common_keys:
        preferred_value = preferred[key]
        fallback_value = fallback[key]

        if isinstance(preferred_value, dict) and isinstance(fallback_value, dict):
            merged[key] = with_fallback(preferred_value, fallback_value)
        else:
            merged[key] = copy.deepcopy(preferred_value)

    return merged


def parse_overrides(serialized_overrides: str) -> Dict[str, Any]:
    if serialized_overrides:
        ext_vars = dict(os.environ)
        return unflatten(json.loads(evaluate_snippet("", serialized_overrides, ext_vars=ext_vars)))
    else:
        return {}


class Params(MutableMapping):
    """
    Represents a parameter dictionary with a history, and contains other functionality around
    parameter passing and validation for AllenNLP.

    There are currently two benefits of a ``Params`` object over a plain dictionary for parameter
    passing:

    #. We handle a few kinds of parameter validation, including making sure that parameters
       representing discrete choices actually have acceptable values, and making sure no extra
       parameters are passed.
    #. We log all parameter reads, including default values.  This gives a more complete
       specification of the actual parameters used than is given in a JSON file, because
       those may not specify what default values were used, whereas this will log them.

    The convention for using a ``Params`` object in AllenNLP is that you will consume the parameters
    as you read them, so that there are none left when you've read everything you expect.  This
    lets us easily validate that you didn't pass in any `extra` parameters, just by making sure
    that the parameter dictionary is empty.  You should do this when you're done handling
    parameters, by calling :func:`Params.assert_empty`.
    """

    # This allows us to check for the presence of "None" as a default argument,
    # which we require because we make a distinction bewteen passing a value of "None"
    # and passing no value to the default parameter of "pop".
    DEFAULT = object()

    def __init__(self,
                 params: Dict[str, Any],
                 history: str = "",
                 loading_from_archive: bool = False,
                 files_to_archive: Dict[str, str] = None) -> None:
        self.params = _replace_none(params)
        self.history = history
        self.loading_from_archive = loading_from_archive
        self.files_to_archive = {} if files_to_archive is None else files_to_archive

    def add_file_to_archive(self, name: str) -> None:
        """
        Any class in its ``from_params`` method can request that some of its
        input files be added to the archive by calling this method.

        For example, if some class ``A`` had an ``input_file`` parameter, it could call

        ```
        params.add_file_to_archive("input_file")
        ```

        which would store the supplied value for ``input_file`` at the key
        ``previous.history.and.then.input_file``. The ``files_to_archive`` dict
        is shared with child instances via the ``_check_is_dict`` method, so that
        the final mapping can be retrieved from the top-level ``Params`` object.

        NOTE: You must call ``add_file_to_archive`` before you ``pop()``
        the parameter, because the ``Params`` instance looks up the value
        of the filename inside itself.

        If the ``loading_from_archive`` flag is True, this will be a no-op.
        """
        if not self.loading_from_archive:
            self.files_to_archive[f"{self.history}{name}"] = cached_path(
                self.get(name))

    @overrides
    def pop(self, key: str, default: Any = DEFAULT) -> Any:
        """
        Performs the functionality associated with dict.pop(key), along with checking for
        returned dictionaries, replacing them with Param objects with an updated history.

        If ``key`` is not present in the dictionary, and no default was specified, we raise a
        ``ConfigurationError``, instead of the typical ``KeyError``.
        """
        if default is self.DEFAULT:
            try:
                value = self.params.pop(key)
            except KeyError:
                raise ConfigurationError(
                    "key \"{}\" is required at location \"{}\"".format(key, self.history))
        else:
            value = self.params.pop(key, default)
        if not isinstance(value, dict):
            logger.info(self.history + key + " = " +
                        str(value))  # type: ignore
        return self._check_is_dict(key, value)

    def pop_int(self, key: str, default: Any = DEFAULT) -> int:
        """
        Performs a pop and coerces to an int.
        """
        value = self.pop(key, default)
        if value is None:
            return None
        else:
            return int(value)

    def pop_float(self, key: str, default: Any = DEFAULT) -> float:
        """
        Performs a pop and coerces to a float.
        """
        value = self.pop(key, default)
        if value is None:
            return None
        else:
            return float(value)

    def pop_bool(self, key: str, default: Any = DEFAULT) -> bool:
        """
        Performs a pop and coerces to a bool.
        """
        value = self.pop(key, default)
        if value is None:
            return None
        elif isinstance(value, bool):
            return value
        elif value == "true":
            return True
        elif value == "false":
            return False
        else:
            raise ValueError("Cannot convert variable to bool: " + value)

    @overrides
    def get(self, key: str, default: Any = DEFAULT):
        """
        Performs the functionality associated with dict.get(key) but also checks for returned
        dicts and returns a Params object in their place with an updated history.
        """
        if default is self.DEFAULT:
            try:
                value = self.params.get(key)
            except KeyError:
                raise ConfigurationError(
                    "key \"{}\" is required at location \"{}\"".format(key, self.history))
        else:
            value = self.params.get(key, default)
        return self._check_is_dict(key, value)

    def pop_choice(self, key: str, choices: List[Any],
                   default_to_first_choice: bool = False) -> Any:
        """
        Gets the value of ``key`` in the ``params`` dictionary, ensuring that the value is one of
        the given choices. Note that this `pops` the key from params, modifying the dictionary,
        consistent with how parameters are processed in this codebase.

        Parameters
        ----------
        key: str
            Key to get the value from in the param dictionary
        choices: List[Any]
            A list of valid options for values corresponding to ``key``.  For example, if you're
            specifying the type of encoder to use for some part of your model, the choices might be
            the list of encoder classes we know about and can instantiate.  If the value we find in
            the param dictionary is not in ``choices``, we raise a ``ConfigurationError``, because
            the user specified an invalid value in their parameter file.
        default_to_first_choice: bool, optional (default=False)
            If this is ``True``, we allow the ``key`` to not be present in the parameter
            dictionary.  If the key is not present, we will use the return as the value the first
            choice in the ``choices`` list.  If this is ``False``, we raise a
            ``ConfigurationError``, because specifying the ``key`` is required (e.g., you `have` to
            specify your model class when running an experiment, but you can feel free to use
            default settings for encoders if you want).
        """
        default = choices[0] if default_to_first_choice else self.DEFAULT
        value = self.pop(key, default)
        if value not in choices:
            key_str = self.history + key
            message = '%s not in acceptable choices for %s: %s' % (
                value, key_str, str(choices))
            raise ConfigurationError(message)
        return value

    def as_dict(self, quiet=False):
        """
        Sometimes we need to just represent the parameters as a dict, for instance when we pass
        them to a Keras layer(so that they can be serialised).

        Parameters
        ----------
        quiet: bool, optional (default = False)
            Whether to log the parameters before returning them as a dict.
        """
        if quiet:
            return self.params

        def log_recursively(parameters, history):
            for key, value in parameters.items():
                if isinstance(value, dict):
                    new_local_history = history + key + "."
                    log_recursively(value, new_local_history)
                else:
                    logger.info(history + key + " = " + str(value))

        logger.info("Converting Params object to dict; logging of default "
                    "values will not occur when dictionary parameters are "
                    "used subsequently.")
        logger.info("CURRENTLY DEFINED PARAMETERS: ")
        log_recursively(self.params, self.history)
        return self.params

    def as_flat_dict(self):
        """
        Returns the parameters of a flat dictionary from keys to values.
        Nested structure is collapsed with periods.
        """
        flat_params = {}

        def recurse(parameters, path):
            for key, value in parameters.items():
                newpath = path + [key]
                if isinstance(value, dict):
                    recurse(value, newpath)
                else:
                    flat_params['.'.join(newpath)] = value

        recurse(self.params, [])
        return flat_params

    def duplicate(self) -> 'Params':
        """
        Uses ``copy.deepcopy()`` to create a duplicate (but fully distinct)
        copy of these Params.
        """
        return Params(copy.deepcopy(self.params))

    def assert_empty(self, class_name: str):
        """
        Raises a ``ConfigurationError`` if ``self.params`` is not empty.  We take ``class_name`` as
        an argument so that the error message gives some idea of where an error happened, if there
        was one.  ``class_name`` should be the name of the `calling` class, the one that got extra
        parameters (if there are any).
        """
        if self.params:
            raise ConfigurationError(
                "Extra parameters passed to {}: {}".format(class_name, self.params))

    def __getitem__(self, key):
        if key in self.params:
            return self._check_is_dict(key, self.params[key])
        else:
            raise KeyError

    def __setitem__(self, key, value):
        self.params[key] = value

    def __delitem__(self, key):
        del self.params[key]

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    def _check_is_dict(self, new_history, value):
        if isinstance(value, dict):
            new_history = self.history + new_history + "."
            return Params(value,
                          history=new_history,
                          loading_from_archive=self.loading_from_archive,
                          files_to_archive=self.files_to_archive)
        if isinstance(value, list):
            value = [self._check_is_dict(
                new_history + '.list', v) for v in value]
        return value

    @staticmethod
    def from_file(params_file: str, params_overrides: str = "", ext_vars: dict = None) -> 'Params':
        """
        Load a `Params` object from a configuration file.

        Parameters
        ----------
        params_file : ``str``
            The path to the configuration file to load.
        params_overrides : ``str``, optional
            A dict of overrides that can be applied to final object.
            e.g. {"model.embedding_dim": 10}
        ext_vars : ``dict``, optional
            Our config files are Jsonnet, which allows specifying external variables
            for later substitution. Typically we substitute these using environment
            variables; however, you can also specify them here, in which case they
            take priority over environment variables.
            e.g. {"HOME_DIR": "/Users/allennlp/home"}
        """
        if ext_vars is None:
            ext_vars = {}

        # redirect to cache, if necessary
        params_file = cached_path(params_file)
        ext_vars = {**dict(os.environ), **ext_vars}

        file_dict = json.loads(evaluate_file(params_file, ext_vars=ext_vars))

        overrides_dict = parse_overrides(params_overrides)
        param_dict = with_fallback(
            preferred=overrides_dict, fallback=file_dict)

        return Params(param_dict)

    def to_file(self, params_file: str, preference_orders: List[List[str]] = None) -> None:
        with open(params_file, "w") as handle:
            json.dump(self.as_ordered_dict(
                preference_orders), handle, indent=4)

    def as_ordered_dict(self, preference_orders: List[List[str]] = None) -> OrderedDict:
        """
        Returns Ordered Dict of Params from list of partial order preferences.

        Parameters
        ----------
        preference_orders: List[List[str]], optional
            ``preference_orders`` is list of partial preference orders. ["A", "B", "C"] means
            "A" > "B" > "C". For multiple preference_orders first will be considered first.
            Keys not found, will have last but alphabetical preference. Default Preferences:
            ``[["dataset_reader", "iterator", "model", "train_data_path", "validation_data_path",
            "test_data_path", "trainer", "vocabulary"], ["type"]]``
        """
        params_dict = self.as_dict(quiet=True)
        if not preference_orders:
            preference_orders = []
            preference_orders.append(["dataset_reader", "iterator", "model",
                                      "train_data_path", "validation_data_path", "test_data_path",
                                      "trainer", "vocabulary"])
            preference_orders.append(["type"])

        def order_func(key):
            # Makes a tuple to use for ordering.  The tuple is an index into each of the
            # `preference_orders`, followed by the key itself.  This gives us integer sorting if
            # you have a key in one of the `preference_orders`, followed by alphabetical ordering
            # if not.
            order_tuple = [order.index(key) if key in order else len(
                order) for order in preference_orders]
            return order_tuple + [key]

        def order_dict(dictionary, order_func):
            # Recursively orders dictionary according to scoring order_func
            result = OrderedDict()
            for key, val in sorted(dictionary.items(), key=lambda item: order_func(item[0])):
                result[key] = order_dict(
                    val, order_func) if isinstance(val, dict) else val
            return result

        return order_dict(params_dict, order_func)


def pop_choice(params: Dict[str, Any],
               key: str,
               choices: List[Any],
               default_to_first_choice: bool = False,
               history: str = "?.") -> Any:
    """
    Performs the same function as :func:`Params.pop_choice`, but is required in order to deal with
    places that the Params object is not welcome, such as inside Keras layers.  See the docstring
    of that method for more detail on how this function works.

    This method adds a ``history`` parameter, in the off-chance that you know it, so that we can
    reproduce :func:`Params.pop_choice` exactly.  We default to using "?." if you don't know the
    history, so you'll have to fix that in the log if you want to actually recover the logged
    parameters.
    """
    value = Params(params, history).pop_choice(
        key, choices, default_to_first_choice)
    return value


def _replace_none(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    for key in dictionary.keys():
        if dictionary[key] == "None":
            dictionary[key] = None
        elif isinstance(dictionary[key], dict):
            dictionary[key] = _replace_none(dictionary[key])
    return dictionary


def takes_arg(obj, arg: str) -> bool:
    """
    Checks whether the provided obj takes a certain arg.
    If it's a class, we're really checking whether its constructor does.
    If it's a function or method, we're checking the object itself.
    Otherwise, we raise an error.
    """
    if inspect.isclass(obj):
        signature = inspect.signature(obj.__init__)
    elif inspect.ismethod(obj) or inspect.isfunction(obj):
        signature = inspect.signature(obj)
    else:
        raise ConfigurationError(f"object {obj} is not callable")
    return arg in signature.parameters


def remove_optional(annotation: type):
    """
    Optional[X] annotations are actually represented as Union[X, NoneType].
    For our purposes, the "Optional" part is not interesting, so here we
    throw it away.
    """
    origin = getattr(annotation, '__origin__', None)
    args = getattr(annotation, '__args__', ())
    if origin == Union and len(args) == 2 and isinstance(None, args[1]):
        return args[0]
    else:
        return annotation


def create_kwargs(cls: Type[T], params: Params, **extras) -> Dict[str, Any]:
    """
    Given some class, a `Params` object, and potentially other keyword arguments,
    create a dict of keyword args suitable for passing to the class's constructor.

    The function does this by finding the class's constructor, matching the constructor
    arguments to entries in the `params` object, and instantiating values for the parameters
    using the type annotation and possibly a from_params method.

    Any values that are provided in the `extras` will just be used as is.
    For instance, you might provide an existing `Vocabulary` this way.
    """
    # Get the signature of the constructor.
    signature = inspect.signature(cls.__init__)
    kwargs: Dict[str, Any] = {}

    # Iterate over all the constructor parameters and their annotations.
    for name, param in signature.parameters.items():
        # Skip "self". You're not *required* to call the first parameter "self",
        # so in theory this logic is fragile, but if you don't call the self parameter
        # "self" you kind of deserve what happens.
        if name == "self":
            continue

        # If the annotation is a compound type like typing.Dict[str, int],
        # it will have an __origin__ field indicating `typing.Dict`
        # and an __args__ field indicating `(str, int)`. We capture both.
        annotation = remove_optional(param.annotation)
        origin = getattr(annotation, '__origin__', None)
        args = getattr(annotation, '__args__', [])

        # The parameter is optional if its default value is not the "no default" sentinel.
        default = param.default
        optional = default != _NO_DEFAULT

        # Some constructors expect extra non-parameter items, e.g. vocab: Vocabulary.
        # We check the provided `extras` for these and just use them if they exist.
        if name in extras:
            kwargs[name] = extras[name]

        # The next case is when the parameter type is itself constructible from_params.
        elif hasattr(annotation, 'from_params'):
            if name in params:
                # Our params have an entry for this, so we use that.
                subparams = params.pop(name)

                if takes_arg(annotation.from_params, 'extras'):
                    # If annotation.params accepts **extras, we need to pass them all along.
                    # For example, `BasicTextFieldEmbedder.from_params` requires a Vocabulary
                    # object, but `TextFieldEmbedder.from_params` does not.
                    subextras = extras
                else:
                    # Otherwise, only supply the ones that are actual args; any additional ones
                    # will cause a TypeError.
                    subextras = {k: v for k, v in extras.items(
                    ) if takes_arg(annotation.from_params, k)}

                # In some cases we allow a string instead of a param dict, so
                # we need to handle that case separately.
                if isinstance(subparams, str):
                    kwargs[name] = annotation.by_name(subparams)()
                else:
                    kwargs[name] = annotation.from_params(params=subparams, **subextras)
            elif not optional:
                # Not optional and not supplied, that's an error!
                raise ConfigurationError(f"expected key {name} for {cls.__name__}")
            else:
                kwargs[name] = default

        # If the parameter type is a Python primitive, just pop it off
        # using the correct casting pop_xyz operation.
        elif annotation == str:
            kwargs[name] = (params.pop(name, default)
                            if optional
                            else params.pop(name))
        elif annotation == int:
            kwargs[name] = (params.pop_int(name, default)
                            if optional
                            else params.pop_int(name))
        elif annotation == bool:
            kwargs[name] = (params.pop_bool(name, default)
                            if optional
                            else params.pop_bool(name))
        elif annotation == float:
            kwargs[name] = (params.pop_float(name, default)
                            if optional
                            else params.pop_float(name))

        # This is special logic for handling types like Dict[str, TokenIndexer],
        # List[TokenIndexer], Tuple[TokenIndexer, Tokenizer], and Set[TokenIndexer],
        # which it creates by instantiating each value from_params and returning the resulting
        # structure.
        elif origin in (Dict, dict) and len(args) == 2 and hasattr(args[-1], 'from_params'):
            value_cls = annotation.__args__[-1]

            value_dict = {}

            for key, value_params in params.pop(name, Params({})).items():
                value_dict[key] = value_cls.from_params(params=value_params, **extras)

            kwargs[name] = value_dict

        elif origin in (List, list) and len(args) == 1 and hasattr(args[0], 'from_params'):
            value_cls = annotation.__args__[0]

            value_list = []

            for value_params in params.pop(name, Params({})):
                value_list.append(value_cls.from_params(params=value_params, **extras))

            kwargs[name] = value_list

        elif origin in (Tuple, tuple) and all(hasattr(arg, 'from_params') for arg in args):
            value_list = []

            for value_cls, value_params in zip(annotation.__args__, params.pop(name, Params({}))):
                value_list.append(value_cls.from_params(params=value_params, **extras))

            kwargs[name] = tuple(value_list)

        elif origin in (Set, set) and len(args) == 1 and hasattr(args[0], 'from_params'):
            value_cls = annotation.__args__[0]

            value_set = set()

            for value_params in params.pop(name, Params({})):
                value_set.add(value_cls.from_params(params=value_params, **extras))

            kwargs[name] = value_set

        else:
            # Pass it on as is and hope for the best.   ¯\_(ツ)_/¯
            if optional:
                kwargs[name] = params.pop(name, default)
            else:
                kwargs[name] = params.pop(name)

    params.assert_empty(cls.__name__)
    return kwargs


class FromParams:
    """
    Mixin to give a from_params method to classes. We create a distinct base class for this
    because sometimes we want non-Registrable classes to be instantiatable from_params.
    """
    @classmethod
    def from_params(cls: Type[T], params: Params, **extras) -> T:
        """
        This is the automatic implementation of `from_params`. Any class that subclasses
        `FromParams` (or `Registrable`, which itself subclasses `FromParams`) gets this
        implementation for free. If you want your class to be instantiated from params in the
        "obvious" way -- pop off parameters and hand them to your constructor with the same names --
        this provides that functionality.

        If you need more complex logic in your from `from_params` method, you'll have to implement
        your own method that overrides this one.
        """
        # pylint: disable=protected-access
        from nsds.common.registrable import Registrable  # import here to avoid circular imports

        logger.info(f"instantiating class {cls} from params {getattr(params, 'params', params)} "
                    f"and extras {extras}")

        if params is None:
            return None

        if isinstance(params, str):
            params = Params({"type": params})

        registered_subclasses = Registrable._registry.get(cls)

        if registered_subclasses is not None:
            # We know ``cls`` inherits from Registrable, so we'll use a cast to make mypy happy.
            # We have to use a disable to make pylint happy.
            # pylint: disable=no-member
            as_registrable = cast(Type[Registrable], cls)
            default_to_first_choice = as_registrable.default_implementation is not None
            choice = params.pop_choice("type",
                                       choices=as_registrable.list_available(),
                                       default_to_first_choice=default_to_first_choice)
            subclass = registered_subclasses[choice]

            # We want to call subclass.from_params. It's possible that it's just the "free"
            # implementation here, in which case it accepts `**extras` and we are not able
            # to make any assumptions about what extra parameters it needs.
            #
            # It's also possible that it has a custom `from_params` method. In that case it
            # won't accept any **extra parameters and we'll need to filter them out.
            if not takes_arg(subclass.from_params, 'extras'):
                # Necessarily subclass.from_params is a custom implementation, so we need to
                # pass it only the args it's expecting.
                extras = {k: v for k, v in extras.items() if takes_arg(subclass.from_params, k)}

            return subclass.from_params(params=params, **extras)
        else:
            # This is not a base class, so convert our params and extras into a dict of kwargs.

            if cls.__init__ == object.__init__:
                # This class does not have an explicit constructor, so don't give it any kwargs.
                # Without this logic, create_kwargs will look at object.__init__ and see that
                # it takes *args and **kwargs and look for those.
                kwargs: Dict[str, Any] = {}
            else:
                # This class has a constructor, so create kwargs for it.
                kwargs = create_kwargs(cls, params, **extras)

            return cls(**kwargs)  # type: ignore


class Registrable(FromParams):
    """
    Any class that inherits from ``Registrable`` gains access to a named registry for its
    subclasses. To register them, just decorate them with the classmethod
    ``@BaseClass.register(name)``.

    After which you can call ``BaseClass.list_available()`` to get the keys for the
    registered subclasses, and ``BaseClass.by_name(name)`` to get the corresponding subclass.
    Note that the registry stores the subclasses themselves; not class instances.
    In most cases you would then call ``from_params(params)`` on the returned subclass.

    You can specify a default by setting ``BaseClass.default_implementation``.
    If it is set, it will be the first element of ``list_available()``.

    Note that if you use this class to implement a new ``Registrable`` abstract class,
    you must ensure that all subclasses of the abstract class are loaded when the module is
    loaded, because the subclasses register themselves in their respective files. You can
    achieve this by having the abstract class and all subclasses in the __init__.py of the
    module in which they reside (as this causes any import of either the abstract class or
    a subclass to load all other subclasses and the abstract class).
    """
    _registry: Dict[Type, Dict[str, Type]] = defaultdict(dict)
    default_implementation: str = None

    @classmethod
    def register(cls: Type[T], name: str):
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass: Type[T]):
            # Add to registry, raise an error if key has already been used.
            if name in registry:
                message = "Cannot register %s as %s; name already in use for %s" % (
                    name, cls.__name__, registry[name].__name__)
                raise ConfigurationError(message)
            registry[name] = subclass
            return subclass
        return add_subclass_to_registry

    @classmethod
    def by_name(cls: Type[T], name: str) -> Type[T]:
        logger.debug(f"instantiating registered subclass {name} of {cls}")
        if name not in Registrable._registry[cls]:
            raise ConfigurationError("%s is not a registered name for %s" % (name, cls.__name__))
        return Registrable._registry[cls].get(name)

    @classmethod
    def list_available(cls) -> List[str]:
        """List default first if it exists"""
        keys = list(Registrable._registry[cls].keys())
        default = cls.default_implementation

        if default is None:
            return keys
        elif default not in keys:
            message = "Default implementation %s is not registered" % default
            raise ConfigurationError(message)
        else:
            return [default] + [k for k in keys if k != default]
