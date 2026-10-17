#!/usr/bin/env python
# -*-coding: utf-8-*-

import threading
from collections import OrderedDict
from typing import Any, Callable, Optional, Type, TypeVar, Union

from llama_index.core.prompts import PromptTemplate

from mind_zone.types import SkipDict
from mind_zone.utils import Logger

T = TypeVar("T")


class Register(object):
    """Registry manager for new modules, tasks, and so on"""

    def __init__(self, name, default: Optional[Union[Callable, Type[T]]] = None):
        self._reg_dict = OrderedDict()
        self._reg_name = name
        self._default = default

    def __setitem__(self, key: Optional[str] = None, value: Callable = None):
        if not callable(value):
            raise Exception("Value of a Registry must be a callable.")

        if key is None:
            key = value.__name__
        if key in self._reg_dict:
            Logger.warning("Key {} already in registry {}.".format(key, self._reg_name))
        self._reg_dict[key] = value

    def __getitem__(self, key: str):
        try:
            if self._default is not None:
                return self._reg_dict.get(key, self._default)
            return self._reg_dict[key]
        except Exception as exc:
            Logger.error(f"Function or class {key} not found: {exc}")
            raise exc

    def __contains__(self, key):
        return key in self._reg_dict

    def register(self, name_or_module: Union[str, Callable]):
        """Decorator to register a function or class"""

        def decorator(key: Optional[str], value: Callable):
            if value in self._reg_dict.values():
                Logger.warning(
                    "Value {} already registered by another key {}.".format(
                        value.__name__, next(k for k, v in self._reg_dict.items() if v == value)
                    )
                )
            self[key] = value
            return value

        if callable(name_or_module):
            # @reg.register, 'name_or_module' is a callable module (function or class)
            return decorator(None, name_or_module)
        # @reg.register('alias'), 'name_or_module' is a string
        return lambda x: decorator(name_or_module, x)

    def keys(self):
        return self._reg_dict.keys()


class StaticMetaClass(type):
    r"""Metaclass for all static modules"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any]):
        for key in ("__init__", "__delattr__", "__setattr__"):
            if key in namespace:
                raise RuntimeError("attribute '{}' is not allowed in static class '{}'".format(key, name))

        # Double-Checked Locking
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls, name, bases, namespace)
        return cls._instance

    def __repr__(cls):
        if hasattr(cls, "__repr__"):
            return getattr(cls, "__repr__")()
        else:
            return super().__repr__()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args: Any, **kwds: Any) -> Any:  # pyright: ignore
        raise RuntimeError("static class '{}' is not intended to be instantiated".format(self.__name__))

    def __setattr__(self, name: str, value: Any, /) -> None:  # pyright: ignore
        raise RuntimeError("assigning to attribute '{}' is forbidden in static class '{}'".format(name, self.__name__))

    def __delattr__(self, name: str, /) -> None:
        raise RuntimeError("deleting attribute '{}' is forbidden in static class '{}'".format(name, self.__name__))


class Registry(metaclass=StaticMetaClass):
    """All registered modules"""

    Component: Register = Register("Component")
    AgentEngine: Register = Register("AgentEngine")
    Template: dict[str, PromptTemplate] = OrderedDict()
    Application: dict[str, Any] = SkipDict()

    @classmethod
    def __repr__(cls):
        return "RegistryManager:\n    " + "\n    ".join(
            ["{}: {}".format(k, type(v).__name__) for k, v in vars(cls).items() if not k.startswith("_")]
        )
