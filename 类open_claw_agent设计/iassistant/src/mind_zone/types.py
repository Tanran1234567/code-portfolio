#!/usr/bin/env python
# -*-coding: utf-8-*-

from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from typing import Any, Self, overload


class BaseEnum(str, Enum):
    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class SkipDict(OrderedDict[str, Any]):

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        if len(args) > 1 or int(not args) + int(not kwargs) < 1:
            raise TypeError("Redundant positional or keyword arguments")
        return super().__new__(cls)

    @overload
    def __init__(self): ...

    @overload
    def __init__(self, map: dict[str, Any]): ...

    @overload
    def __init__(self, **kwargs: Any): ...

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        map = args[0] if args else kwargs
        if not isinstance(map, dict):
            raise ValueError("Unsupported argument type")

        for key, val in map.items():
            super().__setitem__(key, self.__copy__(val))

    @staticmethod
    def __copy__(value: Any) -> Any:
        copy_data = None
        if isinstance(value, (list, set)) and any(isinstance(x, dict) for x in value):
            copy_data = type(value)()
            func_name = "append" if isinstance(value, list) else "add"
            for item in value:
                if isinstance(item, dict):
                    getattr(copy_data, func_name)(SkipDict(item))
                else:
                    getattr(copy_data, func_name)(deepcopy(item))
        elif isinstance(value, dict):
            copy_data = SkipDict(value)
        else:
            copy_data = deepcopy(value)
        return copy_data

    def __check__(self, key: str, /) -> None:
        if key.startswith(".") or key.endswith("."):
            raise KeyError(f"'.' is reserved for SkipDict, please remove prepending or appending dot in {key}")
        if not key:
            raise KeyError("Empty key is not allowed")

    def __contains__(self, key: str, /) -> bool:
        self.__check__(key)
        if "." not in key:
            return super().__contains__(key)

        prefix, sub_key = key.split(".", 1)
        if not super().__contains__(prefix):
            return False

        value = super().__getitem__(prefix)
        if not isinstance(value, (dict, SkipDict)):
            return False
        return sub_key in value

    def __getitem__(self, key: str, /) -> Any:
        self.__check__(key)
        value = self
        try:
            for k in key.split("."):
                if isinstance(value, SkipDict):
                    value = super(SkipDict, value).__getitem__(k)
                else:
                    value = value[k]
            return value
        except:
            raise KeyError(f"Key not found: {key}")

    def __setitem__(self, key: str, value: Any, /) -> None:
        self.__check__(key)
        obj = self
        for k in key.split(".")[:-1]:
            if k not in obj or not isinstance(obj[k], (dict, SkipDict)):
                obj[k] = SkipDict()
            obj = obj[k]

        if isinstance(obj, SkipDict):
            super(SkipDict, obj).__setitem__(key.split(".")[-1], self.__copy__(value))
        else:
            obj[key.split(".")[-1]] = self.__copy__(value)

    def get(self, key: str, default: Any = None, /) -> Any:
        if key in self:
            return self[key]
        return default


if __name__ == "__main__":
    d = {"a": {"b": 2}}
    d = SkipDict(d)
    print(d)
    d["a.b.c"] = 12
    print(d)
    d["a.b.c"] = []
    print(d)
    d["a.b.c.d"] = "abc"
    print(d)
    print(d.get("a.b.c", "unknown"))
