#!/usr/bin/env python
# -*-coding: utf-8-*-

from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class TaskState:
    id: Optional[str] = None
    status: Literal["Pending", "Running", "Canceled", "Failed", "Success"] = "Pending"
    result: Any = None


_Result = namedtuple("_Result", ["id", "success", "result"])
FileResult = _Result
TaskResult = _Result
