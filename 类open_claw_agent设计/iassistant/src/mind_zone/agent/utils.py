#!/usr/bin/env python
# -*-coding: utf-8-*-

from datetime import time
from typing import Any


def update_kwargs(raw_kwargs: dict[str, Any], tick_ts: float) -> dict[str, Any]:
    new_kwargs = raw_kwargs.copy()
    if new_kwargs.get("timeout", 0) > 0:
        new_kwargs["timeout"] = max(0.1, new_kwargs["timeout"] - (time.time() - tick_ts))
    return new_kwargs
