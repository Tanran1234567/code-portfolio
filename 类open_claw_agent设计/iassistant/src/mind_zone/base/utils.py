#!/usr/bin/env python
# -*-coding: utf-8-*-

import inspect
from collections import namedtuple
from pathlib import Path
from typing import Any, Optional, Type, Union

import pymupdf

from mind_zone.api import API_PARAM_NAMES, API_PARAM_NAMES_EX
from mind_zone.register import Registry
from mind_zone.utils import Logger

ParamList = namedtuple("ParamList", ["names", "model"])
__MODEL_ARG_NAMES__ = ["llm", "embed_model", "client"]


def validate_client(client: Optional[dict[str, Any]]) -> bool:
    if not client:
        return False
    return "base_url" in client


def get_init_kwargs(raw_kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    筛选LLM客户端连接参数。

    例如：
    ```python
    processor = SummaryExtractor(
        llm,
        summaries,
        prompt_template,
        num_workers,
        **kwargs,
    )
    ```
    此函数用于筛选`llm`的初始化参数
    """
    llm_kwargs = {k: v for k, v in raw_kwargs.items() if k in API_PARAM_NAMES}
    return llm_kwargs


def get_call_kwargs(
    raw_kwargs: Optional[dict[str, Any]], exclude_args: list[str] = [], serialized: bool = True
) -> dict[str, Any]:
    """
    筛选组件初始化的其它参数（LLM客户端连接参数除外）。
    如果有prompt相关参数，则将prompt名称替换为值。
    """
    if not raw_kwargs:
        return {}

    api_kwargs = {k: v for k, v in raw_kwargs.items() if k not in API_PARAM_NAMES_EX and k not in exclude_args}
    for key in list(api_kwargs.keys()):
        if key.endswith("template") and api_kwargs[key] in Registry.Template:
            template = Registry.Template[api_kwargs[key]]
            api_kwargs[key] = template.get_template() if serialized else template
        if key == "response_format" and not isinstance(api_kwargs["response_format"], dict):
            api_kwargs[key] = {"type": api_kwargs["response_format"]}
    return api_kwargs


def select_kwargs(
    raw_kwargs: dict[str, Any],
    include_args: Optional[list[str]] = None,
    exclude_args: list[str] = [],
    serialized: bool = True,
) -> dict[str, Any]:
    include_args = include_args or list(raw_kwargs.keys())
    kwargs = {k: v for k, v in raw_kwargs.items() if k in include_args and k not in exclude_args}
    for key in list(kwargs.keys()):
        if key.endswith("template") and kwargs[key] in Registry.Template:
            template = Registry.Template[kwargs[key]]
            kwargs[key] = template.get_template() if serialized else template
        if key == "response_format" and not isinstance(kwargs["response_format"], dict):
            kwargs[key] = {"type": kwargs["response_format"]}
    return kwargs


def get_param_names(
    class_: Type, method: str = "__init__", include_args: list[str] = [], exclude_args: list[str] = []
) -> ParamList:
    param_names = [
        p.name for p in list(inspect.signature(getattr(class_, method)).parameters.values()) if p.name != "self"
    ]
    model_names = (set(__MODEL_ARG_NAMES__) | set(include_args)) - set(exclude_args)

    model_key = None
    if model_names & set(param_names):
        model_key = next(p for p in param_names if p in model_names)
    return ParamList(names=param_names, model=model_key)


def get_num_pages(pdf_path: Union[str, Path]):
    try:
        with pymupdf.open(str(pdf_path)) as doc:
            return doc.page_count
    except Exception as exc:
        Logger.warning(f"Failed to open .pdf file: {exc}")
        return 0
