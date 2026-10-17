#!/usr/bin/env python
# -*-coding: utf-8-*-

from typing import Any, Optional

import dramatiq
from llama_index.core.indices.query.query_transform.base import BaseQueryTransform

from mind_zone.api import RemoteAPI
from mind_zone.base import (
    get_init_kwargs,
    get_param_names,
    select_kwargs,
)
from mind_zone.register import Registry
from mind_zone.utils import Logger


@dramatiq.actor(store_results=True, max_retries=1)
async def preprocess_query(
    *,
    query: dict[str, Any],
    method: str,
    client: Optional[dict[str, Any]] = None,
    **kwargs: Any,
):
    if method not in Registry.Component:
        Logger.warning(f"Method '{method}' not supported yet, use default instead")
        method = method.rsplit(".", 1)[0] + ".default"

    processor_cls = Registry.Component[method]
    params = get_param_names(processor_cls)
    if params.model:
        kwargs[params.model] = RemoteAPI(**get_init_kwargs(client))

    query_str = query["query_str"]
    processor: BaseQueryTransform = processor_cls(**select_kwargs(kwargs, include_args=params.names))
    result = processor(query_str, {"history": query.get("history", [])})
    return result.to_dict()


@dramatiq.actor(store_results=True, max_retries=1)
async def analyze_query(
    *,
    query: dict[str, Any],
    method: str,
    client: Optional[dict[str, Any]] = None,
    kwargs: Any,
):
    pass


@dramatiq.actor(store_results=True, max_retries=1)
async def optimize_query(
    *,
    query: dict[str, Any],
    method: str,
    client: Optional[dict[str, Any]] = None,
    kwargs: Any,
):
    pass
