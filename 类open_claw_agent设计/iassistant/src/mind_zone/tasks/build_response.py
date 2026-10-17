#!/usr/bin/env python
# -*-coding: utf-8-*-

from copy import deepcopy
from typing import Any, Optional, Union

import dramatiq
from llama_index.core.base.response.schema import AsyncStreamingResponse
from llama_index.core.schema import NodeWithScore

from mind_zone.api import RemoteAPI
from mind_zone.base import BaseSynthesizer, get_call_kwargs, get_init_kwargs, get_param_names, select_kwargs
from mind_zone.register import Registry
from mind_zone.utils import Logger, cast_from_dict


@dramatiq.actor(store_results=True, max_retries=1)
async def synthesize_response(
    *,
    query: str,
    method: str,
    chunks: list[dict[str, Any]],
    client: Optional[dict[str, Any]] = None,
    **kwargs: Any,
):
    if method not in Registry.Component:
        Logger.warning(f"Method '{method}' not supported yet, use default instead")
        method = method.rsplit(".", 1)[0] + ".default"

    processor_cls = Registry.Component[method]
    params = get_param_names(processor_cls)
    init_kwargs = deepcopy(kwargs)
    if params.model:
        init_kwargs[params.model] = RemoteAPI(**get_init_kwargs(client))

    client = client or {}
    processor: BaseSynthesizer = processor_cls(**select_kwargs(client | init_kwargs, include_args=params.names))
    sorted_chunks = [cast_from_dict(x, "NodeWithScore") for x in chunks]
    response = await processor.asynthesize(query, sorted_chunks, **get_call_kwargs(client | kwargs))
    return response


async def stream_synthesize_response(
    *,
    query: str,
    method: str,
    chunks: list[Union[NodeWithScore, dict[str, Any]]],
    client: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> AsyncStreamingResponse:
    if method not in Registry.Component:
        Logger.warning(f"Method '{method}' not supported yet, use default instead")
        method = method.rsplit(".", 1)[0] + ".default"

    processor_cls = Registry.Component[method]
    params = get_param_names(processor_cls)
    init_kwargs = deepcopy(kwargs)
    if params.model:
        init_kwargs[params.model] = RemoteAPI(**get_init_kwargs(client))
    init_kwargs["streaming"] = True

    conv_fn = lambda x: x if isinstance(x, NodeWithScore) else cast_from_dict(x, "NodeWithScore")

    client = client or {}
    processor: BaseSynthesizer = processor_cls(**select_kwargs(client | init_kwargs, include_args=params.names))
    response = await processor.asynthesize(query, [conv_fn(x) for x in chunks], **get_call_kwargs(client | kwargs))
    return response
