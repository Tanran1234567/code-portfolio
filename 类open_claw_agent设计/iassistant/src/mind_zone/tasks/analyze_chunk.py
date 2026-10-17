#!/usr/bin/env python
# -*-coding: utf-8-*-

from typing import Any, Optional

import dramatiq
from llama_index.core.schema import TextNode

from mind_zone.api import RemoteAPI
from mind_zone.base import BaseExtractor, get_init_kwargs, get_param_names, select_kwargs
from mind_zone.register import Registry
from mind_zone.utils import Logger


@dramatiq.actor(store_results=True, max_retries=1)
async def extract_title(
    *,
    texts: list[str],
    method: str = "default",
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

    client = client or {}
    processor: BaseExtractor = processor_cls(**select_kwargs(client | kwargs, include_args=params.names))
    result_list = await processor.aextract([TextNode(text=x) for x in texts])
    result_list = [{"title": d["document_title"]} for d in result_list]
    return result_list


@dramatiq.actor(store_results=True, max_retries=1)
async def extract_keyword(
    *,
    texts: list[str],
    method: str = "default",
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

    client = client or {}
    processor: BaseExtractor = processor_cls(**select_kwargs(client | kwargs, include_args=params.names))
    result_list = await processor.aextract([TextNode(text=x) for x in texts])
    result_list = [{"keyword": d["excerpt_keywords"]} for d in result_list]
    return result_list


@dramatiq.actor(store_results=True, max_retries=1)
async def extract_summary(
    *,
    texts: list[str],
    method: str = "default",
    client: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> list[str]:
    if method not in Registry.Component:
        Logger.warning(f"Method '{method}' not supported yet, use default instead")
        method = method.rsplit(".", 1)[0] + ".default"

    processor_cls = Registry.Component[method]
    params = get_param_names(processor_cls)
    if params.model:
        kwargs[params.model] = RemoteAPI(**get_init_kwargs(client))

    client = client or {}
    processor: BaseExtractor = processor_cls(**select_kwargs(client | kwargs, include_args=params.names))
    result_list = await processor.aextract([TextNode(text=x) for x in texts])
    result_list = [{"summary": d["section_summary"]} for d in result_list]
    return result_list


@dramatiq.actor(store_results=True, max_retries=1)
async def extract_question(
    *,
    texts: list[str],
    method: str = "default",
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

    client = client or {}
    processor: BaseExtractor = processor_cls(**select_kwargs(client | kwargs, include_args=params.names))
    result_list = await processor.aextract([TextNode(text=x) for x in texts])
    result_list = [{"question": d["questions_this_excerpt_can_answer"]} for d in result_list]
    return result_list
