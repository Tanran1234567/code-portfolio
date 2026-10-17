#!/usr/bin/env python
# -*-coding: utf-8-*-

import asyncio
from typing import Any, Optional, Union

import dramatiq

from mind_zone.api import OpenAIChatClient, RemoteAPI
from mind_zone.base import get_call_kwargs, get_init_kwargs, select_kwargs
from mind_zone.tasks.utils import RESULT_HOLDER_NAME, process_response, wrap_result

__NON_API_PARAM_NAMES__ = ["status_url"]


@dramatiq.actor(store_results=True, max_retries=1)
async def async_chat(
    *,
    client: dict[str, Any],
    messages: dict[str, Any],
    processors: Optional[Union[str, list[str]]] = None,
    result_wrapper: Any = None,
    **kwargs: Any,
):
    api_client = RemoteAPI(**get_init_kwargs(client))
    response = None
    if isinstance(api_client, OpenAIChatClient):
        response = await api_client.achat(
            messages, **get_call_kwargs(client), **select_kwargs(kwargs, exclude_args=__NON_API_PARAM_NAMES__)
        )
    else:
        pass

    if not processors:
        return wrap_result(result_wrapper, RESULT_HOLDER_NAME, response)
    return wrap_result(result_wrapper, RESULT_HOLDER_NAME, process_response(processors, response))


@dramatiq.actor(store_results=True, max_retries=1)
async def async_complete(
    *,
    client: dict[str, Any],
    prompt: str,
    processors: Optional[Union[str, list[str]]] = None,
    result_wrapper: Any = None,
    **kwargs: Any,
):
    api_client = RemoteAPI(**get_init_kwargs(client))
    response = None
    if isinstance(api_client, OpenAIChatClient):
        response = await api_client.acomplete(
            prompt, **get_call_kwargs(client), **select_kwargs(kwargs, exclude_args=__NON_API_PARAM_NAMES__)
        )
    else:
        pass

    if not processors:
        return wrap_result(result_wrapper, RESULT_HOLDER_NAME, response)
    return wrap_result(result_wrapper, RESULT_HOLDER_NAME, process_response(processors, response))


@dramatiq.actor(store_results=True, max_retries=1)
async def batch_chat(
    *,
    client: dict[str, Any],
    msgs_list: list[dict[str, Any]],
    processors: Optional[Union[str, list[str]]] = None,
    result_wrapper: Any = None,
    **kwargs: Any,
):
    api_client = RemoteAPI(**get_init_kwargs(client))
    coro_list = []
    if isinstance(api_client, OpenAIChatClient):
        for messages in msgs_list:
            coro_list.append(
                api_client.achat(
                    messages, **get_call_kwargs(client), **select_kwargs(kwargs, exclude_args=__NON_API_PARAM_NAMES__)
                )
            )
    else:
        pass

    resp_list = await asyncio.gather(*coro_list)
    if not processors:
        return [wrap_result(result_wrapper, RESULT_HOLDER_NAME, x) for x in resp_list]
    return [wrap_result(result_wrapper, RESULT_HOLDER_NAME, process_response(processors, x)) for x in resp_list]


@dramatiq.actor(store_results=True, max_retries=1)
async def batch_complete(
    *,
    client: dict[str, Any],
    prompts: list[str],
    processors: Optional[Union[str, list[str]]] = None,
    result_wrapper: Any = None,
    **kwargs: Any,
):
    api_client = RemoteAPI(**get_init_kwargs(client))
    coro_list = []
    if isinstance(api_client, OpenAIChatClient):
        for prompt in prompts:
            coro_list.append(
                api_client.acomplete(
                    prompt, **get_call_kwargs(client), **select_kwargs(kwargs, exclude_args=__NON_API_PARAM_NAMES__)
                )
            )
    else:
        pass

    resp_list = await asyncio.gather(*coro_list)
    if not processors:
        return [wrap_result(result_wrapper, RESULT_HOLDER_NAME, x) for x in resp_list]
    return [wrap_result(result_wrapper, RESULT_HOLDER_NAME, process_response(processors, x)) for x in resp_list]
