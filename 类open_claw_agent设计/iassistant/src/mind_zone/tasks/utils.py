#!/usr/bin/env python
# -*-coding: utf-8-*-

import json
import re
from typing import Any, Optional, Union

from llama_index.core.llms import ChatResponse, CompletionResponse

from mind_zone.api import RemoteAPI
from mind_zone.base import BaseEmbedding, get_init_kwargs, get_param_names, select_kwargs
from mind_zone.register import Registry
from mind_zone.utils import Logger

RESULT_HOLDER_NAME = "@RESPONSE@"


def get_embed_model(
    *,
    method: Optional[str] = None,
    client: dict[str, Any],
    **kwargs: Any,
) -> BaseEmbedding:
    method = method or client.get("method")
    if method not in Registry.Component:
        Logger.warning(f"Method '{method}' not supported yet, use default instead")
        method = method.rsplit(".", 1)[0] + ".default"

    if method.endswith(".default"):
        processor: BaseEmbedding = RemoteAPI(**get_init_kwargs(client))
    else:
        processor_cls = Registry.Component[method]
        params = get_param_names(processor_cls, include_args=["client"])
        if params.model:
            kwargs[params.model] = RemoteAPI(**get_init_kwargs(client))
        processor: BaseEmbedding = processor_cls(**select_kwargs(client | kwargs, include_args=params.names))
    return processor


def unwrap_response(response: Union[ChatResponse, CompletionResponse]):
    if isinstance(response, ChatResponse):
        return str(response.message.content)
    return str(response)


def convert_to_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_re_list = [
        re.compile(r"```json\s*\n(.*?)\n```", flags=re.DOTALL),  # 标准格式
        re.compile(r"```json\s*(.*?)\s*```", flags=re.DOTALL),  # 紧凑格式
        re.compile(r"```\s*\n(.*?)\n```", flags=re.DOTALL),  # 无语言标识
        re.compile(r"```\s*(.*?)\s*```", flags=re.DOTALL),  # 紧凑无语言标识
    ]
    for pattern in json_re_list:
        match = pattern.search(text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        pass

    return None


ProcessorDict = {
    "unwrap_response": unwrap_response,
    "convert_to_json": convert_to_json,
}


def process_response(processors: Union[str, list[str]], response: Union[ChatResponse, CompletionResponse]):
    if isinstance(processors, str):
        processors = [processors]

    result = response
    for fn_name in processors:
        if fn_name not in ProcessorDict:
            Logger.warning(f"Processor '{fn_name}' not supported yet, skipped")
            continue
        result = ProcessorDict[fn_name](result)
    return result


def wrap_result(wrapper: Any, name: str, value: Any):
    if not wrapper:
        return value

    if isinstance(wrapper, (int, float, bool)):
        raise TypeError("type of wrapper MUST be str, list, tuple or dict")

    def wrap(obj: Any):
        if isinstance(obj, set) or callable(obj):
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        if not obj or isinstance(obj, (int, float, bool)):
            return obj

        if isinstance(obj, str):
            return value if obj == name else obj

        if isinstance(obj, (list, tuple)):
            return [wrap(x) for x in obj]

        if isinstance(obj, dict):
            return {wrap(k): wrap(v) for k, v in obj.items()}
        return

    return wrap(wrapper)
