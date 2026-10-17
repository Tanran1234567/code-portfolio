#!/usr/bin/env python
# -*-coding: utf-8-*-

import asyncio
from typing import Any, Optional, Union

from llama_index.core.schema import Document, MediaResource

from mind_zone.base import get_call_kwargs, get_init_kwargs, select_kwargs
from mind_zone.state import TaskManager, TaskResult, TaskState
from mind_zone.tasks import (
    extract_keyword,
    extract_question,
    extract_summary,
    extract_title,
    get_embeddings,
    load_from_file,
    split_document,
)
from mind_zone.types import SkipDict
from mind_zone.utils import Logger

EXTRACT_TASK_DICT = {
    "title": extract_title,
    "keyword": extract_keyword,
    "summary": extract_summary,
    "question": extract_question,
}


async def parse_document_(
    task_mgr: TaskManager,
    config: SkipDict,
    req_data: dict[str, Any],
    extra_info: dict[str, Any] = {},
    **kwargs: Any,
) -> Union[TaskState, TaskResult]:
    file_id = kwargs.get("file_id", req_data.get("file_id"))
    file_type = kwargs.get("file_type", req_data["file_type"])
    task_state = task_mgr.launch_task(
        load_from_file,
        file_type=file_type,
        file_path=kwargs.get("file_path", req_data["file_path"]),
        method=config[f"read_file{file_type}.method"],
        client=select_kwargs(config[f"read_file{file_type}"]),
        file_id=file_id,
        bundle_id=kwargs.get("bundle_id", req_data.get("bundle_id")),
        status_url=kwargs.get("status_url", req_data.get("status_url")),
        extra_info={"purpose": "parse-document", **extra_info},
    )
    if kwargs.get("wait_kwargs"):
        if task_state.status == "Canceled":
            Logger.warning(f"File {file_id} has already been uploaded.")
            task_id = task_mgr.search_task(file_id=file_id)
        else:
            task_id = task_state.id
        return await task_mgr.wait_for_done(task_id, **kwargs.get("wait_kwargs", {}))
    return task_state


async def split_document_(
    task_mgr: TaskManager,
    config: SkipDict,
    req_data: dict[str, Any] = {},
    parent_id: Optional[str] = None,
    parent_result: Optional[TaskResult] = None,
    extra_info: dict[str, Any] = {},
    **kwargs: Any,
) -> Union[TaskState, TaskResult]:
    """对单个文件分段，大文件在解析时可能被拆分成多个部分"""
    # `contents`表示文件的多个部分
    if parent_result:
        contents = [d.text for d in parent_result.result]
    elif parent_id:
        parent_result = await task_mgr.wait_for_done(parent_id)
        contents = [d.text for d in parent_result.result]
    else:
        assert req_data.get("contents"), "Please provide at least one of `contents`, `parent_id`, and `parent_result`"
        contents = req_data["contents"]

    if (parent_id or parent_result) and not extra_info.get("parent_id"):
        extra_info["parent_id"] = parent_id or parent_result.id

    task_state = task_mgr.launch_task(
        split_document,
        texts=contents,
        method=config["split.text.method"],
        client=select_kwargs(config["split.text"]),
        file_id=kwargs.get("file_id", req_data.get("file_id")),
        extra_info={"purpose": "split-document", **extra_info},
    )
    if kwargs.get("wait_kwargs"):
        return await task_mgr.wait_for_done(task_state.id, **kwargs.get("wait_kwargs", {}))
    return task_state


async def get_embeddings_(
    task_mgr: TaskManager,
    config: SkipDict,
    req_data: dict[str, Any] = {},
    parent_id: Optional[str] = None,
    parent_result: Optional[TaskResult] = None,
    extra_info: dict[str, Any] = {},
    **kwargs: Any,
) -> Union[TaskState, TaskResult]:
    """获取多个分段的Embedding"""
    # `contents`表示文件的分段，split之后的内容
    if parent_result:
        contents = [d.text for d in parent_result.result]
    elif parent_id:
        parent_result = await task_mgr.wait_for_done(parent_id)
        contents = [d.text for d in parent_result.result]
    else:
        null_count = sum(
            [not req_data.get("contents"), not req_data.get("text"), not kwargs.get("contents"), not kwargs.get("text")]
        )
        assert null_count < 4, "Please provide at least one of `text`, `contents`, `parent_id`, and `parent_result`"
        assert null_count > 2, "Please provide either `text` or `contents` only for embedding"
        contents = req_data.get("contents") or req_data.get("text") or kwargs.get("contents") or kwargs.get("text")

    if (parent_id or parent_result) and not extra_info.get("parent_id"):
        extra_info["parent_id"] = parent_id or parent_result.id
    # TODO: 添加prompt
    task_state = task_mgr.launch_task(
        get_embeddings,
        texts=contents,
        method=config["embed.text.method"],
        client=select_kwargs(config["embed.text"]),
        extra_info={"purpose": "get-embeddings", **extra_info},
    )
    if kwargs.get("wait_kwargs"):
        return await task_mgr.wait_for_done(task_state.id, **kwargs.get("wait_kwargs", {}))
    return task_state


async def extract_metadata_(
    task_mgr: TaskManager,
    config: SkipDict,
    req_data: dict[str, Any] = {},
    parent_id: Optional[str] = None,
    parent_result: Optional[TaskResult] = None,
    extra_info: dict[str, Any] = {},
    **kwargs: Any,
) -> Union[dict[str, TaskState], list[dict[str, Any]]]:
    # `contents`表示文件的分段，split之后的内容
    if parent_result:
        contents = [d.text for d in parent_result.result]
    elif parent_id:
        parent_result = await task_mgr.wait_for_done(parent_id)
        contents = [d.text for d in parent_result.result]
    else:
        assert req_data.get("contents") or kwargs.get("contents"), (
            "Please" "provide at least one of `contents`, `parent_id`, and `parent_result`"
        )
        contents = req_data.get("contents") or kwargs.get("contents")

    field_list = kwargs.get("field_list", req_data.get("return_fields"))
    if (parent_id or parent_result) and not extra_info.get("parent_id"):
        extra_info["parent_id"] = parent_id or parent_result.id

    state_dict = {}
    for field in EXTRACT_TASK_DICT.keys():
        if field not in config["extract"]:
            continue
        if field_list and field not in field_list:
            continue

        task_state = task_mgr.launch_task(
            EXTRACT_TASK_DICT[field],
            texts=contents,
            method=config[f"extract.{field}.method"],
            client=get_init_kwargs(config[f"extract.{field}"]),
            **get_call_kwargs(config[f"extract.{field}"]),
            extra_info={"purpose": f"extract-{field}", **extra_info},
        )
        state_dict[field] = task_state

    if not kwargs.get("wait_kwargs"):
        return state_dict

    wait_kwargs = kwargs.get("wait_kwargs", {})
    result_list = await asyncio.gather(*[task_mgr.wait_for_done(x.id, **wait_kwargs) for x in state_dict.values()])
    metadata_list = [{} for _ in range(len(contents))]
    for result in result_list:
        if result.success and result.result:
            for idx, val in enumerate(result.result):
                metadata_list[idx].update(val)
    return metadata_list


async def create_document_(
    content_list: list[Union[str, dict[str, Any], Document]],
    metadata_dict: dict[str, Any],
    embedding_list: list[list[float]],
) -> list[Document]:
    """构造文档，补全元信息、Embedding、内容等"""
    assert "chunks" not in metadata_dict or len(metadata_dict["chunks"]) == len(content_list)
    assert len(embedding_list) == len(content_list)

    chunk_list = []
    for idx, content in enumerate(content_list):
        if isinstance(content, str):
            chunk = Document(text_resource=MediaResource(text=content))
        elif isinstance(content, dict):
            chunk = Document.from_dict(content)
        elif isinstance(content, Document):
            chunk = content.model_copy(deep=True)
        else:
            Logger.warning("Unsupported object type {} to create document".format(type(content).__name__))
            continue

        metadata = chunk.metadata
        if metadata_dict.get("file"):
            metadata = metadata | metadata_dict["file"]
        if metadata_dict.get("chunks"):
            metadata = metadata | metadata_dict["chunks"][idx]

        chunk.metadata = metadata
        chunk.embedding = embedding_list[idx]
        chunk_list.append(chunk)
    return chunk_list
