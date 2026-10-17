#!/usr/bin/env python
# -*-coding: utf-8-*-

from typing import Any, Optional, Union

from llama_index.core.schema import QueryBundle

from mind_zone.base import select_kwargs
from mind_zone.state import TaskManager, TaskResult, TaskState
from mind_zone.tasks import postprocess_nodes, search_documents, preprocess_query
from mind_zone.types import SkipDict


async def query_opt_(
    task_mgr: TaskManager,
    config: SkipDict,
    req_data: dict[str, Any] = {},
    extra_info: dict[str, Any] = {},
    **kwargs: Any,
) -> Union[TaskState, TaskResult]:
    query = {
        "query_str": req_data["text"],
        "history": req_data.get("history", []),
    }
    task_state = task_mgr.launch_task(
        preprocess_query,
        query=query,
        method=config["query.query_opt.rewrite.method"],
        client=select_kwargs(config.get("query.query_opt.rewrite.model")),
    )

    if kwargs.get("wait_kwargs"):
        return await task_mgr.wait_for_done(task_state.id, **kwargs.get("wait_kwargs", {}))
    return task_state


async def search_documents_(
    task_mgr: TaskManager,
    config: SkipDict,
    req_data: dict[str, Any] = {},
    extra_info: dict[str, Any] = {},
    **kwargs: Any,
) -> Union[TaskState, TaskResult]:
    """检索数据库，查询相关文档"""
    filters = kwargs.get("filters")
    if not filters:
        if req_data.get("file_id"):
            filters = [{"key": "file_id", "value": req_data["file_id"]}]
        else:
            filters = [{"key": "bundle_id", "value": req_data["bundle_id"]}]

    query = QueryBundle(query_str=req_data["text"], image_path=req_data.get("image"))
    task_state = task_mgr.launch_task(
        search_documents,
        query=query.to_dict(),
        method=config["query.retrieve.chunks.method"],
        vstore=kwargs.get("ds_config", {}) | config["document.store.text"],
        client=select_kwargs(config.get("query.retrieve.chunks.model") or config.get("document.embed.text", {})),
        filters=filters,
        **select_kwargs(config.get("query.retrieve.chunks.store", {})),
        extra_info={"purpose": "search-documents", **extra_info},
    )
    if kwargs.get("wait_kwargs"):
        return await task_mgr.wait_for_done(task_state.id, **kwargs.get("wait_kwargs", {}))
    return task_state


async def postprocess_nodes_(
    task_mgr: TaskManager,
    config: SkipDict,
    req_data: dict[str, Any] = {},
    parent_id: Optional[str] = None,
    parent_result: Optional[TaskResult] = None,
    extra_info: dict[str, Any] = {},
    **kwargs: Any,
) -> Union[TaskState, TaskResult]:
    """后处理检索到的文档"""
    if parent_result:
        chunk_list = parent_result.result
    elif parent_id:
        parent_result = await task_mgr.wait_for_done(parent_id, return_primitive=True)
        chunk_list = parent_result.result
    else:
        chunk_list = None
    assert chunk_list, "Please provide either `parent_id` or `parent_result`"

    if (parent_id or parent_result) and not extra_info.get("parent_id"):
        extra_info["parent_id"] = parent_id or parent_result.id

    task_state = task_mgr.launch_task(
        postprocess_nodes,
        query=req_data["text"],
        chunks=chunk_list,
        method=config["postprocess.nodes.method"],
        client=select_kwargs(config["postprocess.nodes"]),
        extra_info={"purpose": "postprocess-nodes", **extra_info},
    )
    if kwargs.get("wait_kwargs"):
        return await task_mgr.wait_for_done(task_state.id, **kwargs.get("wait_kwargs", {}))
    return task_state
