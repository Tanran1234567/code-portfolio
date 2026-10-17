#!/usr/bin/env python
# -*-coding: utf-8-*-

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from mind_zone.config import Config
from mind_zone.register import Registry
from mind_zone.routers.common import (
    create_document_,
    extract_metadata_,
    get_embeddings_,
    parse_document_,
    split_document_,
)
from mind_zone.schemas import (
    DocumentDeleteRequest,
    DocumentIndexRequestV1,
    DocumentParseRequest,
    DocumentReindexRequest,
    FormatOutput,
)
from mind_zone.state import TaskManager, TaskResult
from mind_zone.tasks import delete_documents, store_documents
from mind_zone.types import SkipDict
from mind_zone.utils import Logger, truncate_repr

router = APIRouter(prefix="/file")


@router.post("/parse")
async def _parse_file(request: DocumentParseRequest, raw_request: Request):  # pyright: ignore
    req_dict = request.model_dump()
    task_mgr: TaskManager = raw_request.app.TASK_MANAGER
    app_config = Registry.Application[request.app_name]

    task_state = await parse_document_(task_mgr, app_config["document"], req_dict)
    if task_state.status != "Running":
        task_id = task_mgr.search_task(file_id=request.file_id)
    else:
        task_id = task_state.id
    if not task_id:
        return JSONResponse(content=FormatOutput("parse-file"))

    task_result = await task_mgr.wait_for_done(task_id, timeout=150)
    if not task_result.success:
        return JSONResponse(content=FormatOutput("parse-file"))
    Logger.debug("Parse file successfully:\n{}".format(truncate_repr(task_result.result[0].text, 64)))

    return_type = request.return_type
    if return_type == "chunks":
        return await _split_document(task_mgr, app_config["document"], req_dict, task_result)

    elif isinstance(return_type, list) and len(return_type) > 0:
        return await _extract_metadata(task_mgr, app_config["document"], return_type, task_result)

    doc_content = "\n\n".join([d.text for d in task_result.result])
    return JSONResponse(content=FormatOutput("no-error", data={"content": doc_content}))


async def _split_document(task_mgr: TaskManager, config: SkipDict, req_data: dict[str, Any], parent_result: TaskResult):
    task_result = await split_document_(
        task_mgr, config, req_data, parent_result=parent_result, wait_kwargs={"timeout": 30}
    )
    if task_result.success:
        chunk_list = [d.text for d in task_result.result]
        return JSONResponse(content=FormatOutput("no-error", data={"chunk_list": chunk_list}))

    doc_content = "\n\n".join([d.text for d in parent_result.result])
    return JSONResponse(content=FormatOutput("split-doc", data={"content": doc_content}))


async def _extract_metadata(task_mgr: TaskManager, config: SkipDict, field_list: list[str], parent_result: TaskResult):
    metadata_list = await extract_metadata_(
        task_mgr, config, field_list=field_list, parent_result=parent_result, wait_kwargs={"timeout": 90}
    )
    if metadata_list:
        if len(parent_result.result) == 1:
            return JSONResponse(content=FormatOutput("no-error", data={"metadata": metadata_list[0]}))
        return JSONResponse(content=FormatOutput("no-error", data={"metadata": metadata_list}))

    doc_content = "\n\n".join([d.text for d in parent_result.result])
    return JSONResponse(content=FormatOutput("metadata", data={"content": doc_content}))


@router.post("/store")
async def _store_file(request: DocumentIndexRequestV1, raw_request: Request):  # pyright: ignore
    """上传文件或内容，文档分段，提取元信息，计算Embedding，存储到数据库"""
    req_dict = request.model_dump()
    task_mgr: TaskManager = raw_request.app.TASK_MANAGER
    config: Config = raw_request.app.CONFIG
    app_config = Registry.Application[req_dict["app_name"]]

    parse_result = await parse_document_(
        task_mgr, app_config["document"], req_dict, status_url=None, wait_kwargs={"timeout": 150}
    )
    if not parse_result.success:
        return JSONResponse(content=FormatOutput("parse-file"))
    Logger.debug("Parse file successfully:\n{}".format(truncate_repr(parse_result.result[0].text, 64)))

    split_result = await split_document_(
        task_mgr, app_config["document"], req_dict, parent_result=parse_result, wait_kwargs={"timeout": 30}
    )
    if not split_result.success:
        return JSONResponse(content=FormatOutput("split-doc"))

    batch_size = 10
    for idx in range((len(split_result.result) + batch_size - 1) // batch_size):
        split_texts = [d.text for d in split_result.result[batch_size * idx : batch_size * (idx + 1)]]
        embed_state = await get_embeddings_(task_mgr, app_config["document"], req_dict, contents=split_texts)
        metadata_list = await extract_metadata_(
            task_mgr, app_config["document"], req_dict, contents=split_texts, wait_kwargs={"timeout": 150}
        )
        if not metadata_list:
            return JSONResponse(content=FormatOutput("metadata"))

        embed_result = await task_mgr.wait_for_done(embed_state.id, timeout=60)
        if not embed_result.success:
            return JSONResponse(content=FormatOutput("embedding"))

        metadata_dict = {"file": parse_result.result[0].metadata, "chunks": metadata_list}
        chunk_list = await create_document_(
            split_result.result[batch_size * idx : batch_size * (idx + 1)], metadata_dict, embed_result.result
        )
        ds_type = app_config["document.store.text.ds_type"]
        task_kwargs = (
            {"file_id": req_dict["file_id"], "status_url": req_dict.get("status_url")}
            if batch_size * (idx + 1) >= len(split_result.result)
            else {}
        )
        task_state = task_mgr.launch_task(
            store_documents,
            documents=[d.to_dict() for d in chunk_list],
            ds_type=ds_type,
            ds_config=config.get_config(db_type=ds_type) | app_config["document.store.text"],
            **task_kwargs,
        )
        if task_state.status not in ["Running", "Success"]:
            return JSONResponse(content=FormatOutput("store-doc"))

    if req_dict.get("status_url"):
        return JSONResponse(content=FormatOutput("no-error", data={"task_id": task_state.id}))

    task_result = await task_mgr.wait_for_done(task_state.id, timeout=60)
    if task_result.success:
        return JSONResponse(content=FormatOutput("no-error", data={"doc_ids": task_result.result}))
    return JSONResponse(content=FormatOutput("store-doc"))


@router.post("/learn")
async def _learn_file(request: DocumentReindexRequest, raw_request: Request):  # pyright: ignore
    """重新学习，文档分段，提取元信息，计算Embedding，存储到数据库"""
    req_dict = request.model_dump()
    task_mgr: TaskManager = raw_request.app.TASK_MANAGER
    config: Config = raw_request.app.CONFIG
    app_config = Registry.Application[request.app_name]

    file_result = await task_mgr.get_document(
        req_dict["file_id"], timeout=10, return_fields=["file_id", "file_path", "metadata"]
    )
    if file_result.success:
        task_id = task_mgr.search_task(file_id=req_dict["file_id"])
        parse_result = await task_mgr.wait_for_done(task_id)
    else:
        if not file_result.result:
            return JSONResponse(content=FormatOutput("file-not-found"))

        task_mgr.delete_files(file_id=file_result.result["file_id"])
        file_info = {
            "file_id": file_result.result["file_id"],
            "file_path": file_result.result["file_path"],
            "file_type": file_result.result["metadata"]["file_type"],
            "bundle_id": file_result.result.get("bundle_id"),
        }
        parse_result = await parse_document_(task_mgr, app_config["document"], file_info, wait_kwargs={"timeout": 150})
        if not parse_result.success:
            return JSONResponse(content=FormatOutput("parse-file"))
        Logger.debug("Parse file successfully:\n{}".format(truncate_repr(parse_result.result[0].text, 64)))
        task_id = parse_result.id

    split_result = await split_document_(
        task_mgr, app_config["document"], req_dict, parent_id=task_id, wait_kwargs={"timeout": 30}
    )
    if not split_result.success:
        return JSONResponse(content=FormatOutput("split-doc"))

    batch_size = 10
    for idx in range((len(split_result.result) + batch_size - 1) // batch_size):
        split_texts = [d.text for d in split_result.result[batch_size * idx : batch_size * (idx + 1)]]
        embed_state = await get_embeddings_(task_mgr, app_config["document"], req_dict, contents=split_texts)
        metadata_list = await extract_metadata_(
            task_mgr, app_config["document"], req_dict, contents=split_texts, wait_kwargs={"timeout": 150}
        )
        if not metadata_list:
            return JSONResponse(content=FormatOutput("metadata"))

        embed_result = await task_mgr.wait_for_done(embed_state.id, timeout=30)
        if not embed_result.success:
            return JSONResponse(content=FormatOutput("embedding"))

        metadata_dict = {"file": parse_result.result[0].metadata, "chunks": metadata_list}
        chunk_list = await create_document_(
            split_result.result[batch_size * idx : batch_size * (idx + 1)], metadata_dict, embed_result.result
        )
        ds_type = app_config["document.store.text.ds_type"]
        task_kwargs = (
            {"file_id": req_dict["file_id"], "status_url": req_dict.get("status_url")}
            if batch_size * (idx + 1) >= len(split_result.result)
            else {}
        )
        task_state = task_mgr.launch_task(
            store_documents,
            documents=[d.to_dict() for d in chunk_list],
            ds_type=ds_type,
            ds_config=config.get_config(db_type=ds_type) | app_config["document.store.text"],
            **task_kwargs,
        )
        if task_state.status not in ["Running", "Success"]:
            return JSONResponse(content=FormatOutput("store-doc"))

    if req_dict.get("status_url"):
        return JSONResponse(content=FormatOutput("no-error", data={"task_id": task_state.id}))

    task_result = await task_mgr.wait_for_done(task_state.id, timeout=60)
    if task_result.success:
        return JSONResponse(content=FormatOutput("no-error", data={"doc_ids": task_result.result}))
    return JSONResponse(content=FormatOutput("store-doc"))


@router.post("/delete")
async def _delete_file(request: DocumentDeleteRequest, raw_request: Request):  # pyright: ignore
    task_mgr: TaskManager = raw_request.app.TASK_MANAGER
    config: Config = raw_request.app.CONFIG
    app_config = Registry.Application[request.app_name]

    filters = None
    if request.file_id:
        filters = [{"key": "file_id", "value": request.file_id}]
    elif request.bundle_id:
        filters = [{"key": "bundle_id", "value": request.bundle_id}]

    if filters:
        ds_type = app_config["document.store.text.ds_type"]
        task_result = await task_mgr.complete_task(
            delete_documents,
            ds_type=ds_type,
            ds_config=config.get_config(db_type=ds_type) | app_config["document.store.text"],
            filters=filters,
            wait_kwargs={"timeout": 10},
        )
        if not task_result.success:
            return JSONResponse(content=FormatOutput("vstore-del"))

    count = task_mgr.delete_files(file_id=request.file_id, bundle_id=request.bundle_id)
    return JSONResponse(content=FormatOutput("no-error", data={"count": count}))
