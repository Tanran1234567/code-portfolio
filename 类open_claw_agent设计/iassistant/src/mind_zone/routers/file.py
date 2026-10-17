#!/usr/bin/env python
# -*-coding: utf-8-*-

from typing import Optional, Union

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import Field

from mind_zone.agent import Orchestrator
from mind_zone.config import Config
from mind_zone.register import Registry
from mind_zone.schemas import DocumentDeleteRequest, DocumentIndexRequest, FormatOutput
from mind_zone.state import TaskManager
from mind_zone.tasks import delete_documents

router = APIRouter(prefix="/file")


class ParseRequest(DocumentIndexRequest):
    bundle_id: Optional[str] = Field(default=None, description="文件包ID，可以是知识库ID，或其它文件集合ID")
    status_url: Optional[str] = Field(default=None, description="状态回调接口，处理完成后发送回调请求")
    meta_fields: Optional[list[str]] = Field(default=[], exclude=True)


class SplitRequest(DocumentIndexRequest):
    file_type: Optional[str] = Field(default=None, description="文件类型，小写后缀名，如.pdf、.md、.txt等")
    file_path: Optional[Union[str, list[str]]] = Field(
        default=None, description="文件路径，文件系统路径或下载路径，文件有多个片段时，可以传每个片段的路径列表"
    )
    bundle_id: Optional[str] = Field(default=None, description="文件包ID，可以是知识库ID，或其它文件集合ID")
    status_url: Optional[str] = Field(default=None, description="状态回调接口，处理完成后发送回调请求")
    meta_fields: Optional[list[str]] = Field(default=[], exclude=True)


class ExtractRequest(DocumentIndexRequest):
    file_type: Optional[str] = Field(default=None, description="文件类型，小写后缀名，如.pdf、.md、.txt等")
    file_path: Optional[Union[str, list[str]]] = Field(
        default=None, description="文件路径，文件系统路径或下载路径，文件有多个片段时，可以传每个片段的路径列表"
    )
    bundle_id: Optional[str] = Field(default=None, description="文件包ID，可以是知识库ID，或其它文件集合ID")
    status_url: Optional[str] = Field(default=None, description="状态回调接口，处理完成后发送回调请求")


class EmbedRequest(DocumentIndexRequest):
    file_type: Optional[str] = Field(default=None, description="文件类型，小写后缀名，如.pdf、.md、.txt等")
    file_path: Optional[Union[str, list[str]]] = Field(
        default=None, description="文件路径，文件系统路径或下载路径，文件有多个片段时，可以传每个片段的路径列表"
    )
    bundle_id: Optional[str] = Field(default=None, description="文件包ID，可以是知识库ID，或其它文件集合ID")
    status_url: Optional[str] = Field(default=None, description="状态回调接口，处理完成后发送回调请求")
    meta_fields: Optional[list[str]] = Field(default=[], exclude=True)


class StoreRequest(DocumentIndexRequest):
    file_type: Optional[str] = Field(default=None, description="文件类型，小写后缀名，如.pdf、.md、.txt等")
    file_path: Optional[Union[str, list[str]]] = Field(
        default=None, description="文件路径，文件系统路径或下载路径，文件有多个片段时，可以传每个片段的路径列表"
    )
    bundle_id: Optional[str] = Field(default=None, description="文件包ID，可以是知识库ID，或其它文件集合ID")
    status_url: Optional[str] = Field(default=None, description="状态回调接口，处理完成后发送回调请求")


class LearnRequest(DocumentIndexRequest):
    file_type: Optional[str] = Field(default=None, description="文件类型，小写后缀名，如.pdf、.md、.txt等")
    file_path: Optional[Union[str, list[str]]] = Field(
        default=None, description="文件路径，文件系统路径或下载路径，文件有多个片段时，可以传每个片段的路径列表"
    )
    bundle_id: Optional[str] = Field(default=None, description="文件包ID，可以是知识库ID，或其它文件集合ID")
    reparse: bool = Field(default=True, description="是否重新解析")
    status_url: Optional[str] = Field(default=None, description="状态回调接口，处理完成后发送回调请求")


@router.post("/parse")
async def _parse_file(request: ParseRequest, raw_request: Request):  # pyright: ignore
    orch: Orchestrator = raw_request.app.ORCHESTRATOR
    result = await orch.run(router="rag-index-parse", request=request, request_type="rag")
    return JSONResponse(content=result)


@router.post("/split")
async def _split_document(request: SplitRequest, raw_request: Request):  # pyright: ignore
    orch: Orchestrator = raw_request.app.ORCHESTRATOR
    result = await orch.run(router="rag-index-split", request=request, request_type="rag")
    return JSONResponse(content=result)


@router.post("/extract")
async def _extract_metadata(request: ExtractRequest, raw_request: Request):  # pyright: ignore
    orch: Orchestrator = raw_request.app.ORCHESTRATOR
    result = await orch.run(router="rag-index-extract", request=request, request_type="rag")
    return JSONResponse(content=result)


@router.post("/embed")
async def _get_embeddings(request: EmbedRequest, raw_request: Request):  # pyright: ignore
    orch: Orchestrator = raw_request.app.ORCHESTRATOR
    result = await orch.run(router="rag-index-embed", request=request, request_type="rag")
    return JSONResponse(content=result)


@router.post("/store")
async def _store_chunks(request: StoreRequest, raw_request: Request):  # pyright: ignore
    """上传文件，文档分段，提取元信息，计算Embedding，存储到数据库"""
    orch: Orchestrator = raw_request.app.ORCHESTRATOR
    result = await orch.run(router="rag-index-index", request=request, request_type="rag")
    return JSONResponse(content=result)


@router.post("/learn")
async def _reindex_file(request: LearnRequest, raw_request: Request):  # pyright: ignore
    """重新学习，文档分段，提取元信息，计算Embedding，存储到数据库"""
    orch: Orchestrator = raw_request.app.ORCHESTRATOR
    result = await orch.run(router="rag-index-index", request=request, request_type="rag")
    return JSONResponse(content=result)


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
