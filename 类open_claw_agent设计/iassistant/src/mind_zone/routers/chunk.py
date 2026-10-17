#!/usr/bin/env python
# -*-coding: utf-8-*-

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from llama_index.core.schema import Document

from mind_zone.config import Config
from mind_zone.register import Registry
from mind_zone.schemas import DocumentViewRequest, FormatOutput
from mind_zone.state import TaskManager
from mind_zone.tasks import fetch_documents

router = APIRouter(prefix="/chunk")
__EXCLUDED_METADATA_KEYS__ = ["_node_content", "_node_type", "document_id", "doc_id", "ref_doc_id"]


def __get_chunk_info(chunk: Document):
    return {
        "id": chunk.doc_id,
        "content": chunk.text,
        **{k: v for k, v in chunk.metadata.items() if k not in __EXCLUDED_METADATA_KEYS__},
    }


@router.post("/view")
async def _view_document(request: DocumentViewRequest, raw_request: Request):  # pyright: ignore
    task_mgr: TaskManager = raw_request.app.TASK_MANAGER
    config: Config = raw_request.app.CONFIG
    app_config = Registry.Application[request.app_name]

    file_result = await task_mgr.get_document(request.file_id, timeout=5, return_fields=["status"])
    if not file_result.success:
        return JSONResponse(content=FormatOutput("parse-file"))
    if file_result.result["status"] != "ready":
        task_id = task_mgr.search_task(file_id=request.file_id)
        return JSONResponse(content=FormatOutput("no-error", data={"task_id": task_id, "task_type": "parse-file"}))

    filters = [
        {"key": "file_id", "value": request.file_id},
    ]
    if request.chunk_idx is not None:
        filters.append({"key": "chunk_idx", "value": request.chunk_idx})

    ds_type = app_config["document.store.text.ds_type"]
    chunk_list: list[Document] = await fetch_documents(
        ds_type=ds_type,
        ds_config=config.get_config(db_type=ds_type) | app_config["document.store.text"],
        filters=filters,
    )

    chunk_list = [__get_chunk_info(d) for d in chunk_list]
    return JSONResponse(
        content=FormatOutput("no-error", data={"num_chunks": len(chunk_list), "chunk_list": chunk_list})
    )
