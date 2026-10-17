#!/usr/bin/env python
# -*-coding: utf-8-*-

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from llama_index.core.base.response.schema import AsyncStreamingResponse

from mind_zone.agent import Orchestrator
from mind_zone.register import Registry
from mind_zone.routers.common import get_embeddings_, search_documents_
from mind_zone.schemas import FormatOutput, ProgressEvent, QueryAnswerRequest
from mind_zone.state import TaskManager
from mind_zone.utils import Logger

router = APIRouter(prefix="/query")


@router.post("/search")
async def _search_chunks(request: QueryAnswerRequest, raw_request: Request):  # pyright: ignore
    req_dict = request.model_dump()
    task_mgr: TaskManager = raw_request.app.TASK_MANAGER
    app_config = Registry.Application[request.app_name]
    # TODO: query优化
    embed_result = await get_embeddings_(task_mgr, app_config["document"], req_dict, wait_kwargs={"timeout": 30})
    if not embed_result.success:
        return JSONResponse(content=FormatOutput("embedding"))
    # TODO: 添加其它筛选条件
    task_result = await search_documents_(
        task_mgr, app_config, req_dict, parent_id=embed_result.id, wait_kwargs={"timeout": 20}
    )
    if not task_result.success:
        return JSONResponse(content=FormatOutput(""))
    return JSONResponse(content=FormatOutput("no-error", data={"chunk_list": task_result.result}))


@router.post("/respond")
async def _respond_query(request: QueryAnswerRequest, raw_request: Request):  # pyright: ignore
    orch: Orchestrator = raw_request.app.ORCHESTRATOR
    resp_list: list[dict] = []

    handler = orch.run(request=request, stream=True, user_id=request.user_id)
    async for ev in handler.stream_events():
        if isinstance(ev, ProgressEvent):
            resp_list.append({"ckpt_id": ev.ckpt_id, "content_type": ev.content_type, "content": ev.content})

    response = await handler
    if not isinstance(response, AsyncStreamingResponse):
        if callable(response):
            return StreamingResponse(content=response())
        return JSONResponse(content=response or FormatOutput("runtime-err"))

    async def wrap_fn():
        for data_dict in resp_list:
            data_dict = {k: v for k, v in data_dict.items() if v is not None}
            yield json.dumps({"err_code": 0, "finish": False, "data": data_dict}, ensure_ascii=False) + "\n"

        is_reasoning = False
        cache_content = ""
        async for content in response.async_response_gen():
            if not content:
                continue
            cache_content += content

            data_dict = {"ckpt_id": "final-response", "content": content}
            if "<think>" in content:
                is_reasoning = True
                data_dict["content"] = content.split("<think>")[1].lstrip()
            elif "</think>" in content:
                is_reasoning = False
                data_dict["content"] = content.split("</think>")[1].lstrip()
                Logger.debug("Get thinking content: {}".format(cache_content.strip()))
                cache_content = ""

            data_dict["content_type"] = "thinking" if is_reasoning else "response"
            yield json.dumps({"err_code": 0, "finish": False, "data": data_dict}, ensure_ascii=False) + "\n"

        Logger.debug("Get response content: {}".format(cache_content.strip()))
        data_dict = {"ckpt_id": "final-response", "content_type": "response", "content": "\n"}
        yield json.dumps({"err_code": 0, "finish": True, "data": data_dict}, ensure_ascii=False) + "\n"

    return StreamingResponse(content=wrap_fn())

