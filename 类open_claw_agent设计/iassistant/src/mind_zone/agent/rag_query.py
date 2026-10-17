#!/usr/bin/env python
# -*-coding: utf-8-*-

from copy import deepcopy
from typing import Any, Optional, Union

from llama_index.core.base.response.schema import AsyncStreamingResponse
from llama_index.core.schema import NodeWithScore

from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from mind_zone.base import select_kwargs
from mind_zone.config import Config
from mind_zone.schemas import FormatOutput, ProgressEvent, QueryAnswerRequest
from mind_zone.state import TaskManager, TaskResult
from mind_zone.tasks import stream_synthesize_response
from mind_zone.utils import Logger


class RAGSearchEvent(Event):
    pass


class RAGPostprocessEvent(Event):
    pass


class RAGSynthesizeEvent(Event):
    pass


class RAGQueryFlow(Workflow):
    """RAG在线流程，query查询、检索、回复等"""

    def __init__(self, config: Config, task_mgr: TaskManager, timeout: Optional[float] = None):
        super().__init__(timeout=timeout)

        self._config = config
        self._task_mgr = task_mgr

    @step
    async def process(self, ctx: Context, ev: StartEvent) -> Union[StopEvent, "RAGSearchEvent"]:
        request = await ctx.store.get("request")
        app_config = await ctx.store.get("config")
        if not isinstance(request, QueryAnswerRequest):
            Logger.error(f"Invalid request type {type(request).__name__} for RAG query")
            return StopEvent(result=FormatOutput("type-err"))

        req_data = request.model_dump()
        req_data["text"] = ev.query or request.text
        req_data["history"] = ev.context or request.history or []
        await ctx.store.set("req_data", req_data)
        await ctx.store.set("template", ev.template)
        await ctx.store.set("stream", ev.stream)

        from mind_zone.routers.common import query_opt_

        query_opt_result = await query_opt_(
            self._task_mgr,
            app_config,
            req_data,
            wait_kwargs={"timeout": 20},
        )
        if not query_opt_result.success:
            return StopEvent(result=FormatOutput("query-opt", stream=ev.stream) if ev.stream else FormatOutput("query-opt"))

        raw_query = req_data["text"]
        opt_query = query_opt_result.result.get("query_str") or raw_query
        req_data["raw_query"] = raw_query
        req_data["text"] = opt_query
        await ctx.store.set("req_data", req_data)
        Logger.debug(f"原始query: {raw_query}")
        Logger.debug(f"优化query: {opt_query}")
        ctx.write_event_to_stream(
            ProgressEvent(
                ckpt_id="query-rewrite",
                content_type="query",
                content={"raw_query": raw_query, "query": opt_query},
            )
        )
        return RAGSearchEvent()

    @step
    async def retrieve(self, ctx: Context, ev: "RAGSearchEvent") -> Union[StopEvent, "RAGPostprocessEvent"]:
        app_config = await ctx.store.get("config")
        req_data = await ctx.store.get("req_data")
        stream = await ctx.store.get("stream")
        ds_config = self._config.get_config(db_type=app_config["document.store.text.ds_type"]) or {}
        from mind_zone.routers.common import search_documents_

        search_result = await search_documents_(
            self._task_mgr,
            app_config,
            req_data,
            ds_config=ds_config,
            wait_kwargs={"timeout": 20},
        )
        if not search_result.success:
            return StopEvent(result=FormatOutput("retrieval", stream=stream) if stream else FormatOutput("retrieval"))

        await ctx.store.set("search_result", search_result)
        return RAGPostprocessEvent()

    @step
    async def postprocess(self, ctx: Context, ev: "RAGPostprocessEvent") -> Union[StopEvent, "RAGSynthesizeEvent"]:
        app_config = await ctx.store.get("config")
        req_data = await ctx.store.get("req_data")
        stream = await ctx.store.get("stream")
        search_result: TaskResult = await ctx.store.get("search_result")

        if search_result.result and "query.postprocess.nodes" in app_config:
            from mind_zone.routers.common import postprocess_nodes_

            chunk_result = await postprocess_nodes_(
                self._task_mgr,
                app_config["query"],
                req_data,
                parent_id=search_result.id,
                wait_kwargs={"timeout": 90},
            )
            if not chunk_result.success:
                return StopEvent(
                    result=FormatOutput("postprocess", stream=stream) if stream else FormatOutput("postprocess")
                )
        else:
            chunk_result = search_result

        await ctx.store.set("chunk_result", chunk_result)
        chunk_list = self._format_chunks(chunk_result.result)
        ctx.write_event_to_stream(
            ProgressEvent(ckpt_id="after-retrieval", content_type="retrieval", content=chunk_list)
        )
        return RAGSynthesizeEvent()

    @step
    async def synthesize(self, ctx: Context, ev: "RAGSynthesizeEvent") -> StopEvent:
        request: QueryAnswerRequest = await ctx.store.get("request")
        app_config = await ctx.store.get("config")
        chunk_result: TaskResult = await ctx.store.get("chunk_result")
        req_data = await ctx.store.get("req_data")
        template = await ctx.store.get("template")

        synthesize_kwargs: dict[str, Any] = {
            "query": req_data.get("raw_query") or request.text,
            "method": app_config["response.synthesize.text.method"],
            "chunks": chunk_result.result,
            "client": select_kwargs(app_config["response.synthesize.text"]),
            "extra_body": {"chat_template_kwargs": {"enable_thinking": request.thinking}},
        }
        if template:
            synthesize_kwargs["text_qa_template"] = template

        response: AsyncStreamingResponse = await stream_synthesize_response(**synthesize_kwargs)
        return StopEvent(result=response)

    @staticmethod
    def _format_chunks(chunks: list[Union[NodeWithScore, dict[str, Any]]]) -> list[dict[str, Any]]:
        chunk_list = []
        for idx, chunk in enumerate(chunks):
            if isinstance(chunk, NodeWithScore):
                chunk_list.append(
                    {
                        "index": idx,
                        "doc_id": chunk.node_id,
                        "score": chunk.get_score(),
                        "file_id": chunk.metadata.get("file_id"),
                        "bundle_id": chunk.metadata.get("bundle_id"),
                        "text": chunk.text,
                    }
                )
                continue

            node = deepcopy(chunk.get("node") or {})
            metadata = node.get("metadata", {})
            chunk_list.append(
                {
                    "index": idx,
                    "doc_id": chunk.get("node_id") or node.get("id_") or node.get("doc_id"),
                    "score": chunk.get("score"),
                    "file_id": metadata.get("file_id"),
                    "bundle_id": metadata.get("bundle_id"),
                    "text": node.get("text") or chunk.get("text"),
                }
            )
        return chunk_list

