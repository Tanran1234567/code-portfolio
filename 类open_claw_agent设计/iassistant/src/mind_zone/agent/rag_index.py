#!/usr/bin/env python
# -*-coding: utf-8-*-

import asyncio
import time
from typing import Optional, Union

from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from mind_zone.agent.utils import update_kwargs
from mind_zone.base import select_kwargs
from mind_zone.config import Config
from mind_zone.schemas import (
    DocumentIndexRequest,
    EmbedRequestEvent,
    ExtractRequestEvent,
    FormatOutput,
    ParseRequestEvent,
    SplitRequestEvent,
    StoreRequestEvent,
)
from mind_zone.state import TaskManager, TaskState
from mind_zone.tasks import (
    extract_keyword,
    extract_question,
    extract_summary,
    extract_title,
    get_embeddings,
    load_from_file,
    split_document,
    store_documents,
)
from mind_zone.types import SkipDict
from mind_zone.utils import Logger, truncate_repr


class RAGIndexFlow(Workflow):
    """RAG离线流程，文档分快、存储"""

    def __init__(self, config: Config, task_mgr: TaskManager, timeout: Optional[float] = None):
        super().__init__(timeout=timeout)

        self._config = config
        self._task_mgr = task_mgr

        self._extract_fn = {
            "title": extract_title,
            "keyword": extract_keyword,
            "summary": extract_summary,
            "question": extract_question,
        }

        self._routers = {
            "parse": ["rag-index-parse"],
            "split": ["rag-index-split"],
            "extract": ["rag-index-extract"],
            "embed": ["rag-index-embed"],
            "index": ["rag-index-index", "rag-index-reindex"],
        }

    @step
    async def process(
        self, ctx: Context, ev: StartEvent
    ) -> Union[
        ParseRequestEvent, SplitRequestEvent, ExtractRequestEvent, EmbedRequestEvent, StoreRequestEvent, StopEvent
    ]:
        request = await ctx.store.get("request")
        if isinstance(request, DocumentIndexRequest):
            return ParseRequestEvent()
        else:
            Logger.error(f"Invalid request type {type(request).__name__} for RAG indexing")
            return StopEvent()

    @step
    async def parse_file(self, ctx: Context, ev: ParseRequestEvent) -> Union[SplitRequestEvent, StopEvent]:
        tick_ts = time.time()
        router = await ctx.store.get("router")
        config = SkipDict(await ctx.store.get("config"))
        request: DocumentIndexRequest = await ctx.store.get("request")

        file_result = await self._task_mgr.get_document(
            request.file_id, return_fields=["metadata", "file_path", "bundle_id"], **ev.wait_kwargs
        )
        if file_result.result:
            await ctx.store.set("bundle_id", file_result.result.get("bundle_id"))

        if router not in self._routers["parse"] and not request.reparse:
            if file_result.success:
                return SplitRequestEvent(wait_kwargs=update_kwargs(ev.wait_kwargs, tick_ts))

        if not file_result.result and (request.file_type == ".unk" or not request.file_path):
            return StopEvent(result=FormatOutput("value-err"))

        file_type = request.file_type or file_result.result["metadata"].get("file_type")
        task_state = self._task_mgr.launch_task(
            load_from_file,
            file_type=file_type,
            file_path=request.file_path or file_result.result.get("file_path"),
            method=config[f"document.read_file{file_type}.method"],
            client=select_kwargs(config[f"document.read_file{file_type}"]),
            file_id=request.file_id,
            bundle_id=request.bundle_id or file_result.result.get("bundle_id"),
            reparse=request.reparse,
            status_url=request.status_url if router in self._routers["parse"] else None,
            extra_info={"purpose": "parse-document", **ev.extra_info},
        )

        if task_state.status != "Running":
            if task_state.status == "Canceled":
                Logger.warning(f"File {request.file_id} has already been parsed.")
            task_id = self._task_mgr.search_task(file_id=request.file_id)
        else:
            task_id = task_state.id

        if not task_id:
            return StopEvent(result=FormatOutput("parse-file"))

        if router in self._routers["parse"] and request.status_url:
            return StopEvent(result=FormatOutput("no-error", data={"task_id": task_id}))

        if router not in self._routers["parse"]:
            return SplitRequestEvent(task_id=task_id, wait_kwargs=update_kwargs(ev.wait_kwargs, tick_ts))

        task_result = await self._task_mgr.wait_for_done(task_id, **ev.wait_kwargs)
        if not task_result.success:
            return StopEvent(result=FormatOutput("parse-file"))

        Logger.debug("Parse file successfully:\n{}".format(truncate_repr(task_result.result[0].text, 64)))
        contents = "\n\n".join([d.text for d in task_result.result])
        return StopEvent(result=FormatOutput("no-error", data={"content": contents}))

    @step
    async def split_document(
        self, ctx: Context, ev: SplitRequestEvent
    ) -> Optional[Union[ExtractRequestEvent, EmbedRequestEvent, StopEvent]]:
        tick_ts = time.time()
        router = await ctx.store.get("router")
        config = SkipDict(await ctx.store.get("config"))
        request: DocumentIndexRequest = await ctx.store.get("request")

        parse_id = ev.task_id or self._task_mgr.search_task(file_id=request.file_id)
        if not parse_id:
            return StopEvent(result=FormatOutput("parse-file"))

        parse_result = await self._task_mgr.wait_for_done(parse_id)
        if not parse_result.success:
            return StopEvent(result=FormatOutput("parse-file"))

        contents = [d.text for d in parse_result.result]
        extra_info = ev.extra_info.copy() | {"parent_id": parse_id}
        task_state = self._task_mgr.launch_task(
            split_document,
            texts=contents,
            method=config["document.split.text.method"],
            client=select_kwargs(config["document.split.text"]),
            file_id=request.file_id,
            bundle_id=request.bundle_id or await ctx.store.get("bundle_id"),
            status_url=request.status_url if router in self._routers["split"] else None,
            extra_info={"purpose": "split-document", **extra_info},
        )
        if router in self._routers["split"] and request.status_url:
            return StopEvent(result=FormatOutput("no-error", data={"task_id": task_state.id}))

        if router in self._routers["extract"]:
            return ExtractRequestEvent(task_id=task_state.id, wait_kwargs=update_kwargs(ev.wait_kwargs, tick_ts))
        elif router in self._routers["embed"]:
            return EmbedRequestEvent(task_id=task_state.id, wait_kwargs=update_kwargs(ev.wait_kwargs, tick_ts))
        elif router in self._routers["index"]:
            ctx.send_event(
                ExtractRequestEvent(task_id=task_state.id, wait_kwargs=update_kwargs(ev.wait_kwargs, tick_ts)),
            )
            ctx.send_event(
                EmbedRequestEvent(task_id=task_state.id, wait_kwargs=update_kwargs(ev.wait_kwargs, tick_ts)),
            )
            ctx.send_event(StoreRequestEvent(task_ids=[task_state.id], task_type="split_document"))
            return

        task_result = await self._task_mgr.wait_for_done(task_state.id, **ev.wait_kwargs)
        if not task_result.success:
            return StopEvent(result=FormatOutput("split-doc", data={"content": contents}))

        chunk_list = [d.text for d in task_result.result]
        return StopEvent(
            result=FormatOutput("no-error", data={"num_chunks": len(chunk_list), "chunk_list": chunk_list})
        )

    @step
    async def extract_metadata(self, ctx: Context, ev: ExtractRequestEvent) -> Union[StoreRequestEvent, StopEvent]:
        tick_ts = time.time()
        router = await ctx.store.get("router")
        config = SkipDict(await ctx.store.get("config"))
        request: DocumentIndexRequest = await ctx.store.get("request")

        split_id = ev.task_id
        if not split_id:
            return StopEvent(result=FormatOutput("split-doc"))

        split_result = await self._task_mgr.wait_for_done(split_id)
        if not split_result.success:
            return StopEvent(result=FormatOutput("split-doc"))

        batch_size = max(8192 * len(split_result.result) // sum(len(d.text) for d in split_result.result), 1)
        num_batches = (len(split_result.result) + batch_size - 1) // batch_size
        task_states: list[tuple[int, str, TaskState]] = []
        all_fields = set(self._extract_fn.keys())
        for idx in range(num_batches):
            contents = [d.text for d in split_result.result[batch_size * idx : batch_size * (idx + 1)]]
            doc_ids = [d.doc_id for d in split_result.result[batch_size * idx : batch_size * (idx + 1)]]
            for field in self._extract_fn.keys():
                if idx == num_batches - 1:
                    all_fields.remove(field)
                if field not in config.get("document.extract", {}):
                    continue
                if request.meta_fields is not None and field not in request.meta_fields:
                    continue

                extra_info = ev.extra_info.copy() | {
                    "purpose": f"extract-{field}",
                    "parent_id": split_id,
                    "file_id": request.file_id,
                    "doc_ids": doc_ids,
                }
                task_state = self._task_mgr.launch_task(
                    self._extract_fn[field],
                    texts=contents,
                    method=config[f"document.extract.{field}.method"],
                    client=select_kwargs(config[f"document.extract.{field}"]),
                    status_url=request.status_url if router in self._routers["extract"] and not all_fields else None,
                    extra_info=extra_info,
                )
                task_states.append((idx, field, task_state))

        task_ids = [x[-1].id for x in task_states]
        if router in self._routers["extract"] and request.status_url:
            return StopEvent(result=FormatOutput("no-error", data={"task_ids": task_ids}))

        if router not in self._routers["extract"]:
            return StoreRequestEvent(
                task_ids=task_ids, task_type="extract_metadata", wait_kwargs=update_kwargs(ev.wait_kwargs, tick_ts)
            )

        task_results = await asyncio.gather(*[self._task_mgr.wait_for_done(x, **ev.wait_kwargs) for x in task_ids])
        metadata_list = [{"file_id": request.file_id, "chunk_id": d.metadata["chunk_id"]} for d in split_result.result]
        for state, result in zip(task_states, task_results):
            offset = batch_size * state[0]
            if result.success and result.result:
                for idx, val in enumerate(result.result):
                    metadata_list[idx + offset].update(val)

        Logger.debug(
            "Extract metadata successfully: {} chunks with {} fields".format(
                len(metadata_list), len(set(x[1] for x in task_states))
            )
        )
        return StopEvent(result=FormatOutput("no-error", data={"metadata": metadata_list}))

    @step
    async def get_embeddings(self, ctx: Context, ev: EmbedRequestEvent) -> Union[StoreRequestEvent, StopEvent]:
        tick_ts = time.time()
        router = await ctx.store.get("router")
        config = SkipDict(await ctx.store.get("config"))
        request: DocumentIndexRequest = await ctx.store.get("request")

        split_id = ev.task_id
        if not split_id:
            return StopEvent(result=FormatOutput("split-doc"))

        split_result = await self._task_mgr.wait_for_done(split_id)
        if not split_result.success:
            return StopEvent(result=FormatOutput("split-doc"))

        batch_size = max(8192 * len(split_result.result) // sum(len(d.text) for d in split_result.result), 1)
        num_batches = (len(split_result.result) + batch_size - 1) // batch_size
        task_states: list[TaskState] = []
        for idx in range(num_batches):
            contents = [d.text for d in split_result.result[batch_size * idx : batch_size * (idx + 1)]]
            doc_ids = [d.doc_id for d in split_result.result[batch_size * idx : batch_size * (idx + 1)]]
            extra_info = ev.extra_info.copy() | {
                "purpose": "get-embeddings",
                "parent_id": split_id,
                "file_id": request.file_id,
                "doc_ids": doc_ids,
            }
            task_state = self._task_mgr.launch_task(
                get_embeddings,
                texts=contents,
                method=config["document.embed.text.method"],
                client=select_kwargs(config["document.embed.text"]),
                file_id=request.file_id,
                status_url=request.status_url if router in self._routers["embed"] and idx == num_batches - 1 else None,
                extra_info=extra_info,
            )
            task_states.append(task_state)

        task_ids = [x.id for x in task_states]
        if router in self._routers["embed"] and request.status_url:
            return StopEvent(result=FormatOutput("no-error", data={"task_ids": task_ids}))

        if router not in self._routers["embed"]:
            return StoreRequestEvent(
                task_ids=task_ids, task_type="get_embeddings", wait_kwargs=update_kwargs(ev.wait_kwargs, tick_ts)
            )

        task_results = await asyncio.gather(*[self._task_mgr.wait_for_done(x, **ev.wait_kwargs) for x in task_ids])
        embeddings_list = []
        for result in task_results:
            if result.success and result.result:
                embeddings_list.extend(result.result)

        Logger.debug(
            "Extract embeddings successfully: {} chunks with {} dimension".format(
                len(embeddings_list), len(embeddings_list[0])
            )
        )
        return StopEvent(result=FormatOutput("no-error", data={"embeddings": embeddings_list}))

    @step
    async def store_chunk_data(self, ctx: Context, ev: StoreRequestEvent) -> Optional[StopEvent]:
        router = await ctx.store.get("router")
        config = SkipDict(await ctx.store.get("config"))
        request: DocumentIndexRequest = await ctx.store.get("request")

        events = ctx.collect_events(ev, [StoreRequestEvent] * 3)
        if events is None:
            return

        split_task, metadata_tasks, embedding_tasks = None, None, None
        for event in events:
            if event.task_type == "extract_metadata":
                metadata_tasks = event.task_ids
            elif event.task_type == "get_embeddings":
                embedding_tasks = event.task_ids
            elif event.task_type == "split_document":
                split_task = event.task_ids[0]

        ds_type = config["document.store.text.ds_type"]
        task_state = self._task_mgr.launch_task(
            store_documents,
            file_id=request.file_id,
            split_task=split_task,
            metadata_tasks=metadata_tasks or [],
            embedding_tasks=embedding_tasks or [],
            ds_type=ds_type,
            ds_config=self._config.get_config(db_type=ds_type) | config["document.store.text"],
            status_url=request.status_url if router in self._routers["index"] else None,
            extra_info={"purpose": "store-document", **ev.extra_info},
        )

        if request.status_url:
            return StopEvent(result=FormatOutput("no-error", data={"task_id": task_state.id}))

        task_result = await self._task_mgr.wait_for_done(task_state.id, **ev.wait_kwargs)
        if not task_result.success:
            return StopEvent(result=FormatOutput("store-doc"))

        Logger.debug(
            "Store document chunks successfully: {} chunks for file {}".format(len(task_result.result), request.file_id)
        )
        return StopEvent(result=FormatOutput("no-error", data={"node_ids": task_result.result}))
