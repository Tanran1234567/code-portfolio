#!/usr/bin/env python
# -*-coding: utf-8-*-

import asyncio
import re
import unicodedata
from typing import Optional, Union

from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step

from mind_zone.agent.chatbi import ChatBIFlow
from mind_zone.agent.direct import DirectFlow
from mind_zone.agent.intent import UserGroupType, get_intent_choices, get_intent_name, get_intent_template
from mind_zone.agent.rag_index import RAGIndexFlow
from mind_zone.agent.rag_query import RAGQueryFlow
from mind_zone.api import RemoteAPI
from mind_zone.base import get_init_kwargs
from mind_zone.config import Config
from mind_zone.register import Registry
from mind_zone.schemas import (
    DocumentIndexRequest,
    MultiResponseEvent,
    NaiveRequestEvent,
    ProgressEvent,
    QueryAnswerRequest,
    QueryIntent,
    QueryRequestEvent,
)
from mind_zone.state import TaskManager
from mind_zone.utils import Logger


class Orchestrator(Workflow):
    """主流程入口，对query进行分解，派发任务，并聚合结果"""

    def __init__(self, config: Config, task_mgr: TaskManager, timeout: Optional[float] = None):
        """
        Args:
            config: Config, configurations of FastAPI app
            task_mgr: TaskManager, manager for background tasks using dramatiq
        """
        super().__init__(timeout=timeout)

        self._config = config
        self._task_mgr = task_mgr
        # 问答处理工作流
        self._runner = AgentWorker(self._config, self._task_mgr, timeout=200)
        # 非问答处理工作流
        self._handlers = {
            "rag": RAGIndexFlow(self._config, self._task_mgr, timeout=1800),
        }

    @step
    async def dispatch(self, ctx: Context, ev: StartEvent) -> Union[NaiveRequestEvent, MultiResponseEvent]:
        if isinstance(ev.request, DocumentIndexRequest):
            return NaiveRequestEvent(router=ev.router, request=ev.request, request_type=ev.request_type)

        handler = self._runner.run(request=ev.request, stream=ev.get("stream", False), user_id=ev.get("user_id"))

        async for event in handler.stream_events():
            ctx.write_event_to_stream(event)

        return MultiResponseEvent(responses=[await handler])

    @step
    async def handle_request(self, ev: NaiveRequestEvent) -> StopEvent:
        processor: Workflow = self._handlers[ev.request_type]
        new_ctx = Context(processor)
        await new_ctx.store.set("router", ev.router)
        await new_ctx.store.set("config", Registry.Application[ev.request.app_name])
        await new_ctx.store.set("request", ev.request)

        result = await processor.run(ctx=new_ctx)
        return StopEvent(result=result)

    @step
    async def aggregate(self, ev: MultiResponseEvent) -> StopEvent:
        if len(ev.responses) == 1:
            return StopEvent(result=ev.responses[0])

        raise NotImplementedError()


class AgentWorker(Workflow):
    """任务流程，独立执行query改写、分解、扩展、路由等"""

    def __init__(self, config: Config, task_mgr: TaskManager, timeout: Optional[float] = None):
        super().__init__(timeout=timeout)

        self._config = config
        self._task_mgr = task_mgr

        llm_kwargs = get_init_kwargs(Registry.Application["workflow-default.query.route"])
        self._selector = LLMSingleSelector.from_defaults(llm=RemoteAPI(**llm_kwargs))

        self._runners = {
            "direct": DirectFlow(timeout=60),
            "rag": RAGQueryFlow(self._config, self._task_mgr, timeout=180),
            "chatbi": ChatBIFlow(self._config, self._task_mgr, timeout=180),
        }

    @step
    async def process_query(self, ctx: Context, ev: StartEvent) -> QueryRequestEvent:
        request = ev.get("request")
        config = Registry.Application[request.app_name]
        await ctx.store.set("config", config)
        await ctx.store.set("request", request)

        query, context = None, None
        if isinstance(request, QueryAnswerRequest):
            query = request.text
            context = request.history
        else:
            pass

        # query preprocess: refine -> expand -> decompose
        raw_query = query or ""

        query = self._refine_query_text(raw_query)
        if query != raw_query:
            ctx.write_event_to_stream(
                ProgressEvent(
                    ckpt_id="query-refine",
                    content_type="query",
                    content={"query": raw_query, "refined": query},
                )
            )

        pre_expand = query
        query = self._expand_query_text(pre_expand)
        if query != pre_expand:
            ctx.write_event_to_stream(
                ProgressEvent(
                    ckpt_id="query-expand",
                    content_type="query",
                    content={"query": pre_expand, "expanded": query},
                )
            )

        sub_queries = self._decompose_query(query)
        await ctx.store.set("sub_queries", sub_queries)
        if len(sub_queries) > 1:
            ctx.write_event_to_stream(
                ProgressEvent(
                    ckpt_id="query-decompose",
                    content_type="query",
                    content={"queries": sub_queries},
                )
            )

        l1_intent, l2_intent = await self._route_query(query)

        user_group = UserGroupType.STANDARD if ev.get("user_id") else UserGroupType.GUEST
        await ctx.store.set("user_group", user_group)
        template = get_intent_template(l1_intent, l2_intent, user_group=user_group)

        ctx.write_event_to_stream(
            ProgressEvent(ckpt_id="query-routing", content_type="intent", content={"l1": l1_intent, "l2": l2_intent})
        )
        return QueryRequestEvent(
            intent=QueryIntent(l1=l1_intent, l2=l2_intent),
            query=query,
            context=context,
            template=template,
            stream=ev.stream,
        )

    @step
    async def handle_request(self, ctx: Context, ev: QueryRequestEvent) -> StopEvent:
        if not ev.intent.l1:
            Logger.error(f"Unable to determine user's intent from query '{ev.query}'")
            return StopEvent()

        request = await ctx.store.get("request")
        sub_queries = await ctx.store.get("sub_queries", default=[ev.query])
        if len(sub_queries) <= 1:
            return await self._run_single_flow(ctx, request, ev)

        user_group = await ctx.store.get("user_group", default=UserGroupType.GUEST)
        handlers = []
        for query in sub_queries:
            l1_intent, l2_intent = await self._route_query(query)
            template = get_intent_template(l1_intent, l2_intent, user_group=user_group)
            sub_ev = QueryRequestEvent(
                intent=QueryIntent(l1=l1_intent, l2=l2_intent),
                query=query,
                context=ev.context,
                template=template,
                stream=ev.stream,
            )
            handlers.append(self._run_single_flow(ctx, request, sub_ev))

        responses = await asyncio.gather(*handlers)
        return StopEvent(result=[response.result for response in responses])

    async def _run_single_flow(self, ctx: Context, request: QueryAnswerRequest, ev: QueryRequestEvent) -> StopEvent:
        runner: Workflow = self._runners[ev.intent.l1]
        new_ctx = Context(runner)
        await new_ctx.store.set("config", await ctx.store.get("config"))
        await new_ctx.store.set("request", request)
        handler = runner.run(ctx=new_ctx, query=ev.query, context=ev.context, template=ev.template, stream=ev.stream)

        async for event in handler.stream_events():
            ctx.write_event_to_stream(event)

        return StopEvent(result=await handler)

    async def _route_query(self, query: str) -> tuple[str, str]:
        l1_result = await self._selector.aselect(get_intent_choices(), query=query)
        l1_intent = get_intent_name(l1_index=l1_result.ind)
        l2_result = await self._selector.aselect(get_intent_choices(l1_intent), query=query)
        l2_intent = get_intent_name(l1_intent=l1_intent, l2_index=l2_result.ind)
        return l1_intent, l2_intent

    @staticmethod
    def _refine_query_text(query: Optional[str]) -> str:
        """Normalize unicode and horizontal whitespace; preserve newlines for decompose."""
        if not query:
            return ""
        t = unicodedata.normalize("NFKC", query)
        t = t.replace("\u200b", "").replace("\ufeff", "")
        lines: list[str] = []
        for line in t.splitlines():
            line = re.sub(r"[ \t\r\f\v]+", " ", line).strip()
            if line:
                lines.append(line)
        return "\n".join(lines)

    # Strip only discourse markers; do not consume content verbs like 介绍/说明 (avoids
    # "请帮我介绍一下…" -> "一下…" or "请帮我介绍A" -> "A").
    _POLITE_PREFIX = re.compile(
        r"^(?:"
        r"请问一下?|"
        r"请问|"
        r"请(?:你|帮忙|帮我)|"
        r"能否|可不可以|"
        r"麻烦(?:你|帮忙)?|"
        r"想(?:问问|咨询|了解一下)|"
        r"帮我"
        r")[\s，,。.]*",
        re.UNICODE,
    )

    @classmethod
    def _expand_single_line(cls, line: str) -> str:
        s = line.strip()
        if not s:
            return s
        stripped = cls._POLITE_PREFIX.sub("", s).strip()
        base = stripped or s
        extras = re.findall(r"[（(]([^）)]+)[）)]", base)
        if extras:
            tail = " ".join(x.strip() for x in extras if x.strip())
            if tail:
                base = f"{base} {tail}"
        return base

    @classmethod
    def _expand_query_text(cls, query: str) -> str:
        """Strip common Chinese polite openers and append parenthetical hints for retrieval."""
        if not query:
            return ""
        return "\n".join(cls._expand_single_line(line) for line in query.splitlines())

    @staticmethod
    def _decompose_query(query: str) -> list[str]:
        if not query:
            return []

        parts = [x.strip() for x in re.split(r"(?:[；;]|(?:\n+)|(?:另外)|(?:同时)|(?:并且)|(?:以及))", query) if x.strip()]
        if len(parts) <= 1:
            return [query]

        return parts

