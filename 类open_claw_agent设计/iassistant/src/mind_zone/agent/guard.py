#!/usr/bin/env python
# -*-coding: utf-8-*-

import json
from dataclasses import dataclass
from typing import Optional, Any

from llama_index.core.base.response.schema import PydanticResponse
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

from mind_zone.api import OpenAIChatClient, RemoteAPI
from mind_zone.base import get_init_kwargs
from mind_zone.register import Registry
from mind_zone.utils import Logger


@dataclass
class GuardResponse:
    """响应容器"""

    blocked: bool
    rule: str = ""
    reason: str = ""

    @property
    def result(self):
        return "block" if self.blocked else "pass"


class Guard:
    """安全解析器"""

    def __init__(self, tag: str):
        self._tag = tag

    def _parse(self, llm_output: str) -> GuardResponse:
        try:
            review_result = json.loads(llm_output.strip())
            if review_result.get("triggered"):
                rule = review_result.get("rule", "Unknown")
                reason = review_result.get("reason", "")
                Logger.warning("[%s] BLOCK rule=%s reason=%s", self._tag, rule, reason)
                return GuardResponse(blocked=True, rule=rule, reason=reason)

            return GuardResponse(blocked=False)

        except Exception as e:
            Logger.warning("[%s] JSON Parse failed: %s. Blocking by default.", self._tag, e)
            return GuardResponse(blocked=True, rule="格式异常", reason="安全护栏输出非标准JSON")


class GuardFlow(Workflow):
    """
    安全护栏工作流：调用模型 -> 获取文本 -> Guard 解析
    """

    def __init__(self, direction: str, timeout: Optional[float] = None):
        super().__init__(timeout=timeout)

        llm_kwargs = get_init_kwargs(Registry.Application[f"workflow-default.flow.{direction}"])
        self._model: OpenAIChatClient = RemoteAPI(**llm_kwargs)
        self._tag = f"[GUARD: {direction}]"

    @step
    async def check(self, ev: StartEvent) -> StopEvent:
        template: PromptTemplate = ev.template
        format_args = {
            "user_input": ev.query,
            "model_output": ev.response,
        }
        prompt_str = template.format(**format_args)
        response = await self._model.acomplete(prompt_str)

        guard = Guard(tag=self._tag)
        guard_result = guard.parse(response)

        return StopEvent(result=PydanticResponse(response=guard_result))
