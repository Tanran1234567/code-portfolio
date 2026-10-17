#!/usr/bin/env python
# -*-coding: utf-8-*-

from typing import Optional

from llama_index.core.base.response.schema import AsyncStreamingResponse, PydanticResponse
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

from mind_zone.api import OpenAIChatClient, RemoteAPI, astream_completion_response_to_tokens
from mind_zone.base import get_init_kwargs
from mind_zone.register import Registry


class DirectFlow(Workflow):

    def __init__(self, timeout: Optional[float] = None):
        super().__init__(timeout=timeout)

        llm_kwargs = get_init_kwargs(Registry.Application["workflow-default.flow.direct"])
        self._model: OpenAIChatClient = RemoteAPI(**llm_kwargs)

    @step
    async def generate(self, ev: StartEvent) -> StopEvent:
        template: PromptTemplate = ev.template or "{query_str}"
        if ev.stream:
            response = await self._model.astream_complete(template.format(query_str=ev.query))
            result = AsyncStreamingResponse(
                response_gen=await astream_completion_response_to_tokens(response),
            )
        else:
            response = await self._model.acomplete(template.format(query_str=ev.query))
            result = PydanticResponse(response=response)
        return StopEvent(result=result)
