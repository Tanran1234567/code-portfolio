#!/usr/bin/env python
# -*-coding: utf-8-*-

from typing import Optional

from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from mind_zone.config import Config
from mind_zone.state import TaskManager


class ChatBIFlow(Workflow):
    """Chat BI在线流程，query查询、检索、回复等"""

    def __init__(self, config: Config, task_mgr: TaskManager, timeout: Optional[float] = None):
        super().__init__(timeout=timeout)

        self._config = config
        self._task_mgr = task_mgr

    @step
    async def process(self, ctx: Context, ev: StartEvent) -> StopEvent:
        return StopEvent()
