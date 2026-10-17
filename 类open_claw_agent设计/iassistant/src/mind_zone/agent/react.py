#!/usr/bin/env python
# -*-coding: utf-8-*-

from typing import Optional

from llama_index.core.agent import ReActAgent as BaseReActAgent

from mind_zone.config import Config
from mind_zone.state import TaskManager


class ReActAgent(BaseReActAgent):
    """ReAct工作流"""

    def __init__(self, config: Config, task_mgr: TaskManager, timeout: Optional[float] = None):
        super().__init__(timeout=timeout)

        self._config = config
        self._task_mgr = task_mgr
