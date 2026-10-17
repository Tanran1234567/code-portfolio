#!/usr/bin/env python
# -*-coding: utf-8-*-

import sys
from typing import Dict, Optional

from llama_index.core.indices.query.query_transform.base import IdentityQueryTransform
from llama_index.core.llms import LLM
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.schema import QueryBundle

from mind_zone.base.prompts import DEFAULT_HISTORY_QUERY_TRANSFORM_PROMPT
from mind_zone.register import Registry


@Registry.Component.register("query.query_opt.history")
class HistoryQueryTransform(IdentityQueryTransform):
    """
    使用历史会话内容进行查询转换。

    根据对话历史将用户的最新提问改写为一个独立、完整的查询语句，
    使其能够在不依赖上下文的情况下被准确理解和检索。

    用法::

        transform = HistoryQueryTransform(llm=llm)
        new_bundle = transform(
            query_bundle_or_str="有什么玩的地方推荐吗？",
            metadata={"history": [{"role": "user", "content": "北京天气怎么样？"}, {"role": "assistant", "content": "北京今天晴天，25°C。"}]},
        )

    Args:
        llm: 用于生成改写查询的语言模型
        prompt_template: 自定义的查询改写提示模板，
            需包含 ``{history_str}`` 和 ``{query_str}`` 变量
        include_original: 是否将原始查询一同作为 embedding 字符串
        max_history: 截取的最大历史数据
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        prompt_template: Optional[BasePromptTemplate] = None,
        include_original: bool = False,
        max_history: int = 5,
    ) -> None:
        super().__init__()
        from llama_index.core import Settings

        self._llm = llm or Settings.llm
        self._prompt_template = prompt_template or DEFAULT_HISTORY_QUERY_TRANSFORM_PROMPT
        self._include_original = include_original
        if max_history < 0:
            max_history = sys.maxsize
        self._max_history = max_history

    def _format_history(self, history: list[dict[str, str]]) -> str:
        parts: list[str] = []
        # TODO 根据llm max token限制截断
        history = reversed(history)
        for index, msg in enumerate(history, 1):
            if index > self._max_history:
                break
            role = msg.get("role", "")
            content = msg.get("content", "")
            if index > 2 and role == "assistant" and len(content) > 256:
                content = content[:256] + "...[已缩减]"
            parts.append(f"{role}: {content}")
        return "\n".join(reversed(parts))

    def _run(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
        query_str = query_bundle.query_str

        history: Optional[list[dict[str, str]]] = metadata.get("history")
        if not history:
            return query_bundle

        history_str = self._format_history(history)

        new_query_str = self._llm.predict(
            self._prompt_template,
            history_str=history_str,
            query_str=query_str,
        ).strip()

        embedding_strs = [new_query_str]
        if self._include_original:
            embedding_strs.extend(query_bundle.embedding_strs)

        return QueryBundle(
            query_str=new_query_str,
            custom_embedding_strs=embedding_strs,
        )
