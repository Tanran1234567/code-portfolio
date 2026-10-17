#!/usr/bin/env python
# -*-coding: utf-8-*-

from typing import Any, Callable, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.postprocessor import (
    EmbeddingRecencyPostprocessor,
    FixedRecencyPostprocessor,
    LLMRerank,
    SentenceTransformerRerank,
    SimilarityPostprocessor,
    StructuredLLMRerank,
    TimeWeightedPostprocessor,
)
from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor as BasePostProcessor
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

from mind_zone.api import HTTPResponse, HTTPxClient
from mind_zone.base.prompts import DEFAULT_CHOICE_SELECT_PROMPT, RANKGPT_RERANK_PROMPT
from mind_zone.register import Registry
from mind_zone.utils import Logger

dispatcher = get_dispatcher(__name__)


def default_parse_scoring_answer_fn(response: HTTPResponse, top_n: Optional[int] = None):
    results = []
    try:
        resp_json = response.json()
        for result in resp_json.get("results", []):
            results.append({"index": result["index"], "relevance_score": result["relevance_score"]})
    except Exception as exc:
        Logger.warning(f"Failed to parse scoring response: {exc}")

    results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
    if not top_n or top_n <= 0:
        return results

    return results[:top_n]


@Registry.Component.register("postprocess.nodes.scoring-rerank")
class ScoringRerank(BasePostProcessor):

    model: str = Field(description="Cohere model name.")
    top_n: int = Field(description="Top N nodes to return.")
    base_url: Optional[str] = Field(description="Cohere base url.", default=None)

    _client: Optional[HTTPxClient] = PrivateAttr()

    def __init__(
        self,
        client: Optional[HTTPxClient] = None,
        model_name: Optional[str] = None,
        parse_scoring_answer_fn: Optional[Callable] = None,
        top_n: int = 5,
    ):
        super().__init__(top_n=top_n, model=model_name)

        self._client = client
        self._parse_scoring_answer_fn = parse_scoring_answer_fn or default_parse_scoring_answer_fn

    @classmethod
    def class_name(cls) -> str:
        return "ScoringRerank"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        """Postprocess nodes."""
        dispatcher.event(ReRankStartEvent(query=query_bundle, nodes=nodes, top_n=self.top_n, model_name=self.model))

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            texts = [node.node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes]
            response = self._client.post(
                self._client.base_url,
                json={"model": self.model, "query": query_bundle.query_str, "documents": texts},
            )
            if response.status_code != 200:
                Logger.warning(f"Failed to rerank documents using the scoring API at {self._client.base_url}.")
                return []

            results = self._parse_scoring_answer_fn(response, self.top_n)

            new_nodes = []
            for result in results:
                new_node_with_score = NodeWithScore(node=nodes[result["index"]].node, score=result["relevance_score"])
                new_nodes.append(new_node_with_score)
            event.on_end(payload={EventPayload.NODES: new_nodes})

        dispatcher.event(ReRankEndEvent(nodes=new_nodes))
        return new_nodes


@Registry.Component.register("postprocess.nodes.default")
@Registry.Component.register("postprocess.nodes.chatllm-rerank")
class ChatLLMRerank(LLMRerank):

    def __init__(
        self,
        llm: Optional[LLM] = None,
        custom_template: Optional[str] = None,
        choice_batch_size: int = 10,
        format_node_batch_fn: Optional[Callable] = None,
        parse_choice_select_answer_fn: Optional[Callable] = None,
        top_n: int = 5,
    ) -> None:
        if not custom_template:
            choice_select_prompt = DEFAULT_CHOICE_SELECT_PROMPT
        else:
            choice_select_prompt = PromptTemplate(custom_template, prompt_type=PromptType.CHOICE_SELECT)

        super().__init__(
            llm=llm,
            choice_select_prompt=choice_select_prompt,
            choice_batch_size=choice_batch_size,
            format_node_batch_fn=format_node_batch_fn,
            parse_choice_select_answer_fn=parse_choice_select_answer_fn,
            top_n=top_n,
        )


@Registry.Component.register("postprocess.nodes.rankgpt-rerank")
class RankGPTRerank(RankGPTRerank):

    def __init__(
        self,
        llm: Optional[LLM] = None,
        custom_template: Optional[str] = None,
        top_n: int = 5,
        verbose: bool = False,
    ):
        if not custom_template:
            rankgpt_rerank_prompt = RANKGPT_RERANK_PROMPT
        else:
            rankgpt_rerank_prompt = PromptTemplate(custom_template, prompt_type=PromptType.RANKGPT_RERANK)

        super().__init__(top_n=top_n, llm=llm, verbose=verbose, rankgpt_rerank_prompt=rankgpt_rerank_prompt)

    def _get_prefix_prompt(self, query: str, num: int) -> list[ChatMessage]:
        return [
            ChatMessage(
                role="system",
                content="你是RankGPT，一个能根据查询相关性对文本段落进行排序的智能助手。",
            ),
            ChatMessage(
                role="user",
                content=f"我将为你提供{num}个文本段落，每个段落前面都有一个数字标识[]。\n请根据这些段落与如下查询的相关性对它们进行排序。\n{query}",
            ),
            ChatMessage(role="assistant", content="好的，请提供文本段落。"),
        ]

    def create_permutation_instruction(self, item: dict[str, Any]) -> list[ChatMessage]:
        query = item["query"]
        num = len(item["hits"])

        messages = self._get_prefix_prompt(query, num)
        rank = 0
        for hit in item["hits"]:
            rank += 1
            content = hit["content"]
            content = content.replace("Title: Content: ", "")
            content = content.strip()
            # For Japanese should cut by character: content = content[:int(max_length)]
            content = " ".join(content.split()[:300])
            messages.append(ChatMessage(role="user", content=f"[{rank}] {content}"))
            messages.append(ChatMessage(role="assistant", content=f"接收到文本段落[{rank}]。"))
        messages.append(ChatMessage(role="user", content=self._get_post_prompt(query, num)))
        return messages


# TODO: 下面几种类型需要改造
Registry.Component.register("postprocess.nodes.cross-encoder-rerank")(SentenceTransformerRerank)
Registry.Component.register("postprocess.nodes.similarity")(SimilarityPostprocessor)
Registry.Component.register("postprocess.nodes.embedding-recency")(EmbeddingRecencyPostprocessor)
Registry.Component.register("postprocess.nodes.fixed-recency")(FixedRecencyPostprocessor)
Registry.Component.register("postprocess.nodes.structured-llm-rerank")(StructuredLLMRerank)
Registry.Component.register("postprocess.nodes.time-weighted")(TimeWeightedPostprocessor)


@Registry.Component.register("postprocess.nodes.text-api")
class TextAPIPostProcessor(BasePostProcessor):
    """调用API进行后处理"""

    pass
