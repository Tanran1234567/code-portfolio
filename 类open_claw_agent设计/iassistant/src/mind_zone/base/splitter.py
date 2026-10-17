#!/usr/bin/env python
# -*-coding: utf-8-*-

from typing import Any, Callable, Optional

import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser import (
    HTMLNodeParser,
    JSONNodeParser,
    MarkdownNodeParser,
)
from llama_index.core.node_parser import NodeParser as BaseNodeParser  # pyright: ignore
from llama_index.core.node_parser import (
    SemanticDoubleMergingSplitterNodeParser,
    SemanticSplitterNodeParser,
    SentenceWindowNodeParser,
    TextSplitter,
)
from llama_index.core.node_parser.node_utils import default_id_func
from llama_index.core.node_parser.text.semantic_splitter import DEFAULT_OG_TEXT_METADATA_KEY, SentenceCombination
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.core.schema import Document
from llama_index.core.text_splitter import CodeSplitter, SentenceSplitter, TokenTextSplitter

from mind_zone.register import Registry

Registry.Component.register("split.md.default")(MarkdownNodeParser)
Registry.Component.register("split.html.default")(HTMLNodeParser)
Registry.Component.register("split.json.default")(JSONNodeParser)
Registry.Component.register("split.code.default")(CodeSplitter)
Registry.Component.register("split.text.default")(SentenceSplitter)
Registry.Component.register("split.text.token-text")(TokenTextSplitter)
Registry.Component.register("split.text.sentence-window")(SentenceWindowNodeParser)
Registry.Component.register("split.text.semantic-double-merging")(SemanticDoubleMergingSplitterNodeParser)


@Registry.Component.register("split.text.semantic")
class SemanticSplitter(SemanticSplitterNodeParser):
    """Add max_chunk_size restriction to chunks"""

    max_chunk_size: int = Field(
        default=1000,
        description="Maximum length of chunk that can be subjected to verification (number of characters)",
    )

    @classmethod
    def class_name(cls) -> str:
        return "SemanticSplitter"

    @classmethod
    def from_defaults(
        cls,
        embed_model: Optional[BaseEmbedding] = None,
        breakpoint_percentile_threshold: Optional[int] = 95,
        buffer_size: Optional[int] = 1,
        max_chunk_size: int = 1000,
        sentence_splitter: Optional[Callable[[str], list[str]]] = None,
        original_text_metadata_key: str = DEFAULT_OG_TEXT_METADATA_KEY,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> "SemanticSplitter":
        callback_manager = callback_manager or CallbackManager([])

        sentence_splitter = sentence_splitter or split_by_sentence_tokenizer()
        if embed_model is None:
            try:
                from llama_index.embeddings.openai import (  # pants: no-infer-dep
                    OpenAIEmbedding,
                )

                embed_model = embed_model or OpenAIEmbedding()
            except ImportError:
                raise ImportError(
                    "`llama-index-embeddings-openai` package not found, "
                    "please run `pip install llama-index-embeddings-openai`"
                )

        return cls(
            embed_model=embed_model,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            buffer_size=buffer_size,
            max_chunk_size=max_chunk_size,
            sentence_splitter=sentence_splitter,
            original_text_metadata_key=original_text_metadata_key,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            id_func=id_func or default_id_func,
        )

    def _break_large_chunk(self, sentences: list[SentenceCombination], distances: list[float]) -> list[str]:
        if len(distances) <= 0:
            return ["".join([d["sentence"] for d in sentences])]

        max_dist_idx = max(list(enumerate(distances)), key=lambda x: x[1])[0]

        chunks = []
        combined_text = "".join([d["sentence"] for d in sentences[: max_dist_idx + 1]])
        if len(combined_text) <= self.max_chunk_size:
            chunks.append(combined_text)
        else:
            chunks.extend(self._break_large_chunk(sentences[: max_dist_idx + 1], distances[:max_dist_idx]))

        combined_text = "".join([d["sentence"] for d in sentences[max_dist_idx + 1 :]])
        if len(combined_text) <= self.max_chunk_size:
            chunks.append(combined_text)
        else:
            chunks.extend(self._break_large_chunk(sentences[max_dist_idx + 1 :], distances[max_dist_idx + 1 :]))
        return chunks

    def _build_node_chunks(self, sentences: list[SentenceCombination], distances: list[float]) -> list[str]:
        chunks = []
        if len(distances) > 0:
            breakpoint_distance_threshold = np.percentile(distances, self.breakpoint_percentile_threshold)
            indices_above_threshold = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

            # Chunk sentences into semantic groups based on percentile breakpoints
            start_index = 0
            for index in indices_above_threshold:
                group = sentences[start_index : index + 1]
                combined_text = "".join([d["sentence"] for d in group])
                if len(combined_text) > self.max_chunk_size:
                    chunks.extend(self._break_large_chunk(group, distances[start_index:index]))
                else:
                    chunks.append(combined_text)

                start_index = index + 1

            if start_index < len(sentences):
                combined_text = "".join([d["sentence"] for d in sentences[start_index:]])
                if len(combined_text) > self.max_chunk_size:
                    chunks.extend(self._break_large_chunk(sentences[start_index:], distances[start_index:]))
                else:
                    chunks.append(combined_text)
        else:
            # If, for some reason we didn't get any distances (i.e. very, very small documents) just
            # treat the whole document as a single node
            chunks = [" ".join([s["sentence"] for s in sentences])]
        return chunks
