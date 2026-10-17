#!/usr/bin/env python
# -*-coding: utf-8 -*-

import inspect
from itertools import takewhile
from typing import Any, Type

from llama_index.core.retrievers import BasePGRetriever, BaseRetriever, KnowledgeGraphRAGRetriever, VectorIndexRetriever
from llama_index.core.schema import MetadataMode, NodeRelationship, NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import VectorStoreQueryResult

from mind_zone.register import Registry
from mind_zone.utils import Logger

__EXCLUDED_LLM_METADATA_KEYS__ = [
    "file_id",
    "file_name",
    "file_type",
    "file_size",
    "bundle_id",
    "summary",
]

Registry.Component.register("retrieve.kgraph-db.default")(KnowledgeGraphRAGRetriever)
Registry.Component.register("retrieve.pgraph-db.default")(BasePGRetriever)


@Registry.Component.register("retrieve.kv-store.default")
@Registry.Component.register("retrieve.vector-db.default")
@Registry.Component.register("retrieve.vector-db.vector-index")
class VectorStoreRetriever(VectorIndexRetriever):

    def _convert_nodes_to_scored_nodes(self, query_result: VectorStoreQueryResult) -> list[NodeWithScore]:
        """Patch the retrieved nodes to exclude metadata keys when generation."""
        node_with_scores = super()._convert_nodes_to_scored_nodes(query_result)
        for item in node_with_scores:
            item.node.excluded_llm_metadata_keys = list(
                set(item.node.excluded_llm_metadata_keys + __EXCLUDED_LLM_METADATA_KEYS__)
            )
        return node_with_scores


def _create_extended_sibling_retriever(base_cls: Type[BaseRetriever]) -> Type[BaseRetriever]:
    assert issubclass(base_cls, VectorIndexRetriever)

    class ExtendedSiblingRetriever(base_cls):

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """
            Args:
                max_prev: 最大前节点个数，默认为2
                max_next: 最大后节点个数，默认为2
                remove_overlap: 是否对overlap去重，默认为True
                overlap_length: 分块时重叠的长度，默认为100
            """
            super().__init__(*args, **kwargs)

            self._max_prev = kwargs.pop("max_prev", 2)
            self._max_next = kwargs.pop("max_next", 2)
            self._remove_overlap = kwargs.pop("remove_overlap", True)
            self._overlap_length = kwargs.pop("overlap_length", 100)

        def _calculate_overlap(self, text1: str, text2: str) -> int:
            min_len = min(len(text1), len(text2))
            max_overlap = 0
            for i in range(1, min(min_len, self._overlap_length) + 1):
                if text1[-i:] == text2[:i]:
                    max_overlap = i
            return max_overlap

        def _get_sibling_nodes_part(self, node_with_score: NodeWithScore, relationship: NodeRelationship):
            node = node_with_score.node
            sibling_nodes: list[NodeWithScore] = []

            current_node = node
            max_siblings = self._max_prev if relationship == NodeRelationship.PREVIOUS else self._max_next
            for _ in range(max_siblings):
                next_node_info = current_node.relationships.get(relationship) if current_node.relationships else None
                if not next_node_info:
                    break

                next_node_id = next_node_info.node_id
                try:
                    next_node = self._vector_store.get_nodes([next_node_id])[0]
                    if self._remove_overlap:
                        current_content = current_node.get_content(metadata_mode=MetadataMode.NONE)
                        next_content = next_node.get_content(metadata_mode=MetadataMode.NONE)
                        if relationship == NodeRelationship.PREVIOUS:
                            overlap = self._calculate_overlap(next_content, current_content)
                            if overlap > 0:
                                next_node.set_content(next_content[:-overlap])
                        else:
                            overlap = self._calculate_overlap(current_content, next_content)
                            if overlap > 0:
                                next_node.set_content(next_content[overlap:])

                    sibling_nodes.append(NodeWithScore(node=next_node))
                    current_node = next_node

                except Exception as exc:
                    Logger.warning(f"Failed to retrieve node {next_node_id}: {exc}")
                    break

            if relationship == NodeRelationship.PREVIOUS:
                sibling_nodes.reverse()
            return sibling_nodes

        async def _aget_sibling_nodes_part(self, node_with_score: NodeWithScore, relationship: NodeRelationship):
            node = node_with_score.node
            sibling_nodes: list[NodeWithScore] = []

            current_node = node
            max_siblings = self._max_prev if relationship == NodeRelationship.PREVIOUS else self._max_next
            for _ in range(max_siblings):
                next_node_info = current_node.relationships.get(relationship) if current_node.relationships else None
                if not next_node_info:
                    break

                next_node_id = next_node_info.node_id
                try:
                    next_node = (await self._vector_store.aget_nodes([next_node_id]))[0]
                    if self._remove_overlap:
                        current_content = current_node.get_content(metadata_mode=MetadataMode.NONE)
                        next_content = next_node.get_content(metadata_mode=MetadataMode.NONE)
                        if relationship == NodeRelationship.PREVIOUS:
                            overlap = self._calculate_overlap(next_content, current_content)
                            if overlap > 0:
                                next_node.set_content(next_content[:-overlap])
                        else:
                            overlap = self._calculate_overlap(current_content, next_content)
                            if overlap > 0:
                                next_node.set_content(next_content[overlap:])

                    sibling_nodes.append(NodeWithScore(node=next_node))
                    current_node = next_node

                except Exception as exc:
                    Logger.warning(f"Failed to retrieve node {next_node_id}: {exc}")
                    break

            if relationship == NodeRelationship.PREVIOUS:
                sibling_nodes.reverse()
            return sibling_nodes

        def _get_sibling_nodes(self, node_with_score: NodeWithScore) -> list[list[NodeWithScore]]:
            nodes_prev = self._get_sibling_nodes_part(node_with_score, NodeRelationship.PREVIOUS)
            Logger.debug(f"Got {len(nodes_prev)} previous sibling nodes")
            nodes_next = self._get_sibling_nodes_part(node_with_score, NodeRelationship.NEXT)
            Logger.debug(f"Got {len(nodes_next)} next sibling nodes")
            return [nodes_prev, nodes_next]

        async def _aget_sibling_nodes(self, node_with_score: NodeWithScore) -> list[list[NodeWithScore]]:
            nodes_prev = await self._aget_sibling_nodes_part(node_with_score, NodeRelationship.PREVIOUS)
            Logger.debug(f"Got {len(nodes_prev)} previous sibling nodes")
            nodes_next = await self._aget_sibling_nodes_part(node_with_score, NodeRelationship.NEXT)
            Logger.debug(f"Got {len(nodes_next)} next sibling nodes")
            return [nodes_prev, nodes_next]

        def _merge_siblings(
            self,
            initial_node: NodeWithScore,
            nodes_prev: list[NodeWithScore],
            nodes_next: list[NodeWithScore],
            node_ids: set[str],
        ) -> NodeWithScore:
            # 前向遍历所有前向节点，去除重复节点
            nodes_prev = list(takewhile(lambda node: node.node.node_id not in node_ids, reversed(nodes_prev)))
            nodes_prev.reverse()
            node_ids.update([node.node.node_id for node in nodes_prev])

            # 后向遍历所有前向节点，去除重复节点
            nodes_next = list(takewhile(lambda node: node.node.node_id not in node_ids, nodes_next))
            node_ids.update([node.node.node_id for node in nodes_next])

            text = "".join([node.get_content() for node in nodes_prev + [initial_node] + nodes_next])
            initial_node.node.set_content(text)

        def _extend_with_siblings(self, initial_nodes: list[NodeWithScore]) -> list[NodeWithScore]:
            node_ids = set([node.node.node_id for node in initial_nodes])
            initial_nodes.sort(key=lambda node: node.score, reverse=True)

            for node_with_score in initial_nodes:
                nodes_prev, nodes_next = self._get_sibling_nodes(node_with_score)
                self._merge_siblings(node_with_score, nodes_prev, nodes_next, node_ids)

            return initial_nodes

        async def _aextend_with_siblings(self, initial_nodes: list[NodeWithScore]) -> list[NodeWithScore]:
            node_ids = set([node.node.node_id for node in initial_nodes])
            initial_nodes.sort(key=lambda node: node.score, reverse=True)

            for node_with_score in initial_nodes:
                Logger.debug(f"Extending with siblings for node {node_with_score.node.node_id}")
                nodes_prev, nodes_next = await self._aget_sibling_nodes(node_with_score)
                self._merge_siblings(node_with_score, nodes_prev, nodes_next, node_ids)

            return initial_nodes

        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            initial_nodes = super()._retrieve(query_bundle)
            return self._extend_with_siblings(initial_nodes)

        async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            initial_nodes = await super()._aretrieve(query_bundle)
            return await self._aextend_with_siblings(initial_nodes)

    ExtendedSiblingRetriever.__name__ = "ExtendedSibling" + base_cls.__name__

    signature = inspect.signature(ExtendedSiblingRetriever.__init__)
    params = list(inspect.signature(base_cls.__init__).parameters.values())
    extended_params = [
        inspect.Parameter("max_prev", inspect.Parameter.KEYWORD_ONLY, default=2),
        inspect.Parameter("max_next", inspect.Parameter.KEYWORD_ONLY, default=2),
        inspect.Parameter("remove_overlap", inspect.Parameter.KEYWORD_ONLY, default=True),
        inspect.Parameter("overlap_length", inspect.Parameter.KEYWORD_ONLY, default=100),
    ]
    if params[-1].kind == inspect.Parameter.VAR_KEYWORD:
        params = params[:-1] + extended_params + params[-1:]
    else:
        params.extend(extended_params)
    ExtendedSiblingRetriever.__init__.__signature__ = signature.replace(parameters=params)
    return ExtendedSiblingRetriever


Registry.Component.register("retrieve.vector-db.vector-index-extended")(
    _create_extended_sibling_retriever(VectorStoreRetriever)
)
