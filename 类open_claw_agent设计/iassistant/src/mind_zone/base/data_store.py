#!/usr/bin/env python
# -*-coding: utf-8-*-

from typing import Any, Optional

from llama_index.core.schema import BaseNode, Document
from llama_index.core.vector_stores.types import BasePydanticVectorStore as BaseVectorStore  # pyright: ignore
from llama_index.core.vector_stores.types import MetadataFilters
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.opensearch.base import OpensearchVectorClient, OpensearchVectorStore
from opensearchpy import AsyncOpenSearch, NotFoundError, OpenSearch

from mind_zone.register import Registry
from mind_zone.utils import Logger

OPENSEARCH_PIPELINES = {
    "hybrid-rrf": {
        "description": "Reciprocal Rank Fusion (RRF) for Hybrid Search",
        "phase_results_processors": [
            {
                "score-ranker-processor": {
                    "combination": {
                        "technique": "rrf",
                        "rank_constant": 36,
                        "parameters": {
                            "weights": [0.3, 0.7],
                        },
                    },
                }
            }
        ],
    },
    "hybrid-min_max": {
        "description": "Min-Max and Weighted Arithmetic Mean for Hybrid Search",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {
                        "technique": "min_max",
                    },
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {
                            "weights": [0.3, 0.7],
                        },
                    },
                }
            }
        ],
    },
}


class OpensearchClient(OpensearchVectorClient):
    """优化Opensearch查询"""

    def _add_pipelines(self):
        for id, setting in OPENSEARCH_PIPELINES.items():
            try:
                pipeline = self._os_client.search_pipeline.get(id=id)
            except NotFoundError:
                pipeline = None

            if pipeline:
                continue

            try:
                self._os_client.search_pipeline.put(id=id, body=setting)
            except Exception as exc:
                Logger.error(f"Failed to add pipeline {id}: {exc}")

    async def _async_add_pipelines(self):
        for id, setting in OPENSEARCH_PIPELINES.items():
            try:
                pipeline = await self._os_async_client.search_pipeline.get(id=id)
            except NotFoundError:
                pipeline = None

            if pipeline:
                continue

            try:
                await self._os_async_client.search_pipeline.put(id=id, body=setting)
            except Exception as exc:
                Logger.error(f"Failed to add pipeline {id}: {exc}")

    def _update_idx_config(self):
        if "mappings" not in self._idx_conf:
            self._idx_conf["mappings"] = {}
        if "properties" not in self._idx_conf["mappings"]:
            self._idx_conf["mappings"]["properties"] = {}
        if self._text_field not in self._idx_conf["mappings"]["properties"]:
            self._idx_conf["mappings"]["properties"][self._text_field] = {"type": "text"}

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        self._update_idx_config()
        super()._ensure_initialized()
        self._add_pipelines()

    async def _async_ensure_initialized(self) -> None:
        if self._initialized:
            return

        self._update_idx_config()
        await super()._async_ensure_initialized()
        await self._async_add_pipelines()

    def _lexical_search_query(
        self,
        text_field: str,
        query_str: str,
        k: int,
        filters: Optional[MetadataFilters] = None,
        excluded_source_fields: Optional[list[str]] = None,
    ) -> dict:
        # TODO: 改写文本查询
        return super()._lexical_search_query(
            text_field, query_str, k, filters=filters, excluded_source_fields=excluded_source_fields
        )

    def get_nodes(
        self, node_ids: Optional[list[str]] = None, filters: Optional[MetadataFilters] = None
    ) -> dict[str, Any]:
        self._ensure_initialized()
        if not node_ids and not filters:
            return {}

        client: OpenSearch = self._os_client
        if filters:
            query = {"query": {"bool": {"filter": self._parse_filters(filters)}}}
            response = client.search(index=self._index, body=query, params={"_source": False, "size": 10000})
            if not isinstance(response, dict) or not response.get("hits", {}).get("hits", []):
                return {}

            filtered = []
            for hit in response["hits"]["hits"]:
                filtered.append(hit["_id"])
            node_ids = filtered if node_ids is None else [x for x in node_ids if x in filtered]

        return client.mget(index=self._index, body={"ids": node_ids})

    async def aget_nodes(
        self, node_ids: Optional[list[str]] = None, filters: Optional[MetadataFilters] = None
    ) -> dict[str, Any]:
        await self._async_ensure_initialized()
        if not node_ids and not filters:
            return {}

        client: AsyncOpenSearch = self._os_async_client
        if filters:
            query = {"query": {"bool": {"filter": self._parse_filters(filters)}}}
            response = await client.search(index=self._index, body=query, params={"_source": False, "size": 10000})
            if not isinstance(response, dict) or not response.get("hits", {}).get("hits", []):
                return {}

            filtered = []
            for hit in response["hits"]["hits"]:
                filtered.append(hit["_id"])
            node_ids = filtered if node_ids is None else [x for x in node_ids if x in filtered]

        return await client.mget(index=self._index, body={"ids": node_ids})


@Registry.Component.register("store.opensearch")
class OpensearchStore(OpensearchVectorStore):

    def __init__(self, *, endpoint: str, index_name: str, embedding_dim: int, **kwargs: Any) -> None:
        if not kwargs.get("search_pipeline"):
            kwargs["search_pipeline"] = "hybrid-rrf"
        super().__init__(OpensearchClient(endpoint, index_name, embedding_dim, **kwargs))

    def _to_nodes(self, hits: dict[str, Any]) -> list[BaseNode]:
        """Convert OpenSearch hits to LlamaIndex nodes"""

        def _to_node(doc: dict[str, Any]) -> BaseNode:
            return Document(
                id_=doc["_id"],
                text=doc["_source"][self.client._text_field],
                embedding=doc["_source"][self.client._embedding_field],
                relationships=doc["_source"]["metadata"].pop("relationships"),
                metadata={
                    k: v for k, v in doc["_source"]["metadata"].items() if k not in ["_node_content", "_node_type"]
                },
            )

        return [_to_node(doc) for doc in hits.get("docs", [])]

    def get_nodes(
        self, node_ids: Optional[list[str]] = None, filters: Optional[MetadataFilters] = None
    ) -> list[BaseNode]:
        nodes = self.client.get_nodes(node_ids, filters)
        return self._to_nodes(nodes)

    async def aget_nodes(
        self, node_ids: Optional[list[str]] = None, filters: Optional[MetadataFilters] = None
    ) -> list[BaseNode]:
        nodes = await self.client.aget_nodes(node_ids, filters)
        return self._to_nodes(nodes)


@Registry.Component.register("store.milvus")
class MilvusStore(MilvusVectorStore):

    def __init__(self, *, endpoint: str, index_name: str, embedding_dim: int, **kwargs: Any) -> None:
        super().__init__(uri=endpoint, collection_name=index_name, upsert_mode=True, dim=embedding_dim, **kwargs)
