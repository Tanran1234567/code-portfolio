#!/usr/bin/env python
# -*-coding: utf-8-*-

from typing import Any, Optional, Union

import dramatiq
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores.types import MetadataFilters

from mind_zone.api import RemoteAPI
from mind_zone.base import (
    BasePostProcessor,
    BaseRetriever,
    BaseVectorStore,
    OpensearchStore,
    get_init_kwargs,
    get_param_names,
    select_kwargs,
)
from mind_zone.register import Registry
from mind_zone.tasks.utils import get_embed_model
from mind_zone.utils import Logger, cast_from_dict


@dramatiq.actor(store_results=True, max_retries=1)
async def search_documents(
    *,
    query: Union[str, dict[str, Any]],
    method: str,
    vstore: Optional[dict[str, Any]] = None,
    client: Optional[dict[str, Any]] = None,
    filters: Optional[list[dict[str, Any]]] = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Args:
        client: Embedding model client
    """
    if method not in Registry.Component:
        Logger.warning(f"Method '{method}' not supported yet, use default instead")
        method = method.rsplit(".", 1)[0] + ".default"

    processor_cls = Registry.Component[method]
    params = get_param_names(processor_cls)

    args = []
    if issubclass(processor_cls, VectorIndexRetriever):
        assert vstore, "Please provide a vector database settings"
        ds_type = vstore["ds_type"]
        store: BaseVectorStore = Registry.Component[ds_type](**vstore)
        args.append(VectorStoreIndex.from_vector_store(store, embed_model=get_embed_model(client=client)))
    elif params.model:
        kwargs[params.model] = RemoteAPI(**get_init_kwargs(client))

    if filters:
        kwargs["filters"] = MetadataFilters.from_dicts(filters)

    processor: BaseRetriever = processor_cls(*args, **select_kwargs(kwargs, include_args=params.names))
    sorted_chunks = await processor.aretrieve(query if isinstance(query, str) else QueryBundle.from_dict(query))

    if "store" in locals() and isinstance(store, OpensearchStore):
        await store.aclose()
    return [x.to_dict() for x in sorted_chunks]


@dramatiq.actor(store_results=True, max_retries=1)
async def postprocess_nodes(
    *,
    query: Union[str, dict[str, Any]],
    chunks: list[dict[str, Any]],
    method: str,
    client: Optional[dict[str, Any]] = None,
    **kwargs: Any,
):
    if method not in Registry.Component:
        Logger.warning(f"Method '{method}' not supported yet, use default instead")
        method = method.rsplit(".", 1)[0] + ".default"

    processor_cls = Registry.Component[method]
    params = get_param_names(processor_cls)
    if params.model:
        kwargs[params.model] = RemoteAPI(**get_init_kwargs(client))

    client = client or {}
    processor: BasePostProcessor = processor_cls(**select_kwargs(client | kwargs, include_args=params.names))
    sorted_chunks = await processor.apostprocess_nodes(
        [cast_from_dict(x, "NodeWithScore") for x in chunks],
        query_str=query if isinstance(query, str) else None,
        query_bundle=QueryBundle.from_dict(query) if isinstance(query, dict) else None,
    )
    return [x.to_dict() for x in sorted_chunks]
