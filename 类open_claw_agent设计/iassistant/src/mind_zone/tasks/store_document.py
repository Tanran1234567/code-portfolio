#!/usr/bin/env python
# -*-coding: utf-8-*-

from typing import Any, Union

import dramatiq
import numpy as np
from llama_index.core.schema import BaseNode, Document
from llama_index.core.vector_stores.types import MetadataFilters

from mind_zone.base import BaseVectorStore, OpensearchStore
from mind_zone.register import Registry
from mind_zone.state import get_task_manager
from mind_zone.tasks.utils import get_embed_model
from mind_zone.utils import Logger


@dramatiq.actor(store_results=True, max_retries=1)
async def get_embeddings(
    *,
    texts: Union[str, list[str]],
    method: str,
    client: dict[str, Any],
    **kwargs: Any,
) -> Union[list[float], list[list[float]]]:
    processor = get_embed_model(method=method, client=client, **kwargs)
    if isinstance(texts, str):
        embeddings = await processor.aget_query_embedding(texts)
    else:
        embeddings = await processor.aget_text_embedding_batch(texts)
    return np.array(embeddings).tolist()


async def store_doc_nodes(
    *,
    file_id: str,
    node_list: list[Document],
    ds_type: str,
    ds_config: dict[str, Any],
    **kwargs: Any,  # pyright: ignore
):
    if ds_type not in Registry.Component:
        Logger.warning(f"Data store '{ds_type}' not supported yet, use opensearch instead")
        ds_type = ds_type.rsplit(".", 1)[0] + ".opensearch"

    for doc in node_list:
        # 将relationships字段移动到metadata字段，因为OpensearchVecClient不支持relationships字段
        doc.metadata["relationships"] = doc.relationships
    if ds_config.get("embedding_dim"):
        assert node_list[0].embedding is not None or ds_config["embedding_dim"] == len(node_list[0].embedding)

    store: BaseVectorStore = Registry.Component[ds_type](**ds_config)
    filters = [{"key": "file_id", "value": file_id}]
    await store.adelete_nodes(filters=MetadataFilters.from_dicts(filters), **kwargs)

    node_ids = await store.async_add(node_list)
    if isinstance(store, OpensearchStore):
        await store.aclose()
    return node_ids


@dramatiq.actor(store_results=True)
async def store_documents(
    *,
    file_id: str,
    split_task: str,
    metadata_tasks: list[str] = [],
    embedding_tasks: list[str] = [],
    ds_type: str,
    ds_config: dict[str, Any],
    **kwargs: Any,  # pyright: ignore
):
    task_mgr = get_task_manager()
    split_result = await task_mgr.wait_for_done(split_task, timeout=60)
    if not split_result.success:
        raise RuntimeError(f"Unable to split chunks for document {file_id}")

    extra_data: dict[str, dict[str, Any]] = {}
    for task_id in metadata_tasks:
        task_result = await task_mgr.wait_for_done(task_id, return_fields=["outputs", "other_info"], timeout=600)
        if not task_result.success:
            raise RuntimeError(f"Unable to extract metadata for document {file_id}")

        for doc_id, metadata in zip(task_result.result["other_info"]["doc_ids"], task_result.result["outputs"]):
            if doc_id not in extra_data:
                extra_data[doc_id] = {}
            extra_data[doc_id].update(metadata)

    for task_id in embedding_tasks:
        task_result = await task_mgr.wait_for_done(task_id, return_fields=["outputs", "other_info"], timeout=600)
        if not task_result.success:
            raise RuntimeError(f"Unable to compute embeddings for document {file_id}")

        for doc_id, embeddings in zip(task_result.result["other_info"]["doc_ids"], task_result.result["outputs"]):
            if doc_id not in extra_data:
                extra_data[doc_id] = {}
            extra_data[doc_id].update({"embedding": embeddings})

    file_result = await task_mgr.get_document(file_id)
    if not file_result.success:
        raise RuntimeError(f"CANNOT find contents for file {file_id}")

    assert len(split_result.result) == len(extra_data)
    doc_metadata = file_result.result[0].metadata or {}

    chunk_list: list[Document] = []
    for chunk in split_result.result:
        chunk: Document
        assert chunk.doc_id in extra_data

        chunk.metadata.update(doc_metadata)
        chunk.metadata.update({k: v for k, v in extra_data[chunk.doc_id].items() if k != "embedding"})
        chunk.embedding = extra_data[chunk.doc_id].get("embedding")
        chunk_list.append(chunk)
    node_ids = await store_doc_nodes(
        file_id=file_id, node_list=chunk_list, ds_type=ds_type, ds_config=ds_config, **kwargs
    )
    return node_ids


@dramatiq.actor(store_results=False, max_retries=1)
async def delete_documents(
    *,
    ds_type: str,
    ds_config: dict[str, Any],
    filters: dict[str, Any],
    **kwargs: Any,  # pyright: ignore
):
    if ds_type not in Registry.Component:
        Logger.warning(f"Data store '{ds_type}' not supported yet, use opensearch instead")
        ds_type = ds_type.rsplit(".", 1)[0] + ".opensearch"

    store: BaseVectorStore = Registry.Component[ds_type](**ds_config)
    await store.adelete_nodes(filters=MetadataFilters.from_dicts(filters), **kwargs)
    if isinstance(store, OpensearchStore):
        await store.aclose()
    return


async def fetch_documents(
    *,
    ds_type: str,
    ds_config: dict[str, Any],
    filters: dict[str, Any],
    **kwargs: Any,  # pyright: ignore
) -> list[BaseNode]:
    if ds_type not in Registry.Component:
        Logger.warning(f"Data store '{ds_type}' not supported yet, use opensearch instead")
        ds_type = ds_type.rsplit(".", 1)[0] + ".opensearch"

    store: BaseVectorStore = Registry.Component[ds_type](**ds_config)
    node_list = await store.aget_nodes(filters=MetadataFilters.from_dicts(filters), **kwargs)
    if isinstance(store, OpensearchStore):
        await store.aclose()
    return node_list
