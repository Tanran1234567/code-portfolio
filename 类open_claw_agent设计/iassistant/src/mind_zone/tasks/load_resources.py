#!/usr/bin/env python
# -*-coding: utf-8-*-

from pathlib import Path
from typing import Any, Optional, Union

import dramatiq
from llama_index.core import Document
from llama_index.core.schema import MediaResource

from mind_zone.api import RemoteAPI
from mind_zone.base import (
    BaseNodeParser,
    BaseReader,
    FileType,
    get_call_kwargs,
    get_init_kwargs,
    get_metadata,
    get_param_names,
    select_kwargs,
    validate_client,
)
from mind_zone.register import Registry
from mind_zone.utils import Logger, download_file


@dramatiq.actor(store_results=True, time_limit=3600000)
def load_from_file(
    *,
    file_type: FileType,
    file_path: Union[str, list[str]],
    method: str = "",
    client: Optional[dict[str, Any]] = None,
    file_id: Optional[str] = None,
    bundle_id: Optional[str] = None,
    **kwargs: Any,
) -> list[dict]:
    """
    Args:
        file_path: If a file is splitted into multiple parts, then provide paths of all parts as a list
    """
    if not method:
        method = f"read_file{file_type}.default"

    if method not in Registry.Component:
        Logger.warning(f"Method '{method}' not supported yet, use default instead")
        method = method.rsplit(".", 1)[0] + ".default"

    if method not in Registry.Component:
        raise RuntimeError(f"Unsupported type {file_type} for parsing")

    processor_cls = Registry.Component[method]
    params = get_param_names(processor_cls)
    if processor_cls.__name__ == "PDFReader":
        processor: BaseReader = processor_cls(return_full_document=True)
    elif validate_client(client):
        api_client = RemoteAPI(**get_init_kwargs(client))
        client = client or {}
        processor: BaseReader = processor_cls(api_client, **select_kwargs(client | kwargs, include_args=params.names))
    else:
        processor: BaseReader = processor_cls()

    file_path_list = [file_path] if isinstance(file_path, str) else file_path
    documents = []
    for part_path in file_path_list:
        if any(part_path.startswith(p) for p in ["http://", "https://", "ftp://"]):
            part_path = download_file(part_path)

        metadata = get_metadata(file_path=part_path)
        if validate_client(client):
            documents.extend(processor.load_data(Path(part_path), metadata, **get_call_kwargs(client)))
        else:
            documents.extend(processor.load_data(Path(part_path), metadata))

    is_tabular = file_type in [".csv", ".xls", ".xlsx"]
    for idx, doc in enumerate(documents):
        if file_id:
            doc.doc_id = file_id + f"_part_{idx}"
            doc.metadata["file_id"] = file_id
        if bundle_id:
            doc.metadata["bundle_id"] = bundle_id

        if is_tabular:
            # 不保存表格文件的解析结果
            doc.set_content("")
    return [d.to_dict() for d in documents]


@dramatiq.actor(store_results=True, time_limit=1800000)
async def split_document(
    *,
    texts: list[str],
    method: str = "",
    client: Optional[dict[str, Any]] = None,
    file_id: Optional[str] = None,
    bundle_id: Optional[str] = None,
    **kwargs: Any,
) -> list[dict]:
    if method not in Registry.Component:
        Logger.warning(f"Method '{method}' not supported yet, use default instead")
        method = method.rsplit(".", 1)[0] + ".default"

    processor_cls = Registry.Component[method]
    Logger.debug(f"Using {processor_cls.__name__} ({method}) to split document")

    params = get_param_names(processor_cls, method="from_defaults")
    if params.model:
        kwargs[params.model] = RemoteAPI(**get_init_kwargs(client))

    client = client or {}
    cls_kwargs = select_kwargs(client | kwargs, include_args=params.names)
    # TODO: sentence_splitter 自定义
    processor: BaseNodeParser = getattr(processor_cls, "from_defaults")(**cls_kwargs)
    chunk_nodes = await processor.acall(
        [Document(text_resource=MediaResource(text=x)) for x in texts], **get_call_kwargs(client)
    )

    chunk_list = []
    for idx, chunk in enumerate(chunk_nodes):
        metadata = chunk.metadata
        metadata["chunk_idx"] = idx
        if file_id:
            metadata["file_id"] = file_id
        if bundle_id:
            metadata["bundle_id"] = bundle_id
        chunk_list.append(
            Document(
                id_=chunk.id_,
                text_resource=MediaResource(text=chunk.get_content()),
                metadata=metadata,
                relationships=chunk.relationships,
            )
        )
    return [d.to_dict() for d in chunk_list]
