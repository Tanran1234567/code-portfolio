#!/usr/bin/env python
# -*-coding: utf-8-*-

from typing import Optional, Union

from pydantic import BaseModel, Field


class DocumentIndexRequest(BaseModel):
    app_name: str = Field(default="", description="应用名称")
    file_id: str = Field(default="", description="文件ID")
    file_type: str = Field(default=".unk", description="文件类型，小写后缀名，如.pdf、.md、.txt等")
    file_path: Union[str, list[str]] = Field(
        default="", description="文件路径，文件系统路径或下载路径，文件有多个片段时，可以传每个片段的路径列表"
    )
    bundle_id: str = Field(default="", description="文件包ID，可以是知识库ID，或其它文件集合ID")
    reparse: bool = Field(default=False, description="是否重新解析")
    meta_fields: Optional[list[str]] = Field(
        default=["title", "keyword", "summary", "question"],
        description=(
            "需要提取元数据字段：title、keyword、summary、question代表片段的标题、关键词、摘要、能回答的问题等"
        ),
    )
    status_url: str = Field(default="", description="状态回调接口，处理完成后发送回调请求")


class DocumentParseRequest(BaseModel):
    app_name: str = Field(description="应用名称")
    file_id: str = Field(description="文件ID")
    file_type: str = Field(description="文件类型，小写后缀名，如.pdf、.md、.txt等")
    file_path: Union[str, list[str]] = Field(
        description="文件路径，文件系统路径或下载路径，文件有多个片段时，可以传每个片段的路径列表"
    )
    bundle_id: Optional[str] = Field(default=None, description="文件包ID，可以是知识库ID，或其它文件集合ID")
    return_type: Optional[Union[str, list[str]]] = Field(
        default=None,
        deprecated="1.0.0版本之后不可用",
        description=(
            "需要返回的内容，"
            "chunks代表分段结果；"
            "title、keyword、summary、question代表片段的元数据，标题、关键词、摘要、能回答的问题等；"
            "可传入chunks或者包含title、keyword、summary、question任意组合的列表"
        ),
    )
    reparse: bool = Field(default=False, description="是否重新解析")
    status_url: Optional[str] = Field(default=None, description="状态回调接口，处理完成后发送回调请求")


class DocumentIndexRequestV1(BaseModel):
    app_name: str = Field(description="应用名称")
    file_id: str = Field(description="文件ID")
    file_type: str = Field(description="文件类型，小写后缀名，如.pdf、.md、.txt等")
    file_path: Union[str, list[str]] = Field(
        description="文件路径，文件系统路径或下载路径，文件有多个片段时，可以传每个片段的路径列表"
    )
    bundle_id: Optional[str] = Field(default=None, description="文件包ID，可以是知识库ID，或其它文件集合ID")
    status_url: Optional[str] = Field(default=None, description="状态回调接口，处理完成后发送回调请求")


class DocumentReindexRequest(BaseModel):
    app_name: str = Field(description="应用名称")
    file_id: str = Field(description="文件ID")
    status_url: Optional[str] = Field(default=None, description="状态回调接口，处理完成后发送回调请求")


class DocumentDeleteRequest(BaseModel):
    app_name: str = Field(default=None, description="应用名称")
    file_id: Optional[str] = Field(default=None, description="文件ID")
    bundle_id: Optional[str] = Field(default=None, description="文件包ID，可以是知识库ID，或其它文件集合ID")


class DocumentViewRequest(BaseModel):
    app_name: str = Field(description="应用名称")
    file_id: str = Field(description="文件ID")
    chunk_idx: Optional[int] = Field(default=None, description="文档分块序号")


class QueryAnswerRequest(BaseModel):
    app_name: str = Field(description="应用名称")
    text: str = Field(description="查询语句")
    user_id: Optional[str] = Field(default=None, description="用户ID")
    history: Optional[list[dict[str, str]]] = Field(default=None, description="历史会话的内容")
    thinking: bool = Field(default=False, description="是否启用深度思考")
    file_id: Optional[str] = Field(default=None, description="文件ID")
    bundle_id: Optional[str] = Field(default=None, description="文件包ID，可以是知识库ID，或其它文件集合ID")


class QueryRespondRequest(BaseModel):
    app_name: str = Field(description="应用名称")
    text: str = Field(description="查询语句")
    history: Optional[list[dict[str, str]]] = Field(default=None, description="历史会话的内容")
    thinking: bool = Field(default=False, description="是否启用深度思考")
    file_id: Optional[str] = Field(default=None, description="文件ID")
    bundle_id: Optional[str] = Field(default=None, description="文件包ID，可以是知识库ID，或其它文件集合ID")
