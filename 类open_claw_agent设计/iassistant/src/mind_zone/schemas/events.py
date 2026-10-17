#!/usr/bin/env python
# -*-coding: utf-8-*-

from collections import namedtuple
from typing import Any, Optional, Union

from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.schema import QueryType
from llama_index.core.workflow import Event
from pydantic.main import BaseModel

QueryIntent = namedtuple("QueryIntent", ["l1", "l2"])


class NaiveRequestEvent(Event):
    router: str
    request: BaseModel
    request_type: str


class ParseRequestEvent(Event):
    extra_info: dict[str, Any] = {}
    wait_kwargs: dict[str, Any] = {}


class SplitRequestEvent(Event):
    task_id: Optional[str] = None
    extra_info: dict[str, Any] = {}
    wait_kwargs: dict[str, Any] = {}


class ExtractRequestEvent(Event):
    task_id: Optional[str] = None
    extra_info: dict[str, Any] = {}
    wait_kwargs: dict[str, Any] = {}


class EmbedRequestEvent(Event):
    task_id: Optional[str] = None
    extra_info: dict[str, Any] = {}
    wait_kwargs: dict[str, Any] = {}


class StoreRequestEvent(Event):
    task_ids: Optional[list[str]] = None
    task_type: Optional[str] = None
    extra_info: dict[str, Any] = {}
    wait_kwargs: dict[str, Any] = {}


class QueryRequestEvent(Event):
    intent: QueryIntent
    query: QueryType
    context: Optional[list[Any]] = None
    template: Optional[PromptTemplate] = None
    stream: bool = False


class ProgressEvent(Event):
    ckpt_id: str
    content: Union[list[Any], dict[str, Any]]
    content_type: Optional[str] = None


class ResponseEvent(Event):
    response: RESPONSE_TYPE


class MultiResponseEvent(Event):
    responses: list[RESPONSE_TYPE]


class DirectEvent(Event):
    query: QueryType
    context: Optional[list[str]] = None
    template: Optional[PromptTemplate] = None


class ChatBIEvent(Event):
    query: QueryType
    context: Optional[list[Any]] = None
    user_id: Optional[str] = None


class RAGQueryEvent(Event):
    query: QueryType
    context: Optional[list[Any]] = None
    user_id: Optional[str] = None
