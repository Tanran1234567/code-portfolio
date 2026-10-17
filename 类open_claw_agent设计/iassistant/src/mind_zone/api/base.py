#!/usr/bin/env python
# -*-coding: utf-8-*-

import os
import threading
from collections import OrderedDict
from typing import Any, Optional

from mind_zone.api.grpc import GRPCClient
from mind_zone.api.httpx import HTTPxClient
from mind_zone.api.openai import OpenAIChatClient, OpenAIClient, OpenAIEmbeddingClient
from mind_zone.api.types import APIClient, APIType


class __InnerManager__:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        # Double-Checked Locking
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._api_dict = OrderedDict()
        return cls._instance

    @classmethod
    def _infer_type(cls, **kwargs) -> APIType:
        if all(x in kwargs for x in ["api_key", "base_url", "model_name"]):
            return APIType.OPENAI

        if all(x in kwargs for x in ["provider", "base_url", "model_name"]):
            return APIType.OPENAI

        return APIType.HTTPx

    def _add_client(self, api_type: Optional[APIType] = None, **kwargs: Any):
        api_type = api_type or self._infer_type(**kwargs)

        if api_type == APIType.OPENAI:
            model_type = kwargs.get("model_type")
            match model_type:
                case "embedding":
                    client = OpenAIEmbeddingClient.from_dict(**kwargs)
                case _:
                    if model_type == "chat":
                        kwargs["is_chat_model"] = True
                    elif model_type == "function":
                        kwargs["is_function_calling_model"] = True
                    client = OpenAIChatClient.from_dict(**kwargs)
        elif api_type == APIType.GRPC:
            client = GRPCClient.from_dict(kwargs)
        else:
            client = HTTPxClient.from_dict(kwargs)

        if client.client_id not in self._api_dict:
            self._api_dict[client.client_id] = client

    def __call__(self, **kwargs: Any) -> APIClient:
        api_type: APIType = kwargs.get("api_type") or self._infer_type(**kwargs)

        if kwargs.get("provider") and kwargs["provider"] != "local":
            kwargs["api_key"] = os.getenv(f"{kwargs.pop('provider').upper()}_API_AUTH_KEY")

        if api_type == APIType.OPENAI:
            client_id = OpenAIClient.get_client_id(**kwargs)
        elif api_type == APIType.GRPC:
            client_id = GRPCClient.get_client_id(**kwargs)
        else:
            api_type = APIType.HTTPx
            client_id = HTTPxClient.get_client_id(**kwargs)

        if client_id not in self._api_dict:
            self._add_client(api_type=api_type, **kwargs)
        return self._api_dict[client_id]

    def close(self):
        for client in self._api_dict.values():
            if hasattr(client, "close"):
                getattr(client, "close")()

    async def aclose(self):
        for client in self._api_dict.values():
            if hasattr(client, "aclose"):
                await getattr(client, "aclose")()


RemoteAPI = __InnerManager__()
