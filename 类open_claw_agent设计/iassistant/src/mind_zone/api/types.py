#!/usr/bin/env python
# -*-coding: utf-8-*-

from typing import Any

from llama_index.core.schema import BaseComponent

from mind_zone.types import BaseEnum

API_PARAM_NAMES = ["api_type", "provider", "base_url", "model_name", "model_type", "api_version"]
API_PARAM_NAMES_EX = API_PARAM_NAMES + ["method", "custom_template"]


class APIType(BaseEnum):
    OPENAI = "openai"
    HTTPx = "httpx"
    GRPC = "grpc"


class APIClient(BaseComponent):
    """接口客户端基类"""

    @classmethod
    def class_name(cls) -> str:
        return "APIClient"

    @classmethod
    def get_client_id(cls, *args: Any, **kwargs: Any):  # pyright: ignore
        raise NotImplementedError()

    @property
    def client_id(self):
        raise NotImplementedError()
