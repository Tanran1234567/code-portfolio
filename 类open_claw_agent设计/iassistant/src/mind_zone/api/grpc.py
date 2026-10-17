#!/usr/bin/env python
# -*-coding: utf-8-*-

import threading
from collections import OrderedDict
from hashlib import md5
from typing import Any, Optional

from grpc.aio import Channel, insecure_channel
from llama_index.core.bridge.pydantic import Field, PrivateAttr

from mind_zone.api.types import APIClient


class GRPCClient(APIClient):
    base_url: str = Field(description="The endpoint of service to use.")

    _client_id: Optional[str] = PrivateAttr()
    _options: Optional[list[tuple[str, Any]]] = PrivateAttr()
    _chan_dict: dict[str, Any] = PrivateAttr()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._client_id = None
        self._options = kwargs.get("options")
        self._chan_dict = OrderedDict()

        self._pool_size = kwargs.get("pool_size", 8)
        self._lock = threading.Lock()

    @classmethod
    def class_name(cls) -> str:
        return "GRPCClient"

    @property
    def client_id(self):
        if self._client_id is None:
            self._client_id = self.get_client_id(self.base_url)
        return self._client_id

    @classmethod
    def get_client_id(cls, base_url: str = "", **kwargs: Any):  # pyright: ignore
        hash_str = "grpc_url={}".format(base_url)
        return md5(hash_str.encode("utf-8")).hexdigest()

    def acquire(self) -> Channel:
        with self._lock:
            if not self._chan_dict:
                for _ in range(self._pool_size):
                    chan = insecure_channel(self.base_url, options=self._options)
                    self._chan_dict[hex(id(chan))] = {"object": chan, "count": 0}

            chan_id = min(self._chan_dict.keys(), key=lambda x: self._chan_dict[x]["count"])
            self._chan_dict[chan_id]["count"] += 1
            return self._chan_dict[chan_id]["object"]

    def release(self, channel: Channel):
        with self._lock:
            chan_id = hex(id(channel))
            if chan_id in self._chan_dict:
                self._chan_dict[chan_id]["count"] = max(0, self._chan_dict[chan_id]["count"] - 1)

    async def aclose(self):
        with self._lock:
            for chan_id in self._chan_dict.keys():
                channel: Channel = self._chan_dict[chan_id]["object"]
                await channel.close()

            self._chan_dict = {}
