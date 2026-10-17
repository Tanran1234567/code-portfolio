#!/usr/bin/env python
# -*-coding: utf-8-*-

import threading
from typing import Optional, Sequence, Union

from pymongo import MongoClient


class MongoDBStore:
    _instance = None
    _lock = threading.Lock()

    def __new__(
        cls,
        host: Optional[Union[str, Sequence[str]]] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        # Double-Checked Locking
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._client = MongoClient(host=host, port=port, username=username, password=password)
        return cls._instance

    @property
    def client(self) -> MongoClient:
        return self._client

    def close(self):
        if hasattr(self, "_client"):
            self._client.close()
