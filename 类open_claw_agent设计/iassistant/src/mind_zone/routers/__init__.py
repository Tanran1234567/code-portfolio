#!/usr/bin/env python
# -*-coding: utf-8-*-

__author__ = ["super2nerd <super2nerd@icloud.com>"]

from mind_zone.routers.chunk import router as chunk_router
from mind_zone.routers.file import router as file_router
from mind_zone.routers.internal.task import router as task_router
from mind_zone.routers.query import router as query_router

__all__ = [
    "chunk_router",
    "file_router",
    "query_router",
    "task_router",
]
