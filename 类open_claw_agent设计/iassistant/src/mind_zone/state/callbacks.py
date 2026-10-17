#!/usr/bin/env python
# -*-coding: utf-8-*-

import os
from typing import Any

import dramatiq
import httpx


@dramatiq.actor(max_retries=2)
async def send_for_update(msg_dict: dict[str, Any], result: Any):
    async with httpx.AsyncClient(timeout=15, limits=httpx.Limits(max_keepalive_connections=100)) as client:
        await client.post(
            "http://{}:{}/task/-update".format(os.getenv("MZ_FASTAPI_HOST"), os.getenv("MZ_FASTAPI_PORT")),
            json={
                "task_id": msg_dict["message_id"],
                "task_type": msg_dict["actor_name"],
                "inputs": dict(zip(msg_dict["options"]["params"], msg_dict["args"])) | msg_dict["kwargs"],
                "result": result,
            },
        )


@dramatiq.actor(max_retries=2)
async def send_on_success(msg_dict: dict[str, Any], result: Any):
    async with httpx.AsyncClient(timeout=15, limits=httpx.Limits(max_keepalive_connections=100)) as client:
        await client.post(
            "http://{}:{}/task/-update".format(os.getenv("MZ_FASTAPI_HOST"), os.getenv("MZ_FASTAPI_PORT")),
            json={
                "task_id": msg_dict["message_id"],
                "task_type": msg_dict["actor_name"],
                "success": True,
                "inputs": dict(zip(msg_dict["options"]["params"], msg_dict["args"])) | msg_dict["kwargs"],
                "result": result,
            },
        )


@dramatiq.actor(max_retries=2)
async def send_on_failure(msg_dict: dict[str, Any], exc_data: dict[str, Any]):
    async with httpx.AsyncClient(timeout=15, limits=httpx.Limits(max_keepalive_connections=100)) as client:
        await client.post(
            "http://{}:{}/task/-update".format(os.getenv("MZ_FASTAPI_HOST"), os.getenv("MZ_FASTAPI_PORT")),
            json={
                "task_id": msg_dict["message_id"],
                "task_type": msg_dict["actor_name"],
                "success": False,
                "inputs": dict(zip(msg_dict["options"]["params"], msg_dict["args"])) | msg_dict["kwargs"],
                "exc_data": exc_data,
            },
        )
