#!/usr/bin/env python
# -*-coding: utf-8-*-

import asyncio
import threading
import time
from typing import Any, Optional, Sequence, Union

from dramatiq.actor import Actor
from llama_index.core import Document

from mind_zone.state.db_store import MongoDBStore
from mind_zone.state.handlers import cancel_task, finish_task, init_state, launch_task, update_task
from mind_zone.state.types import FileResult, TaskResult, TaskState
from mind_zone.utils import Logger, cast_from_dict, truncate_repr


class TaskManager:
    """后台任务管理器"""

    def __init__(
        self,
        host: Optional[Union[str, Sequence[str]]] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self._state = MongoDBStore(host=host, port=port, username=username, password=password)
        db_name, file_db_name, task_db_name = init_state(self._state)
        self._state_db = self._state.client.get_database(db_name)
        self._file_db_name = file_db_name
        self._task_db_name = task_db_name

    def close(self):
        self._state.close()

    @property
    def state_db(self):
        return self._state_db

    @property
    def file_client(self):
        return self.state_db.get_collection(self._file_db_name)

    @property
    def task_client(self):
        return self.state_db.get_collection(self._task_db_name)

    def launch_task(self, task_actor: Actor, *args: Any, **kwargs: Any) -> TaskState:
        extra_info: dict[str, Any] = kwargs.pop("extra_info", {})
        Logger.debug(
            {"task": task_actor.actor_name, "args": args, "kwargs": truncate_repr(kwargs), "extra_info": extra_info}
        )
        task_state = launch_task(self._state_db, task_actor, args, kwargs, extra_info)
        return task_state

    def update_task(self, task_id: str, result: Any):
        update_task(self._state_db, task_id, result)
        return

    def finish_task(self, task_id: str, result: Any):
        finish_task(self._state_db, task_id, result)
        return

    def cancel_task(self, task_id: str, exc_data: Optional[dict[str, Any]] = None, abortable: bool = False):
        cancel_task(self._state_db, task_id, exc_data, abortable)
        return

    def search_task(
        self,
        file_id: Optional[str] = None,
        child_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        purpose: str = "parse-document",
        filters: Optional[dict[str, Any]] = None,
    ) -> Optional[Union[str, dict[str, str], list[str]]]:
        """通过file_id找对应task，通过child_id找父任务，通过parent_id找子任务"""
        assert (
            len([x for x in (file_id, child_id, parent_id, filters) if x]) == 1
        ), "Provide only one query from `file_id`, `child_id`, `parent_id` or `filters`"
        if file_id:
            file_info: dict = self.file_client.find_one({"file_id": file_id})
            if file_info:
                return file_info.get("task_dict", {}).get(purpose)
            return None

        if child_id:
            task_info: dict = self.task_client.find_one({"task_id": child_id})
            if task_info:
                return task_info.get("parent")
            return None

        if parent_id:
            task_info: dict = self.task_client.find_one({"task_id": parent_id})
            if task_info:
                if purpose:
                    return {k: v for k, v in task_info.get("children", {}).items() if v["purpose"] == purpose}
                return task_info.get("children", {})
            return None

        if filters:
            task_list = []
            for info in self.task_client.find(filters):
                task_list.append(info["task_id"])
            return task_list

        return None

    def search_files(self, bundle_id: str) -> list[str]:
        return [d["file_id"] for d in self.file_client.find({"bundle_id": bundle_id})]

    def delete_files(self, file_id: Optional[str] = None, bundle_id: Optional[str] = None) -> int:
        if file_id:
            return self.file_client.delete_one({"file_id": file_id}).deleted_count
        if bundle_id:
            return self.file_client.delete_many({"bundle_id": bundle_id}).deleted_count
        return 0

    def get_progress(self, task_id: str) -> float:
        outputs = self.task_client.find_one({"task_id": task_id})
        if not outputs:
            return -1
        return outputs["progress"]

    async def get_document(
        self,
        file_id: str,
        timeout: float = -1,
        interval: float = 0.5,
        return_fields: Optional[list[str]] = None,
    ) -> FileResult:
        """timeout < 0阻塞等待"""
        tick_ts = time.time()
        while True:
            file_info = self.file_client.find_one({"file_id": file_id})
            if not file_info:
                return FileResult(id=None, success=False, result=None)

            if not return_fields:
                if file_info["status"] == "ready":
                    outputs = [Document.from_dict(x) for x in file_info["outputs"]]
                    return FileResult(id=file_id, success=True, result=outputs)
            else:
                return FileResult(
                    id=file_id, success=True, result={k: v for k, v in file_info.items() if k in return_fields}
                )

            if timeout >= 0 and time.time() - tick_ts >= timeout:
                return FileResult(id=file_id, success=False, result=None)
            # NOTE: 不能使用time.sleep，会阻塞主进程
            await asyncio.sleep(interval)

    async def wait_for_done(
        self,
        task_id: str,
        timeout: float = -1,
        interval: float = 0.5,
        return_fields: Optional[list[str]] = None,
        return_primitive: bool = False,
    ) -> TaskResult:
        """
        Args:
            timeout < 0阻塞等待
        Returns:
            success: bool
                - True 代表任务已完成
                - False 代表不存在，或者未完成
        """
        tick_ts = time.time()
        while True:
            task_info = self.task_client.find_one({"task_id": task_id})
            if not task_info:
                return TaskResult(id=None, success=False, result=None)

            if task_info["progress"] >= 0.99:
                if not return_fields:
                    outputs = task_info["outputs"]
                    if not return_primitive:
                        if task_info["task_type"] in ["load_from_file", "split_document"]:
                            outputs = [cast_from_dict(x, "Document") for x in outputs]
                        elif task_info["task_type"] in ["search_documents", "postprocess_nodes"]:
                            outputs = [cast_from_dict(x, "NodeWithScore") for x in outputs]
                    return TaskResult(id=task_id, success=True, result=outputs)

                return TaskResult(
                    id=task_id, success=True, result={k: v for k, v in task_info.items() if k in return_fields}
                )

            if timeout >= 0 and time.time() - tick_ts >= timeout:
                if not return_fields:
                    return TaskResult(id=task_id, success=False, result=task_info)

                return TaskResult(
                    id=task_id, success=False, result={k: v for k, v in task_info.items() if k in return_fields}
                )
            # NOTE: 不能使用time.sleep，会阻塞主进程
            await asyncio.sleep(interval)

    async def complete_task(self, task_actor: Actor, *args: Any, **kwargs: Any) -> TaskResult:
        wait_kwargs: dict[str, Any] = kwargs.pop("wait_kwargs", {})
        task_state = self.launch_task(task_actor, *args, **kwargs)
        task_result = await self.wait_for_done(
            task_state.id,
            timeout=wait_kwargs.get("timeout", -1),
            interval=wait_kwargs.get("interval", 0.5),
            return_fields=wait_kwargs.get("return_fields"),
            return_primitive=wait_kwargs.get("return_primitive", False),
        )
        return task_result


__TASK_MGR__ = None
__INIT_LOCK__ = threading.Lock()


def get_task_manager(**kwargs: Any):
    global __TASK_MGR__
    if __TASK_MGR__ is None:
        with __INIT_LOCK__:
            if __TASK_MGR__ is None:
                if not kwargs:
                    Logger.error("Configurations CANNOT be empty when initializing TaskManager")
                    return None

                __TASK_MGR__ = TaskManager(**kwargs)
    return __TASK_MGR__
