#!/usr/bin/env python
# -*-coding: utf-8-*-

import inspect
import os
from datetime import datetime
from typing import Any, Optional, Union

from dramatiq import Actor, Message
from dramatiq_abort import abort
from pymongo.synchronous.database import Database

from mind_zone.state.db_store import MongoDBStore
from mind_zone.state.types import TaskState
from mind_zone.utils import Logger, truncate_repr

__FILE_DB_NAME__ = "raw_file_info"
__TASK_DB_NAME__ = "task_mgr_info"

__TASK_META_KEYS__ = ["on_success", "on_failure", "abortable", "parent_id", "purpose", "metadata"]


def init_state(db: MongoDBStore):
    svc_name = os.getenv("MZ_SVC_NAME", "mind_zone").replace("-", "_")
    env = os.getenv("MZ_ENV", "dev")
    db_name = svc_name if env == "prod" else f"{svc_name}_{env}"

    database = db.client.get_database(db_name)
    file_store = database.get_collection(__FILE_DB_NAME__)
    file_store.create_index("file_id", unique=True)
    task_store = database.get_collection(__TASK_DB_NAME__)
    task_store.create_index("task_id", unique=True)
    return db_name, __FILE_DB_NAME__, __TASK_DB_NAME__


def _send_actor(
    actor: Actor,
    args: tuple[Any],
    kwargs: dict[str, Any],
    on_success: Optional[Union[str, Actor]] = None,
    on_failure: Optional[Union[str, Actor]] = None,
) -> Message:
    params = list(inspect.signature(actor.fn).parameters.keys())
    message = actor.send_with_options(
        args=args, kwargs=kwargs, on_failure=on_failure, on_success=on_success, params=params
    )
    return message


def _launch_for_file(
    db: Database,
    actor: Actor,
    fn_args: tuple[Any],
    fn_kwargs: dict[str, Any],
    extra_info: dict[str, Any] = {},
) -> dict[str, Any]:
    arg_dict = dict(zip(list(inspect.signature(actor.fn).parameters.keys())[: len(fn_args)], fn_args)) | fn_kwargs
    if not (
        arg_dict.get("file_id")
        or (isinstance(arg_dict.get("result_wrapper"), dict) and arg_dict["result_wrapper"].get("file_id"))
    ):
        return {}

    file_store = db.get_collection(__FILE_DB_NAME__)
    file_id = arg_dict.get("file_id") or arg_dict["result_wrapper"].get("file_id")
    file_info: dict = file_store.find_one({"file_id": file_id}) or {}

    if actor.actor_name == "load_from_file":
        task_store = db.get_collection(__TASK_DB_NAME__)
        if (
            file_info
            and task_store.find_one({"task_id": file_info["task_dict"]["parse-document"]})
            and not arg_dict.get("reparse", False)
        ):
            # NOTE: 如果重复上传文件，取消当前的解析任务
            Logger.warning(f"File {file_id} has already been uploaded.")
            return {"returns": TaskState(id=None, status="Canceled")}

        metadata = arg_dict.get("metadata") or {"file_type": arg_dict["file_type"], "created_time": datetime.now()}
        file_info = {
            "file_id": file_id,
            "file_path": arg_dict["file_path"],
            "status": "not-ready",
            "metadata": metadata,
        }
        if arg_dict.get("bundle_id"):
            file_info["bundle_id"] = arg_dict["bundle_id"]

    message = _send_actor(
        actor,
        fn_args,
        fn_kwargs,
        on_success=extra_info.get("on_success", "send_on_success"),
        on_failure=extra_info.get("on_failure", "send_on_failure"),
    )
    if "task_dict" not in file_info:
        file_info["task_dict"] = {}
    purpose = extra_info.get("purpose", "default")
    if purpose in file_info["task_dict"]:
        Logger.error(
            "Task ({}) for {} already exists, overridden by new task ({}).".format(
                file_info["task_dict"][purpose], purpose, message.message_id
            )
        )
    file_info["task_dict"][purpose] = message.message_id

    attr_dict = {
        "metadata": arg_dict.get("metadata") or {},
        "other_info": arg_dict.get("other_info") or {},
    }
    for key, val in attr_dict.items():
        file_info[key] = file_info.get(key, {}) | val
    file_store.replace_one({"file_id": file_id}, file_info, upsert=True)
    return {"file_id": file_id, "message": message}


def launch_task(
    db: Database,
    actor: Actor,
    fn_args: tuple[Any],
    fn_kwargs: dict[str, Any],
    extra_info: dict[str, Any] = {},
) -> TaskState:
    """`kwargs` 是后台任务的启动参数"""
    output_dict = _launch_for_file(db, actor, fn_args, fn_kwargs, extra_info)
    if output_dict.get("returns"):
        return output_dict["returns"]

    message = output_dict.get("message") or _send_actor(
        actor,
        fn_args,
        fn_kwargs,
        on_success=extra_info.get("on_success", "send_on_success"),
        on_failure=extra_info.get("on_failure", "send_on_failure"),
    )
    task_info = {
        "task_id": message.message_id,
        "task_type": actor.actor_name,
        "created_time": message.message_datetime,
        "abortable": extra_info.get("abortable", False),
        "progress": 0.0,
    }
    if output_dict.get("file_id"):
        task_info["src_file"] = output_dict["file_id"]

    task_store = db.get_collection(__TASK_DB_NAME__)
    parent_id = extra_info.get("parent_id")
    if parent_id:
        task_info["parent"] = parent_id
        temp_info: dict = task_store.find_one({"task_id": parent_id})
        if temp_info:
            if "children" not in temp_info:
                temp_info["children"] = {}
            temp_info["children"][message.message_id] = {
                "purpose": extra_info.get("purpose", "default"),
                "status": "Running",
            }
            task_store.update_one({"task_id": parent_id}, {"$set": temp_info})

    metadata = extra_info.get("metadata") or {}
    arg_dict = dict(zip(list(inspect.signature(actor.fn).parameters.keys())[: len(fn_args)], fn_args)) | fn_kwargs
    metadata.update({x: arg_dict[x] for x in ["job_id", "source_id"] if arg_dict.get(x) is not None})

    task_info["metadata"] = metadata
    task_info["other_info"] = {k: v for k, v in extra_info.items() if k not in __TASK_META_KEYS__}
    task_store.insert_one(task_info)
    return TaskState(id=message.message_id, status="Running")


def update_task(db: Database, task_id: str, result: Any) -> TaskState:  # pyright: ignore
    task_store = db.get_collection(__TASK_DB_NAME__)
    task_info: dict = task_store.find_one({"task_id": task_id})
    if not task_info:
        Logger.warning(f"Task {task_id} not found in {__TASK_DB_NAME__} database.")
        return TaskState(id=task_id, status="Failed")
    # TODO: 更新进度
    Logger.debug("Update task with info: {}".format(truncate_repr(task_info)))
    task_store.update_one({"task_id": task_id}, {"$set": task_info})
    return TaskState(id=task_id, status="Running")


def _finish_for_file(db: Database, task_info: dict[str, Any], result: Any):
    file_store = db.get_collection(__FILE_DB_NAME__)
    if task_info["task_type"] == "load_from_file" and task_info.get("src_file"):
        file_store.update_one({"file_id": task_info["src_file"]}, {"$set": {"status": "ready", "outputs": result}})

    if task_info["task_type"] == "store_documents":
        # TODO: 保存向量数据库index
        pass

    if isinstance(result, dict) and result.get("file_id"):
        file_id = result.pop("file_id")
        other_info: dict = file_store.find_one({"file_id": file_id}).get("other_info", {})
        if set(other_info.keys()) & set(result.keys()):
            keys_str = ",".join([str(x) for x in set(other_info.keys()) & set(result.keys())])
            Logger.warning("Duplicate keys {} found for file {}".format(keys_str, file_id))

        other_info.update(result)
        file_store.update_one({"file_id": file_id}, {"$set": {"other_info": other_info}})
    return


def finish_task(db: Database, task_id: str, result: Any) -> TaskState:
    task_store = db.get_collection(__TASK_DB_NAME__)
    task_info: dict = task_store.find_one({"task_id": task_id})
    if not task_info:
        Logger.warning(f"Task {task_id} not found in {__TASK_DB_NAME__} database.")
        return TaskState(id=task_id, status="Success")

    _finish_for_file(db, task_info, result)
    task_store.update_one({"task_id": task_id}, {"$set": {"progress": 1.0, "outputs": result}})
    Logger.debug("Update task with info: {}".format(truncate_repr(task_info)))

    parent_id = task_info.get("parent")
    if parent_id:
        temp_info: dict = task_store.find_one({"task_id": parent_id})
        if temp_info and temp_info.get("children"):
            if temp_info["task_type"] in ["custom_process_video"]:
                temp_info["children"].pop(task_id, None)
            else:
                temp_info["children"][task_id]["status"] = "Success"
            task_store.update_one({"task_id": parent_id}, {"$set": {"children": temp_info["children"]}})

    return TaskState(id=task_id, status="Success")


def _cancel_for_file(db: Database, task_info: dict[str, Any]):
    if task_info["task_type"] == "load_from_file" and task_info.get("src_file"):
        file_store = db.get_collection(__FILE_DB_NAME__)
        file_store.delete_one({"file_id": task_info["src_file"]})
    return


def cancel_task(
    db: Database,
    task_id: str,
    exc_data: Optional[dict[str, Any]] = None,
    abortable: bool = False,
) -> TaskState:
    if abortable:
        abort(task_id, abort_timeout=10000)
        Logger.info(f"Abort running task {task_id}.")

    task_store = db.get_collection(__TASK_DB_NAME__)
    task_info: dict = task_store.find_one({"task_id": task_id})
    if not task_info:
        Logger.warning(f"Task {task_id} not found {__TASK_DB_NAME__} database.")
        return TaskState(id=task_id, status="Canceled")

    if not abortable and task_info.get("abortable"):
        abort(task_id, abort_timeout=10000)
        Logger.info(f"Abort running task {task_id}.")

    _cancel_for_file(db, task_info)
    parent_id = task_info.get("parent")
    if parent_id:
        temp_info: dict = task_store.find_one({"task_id": parent_id})
        if temp_info and temp_info.get("children"):
            temp_info["children"].pop(task_id, None)
            task_store.update_one({"task_id": parent_id}, {"$set": temp_info})

    if os.getenv("MZ_ENV", "dev") == "dev":
        task_store.update_one({"task_id": task_id}, {"$set": {"exception": exc_data}})
    else:
        task_store.delete_one({"task_id": task_id})
    return TaskState(id=task_id, status="Canceled")
