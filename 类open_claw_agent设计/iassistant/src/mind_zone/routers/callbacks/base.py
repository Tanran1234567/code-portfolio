#!/usr/bin/env python
# -*-coding: utf-8-*-

from typing import Any

import httpx


def report_status(req_data: dict[str, Any], task_type: str, success: bool, progress: float):
    """返回任务的执行状态"""
    if not req_data.get("status_url"):
        return

    file_or_bundle_id = req_data.get("file_id")
    if not file_or_bundle_id and isinstance(req_data.get("result_wrapper"), dict):
        file_or_bundle_id = req_data["result_wrapper"].get("file_id")
    if not file_or_bundle_id:
        file_or_bundle_id = req_data.get("bundle_id")
    if not file_or_bundle_id and isinstance(req_data.get("result_wrapper"), dict):
        file_or_bundle_id = req_data["result_wrapper"].get("bundle_id")

    if file_or_bundle_id:
        if task_type == "load_from_file":
            data_dict = {"taskId": file_or_bundle_id, "status": "success" if success else "failed"}
        elif task_type == "store_documents":
            data_dict = {"id": file_or_bundle_id, "status": 2 if success else 3}
        with httpx.Client() as client:
            client.get(req_data["status_url"], params=data_dict)
    return


def submit_results():
    pass
