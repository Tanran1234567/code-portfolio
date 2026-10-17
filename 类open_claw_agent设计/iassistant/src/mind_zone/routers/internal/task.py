#!/usr/bin/env python
# -*-coding: utf-8-*-

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from mind_zone.routers.callbacks import report_status
from mind_zone.schemas import FormatOutput
from mind_zone.state import TaskManager

router = APIRouter(prefix="/task")


@router.post("/-update")
async def _update_task(request: Request):  # pyright: ignore
    req_dict = await request.json()
    task_mgr: TaskManager = request.app.TASK_MANAGER

    task_id = req_dict["task_id"]
    task_type = req_dict["task_type"]
    progress = task_mgr.get_progress(task_id)
    if req_dict.get("success") is None:
        task_mgr.update_task(task_id, req_dict["result"])
    elif req_dict["success"]:
        task_mgr.finish_task(task_id, req_dict["result"])
        report_status(req_dict["inputs"], task_type, req_dict["success"], progress)
    else:
        task_mgr.cancel_task(task_id, req_dict["exc_data"])
        report_status(req_dict["inputs"], task_type, req_dict["success"], progress)
    return


@router.get("/status")
async def _get_task_status(request: Request):  # pyright: ignore
    req_dict = request.query_params
    task_mgr: TaskManager = request.app.TASK_MANAGER

    task_id = req_dict["task_id"]
    progress = task_mgr.get_progress(task_id)
    data_dict = {"task_id": task_id, "progress": progress, "status": 0 if progress >= 0 else -1}
    if progress >= 0.99:
        task_result = await task_mgr.wait_for_done(task_id, timeout=3, return_primitive=True)
        if task_result.success:
            data_dict["result"] = task_result.result
        data_dict["status"] = 1 if task_result.success else -1

    return JSONResponse(content=FormatOutput("no-error", data=data_dict))
