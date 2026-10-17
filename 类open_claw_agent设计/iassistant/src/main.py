#!/usr/bin/env python
# -*-coding: utf-8-*-

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Awaitable, Callable

WORK_DIR = Path(__file__).parent.parent.absolute()
PY_VERSION = next(WORK_DIR.joinpath(".venv/lib").iterdir()).name
sys.path.insert(0, str(WORK_DIR / ".venv" / "lib" / PY_VERSION / "site-packages"))

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse, Response

load_dotenv(
    dotenv_path=WORK_DIR / "config" / ".env.{}".format(os.getenv("MZ_ENV", "dev")),
    override=True,
    verbose=True,
)
os.environ["MZ_SVC_NAME"] = "iassistant"
os.environ["MZ_CACHE_DIR"] = str(WORK_DIR / ".cache")
Path(os.getenv("MZ_CACHE_DIR")).mkdir(parents=True, exist_ok=True)

from mind_zone import (
    Config,
    FormatOutput,
    Logger,
    Orchestrator,
    RemoteAPI,
    TaskManager,
    chunk_router,
    create_nacos_client,
    deregister_service,
    file_router,
    query_router,
    register_service,
    task_router,
)
from mind_zone.monitoring import configure_monitoring


@asynccontextmanager
async def lifespan(app: FastAPI):
    nacos_client = await create_nacos_client(WORK_DIR / "config" / "config.yaml")
    ephemeral = os.getenv("MZ_ENV", "dev") != "dev"
    svc_host, svc_port = await register_service(nacos_client, ephemeral=ephemeral)

    app.CONFIG = Config.from_yaml(WORK_DIR / "config" / "config.yaml")
    app.TASK_MANAGER = TaskManager(**app.CONFIG.task_mgr, password=os.getenv("MZ_MONGODB_PASSWD"))
    app.ORCHESTRATOR = Orchestrator(app.CONFIG, app.TASK_MANAGER)

    yield

    app.TASK_MANAGER.close()
    RemoteAPI.close()
    await RemoteAPI.aclose()

    await deregister_service(nacos_client, svc_host, svc_port, ephemeral=ephemeral)


app = FastAPI(lifespan=lifespan)
configure_monitoring(app)
# 生产环境配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)

app.include_router(chunk_router)
app.include_router(file_router)
app.include_router(query_router)
app.include_router(task_router)


@app.middleware("http")
async def postprocess_response(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    start_ts = time.time()
    response = await call_next(request)
    end_ts = time.time()
    if isinstance(response, JSONResponse):
        content = json.loads(response.body.decode("utf-8"))
        content["timestamp"] = end_ts
        content["cost_time"] = end_ts - start_ts
        response = JSONResponse(content=content)
        Logger.info("Return response: {}".format(repr(content)))
    return response


@app.exception_handler(Exception)
async def handle_exception(request: Request, exc: Exception):  # pyright: ignore
    if isinstance(exc, RuntimeError):
        return JSONResponse(content=FormatOutput("runtime-err", success=False))
    if isinstance(exc, TypeError):
        return JSONResponse(content=FormatOutput("type-err", success=False))
    if isinstance(exc, ValueError):
        return JSONResponse(content=FormatOutput("value-err", success=False))
    if isinstance(exc, IndexError):
        return JSONResponse(content=FormatOutput("index-err", success=False))
    if isinstance(exc, OSError):
        return JSONResponse(content=FormatOutput("os-err", success=False))
    if isinstance(exc, SyntaxError):
        return JSONResponse(content=FormatOutput("syntax-err", success=False))
    return JSONResponse(content=FormatOutput("other-err", success=False))


@app.get("/health")
async def index():
    return {"msg": "欢迎使用个人智能助手！"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("MZ_FASTAPI_PORT")),
        # reload=True,
        reload=False,
        reload_delay=5,
    )
