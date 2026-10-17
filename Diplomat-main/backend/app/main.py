from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware

from .chair.router import router as chair_router
from .dataset_loader import list_dataset_scenarios
from .diplomat.router import router as diplomat_router
from .evaluator.router import router as evaluator_router
from .memory.router import router as memory_router
from .models import Background, Issues, Speech
from .storage import db


app = FastAPI(title="Diplomat API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/scenarios/{scenario_id}/background", response_model=Background)
def get_background(scenario_id: str = Path(...)) -> Background:
    store = db.get_or_create(scenario_id)
    if not store.background:
        raise HTTPException(status_code=404, detail="background not found")
    return store.background


@app.get("/api/v1/scenarios/{scenario_id}/issues", response_model=Issues)
def get_issues(scenario_id: str = Path(...)) -> Issues:
    store = db.get_or_create(scenario_id)
    if not store.issues:
        raise HTTPException(status_code=404, detail="issues not found")
    return store.issues


@app.get("/api/v1/scenarios/{scenario_id}/secrets")
def get_secrets(
    scenario_id: str = Path(...),
    full: bool = Query(default=False),
) -> Dict[str, Any]:
    """
    - full=0（默认）：返回 {国家: notes字符串}，便于前端 textarea 直接展示
    - full=1：返回 dataset 原始 secrets 对象（含 decision_factors 等）
    """
    store = db.get_or_create(scenario_id)
    if not store.secrets:
        raise HTTPException(status_code=404, detail="secrets not found")
    if full:
        return store.secrets

    out: Dict[str, str] = {}
    for country, obj in (store.secrets or {}).items():
        if isinstance(obj, dict):
            out[str(country)] = str(obj.get("notes") or "")
        else:
            out[str(country)] = str(obj or "")
    return out


@app.get("/api/v1/datasets/scenarios", response_model=List[str])
def list_scenarios_from_dataset() -> List[str]:
    """
    列出仓库根目录 dataset/ 下的 *.json（仅文件名 stem，例如 demo-001）。
    """
    return list_dataset_scenarios()


@app.get("/api/v1/scenarios/{scenario_id}/speeches", response_model=List[Speech])
def list_speeches(
    scenario_id: str = Path(...),
    round: Optional[int] = Query(default=None, ge=1),
    speaker: Optional[str] = Query(default=None),
) -> List[Speech]:
    store = db.get_or_create(scenario_id)
    items = store.speeches
    if round is not None:
        items = [s for s in items if s.round == round]
    if speaker is not None:
        items = [s for s in items if s.speaker == speaker]
    return items


@app.post("/api/v1/scenarios/{scenario_id}/speeches", response_model=Speech)
def create_speech(scenario_id: str, speech: Speech) -> Speech:
    store = db.get_or_create(scenario_id)
    store.speeches.append(speech)
    return speech


app.include_router(memory_router)
app.include_router(chair_router)
app.include_router(diplomat_router)
app.include_router(evaluator_router)
