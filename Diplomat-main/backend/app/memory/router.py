from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Path, Query, Header

from ..models import MemoryGenerateInput, RoundSummary, RoundSummaryResponse
from ..storage import db
from .service import generate_round_summary as generate_round_summary_service


router = APIRouter(tags=["memory"])


@router.get("/api/v1/scenarios/{scenario_id}/round-summaries", response_model=List[RoundSummary])
def list_round_summaries(
    scenario_id: str = Path(...),
    round: Optional[int] = Query(default=None, ge=1),
) -> List[RoundSummary]:
    store = db.get_or_create(scenario_id)
    if round is not None:
        rs = store.round_summaries.get(round)
        return [rs] if rs else []
    return [store.round_summaries[k] for k in sorted(store.round_summaries.keys())]


@router.get("/api/v1/scenarios/{scenario_id}/round-summaries/{round}", response_model=RoundSummary)
def get_round_summary(
    scenario_id: str = Path(...),
    round: int = Path(..., ge=1),
) -> RoundSummary:
    store = db.get_or_create(scenario_id)
    rs = store.round_summaries.get(round)
    if not rs:
        raise HTTPException(status_code=404, detail="round summary not found")
    return rs


@router.post(
    "/api/v1/scenarios/{scenario_id}/round-summaries",
    response_model=RoundSummaryResponse,
    response_model_exclude_none=True,
)
def generate_round_summary(
    scenario_id: str,
    payload: MemoryGenerateInput,
    x_debug_prompts: str | None = Header(default=None, alias="X-Debug-Prompts"),
) -> RoundSummaryResponse:
    debug_enabled = str(x_debug_prompts or "").strip().lower() in ("1", "true", "yes", "on")
    store = db.get_or_create(scenario_id)
    rs, dbg = generate_round_summary_service(payload)
    store.round_summaries[payload.round] = rs
    return RoundSummaryResponse(**rs.model_dump(), debug=dbg if debug_enabled else None)

