from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Header

from ..models import EvaluationInput, EvaluationOutput, EvaluationResponse
from ..storage import db
from .service import evaluate as evaluate_service


router = APIRouter(tags=["evaluator"])


@router.post(
    "/api/v1/scenarios/{scenario_id}/evaluations",
    response_model=EvaluationResponse,
    response_model_exclude_none=True,
)
def generate_evaluation(
    scenario_id: str,
    payload: EvaluationInput,
    x_debug_prompts: str | None = Header(default=None, alias="X-Debug-Prompts"),
) -> EvaluationResponse:
    debug_enabled = str(x_debug_prompts or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    store = db.get_or_create(scenario_id)
    out, dbg = evaluate_service(payload)
    store.evaluations.append(out)
    return EvaluationResponse(**out.model_dump(), debug=dbg if debug_enabled else None)


@router.get(
    "/api/v1/scenarios/{scenario_id}/evaluations/latest",
    response_model=EvaluationOutput,
)
def get_latest_evaluation(scenario_id: str = Path(...)) -> EvaluationOutput:
    store = db.get_or_create(scenario_id)
    if not store.evaluations:
        raise HTTPException(status_code=404, detail="evaluation not found")
    return store.evaluations[-1]
