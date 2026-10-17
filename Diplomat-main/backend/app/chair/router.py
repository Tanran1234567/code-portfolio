from __future__ import annotations

from fastapi import APIRouter, Header, Path

from ..models import ChairGenerateInput, ChairGenerateResponse, Speech
from ..storage import db
from .service import generate_chair_message


router = APIRouter(tags=["chair"])


@router.post(
    "/api/v1/scenarios/{scenario_id}/chair/messages",
    response_model=ChairGenerateResponse,
    response_model_exclude_none=True,
)
def chair_generate_message(
    scenario_id: str = Path(...),
    payload: ChairGenerateInput = ...,
    x_debug_prompts: str | None = Header(default=None, alias="X-Debug-Prompts"),
) -> ChairGenerateResponse:
    debug_enabled = str(x_debug_prompts or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    out, dbg = generate_chair_message(payload)

    # 持久化：主席输出也作为 speech 存档（便于导出与回放）
    if out.message.strip():
        store = db.get_or_create(scenario_id)
        store.speeches.append(
            Speech(
                round=int(payload.round or 1),
                speaker="主席",
                content=out.message.strip(),
                stage=payload.stage,
                visibility="public",
            )
        )

    return ChairGenerateResponse(
        **out.model_dump(), debug=dbg if debug_enabled else None
    )
