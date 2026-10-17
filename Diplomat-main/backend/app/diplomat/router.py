from __future__ import annotations

from fastapi import APIRouter, Header

from ..models import AgentSpeechGenerateInput, AgentSpeechGenerateResponse, Speech
from ..storage import db
from .service import generate_speech as generate_speech_service


router = APIRouter(tags=["diplomat"])


@router.post(
    "/api/v1/scenarios/{scenario_id}/agents/{agent_id}/speeches",
    response_model=AgentSpeechGenerateResponse,
    response_model_exclude_none=True,
)
def agent_generate_speech(
    scenario_id: str,
    agent_id: str,
    payload: AgentSpeechGenerateInput,
    x_debug_prompts: str | None = Header(default=None, alias="X-Debug-Prompts"),
) -> AgentSpeechGenerateResponse:
    debug_enabled = str(x_debug_prompts or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    out, dbg = generate_speech_service(agent_id=agent_id, payload=payload)
    store = db.get_or_create(scenario_id)
    visibility = (
        "private"
        if str(payload.stage or "") == "consultation_private_message"
        else "public"
    )
    store.speeches.append(
        Speech(
            round=out.round,
            speaker=agent_id,
            content=out.speech,
            stage=payload.stage,
            visibility=visibility,
        )
    )
    return AgentSpeechGenerateResponse(
        **out.model_dump(), debug=dbg if debug_enabled else None
    )
