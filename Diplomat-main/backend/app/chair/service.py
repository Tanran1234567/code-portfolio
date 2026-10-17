from __future__ import annotations

import json
from typing import Optional, Tuple

from ..llm_client import LLMClient
from ..models import (
    ChairAdjudicationItem,
    ChairGenerateInput,
    ChairGenerateOutput,
    PromptDebug,
)
from ..prompt_loader import format_user, load_prompt, to_pretty_json


def generate_chair_message(
    payload: ChairGenerateInput,
) -> Tuple[ChairGenerateOutput, Optional[PromptDebug]]:
    llm = LLMClient()

    # 选择 prompt
    prompt_key = payload.stage
    cfg = load_prompt(name="chair", prompt_key=prompt_key)
    system = cfg.system

    variables = {
        "round": payload.round or "",
        "question": payload.question or "",
        "background_json": to_pretty_json(payload.background or {}),
        "issues_json": to_pretty_json(payload.issues or {}),
        "constraints_json": to_pretty_json(payload.constraints or {}),
        "speeches_json": to_pretty_json([s.model_dump() for s in payload.speeches]),
    }

    user = format_user(cfg.user_template, variables)
    temp = float(cfg.temperature)
    dbg = PromptDebug(
        system=system, user=user, model=(cfg.model or llm.model), temperature=temp
    )

    if llm.api_key and llm.api_base:
        text = llm.chat(system=system, user=user, temperature=temp, timeout=90) or ""
        text = text.strip()
        if text:
            if payload.stage == "adjudicate":
                try:
                    obj = json.loads(text)
                    items = []
                    for it in obj.get("adjudication") or []:
                        if isinstance(it, dict):
                            items.append(ChairAdjudicationItem(**it))
                    return (
                        ChairGenerateOutput(
                            stage=payload.stage,
                            round=payload.round,
                            message=str(obj.get("message") or "").strip(),
                            adjudication=items or None,
                        ),
                        dbg,
                    )
                except Exception:
                    # 解析失败：退化为纯文本 message
                    return (
                        ChairGenerateOutput(
                            stage=payload.stage, round=payload.round, message=text
                        ),
                        dbg,
                    )

            return (
                ChairGenerateOutput(
                    stage=payload.stage, round=payload.round, message=text
                ),
                dbg,
            )

    # stub
    if payload.stage == "rule_qa":
        return (
            ChairGenerateOutput(
                stage="rule_qa",
                round=payload.round,
                message="[stub] 主席：我仅回答流程与规则问题；议题实质内容与裁定请留待最后裁定阶段。",
            ),
            None,
        )
    if payload.stage == "adjudicate":
        return (
            ChairGenerateOutput(
                stage="adjudicate",
                round=payload.round,
                message="[stub] 主席：将按议题逐项裁定（当前为无 LLM stub）。",
                adjudication=[],
            ),
            None,
        )
    return (
        ChairGenerateOutput(
            stage="stage_notice",
            round=payload.round,
            message="[stub] 主席：请各代表团遵守时间与流程。",
        ),
        None,
    )
