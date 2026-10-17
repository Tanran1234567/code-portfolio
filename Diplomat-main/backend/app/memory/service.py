from __future__ import annotations

from typing import Optional, Tuple

from ..llm_client import LLMClient
from ..models import MemoryGenerateInput, PromptDebug, RoundSignals, RoundSummary
from ..prompt_loader import format_user, load_prompt, to_pretty_json


def generate_round_summary(payload: MemoryGenerateInput) -> Tuple[RoundSummary, Optional[PromptDebug]]:
    """
    记忆模块（摘要器）核心逻辑入口。
    约定：不做 IO（数据库/HTTP/LLM），只做纯计算，方便单测与替换实现。
    """
    llm = LLMClient()
    if llm.api_key and llm.api_base:
        cfg = load_prompt("memory")
        system = cfg.system
        joined = "\n".join([f"{s.speaker}: {s.content}" for s in payload.speeches])
        user = format_user(
            cfg.user_template,
            {
                "round": payload.round,
                "speeches_text": joined,
                "speeches_json": to_pretty_json([s.model_dump() for s in payload.speeches]),
            },
        )
        temp = float(cfg.temperature)
        dbg = PromptDebug(system=system, user=user, model=(cfg.model or llm.model), temperature=temp)
        text = llm.chat(system=system, user=user, temperature=temp, timeout=60) or ""
        if text.strip():
            return (
                RoundSummary(
                    round=payload.round,
                    summary=text.strip(),
                    signals=RoundSignals(
                        alliances=[], conflicts=[], concessions=[], escalation_risk=None
                    ),
                ),
                dbg,
            )

    lines = [f"{s.speaker}: {s.content}" for s in payload.speeches]
    summary = " / ".join(lines) if lines else "No speeches."
    return (
        RoundSummary(
            round=payload.round,
            summary=summary,
            signals=RoundSignals(alliances=[], conflicts=[], concessions=[], escalation_risk=None),
        ),
        None,
    )

