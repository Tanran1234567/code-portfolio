from __future__ import annotations

from typing import Optional, Tuple

from ..llm_client import LLMClient
from ..prompt_loader import format_user, load_prompt, to_pretty_json
from ..models import (
    AgentSpeechGenerateInput,
    AgentSpeechGenerateOutput,
    PromptDebug,
)


def generate_strategy(
    agent_id: str, payload: AgentSpeechGenerateInput
) -> Tuple[AgentSpeechGenerateOutput, Optional[PromptDebug]]:
    """
    外交官模块核心逻辑入口（每个国家/agent 生成一段 speech）。
    约定：service 层可替换为 LLM / 规则系统 / 多智能体推理；router 负责持久化与 HTTP。
    """
    round_no = payload.round
    print(payload.stage)
    if (not payload.stage.startswith("negotiation_round_")):
        return (
        AgentSpeechGenerateOutput(agent_id=agent_id, round=round_no, speech=""),
        None,
    )
    # 默认：有 LLM 配置就走 LLM；否则回退 stub，保证接口可跑。
    llm = LLMClient()
    if llm.api_key and llm.api_base:
        # cfg = load_prompt("diplomat")
        cfg = load_prompt(name="diplomat", prompt_key="strategy")
        system = cfg.system
        user = format_user(
            cfg.user_template,
            {
                "agent_id": agent_id,
                "round": round_no,
                "stage": payload.stage or "",
                "background_json": to_pretty_json(payload.background),
                "issues_json": to_pretty_json(payload.issues),
                "speech_history_json": to_pretty_json(payload.speech_history),
                "shared_memory_json": to_pretty_json(payload.shared_memory),
                "constraints_json": to_pretty_json(payload.constraints),
                "secret_json": to_pretty_json(payload.secret),
            },
        )
        temp = float(cfg.temperature)
        dbg = PromptDebug(
            system=system, user=user, model=(cfg.model or llm.model), temperature=temp
        )
        text = llm.chat(system=system, user=user, temperature=temp, timeout=60) or ""
        if text.strip():
            return (
                AgentSpeechGenerateOutput(
                    agent_id=agent_id, round=round_no, speech=text.strip()
                ),
                dbg,
            )

    text = f"[stub] Agent {agent_id} speech for round {round_no}."
    return (
        AgentSpeechGenerateOutput(agent_id=agent_id, round=round_no, speech=text),
        None,
    )
