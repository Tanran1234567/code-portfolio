from __future__ import annotations

from typing import Optional, Tuple

from ..llm_client import LLMClient
from ..prompt_loader import format_user, load_prompt, to_pretty_json
from ..models import (
    AgentSpeechGenerateInput,
    AgentSpeechGenerateOutput,
    PromptDebug,
)
from .attitude import generate_attitude
from .strategy import generate_strategy


def generate_speech(
    agent_id: str, payload: AgentSpeechGenerateInput
) -> Tuple[AgentSpeechGenerateOutput, Optional[PromptDebug]]:
    """
    外交官模块核心逻辑入口（每个国家/agent 生成一段 speech）。
    基于态度生成器+策略生成器的输出，拼接生成正式外交辞令。
    约定：service 层可替换为 LLM / 规则系统 / 多智能体推理；router 负责持久化与 HTTP。
    """
    round_no = payload.round

    # 默认：有 LLM 配置就走 LLM；否则回退 stub，保证接口可跑。
    llm = LLMClient()
    if llm.api_key and llm.api_base:
        # 核心新增：嵌套调用态度生成器、策略生成器，获取态度/策略文本
        # 复用相同的agent_id和payload，保证上下文一致（同一轮次、同一背景）
        attitude_output, _ = generate_attitude(
            agent_id, payload
        )  # 调用态度生成器，忽略debug
        strategy_output, _ = generate_strategy(
            agent_id, payload
        )  # 调用策略生成器，忽略debug
        attitude_content = attitude_output.speech  # 提取态度文本
        strategy_content = strategy_output.speech  # 提取策略文本

        print(f"{'=' * 40} ATTITUDE {'=' * 40}")
        print(attitude_content)
        print("\n")
        print(f"{'=' * 40} STRATEGY {'=' * 40}")
        print(strategy_content)

        # 原有逻辑：加载speech的prompt配置
        cfg = load_prompt(name="diplomat", prompt_key="speech")
        system = cfg.system
        # 改造模板渲染：新增attitude_content/strategy_content参数，实现动态拼接
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
                "attitude_content": attitude_content,  # 注入态度文本
                "strategy_content": strategy_content,  # 注入策略文本
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
                    agent_id=agent_id, round=round_no, speech=(text.strip())
                ),
                dbg,
            )

    # Stub兜底：基于前两个生成器的stub逻辑，生成统一的辞令stub
    text = f"[stub] Agent {agent_id} official speech for round {round_no} (based on attitude and strategy)."
    return (
        AgentSpeechGenerateOutput(agent_id=agent_id, round=round_no, speech=text),
        None,
    )


# def generate_speech(
#     agent_id: str, payload: AgentSpeechGenerateInput
# ) -> Tuple[AgentSpeechGenerateOutput, Optional[PromptDebug]]:
#     """
#     外交官模块核心逻辑入口（每个国家/agent 生成一段 speech）。
#     约定：service 层可替换为 LLM / 规则系统 / 多智能体推理；router 负责持久化与 HTTP。
#     """
#     round_no = payload.round

#     # 默认：有 LLM 配置就走 LLM；否则回退 stub，保证接口可跑。
#     llm = LLMClient()
#     if llm.api_key and llm.api_base:
#         # cfg = load_prompt("diplomat")
#         cfg = load_prompt(name="diplomat", prompt_key="speech")
#         system = cfg.system
#         user = format_user(
#             cfg.user_template,
#             {
#                 "agent_id": agent_id,
#                 "round": round_no,
#                 "background_json": to_pretty_json(payload.background),
#                 "issues_json": to_pretty_json(payload.issues),
#                 "speech_history_json": to_pretty_json(payload.speech_history),
#                 "shared_memory_json": to_pretty_json(payload.shared_memory),
#             },
#         )
#         temp = float(cfg.temperature)
#         dbg = PromptDebug(
#             system=system, user=user, model=(cfg.model or llm.model), temperature=temp
#         )
#         text = llm.chat(system=system, user=user, temperature=temp, timeout=60) or ""
#         if text.strip():
#             return (
#                 AgentSpeechGenerateOutput(
#                     # agent_id=agent_id, round=round_no, speech=text.strip()
#                 ),
#                 dbg,
#             )

#     text = f"[stub] Agent {agent_id} speech for round {round_no}."
#     return (
#         AgentSpeechGenerateOutput(agent_id=agent_id, round=round_no, speech=text),
#         None,
#     )
