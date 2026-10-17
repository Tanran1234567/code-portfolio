from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class CountryProfile(BaseModel):
    profile: str


class Background(BaseModel):
    scenario_id: str
    world_background: str
    countries: Dict[str, CountryProfile]


class Issue(BaseModel):
    issue_id: str
    title: str
    description: str


class Issues(BaseModel):
    issues: List[Issue]


class Speech(BaseModel):
    round: int = Field(ge=1)
    speaker: str
    content: str
    # 可选：用于区分开场陈述/谈判/新闻发布/主席裁定/自由磋商等阶段
    stage: Optional[str] = None
    # 可选：标记私密内容（例如自由磋商的小窗私聊）。前端可据此决定展示范围。
    visibility: Optional[Literal["public", "private"]] = "public"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class RoundSignals(BaseModel):
    alliances: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    concessions: List[str] = Field(default_factory=list)
    escalation_risk: Optional[Literal["low", "medium", "high"]] = None


class RoundSummary(BaseModel):
    round: int = Field(ge=1)
    summary: str
    signals: RoundSignals


class PromptDebug(BaseModel):
    system: Optional[str] = None
    user: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None


class AgentSpeechGenerateInput(BaseModel):
    # agent_id 可由 Path 推断，保留字段是为了兼容你文档里的 JSON 定义
    agent_id: Optional[str] = None
    round: int = Field(ge=1)
    background: Dict[str, Any]
    issues: Dict[str, Any]
    speech_history: List[Dict[str, Any]] = Field(default_factory=list)
    shared_memory: List[Dict[str, Any]] = Field(default_factory=list)
    # 新增：当前发言所处阶段（opening/negotiation_1/mid_press/negotiation_2/final_press/...）
    stage: Optional[str] = None
    # 新增：主席/赛制对本次发言的约束（如 max_chars、speaking_time_minutes 等）
    constraints: Dict[str, Any] = Field(default_factory=dict)
    # 新增：本方私密信息（不会分享给他方）。建议为字符串或结构化对象。
    secret: Optional[Any] = None


class AgentSpeechGenerateOutput(BaseModel):
    agent_id: str
    round: int
    speech: str


class AgentSpeechGenerateResponse(AgentSpeechGenerateOutput):
    debug: Optional[PromptDebug] = None


class ChairGenerateInput(BaseModel):
    """
    主席/主持人模块输入：
    - rule_qa：回答规则/流程问题（不得就议题内容表态）
    - adjudicate：按议题裁定是否达成、达成内容是什么
    - stage_notice：阶段提示/时间提醒
    """

    stage: Literal["rule_qa", "adjudicate", "stage_notice"]
    round: Optional[int] = Field(default=None, ge=1)
    # 规则/流程问题（仅 stage=rule_qa 时使用）
    question: Optional[str] = None
    # 背景与议题（用于裁定/提示时参考）
    background: Optional[Dict[str, Any]] = None
    issues: Optional[Dict[str, Any]] = None
    # 全部公开发言记录（裁定依据主要来自公开发言与已形成的共识文本）
    speeches: List[Speech] = Field(default_factory=list)
    # 主席阶段控制参数（例如剩余时间、每轮限制等）
    constraints: Dict[str, Any] = Field(default_factory=dict)


class ChairAdjudicationItem(BaseModel):
    issue_id: str
    title: Optional[str] = None
    status: Literal["达成", "未达成", "部分达成"]
    content: str = ""


class ChairGenerateOutput(BaseModel):
    stage: str
    round: Optional[int] = None
    message: str
    adjudication: Optional[List[ChairAdjudicationItem]] = None


class ChairGenerateResponse(ChairGenerateOutput):
    debug: Optional[PromptDebug] = None


class MemoryGenerateInput(BaseModel):
    round: int = Field(ge=1)
    speeches: List[Speech]


class EvaluationInput(BaseModel):
    scenario_id: str
    rubrics: List[Dict[str, Any]] = Field(default_factory=list)
    speeches: List[Speech] = Field(default_factory=list)


class EvaluationOutput(BaseModel):
    scores: Dict[str, int]
    feedback: Dict[str, str]


class RoundSummaryResponse(RoundSummary):
    debug: Optional[PromptDebug] = None


class EvaluationResponse(EvaluationOutput):
    debug: Optional[PromptDebug] = None
