from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import Background, EvaluationOutput, Issues, RoundSummary, Speech
from .dataset_loader import load_scenario_from_dataset


@dataclass
class ScenarioStore:
    background: Optional[Background] = None
    issues: Optional[Issues] = None
    # dataset 新版支持：私密信息（不进入公开接口的 response_model）
    secrets: Optional[Dict[str, Any]] = None
    speeches: List[Speech] = field(default_factory=list)
    round_summaries: Dict[int, RoundSummary] = field(default_factory=dict)
    evaluations: List[EvaluationOutput] = field(default_factory=list)


class InMemoryDB:
    def __init__(self) -> None:
        self.scenarios: Dict[str, ScenarioStore] = {}

    def get_or_create(self, scenario_id: str) -> ScenarioStore:
        if scenario_id not in self.scenarios:
            store = ScenarioStore()
            # 尝试从仓库根目录 dataset/{scenario_id}.json 自动加载（失败则保持空 store）
            try:
                background, issues, secrets = load_scenario_from_dataset(scenario_id)
                store.background = background
                store.issues = issues
                store.secrets = secrets
            except FileNotFoundError:
                pass
            except Exception:
                # 解析失败不阻断服务启动/使用；保持空 store 让调用方按需注入
                pass
            self.scenarios[scenario_id] = store
        return self.scenarios[scenario_id]


db = InMemoryDB()
