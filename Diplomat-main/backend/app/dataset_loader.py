from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .models import Background, CountryProfile, Issue, Issues


def _repo_root() -> Path:
    # e:\Diplomat\backend\app\dataset_loader.py -> parents[0]=app, [1]=backend, [2]=repo root
    return Path(__file__).resolve().parents[2]


def dataset_dir() -> Path:
    return _repo_root() / "dataset"


def normalize_scenario_id(scenario_id: str) -> str:
    sid = str(scenario_id or "").strip()
    if sid.lower().endswith(".json"):
        sid = sid[: -len(".json")]
    return sid


def load_dataset_raw(scenario_id: str) -> Dict[str, Any]:
    sid = normalize_scenario_id(scenario_id)
    path = dataset_dir() / f"{sid}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"dataset file must be a JSON object: {path}")
    return raw


def parse_background(raw: Dict[str, Any], scenario_id: str) -> Background:
    bg = raw.get("background") or {}
    if not isinstance(bg, dict):
        bg = {}

    countries_raw = bg.get("countries") or {}
    countries: Dict[str, CountryProfile] = {}
    if isinstance(countries_raw, dict):
        for k, v in countries_raw.items():
            countries[str(k)] = CountryProfile(profile=str(v or ""))

    return Background(
        scenario_id=str(raw.get("scenario_id") or scenario_id),
        world_background=str(bg.get("world_background") or ""),
        countries=countries,
    )


def parse_issues(raw: Dict[str, Any]) -> Issues:
    issues_raw = raw.get("issues") or []
    items: List[Issue] = []
    if isinstance(issues_raw, list):
        for idx, it in enumerate(issues_raw):
            if not isinstance(it, dict):
                continue
            title = str(it.get("title") or "").strip()
            desc = str(it.get("description") or "").strip()
            items.append(
                Issue(
                    issue_id=f"i-{idx + 1}",
                    title=title or f"议题 {idx + 1}",
                    description=desc,
                )
            )
    return Issues(issues=items)


def parse_secrets_notes(raw: Dict[str, Any]) -> Dict[str, str]:
    """
    dataset 新版 secrets 结构通常为：
      secrets: { "国家名": { "notes": "...", "decision_factors": {...} }, ... }
    前端当前 textarea 只适合 string，所以这里默认抽取 notes 作为 string。
    """
    sec = raw.get("secrets") or {}
    out: Dict[str, str] = {}
    if not isinstance(sec, dict):
        return out

    for country, obj in sec.items():
        if isinstance(obj, dict):
            notes = obj.get("notes")
            out[str(country)] = str(notes or "")
        else:
            out[str(country)] = str(obj or "")
    return out


def load_scenario_from_dataset(
    scenario_id: str,
) -> Tuple[Background, Issues, Dict[str, Any]]:
    """
    返回：
    - background: 适配后端 Background 模型
    - issues:     适配后端 Issues 模型（自动生成 issue_id）
    - secrets:    原始 secrets（保持 dict 结构，便于未来扩展）
    """
    sid = normalize_scenario_id(scenario_id)
    raw = load_dataset_raw(sid)
    background = parse_background(raw, scenario_id=sid)
    issues = parse_issues(raw)
    secrets = raw.get("secrets") if isinstance(raw.get("secrets"), dict) else {}
    return background, issues, secrets  # type: ignore[return-value]


def list_dataset_scenarios() -> List[str]:
    d = dataset_dir()
    if not d.exists() or not d.is_dir():
        return []
    # dataset/demo-001.json -> demo-001
    return sorted([p.stem for p in d.glob("*.json") if p.is_file()])
