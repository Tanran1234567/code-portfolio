from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PromptConfig:
    name: str
    version: str
    system: str
    user_template: str
    temperature: float
    model: Optional[str] = None


_CACHE: Dict[str, tuple[float, PromptConfig]] = {}


def _app_dir() -> Path:
    return Path(__file__).resolve().parent


# def load_prompt(name: str) -> PromptConfig:
#     """
#     从各模块目录加载 prompt：
#       - backend/app/{name}/prompt.json
#     带简单 mtime 缓存，便于开发时热更新。
#     """
#     app_dir = _app_dir()
#     candidates = [app_dir / name / "prompt.json"]
#     path = None
#     for p in candidates:
#         if p.exists():
#             path = p
#             break
#     if path is None:
#         raise FileNotFoundError(f"prompt not found for '{name}' (tried: {candidates})")

#     mtime = path.stat().st_mtime
#     cached = _CACHE.get(name)
#     if cached and cached[0] == mtime:
#         return cached[1]

#     raw = json.loads(path.read_text(encoding="utf-8"))
#     cfg = PromptConfig(
#         name=str(raw.get("name") or name),
#         version=str(raw.get("version") or "v1"),
#         system=str(raw.get("system") or ""),
#         user_template=str(raw.get("user_template") or ""),
#         temperature=float(
#             raw.get("temperature") if raw.get("temperature") is not None else 0.7
#         ),
#         model=(raw.get("model") if raw.get("model") else None),
#     )
#     _CACHE[name] = (mtime, cfg)
#     return cfg


def load_prompt(
    name: str,
    prompt_key: Optional[str] = None  # 新增：要加载的配置项标识（如"attitude"/"speech"）
) -> PromptConfig:
    """
    从各模块目录加载 prompt（适配单/多配置）：
      - backend/app/{name}/prompt.json（支持单对象/数组格式）
    带简单 mtime 缓存，便于开发时热更新。
    新增：prompt_key指定加载数组中某一条配置（按name字段匹配）
    """
    app_dir = _app_dir()
    candidates = [app_dir / name / "prompt.json"]
    path = None
    for p in candidates:
        if p.exists():
            path = p
            break
    if path is None:
        raise FileNotFoundError(f"prompt not found for '{name}' (tried: {candidates})")

    # 1. 检查文件修改时间，判断是否使用缓存
    mtime = path.stat().st_mtime
    cached = _CACHE.get(name)
    if cached and cached[0] == mtime:
        # 缓存命中：从缓存的配置字典中筛选目标项
        cfg_dict = cached[1]
        target_cfg = cfg_dict.get(prompt_key) if prompt_key else next(iter(cfg_dict.values()))
        return target_cfg

    # 2. 缓存未命中：读取并解析JSON（支持单对象/数组）
    raw = json.loads(path.read_text(encoding="utf-8"))
    cfg_dict = {}  # 新缓存格式：{配置项name: PromptConfig对象}

    # 处理数组格式（你的新需求）
    if isinstance(raw, list):
        for item in raw:
            item_name = str(item.get("name") or f"{name}_default")
            cfg = PromptConfig(
                name=item_name,
                version=str(item.get("version") or "v1"),
                system=str(item.get("system") or ""),
                user_template=str(item.get("user_template") or ""),
                temperature=float(item.get("temperature")) if item.get("temperature") is not None else 0.7,
                model=item.get("model") or None,
            )
            cfg_dict[item_name] = cfg
    # 兼容原单对象格式（不破坏旧代码）
    elif isinstance(raw, dict):
        cfg = PromptConfig(
            name=str(raw.get("name") or name),
            version=str(raw.get("version") or "v1"),
            system=str(raw.get("system") or ""),
            user_template=str(raw.get("user_template") or ""),
            temperature=float(raw.get("temperature")) if raw.get("temperature") is not None else 0.7,
            model=raw.get("model") or None,
        )
        cfg_dict[cfg.name] = cfg
    else:
        raise ValueError(f"prompt.json must be JSON object or array, got {type(raw)}")

    # 3. 更新缓存（键：模块name；值：(mtime, 配置字典{prompt_key: PromptConfig})）
    _CACHE[name] = (mtime, cfg_dict)

    # 4. 返回目标配置：指定prompt_key则返回对应项，否则返回第一个配置
    if prompt_key:
        if prompt_key not in cfg_dict:
            raise KeyError(f"Prompt key '{prompt_key}' not found in {name}/prompt.json (available keys: {list(cfg_dict.keys())})")
        return cfg_dict[prompt_key]
    return next(iter(cfg_dict.values()))


def to_pretty_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def format_user(template: str, variables: Dict[str, Any]) -> str:
    """
    用 str.format 渲染模板。variables 缺失时会 KeyError，便于尽早发现模板问题。
    """
    return template.format(**variables)
