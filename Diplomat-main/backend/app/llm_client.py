from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


class LLMClient:
    """
    公用 LLM Client（OpenAI SDK，兼容 OpenRouter /chat/completions）。

    重要：不在代码里内置默认 key，避免把密钥写入仓库。
    - OPENROUTER_API_KEY（推荐）
    - OPENROUTER_BASE_URL（默认 https://openrouter.ai/api/v1）
    - OPENROUTER_MODEL（默认 openai/gpt-5.2）
    - OPENROUTER_VISION_MODEL（可选）
    - OPENROUTER_REASONING（1/true/yes/on 开启 reasoning）

    兼容旧变量名（如你之前的配置）：
    - OPENAI_API_KEY
    - OPENAI_API_BASE
    - OPENAI_MODEL / OPENAI_CHAT_MODEL
    - OPENAI_VISION_MODEL / OPENAI_VLM_MODEL
    - LLM_DEBUG_PROMPTS（1/true/yes/on）
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        vision_model: Optional[str] = None,
    ):
        # 尝试加载 .env（如果用户没手动 export 环境变量，也能直接跑）
        try:
            from dotenv import load_dotenv

            # 默认从当前工作目录向上寻找 .env
            load_dotenv(override=False)
        except Exception:
            pass

        self.api_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY", "")
            or os.getenv("OPENAI_API_KEY", "")
        )
        self.api_base = (
            api_base
            or os.getenv("OPENROUTER_BASE_URL", "")
            or os.getenv("OPENAI_API_BASE", "")
            or "https://openrouter.ai/api/v1"
        ).rstrip("/")

        self.model = (
            model
            or os.getenv("OPENROUTER_MODEL")
            or os.getenv("OPENAI_MODEL")
            or os.getenv("OPENAI_CHAT_MODEL")
            or "openai/gpt-5.2"
        )
        self.vision_model = (
            vision_model
            or os.getenv("OPENROUTER_VISION_MODEL")
            or os.getenv("OPENAI_VISION_MODEL")
            or os.getenv("OPENAI_VLM_MODEL")
            or self.model
        )

        dbg = str(os.getenv("LLM_DEBUG_PROMPTS", "0") or "").strip().lower()
        self.debug_prompts = dbg in ("1", "true", "yes", "on")

        reasoning = str(os.getenv("OPENROUTER_REASONING", "0") or "").strip().lower()
        self.reasoning_enabled = reasoning in ("1", "true", "yes", "on")

        self._client = None
        if self.api_key and self.api_base:
            try:
                from openai import OpenAI

                self._client = OpenAI(base_url=self.api_base, api_key=self.api_key)
            except Exception:
                self._client = None

    def _debug_print(self, title: str, payload: Dict[str, Any]) -> None:
        if not getattr(self, "debug_prompts", False):
            return
        try:
            body = json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            body = str(payload)
        try:
            print("\n" + ("=" * 24) + f" {title} " + ("=" * 24), flush=True)
            print(body, flush=True)
            print(("=" * 24) + " END " + ("=" * 27) + "\n", flush=True)
        except Exception:
            return

    def chat_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 1.0,
        timeout: int = 20,
        model: Optional[str] = None,
        reasoning_enabled: Optional[bool] = None,
    ) -> Optional[str]:
        if not self._client or not messages:
            return None
        try:
            use_reasoning = (
                self.reasoning_enabled
                if reasoning_enabled is None
                else bool(reasoning_enabled)
            )
            extra_body = {"reasoning": {"enabled": True}} if use_reasoning else None

            self._debug_print(
                "LLM chat_messages",
                {
                    "base_url": self.api_base,
                    "model": model or self.model,
                    "temperature": float(temperature),
                    "messages": messages,
                    "extra_body": extra_body,
                },
            )

            resp = self._client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=float(temperature),
                timeout=timeout,
                extra_body=extra_body,
            )
            msg = resp.choices[0].message if resp and resp.choices else None
            return (getattr(msg, "content", None) or "") if msg else None
        except Exception:
            return None

    def chat_messages_with_reasoning_details(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 1.0,
        timeout: int = 20,
        model: Optional[str] = None,
        reasoning_enabled: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        返回可“接力”的 assistant message：
        {
          "role": "assistant",
          "content": "...",
          "reasoning_details": <原样返回，供下一次 messages 继续使用>
        }
        """
        if not self._client or not messages:
            return None
        try:
            use_reasoning = (
                self.reasoning_enabled
                if reasoning_enabled is None
                else bool(reasoning_enabled)
            )
            extra_body = {"reasoning": {"enabled": True}} if use_reasoning else None
            resp = self._client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=float(temperature),
                timeout=timeout,
                extra_body=extra_body,
            )
            msg = resp.choices[0].message if resp and resp.choices else None
            if not msg:
                return None
            out: Dict[str, Any] = {
                "role": "assistant",
                "content": getattr(msg, "content", None),
            }
            # OpenRouter 会在 message 上提供 reasoning_details；按原样透传
            if hasattr(msg, "reasoning_details"):
                out["reasoning_details"] = getattr(msg, "reasoning_details")
            return out
        except Exception:
            return None

    def chat(
        self, system: str, user: str, *, temperature: float = 1.0, timeout: int = 20
    ) -> Optional[str]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.chat_messages(
            messages, temperature=temperature, timeout=timeout, model=self.model
        )

    def chat_vision(
        self,
        content: List[Dict[str, Any]],
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        timeout: int = 60,
        model: Optional[str] = None,
    ) -> Optional[str]:
        if not content:
            return None
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        return self.chat_messages(
            messages,
            temperature=temperature,
            timeout=timeout,
            model=model or self.vision_model,
        )

    def chat_messages_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 1.0,
        timeout: int = 60,
        model: Optional[str] = None,
        reasoning_enabled: Optional[bool] = None,
    ):
        """
        通用消息流式接口：支持包含图像内容的 messages。
        逐条 yield 文本 delta（若后端以 SSE 方式返回）。
        """
        if not self._client or not messages:
            return
        try:
            use_reasoning = (
                self.reasoning_enabled
                if reasoning_enabled is None
                else bool(reasoning_enabled)
            )
            extra_body = {"reasoning": {"enabled": True}} if use_reasoning else None
            resp = self._client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=float(temperature),
                timeout=timeout,
                extra_body=extra_body,
                stream=True,
            )
            for ev in resp:
                try:
                    delta = ev.choices[0].delta.content
                    if delta:
                        yield delta
                except Exception:
                    continue
        except Exception:
            return
