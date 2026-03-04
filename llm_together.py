# llm_together.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import os

from openai import OpenAI


@dataclass
class TogetherConfig:
    api_key_env: str = "TOGETHER_API_KEY"
    base_url: str = "https://api.together.ai/v1"
    model: str = "Qwen/Qwen2.5-7B-Instruct-Turbo"
    temperature: float = 0.0
    max_tokens: int = 350


class TogetherLLM:
    def __init__(self, cfg: TogetherConfig):
        api_key = os.getenv(cfg.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Missing Together API key. Set {cfg.api_key_env} as an environment variable."
            )

        self.cfg = cfg
        self.client = OpenAI(api_key=api_key, base_url=cfg.base_url)

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=int(max_tokens or self.cfg.max_tokens),
        )
        return (resp.choices[0].message.content or "").strip()