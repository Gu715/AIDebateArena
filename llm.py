# llm.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from openai import OpenAI

@dataclass
class ChatMessage:
    role: str   # "system" | "user" | "assistant"
    content: str

class OpenAICompatLLM:
    """
    针对 OpenAI 兼容 Chat Completions API 的轻封装：
    - .invoke(messages) -> str
    - 支持 base_url / api_key / model / temperature
    """
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.7):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def invoke(self, messages: List[ChatMessage]) -> str:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=self.temperature,
            stream=False,
            extra_body={"enable_thinking": False},
        )
        return (resp.choices[0].message.content or "").strip()
