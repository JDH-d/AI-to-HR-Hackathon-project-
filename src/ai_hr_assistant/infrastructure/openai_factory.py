from __future__ import annotations

from openai import OpenAI

from ai_hr_assistant.config import AppSettings


def build_openai_client(settings: AppSettings) -> OpenAI | None:
    if not settings.openai_api_key:
        return None

    return OpenAI(api_key=settings.openai_api_key)
