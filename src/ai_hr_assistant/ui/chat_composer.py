from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).resolve().parent / "components" / "chat_composer"
_chat_composer = components.declare_component(
    "ai_hr_chat_composer",
    path=str(_COMPONENT_DIR),
)


def chat_composer(
    *,
    placeholder: str,
    submit_label: str = "↑",
    disabled: bool = False,
    reset_nonce: int = 0,
    key: str,
) -> dict[str, Any] | None:
    payload = _chat_composer(
        placeholder=placeholder,
        submit_label=submit_label,
        disabled=disabled,
        reset_nonce=reset_nonce,
        key=key,
        default={
            "text": "",
            "submit_token": "",
        },
    )

    if not isinstance(payload, dict):
        return None

    text = payload.get("text", "")
    submit_token = payload.get("submit_token", "")
    if not isinstance(text, str) or not isinstance(submit_token, str):
        return None

    return {
        "text": text,
        "submit_token": submit_token,
    }
