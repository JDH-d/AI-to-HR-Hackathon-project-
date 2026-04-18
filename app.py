from __future__ import annotations

import html
import sys
import time
from pathlib import Path
from urllib.parse import quote

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_hr_assistant.config import AppSettings
from ai_hr_assistant.services.hr_assistant import HRAssistantService
from ai_hr_assistant.ui.chat_composer import chat_composer

WELCOME_MESSAGE = (
    "\u041f\u0440\u0438\u0432\u0435\u0442. \u042f AI to HR \u2014 "
    "\u0430\u0441\u0441\u0438\u0441\u0442\u0435\u043d\u0442 \u043f\u043e "
    "\u0432\u043d\u0443\u0442\u0440\u0435\u043d\u043d\u0438\u043c HR-\u0432\u043e\u043f\u0440\u043e\u0441\u0430\u043c "
    "\u043a\u043e\u043c\u043f\u0430\u043d\u0438\u0438. \u041c\u043e\u0433\u0443 "
    "\u043f\u043e\u043c\u043e\u0447\u044c \u0441 \u0431\u043e\u043b\u044c\u043d\u0438\u0447\u043d\u044b\u043c\u0438, "
    "\u043e\u0442\u043f\u0443\u0441\u043a\u0430\u043c\u0438, \u0441\u043f\u0440\u0430\u0432\u043a\u0430\u043c\u0438, "
    "\u0443\u0434\u0430\u043b\u0435\u043d\u043d\u043e\u0439 \u0440\u0430\u0431\u043e\u0442\u043e\u0439 \u0438 "
    "\u043e\u0431\u0443\u0447\u0435\u043d\u0438\u0435\u043c. \u0417\u0430\u0434\u0430\u0439\u0442\u0435 "
    "\u0432\u043e\u043f\u0440\u043e\u0441, \u0438 \u044f \u043e\u0442\u0432\u0435\u0447\u0443 \u043f\u043e "
    "\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u043c \u043a\u043e\u043c\u043f\u0430\u043d\u0438\u0438."
)

DEFAULT_HANDOFF_SUBJECT = (
    "\u041d\u0443\u0436\u043d\u0430 \u043f\u043e\u043c\u043e\u0449\u044c HR "
    "\u043f\u043e \u0432\u043e\u043f\u0440\u043e\u0441\u0443 \u0441\u043e\u0442\u0440\u0443\u0434\u043d\u0438\u043a\u0430"
)
DEFAULT_HANDOFF_BODY = (
    "\u041a\u043e\u043b\u043b\u0435\u0433\u0438, \u043f\u0440\u043e\u0448\u0443 "
    "\u043f\u043e\u043c\u043e\u0447\u044c \u0441 \u0432\u043e\u043f\u0440\u043e\u0441\u043e\u043c "
    "\u0441\u043e\u0442\u0440\u0443\u0434\u043d\u0438\u043a\u0430.\n\n"
    "\u041f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0439 \u0432\u043e\u043f\u0440\u043e\u0441:\n{question}"
)
SOURCES_LABEL = "\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u0438"
SIMILAR_SOURCES_LABEL = "\u041f\u043e\u0445\u043e\u0436\u0438\u0435 \u0444\u0440\u0430\u0433\u043c\u0435\u043d\u0442\u044b"
CONFIDENCE_LABEL = "\u0423\u0432\u0435\u0440\u0435\u043d\u043d\u043e\u0441\u0442\u044c \u043e\u0442\u0432\u0435\u0442\u0430"
COMPOSER_PLACEHOLDER = (
    "\u0421\u043f\u0440\u043e\u0441\u0438\u0442\u0435 \u043f\u0440\u043e "
    "\u043e\u0442\u043f\u0443\u0441\u043a, \u0431\u043e\u043b\u044c\u043d\u0438\u0447\u043d\u044b\u0439, "
    "\u0441\u043f\u0440\u0430\u0432\u043a\u0438, \u0443\u0434\u0430\u043b\u0435\u043d\u043a\u0443 "
    "\u0438\u043b\u0438 \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u0435"
)
HANDOFF_LABEL = "\u041f\u0435\u0440\u0435\u0434\u0430\u0442\u044c HR"
STREAM_ERROR_MESSAGE = (
    "\u041d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c "
    "\u0437\u0430\u0432\u0435\u0440\u0448\u0438\u0442\u044c \u043e\u0442\u0432\u0435\u0442. "
    "\u041f\u043e\u043f\u0440\u043e\u0431\u0443\u0439\u0442\u0435 "
    "\u043f\u043e\u0432\u0442\u043e\u0440\u0438\u0442\u044c \u0437\u0430\u043f\u0440\u043e\u0441."
)
SEARCHING_MESSAGE = (
    "\u0418\u0449\u0443 \u043e\u0442\u0432\u0435\u0442 \u0432 "
    "\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u0445 "
    "\u043a\u043e\u043c\u043f\u0430\u043d\u0438\u0438..."
)


def build_handoff_link(settings: AppSettings, last_question: str) -> str:
    if settings.hr_contact_url:
        return settings.hr_contact_url

    subject = quote(DEFAULT_HANDOFF_SUBJECT)
    body = quote(
        DEFAULT_HANDOFF_BODY.format(
            question=last_question or "\u0412\u043e\u043f\u0440\u043e\u0441 \u043d\u0435 \u0443\u043a\u0430\u0437\u0430\u043d",
        )
    )
    return f"mailto:{settings.hr_contact_email}?subject={subject}&body={body}"


def render_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --app-bg: #17181b;
                --app-bg-deep: #111214;
                --rail-bg: rgba(17, 18, 20, 0.92);
                --surface: #232529;
                --surface-strong: #2a2d33;
                --surface-soft: #1d2024;
                --border: rgba(255, 255, 255, 0.08);
                --border-strong: rgba(255, 255, 255, 0.14);
                --text: #f4f5f7;
                --text-soft: #a9afb8;
                --text-faint: #7f8792;
                --shadow-soft: 0 16px 48px rgba(0, 0, 0, 0.22);
                --rail-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.05);
                --rail-width: 272px;
                --chat-max-width: 860px;
                --message-max-width: 720px;
                --content-gutter: clamp(24px, 4vw, 48px);
            }

            html, body, [class*="css"] {
                font-family: "Segoe UI Variable Text", "Segoe UI", "Inter", sans-serif;
            }

            .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stMain"] {
                background:
                    radial-gradient(circle at top, rgba(255, 255, 255, 0.05), transparent 28%),
                    linear-gradient(180deg, #1b1d20 0%, var(--app-bg) 18%, var(--app-bg-deep) 100%);
                color: var(--text);
            }

            [data-testid="stHeader"],
            [data-testid="stToolbar"],
            [data-testid="collapsedControl"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"],
            footer,
            #MainMenu {
                display: none;
            }

            [data-testid="stSidebar"] {
                display: none;
            }

            .app-shell {
                position: fixed;
                inset: 0;
                pointer-events: none;
                z-index: 0;
            }

            .app-rail {
                position: absolute;
                left: 0;
                top: 0;
                bottom: 0;
                width: var(--rail-width);
                background: var(--rail-bg);
                box-shadow: var(--rail-shadow);
                padding: 1.2rem 1rem;
                box-sizing: border-box;
                backdrop-filter: blur(16px);
            }

            .app-logo {
                color: var(--text);
                font-size: 1rem;
                font-weight: 650;
                letter-spacing: 0.01em;
                line-height: 1.2;
            }

            .app-rail-empty {
                margin-top: 1.2rem;
                min-height: 120px;
                border-radius: 1.15rem;
                border: 1px dashed rgba(255, 255, 255, 0.06);
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.02), rgba(255, 255, 255, 0.01));
            }

            .block-container {
                position: relative;
                z-index: 1;
                width: calc(100% - var(--rail-width));
                max-width: none;
                margin: 0 0 0 var(--rail-width);
                padding: clamp(28px, 4vw, 40px) var(--content-gutter) 10rem;
                box-sizing: border-box;
            }

            .chat-turn {
                width: min(100%, var(--chat-max-width));
                margin: 0 auto 1.35rem;
                display: flex;
                justify-content: flex-start;
            }

            .chat-turn:first-of-type {
                margin-top: clamp(18px, 4vh, 42px);
            }

            .chat-turn.user {
                justify-content: flex-end;
            }

            .message-stack {
                min-width: 0;
                display: flex;
                flex-direction: column;
                gap: 0.72rem;
                align-items: flex-start;
                width: 100%;
            }

            .chat-turn.user .message-stack {
                align-items: flex-end;
            }

            .message-card,
            .message-sources,
            .message-meta,
            .message-actions {
                width: min(100%, var(--message-max-width));
            }

            .message-card {
                padding: 1rem 1.15rem;
                border-radius: 1.35rem;
                border: 1px solid var(--border);
                background: var(--surface);
                box-shadow: var(--shadow-soft);
                line-height: 1.62;
                font-size: 0.98rem;
                color: var(--text);
                word-break: break-word;
            }

            .message-card.assistant {
                background: linear-gradient(180deg, rgba(42, 45, 50, 0.98), rgba(35, 37, 41, 0.98));
            }

            .message-card.user {
                background: linear-gradient(180deg, rgba(47, 50, 57, 0.98), rgba(40, 43, 50, 0.98));
            }

            .message-card p {
                margin: 0;
            }

            .message-sources {
                border: 1px solid var(--border);
                border-radius: 1.15rem;
                background: rgba(30, 33, 37, 0.9);
                overflow: hidden;
            }

            .message-sources summary {
                list-style: none;
                cursor: pointer;
                padding: 0.9rem 1rem;
                color: var(--text);
                font-size: 0.94rem;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 0.7rem;
            }

            .message-sources summary::-webkit-details-marker {
                display: none;
            }

            .message-sources summary::before {
                content: "›";
                color: var(--text-soft);
                font-size: 1.1rem;
                line-height: 1;
                transform: rotate(0deg);
                transition: transform 140ms ease;
            }

            .message-sources[open] summary::before {
                transform: rotate(90deg);
            }

            .message-sources[open] summary {
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            }

            .source-list {
                padding: 0.2rem 0;
            }

            .source-item {
                padding: 0.9rem 1rem 1rem;
            }

            .source-item + .source-item {
                border-top: 1px solid rgba(255, 255, 255, 0.05);
            }

            .source-title {
                color: var(--text);
                font-size: 0.92rem;
                font-weight: 600;
                margin-bottom: 0.32rem;
            }

            .source-snippet {
                color: var(--text-soft);
                font-size: 0.91rem;
                line-height: 1.56;
                margin-bottom: 0.45rem;
            }

            .message-meta {
                color: var(--text-soft);
                font-size: 0.84rem;
                line-height: 1.4;
                padding: 0 0.15rem;
            }

            .message-meta strong {
                color: var(--text);
                font-weight: 600;
            }

            .message-actions {
                padding-top: 0.15rem;
            }

            .handoff-button {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-height: 40px;
                padding: 0.65rem 1rem;
                border-radius: 999px;
                border: 1px solid var(--border-strong);
                background: rgba(33, 38, 47, 0.92);
                color: var(--text);
                text-decoration: none;
                font-size: 0.92rem;
                font-weight: 600;
                transition: background 140ms ease, border-color 140ms ease, transform 140ms ease;
            }

            .handoff-button:hover {
                background: rgba(39, 45, 56, 0.98);
                border-color: rgba(255, 255, 255, 0.16);
                transform: translateY(-1px);
            }

            iframe[data-testid="stCustomComponentV1"],
            iframe.stCustomComponentV1 {
                position: fixed !important;
                left: calc(var(--rail-width) + ((100vw - var(--rail-width)) / 2));
                bottom: 1.25rem;
                transform: translateX(-50%);
                width: min(var(--chat-max-width), calc(100vw - var(--rail-width) - (2 * var(--content-gutter))));
                margin: 0;
                z-index: 20;
            }

            iframe[data-testid="stCustomComponentV1"],
            iframe.stCustomComponentV1 {
                border: none !important;
                background: transparent !important;
                border-radius: 1.5rem !important;
                overflow: hidden !important;
            }

            @media (max-width: 900px) {
                :root {
                    --rail-width: 0px;
                    --chat-max-width: 100%;
                    --message-max-width: 100%;
                }

                .app-rail {
                    display: none;
                }

                .block-container {
                    width: 100%;
                    margin-left: 0;
                    padding-top: 1.25rem;
                    padding-bottom: 9rem;
                }

                iframe[data-testid="stCustomComponentV1"],
                iframe.stCustomComponentV1 {
                    left: 50%;
                    width: calc(100vw - 2rem);
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_shell() -> None:
    st.markdown(
        """
        <div class="app-shell" aria-hidden="true">
            <div class="app-rail">
                <div class="app-logo">AI to HR</div>
                <div class="app-rail-empty"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE,
                "sources": [],
                "fallback": False,
                "confidence": 1.0,
                "show_handoff": False,
            }
        ]

    if "composer_reset_nonce" not in st.session_state:
        st.session_state.composer_reset_nonce = 0

    if "last_submit_token" not in st.session_state:
        st.session_state.last_submit_token = ""


def get_assistant(settings: AppSettings) -> HRAssistantService:
    assistant = st.session_state.get("assistant_service")
    if assistant is None:
        assistant = HRAssistantService.from_settings(settings)
        st.session_state.assistant_service = assistant
    return assistant


def format_rich_text(content: str) -> str:
    return html.escape(str(content)).replace("\n", "<br/>")


def build_sources_html(message: dict) -> str:
    sources = message.get("sources") or []
    if not sources:
        return ""

    label = SOURCES_LABEL if not message.get("fallback") else SIMILAR_SOURCES_LABEL
    items_html = "".join(
        f"""
        <div class="source-item">
            <div class="source-title">{html.escape(str(source.get("title", "")))}</div>
            <div class="source-snippet">{format_rich_text(source.get("snippet", ""))}</div>
        </div>
        """
        for source in sources
    )
    return f"""
    <details class="message-sources">
        <summary>{label}</summary>
        <div class="source-list">{items_html}</div>
    </details>
    """


def render_chat_turn(
    message: dict,
    *,
    settings: AppSettings | None = None,
    show_handoff: bool = False,
    last_question: str = "",
    target=None,
) -> None:
    role = "assistant" if message["role"] == "assistant" else "user"

    extras: list[str] = []
    if role == "assistant":
        sources_html = build_sources_html(message)
        if sources_html:
            extras.append(sources_html)

        if message.get("fallback"):
            extras.append(
                f"""
                <div class="message-meta">
                    {CONFIDENCE_LABEL}: <strong>{message.get("confidence", 0.0):.2f}</strong>
                </div>
                """
            )

        if show_handoff and settings and last_question:
            handoff_link = html.escape(build_handoff_link(settings, last_question), quote=True)
            extras.append(
                f"""
                <div class="message-actions">
                    <a class="handoff-button" href="{handoff_link}" target="_blank" rel="noopener noreferrer">
                        {HANDOFF_LABEL}
                    </a>
                </div>
                """
            )

    extras_html = "".join(extras)
    html_content = f"""
    <article class="chat-turn {role}">
        <div class="message-stack">
            <div class="message-card {role}">
                <p>{format_rich_text(message["content"])}</p>
            </div>
            {extras_html}
        </div>
    </article>
    """
    if target is None:
        st.html(html_content)
        return

    target.html(html_content)


def render_messages(settings: AppSettings) -> None:
    last_assistant_index = max(
        (index for index, item in enumerate(st.session_state.messages) if item["role"] == "assistant"),
        default=-1,
    )
    last_user_message = next(
        (
            item["content"]
            for item in reversed(st.session_state.messages)
            if item["role"] == "user"
        ),
        "",
    )

    for index, message in enumerate(st.session_state.messages):
        render_chat_turn(
            message,
            settings=settings,
            show_handoff=bool(message.get("show_handoff")) and index == last_assistant_index,
            last_question=last_user_message,
        )


def render_composer() -> str | None:
    payload = chat_composer(
        placeholder=COMPOSER_PLACEHOLDER,
        submit_label="\u2191",
        reset_nonce=st.session_state.composer_reset_nonce,
        key=f"hr_chat_composer_{st.session_state.composer_reset_nonce}",
    )
    if not payload:
        return None

    submit_token = payload.get("submit_token", "")
    if not submit_token or submit_token == st.session_state.last_submit_token:
        return None

    st.session_state.last_submit_token = submit_token
    prompt = payload.get("text", "").strip()
    return prompt or None


def build_response_message(answer) -> dict:
    return {
        "role": "assistant",
        "content": answer.text,
        "sources": [
            {
                "title": source.title,
                "snippet": source.snippet,
                "score": source.score,
            }
            for source in answer.sources
        ],
        "fallback": answer.is_fallback,
        "confidence": answer.confidence,
        "show_handoff": answer.show_handoff,
    }


def stream_assistant_response(
    assistant: HRAssistantService,
    settings: AppSettings,
    prompt: str,
) -> dict:
    assistant_placeholder = st.empty()
    partial_message = {
        "role": "assistant",
        "content": SEARCHING_MESSAGE,
        "sources": [],
        "fallback": False,
        "confidence": 0.0,
        "show_handoff": False,
    }
    render_chat_turn(partial_message, target=assistant_placeholder)

    streamed_text = ""
    last_render_at = 0.0
    final_message: dict | None = None

    for event in assistant.ask_stream(prompt):
        if event.delta:
            streamed_text += event.delta
            now = time.monotonic()
            should_render = (
                (now - last_render_at) >= 0.05
                or event.delta.endswith(("\n", ".", "!", "?", ":"))
            )
            if should_render:
                partial_message["content"] = streamed_text or SEARCHING_MESSAGE
                render_chat_turn(partial_message, target=assistant_placeholder)
                last_render_at = now

        if event.answer is not None:
            final_message = build_response_message(event.answer)
            render_chat_turn(
                final_message,
                settings=settings,
                show_handoff=bool(final_message.get("show_handoff")),
                last_question=prompt,
                target=assistant_placeholder,
            )

    if final_message is not None:
        return final_message

    partial_message["content"] = streamed_text or STREAM_ERROR_MESSAGE
    partial_message["fallback"] = not bool(streamed_text)
    partial_message["confidence"] = 0.0
    partial_message["show_handoff"] = False
    render_chat_turn(
        partial_message,
        settings=settings,
        show_handoff=False,
        last_question=prompt,
        target=assistant_placeholder,
    )
    return partial_message


def main() -> None:
    st.set_page_config(
        page_title="AI to HR",
        page_icon="AI",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_styles()
    render_shell()
    init_state()

    settings = AppSettings.from_env()
    assistant = get_assistant(settings)

    render_messages(settings)

    prompt = render_composer()
    if not prompt:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    st.session_state.composer_reset_nonce += 1

    render_chat_turn({"role": "user", "content": prompt})

    response_message = stream_assistant_response(
        assistant=assistant,
        settings=settings,
        prompt=prompt,
    )
    st.session_state.messages.append(response_message)
    st.rerun()


if __name__ == "__main__":
    main()
