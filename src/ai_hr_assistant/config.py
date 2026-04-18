from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class AppSettings:
    app_name: str
    openai_api_key: str | None
    chat_model: str
    embedding_model: str
    documents_dir: Path
    index_file: Path
    similarity_threshold: float
    top_k_sources: int
    rag_debug: bool
    hr_contact_email: str
    hr_contact_label: str
    hr_contact_url: str | None

    @classmethod
    def from_env(cls) -> "AppSettings":
        root_dir = Path(__file__).resolve().parents[2]
        load_dotenv(root_dir / ".env")
        documents_dir = root_dir / "documents"
        index_dir = root_dir / "data" / "index"
        index_file = index_dir / "vector_index.json"

        documents_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            app_name="AI HR Assistant",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            documents_dir=documents_dir,
            index_file=index_file,
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.33")),
            top_k_sources=int(os.getenv("TOP_K_SOURCES", "4")),
            rag_debug=os.getenv("RAG_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"},
            hr_contact_email=os.getenv("HR_CONTACT_EMAIL", "hr@example.com"),
            hr_contact_label=os.getenv("HR_CONTACT_LABEL", "Команда HR"),
            hr_contact_url=os.getenv("HR_CONTACT_URL") or None,
        )
