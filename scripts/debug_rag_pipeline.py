from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_hr_assistant.config import AppSettings
from ai_hr_assistant.services.document_loader import load_documents, split_text
from ai_hr_assistant.services.hr_assistant import HRAssistantService

QUERIES = [
    "Как оформить больничный?",
    "Как взять отпуск?",
    "Как получить справку с места работы?",
    "Можно ли работать удаленно?",
    "Как проходит обучение нового сотрудника?",
]


def build_context(retrieved_chunks: list) -> str:
    context_blocks = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[Источник {index}: {chunk.title}]\n"
            f"Релевантность: {chunk.score:.2f}\n"
            f"{chunk.text}"
        )
    return "\n\n".join(context_blocks)


def safe_console_text(text: str) -> str:
    return text.encode("cp1251", errors="replace").decode("cp1251")


def main() -> None:
    settings = AppSettings.from_env()
    assistant = HRAssistantService.from_settings(settings)
    status = assistant.ensure_knowledge_base(force_rebuild=True)

    print("== Load / Chunk / Index ==")
    print(
        f"ready={status.ready} documents={status.document_count} "
        f"chunks={status.chunk_count} message={status.message}"
    )

    documents = load_documents(settings.documents_dir)
    for document in documents:
        chunks = split_text(document.text)
        print(
            f"DOC {document.title} chars={len(document.text)} chunks={len(chunks)} "
            f"sample={' '.join(document.text.split())[:160]}"
        )
        for index, chunk in enumerate(chunks[:2], start=1):
            print(f"  chunk{index} chars={len(chunk)} sample={' '.join(chunk.split())[:180]}")

    print("\n== Retrieval / Prompt / Answer ==")
    for query in QUERIES:
        retrieved = assistant.knowledge_base.search(query, settings.top_k_sources)
        print(f"\nQ: {query}")
        print(
            f"top_score={(retrieved[0].score if retrieved else 0.0):.4f} "
            f"threshold={settings.similarity_threshold:.4f}"
        )
        for index, chunk in enumerate(retrieved, start=1):
            print(
                f"  top{index} score={chunk.score:.4f} title={chunk.title} "
                f"source={chunk.source_path}"
            )
            print(f"    snippet={' '.join(chunk.text.split())[:220]}")

        context = build_context(retrieved)
        print(f"  context_chars={len(context)} context_in_prompt={bool(context.strip())}")

        answer = assistant.ask(query)
        print(
            f"  fallback={answer.is_fallback} confidence={answer.confidence:.4f} "
            f"sources={[source.title for source in answer.sources]}"
        )
        preview = answer.text[:400].replace(chr(10), " | ")
        print(f"  answer={safe_console_text(preview)}")


if __name__ == "__main__":
    main()
