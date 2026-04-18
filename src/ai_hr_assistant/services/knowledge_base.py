from __future__ import annotations

import hashlib
import json
import logging
import math
import re

from openai import OpenAI

from ai_hr_assistant.config import AppSettings
from ai_hr_assistant.domain.models import DocumentChunk, KnowledgeBaseStatus, RetrievedChunk
from ai_hr_assistant.services.document_loader import load_documents, split_text

logger = logging.getLogger(__name__)
STOP_WORDS = {
    "а",
    "без",
    "в",
    "во",
    "для",
    "и",
    "или",
    "как",
    "когда",
    "ли",
    "можно",
    "на",
    "не",
    "но",
    "о",
    "по",
    "с",
    "со",
    "что",
    "это",
}

QUERY_TERM_EXPANSIONS = {
    "отпуск": {"отпуск", "отгул", "absence"},
    "больнич": {"больнич", "нетрудоспособ", "болезн", "забол"},
    "забол": {"забол", "больнич", "нетрудоспособ", "болезн"},
    "справк": {"справк", "документ", "копи"},
    "удален": {"удален", "удалён", "дистанцион", "гибрид", "из дома"},
    "удалён": {"удален", "удалён", "дистанцион", "гибрид", "из дома"},
    "отгул": {"отгул", "отсутств", "timeoff"},
    "график": {"график", "расписан", "режим"},
    "обучен": {"обучен", "курс", "компенсац"},
    "компенс": {"компенс", "обучен", "расход"},
}


class KnowledgeBase:
    def __init__(self, settings: AppSettings, client: OpenAI | None) -> None:
        self.settings = settings
        self.client = client
        self._chunks: list[DocumentChunk] = []
        self._document_count = 0

    def ensure_index(self, force_rebuild: bool = False) -> KnowledgeBaseStatus:
        if not force_rebuild and self._chunks:
            return KnowledgeBaseStatus(
                ready=True,
                document_count=self._document_count,
                chunk_count=len(self._chunks),
                message="База знаний готова",
            )

        documents = load_documents(self.settings.documents_dir)
        self._document_count = len(documents)
        self._log_document_diagnostics(documents)
        manifest = self._build_manifest(documents)

        if not force_rebuild and self._load_if_current(manifest):
            return KnowledgeBaseStatus(
                ready=bool(self._chunks),
                document_count=self._document_count,
                chunk_count=len(self._chunks),
                message="База знаний готова",
            )

        if not documents:
            self._chunks = []
            self._write_index({"manifest": manifest, "chunks": []})
            return KnowledgeBaseStatus(
                ready=False,
                document_count=0,
                chunk_count=0,
                message="Документы не найдены",
            )

        if not self.client:
            self._chunks = []
            return KnowledgeBaseStatus(
                ready=False,
                document_count=self._document_count,
                chunk_count=0,
                message="Нет OPENAI_API_KEY для построения индекса",
            )

        raw_chunks: list[dict[str, str]] = []
        for document in documents:
            document_chunks = split_text(document.text)
            self._log_chunk_diagnostics(document.title, document_chunks)
            for index, chunk_text in enumerate(document_chunks, start=1):
                raw_chunks.append(
                    {
                        "chunk_id": f"{document.document_id}-{index}",
                        "document_id": document.document_id,
                        "title": document.title,
                        "source_path": document.source_path,
                        "text": chunk_text,
                    }
                )

        embeddings = self._embed_texts([item["text"] for item in raw_chunks])
        if self.settings.rag_debug:
            logger.info(
                "RAG embeddings created: chunks=%s embeddings=%s",
                len(raw_chunks),
                len(embeddings),
            )
        self._chunks = [
            DocumentChunk(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                title=item["title"],
                source_path=item["source_path"],
                text=item["text"],
                embedding=embedding,
            )
            for item, embedding in zip(raw_chunks, embeddings, strict=True)
        ]

        self._write_index(
            {
                "manifest": manifest,
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "source_path": chunk.source_path,
                        "text": chunk.text,
                        "embedding": chunk.embedding,
                    }
                    for chunk in self._chunks
                ],
            }
        )

        return KnowledgeBaseStatus(
            ready=True,
            document_count=self._document_count,
            chunk_count=len(self._chunks),
            message="База знаний обновлена",
        )

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not query.strip() or not self._chunks or not self.client:
            return []

        query_embedding = self._embed_texts([query])[0]
        query_terms = self._extract_query_terms(query)
        scored_chunks = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                source_path=chunk.source_path,
                text=chunk.text,
                score=self._score_chunk(query_embedding, query_terms, chunk),
            )
            for chunk in self._chunks
        ]
        scored_chunks.sort(key=lambda item: item.score, reverse=True)
        top_chunks = scored_chunks[:top_k]
        if self.settings.rag_debug:
            for index, chunk in enumerate(top_chunks, start=1):
                logger.info(
                    "RAG retrieval #%s: score=%.4f title=%s source=%s snippet=%s",
                    index,
                    chunk.score,
                    chunk.title,
                    chunk.source_path,
                    " ".join(chunk.text.split())[:220],
                )
        return top_chunks

    def _load_if_current(self, manifest: str) -> bool:
        if not self.settings.index_file.exists():
            return False

        payload = json.loads(self.settings.index_file.read_text(encoding="utf-8"))
        if payload.get("manifest") != manifest:
            return False

        self._chunks = [
            DocumentChunk(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                title=item["title"],
                source_path=item["source_path"],
                text=item["text"],
                embedding=item["embedding"],
            )
            for item in payload.get("chunks", [])
        ]
        return True

    def _build_manifest(self, documents: list) -> str:
        source = "|".join(
            f"{document.source_path}:{document.checksum}"
            for document in sorted(documents, key=lambda item: item.source_path)
        )
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.client:
            return [[] for _ in texts]

        vectors: list[list[float]] = []
        batch_size = 32

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=batch,
            )
            vectors.extend([item.embedding for item in response.data])

        return vectors

    def _log_document_diagnostics(self, documents: list) -> None:
        if not self.settings.rag_debug:
            return

        logger.info("RAG document loading: documents=%s", len(documents))
        for document in documents:
            logger.info(
                "RAG document loaded: title=%s chars=%s source=%s snippet=%s",
                document.title,
                len(document.text),
                document.source_path,
                " ".join(document.text.split())[:180],
            )

    def _log_chunk_diagnostics(self, title: str, chunks: list[str]) -> None:
        if not self.settings.rag_debug:
            return

        logger.info("RAG chunking: title=%s chunks=%s", title, len(chunks))
        for index, chunk in enumerate(chunks[:3], start=1):
            logger.info(
                "RAG chunk sample #%s: title=%s chars=%s snippet=%s",
                index,
                title,
                len(chunk),
                " ".join(chunk.split())[:220],
            )

    def _write_index(self, payload: dict) -> None:
        self.settings.index_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0

        dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))

        if left_norm == 0 or right_norm == 0:
            return 0.0

        return dot_product / (left_norm * right_norm)

    def _score_chunk(
        self,
        query_embedding: list[float],
        query_terms: set[str],
        chunk: DocumentChunk,
    ) -> float:
        semantic_score = self._cosine_similarity(query_embedding, chunk.embedding)
        lexical_score = self._lexical_score(query_terms, chunk.title, chunk.text)
        return min(1.0, semantic_score + lexical_score)

    @staticmethod
    def _extract_query_terms(query: str) -> set[str]:
        terms: set[str] = set()
        for token in re.findall(r"\w+", query.lower()):
            if len(token) < 4 or token in STOP_WORDS:
                continue
            terms.add(token)
            if len(token) >= 5:
                terms.add(token[:5])
            if len(token) >= 6:
                terms.add(token[:6])

        expanded_terms = set(terms)
        for term in terms:
            for stem, expansions in QUERY_TERM_EXPANSIONS.items():
                if stem in term:
                    expanded_terms.update(expansions)

        for term in list(expanded_terms):
            if len(term) >= 5:
                expanded_terms.add(term[:5])
            if len(term) >= 6:
                expanded_terms.add(term[:6])
        return expanded_terms

    @staticmethod
    def _lexical_score(query_terms: set[str], title: str, text: str) -> float:
        if not query_terms:
            return 0.0

        title_text = title.lower()
        body_text = text.lower()
        title_hits = sum(1 for term in query_terms if term in title_text)
        body_hits = sum(1 for term in query_terms if term in body_text)
        return min(0.24, (title_hits * 0.08) + (body_hits * 0.03))
