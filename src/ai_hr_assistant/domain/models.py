from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Document:
    document_id: str
    title: str
    source_path: str
    text: str
    checksum: str


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    title: str
    source_path: str
    text: str
    embedding: list[float]


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    title: str
    source_path: str
    text: str
    score: float


@dataclass(slots=True)
class SourceReference:
    title: str
    source_path: str
    snippet: str
    score: float


@dataclass(slots=True)
class AssistantAnswer:
    text: str
    sources: list[SourceReference]
    is_fallback: bool
    confidence: float
    response_kind: str = "answer"
    show_handoff: bool = False


@dataclass(slots=True)
class AssistantStreamEvent:
    delta: str = ""
    answer: AssistantAnswer | None = None


@dataclass(slots=True)
class KnowledgeBaseStatus:
    ready: bool
    document_count: int
    chunk_count: int
    message: str
