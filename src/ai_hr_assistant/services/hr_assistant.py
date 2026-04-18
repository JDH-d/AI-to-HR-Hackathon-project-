from __future__ import annotations

import logging
import re
from collections.abc import Iterator

from openai import OpenAI, OpenAIError

from ai_hr_assistant.config import AppSettings
from ai_hr_assistant.domain.models import (
    AssistantAnswer,
    AssistantStreamEvent,
    KnowledgeBaseStatus,
    SourceReference,
)
from ai_hr_assistant.infrastructure.openai_factory import build_openai_client
from ai_hr_assistant.services.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

ROUTE_GREETING = "greeting"
ROUTE_HR = "hr_request"
ROUTE_OFFTOPIC = "offtopic"
MIN_HR_CONFIDENT_ANSWER_SCORE = 0.4

SYSTEM_PROMPT = """
Ты AI HR Assistant для сотрудников компании.
Отвечай только по переданным фрагментам внутренних документов.
Не придумывай политики, сроки, суммы, исключения или процессы, которых нет в контексте.
Если в фрагментах есть релевантные шаги, условия, каналы обращения, список документов, сроки или ответственные роли,
собери из них практичный ответ для сотрудника.
Возвращай ровно INSUFFICIENT_CONTEXT только если фрагменты реально не относятся к вопросу
или в них недостаточно данных даже для краткого полезного ответа по документам.
Пиши по-русски, кратко, уверенно, простыми словами и без канцелярита.
Начинай сразу с сути, не повторяй вопрос пользователя.
Если уместно, давай 2-5 коротких пунктов с конкретными действиями.
Не используй фразы вроде "из имеющихся фрагментов", "согласно предоставленным документам",
"можно выделить следующие моменты" и похожие бюрократические формулировки.
Не вставляй в основной текст названия файлов, номера разделов и служебные ссылки на документы:
источники будут показаны отдельно.
Не акцентируй неполноту данных и не добавляй разделы вроде "Что не указано".
Не пиши, что "в документе сказано", "в документе упоминается", "в источниках указано" или "данные неполные",
если уже можно дать полезный и внятный ответ по сути вопроса.
""".strip()

FORMAL_INTRO_PATTERNS = (
    re.compile(
        r"^из имеющихся фрагментов(?: документов)? можно собрать следующие(?: практические)? моменты:\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^согласно (?:предоставленным|имеющимся|найденным) (?:фрагментам|документам)(?: документов)?[,:]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^из (?:предоставленных|найденных|имеющихся) фрагментов(?: документов)? (?:следует|видно),? что\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^можно (?:выделить|собрать) следующие(?: практические)? моменты:\s*",
        re.IGNORECASE,
    ),
)
SOURCE_REFERENCE_PATTERN = re.compile(
    r"\s*\((?=[^)]*\.(?:docx|pdf|txt|md)\b)[^)]*\)",
    re.IGNORECASE,
)
SECTION_TO_REMOVE_PATTERNS = (
    re.compile(
        r"(?:\n\s*)?(?:что именно не хватает в фрагментах|чего не хватает в фрагментах|"
        r"чего не хватает в документах|что не указано)\s*:\s*(?:\n(?:- .+|• .+|[0-9]+\.\s+.+))+",
        re.IGNORECASE,
    ),
)
LEADING_LABEL_PATTERNS = (
    re.compile(r"^ответ:\s*", re.IGNORECASE),
)
SECTION_HEADING_PATTERNS = (
    re.compile(r"^коротко про действия:\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^что делать:\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^порядок действий:\s*$", re.IGNORECASE | re.MULTILINE),
)
HEDGING_PHRASES = (
    (re.compile(r"\bв документе (?:не )?указано,? что\b", re.IGNORECASE), ""),
    (re.compile(r"\bв документе упоминается\b", re.IGNORECASE), ""),
    (re.compile(r"\bв источниках указано,? что\b", re.IGNORECASE), ""),
    (re.compile(r"\bв документах указано,? что\b", re.IGNORECASE), ""),
    (re.compile(r"\bв документах сказано,? что\b", re.IGNORECASE), ""),
    (re.compile(r"\bпо документам\b", re.IGNORECASE), ""),
)
SECTION_HEADINGS_TO_DROP = (
    "коротко про действия",
    "что делать",
    "порядок действий",
)

FALLBACK_MESSAGE = (
    "\u042f \u043d\u0435 \u0441\u043c\u043e\u0433 \u0443\u0432\u0435\u0440\u0435\u043d\u043d\u043e \u043e\u0442\u0432\u0435\u0442\u0438\u0442\u044c "
    "\u043f\u043e \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u043c \u043a\u043e\u043c\u043f\u0430\u043d\u0438\u0438. "
    "\u041b\u0443\u0447\u0448\u0435 \u043f\u0435\u0440\u0435\u0434\u0430\u0442\u044c \u044d\u0442\u043e\u0442 \u0432\u043e\u043f\u0440\u043e\u0441 HR "
    "\u0434\u043b\u044f \u0442\u043e\u0447\u043d\u043e\u0433\u043e \u043e\u0442\u0432\u0435\u0442\u0430."
)

GREETING_MESSAGE = (
    "\u041f\u0440\u0438\u0432\u0435\u0442. \u042f \u043f\u043e\u043c\u043e\u0433\u0430\u044e \u0441 "
    "\u0432\u043d\u0443\u0442\u0440\u0435\u043d\u043d\u0438\u043c\u0438 HR-\u0432\u043e\u043f\u0440\u043e\u0441\u0430\u043c\u0438 "
    "\u043a\u043e\u043c\u043f\u0430\u043d\u0438\u0438: \u043e\u0442\u043f\u0443\u0441\u043a, "
    "\u0431\u043e\u043b\u044c\u043d\u0438\u0447\u043d\u044b\u0439, \u0441\u043f\u0440\u0430\u0432\u043a\u0438, "
    "\u0443\u0434\u0430\u043b\u0435\u043d\u043a\u0430, \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u0435. "
    "\u041d\u0430\u043f\u0438\u0448\u0438\u0442\u0435, \u0447\u0442\u043e \u0438\u043c\u0435\u043d\u043d\u043e \u043d\u0443\u0436\u043d\u043e."
)

OFFTOPIC_MESSAGE = (
    "\u042f \u043f\u043e\u043c\u043e\u0433\u0430\u044e \u0442\u043e\u043b\u044c\u043a\u043e \u0441 "
    "\u0432\u043d\u0443\u0442\u0440\u0435\u043d\u043d\u0438\u043c\u0438 HR-\u0432\u043e\u043f\u0440\u043e\u0441\u0430\u043c\u0438 "
    "\u043a\u043e\u043c\u043f\u0430\u043d\u0438\u0438. \u041c\u043e\u0433\u0443 \u043f\u043e\u0434\u0441\u043a\u0430\u0437\u0430\u0442\u044c "
    "\u043f\u043e \u043e\u0442\u043f\u0443\u0441\u043a\u0430\u043c, \u0431\u043e\u043b\u044c\u043d\u0438\u0447\u043d\u044b\u043c, "
    "\u0441\u043f\u0440\u0430\u0432\u043a\u0430\u043c, \u0443\u0434\u0430\u043b\u0435\u043d\u043a\u0435, "
    "\u043e\u0442\u0433\u0443\u043b\u0430\u043c, \u0433\u0440\u0430\u0444\u0438\u043a\u0443 \u0438 "
    "\u043e\u0431\u0443\u0447\u0435\u043d\u0438\u044e."
)

SERVICE_UNAVAILABLE_MESSAGE = (
    "\u0421\u0435\u0439\u0447\u0430\u0441 \u043d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u043f\u043e\u043b\u0443\u0447\u0438\u0442\u044c "
    "\u043e\u0442\u0432\u0435\u0442 \u043e\u0442 OpenAI API. \u041f\u043e\u043f\u0440\u043e\u0431\u0443\u0439\u0442\u0435 "
    "\u043f\u043e\u0432\u0442\u043e\u0440\u0438\u0442\u044c \u0437\u0430\u043f\u0440\u043e\u0441 \u0447\u0435\u0440\u0435\u0437 "
    "\u043c\u0438\u043d\u0443\u0442\u0443 \u0438\u043b\u0438 \u043f\u0435\u0440\u0435\u0434\u0430\u0439\u0442\u0435 "
    "\u0432\u043e\u043f\u0440\u043e\u0441 HR."
)

NO_API_KEY_MESSAGE = (
    "\u041f\u0440\u0438\u043b\u043e\u0436\u0435\u043d\u0438\u0435 \u043d\u0435 \u043c\u043e\u0436\u0435\u0442 "
    "\u043e\u0431\u0440\u0430\u0442\u0438\u0442\u044c\u0441\u044f \u043a OpenAI \u0431\u0435\u0437 `OPENAI_API_KEY`. "
    "\u0414\u043e\u0431\u0430\u0432\u044c\u0442\u0435 \u043a\u043b\u044e\u0447 \u0432 `.env`, \u0437\u0430\u0442\u0435\u043c "
    "\u043f\u043e\u0432\u0442\u043e\u0440\u0438\u0442\u0435 \u0437\u0430\u043f\u0440\u043e\u0441."
)

NO_INDEX_API_KEY_MESSAGE = (
    "\u041f\u0440\u0438\u043b\u043e\u0436\u0435\u043d\u0438\u0435 \u043d\u0435 \u043c\u043e\u0436\u0435\u0442 "
    "\u043f\u043e\u0441\u0442\u0440\u043e\u0438\u0442\u044c \u0431\u0430\u0437\u0443 \u0437\u043d\u0430\u043d\u0438\u0439 "
    "\u0431\u0435\u0437 `OPENAI_API_KEY`. \u0414\u043e\u0431\u0430\u0432\u044c\u0442\u0435 \u043a\u043b\u044e\u0447 "
    "\u0432 `.env`, \u0437\u0430\u0442\u0435\u043c \u043f\u0435\u0440\u0435\u0438\u043d\u0434\u0435\u043a\u0441\u0438\u0440\u0443\u0439\u0442\u0435 "
    "\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u044b."
)

NO_DOCUMENTS_MESSAGE = (
    "\u0412 \u0431\u0430\u0437\u0435 \u0437\u043d\u0430\u043d\u0438\u0439 \u043f\u043e\u043a\u0430 \u043d\u0435\u0442 "
    "\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u043e\u0432. \u0417\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u0435 "
    "HR-\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u044b \u0438 \u043f\u043e\u0432\u0442\u043e\u0440\u0438\u0442\u0435 "
    "\u0432\u043e\u043f\u0440\u043e\u0441."
)

GREETING_PHRASES = {
    "привет",
    "здравствуй",
    "здравствуйте",
    "добрый день",
    "доброе утро",
    "добрый вечер",
    "день добрый",
    "помоги",
    "помоги пожалуйста",
    "нужна помощь",
    "помощь",
}

GREETING_TOKENS = {
    "привет",
    "здравствуйте",
    "здравствуй",
    "добрый",
    "день",
    "вечер",
    "утро",
    "помоги",
    "помощь",
    "нужна",
    "нужно",
    "пожалуйста",
}

HR_KEYWORD_STEMS = {
    "hr",
    "эйчар",
    "кадр",
    "отпуск",
    "отгул",
    "больнич",
    "забол",
    "боле",
    "справк",
    "документ",
    "удален",
    "удалён",
    "дистан",
    "гибрид",
    "график",
    "обуч",
    "компенс",
    "льгот",
    "дмс",
    "полис",
    "декрет",
    "беремен",
    "родител",
    "оклад",
    "зарплат",
    "аванс",
    "workday",
}

HR_CONTEXT_STEMS = {
    "работ",
    "компан",
    "сотруд",
    "офис",
    "заяв",
    "оформ",
    "соглас",
    "уведом",
}

GENERIC_QUESTION_STEMS = {
    "как",
    "куда",
    "какой",
    "какие",
    "какая",
    "какое",
    "кто",
    "что",
    "если",
    "именно",
    "нужн",
    "можно",
    "писа",
    "оформ",
    "дела",
    "моем",
    "случ",
    "моег",
    "этот",
    "этом",
}
STRICT_DETAIL_PATTERNS = (
    re.compile(r"\bкакие документы\b", re.IGNORECASE),
    re.compile(r"\bкто(?:\s+именно)?\s+(?:согласует|утверждает|одобряет)\b", re.IGNORECASE),
    re.compile(r"\bкуда(?:\s+именно)?\s+(?:писать|отправлять|подавать)\b", re.IGNORECASE),
    re.compile(r"\bкакой(?:\s+именно)?\s+(?:канал|формат|адрес|маршрут)\b", re.IGNORECASE),
    re.compile(r"\bесли\b", re.IGNORECASE),
)
PERSONAL_CASE_PATTERNS = (
    re.compile(r"\bв моем случае\b", re.IGNORECASE),
    re.compile(r"\bв моей ситуации\b", re.IGNORECASE),
    re.compile(r"\bименно для меня\b", re.IGNORECASE),
)
CHANNEL_REQUEST_PATTERNS = (
    re.compile(r"\bкуда(?:\s+именно)?\s+(?:писать|сообщать|отправлять|подавать)\b", re.IGNORECASE),
    re.compile(r"\bкакой(?:\s+именно)?\s+канал\b", re.IGNORECASE),
)
DOCUMENT_REQUEST_PATTERNS = (
    re.compile(r"\bкакие документы\b", re.IGNORECASE),
    re.compile(r"\bкакой пакет документов\b", re.IGNORECASE),
)
APPROVAL_REQUEST_PATTERNS = (
    re.compile(r"\bкто(?:\s+именно)?\s+(?:согласует|утверждает|одобряет)\b", re.IGNORECASE),
    re.compile(r"\bчье(?:\s+именно)?\s+согласование\b", re.IGNORECASE),
)
CHANNEL_EVIDENCE_STEMS = {
    "канал",
    "мессендж",
    "звон",
    "почт",
    "hr-сервис",
    "hr сервис",
    "уведом",
    "напис",
    "отправ",
}
DOCUMENT_EVIDENCE_STEMS = {
    "документ",
    "сведени",
    "данны",
    "список",
    "переч",
    "комплект",
    "предостав",
    "справк",
}
APPROVAL_EVIDENCE_STEMS = {
    "соглас",
    "утверж",
    "одобр",
    "руковод",
    "hr",
    "финанс",
    "l&d",
}
GENERIC_CHUNK_MARKERS = (
    "внутренний hr-регламент компании",
    "1. общие положения",
    "2. цели и принципы процесса",
    "6. роль руководителя",
    "7. роль hr и смежных функций",
    "8. сроки, сервисные ожидания и контроль статуса",
    "9. исключения и пограничные сценарии",
    "11. актуализация документа",
)

OFFTOPIC_KEYWORD_STEMS = {
    "погод",
    "дракон",
    "шахмат",
    "футбол",
    "хокке",
    "новост",
    "полит",
    "гороскоп",
    "астрол",
    "рецепт",
    "фильм",
    "сериал",
    "музык",
    "песн",
    "анекдот",
    "биткоин",
    "крипт",
    "акци",
    "бирж",
}


class HRAssistantService:
    def __init__(
        self,
        settings: AppSettings,
        client: OpenAI | None,
        knowledge_base: KnowledgeBase,
    ) -> None:
        self.settings = settings
        self.client = client
        self.knowledge_base = knowledge_base

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "HRAssistantService":
        client = build_openai_client(settings)
        knowledge_base = KnowledgeBase(settings=settings, client=client)
        return cls(settings=settings, client=client, knowledge_base=knowledge_base)

    def ensure_knowledge_base(self, force_rebuild: bool = False) -> KnowledgeBaseStatus:
        return self.knowledge_base.ensure_index(force_rebuild=force_rebuild)

    def ask(self, question: str) -> AssistantAnswer:
        final_answer: AssistantAnswer | None = None
        for event in self.ask_stream(question):
            if event.answer is not None:
                final_answer = event.answer

        if final_answer is None:
            return self._service_error_answer(confidence=0.0, sources=[])

        return final_answer

    def ask_stream(self, question: str) -> Iterator[AssistantStreamEvent]:
        question = question.strip()
        route = self._route_question(question)
        question_looks_hr = self._looks_like_hr_question(question)

        if route == ROUTE_GREETING:
            yield AssistantStreamEvent(answer=self._greeting_answer())
            return

        if route == ROUTE_OFFTOPIC:
            yield AssistantStreamEvent(answer=self._offtopic_answer())
            return

        if not self.client:
            yield AssistantStreamEvent(
                answer=AssistantAnswer(
                    text=NO_API_KEY_MESSAGE,
                    sources=[],
                    is_fallback=True,
                    confidence=0.0,
                    response_kind="service_error",
                )
            )
            return

        try:
            status = self.ensure_knowledge_base()
        except OpenAIError:
            yield AssistantStreamEvent(answer=self._service_error_answer(confidence=0.0, sources=[]))
            return

        if not status.ready:
            yield AssistantStreamEvent(
                answer=AssistantAnswer(
                    text=self._status_based_message(status),
                    sources=[],
                    is_fallback=True,
                    confidence=0.0,
                    response_kind="status",
                )
            )
            return

        try:
            retrieved_chunks = self.knowledge_base.search(
                query=question,
                top_k=self.settings.top_k_sources,
            )
        except OpenAIError:
            yield AssistantStreamEvent(answer=self._service_error_answer(confidence=0.0, sources=[]))
            return

        if not retrieved_chunks:
            yield AssistantStreamEvent(
                answer=self._fallback_for_question(
                    question=question,
                    confidence=0.0,
                    sources=[],
                )
            )
            return

        top_score = retrieved_chunks[0].score
        confidence = max(0.0, min(1.0, top_score))
        sources = self._build_sources(question, retrieved_chunks)
        if self.settings.rag_debug:
            logger.info(
                "RAG fallback gate: query=%s top_score=%.4f threshold=%.4f top_sources=%s",
                question,
                top_score,
                self.settings.similarity_threshold,
                [source.title for source in sources],
            )

        if top_score < self.settings.similarity_threshold:
            yield AssistantStreamEvent(
                answer=self._fallback_for_question(
                    question=question,
                    confidence=confidence,
                    sources=sources,
                )
            )
            return

        if question_looks_hr and top_score < MIN_HR_CONFIDENT_ANSWER_SCORE:
            yield AssistantStreamEvent(
                answer=self._fallback_answer(confidence=confidence, sources=sources)
            )
            return

        if not self._has_sufficient_question_coverage(question, retrieved_chunks):
            if self.settings.rag_debug:
                logger.info("RAG coverage gate rejected answer: query=%s", question)
            yield AssistantStreamEvent(
                answer=self._fallback_for_question(
                    question=question,
                    confidence=confidence,
                    sources=sources,
                )
            )
            return

        try:
            answer_text = ""
            for delta in self._generate_grounded_answer_stream(question, retrieved_chunks):
                answer_text += delta
                yield AssistantStreamEvent(delta=delta)
        except OpenAIError:
            yield AssistantStreamEvent(
                answer=self._service_error_answer(confidence=confidence, sources=sources)
            )
            return

        if not answer_text or answer_text.strip() == "INSUFFICIENT_CONTEXT":
            if self.settings.rag_debug:
                logger.info("RAG model returned insufficient context: query=%s", question)
            yield AssistantStreamEvent(
                answer=self._fallback_for_question(
                    question=question,
                    confidence=confidence,
                    sources=sources,
                )
            )
            return

        answer_text = self._polish_answer_text(question, answer_text)
        if not answer_text:
            yield AssistantStreamEvent(
                answer=self._fallback_for_question(
                    question=question,
                    confidence=confidence,
                    sources=sources,
                )
            )
            return

        yield AssistantStreamEvent(
            answer=AssistantAnswer(
                text=answer_text,
                sources=sources,
                is_fallback=False,
                confidence=confidence,
                response_kind="answer",
            )
        )

    def _generate_grounded_answer_stream(
        self,
        question: str,
        retrieved_chunks: list,
    ) -> Iterator[str]:
        if not self.client:
            return

        user_prompt = self._build_user_prompt(question, retrieved_chunks)
        request_kwargs = {
            "model": self.settings.chat_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
        }
        if self.settings.chat_model.startswith("gpt-5"):
            request_kwargs["reasoning_effort"] = "minimal"

        response = self._create_chat_completion_stream(request_kwargs)

        for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

    def _build_user_prompt(self, question: str, retrieved_chunks: list) -> str:
        context_blocks = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            context_blocks.append(
                f"[Фрагмент {index}: {chunk.title}]\n"
                f"Релевантность: {chunk.score:.2f}\n"
                f"{chunk.text}"
            )

        context = "\n\n".join(context_blocks)
        if self.settings.rag_debug:
            logger.info(
                "RAG prompt assembly: query=%s context_chars=%s chunks=%s",
                question,
                len(context),
                len(retrieved_chunks),
            )

        return (
            "Ниже приведены фрагменты внутренних HR-документов компании.\n\n"
            f"{context}\n\n"
            f"Вопрос сотрудника: {question}\n\n"
            "Сформируй ответ только по этим фрагментам.\n"
            "Объясни коротко, уверенно и простыми словами, как коллеге без HR-бэкграунда.\n"
            "Начни сразу с ответа и не повторяй вопрос пользователя.\n"
            "Если можно, оформи ответ в 2-5 коротких пунктов с конкретными действиями.\n"
            "Дай цельный ответ так, будто это уже рабочий внутренний помощник, а не анализ фрагментов.\n"
            "Не подчеркивай, что данных мало или что чего-то не хватает, если по фрагментам можно ответить по сути.\n"
            "Если вопрос просит точные детали, а во фрагментах есть только общий процесс без прямого покрытия этого кейса, "
            "верни только INSUFFICIENT_CONTEXT.\n"
            "Не придумывай конкретные документы, каналы, роли, сроки, согласующих или исключения, "
            "если они не подтверждены фрагментами.\n"
            "Не используй бюрократические вводные фразы вроде 'из имеющихся фрагментов' "
            "или 'согласно предоставленным документам'.\n"
            "Не вставляй в основной текст названия файлов, номера разделов и ссылки на документы.\n"
            "Не добавляй блоки 'Что не указано', 'Чего не хватает' и похожие разделы.\n"
            "Верни только INSUFFICIENT_CONTEXT, если фрагменты действительно не помогают ответить."
        )

    def _build_sources(self, question: str, retrieved_chunks: list) -> list[SourceReference]:
        query_terms = self._source_query_terms(question)
        best_chunks_by_title: dict[str, tuple[tuple[int, float], object]] = {}

        for chunk in retrieved_chunks:
            priority = (self._count_term_hits(chunk.text, query_terms), chunk.score)
            current = best_chunks_by_title.get(chunk.title)
            if current is None or priority > current[0]:
                best_chunks_by_title[chunk.title] = (priority, chunk)

        ranked_chunks = sorted(
            (item[1] for item in best_chunks_by_title.values()),
            key=lambda chunk: chunk.score,
            reverse=True,
        )

        return [
            SourceReference(
                title=chunk.title,
                source_path=chunk.source_path,
                snippet=self._build_source_snippet(chunk.text, query_terms),
                score=chunk.score,
            )
            for chunk in ranked_chunks
        ]

    def _fallback_answer(
        self,
        confidence: float,
        sources: list[SourceReference],
    ) -> AssistantAnswer:
        return AssistantAnswer(
            text=FALLBACK_MESSAGE,
            sources=sources,
            is_fallback=True,
            confidence=confidence,
            response_kind="hr_fallback",
            show_handoff=True,
        )

    def _service_error_answer(
        self,
        confidence: float,
        sources: list[SourceReference],
    ) -> AssistantAnswer:
        return AssistantAnswer(
            text=SERVICE_UNAVAILABLE_MESSAGE,
            sources=sources,
            is_fallback=True,
            confidence=confidence,
            response_kind="service_error",
        )

    def _fallback_for_question(
        self,
        *,
        question: str,
        confidence: float,
        sources: list[SourceReference],
    ) -> AssistantAnswer:
        if self._route_question(question) == ROUTE_HR or self._looks_like_hr_question(question):
            return self._fallback_answer(confidence=confidence, sources=sources)
        return self._offtopic_answer()

    @staticmethod
    def _status_based_message(status: KnowledgeBaseStatus) -> str:
        if "OPENAI_API_KEY" in status.message:
            return NO_INDEX_API_KEY_MESSAGE
        if status.document_count == 0:
            return NO_DOCUMENTS_MESSAGE
        return FALLBACK_MESSAGE

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return re.findall(r"[a-zа-я0-9]+", cls._normalize_text(text))

    @classmethod
    def _route_question(cls, question: str) -> str:
        if cls._is_greeting(question):
            return ROUTE_GREETING
        if cls._is_clear_offtopic(question):
            return ROUTE_OFFTOPIC
        return ROUTE_HR

    @classmethod
    def _is_greeting(cls, question: str) -> bool:
        normalized = cls._normalize_text(question)
        if not normalized:
            return True
        if normalized in GREETING_PHRASES:
            return True

        tokens = cls._tokenize(question)
        return bool(tokens) and len(tokens) <= 4 and all(token in GREETING_TOKENS for token in tokens)

    @classmethod
    def _looks_like_hr_question(cls, question: str) -> bool:
        normalized = cls._normalize_text(question)
        tokens = cls._tokenize(question)
        if not normalized or not tokens:
            return False

        hr_score = sum(
            2 for token in tokens if any(stem in token for stem in HR_KEYWORD_STEMS)
        )
        hr_score += sum(
            1 for token in tokens if any(stem in token for stem in HR_CONTEXT_STEMS)
        )

        phrase_signals = (
            "с работы",
            "с места работы",
            "из дома",
            "по болезни",
            "на удаленке",
            "на удалёнке",
        )
        if any(phrase in normalized for phrase in phrase_signals):
            hr_score += 1

        return hr_score >= 2

    @classmethod
    def _question_requires_strict_coverage(cls, question: str) -> bool:
        normalized = cls._normalize_text(question)
        if any(pattern.search(normalized) for pattern in PERSONAL_CASE_PATTERNS):
            return True
        return any(pattern.search(normalized) for pattern in STRICT_DETAIL_PATTERNS)

    @classmethod
    def _question_intents(cls, question: str) -> set[str]:
        normalized = cls._normalize_text(question)
        intents: set[str] = set()
        if any(pattern.search(normalized) for pattern in CHANNEL_REQUEST_PATTERNS):
            intents.add("channel")
        if any(pattern.search(normalized) for pattern in DOCUMENT_REQUEST_PATTERNS):
            intents.add("documents")
        if any(pattern.search(normalized) for pattern in APPROVAL_REQUEST_PATTERNS):
            intents.add("approval")
        return intents

    @classmethod
    def _extract_specific_question_terms(cls, question: str) -> list[str]:
        terms: list[str] = []
        for token in cls._tokenize(question):
            if len(token) < 4:
                continue
            if any(stem in token for stem in GENERIC_QUESTION_STEMS):
                continue
            if token not in terms:
                terms.append(token)
        return terms

    @classmethod
    def _question_terms_supported(cls, question: str, retrieved_chunks: list) -> tuple[list[str], list[str]]:
        combined_text = cls._normalize_text(
            " ".join(f"{chunk.title} {chunk.text}" for chunk in retrieved_chunks)
        )
        supported_terms: list[str] = []
        missing_terms: list[str] = []
        for term in cls._extract_specific_question_terms(question):
            if term in combined_text:
                supported_terms.append(term)
                continue
            if len(term) >= 5 and term[:5] in combined_text:
                supported_terms.append(term)
                continue
            missing_terms.append(term)
        return supported_terms, missing_terms

    @classmethod
    def _chunk_is_generic(cls, text: str) -> bool:
        normalized = cls._normalize_text(text)
        return any(marker in normalized for marker in GENERIC_CHUNK_MARKERS)

    @classmethod
    def _intents_have_evidence(cls, intents: set[str], retrieved_chunks: list) -> bool:
        if not intents:
            return True

        combined_text = cls._normalize_text(
            " ".join(f"{chunk.title} {chunk.text}" for chunk in retrieved_chunks)
        )
        evidence_by_intent = {
            "channel": CHANNEL_EVIDENCE_STEMS,
            "documents": DOCUMENT_EVIDENCE_STEMS,
            "approval": APPROVAL_EVIDENCE_STEMS,
        }
        return all(
            any(stem in combined_text for stem in evidence_by_intent[intent])
            for intent in intents
        )

    @classmethod
    def _has_sufficient_question_coverage(cls, question: str, retrieved_chunks: list) -> bool:
        if not retrieved_chunks:
            return False

        supported_terms, missing_terms = cls._question_terms_supported(question, retrieved_chunks)
        strict_coverage = cls._question_requires_strict_coverage(question)
        intents = cls._question_intents(question)
        top_chunks = retrieved_chunks[:2]
        generic_top_chunks = sum(1 for chunk in top_chunks if cls._chunk_is_generic(chunk.text))

        if any(pattern.search(cls._normalize_text(question)) for pattern in PERSONAL_CASE_PATTERNS):
            return False

        if strict_coverage and missing_terms:
            return False

        if strict_coverage and not cls._intents_have_evidence(intents, retrieved_chunks):
            return False

        if strict_coverage and generic_top_chunks == len(top_chunks) and len(supported_terms) < 2:
            return False

        if not strict_coverage and not supported_terms and generic_top_chunks == len(top_chunks):
            return False

        return True

    @classmethod
    def _is_clear_offtopic(cls, question: str) -> bool:
        normalized = cls._normalize_text(question)
        if not normalized:
            return False
        if cls._looks_like_hr_question(question):
            return False

        tokens = cls._tokenize(question)
        if any(any(stem in token for stem in OFFTOPIC_KEYWORD_STEMS) for token in tokens):
            return True

        if re.search(r"\d+\s*[\+\-\*/]\s*\d+", normalized):
            return True

        opinion_patterns = (
            "что ты думаешь",
            "как ты относишься",
            "расскажи анекдот",
        )
        return any(pattern in normalized for pattern in opinion_patterns)

    @staticmethod
    def _greeting_answer() -> AssistantAnswer:
        return AssistantAnswer(
            text=GREETING_MESSAGE,
            sources=[],
            is_fallback=False,
            confidence=1.0,
            response_kind="greeting",
        )

    @staticmethod
    def _offtopic_answer() -> AssistantAnswer:
        return AssistantAnswer(
            text=OFFTOPIC_MESSAGE,
            sources=[],
            is_fallback=False,
            confidence=1.0,
            response_kind="offtopic",
        )

    def _create_chat_completion_stream(self, request_kwargs: dict) -> object:
        try:
            return self.client.chat.completions.create(**request_kwargs)
        except TypeError as error:
            if "reasoning_effort" not in request_kwargs:
                raise
            logger.info("Chat completions API does not support reasoning_effort, retrying without it")
            fallback_kwargs = dict(request_kwargs)
            fallback_kwargs.pop("reasoning_effort", None)
            return self.client.chat.completions.create(**fallback_kwargs)

    @classmethod
    def _source_query_terms(cls, question: str) -> list[str]:
        terms = []
        for token in cls._tokenize(question):
            if len(token) < 4:
                continue
            variants = [token]
            if len(token) >= 5:
                variants.append(token[:5])
            if len(token) >= 6:
                variants.append(token[:6])
            for variant in variants:
                if variant not in terms:
                    terms.append(variant)
        return sorted(terms, key=len, reverse=True)

    @staticmethod
    def _count_term_hits(text: str, query_terms: list[str]) -> int:
        normalized_text = text.lower().replace("ё", "е")
        return sum(normalized_text.count(term) for term in query_terms)

    @staticmethod
    def _build_source_snippet(text: str, query_terms: list[str], window: int = 280) -> str:
        compact_text = " ".join(text.split())
        if not compact_text:
            return ""

        lowered_text = compact_text.lower().replace("ё", "е")
        start = 0
        for term in query_terms:
            position = lowered_text.find(term)
            if position != -1:
                start = max(0, position - 80)
                break

        end = min(len(compact_text), start + window)
        snippet = compact_text[start:end].strip()
        if start > 0:
            snippet = f"...{snippet}"
        if end < len(compact_text):
            snippet = f"{snippet}..."
        return snippet

    @classmethod
    def _polish_answer_text(cls, question: str, text: str) -> str:
        cleaned = text.replace("\r\n", "\n").strip()
        if not cleaned:
            return ""

        lines = [line.strip() for line in cleaned.split("\n")]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()

        normalized_question = cls._normalize_text(question).rstrip("?.!")
        if lines and cls._normalize_text(lines[0]).rstrip("?.!") == normalized_question:
            lines.pop(0)
            while lines and not lines[0]:
                lines.pop(0)

        cleaned = "\n".join(lines)

        for pattern in FORMAL_INTRO_PATTERNS:
            cleaned = pattern.sub("", cleaned, count=1)

        for pattern in LEADING_LABEL_PATTERNS:
            cleaned = pattern.sub("", cleaned, count=1)

        for pattern in SECTION_HEADING_PATTERNS:
            cleaned = pattern.sub("", cleaned)

        cleaned = SOURCE_REFERENCE_PATTERN.sub("", cleaned)

        for pattern in SECTION_TO_REMOVE_PATTERNS:
            cleaned = pattern.sub("", cleaned)

        for pattern, replacement in HEDGING_PHRASES:
            cleaned = pattern.sub(replacement, cleaned)

        cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r" +([,.:;!?])", r"\1", cleaned)
        cleaned = re.sub(r"^[,:;\s]+", "", cleaned)
        cleaned = re.sub(r"\b(?:сказано|упоминается)\s+о\b", "есть", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"(?im)^\s*(?:[-•]\s*)?(?:коротко про действия|что делать|порядок действий)\s*:\s*$\n?",
            "",
            cleaned,
        )

        cleaned_lines = []
        for line in cleaned.split("\n"):
            stripped_line = line.strip()
            normalized_line = cls._normalize_text(stripped_line.lstrip("-• ").strip())
            if any(heading in normalized_line for heading in SECTION_HEADINGS_TO_DROP):
                continue
            cleaned_lines.append(
                line.rstrip(" -") if stripped_line.startswith("- ") else line.rstrip()
            )

        cleaned = "\n".join(cleaned_lines).strip()
        return cleaned
