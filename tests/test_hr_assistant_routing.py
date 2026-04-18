from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_hr_assistant.config import AppSettings
from ai_hr_assistant.domain.models import KnowledgeBaseStatus, RetrievedChunk
from ai_hr_assistant.services.hr_assistant import HRAssistantService
from ai_hr_assistant.services.knowledge_base import KnowledgeBase


class DummyKnowledgeBase:
    def ensure_index(self, force_rebuild: bool = False) -> KnowledgeBaseStatus:
        return KnowledgeBaseStatus(
            ready=True,
            document_count=5,
            chunk_count=20,
            message="ready",
        )

    def search(self, query: str, top_k: int) -> list:
        return []


class RetrievedChunksKnowledgeBase(DummyKnowledgeBase):
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return self._chunks[:top_k]


class FakeChatCompletions:
    def __init__(self, response_text: str = "Готовый ответ") -> None:
        self.calls: list[dict] = []
        self.response_text = response_text

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "reasoning_effort" in kwargs:
            raise TypeError("Completions.create() got an unexpected keyword argument 'reasoning_effort'")
        return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=self.response_text))])]


class FakeClient:
    def __init__(self, response_text: str = "Готовый ответ") -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletions(response_text=response_text))


def build_settings() -> AppSettings:
    return AppSettings(
        app_name="AI HR Assistant",
        openai_api_key="test-key",
        chat_model="gpt-5-nano",
        embedding_model="text-embedding-3-small",
        documents_dir=Path("documents"),
        index_file=Path("data/index/vector_index.json"),
        similarity_threshold=0.33,
        top_k_sources=4,
        rag_debug=False,
        hr_contact_email="hr@example.com",
        hr_contact_label="HR",
        hr_contact_url=None,
    )


class HRAssistantRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assistant = HRAssistantService(
            settings=build_settings(),
            client=object(),
            knowledge_base=DummyKnowledgeBase(),
        )

    def test_greeting_returns_friendly_prompt_without_handoff(self) -> None:
        answer = self.assistant.ask("Добрый день")

        self.assertEqual(answer.response_kind, "greeting")
        self.assertFalse(answer.show_handoff)
        self.assertFalse(answer.is_fallback)

    def test_offtopic_returns_soft_refusal_without_handoff(self) -> None:
        answer = self.assistant.ask("Какая сегодня погода?")

        self.assertEqual(answer.response_kind, "offtopic")
        self.assertFalse(answer.show_handoff)
        self.assertFalse(answer.is_fallback)

    def test_hr_question_without_answer_offers_handoff(self) -> None:
        answer = self.assistant.ask("Мне нужна справка с работы")

        self.assertEqual(answer.response_kind, "hr_fallback")
        self.assertTrue(answer.show_handoff)
        self.assertTrue(answer.is_fallback)

    def test_conversational_hr_phrasing_still_routes_to_hr_fallback(self) -> None:
        answer = self.assistant.ask("Куда писать если заболел?")

        self.assertEqual(answer.response_kind, "hr_fallback")
        self.assertTrue(answer.show_handoff)

    def test_chat_completion_retries_without_reasoning_effort(self) -> None:
        knowledge_base = RetrievedChunksKnowledgeBase(
            [
                RetrievedChunk(
                    chunk_id="doc-1",
                    title="01_sick_leave_policy.docx",
                    source_path="documents/01_sick_leave_policy.docx",
                    text="Если сотрудник заболел, он уведомляет руководителя и HR в день обращения.",
                    score=0.81,
                )
            ]
        )
        client = FakeClient()
        assistant = HRAssistantService(
            settings=build_settings(),
            client=client,
            knowledge_base=knowledge_base,
        )

        answer = assistant.ask("Как оформить больничный?")

        self.assertEqual(answer.response_kind, "answer")
        self.assertFalse(answer.is_fallback)
        self.assertEqual(answer.text, "Готовый ответ")
        self.assertEqual(len(client.chat.completions.calls), 2)
        self.assertIn("reasoning_effort", client.chat.completions.calls[0])
        self.assertNotIn("reasoning_effort", client.chat.completions.calls[1])

    def test_generated_answer_is_simplified_for_end_user(self) -> None:
        knowledge_base = RetrievedChunksKnowledgeBase(
            [
                RetrievedChunk(
                    chunk_id="doc-1",
                    title="01_sick_leave_policy.docx",
                    source_path="documents/01_sick_leave_policy.docx",
                    text="Если сотрудник заболел, он уведомляет руководителя и HR в день обращения.",
                    score=0.81,
                )
            ]
        )
        client = FakeClient(
            response_text=(
                "Как оформить больничный?\n\n"
                "Из имеющихся фрагментов можно собрать следующие практические моменты:\n"
                "- Нужно сообщить руководителю и HR в день обращения "
                "(01_sick_leave_policy.docx, п.1).\n\n"
                "Что именно не хватает в фрагментах:\n"
                "- Не указано, какие документы прикладывать "
                "(01_sick_leave_policy.docx, раздел 2)."
            )
        )
        assistant = HRAssistantService(
            settings=build_settings(),
            client=client,
            knowledge_base=knowledge_base,
        )

        answer = assistant.ask("Как оформить больничный?")

        self.assertNotIn("Из имеющихся фрагментов", answer.text)
        self.assertNotIn("01_sick_leave_policy.docx", answer.text)
        self.assertNotIn("Как оформить больничный?", answer.text)
        self.assertIn("- Нужно сообщить руководителю и HR в день обращения.", answer.text)
        self.assertNotIn("Что не указано", answer.text)
        self.assertNotIn("Не указано", answer.text)

    def test_polish_answer_text_removes_formal_intro_source_references_and_missing_info_block(self) -> None:
        polished = HRAssistantService._polish_answer_text(
            "Как оформить больничный?",
            (
                "Как оформить больничный?\n\n"
                "Согласно предоставленным документам:\n"
                "- Нужно сообщить руководителю "
                "(01_sick_leave_policy.docx, п.1).\n\n"
                "Чего не хватает в документах:\n"
                "- Не указан срок передачи номера ЭЛН "
                "(01_sick_leave_policy.docx, раздел 3)."
            ),
        )

        self.assertEqual(
            polished,
            "- Нужно сообщить руководителю.",
        )

    def test_polish_answer_text_removes_labels_and_document_hedging(self) -> None:
        polished = HRAssistantService._polish_answer_text(
            "Расскажи как выйти в отпуск",
            (
                "Ответ: оформить отпуск можно по установленной процедуре.\n\n"
                "Коротко про действия:\n"
                "- В документах указано, что нужно заранее подать заявку.\n"
                "- В документе упоминается согласование с руководителем."
            ),
        )

        self.assertEqual(
            polished,
            "оформить отпуск можно по установленной процедуре.\n\n"
            "- нужно заранее подать заявку.\n"
            "- согласование с руководителем.",
        )

    def test_sources_prefer_relevant_snippet_over_generic_intro(self) -> None:
        assistant = HRAssistantService(
            settings=build_settings(),
            client=object(),
            knowledge_base=DummyKnowledgeBase(),
        )
        retrieved_chunks = [
            RetrievedChunk(
                chunk_id="doc-1",
                title="03_certificates_and_documents_policy.docx",
                source_path="documents/03_certificates_and_documents_policy.docx",
                text="Внутренний HR-регламент компании. Документ предназначен для внутреннего использования.",
                score=0.62,
            ),
            RetrievedChunk(
                chunk_id="doc-2",
                title="03_certificates_and_documents_policy.docx",
                source_path="documents/03_certificates_and_documents_policy.docx",
                text="Для получения справки с работы сотрудник подает запрос через HR-сервис и указывает тип документа.",
                score=0.58,
            ),
        ]

        sources = assistant._build_sources("Мне нужна справка с работы", retrieved_chunks)

        self.assertEqual(len(sources), 1)
        self.assertIn("справки", sources[0].snippet.lower())

    def test_specific_training_reimbursement_case_falls_back_to_hr(self) -> None:
        knowledge_base = RetrievedChunksKnowledgeBase(
            [
                RetrievedChunk(
                    chunk_id="doc-1",
                    title="05_learning_and_compensation_policy.docx",
                    source_path="documents/05_learning_and_compensation_policy.docx",
                    text=(
                        "Положение об обучении сотрудников и компенсации расходов. "
                        "Базовыми каналами являются процесс согласования, HR/L&D сервис "
                        "и согласование с руководителем. "
                        "Для сопровождения процесса сотрудник предоставляет описание программы, "
                        "стоимость, ссылку на курс, ожидаемый результат и подтверждение оплаты."
                    ),
                    score=0.82,
                ),
                RetrievedChunk(
                    chunk_id="doc-2",
                    title="05_learning_and_compensation_policy.docx",
                    source_path="documents/05_learning_and_compensation_policy.docx",
                    text=(
                        "Типовые вопросы: как согласовать профессиональный курс, "
                        "в каких случаях компания компенсирует обучение частично, "
                        "какие постусловия могут действовать после дорогостоящего обучения."
                    ),
                    score=0.76,
                ),
            ]
        )
        client = FakeClient()
        assistant = HRAssistantService(
            settings=build_settings(),
            client=client,
            knowledge_base=knowledge_base,
        )

        answer = assistant.ask("Как оформить компенсацию обучения, если курс уже оплачен?")

        self.assertEqual(answer.response_kind, "hr_fallback")
        self.assertTrue(answer.show_handoff)
        self.assertTrue(answer.is_fallback)
        self.assertEqual(len(client.chat.completions.calls), 0)

    def test_uncovered_maternity_question_falls_back_to_hr(self) -> None:
        knowledge_base = RetrievedChunksKnowledgeBase(
            [
                RetrievedChunk(
                    chunk_id="doc-1",
                    title="03_certificates_and_documents_policy.docx",
                    source_path="documents/03_certificates_and_documents_policy.docx",
                    text=(
                        "Регламент оформления справок, копий и кадровых документов. "
                        "Базовыми каналами являются HR-сервис и официальный почтовый ящик HR Operations."
                    ),
                    score=0.79,
                )
            ]
        )
        client = FakeClient()
        assistant = HRAssistantService(
            settings=build_settings(),
            client=client,
            knowledge_base=knowledge_base,
        )

        answer = assistant.ask("Какие документы нужны для оформления декрета?")

        self.assertEqual(answer.response_kind, "hr_fallback")
        self.assertTrue(answer.show_handoff)
        self.assertTrue(answer.is_fallback)
        self.assertEqual(len(client.chat.completions.calls), 0)

    def test_general_sick_leave_channel_question_still_gets_answer(self) -> None:
        knowledge_base = RetrievedChunksKnowledgeBase(
            [
                RetrievedChunk(
                    chunk_id="doc-1",
                    title="01_sick_leave_policy.docx",
                    source_path="documents/01_sick_leave_policy.docx",
                    text=(
                        "Базовыми каналами для запуска процесса являются корпоративный мессенджер, "
                        "звонок руководителю, уведомление HR-партнера и фиксация номера ЭЛН. "
                        "Сотрудник должен передать минимально необходимую информацию."
                    ),
                    score=0.84,
                ),
                RetrievedChunk(
                    chunk_id="doc-2",
                    title="01_sick_leave_policy.docx",
                    source_path="documents/01_sick_leave_policy.docx",
                    text=(
                        "Если сотрудник заболел, он сообщает о заболевании через официальный канал. "
                        "Если больничный продлен, сотрудник обновляет статус через тот же официальный канал, "
                        "который использовался для запуска процесса."
                    ),
                    score=0.72,
                ),
            ]
        )
        client = FakeClient(response_text="Сообщи о заболевании через официальный канал команды и HR.")
        assistant = HRAssistantService(
            settings=build_settings(),
            client=client,
            knowledge_base=knowledge_base,
        )

        answer = assistant.ask("Куда писать если заболел?")

        self.assertEqual(answer.response_kind, "answer")
        self.assertFalse(answer.is_fallback)
        self.assertEqual(answer.text, "Сообщи о заболевании через официальный канал команды и HR.")

    def test_fallback_for_specific_hr_question_is_not_offtopic(self) -> None:
        assistant = HRAssistantService(
            settings=build_settings(),
            client=object(),
            knowledge_base=DummyKnowledgeBase(),
        )

        answer = assistant.ask("Какие документы нужны для оформления декрета?")

        self.assertEqual(answer.response_kind, "hr_fallback")
        self.assertTrue(answer.show_handoff)
        self.assertTrue(answer.is_fallback)


class KnowledgeBaseQueryExpansionTests(unittest.TestCase):
    def test_sick_leave_query_terms_include_hr_synonyms(self) -> None:
        terms = KnowledgeBase._extract_query_terms("Я заболел, что делать?")

        self.assertIn("забол", terms)
        self.assertIn("больнич", terms)
        self.assertIn("нетрудоспособ", terms)

    def test_remote_work_query_terms_expand_colloquial_wording(self) -> None:
        terms = KnowledgeBase._extract_query_terms("Как уйти на удаленку?")

        self.assertIn("удален", terms)
        self.assertIn("дистанцион", terms)
        self.assertIn("гибрид", terms)


if __name__ == "__main__":
    unittest.main()
