# AI HR Assistant

MVP веб-приложения для HR Q&A на русском языке: один экран чата на `Streamlit`, ответы на основе внутренних документов компании через `OpenAI API + RAG`, показ источников, fallback при низкой уверенности и кнопка передачи вопроса HR.

## Быстрый запуск

1. Создать и активировать виртуальное окружение.
2. Установить зависимости:

```bash
pip install -r requirements.txt
```

3. Заполнить `.env` на основе `.env.example`.
4. Положить HR-документы в папку `documents`.
5. Запустить приложение:

```bash
streamlit run app.py
```

## Ручная переиндексация

Если нужно отдельно пересобрать индекс документов:

```bash
python scripts/rebuild_index.py
```

## Архитектура

- `app.py` — UI и пользовательский сценарий чата
- `src/ai_hr_assistant/config.py` — конфигурация приложения
- `src/ai_hr_assistant/domain/models.py` — доменные модели
- `src/ai_hr_assistant/infrastructure/openai_factory.py` — создание OpenAI клиента
- `src/ai_hr_assistant/services/document_loader.py` — чтение и чанкинг документов
- `src/ai_hr_assistant/services/knowledge_base.py` — индексация, хранение и retrieval
- `src/ai_hr_assistant/services/hr_assistant.py` — orchestration ответа, fallback и источники
- `scripts/rebuild_index.py` — вспомогательный CLI для обновления базы знаний

## Структура проекта

```text
.
├── app.py
├── data
│   └── index
├── documents
├── scripts
├── src
│   └── ai_hr_assistant
│       ├── domain
│       ├── infrastructure
│       └── services
├── .env.example
├── pyproject.toml
└── requirements.txt
```
