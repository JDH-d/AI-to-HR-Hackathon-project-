from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_hr_assistant.config import AppSettings
from ai_hr_assistant.services.hr_assistant import HRAssistantService


def main() -> None:
    settings = AppSettings.from_env()
    assistant = HRAssistantService.from_settings(settings)
    status = assistant.ensure_knowledge_base(force_rebuild=True)
    print(
        "Индекс обновлен: "
        f"документов={status.document_count}, "
        f"чанков={status.chunk_count}, "
        f"статус={status.message}"
    )


if __name__ == "__main__":
    main()
