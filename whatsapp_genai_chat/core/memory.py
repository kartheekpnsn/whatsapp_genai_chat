import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_FIELDS = ["timestamp", "query", "response"]


class MemoryStore:
    def __init__(self, csv_path: Path, window_minutes: int = 30) -> None:
        self.csv_path = Path(csv_path)
        self.window_minutes = window_minutes

    def append(self, query: str, response: str) -> None:
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self.csv_path.exists()
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "query": query,
                    "response": response,
                })
        except Exception:
            logger.warning("MemoryStore.append failed", exc_info=True)

    def load_recent(self) -> list[dict]:
        return []
