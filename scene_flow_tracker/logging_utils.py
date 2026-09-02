from __future__ import annotations

import logging
from pathlib import Path


def configure_process_logging(log_file: str | None = None, level: str | int = "INFO") -> None:
    """Configure logging for the current process.

    Spawned multiprocessing workers start with fresh logging state, so the parent
    process passes the log file through cfg and workers call this on startup.
    """
    numeric_level = getattr(logging, str(level).upper(), level)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(processName)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
