"""
Structured logging for LoraForge.
Outputs JSON to file (machine-readable) and colored text to console (human-readable).
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)


class GPUContextFilter(logging.Filter):
    """Injects GPU memory stats into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            record.gpu_allocated_gb = f"{mem:.2f}"
            record.gpu_reserved_gb = f"{reserved:.2f}"
        else:
            record.gpu_allocated_gb = "N/A"
            record.gpu_reserved_gb = "N/A"
        return True


class JSONFileHandler(logging.Handler):
    """Writes structured JSON logs to a file — one JSON object per line."""

    def __init__(self, log_path: Path) -> None:
        super().__init__()
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "gpu_allocated_gb": getattr(record, "gpu_allocated_gb", "N/A"),
            "gpu_reserved_gb": getattr(record, "gpu_reserved_gb", "N/A"),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        with self.log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")


def get_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: str = "INFO",
) -> logging.Logger:
    """
    Get a structured logger for a module.

    Args:
        name: Logger name (typically __name__).
        log_dir: Directory to write JSON log file. If None, only console logging.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured Logger instance.

    Example:
        >>> logger = get_logger(__name__, log_dir=Path("./logs"))
        >>> logger.info("Baseline benchmark complete", extra={"batch_size": 4})
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger  # Already configured

    gpu_filter = GPUContextFilter()

    # Rich console handler (colored, human-readable)
    console_handler = RichHandler(
        console=_console,
        show_time=True,
        show_path=False,
        markup=True,
    )
    console_handler.addFilter(gpu_filter)
    logger.addHandler(console_handler)

    # JSON file handler (machine-readable, for benchmark result parsing)
    if log_dir is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"{name.replace('.', '_')}_{timestamp}.jsonl"
        json_handler = JSONFileHandler(log_file)
        json_handler.addFilter(gpu_filter)
        logger.addHandler(json_handler)

    logger.propagate = False
    return logger
