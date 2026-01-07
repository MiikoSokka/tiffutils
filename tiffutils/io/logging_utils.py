# tiffutils/logging_utils.py
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field

DEFAULT_FMT = "%(asctime)s [%(levelname)s] [%(name)s:%(funcName)s] %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name)

def level_from_verbose(verbose: int) -> int:
    # 0=WARNING, 1=INFO, 2+=DEBUG
    if verbose <= 0:
        return logging.WARNING
    if verbose == 1:
        return logging.INFO
    return logging.DEBUG

def setup_root_logging(verbose: int = 1, *, stream=None, fmt: str = DEFAULT_FMT) -> None:
    """
    Use this in your pipeline entrypoint (not inside library code).
    Robust against other libraries calling logging.basicConfig().
    """
    level = level_from_verbose(verbose)

    handler = logging.StreamHandler(stream)  # stream=None -> sys.stderr
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=DEFAULT_DATEFMT))

    # force=True ensures we override any previous logging configuration reliably
    logging.basicConfig(
        level=level,
        handlers=[handler],
        format=fmt,
        datefmt=DEFAULT_DATEFMT,
        force=True,
    )

class ContextLogger(logging.LoggerAdapter):
    """
    Adds stable context like sample/run/fov to every line.
    """
    def process(self, msg, kwargs):
        ctx = " ".join([f"{k}={v}" for k, v in self.extra.items() if v is not None])
        return (f"[{ctx}] {msg}" if ctx else msg), kwargs

@dataclass
class Timer:
    t0: float = field(default_factory=time.perf_counter)
    def s(self) -> float:
        return time.perf_counter() - self.t0