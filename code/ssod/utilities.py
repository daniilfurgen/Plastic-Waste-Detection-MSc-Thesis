import logging
from pathlib import Path


LOGGER_NAME = "msc_thesis"
_HANDLER_MARKER = "_msc_thesis_handler"


def get_logger(
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return the shared project logger, configuring it on first use."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    owned_handlers = [
        handler
        for handler in logger.handlers
        if getattr(handler, _HANDLER_MARKER, False)
    ]

    if not any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in owned_handlers
    ):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        setattr(stream_handler, _HANDLER_MARKER, True)
        logger.addHandler(stream_handler)

    for handler in owned_handlers:
        handler.setLevel(level)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path = str(log_path.resolve())
        has_file_handler = any(
            isinstance(handler, logging.FileHandler)
            and handler.baseFilename == resolved_path
            for handler in owned_handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            setattr(file_handler, _HANDLER_MARKER, True)
            logger.addHandler(file_handler)

    return logger
