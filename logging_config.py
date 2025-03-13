import datetime
import logging
import pathlib


class LoggingDateTimeFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str) -> str:
        """Override formatTime to ensure UTC timezone."""
        dt = datetime.datetime.fromtimestamp(record.created, datetime.timezone.utc)
        return f"{dt.strftime('%Y:%m:%d %H:%M:%S.%f UTC')}"


def setup_logging(
    file_name: str,
    log_level: str = logging.INFO,
    logs_dir: pathlib.Path = pathlib.Path("logs"),
):
    """Configures logging for the FastAPI app with timestamped logfiles."""
    # Create logs directory if it doesn't exist
    logs_dir.mkdir(exist_ok=True)

    # Generate timestamp for logfile name
    # timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    # log_file = logs_dir / f"app_{timestamp}.log"
    log_file = logs_dir / f"{file_name}.log"

    log_formatter = LoggingDateTimeFormatter("[%(levelname)s] %(asctime)s: %(message)s")

    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(asctime)s: %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    # Apply the custom formatter to all handlers
    for handler in logging.getLogger().handlers:
        handler.setFormatter(log_formatter)

    # Get the main logger
    logger = logging.getLogger(__name__)
    logger.debug("Logging is set up and ready!")

    return logger
