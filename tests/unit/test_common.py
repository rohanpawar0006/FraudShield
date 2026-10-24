"""Unit tests for src/common config and logging utilities."""

import json
import logging

from src.common.config import Settings
from src.common.logging import JSONFormatter, get_logger


def test_settings_defaults() -> None:
    """Verify default configuration settings."""
    settings = Settings()
    assert settings.ENV == "development"
    assert settings.API_PORT == 8000
    assert "postgresql://" in settings.postgres_uri
    assert settings.ALERT_THRESHOLD_HIGH == 0.7
    assert settings.ALERT_THRESHOLD_MED == 0.3


def test_json_formatter_with_context() -> None:
    """Verify structured JSON formatter includes extra fields."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_path.py",
        lineno=10,
        msg="Test transaction scored",
        args=(),
        exc_info=None,
    )
    record.txn_id = "TXN_12345"
    record.request_id = "REQ_999"

    formatted = formatter.format(record)
    log_data = json.loads(formatted)

    assert log_data["level"] == "INFO"
    assert log_data["message"] == "Test transaction scored"
    assert log_data["txn_id"] == "TXN_12345"
    assert log_data["request_id"] == "REQ_999"
    assert "timestamp" in log_data


def test_get_logger_singleton() -> None:
    """Verify get_logger returns logger with JSON handler."""
    logger = get_logger("unit_test_logger")
    assert logger.name == "unit_test_logger"
    assert len(logger.handlers) >= 1
    assert isinstance(logger.handlers[0].formatter, JSONFormatter)
