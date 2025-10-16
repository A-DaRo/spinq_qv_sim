"""
Logging configuration for spinq_qv.

Provides structured JSON logging with metadata (git hash, versions, timestamps).
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter with metadata."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with timestamp and metadata."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Include extra fields if present
        if hasattr(record, "metadata"):
            log_data["metadata"] = record.metadata
        
        return json.dumps(log_data)


def setup_logger(
    name: str = "spinq_qv",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    json_format: bool = False,
) -> logging.Logger:
    """
    Set up and configure a logger.
    
    Args:
        name: Logger name (default: "spinq_qv")
        level: Logging level (default: INFO)
        log_file: Optional file path for log output
        json_format: Use JSON-structured logging (default: False)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if json_format:
        console_formatter = StructuredFormatter()
    else:
        console_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        if json_format:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "spinq_qv") -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.
    
    Args:
        name: Logger name (default: "spinq_qv")
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logger(name)
    return logger
