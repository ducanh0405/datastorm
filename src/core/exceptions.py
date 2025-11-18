"""
Custom Exceptions for SmartGrocy
=================================
Custom exception classes for better error handling.
"""

from typing import Any


class SmartGrocyException(Exception):
    """Base exception for SmartGrocy"""
    pass


class PipelineError(SmartGrocyException):
    """Exception raised during pipeline execution"""

    def __init__(self, message: str, stage: str = None, original_error: Exception = None):
        super().__init__(message)
        self.stage = stage
        self.original_error = original_error

    def __str__(self):
        msg = super().__str__()
        if self.stage:
            msg = f"[{self.stage}] {msg}"
        if self.original_error:
            msg = f"{msg}\nOriginal error: {self.original_error}"
        return msg


class ValidationError(SmartGrocyException):
    """Exception raised during data validation"""

    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value

    def __str__(self):
        msg = super().__str__()
        if self.field:
            msg = f"Field '{self.field}': {msg}"
        if self.value is not None:
            msg = f"{msg} (value: {self.value})"
        return msg


class ConfigurationError(SmartGrocyException):
    """Exception raised for configuration issues"""
    pass


class DataQualityError(SmartGrocyException):
    """Exception raised for data quality issues"""

    def __init__(self, message: str, quality_score: float = None):
        super().__init__(message)
        self.quality_score = quality_score

