"""
Core Module - Base Classes and Interfaces
==========================================
Provides base classes and interfaces for the SmartGrocy system.
"""

from src.core.base import BaseConfig, BaseModule, BasePipeline
from src.core.exceptions import PipelineError, SmartGrocyException, ValidationError

__all__ = [
    'BasePipeline',
    'BaseModule',
    'BaseConfig',
    'SmartGrocyException',
    'PipelineError',
    'ValidationError',
]

