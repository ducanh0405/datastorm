"""
Base Classes for SmartGrocy
===========================
Abstract base classes and interfaces for pipelines, modules, and configurations.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class BaseConfig(ABC):
    """Base configuration class"""
    name: str
    description: str = ""

    def validate(self) -> bool:
        """Validate configuration"""
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class BasePipeline(ABC):
    """Base class for all pipelines"""

    def __init__(self, config: BaseConfig | None = None, logger: logging.Logger | None = None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._validate_setup()

    def _validate_setup(self) -> None:
        """Validate pipeline setup"""
        if self.config and hasattr(self.config, 'validate'):
            if not self.config.validate():
                raise ValueError(f"Invalid configuration for {self.__class__.__name__}")

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Run the pipeline"""
        pass

    def setup(self) -> None:
        """Setup pipeline (override if needed)"""
        pass

    def cleanup(self) -> None:
        """Cleanup pipeline (override if needed)"""
        pass

    def __enter__(self):
        """Context manager entry"""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


class BaseModule(ABC):
    """Base class for business modules"""

    def __init__(self, config: BaseConfig | None = None, logger: logging.Logger | None = None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """Process data"""
        pass

    def validate_input(self, data: Any) -> bool:
        """Validate input data"""
        return True

    def validate_output(self, output: Any) -> bool:
        """Validate output data"""
        return True

