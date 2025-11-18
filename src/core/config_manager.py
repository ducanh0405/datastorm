"""
Configuration Manager
=====================
Centralized configuration management with validation and type safety.
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.exceptions import ConfigurationError


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str
    file_format: str = 'parquet'
    temporal_unit: str = 'hour'
    time_column: str = ''
    target_column: str = ''
    groupby_keys: list[str] = field(default_factory=list)
    required_columns: list[str] = field(default_factory=list)

    # Feature workstream toggles
    has_relational: bool = False
    has_stockout: bool = False
    has_weather: bool = False
    has_price_promo: bool = False
    has_behavior: bool = False

    # WS2 Config
    lag_periods: list[int] = field(default_factory=lambda: [1, 24, 48])
    rolling_windows: list[int] = field(default_factory=lambda: [24, 168])
    has_intraday_patterns: bool = False

    def validate(self) -> bool:
        """Validate configuration"""
        if not self.time_column:
            raise ConfigurationError("time_column is required")
        if not self.target_column:
            raise ConfigurationError("target_column is required")
        if not self.groupby_keys:
            raise ConfigurationError("groupby_keys cannot be empty")
        return True


@dataclass
class TrainingConfig:
    """Training configuration"""
    model_type: str = 'lightgbm'
    quantiles: list[float] = field(default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95])
    train_test_split: dict[str, Any] = field(default_factory=lambda: {'cutoff_percentile': 0.8})
    random_state: int = 42

    def validate(self) -> bool:
        """Validate configuration"""
        if self.model_type not in ['lightgbm', 'catboost', 'random_forest']:
            raise ConfigurationError(f"Invalid model_type: {self.model_type}")
        if not all(0 < q < 1 for q in self.quantiles):
            raise ConfigurationError("All quantiles must be between 0 and 1")
        return True


@dataclass
class PathsConfig:
    """Paths configuration"""
    project_root: Path
    raw_data: Path
    processed_data: Path
    models: Path
    reports: Path
    logs: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> 'PathsConfig':
        """Create from project root"""
        return cls(
            project_root=project_root,
            raw_data=project_root / 'data' / '2_raw',
            processed_data=project_root / 'data' / '3_processed',
            models=project_root / 'models',
            reports=project_root / 'reports',
            logs=project_root / 'logs',
        )

    def ensure_directories(self) -> None:
        """Ensure all directories exist"""
        for path in [self.raw_data, self.processed_data, self.models,
                     self.reports, self.logs]:
            path.mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """Centralized configuration manager"""

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent
        self.paths = PathsConfig.from_project_root(self.project_root)
        self.paths.ensure_directories()

        self.datasets: dict[str, DatasetConfig] = {}
        self.training = TrainingConfig()
        self.active_dataset: str | None = None
        self.logger = logging.getLogger(__name__)

    def register_dataset(self, name: str, config: DatasetConfig) -> None:
        """Register a dataset configuration"""
        config.validate()
        self.datasets[name] = config
        self.logger.info(f"Registered dataset: {name}")

    def set_active_dataset(self, name: str) -> None:
        """Set active dataset"""
        if name not in self.datasets:
            raise ConfigurationError(f"Dataset '{name}' not found")
        self.active_dataset = name
        self.logger.info(f"Active dataset set to: {name}")

    def get_active_dataset_config(self) -> DatasetConfig:
        """Get active dataset configuration"""
        if not self.active_dataset:
            raise ConfigurationError("No active dataset set")
        return self.datasets[self.active_dataset]

    def load_from_dict(self, config_dict: dict[str, Any]) -> None:
        """Load configuration from dictionary"""
        # Load datasets
        if 'datasets' in config_dict:
            for name, ds_config in config_dict['datasets'].items():
                self.register_dataset(name, DatasetConfig(**ds_config))

        # Load training config
        if 'training' in config_dict:
            self.training = TrainingConfig(**config_dict['training'])

        # Set active dataset
        if 'active_dataset' in config_dict:
            self.set_active_dataset(config_dict['active_dataset'])

    def load_from_file(self, config_path: Path) -> None:
        """Load configuration from JSON file"""
        if not config_path.exists():
            raise ConfigurationError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config_dict = json.load(f)

        self.load_from_dict(config_dict)
        self.logger.info(f"Loaded configuration from: {config_path}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            'active_dataset': self.active_dataset,
            'datasets': {
                name: {
                    k: v for k, v in ds.__dict__.items()
                    if not k.startswith('_')
                }
                for name, ds in self.datasets.items()
            },
            'training': {
                k: v for k, v in self.training.__dict__.items()
                if not k.startswith('_')
            }
        }

    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to JSON file"""
        config_dict = self.to_dict()
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        self.logger.info(f"Saved configuration to: {config_path}")

