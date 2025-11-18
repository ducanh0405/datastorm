"""
Unified CLI for SmartGrocy
===========================
Single entry point for all SmartGrocy operations.
"""
import argparse
import logging
import sys
from pathlib import Path

# Setup project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import setup_logging, setup_project_path

setup_project_path()
setup_logging()

logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """Setup argument parser"""
    parser = argparse.ArgumentParser(
        prog='smartgrocy',
        description='SmartGrocy E-Grocery Forecasting System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python -m src.cli.main pipeline --full-data

  # Run business modules
  python -m src.cli.main business --forecasts reports/predictions_test_set.csv

  # Run tests
  python -m src.cli.main test

  # Show configuration
  python -m src.cli.main config show
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run ML pipeline')
    pipeline_parser.add_argument('--full-data', action='store_true', help='Use full dataset')
    pipeline_parser.add_argument('--sample', type=float, default=1.0, help='Sample fraction (0.1 = 10%%)')
    pipeline_parser.add_argument('--use-v2', action='store_true', help='Use v2 orchestrator with GX')
    pipeline_parser.add_argument('--no-cache', action='store_true', help='Disable caching')

    # Business command
    business_parser = subparsers.add_parser('business', help='Run business modules')
    business_parser.add_argument('--forecasts', type=str, default='reports/predictions_test_set.csv',
                                help='Path to forecasts file')
    business_parser.add_argument('--inventory-only', action='store_true', help='Only run inventory optimization')
    business_parser.add_argument('--pricing-only', action='store_true', help='Only run dynamic pricing')
    business_parser.add_argument('--llm-only', action='store_true', help='Only run LLM insights')
    business_parser.add_argument('--no-llm', action='store_true', help='Do not use LLM API')

    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    test_parser.add_argument('--coverage', action='store_true', help='Generate coverage report')

    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_action', help='Config actions')
    config_subparsers.add_parser('show', help='Show current configuration')
    config_subparsers.add_parser('validate', help='Validate configuration')

    return parser


def run_pipeline(args) -> int:
    """Run ML pipeline"""
    try:
        from run_end_to_end import run_ml_pipeline
        success = run_ml_pipeline(args)
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


def run_business(args) -> int:
    """Run business modules"""
    try:
        from run_end_to_end import run_business_modules
        success = run_business_modules(args)
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Business modules failed: {e}", exc_info=True)
        return 1


def run_tests(args) -> int:
    """Run tests"""
    try:
        import subprocess
        cmd = [sys.executable, 'run_all_tests.py']
        if args.quick:
            cmd.append('--quick')
        if args.coverage:
            cmd.append('--coverage')
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode
    except Exception as e:
        logger.error(f"Tests failed: {e}", exc_info=True)
        return 1


def handle_config(args) -> int:
    """Handle config commands"""
    try:
        from src.config import ACTIVE_DATASET, DATASET_CONFIGS
        from src.core.config_manager import ConfigManager

        manager = ConfigManager()

        # Load existing config
        for name, ds_config_dict in DATASET_CONFIGS.items():
            from src.core.config_manager import DatasetConfig
            ds_config = DatasetConfig(**ds_config_dict)
            manager.register_dataset(name, ds_config)

        manager.set_active_dataset(ACTIVE_DATASET)

        if args.config_action == 'show':
            print("\n=== SmartGrocy Configuration ===")
            print(f"Active Dataset: {manager.active_dataset}")
            print("\nAvailable Datasets:")
            for name, config in manager.datasets.items():
                print(f"  - {name}: {config.name}")
            print("\nTraining Config:")
            print(f"  Model Type: {manager.training.model_type}")
            print(f"  Quantiles: {manager.training.quantiles}")
            return 0

        elif args.config_action == 'validate':
            try:
                manager.get_active_dataset_config().validate()
                manager.training.validate()
                print("✓ Configuration is valid")
                return 0
            except Exception as e:
                print(f"✗ Configuration validation failed: {e}")
                return 1

    except Exception as e:
        logger.error(f"Config command failed: {e}", exc_info=True)
        return 1


def main() -> int:
    """Main entry point"""
    parser = setup_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == 'pipeline':
            return run_pipeline(args)
        elif args.command == 'business':
            return run_business(args)
        elif args.command == 'test':
            return run_tests(args)
        elif args.command == 'config':
            return handle_config(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


def cli():
    """CLI entry point"""
    sys.exit(main())


if __name__ == '__main__':
    cli()

