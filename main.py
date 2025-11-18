#!/usr/bin/env python3
"""
SmartGrocy - Main Entry Point
==============================
Main entry point for SmartGrocy E-Grocery Forecasting System.

Usage:
    # Run full pipeline
    python main.py pipeline --full-data
    
    # Run business modules
    python main.py business --forecasts reports/predictions_test_set.csv
    
    # Run tests
    python main.py test
    
    # Show help
    python main.py --help
"""
import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_pipeline(args):
    """Run the main pipeline"""
    cmd = [sys.executable, '-m', 'src.pipelines._00_modern_orchestrator']
    if args.full_data:
        cmd.append('--full-data')
    if args.sample:
        cmd.extend(['--sample', str(args.sample)])
    if args.use_v2:
        cmd.append('--use-v2')
    if args.no_cache:
        cmd.append('--no-cache')
    
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def run_business_modules(args):
    """Run business modules"""
    cmd = [sys.executable, 'run_business_modules.py']
    if args.forecasts:
        cmd.extend(['--forecasts', args.forecasts])
    if args.inventory_only:
        cmd.append('--inventory-only')
    if args.pricing_only:
        cmd.append('--pricing-only')
    
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def run_tests(args):
    """Run tests"""
    cmd = [sys.executable, 'run_all_tests.py']
    if args.verbose:
        cmd.append('--verbose')
    
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='SmartGrocy - E-Grocery Forecasting System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py pipeline --full-data
  
  # Run pipeline with 10% sample
  python main.py pipeline --full-data --sample 0.1
  
  # Run business modules
  python main.py business
  
  # Run tests
  python main.py test
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run the main pipeline')
    pipeline_parser.add_argument('--full-data', action='store_true',
                                help='Use full dataset')
    pipeline_parser.add_argument('--sample', type=float,
                                help='Sample fraction (0.1 = 10%%)')
    pipeline_parser.add_argument('--use-v2', action='store_true',
                                help='Use v2 orchestrator with GX')
    pipeline_parser.add_argument('--no-cache', action='store_true',
                                help='Disable caching')
    
    # Business modules command
    business_parser = subparsers.add_parser('business', help='Run business modules')
    business_parser.add_argument('--forecasts', type=str,
                                default='reports/predictions_test_set.csv',
                                help='Path to forecasts file')
    business_parser.add_argument('--inventory-only', action='store_true',
                                help='Only run inventory optimization')
    business_parser.add_argument('--pricing-only', action='store_true',
                                help='Only run dynamic pricing')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--verbose', action='store_true',
                            help='Verbose output')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'pipeline':
        run_pipeline(args)
    elif args.command == 'business':
        run_business_modules(args)
    elif args.command == 'test':
        run_tests(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

