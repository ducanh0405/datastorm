#!/usr/bin/env python3
"""
SmartGrocy Business Modules Runner
==================================
Chạy các business modules (Inventory Optimization + Dynamic Pricing)
sau khi có forecasts từ pipeline

Usage:
    python run_business_modules.py --forecasts reports/predictions_test_set.csv
    python run_business_modules.py --inventory-only
    python run_business_modules.py --pricing-only
"""
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import setup_project_path, setup_logging, OUTPUT_FILES
setup_project_path()
setup_logging()

logger = logging.getLogger(__name__)


def print_banner(text: str):
    """Print formatted banner"""
    logger.info("\n" + "="*70)
    logger.info(f"  {text}")
    logger.info("="*70 + "\n")


def prepare_sample_inventory_data(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo sample inventory data từ forecasts (cho demo)
    """
    logger.info("Tạo sample inventory data từ forecasts...")
    
    # Group by product-store để tính average inventory
    inventory_data = forecasts_df.groupby(['product_id', 'store_id']).agg({
        'sales_quantity': ['mean', 'std']
    }).reset_index()
    
    inventory_data.columns = ['product_id', 'store_id', 'avg_demand', 'demand_std']
    
    # Simulate current inventory (1.5x to 3x average demand)
    np.random.seed(42)
    inventory_data['current_inventory'] = (
        inventory_data['avg_demand'] * np.random.uniform(1.5, 3.0, len(inventory_data))
    ).astype(int)
    
    # Calculate inventory ratio
    inventory_data['inventory_ratio'] = (
        inventory_data['current_inventory'] / inventory_data['avg_demand']
    )
    
    logger.info(f"✓ Created inventory data for {len(inventory_data)} product-store pairs")
    return inventory_data


def prepare_pricing_data(forecasts_df: pd.DataFrame, inventory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn bị data cho dynamic pricing
    """
    logger.info("Chuẩn bị pricing data...")
    
    # Merge forecasts với inventory
    pricing_data = forecasts_df.merge(
        inventory_df[['product_id', 'store_id', 'inventory_ratio', 'current_inventory']],
        on=['product_id', 'store_id'],
        how='left'
    )
    
    # Calculate demand ratio (forecast vs historical average)
    if 'forecast_q50' in pricing_data.columns:
        pricing_data['demand_ratio'] = (
            pricing_data['forecast_q50'] / pricing_data['sales_quantity'].mean()
        )
    else:
        # Fallback: use sales_quantity as proxy
        pricing_data['demand_ratio'] = (
            pricing_data['sales_quantity'] / pricing_data['sales_quantity'].mean()
        )
    
    # Simulate costs (50-70% of average price)
    if 'price' in pricing_data.columns:
        pricing_data['cost'] = pricing_data['price'] * np.random.uniform(0.5, 0.7, len(pricing_data))
        pricing_data['current_price'] = pricing_data['price']
    else:
        # Fallback: simulate prices
        np.random.seed(42)
        pricing_data['current_price'] = np.random.uniform(5, 50, len(pricing_data))
        pricing_data['cost'] = pricing_data['current_price'] * np.random.uniform(0.5, 0.7, len(pricing_data))
    
    # Fill missing inventory ratios
    pricing_data['inventory_ratio'] = pricing_data['inventory_ratio'].fillna(1.0)
    pricing_data['demand_ratio'] = pricing_data['demand_ratio'].fillna(1.0)
    
    logger.info(f"✓ Prepared pricing data for {len(pricing_data)} records")
    return pricing_data


def run_inventory_optimization(forecasts_path: str, output_path: str = None):
    """Chạy inventory optimization"""
    print_banner("MODULE 2: INVENTORY OPTIMIZATION")
    
    try:
        from src.modules.inventory_optimization import InventoryOptimizer, InventoryConfig
        
        # Load forecasts
        logger.info(f"Loading forecasts from: {forecasts_path}")
        if Path(forecasts_path).suffix == '.parquet':
            forecasts = pd.read_parquet(forecasts_path)
        else:
            forecasts = pd.read_csv(forecasts_path)
        
        logger.info(f"✓ Loaded {len(forecasts):,} forecast records")
        
        # Prepare sample inventory data
        inventory_data = prepare_sample_inventory_data(forecasts)
        
        # Initialize optimizer
        config = InventoryConfig(
            service_level=0.95,
            ordering_cost=50.0,
            holding_cost_rate=0.20,
            lead_time_days=7,
            unit_cost=10.0
        )
        optimizer = InventoryOptimizer(config)
        
        # Optimize for sample products
        logger.info("Optimizing inventory for sample products...")
        results = []
        
        # Sample first 10 product-store pairs for demo
        sample_pairs = inventory_data.head(10)
        
        for _, row in sample_pairs.iterrows():
            try:
                # Get forecasts for this product-store
                product_forecasts = forecasts[
                    (forecasts['product_id'] == row['product_id']) &
                    (forecasts['store_id'] == row['store_id'])
                ]
                
                if len(product_forecasts) > 0:
                    # Calculate forecast stats
                    if 'forecast_q50' in product_forecasts.columns:
                        avg_demand = product_forecasts['forecast_q50'].mean()
                        demand_std = product_forecasts['forecast_q50'].std()
                    else:
                        avg_demand = product_forecasts['sales_quantity'].mean()
                        demand_std = product_forecasts['sales_quantity'].std()
                    
                    # Optimize
                    result = optimizer.optimize_inventory_from_forecast(
                        product_forecasts,
                        row['product_id'],
                        row['store_id']
                    )
                    
                    result['current_inventory'] = row['current_inventory']
                    result['inventory_ratio'] = row['inventory_ratio']
                    results.append(result)
            except Exception as e:
                logger.warning(f"Failed to optimize {row['product_id']}-{row['store_id']}: {e}")
        
        if results:
            results_df = pd.DataFrame(results)
            output_file = output_path or 'reports/inventory_recommendations.csv'
            results_df.to_csv(output_file, index=False)
            logger.info(f"✅ Inventory optimization complete")
            logger.info(f"   Results saved to: {output_file}")
            logger.info(f"   Products optimized: {len(results_df)}")
            
            # Print summary
            if 'should_reorder' in results_df.columns:
                reorder_count = results_df['should_reorder'].sum()
                logger.info(f"   Products needing reorder: {reorder_count}")
            
            return results_df
        else:
            logger.warning("No results generated")
            return None
            
    except ImportError as e:
        logger.error(f"Failed to import inventory optimization module: {e}")
        return None
    except Exception as e:
        logger.error(f"Inventory optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_dynamic_pricing(forecasts_path: str, inventory_df: pd.DataFrame = None, output_path: str = None):
    """Chạy dynamic pricing"""
    print_banner("MODULE 3: DYNAMIC PRICING")
    
    try:
        from src.modules.dynamic_pricing import DynamicPricingEngine, PricingConfig
        
        # Load forecasts
        logger.info(f"Loading forecasts from: {forecasts_path}")
        if Path(forecasts_path).suffix == '.parquet':
            forecasts = pd.read_parquet(forecasts_path)
        else:
            forecasts = pd.read_csv(forecasts_path)
        
        # Prepare inventory data if not provided
        if inventory_df is None:
            inventory_df = prepare_sample_inventory_data(forecasts)
        
        # Prepare pricing data
        pricing_data = prepare_pricing_data(forecasts, inventory_df)
        
        # Initialize pricing engine
        config = PricingConfig(
            inventory_high_threshold=2.0,
            max_discount=0.50,
            min_profit_margin=0.10
        )
        engine = DynamicPricingEngine(config)
        
        # Sample data for demo (first 20 records)
        sample_data = pricing_data.head(20)
        
        logger.info(f"Optimizing prices for {len(sample_data)} products...")
        
        # Batch optimize
        results = engine.batch_optimize(sample_data)
        
        if results is not None and len(results) > 0:
            output_file = output_path or 'reports/pricing_recommendations.csv'
            results.to_csv(output_file, index=False)
            logger.info(f"✅ Dynamic pricing complete")
            logger.info(f"   Results saved to: {output_file}")
            logger.info(f"   Products optimized: {len(results)}")
            
            # Print summary
            if 'action' in results.columns:
                action_counts = results['action'].value_counts()
                logger.info("   Pricing actions:")
                for action, count in action_counts.items():
                    logger.info(f"     - {action}: {count}")
            
            # Show sample recommendations
            logger.info("\n   Sample recommendations:")
            sample_cols = ['product_id', 'current_price', 'recommended_price', 'discount_pct', 'action']
            available_cols = [c for c in sample_cols if c in results.columns]
            logger.info(f"\n{results[available_cols].head(5).to_string(index=False)}")
            
            return results
        else:
            logger.warning("No pricing results generated")
            return None
            
    except ImportError as e:
        logger.error(f"Failed to import dynamic pricing module: {e}")
        return None
    except Exception as e:
        logger.error(f"Dynamic pricing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SmartGrocy Business Modules Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Chạy cả inventory và pricing
  python run_business_modules.py --forecasts reports/predictions_test_set.csv
  
  # Chỉ chạy inventory
  python run_business_modules.py --inventory-only
  
  # Chỉ chạy pricing
  python run_business_modules.py --pricing-only
        """
    )
    
    parser.add_argument('--forecasts', type=str,
                       default='reports/predictions_test_set.csv',
                       help='Path to forecasts file')
    parser.add_argument('--inventory-only', action='store_true',
                       help='Chỉ chạy inventory optimization')
    parser.add_argument('--pricing-only', action='store_true',
                       help='Chỉ chạy dynamic pricing')
    parser.add_argument('--output-dir', type=str,
                       default='reports',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Print header
    print_banner("🚀 SMARTGROCY BUSINESS MODULES")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check forecasts file
    forecasts_path = Path(args.forecasts)
    if not forecasts_path.exists():
        logger.error(f"Forecasts file not found: {forecasts_path}")
        logger.info("Please run prediction pipeline first:")
        logger.info("  python src\\pipelines\\_05_prediction.py")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    inventory_df = None
    
    # Run inventory optimization
    if not args.pricing_only:
        inventory_results = run_inventory_optimization(
            str(forecasts_path),
            str(output_dir / 'inventory_recommendations.csv')
        )
        if inventory_results is not None:
            # Prepare inventory data for pricing
            inventory_df = prepare_sample_inventory_data(
                pd.read_csv(forecasts_path) if forecasts_path.suffix == '.csv' 
                else pd.read_parquet(forecasts_path)
            )
    
    # Run dynamic pricing
    if not args.inventory_only:
        pricing_results = run_dynamic_pricing(
            str(forecasts_path),
            inventory_df,
            str(output_dir / 'pricing_recommendations.csv')
        )
    
    # Summary
    print_banner("✅ BUSINESS MODULES HOÀN THÀNH")
    logger.info(f"Output files:")
    logger.info(f"  - Inventory: {output_dir / 'inventory_recommendations.csv'}")
    logger.info(f"  - Pricing: {output_dir / 'pricing_recommendations.csv'}")
    logger.info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

