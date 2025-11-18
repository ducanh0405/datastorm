#!/usr/bin/env python3
"""
SmartGrocy Business Modules Runner
==================================
Chạy các business modules (Inventory Optimization + Dynamic Pricing)
sau khi có forecasts từ pipeline

Usage:
    python run_business_modules.py --forecasts reports/predictions_test_set.parquet
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


def run_llm_insights(
    forecasts_path: str,
    inventory_path: str,
    pricing_path: str,
    output_path: str = None,
    top_n: int = 10,
    use_llm: bool = True
):
    """Chạy LLM insights module (Module 4)"""
    print_banner("MODULE 4: LLM INSIGHTS GENERATION")
    
    try:
        from src.modules.llm_insights import LLMInsightGenerator
        
        # Load data from all three modules
        logger.info(f"Loading forecasts from: {forecasts_path}")
        if Path(forecasts_path).suffix == '.parquet':
            forecasts_df = pd.read_parquet(forecasts_path)
        else:
            forecasts_df = pd.read_csv(forecasts_path)
        logger.info(f"✓ Loaded {len(forecasts_df):,} forecast records")
        
        logger.info(f"Loading inventory results from: {inventory_path}")
        if Path(inventory_path).exists():
            if Path(inventory_path).suffix == '.parquet':
                inventory_df = pd.read_parquet(inventory_path)
            else:
                inventory_df = pd.read_csv(inventory_path)
            logger.info(f"✓ Loaded {len(inventory_df):,} inventory records")
        else:
            logger.warning(f"Inventory file not found: {inventory_path}")
            logger.info("Creating sample inventory data...")
            inventory_df = prepare_sample_inventory_data(forecasts_df)
        
        logger.info(f"Loading pricing results from: {pricing_path}")
        if Path(pricing_path).exists():
            if Path(pricing_path).suffix == '.parquet':
                pricing_df = pd.read_parquet(pricing_path)
            else:
                pricing_df = pd.read_csv(pricing_path)
            logger.info(f"✓ Loaded {len(pricing_df):,} pricing records")
        else:
            logger.warning(f"Pricing file not found: {pricing_path}")
            logger.info("Creating sample pricing data...")
            pricing_df = prepare_pricing_data(forecasts_df, inventory_df)
        
        # Initialize LLM generator
        generator = LLMInsightGenerator(use_llm_api=use_llm)
        
        # Load SHAP values if available
        shap_dict = None
        shap_path = Path('reports/shap_values/shap_values.json')
        if shap_path.exists():
            try:
                import json
                with open(shap_path, 'r') as f:
                    shap_data = json.load(f)
                    # Convert to dict format if needed
                    shap_dict = shap_data
                logger.info(f"✓ Loaded SHAP values for {len(shap_dict)} products")
            except Exception as e:
                logger.warning(f"Could not load SHAP values: {e}")
        
        # Generate insights for top N products
        logger.info(f"Generating comprehensive insights for top {top_n} products...")
        
        insights_df = generator.batch_generate_comprehensive(
            forecasts_df,
            inventory_df,
            pricing_df,
            shap_dict=shap_dict,
            top_n=top_n,
            use_llm=use_llm
        )
        
        if insights_df is not None and len(insights_df) > 0:
            output_file = output_path or 'reports/llm_insights.csv'
            insights_df.to_csv(output_file, index=False)
            logger.info(f"✅ LLM insights generation complete")
            logger.info(f"   Results saved to: {output_file}")
            logger.info(f"   Products analyzed: {len(insights_df)}")
            
            # Show sample insights
            logger.info("\n   Sample insights:")
            for idx, row in insights_df.head(3).iterrows():
                logger.info(f"\n   Product: {row.get('product_id', 'Unknown')}")
                insight_text = row.get('insight_text', row.get('insight', ''))
                if insight_text:
                    # Show first 200 characters
                    preview = insight_text[:200] + "..." if len(insight_text) > 200 else insight_text
                    logger.info(f"   Preview: {preview}")
            
            return insights_df
        else:
            logger.warning("No insights generated")
            return None
            
    except ImportError as e:
        logger.error(f"Failed to import LLM insights module: {e}")
        logger.info("Make sure google-generativeai is installed: pip install google-generativeai")
        return None
    except Exception as e:
        logger.error(f"LLM insights generation failed: {e}")
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
  # Chạy cả inventory, pricing và LLM insights
  python run_business_modules.py --forecasts reports/predictions_test_set.csv
  
  # Chỉ chạy inventory
  python run_business_modules.py --inventory-only
  
  # Chỉ chạy pricing
  python run_business_modules.py --pricing-only
  
  # Chỉ chạy LLM insights (cần có kết quả từ modules 1,2,3)
  python run_business_modules.py --llm-only
  
  # Chạy không dùng LLM API (rule-based)
  python run_business_modules.py --no-llm
        """
    )
    
    parser.add_argument('--forecasts', type=str,
                       default='reports/predictions_test_set.parquet',
                       help='Path to forecasts file (Module 1 output)')
    parser.add_argument('--inventory-only', action='store_true',
                       help='Chỉ chạy inventory optimization (Module 2)')
    parser.add_argument('--pricing-only', action='store_true',
                       help='Chỉ chạy dynamic pricing (Module 3)')
    parser.add_argument('--llm-only', action='store_true',
                       help='Chỉ chạy LLM insights (Module 4)')
    parser.add_argument('--no-llm', action='store_true',
                       help='Không dùng LLM API, chỉ dùng rule-based')
    parser.add_argument('--output-dir', type=str,
                       default='reports',
                       help='Output directory for results')
    parser.add_argument('--top-n', type=int,
                       default=10,
                       help='Số lượng sản phẩm để generate insights (default: 10)')
    
    args = parser.parse_args()
    
    # Print header
    print_banner("🚀 SMARTGROCY BUSINESS MODULES")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check forecasts file (unless LLM-only mode)
    if not args.llm_only:
        forecasts_path = Path(args.forecasts)
        if not forecasts_path.exists():
            logger.error(f"Forecasts file not found: {forecasts_path}")
            logger.info("Please run prediction pipeline first:")
            logger.info("  python src\\pipelines\\_05_prediction.py")
            sys.exit(1)
    
    inventory_df = None
    inventory_results = None
    pricing_results = None
    
    # Run inventory optimization (Module 2)
    if not args.pricing_only and not args.llm_only:
        inventory_results = run_inventory_optimization(
            str(Path(args.forecasts)),
            str(output_dir / 'inventory_recommendations.csv')
        )
        if inventory_results is not None:
            # Prepare inventory data for pricing
            forecasts_df = pd.read_csv(args.forecasts) if Path(args.forecasts).suffix == '.csv' else pd.read_parquet(args.forecasts)
            inventory_df = prepare_sample_inventory_data(forecasts_df)
    
    # Run dynamic pricing (Module 3)
    if not args.inventory_only and not args.llm_only:
        if inventory_df is None and Path(args.forecasts).exists():
            forecasts_df = pd.read_csv(args.forecasts) if Path(args.forecasts).suffix == '.csv' else pd.read_parquet(args.forecasts)
            inventory_df = prepare_sample_inventory_data(forecasts_df)
        
        pricing_results = run_dynamic_pricing(
            str(Path(args.forecasts)),
            inventory_df,
            str(output_dir / 'pricing_recommendations.csv')
        )
    
    # Run LLM insights (Module 4)
    if not args.inventory_only and not args.pricing_only:
        # Determine paths
        forecasts_path = args.forecasts
        inventory_path = str(output_dir / 'inventory_recommendations.csv')
        pricing_path = str(output_dir / 'pricing_recommendations.csv')
        
        # If LLM-only mode, check if files exist
        if args.llm_only:
            if not Path(forecasts_path).exists():
                logger.error(f"Forecasts file not found: {forecasts_path}")
                sys.exit(1)
            if not Path(inventory_path).exists():
                logger.error(f"Inventory file not found: {inventory_path}")
                logger.info("Please run Module 2 first or use --forecasts with full pipeline")
                sys.exit(1)
            if not Path(pricing_path).exists():
                logger.error(f"Pricing file not found: {pricing_path}")
                logger.info("Please run Module 3 first or use --forecasts with full pipeline")
                sys.exit(1)
        
        llm_results = run_llm_insights(
            forecasts_path,
            inventory_path,
            pricing_path,
            output_path=str(output_dir / 'llm_insights.csv'),
            top_n=args.top_n,
            use_llm=not args.no_llm
        )
    
    # Summary
    print_banner("✅ BUSINESS MODULES HOÀN THÀNH")
    logger.info(f"Output files:")
    if inventory_results is not None or Path(output_dir / 'inventory_recommendations.csv').exists():
        logger.info(f"  - Module 2 (Inventory): {output_dir / 'inventory_recommendations.csv'}")
    if pricing_results is not None or Path(output_dir / 'pricing_recommendations.csv').exists():
        logger.info(f"  - Module 3 (Pricing): {output_dir / 'pricing_recommendations.csv'}")
    if Path(output_dir / 'llm_insights.csv').exists():
        logger.info(f"  - Module 4 (LLM Insights): {output_dir / 'llm_insights.csv'}")
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

