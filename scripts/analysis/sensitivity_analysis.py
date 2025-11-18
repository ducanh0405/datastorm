#!/usr/bin/env python3
"""
Sensitivity Analysis for KPI Benchmarking
=========================================
Analyze performance across product groups, regions, and scenarios.

Author: SmartGrocy Team
Date: 2025-11-18
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


class SensitivityAnalyzer:
    """Analyze model sensitivity across dimensions."""
    
    def __init__(self):
        self.results = {}
        
    def analyze_by_product_group(
        self,
        df: pd.DataFrame,
        metrics: List[str] = ['mae', 'rmse', 'r2_score']
    ) -> pd.DataFrame:
        """
        Analyze performance by product category.
        
        Args:
            df: Predictions with actual values and product categories
            metrics: List of metrics to calculate
            
        Returns:
            Summary dataframe by product group
        """
        logger.info("\nAnalyzing by product group...")
        
        if 'product_category' not in df.columns:
            logger.warning("No product_category column, using default")
            df['product_category'] = 'General'
        
        groups = df.groupby('product_category')
        results = []
        
        for name, group in groups:
            if 'actual' in group.columns and 'predicted' in group.columns:
                mae = np.mean(np.abs(group['actual'] - group['predicted']))
                rmse = np.sqrt(np.mean((group['actual'] - group['predicted'])**2))
                r2 = 1 - (np.sum((group['actual'] - group['predicted'])**2) / 
                         np.sum((group['actual'] - np.mean(group['actual']))**2))
            else:
                mae, rmse, r2 = 0, 0, 0
            
            results.append({
                'product_group': name,
                'n_products': len(group['product_id'].unique()) if 'product_id' in group.columns else len(group),
                'n_predictions': len(group),
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2,
                'avg_demand': group['actual'].mean() if 'actual' in group.columns else 0
            })
        
        summary = pd.DataFrame(results)
        logger.info(f"  ✓ Analyzed {len(summary)} product groups")
        
        return summary
    
    def analyze_by_region(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze performance by store region.
        """
        logger.info("\nAnalyzing by region...")
        
        if 'region' not in df.columns and 'store_id' in df.columns:
            # Infer region from store_id (placeholder logic)
            df['region'] = df['store_id'].astype(str).str[0]
        elif 'region' not in df.columns:
            df['region'] = 'Default'
        
        groups = df.groupby('region')
        results = []
        
        for name, group in groups:
            if 'actual' in group.columns and 'predicted' in group.columns:
                mae = np.mean(np.abs(group['actual'] - group['predicted']))
                mape = np.mean(np.abs((group['actual'] - group['predicted']) / group['actual'])) * 100
            else:
                mae, mape = 0, 0
            
            results.append({
                'region': name,
                'n_stores': len(group['store_id'].unique()) if 'store_id' in group.columns else 1,
                'n_predictions': len(group),
                'mae': mae,
                'mape': mape,
                'total_demand': group['actual'].sum() if 'actual' in group.columns else 0
            })
        
        summary = pd.DataFrame(results)
        logger.info(f"  ✓ Analyzed {len(summary)} regions")
        
        return summary
    
    def scenario_analysis(
        self,
        df: pd.DataFrame,
        scenarios: Dict[str, Dict] = None
    ) -> pd.DataFrame:
        """
        Run what-if scenarios.
        
        Example scenarios:
        - Best case: demand +20%, price -10%
        - Worst case: demand -20%, price +10%
        - Promotion: price -30%, demand +50%
        """
        logger.info("\nRunning scenario analysis...")
        
        if scenarios is None:
            scenarios = {
                'baseline': {'demand_change': 0, 'price_change': 0},
                'best_case': {'demand_change': 0.2, 'price_change': -0.1},
                'worst_case': {'demand_change': -0.2, 'price_change': 0.1},
                'promotion': {'demand_change': 0.5, 'price_change': -0.3}
            }
        
        results = []
        
        for scenario_name, params in scenarios.items():
            df_scenario = df.copy()
            
            # Apply changes
            if 'predicted' in df_scenario.columns:
                df_scenario['predicted_adj'] = df_scenario['predicted'] * (1 + params['demand_change'])
            
            if 'price' in df_scenario.columns:
                df_scenario['price_adj'] = df_scenario['price'] * (1 + params['price_change'])
            
            # Calculate revenue impact
            if 'predicted_adj' in df_scenario.columns and 'price_adj' in df_scenario.columns:
                revenue_baseline = (df['predicted'] * df['price']).sum() if 'price' in df.columns else 0
                revenue_scenario = (df_scenario['predicted_adj'] * df_scenario['price_adj']).sum()
                revenue_change = ((revenue_scenario - revenue_baseline) / revenue_baseline * 100) if revenue_baseline > 0 else 0
            else:
                revenue_change = 0
            
            results.append({
                'scenario': scenario_name,
                'demand_change': f"{params['demand_change']:+.0%}",
                'price_change': f"{params['price_change']:+.0%}",
                'revenue_change': f"{revenue_change:+.1f}%",
                'avg_demand': df_scenario['predicted_adj'].mean() if 'predicted_adj' in df_scenario.columns else 0
            })
        
        summary = pd.DataFrame(results)
        logger.info(f"  ✓ Analyzed {len(summary)} scenarios")
        
        return summary
    
    def generate_sensitivity_report(
        self,
        predictions_df: pd.DataFrame,
        output_dir: str = 'reports/sensitivity'
    ):
        """
        Generate complete sensitivity analysis report.
        """
        logger.info("\n" + "="*70)
        logger.info("SENSITIVITY ANALYSIS REPORT")
        logger.info("="*70)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # By product group
        product_summary = self.analyze_by_product_group(predictions_df)
        product_summary.to_csv(f"{output_dir}/by_product_group.csv", index=False)
        logger.info(f"\n✅ Product group analysis saved")
        
        # By region
        region_summary = self.analyze_by_region(predictions_df)
        region_summary.to_csv(f"{output_dir}/by_region.csv", index=False)
        logger.info(f"✅ Region analysis saved")
        
        # Scenario analysis
        scenario_summary = self.scenario_analysis(predictions_df)
        scenario_summary.to_csv(f"{output_dir}/scenarios.csv", index=False)
        logger.info(f"✅ Scenario analysis saved")
        
        # Combined summary
        with open(f"{output_dir}/summary.txt", 'w') as f:
            f.write("SENSITIVITY ANALYSIS SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write("BY PRODUCT GROUP:\n")
            f.write(product_summary.to_string(index=False))
            f.write("\n\n")
            
            f.write("BY REGION:\n")
            f.write(region_summary.to_string(index=False))
            f.write("\n\n")
            
            f.write("SCENARIO ANALYSIS:\n")
            f.write(scenario_summary.to_string(index=False))
        
        logger.info(f"✅ Summary report saved: {output_dir}/summary.txt")
        
        return {
            'product_groups': product_summary,
            'regions': region_summary,
            'scenarios': scenario_summary
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Sample data
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'product_id': [f'P{i:03d}' for i in range(n)],
        'product_category': np.random.choice(['Fresh', 'Dairy', 'Packaged', 'Frozen'], n),
        'store_id': [f'S{i:02d}' for i in np.random.randint(1, 11, n)],
        'region': np.random.choice(['North', 'South', 'Central'], n),
        'actual': np.random.randint(50, 200, n),
        'predicted': np.random.randint(45, 205, n),
        'price': np.random.randint(5000, 15000, n)
    })
    
    # Run analysis
    analyzer = SensitivityAnalyzer()
    results = analyzer.generate_sensitivity_report(df)
    
    print("\n" + "="*70)
    print("SAMPLE RESULTS")
    print("="*70)
    print("\nBy Product Group:")
    print(results['product_groups'])
    print("\nBy Region:")
    print(results['regions'])
    print("\nScenario Analysis:")
    print(results['scenarios'])
