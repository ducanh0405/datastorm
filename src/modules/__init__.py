#!/usr/bin/env python3
"""
SmartGrocy Business Modules
===========================

This package contains business logic modules for e-grocery optimization:

Module 1: Demand Forecasting (in src/pipelines/_03_model_training.py)
Module 2: Inventory Optimization (inventory_optimization.py)
Module 3: Dynamic Pricing (dynamic_pricing.py)
Module 4: LLM Insights (llm_insights.py)

Usage:
    from src.modules.inventory_optimization import InventoryOptimizer
    from src.modules.dynamic_pricing import DynamicPricingEngine
    from src.modules.llm_insights import LLMInsightGenerator

Author: SmartGrocy Team
Date: 2025-11-15
"""

__version__ = "1.0.0"
__all__ = [
    "InventoryOptimizer",
    "DynamicPricingEngine", 
    "LLMInsightGenerator"
]
