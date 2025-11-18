#!/usr/bin/env python3
"""
Metrics Validator for Module 4: LLM Insights
============================================

Validates all metrics before using in insight generation.
Ensures data quality and prevents silent failures.

Author: SmartGrocy Team
Date: 2025-11-18
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when metric validation fails"""
    pass


@dataclass
class ValidationResult:
    """Result of metric validation"""
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    validated_data: dict | None = None

    def __str__(self):
        if self.is_valid:
            return "[PASS] Validation passed"
        return f"[FAIL] Validation failed: {', '.join(self.errors)}"


class MetricsValidator:
    """
    Comprehensive metrics validator for Module 4.

    Validates:
    - Forecast metrics (Q05-Q95, trends)
    - Inventory metrics (safety stock, ROP, EOQ)
    - Pricing metrics (prices, margins, discounts)
    - SHAP values (feature importance)
    """

    # Validation thresholds
    FORECAST_MIN_VALUE = 0.0
    FORECAST_MAX_VALUE = 1_000_000.0
    UNCERTAINTY_MAX_PCT = 200.0  # Max 200% uncertainty
    PRICE_MIN_VALUE = 0.01
    PRICE_MAX_VALUE = 1_000_000.0
    MARGIN_MIN_VALUE = -1.0  # Allow negative margin (loss)
    MARGIN_MAX_VALUE = 1.0   # Max 100% margin
    DISCOUNT_MIN_VALUE = 0.0
    DISCOUNT_MAX_VALUE = 1.0  # Max 100% discount

    @staticmethod
    def validate_forecast_metrics(data: dict) -> ValidationResult:
        """
        Validate forecast metrics from Module 1.

        Args:
            data: Dict with forecast metrics

        Returns:
            ValidationResult with validation status and cleaned data

        Required fields:
            - q50: Median forecast (required)
            - q05: Lower bound (optional, will use q50*0.7 if missing)
            - q95: Upper bound (optional, will use q50*1.3 if missing)

        Calculated fields:
            - uncertainty: q95 - q50
            - uncertainty_pct: (uncertainty / q50) * 100
            - confidence_level: HIGH/MODERATE/LOW based on uncertainty
        """
        errors = []
        warnings = []
        validated = data.copy()

        # Check required field: q50
        if 'q50' not in data:
            errors.append("Missing required field: q50 (median forecast)")
            return ValidationResult(False, errors, warnings)

        # Validate q50
        q50 = data['q50']
        if not isinstance(q50, int | float):
            errors.append(f"q50 must be numeric, got {type(q50).__name__}")
        elif q50 < MetricsValidator.FORECAST_MIN_VALUE:
            errors.append(f"q50 cannot be negative: {q50}")
        elif q50 > MetricsValidator.FORECAST_MAX_VALUE:
            warnings.append(f"q50 unusually high: {q50:,.0f}")

        # Validate or estimate q05
        if 'q05' not in data:
            validated['q05'] = q50 * 0.7  # Conservative estimate
            warnings.append("q05 missing, estimated as q50 * 0.7")
        else:
            q05 = data['q05']
            if not isinstance(q05, int | float):
                errors.append(f"q05 must be numeric, got {type(q05).__name__}")
            elif q05 < 0:
                errors.append(f"q05 cannot be negative: {q05}")
            elif q05 > q50:
                warnings.append(f"q05 ({q05:.1f}) should be < q50 ({q50:.1f})")

        # Validate or estimate q95
        if 'q95' not in data:
            validated['q95'] = q50 * 1.3  # Conservative estimate
            warnings.append("q95 missing, estimated as q50 * 1.3")
        else:
            q95 = data['q95']
            if not isinstance(q95, int | float):
                errors.append(f"q95 must be numeric, got {type(q95).__name__}")
            elif q95 < 0:
                errors.append(f"q95 cannot be negative: {q95}")
            elif q95 < q50:
                warnings.append(f"q95 ({q95:.1f}) should be > q50 ({q50:.1f})")

        # If errors, return early
        if errors:
            return ValidationResult(False, errors, warnings)

        # Calculate derived metrics
        try:
            q05_val = validated['q05']
            q50_val = validated['q50']
            q95_val = validated['q95']

            validated['uncertainty'] = q95_val - q50_val

            if q50_val > 0:
                validated['uncertainty_pct'] = (validated['uncertainty'] / q50_val) * 100
            else:
                validated['uncertainty_pct'] = 0.0
                warnings.append("q50 is zero, cannot calculate uncertainty_pct")

            # Confidence level
            unc_pct = validated['uncertainty_pct']
            if unc_pct < 10:
                validated['confidence_level'] = 'HIGH'
            elif unc_pct < 25:
                validated['confidence_level'] = 'MODERATE'
            else:
                validated['confidence_level'] = 'LOW'

            # Validate uncertainty
            if unc_pct > MetricsValidator.UNCERTAINTY_MAX_PCT:
                warnings.append(
                    f"Very high uncertainty: {unc_pct:.1f}% "
                    f"(forecast range: {q05_val:.1f} - {q95_val:.1f})"
                )

            # Trend indicators (optional)
            for field in ['vs_yesterday', 'vs_last_week', 'vs_monthly_avg']:
                if field in data and isinstance(data[field], int | float):
                    validated[field] = data[field]

        except Exception as e:
            errors.append(f"Error calculating derived metrics: {e}")
            return ValidationResult(False, errors, warnings)

        return ValidationResult(True, errors, warnings, validated)

    @staticmethod
    def validate_inventory_metrics(data: dict) -> ValidationResult:
        """
        Validate inventory metrics from Module 2.

        Required fields:
            - current_inventory: Current stock level
            - safety_stock: Safety stock level
            - reorder_point: Reorder point

        Optional fields:
            - eoq: Economic order quantity
            - should_reorder: Boolean flag
            - stockout_risk_pct: Stockout risk %
            - overstock_risk_pct: Overstock risk %
        """
        errors = []
        warnings = []
        validated = data.copy()

        # Validate current_inventory
        if 'current_inventory' not in data:
            errors.append("Missing required field: current_inventory")
        else:
            inv = data['current_inventory']
            if not isinstance(inv, int | float):
                errors.append(f"current_inventory must be numeric, got {type(inv).__name__}")
            elif inv < 0:
                errors.append(f"current_inventory cannot be negative: {inv}")
            validated['current_inventory'] = float(inv) if not errors else 0.0

        # Validate safety_stock
        if 'safety_stock' not in data:
            warnings.append("safety_stock missing, using 0")
            validated['safety_stock'] = 0.0
        else:
            ss = data['safety_stock']
            if not isinstance(ss, int | float):
                errors.append(f"safety_stock must be numeric, got {type(ss).__name__}")
            elif ss < 0:
                errors.append(f"safety_stock cannot be negative: {ss}")
            validated['safety_stock'] = float(ss) if not errors else 0.0

        # Validate reorder_point
        if 'reorder_point' not in data:
            warnings.append("reorder_point missing, using safety_stock")
            validated['reorder_point'] = validated.get('safety_stock', 0.0)
        else:
            rop = data['reorder_point']
            if not isinstance(rop, int | float):
                errors.append(f"reorder_point must be numeric, got {type(rop).__name__}")
            elif rop < 0:
                errors.append(f"reorder_point cannot be negative: {rop}")
            validated['reorder_point'] = float(rop) if not errors else 0.0

        # Validate optional fields
        if 'eoq' in data:
            eoq = data['eoq']
            if isinstance(eoq, int | float) and eoq >= 0:
                validated['eoq'] = float(eoq)

        if 'stockout_risk_pct' in data:
            risk = data['stockout_risk_pct']
            if isinstance(risk, int | float):
                validated['stockout_risk_pct'] = max(0.0, min(100.0, float(risk)))

        if 'overstock_risk_pct' in data:
            risk = data['overstock_risk_pct']
            if isinstance(risk, int | float):
                validated['overstock_risk_pct'] = max(0.0, min(100.0, float(risk)))

        # should_reorder logic
        if 'should_reorder' not in data:
            # Calculate based on current_inventory vs reorder_point
            validated['should_reorder'] = (
                validated['current_inventory'] <= validated['reorder_point']
            )
        else:
            validated['should_reorder'] = bool(data['should_reorder'])

        # Calculate inventory ratio if forecast available
        if 'avg_demand' in data or 'forecast_q50' in data:
            avg_demand = data.get('avg_demand', data.get('forecast_q50', 1.0))
            if avg_demand > 0:
                validated['inventory_ratio'] = validated['current_inventory'] / avg_demand

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            validated_data=validated
        )

    @staticmethod
    def validate_pricing_metrics(data: dict) -> ValidationResult:
        """
        Validate pricing metrics from Module 3.

        Required fields:
            - current_price: Current selling price
            - unit_cost: Cost per unit

        Optional fields:
            - recommended_price: New recommended price
            - discount_pct: Discount percentage
            - action: Pricing action

        Calculated fields:
            - current_margin: (current_price - unit_cost) / current_price
            - new_margin: (recommended_price - unit_cost) / recommended_price
            - price_change_pct: ((new - old) / old) * 100
        """
        errors = []
        warnings = []
        validated = data.copy()

        # Validate current_price
        if 'current_price' not in data:
            errors.append("Missing required field: current_price")
        else:
            price = data['current_price']
            if not isinstance(price, int | float):
                errors.append(f"current_price must be numeric, got {type(price).__name__}")
            elif price < MetricsValidator.PRICE_MIN_VALUE:
                errors.append(f"current_price too low: {price}")
            elif price > MetricsValidator.PRICE_MAX_VALUE:
                warnings.append(f"current_price unusually high: {price:,.2f}")
            validated['current_price'] = float(price) if not errors else 0.0

        # Validate unit_cost
        if 'unit_cost' not in data:
            warnings.append("unit_cost missing, assuming 50% of current_price")
            validated['unit_cost'] = validated['current_price'] * 0.5
        else:
            cost = data['unit_cost']
            if not isinstance(cost, int | float):
                errors.append(f"unit_cost must be numeric, got {type(cost).__name__}")
            elif cost < 0:
                errors.append(f"unit_cost cannot be negative: {cost}")
            validated['unit_cost'] = float(cost) if not errors else 0.0

        # Validate recommended_price (optional)
        if 'recommended_price' in data:
            rec_price = data['recommended_price']
            if not isinstance(rec_price, int | float):
                warnings.append("recommended_price must be numeric, ignoring")
            elif rec_price < MetricsValidator.PRICE_MIN_VALUE:
                warnings.append(f"recommended_price too low: {rec_price}")
            else:
                validated['recommended_price'] = float(rec_price)
        else:
            validated['recommended_price'] = validated['current_price']

        # Validate discount_pct (optional)
        if 'discount_pct' in data:
            disc = data['discount_pct']
            if isinstance(disc, int | float):
                if disc < MetricsValidator.DISCOUNT_MIN_VALUE or disc > MetricsValidator.DISCOUNT_MAX_VALUE:
                    warnings.append(f"discount_pct out of range [0-1]: {disc}")
                validated['discount_pct'] = max(0.0, min(1.0, float(disc)))

        # If errors, return early
        if errors:
            return ValidationResult(False, errors, warnings)

        # Calculate margins
        try:
            current_price = validated['current_price']
            unit_cost = validated['unit_cost']
            recommended_price = validated['recommended_price']

            # Current margin
            if current_price > 0:
                validated['current_margin'] = (current_price - unit_cost) / current_price
            else:
                validated['current_margin'] = 0.0
                warnings.append("current_price is zero, cannot calculate margin")

            # New margin
            if recommended_price > 0:
                validated['new_margin'] = (recommended_price - unit_cost) / recommended_price
            else:
                validated['new_margin'] = validated['current_margin']

            # Price change
            if current_price > 0:
                validated['price_change'] = recommended_price - current_price
                validated['price_change_pct'] = (validated['price_change'] / current_price) * 100
            else:
                validated['price_change'] = 0.0
                validated['price_change_pct'] = 0.0

            # Validate margins
            if validated['current_margin'] < 0:
                warnings.append(
                    f"Current pricing results in loss: margin = {validated['current_margin']:.1%}"
                )

            if validated['new_margin'] < 0:
                warnings.append(
                    f"Recommended pricing results in loss: new margin = {validated['new_margin']:.1%}"
                )

        except Exception as e:
            errors.append(f"Error calculating pricing metrics: {e}")
            return ValidationResult(False, errors, warnings)

        return ValidationResult(True, errors, warnings, validated)

    @staticmethod
    def validate_shap_values(data: dict) -> ValidationResult:
        """
        Validate SHAP values for feature importance.

        Args:
            data: Dict mapping feature names to SHAP values

        Returns:
            ValidationResult with top features sorted by absolute value
        """
        errors = []
        warnings = []

        if not data:
            warnings.append("No SHAP values provided")
            return ValidationResult(True, errors, warnings, {})

        validated = {}

        for feature, value in data.items():
            if not isinstance(value, int | float):
                warnings.append(f"SHAP value for '{feature}' is not numeric: {type(value).__name__}")
                continue

            # Convert to float
            validated[feature] = float(value)

        # Sort by absolute value
        sorted_features = sorted(
            validated.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        validated_sorted = dict(sorted_features)

        return ValidationResult(True, errors, warnings, validated_sorted)

    @staticmethod
    def validate_comprehensive(
        forecast_data: dict,
        inventory_data: dict,
        pricing_data: dict,
        shap_values: dict | None = None
    ) -> tuple[ValidationResult, ValidationResult, ValidationResult, ValidationResult]:
        """
        Validate all metrics comprehensively.

        Returns:
            Tuple of (forecast_result, inventory_result, pricing_result, shap_result)
        """
        forecast_result = MetricsValidator.validate_forecast_metrics(forecast_data)
        inventory_result = MetricsValidator.validate_inventory_metrics(inventory_data)
        pricing_result = MetricsValidator.validate_pricing_metrics(pricing_data)
        shap_result = MetricsValidator.validate_shap_values(shap_values or {})

        return forecast_result, inventory_result, pricing_result, shap_result


if __name__ == "__main__":
    # Test validation
    logging.basicConfig(level=logging.INFO)

    # Test forecast validation
    print("\n" + "="*70)
    print("TEST 1: FORECAST METRICS VALIDATION")
    print("="*70)

    forecast_data = {
        'q50': 150.0,
        'q05': 100.0,
        'q95': 200.0,
        'vs_yesterday': 15.5
    }

    result = MetricsValidator.validate_forecast_metrics(forecast_data)
    print(f"\nResult: {result}")
    if result.is_valid:
        print(f"Validated data: {result.validated_data}")
        print(f"Warnings: {result.warnings}")

    # Test with missing fields
    print("\n" + "="*70)
    print("TEST 2: MISSING FIELDS")
    print("="*70)

    incomplete_data = {'q50': 150.0}
    result = MetricsValidator.validate_forecast_metrics(incomplete_data)
    print(f"\nResult: {result}")
    print(f"Warnings: {result.warnings}")
    print(f"Estimated q05: {result.validated_data.get('q05', 'N/A')}")
    print(f"Estimated q95: {result.validated_data.get('q95', 'N/A')}")

    # Test inventory validation
    print("\n" + "="*70)
    print("TEST 3: INVENTORY METRICS VALIDATION")
    print("="*70)

    inventory_data = {
        'current_inventory': 120.0,
        'safety_stock': 30.0,
        'reorder_point': 100.0,
        'stockout_risk_pct': 45.0
    }

    result = MetricsValidator.validate_inventory_metrics(inventory_data)
    print(f"\nResult: {result}")
    print(f"Should reorder: {result.validated_data.get('should_reorder', 'N/A')}")

    # Test pricing validation
    print("\n" + "="*70)
    print("TEST 4: PRICING METRICS VALIDATION")
    print("="*70)

    pricing_data = {
        'current_price': 50000.0,
        'unit_cost': 30000.0,
        'recommended_price': 45000.0,
        'discount_pct': 0.10
    }

    result = MetricsValidator.validate_pricing_metrics(pricing_data)
    print(f"\nResult: {result}")
    if result.is_valid:
        print(f"Current margin: {result.validated_data['current_margin']:.1%}")
        print(f"New margin: {result.validated_data['new_margin']:.1%}")
        print(f"Price change: {result.validated_data['price_change_pct']:.1f}%")

    print("\n[SUCCESS] All validation tests completed")
