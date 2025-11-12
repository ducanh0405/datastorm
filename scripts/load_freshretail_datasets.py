"""
FreshRetailNet-50K Dataset Loader
=================================
Load FreshRetailNet-50K dataset using HuggingFace datasets library.

Usage:
    python scripts/load_freshretail_datasets.py [--sample] [--max-rows N]
"""

import pandas as pd
from pathlib import Path
import json
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_NAME = "Dingdong-Inc/FreshRetailNet-50K"


def load_dataset(sample_mode: bool = False, max_rows: int = None) -> dict:
    """Load FreshRetailNet-50K dataset."""
    try:
        from datasets import load_dataset

        if sample_mode:
            logger.info("Loading SAMPLE dataset (10K rows)...")
            dataset = load_dataset(DATASET_NAME, split="train[:10000]")
            dataframes = {"train": dataset.to_pandas()}
        else:
            logger.info("Loading FULL dataset...")
            dataset = load_dataset(DATASET_NAME)

            dataframes = {}
            for split_name, split_data in dataset.items():
                df = split_data.to_pandas()
                if max_rows and len(df) > max_rows:
                    df = df.head(max_rows)
                    logger.info(f"Limited {split_name} to {max_rows} rows")
                dataframes[split_name] = df

        # Convert array columns to strings for CSV compatibility
        for split_name, df in dataframes.items():
            array_cols = ['hours_sale', 'hours_stock_status']
            for col in array_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)

        logger.info("Dataset loaded successfully!")
        return dataframes

    except ImportError:
        logger.error("❌ Install datasets: pip install datasets")
        return {}
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return {}


def clean_old_data(data_dir: Path) -> None:
    """Clean old Dunnhumby data files."""
    old_files = [
        "transaction_data.csv", "product.csv", "causal_data.csv",
        "campaign_desc.csv", "campaign_table.csv", "hh_demographic.csv",
        "coupon.csv", "coupon_redempt.csv",
        "freshretail_transactions.csv", "freshretail_transactions.parquet",
        "freshretail_metadata.json"
    ]

    cleaned = []
    for filename in old_files:
        file_path = data_dir / filename
        if file_path.exists():
            file_path.unlink()
            cleaned.append(filename)
            logger.info(f"Removed old file: {filename}")

    if cleaned:
        logger.info(f"🗑️ Cleaned {len(cleaned)} old files")
    else:
        logger.info("ℹ️ No old files to clean")


def save_datasets(dataframes: Dict[str, pd.DataFrame], data_dir: Path) -> bool:
    """Save all dataset splits to files."""
    try:
        total_rows = 0

        for split_name, df in dataframes.items():
            logger.info(f"Saving {split_name} split...")

            # Save as CSV (human readable)
            csv_file = data_dir / f"freshretail_{split_name}.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"✓ Saved CSV: {csv_file}")

            # Save as Parquet (efficient storage)
            parquet_file = data_dir / f"freshretail_{split_name}.parquet"
            df.to_parquet(parquet_file, index=False)
            logger.info(f"✓ Saved Parquet: {parquet_file}")

            total_rows += len(df)

        # Create comprehensive metadata
        metadata = {
            "dataset": "FreshRetailNet-50K",
            "source": "Dingdong-Inc/FreshRetailNet-50K",
            "load_method": "huggingface_datasets",
            "splits": list(dataframes.keys()),
            "total_rows": total_rows,
            "columns": list(dataframes[list(dataframes.keys())[0]].columns) if dataframes else [],
            "splits_info": {
                split_name: {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
                } for split_name, df in dataframes.items()
            },
            "column_dtypes": {
                split_name: df.dtypes.astype(str).to_dict()
                for split_name, df in dataframes.items()
            },
            "sample_data": {
                split_name: df.head(3).to_dict('records')
                for split_name, df in dataframes.items()
            }
        }

        metadata_file = data_dir / "freshretail_datasets_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"✓ Saved metadata: {metadata_file}")

        return True

    except Exception as e:
        logger.error(f"Failed to save datasets: {e}")
        return False


def show_dataset_info(dataframes: Dict[str, pd.DataFrame]) -> None:
    """Display information about the loaded datasets."""
    print("\n" + "=" * 70)
    print("📊 FRESHRETAILNET-50K DATASET INFORMATION")
    print("=" * 70)

    for split_name, df in dataframes.items():
        print(f"\n🔹 {split_name.upper()} SPLIT:")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Memory: {df.memory_usage(deep=True).sum() / (1024 * 1024):.1f} MB")

        print("   Columns:")
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            print(f"{i:2d}. {col}: {dtype}")
        print("   Sample data:")
        print(df.head(2).to_string(index=False))

    print(f"\n📈 TOTAL: {sum(len(df) for df in dataframes.values()):,} rows across {len(dataframes)} splits")


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Load FreshRetailNet-50K dataset")
    parser.add_argument("--sample", action="store_true", help="Load only 10K sample rows")
    parser.add_argument("--max-rows", type=int, help="Maximum rows per split")
    args = parser.parse_args()

    print("=" * 60)
    print("FRESHRETAILNET-50K DATASET LOADER")
    print("=" * 60)

    # Check dependencies
    try:
        import datasets
        logger.info(f"✅ datasets library v{datasets.__version__}")
    except ImportError:
        logger.error("❌ Install: pip install datasets")
        return

    # Setup directories
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "2_raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📁 Data directory: {data_dir}")

    # Clean old data
    logger.info("🧹 Cleaning old files...")
    clean_old_data(data_dir)

    # Load dataset
    logger.info("📥 Loading dataset...")
    if args.sample:
        logger.info("Using SAMPLE mode (10K rows)")
    elif args.max_rows:
        logger.info(f"Limited to {args.max_rows} rows per split")

    dataframes = load_dataset(sample_mode=args.sample, max_rows=args.max_rows)

    if not dataframes:
        logger.error("❌ Dataset loading failed")
        return

    # Display info
    show_dataset_info(dataframes)

    # Save datasets
    logger.info("💾 Saving to disk...")
    if save_datasets(dataframes, data_dir):
        print("\n✅ SUCCESS!")
        print(f"📁 Files saved to: {data_dir}")

        # Show file sizes
        saved_files = list(data_dir.glob("freshretail_*"))
        print(f"\n📋 Created {len(saved_files)} files:")
        for f in sorted(saved_files):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   - {f.name}: {size_mb:.1f} MB")

        # Summary
        total_rows = sum(len(df) for df in dataframes.values())
        print(f"\n📊 Summary: {total_rows:,} rows, {len(dataframes)} splits")

        print("\n🚀 Next steps:")
        print("   python scripts/export_freshretail_data.py")
        print("   python -c \"import pandas as pd; pd.read_csv('data/2_raw/freshretail_train.csv').head()\"")
    else:
        logger.error("❌ Save failed")


if __name__ == "__main__":
    main()
