import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def read_parquet_sample(path: str, nrows: int = 1000, columns: Optional[List[str]] = None) -> pd.DataFrame:
	"""
	Stream-read approximately `nrows` rows from a Parquet file without loading the entire file.
	If the file has multiple row groups, batches are accumulated until at least `nrows` rows.
	"""
	pf = pq.ParquetFile(path)

	if nrows <= 0:
		return pd.DataFrame(columns=[f.name for f in pf.schema])

	rows_needed = nrows
	batches: List[pa.RecordBatch] = []

	# Use a reasonable batch size to avoid over-reading; cap by nrows
	batch_size = max(1, min(10000, nrows))
	for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
		batches.append(batch)
		rows_needed -= batch.num_rows
		if rows_needed <= 0:
			break

	if not batches:
		# Empty file or no rows selected with provided columns
		if columns:
			return pd.DataFrame(columns=columns)
		return pd.DataFrame(columns=[f.name for f in pf.schema])

	table = pa.Table.from_batches(batches)
	if table.num_rows > nrows:
		table = table.slice(0, nrows)
	return table.to_pandas(types_mapper=None)  # rely on defaults


def main() -> None:
	parser = argparse.ArgumentParser(description="Peek a small sample of rows from a Parquet file efficiently.")
	parser.add_argument(
		"-p",
		"--path",
		type=str,
		default="data/3_processed/master_feature_table.parquet",
		help="Path to Parquet file.",
	)
	parser.add_argument(
		"-n",
		"--nrows",
		type=int,
		default=1000,
		help="Number of rows to read (approximate, capped).",
	)
	parser.add_argument(
		"-c",
		"--columns",
		type=str,
		default=None,
		help="Optional comma-separated list of columns to read.",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=str,
		default=None,
		help="Optional output file to save the sample (supports .csv or .parquet).",
	)
	parser.add_argument(
		"--show-dtypes",
		action="store_true",
		help="Print dtypes of the sample dataframe.",
	)

	args = parser.parse_args()

	cols: Optional[List[str]] = [c.strip() for c in args.columns.split(",")] if args.columns else None

	df = read_parquet_sample(args.path, nrows=args.nrows, columns=cols)
	print(f"Sampled shape: {df.shape}")
	print("Preview:")
	print(df.head(10))

	if args.show_dtypes:
		print("\nDtypes:")
		print(df.dtypes)

	if args.output:
		out_path = Path(args.output)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		suffix = out_path.suffix.lower()
		if suffix == ".csv":
			df.to_csv(out_path, index=False)
		elif suffix == ".parquet":
			df.to_parquet(out_path, index=False)
		else:
			# Default to CSV if unknown suffix
			df.to_csv(out_path.with_suffix(".csv"), index=False)
			out_path = out_path.with_suffix(".csv")
		print(f"\nSaved sample to: {out_path}")


if __name__ == "__main__":
	main()


