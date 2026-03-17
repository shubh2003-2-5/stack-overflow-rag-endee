import argparse
import os

import pandas as pd

from .config import PROCESSED_DATA_PATH, RAW_DATA_PATH, SAMPLE_SIZE
from .utils import reservoir_sample


def sample_csv(
    input_path: str = RAW_DATA_PATH,
    output_path: str = PROCESSED_DATA_PATH,
    sample_size: int = SAMPLE_SIZE,
):
    """Sample rows from a large CSV into a smaller file.

    Uses reservoir sampling to avoid loading the full dataset into memory.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    # Read in chunks and yield rows as dicts
    def iter_rows():
        # Use error-tolerant reading to handle malformed encoding in large CSVs
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            for chunk in pd.read_csv(
                f,
                chunksize=10000,
                dtype=str,
                engine="python",
            ):
                # Accept either lowercase or title-case column names
                cols = chunk.columns.str.lower()
                chunk.columns = cols
                chunk = chunk.dropna(subset=["title", "body"], how="any")
                for row in chunk.to_dict(orient="records"):
                    yield row

    sampled = reservoir_sample(iter_rows(), sample_size)

    if not sampled:
        raise ValueError("No rows were sampled. Check the input CSV and required columns.")

    df = pd.DataFrame(sampled)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sampled {len(df)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sample StackOverflow CSV data")
    parser.add_argument("--input", default=RAW_DATA_PATH, help="Path to raw CSV")
    parser.add_argument("--output", default=PROCESSED_DATA_PATH, help="Path to sampled CSV")
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE, help="Number of rows to sample")

    args = parser.parse_args()
    sample_csv(input_path=args.input, output_path=args.output, sample_size=args.size)


if __name__ == "__main__":
    main()
