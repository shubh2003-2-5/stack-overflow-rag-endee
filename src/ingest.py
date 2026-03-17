import os
import uuid

import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

from .config import (
    COLLECTION_NAME,
    ENDPOINT,
    ENDPOINT_TOKEN,
    MODEL_NAME,
    PROCESSED_DATA_PATH,
)
from .utils import chunked

try:
    from endee import Endee, Precision
    from endee.exceptions import ConflictException
except ImportError as e:
    raise ImportError(
        "Please install endee (pip install endee) to use ingestion."
    ) from e


def _get_client() -> Endee:
    """Create an Endee client instance using configured endpoint/token."""

    kwargs = {}
    if ENDPOINT:
        kwargs["endpoint"] = ENDPOINT
    if ENDPOINT_TOKEN:
        kwargs["token"] = ENDPOINT_TOKEN

    # return Endee(**kwargs)
    return Endee()


def _ensure_index(client: Endee, index_name: str, dimension: int):
    """Create index if it does not exist."""
    try:
        client.create_index(
            name=index_name,
            dimension=dimension,
            space_type="cosine",
            precision=Precision.FLOAT16,
        )
    except ConflictException:
        # Index already exists, that's fine
        pass
    except Exception as e:
        # Re-raise other exceptions (e.g., connection issues)
        raise e


def ingest(
    csv_path: str = PROCESSED_DATA_PATH,
    index_name: str = COLLECTION_NAME,
    batch_size: int = 256,
):
    """Ingest sampled CSV into Endee index.

    If embeddings.parquet exists, use pre-computed embeddings.
    Otherwise, compute embeddings on the fly.
    """

    embeddings_path = os.path.join(os.path.dirname(PROCESSED_DATA_PATH), "embeddings.parquet")

    if os.path.exists(embeddings_path):
        # Use pre-computed embeddings
        print("Using pre-computed embeddings from embeddings.parquet")
        df = pd.read_parquet(embeddings_path)
        # Ensure embeddings are lists
        df["embedding"] = df["embedding"].apply(lambda x: x if isinstance(x, list) else x.tolist())
    else:
        # Compute embeddings on the fly
        print("Computing embeddings on the fly")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Sampled data not found: {csv_path}")

        df = pd.read_csv(csv_path, dtype=str).fillna("")
        df["text"] = df["title"].astype(str) + "\n\n" + df["body"].astype(str)

        model = SentenceTransformer(MODEL_NAME)
        embeddings = []
        for start in range(0, len(df), batch_size):
            batch_texts = df.loc[start : start + batch_size - 1, "text"].tolist()
            batch_embeds = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
            embeddings.extend(batch_embeds.tolist())
        df["embedding"] = embeddings

    client = _get_client()
    _ensure_index(client, index_name, dimension=384)  # Fixed dimension for all-MiniLM-L6-v2

    index = client.get_index(name=index_name)

    # Ingest in batches
    for batch_num, batch in enumerate(chunked(df.to_dict(orient="records"), batch_size), start=1):
        ops = []
        for row in batch:
            ops.append(
                {
                    "id": row.get("id") or str(uuid.uuid4()),
                    "vector": row["embedding"],
                    "meta": {
                        "title": row.get("title", ""),
                        "tags": row.get("tags", ""),
                        "text": row.get("text", ""),
                    },
                }
            )

        index.upsert(ops)
        print(f"Ingested batch {batch_num} ({len(ops)} vectors)")

    print("Ingestion complete")


if __name__ == "__main__":
    ingest()
