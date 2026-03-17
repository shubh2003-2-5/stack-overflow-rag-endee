import os

import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import MODEL_NAME, PROCESSED_DATA_PATH


def embed_to_disk(
    input_csv: str = PROCESSED_DATA_PATH,
    output_path: str = os.path.join(os.path.dirname(PROCESSED_DATA_PATH), "embeddings.parquet"),
    batch_size: int = 256,
):
    """Generate embeddings for title+body and store locally.

    This lets you run the embedding step once and ingest into Endee later.
    """

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Sampled CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, dtype=str).fillna("")
    df["text"] = df["title"].astype(str) + "\n\n" + df["body"].astype(str)

    model = SentenceTransformer(MODEL_NAME)

    embeddings = []
    for start in range(0, len(df), batch_size):
        batch_texts = df.loc[start : start + batch_size - 1, "text"].tolist()
        batch_embeds = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=True)
        embeddings.extend(batch_embeds.tolist())

    df["embedding"] = embeddings

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved embeddings to {output_path} (rows={len(df)})")


if __name__ == "__main__":
    embed_to_disk()
