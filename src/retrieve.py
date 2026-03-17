from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer

from .config import COLLECTION_NAME, MODEL_NAME, TOP_K

try:
    from endee import Endee
except ImportError as e:
    raise ImportError(
        "Please install endee (pip install endee) to use retrieval."
    ) from e


def _get_client():
    from .config import ENDPOINT, ENDPOINT_TOKEN

    kwargs = {}
    if ENDPOINT:
        kwargs["endpoint"] = ENDPOINT
    if ENDPOINT_TOKEN:
        kwargs["token"] = ENDPOINT_TOKEN

    return Endee()


def retrieve(
    query: str, top_k: Optional[int] = None, collection_name: str = COLLECTION_NAME
) -> List[Dict]:
    """Retrieve top-k documents for a query from Endee."""

    if not query or not query.strip():
        return []

    top_k = top_k or TOP_K

    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(query, convert_to_numpy=True).tolist()

    client = _get_client()
    index = client.get_index(name=collection_name)

    results = index.query(vector=query_embedding, top_k=top_k)

    # Each result contains id, similarity, distance, meta, norm, vector
    return results
