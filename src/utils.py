import math
from typing import Iterable, List, TypeVar

T = TypeVar("T")


def chunked(iterable: Iterable[T], size: int) -> Iterable[List[T]]:
    """Yield successive chunks from iterable."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def reservoir_sample(source_iterable: Iterable[T], k: int) -> List[T]:
    """Memory-efficient reservoir sampling of k items from an iterator."""
    import random

    reservoir: List[T] = []
    for i, item in enumerate(source_iterable, start=1):
        if i <= k:
            reservoir.append(item)
        else:
            j = random.randrange(i)
            if j < k:
                reservoir[j] = item
    return reservoir


def ensure_list(value):
    """Ensure value is a list (wrap scalar into list)."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
