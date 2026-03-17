import os
from typing import List, Dict

from .config import OPENAI_API_KEY, OPENAI_MODEL, TOP_K
from .retrieve import retrieve

try:
    import openai
except ImportError:
    openai = None


def _build_context(docs: List[Dict], max_chars: int = 3000) -> str:
    """Build a single context string from retrieved documents."""
    lines = []
    total = 0
    for doc in docs:
        meta = doc.get("meta", {})
        title = meta.get("title") or "(no title)"
        snippet = (meta.get("text") or "").replace("\n", " ")
        snippet = snippet[:500]
        block = f"Title: {title}\nSnippet: {snippet}\nScore: {doc.get('similarity'):.4f}\n"
        if total + len(block) > max_chars:
            break
        lines.append(block)
        total += len(block)
    return "\n---\n".join(lines)


def answer_query(query: str, top_k: int = TOP_K) -> Dict[str, str]:
    """Retrieve context and build an answer (using OpenAI if available)."""
    docs = retrieve(query, top_k=top_k)
    context = _build_context(docs)

    if not query or not query.strip():
        return {"answer": "Please provide a non-empty query.", "source": ""}

    if OPENAI_API_KEY and openai:
        openai.api_key = OPENAI_API_KEY
        prompt = (
            "You are an assistant that answers questions using the provided context.\n"
            "Use the context to answer the question concisely. If the answer is not in the context, say you don\'t know.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        try:
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=256,
                temperature=0.2,
            )
            answer = resp["choices"][0]["message"]["content"].strip()
            return {"answer": answer, "source": "openai"}
        except Exception as e:
            return {
                "answer": f"OpenAI call failed: {e}. Returning retrieved passages.",
                "source": "openai_error",
            }

    # Fallback: return top results as a simple answer
    if not docs:
        return {"answer": "No documents found for that query.", "source": "fallback"}

    snippets = []
    for d in docs:
        meta = d.get("meta", {})
        title = meta.get("title") or "(no title)"
        text = (meta.get("text") or "").replace("\n", " ")
        snippets.append(f"* {title}: {text[:300]}")

    answer = (
        "No OpenAI API key found. Here are the top retrieved documents and their snippets:\n"
        + "\n".join(snippets)
    )
    return {"answer": answer, "source": "fallback"}
