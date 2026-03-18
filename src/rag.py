import os
from typing import List, Dict, Optional

from .config import OPENAI_API_KEY, OPENAI_MODEL, TOP_K
from .retrieve import retrieve

try:
    import openai
    from openai import AzureOpenAI
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


def answer_query(
    query: str,
    top_k: int = TOP_K,
    openai_api_key: Optional[str] = None,
) -> Dict[str, str]:
    """Retrieve context and build an answer (using OpenAI if available)."""
    docs = retrieve(query, top_k=top_k)
    context = _build_context(docs)
    if not query or not query.strip():
        return {"answer": "Please provide a non-empty query.", "source": ""}

    # Prefer API key passed in at runtime (e.g., from a UI field), otherwise fall back to env var.
    api_key = (openai_api_key.strip() if openai_api_key else None) or OPENAI_API_KEY
    if api_key and openai:
        print(f"API key found. Query and context will be sent to LLM for response generation")
        openai.api_key = api_key
        prompt = (
            "You are an assistant that answers questions using the provided context.\n"
            "Use the context to answer the question concisely. If the answer is not in the context, say you don\'t know.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        try:
            endpoint = "https://vikash0837-9992-resource.cognitiveservices.azure.com/"
            model_name = "gpt-4.1"
            deployment = "gpt-4.1-2"

            subscription_key = api_key
            api_version = "2024-12-01-preview"
            client = AzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    api_key=subscription_key,
                )
            
            resp = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=13107,
                temperature=1.0,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                model=deployment,
            )
            answer = resp.choices[0].message.content.strip()
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
