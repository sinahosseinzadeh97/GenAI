"""Chain-of-Thought prompt templates for the LegalRAGAgent.

All templates follow the structured CoT format::

    [UNDERSTAND] → [RETRIEVE] → [REASON] → [ANSWER] → [CITE]

Functions
---------
build_prompt
    Assemble a list of OpenAI-style chat messages from a query + contexts.
"""

from __future__ import annotations

from src.vectorstore.schema import SearchResult

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are LexReview, an expert legal document analysis assistant.
Your role is to answer questions about legal contracts and documents with
precision, grounding every claim in the provided source excerpts.

Rules:
1. Base your answer ONLY on the provided context passages.
2. If the context does not contain enough information, say so explicitly.
3. Always cite specific passages using [CHUNK: <chunk_id>] inline notation.
4. Use Chain-of-Thought reasoning — show your reasoning steps clearly.
5. Be concise but thorough. Legal precision matters.
"""

# ── CoT template ──────────────────────────────────────────────────────────────

_COT_TEMPLATE = """\
## Legal Query
{query}

## Retrieved Context Passages
{context_block}

## Instructions
Reason through the following steps, then provide your final answer.

[UNDERSTAND] What is the user asking about? Identify key legal concepts.
[RETRIEVE] Which of the context passages are most relevant and why?
[REASON] How do the relevant passages answer the question? Note any ambiguities.
[ANSWER] Provide a clear, precise answer grounded in the context.
[CITE] List the chunk IDs you relied on: [CHUNK: <id>], [CHUNK: <id>], ...

Begin your response:
"""


def _format_context(results: list[SearchResult]) -> str:
    """Format retrieved chunks into a numbered context block.

    Args:
        results: Reranked search results to include in the prompt.

    Returns:
        Multi-line string with each chunk prefixed by its ID and rank.
    """
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        source = r.metadata.get("source", "unknown")
        lines.append(
            f"[{i}] CHUNK_ID={r.chunk_id} | SOURCE={source} | SCORE={r.score:.3f}\n"
            f"{r.content.strip()}\n"
        )
    return "\n---\n".join(lines)


def build_prompt(
    query: str,
    contexts: list[SearchResult],
) -> list[dict[str, str]]:
    """Assemble OpenAI-style chat messages for the LegalRAGAgent.

    Args:
        query:    User's legal query string.
        contexts: Reranked retrieval results to include as context.

    Returns:
        List of ``{"role": ..., "content": ...}`` dicts ready for the
        OpenAI Chat Completions API.

    Example::

        messages = build_prompt("What is the termination clause?", results)
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages
        )
    """
    context_block = _format_context(contexts) if contexts else "(No context retrieved)"
    user_content = _COT_TEMPLATE.format(
        query=query,
        context_block=context_block,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
