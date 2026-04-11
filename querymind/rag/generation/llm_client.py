import os
import anthropic

def generate_answer(query: str, contexts: list[dict]) -> str:
    client = anthropic.Anthropic()
    
    context_text = ""
    for c in contexts:
        context_text += f"\n[Source: {c['filename']} (Page {c['page_number']})]\n{c['content']}\n"
    
    prompt = f"""You are a helpful assistant for supplier contracts.
Answer based on the provided context only.
Always cite sources as [Source: filename (Page N)].
If context doesn't contain the answer, say so.

Context:
{context_text}

Question:
{query}"""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
