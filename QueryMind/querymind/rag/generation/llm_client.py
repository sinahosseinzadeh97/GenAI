import os
from openai import OpenAI

def generate_answer(query: str, contexts: list[dict]) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    return response.choices[0].message.content
