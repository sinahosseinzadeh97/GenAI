import json

INSIGHT_SYSTEM_PROMPT = """
You are a data analyst assistant.
You are given the results of a database query.
Your job is to explain what was found and provide a brief insight.

Respond ONLY with a valid JSON object. No markdown. No explanation outside JSON.
No ```json fences. Raw JSON only.

Format:
{
  "explanation": "one sentence describing what this query returns",
  "insight": "one notable finding from the data",
  "suggestion": "one follow-up question the user might want to ask"
}

Rules:
- Keep each field under 2 sentences
- Be specific, reference actual values from the data
- If row_count is 0, say so clearly in explanation
- Never invent data that is not in sample_rows
"""

def build_insight_prompt(
    nl_query: str,
    sql: str,
    row_count: int,
    sample_rows: list[dict]
) -> str:
    return (
        f"Original question: {nl_query}\n"
        f"SQL executed: {sql}\n"
        f"Total rows returned: {row_count}\n"
        f"Sample data (first 3 rows):\n{json.dumps(sample_rows[:3], indent=2)}"
    )
