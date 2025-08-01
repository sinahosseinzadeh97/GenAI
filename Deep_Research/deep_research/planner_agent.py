"""Planner Agent – query decomposition

Generates a structured web‑search plan (3‑6 focused queries) that feeds the
research pipeline. This revision emphasises **output reliability**, **prompt
clarity**, and minimal runtime overhead (no I/O).
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from .agents import Agent, ModelSettings

# ───────────────────────────────────────────────────────────────────────────
# Pydantic schemas – kept minimal for speed but strict enough to validate
# ───────────────────────────────────────────────────────────────────────────
class WebSearchItem(BaseModel):
    """Single search query + motivation."""

    query: str = Field(description="The search query (≤100 chars, no quotes)")
    reason: str = Field(description="Why this particular query advances the research goal")


class WebSearchPlan(BaseModel):
    """List wrapper so downstream agents can reference .searches."""

    searches: List[WebSearchItem] = Field(description="3–6 searches to perform")


# ───────────────────────────────────────────────────────────────────────────
# Prompt engineering – concise & deterministic
# ───────────────────────────────────────────────────────────────────────────
INSTRUCTIONS = (
    "You are an expert research strategist. Break the ❰User query❱ into 3–6 DISTINCT web searches\n"
    "that collectively provide comprehensive coverage (definitions, recent data, opposing views,\n"
    "expert analyses, statistics…).\n\n"
    "Output MUST be valid JSON **matching exactly** this schema (no extra keys, no Markdown):\n"
    "{\n"
    "  \"searches\": [\n"
    "    { \"query\": <string>, \"reason\": <string> },\n"
    "    …\n"
    "  ]\n"
    "}\n\n"
    "Guidelines:\n"
    "• Keep each `query` concise (≤100 characters) and remove redundant phrasing like ‘latest’.\n"
    "• Start each `reason` with an action verb (e.g. ‘Identify’, ‘Compare’, ‘Gather’).\n"
    "• Avoid overlap – each search should target a unique facet.\n"
    "• Do NOT wrap the JSON in code fences. Return JSON only."
)

# ───────────────────────────────────────────────────────────────────────────
# Agent instance – conservative temperature for stability
# ───────────────────────────────────────────────────────────────────────────
planner_agent = Agent(
    name="Planner Agent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=WebSearchPlan,
    model_settings=ModelSettings(
        temperature=0.3,
        top_p=0.9,
    ),
)
