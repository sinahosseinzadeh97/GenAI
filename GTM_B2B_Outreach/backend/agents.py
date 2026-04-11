import json
from typing import Any, Dict, List, Optional

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.exa import ExaTools


def create_company_finder_agent() -> Agent:
    exa_tools = ExaTools(category="company")
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[exa_tools],
        session_id="gtm_outreach_company_finder",
        instructions=[
            "You are CompanyFinderAgent. Use ExaTools to search the web for companies that match the targeting criteria.",
            "Return ONLY valid JSON with key 'companies' as a list; respect the requested limit provided in the user prompt.",
            "Each item must have: name, website, why_fit (1-2 lines).",
        ],
    )


def create_contact_finder_agent() -> Agent:
    exa_tools = ExaTools()
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[exa_tools],
        session_id="gtm_outreach_contact_finder",
        instructions=[
            "You are ContactFinderAgent. Use ExaTools to find 1-2 relevant decision makers per company and their emails if available.",
            "Prioritize roles from Founder's Office, GTM (Marketing/Growth), Sales leadership, Partnerships/Business Development, and Product Marketing.",
            "Search queries can include patterns like '<Company> email format', 'contact', 'team', 'leadership', and role titles.",
            "If direct emails are not found, infer likely email using common formats (e.g., first.last@domain), but mark inferred=true.",
            "Return ONLY valid JSON with key 'companies' as a list; each has: name, contacts: [{full_name, title, email, inferred}]",
        ],
    )


def get_email_style_instruction(style_key: str) -> str:
    styles = {
        "Professional": "Style: Professional. Clear, respectful, and businesslike. Short paragraphs; no slang.",
        "Casual": "Style: Casual. Friendly, approachable, first-name basis. No slang or emojis; keep it human.",
        "Cold": "Style: Cold email. Strong hook in opening 2 lines, tight value proposition, minimal fluff, strong CTA.",
        "Consultative": "Style: Consultative. Insight-led, frames observed problems and tailored solution hypotheses; soft CTA.",
    }
    return styles.get(style_key, styles["Professional"])


def create_email_writer_agent(style_key: str = "Professional") -> Agent:
    style_instruction = get_email_style_instruction(style_key)
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[],
        session_id="gtm_outreach_email_writer",
        instructions=[
            "You are EmailWriterAgent. Write concise, personalized B2B outreach emails.",
            style_instruction,
            "Return ONLY valid JSON with key 'emails' as a list of items: {company, contact, subject, body}.",
            "Length: 120-160 words. Include 1-2 lines of strong personalization referencing research insights (company website and Reddit findings).",
            "CTA: suggest a short intro call; include sender company name and calendar link if provided.",
        ],
    )


def create_research_agent() -> Agent:
    """Agent to gather interesting insights from company websites and Reddit."""
    exa_tools = ExaTools()
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[exa_tools],
        session_id="gtm_outreach_researcher",
        instructions=[
            "You are ResearchAgent. For each company, collect concise, valuable insights from:",
            "1) Their official website (about, blog, product pages)",
            "2) Reddit discussions (site:reddit.com mentions)",
            "Summarize 2-4 interesting, non-generic points per company that a human would bring up in an email to show genuine effort.",
            "Return ONLY valid JSON with key 'companies' as a list; each has: name, insights: [strings].",
        ],
    )


def extract_json_or_raise(text: str) -> Dict[str, Any]:
    """Extract JSON from a model response. Handles various formats."""
    import re
    
    # Try direct parse first
    try:
        return json.loads(text)
    except:
        pass
    
    # Try to find JSON in markdown code block
    json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    if matches:
        try:
            return json.loads(matches[0])
        except:
            pass
    
    # Try to locate a JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except:
            pass
    
    # Last resort: try to fix common JSON issues
    try:
        # Remove trailing commas
        fixed = re.sub(r',\s*}', '}', text)
        fixed = re.sub(r',\s*]', ']', fixed)
        return json.loads(fixed)
    except:
        pass
    
    # Give up and return empty structure
    print(f"⚠️  WARNING: Could not parse JSON. Response: {text[:200]}...")
    return {"companies": [], "contacts": [], "research": [], "emails": []}


def run_company_finder(agent: Agent, target_desc: str, offering_desc: str, max_companies: int) -> List[Dict[str, str]]:
    prompt = (
        f"Find exactly {max_companies} companies that are a strong B2B fit given the user inputs.\n"
        f"Targeting: {target_desc}\n"
        f"Offering: {offering_desc}\n"
        "For each, provide: name, website, why_fit (1-2 lines)."
    )
    resp = agent.run(prompt)
    data = extract_json_or_raise(str(resp.content))
    companies = data.get("companies", [])
    return companies[: max(1, min(max_companies, 10))]


def run_contact_finder(agent: Agent, companies: List[Dict[str, str]], target_desc: str, offering_desc: str) -> List[Dict[str, Any]]:
    prompt = (
        "For each company below, find 2-3 relevant decision makers and emails (if available). Ensure at least 2 per company when possible, and cap at 3.\n"
        "If not available, infer likely email and mark inferred=true.\n"
        f"Targeting: {target_desc}\nOffering: {offering_desc}\n"
        f"Companies JSON: {json.dumps(companies, ensure_ascii=False)}\n"
        "Return JSON: {companies: [{name, contacts: [{full_name, title, email, inferred}]}]}"
    )
    resp = agent.run(prompt)
    data = extract_json_or_raise(str(resp.content))
    return data.get("companies", [])


def run_research(agent: Agent, companies: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    prompt = (
        "For each company, gather 2-4 interesting insights from their website and Reddit that would help personalize outreach.\n"
        f"Companies JSON: {json.dumps(companies, ensure_ascii=False)}\n"
        "Return JSON: {companies: [{name, insights: [string, ...]}]}"
    )
    resp = agent.run(prompt)
    data = extract_json_or_raise(str(resp.content))
    return data.get("companies", [])


def run_email_writer(agent: Agent, contacts_data: List[Dict[str, Any]], research_data: List[Dict[str, Any]], offering_desc: str, sender_name: str, sender_company: str, calendar_link: Optional[str]) -> List[Dict[str, str]]:
    prompt = (
        "Write personalized outreach emails for the following contacts.\n"
        f"Sender: {sender_name} at {sender_company}.\n"
        f"Offering: {offering_desc}.\n"
        f"Calendar link: {calendar_link or 'N/A'}.\n"
        f"Contacts JSON: {json.dumps(contacts_data, ensure_ascii=False)}\n"
        f"Research JSON: {json.dumps(research_data, ensure_ascii=False)}\n"
        "Return JSON with key 'emails' as a list of {company, contact, subject, body}."
    )
    resp = agent.run(prompt)
    data = extract_json_or_raise(str(resp.content))
    return data.get("emails", [])
