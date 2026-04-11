import os
from typing import Any, Callable, Dict, List, Optional
from dotenv import load_dotenv
from models import RunPayload, PipelineResult

load_dotenv()

# Ensure keys exist
for var in ("OPENAI_API_KEY", "EXA_API_KEY"):
    if not os.getenv(var):
        print(f"[WARN] {var} not set in environment.")

# Import functions from agents.py
from agents import (
    create_company_finder_agent,
    create_contact_finder_agent,
    create_research_agent,
    create_email_writer_agent,
    run_company_finder,
    run_contact_finder,
    run_research,
    run_email_writer,
)


def orchestrate(
    p: RunPayload,
    on_progress: Optional[Callable[[int], None]] = None,
) -> PipelineResult:
    """Orchestrate the full pipeline with progress callbacks."""
    def _set(x: int):
        if on_progress:
            on_progress(x)

    company_agent = create_company_finder_agent()
    contact_agent = create_contact_finder_agent()
    research_agent = create_research_agent()
    email_agent = create_email_writer_agent(p.email_style)

    _set(5)
    companies = run_company_finder(
        company_agent,
        p.target_desc.strip(),
        p.offering_desc.strip(),
        max_companies=int(p.num_companies),
    )
    _set(30)

    contacts = run_contact_finder(
        contact_agent, companies, p.target_desc.strip(), p.offering_desc.strip()
    ) if companies else []
    _set(55)

    research = run_research(research_agent, companies) if companies else []
    _set(80)

    # Generate emails even if no specific contacts found - use general company contacts
    if not contacts and companies:
        # Create generic contacts for each company
        contacts = [
            {
                "name": company["name"],
                "contacts": [
                    {
                        "full_name": "Team",
                        "title": "Decision Maker",
                        "email": f"contact@{company['website'].replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]}",
                        "inferred": True
                    }
                ]
            }
            for company in companies
        ]
    
    emails = run_email_writer(
        email_agent,
        contacts,
        research,
        p.offering_desc.strip(),
        p.sender_name.strip() or "Sales Team",
        p.sender_company.strip() or "Our Company",
        (p.calendar_link or "").strip() or None,
    ) if contacts else []
    _set(100)

    return PipelineResult(
        companies=companies,
        contacts=contacts,
        research=research,
        emails=emails,
    )
