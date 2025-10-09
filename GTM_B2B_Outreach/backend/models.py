from typing import List, Optional, Any
from pydantic import BaseModel, HttpUrl

class RunPayload(BaseModel):
    target_desc: str
    offering_desc: str
    sender_name: str
    sender_company: str
    calendar_link: Optional[str] = None
    num_companies: int = 5
    email_style: str = "Professional"

class Company(BaseModel):
    name: str
    website: str
    why_fit: str

class Contact(BaseModel):
    full_name: str
    title: str
    email: str
    inferred: bool = False

class CompanyContacts(BaseModel):
    name: str
    contacts: List[Contact] = []

class CompanyResearch(BaseModel):
    name: str
    insights: List[str] = []

class PipelineResult(BaseModel):
    companies: List[Company] = []
    contacts: List[CompanyContacts] = []
    research: List[CompanyResearch] = []
    emails: List[Any] = []  # {company, contact, subject, body}
