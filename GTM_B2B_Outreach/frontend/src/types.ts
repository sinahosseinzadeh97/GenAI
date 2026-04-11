export type RunPayload = {
  target_desc: string;
  offering_desc: string;
  sender_name: string;
  sender_company: string;
  calendar_link?: string;
  num_companies: number;
  email_style: 'Professional' | 'Casual' | 'Cold' | 'Consultative';
};

export type Company = { 
  name: string; 
  website: string; 
  why_fit: string 
};

export type Contact = { 
  full_name: string; 
  title: string; 
  email: string; 
  inferred?: boolean 
};

export type CompanyContacts = { 
  name: string; 
  contacts: Contact[] 
};

export type CompanyResearch = { 
  name: string; 
  insights: string[] 
};

export type PipelineResult = {
  companies: Company[];
  contacts: CompanyContacts[];
  research: CompanyResearch[];
  emails: { 
    company: string; 
    contact: {
      full_name: string;
      title: string;
      email: string;
    } | string; 
    subject: string; 
    body: string 
  }[];
};
