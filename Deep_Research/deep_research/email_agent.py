from agents import Agent, ModelSettings

INSTRUCTIONS = (
    "You are an email composer. Given a report, compose a professional email that summarizes "
    "the key findings and includes the full report. The email should be well-formatted, "
    "professional, and include a clear subject line. Structure the email with a greeting, "
    "executive summary, key findings, and a professional closing."
)

email_agent = Agent(
    name="Email Agent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    model_settings=ModelSettings(temperature=0.5),
)