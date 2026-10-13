"""Strip personal data before any text is sent to a third-party LLM API.

Vehicle registrations, emails and phone numbers are personal data under UK GDPR,
and none of them help the model decide *why* someone is appealing.
"""
import re

# Current UK format, e.g. AB12 CDE / AB12CDE
UK_PLATE = re.compile(r"\b[A-Z]{2}[0-9]{2}\s?[A-Z]{3}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
UK_PHONE = re.compile(r"(?:\+44\s?7\d{3}|\b07\d{3})\s?\d{3}\s?\d{3}\b")


def redact(text: str) -> str:
    text = EMAIL.sub("[EMAIL]", text)
    text = UK_PHONE.sub("[PHONE]", text)
    text = UK_PLATE.sub("[REG]", text)
    return text
