"""Trust checks for a brochure's contact and legal details.

The original-vs-generated evaluation called the generated brochure's
'@gmail.com' contact "unbelievable" for a luxury property, and credited
the original's RERA and registration details for building trust. These
checks surface those gaps to the person as warnings. They never block
generation and never invent a replacement — only the developer can
supply real contact and registration details.
"""
from __future__ import annotations

from typing import Optional

from src.data.schemas import PropertyInput, PropertyKnowledge

CONSUMER_EMAIL_DOMAINS = {
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.in", "hotmail.com", "outlook.com",
    "live.com", "rediffmail.com", "icloud.com", "aol.com", "protonmail.com", "proton.me",
}


def check(prop: PropertyInput, knowledge: Optional[PropertyKnowledge] = None) -> list[str]:
    warnings: list[str] = []
    email = (prop.contact.email if prop.contact else "") or ""
    domain = email.rsplit("@", 1)[-1].strip().lower() if "@" in email else ""
    if domain in CONSUMER_EMAIL_DOMAINS:
        warnings.append(f"Contact email uses a personal domain ({domain}). A company-domain address "
                        "reads as far more credible on a premium brochure.")
    rera = (prop.rera_number or "") or (knowledge.rera_number if knowledge else "")
    if not rera:
        warnings.append("No RERA registration number provided. Registered projects should show it; "
                        "add it in the property details or include it in an uploaded document.")
    if not prop.developer:
        warnings.append("No developer name provided; it is shown on the contact page when available.")
    contact = prop.contact
    if not contact or not (contact.phone or contact.email or contact.website):
        warnings.append("No contact details provided, so the brochure has no way for a buyer to respond.")
    return warnings
