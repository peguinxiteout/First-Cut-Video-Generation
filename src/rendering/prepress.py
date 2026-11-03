"""What a printer needs that a browser does not emit.

The first print review ended on this:

    "The PDF is RGB with no output intent, so the press has to guess the
     separation. Convert to CMYK against the house profile."

Half of that is ours and half of it is the printer's, and the half that is
ours is the half that was missing.

WHAT A COLOUR CONVERSION ACTUALLY NEEDS

Separating RGB into CMYK is not arithmetic, it is a measurement of a
particular press on a particular paper: the same file separated for coated
sheet-fed and for uncoated web gives two different sets of plates, and
choosing the wrong one is worse than shipping RGB, because a printer who
receives RGB knows to ask and a printer who receives the wrong CMYK does
not. The measurement lives in an ICC profile, and the profile that matters
is the one the press is characterised to — FOGRA39 or FOGRA51 for most of
Europe and India, GRACoL for North America — which the print house
supplies and which is licensed, so it cannot ship in this repository.

WHAT WE CAN STATE, AND NOW DO

A PDF can declare the press condition it was PREPARED FOR, as an output
intent. That is the thing the review was really asking for: the RIP then
knows which separation to apply and, just as importantly, a preflight
check at the printer's end can say "this file is built for FOGRA39" rather
than "this file says nothing". Where the house profile IS available, it is
embedded and the file is a PDF/X candidate; where it is not, the intent
names a registered characterisation from the ICC registry, which is enough
for the RIP to pick the right separation.

So:

    no profile supplied    the print PDF declares its intended condition
                           by name, and the press separates to it
    profile supplied       the profile is embedded with the intent, and
                           the file carries its own definition of the
                           condition it was built for

Set BROCHURE_PRINT_PROFILE to an .icc path to get the second. The
condition's name is BROCHURE_PRINT_CONDITION, defaulting to the coated
condition most Indian offset houses run.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# The characterisation most coated-stock offset work in India and Europe
# is run to. It is a registered ICC condition, so naming it is meaningful
# even with no profile embedded.
DEFAULT_CONDITION = "FOGRA39L"
DEFAULT_INFO = "Coated FOGRA39 (ISO 12647-2:2004)"
REGISTRY = "http://www.color.org"


def _settings() -> tuple[str, str, Path | None]:
    condition = os.environ.get("BROCHURE_PRINT_CONDITION", DEFAULT_CONDITION).strip()
    info = os.environ.get("BROCHURE_PRINT_CONDITION_INFO", DEFAULT_INFO).strip()
    raw = os.environ.get("BROCHURE_PRINT_PROFILE", "").strip()
    profile = Path(raw) if raw else None
    if profile is not None and not profile.is_file():
        logger.warning("BROCHURE_PRINT_PROFILE is set to %s, which is not a file; "
                       "the print PDF will declare its condition by name instead", profile)
        profile = None
    return condition or DEFAULT_CONDITION, info or DEFAULT_INFO, profile


def apply_output_intent(pdf_path: Path) -> bool:
    """Declare on the print PDF which press condition it is built for.

    Returns True when the file was changed. Every failure is a warning and
    never an error: a missing output intent is a conversation with the
    printer, and a brochure that did not render is not."""
    try:
        import pikepdf
    except ImportError:  # pragma: no cover - the reading copy is the deliverable
        logger.info("pikepdf is not installed; the print PDF carries no output intent")
        return False

    condition, info, profile = _settings()
    try:
        with pikepdf.open(str(pdf_path), allow_overwriting_input=True) as pdf:
            intent = pikepdf.Dictionary(
                Type=pikepdf.Name("/OutputIntent"),
                S=pikepdf.Name("/GTS_PDFX"),
                OutputConditionIdentifier=pikepdf.String(condition),
                OutputCondition=pikepdf.String(info),
                Info=pikepdf.String(info),
                RegistryName=pikepdf.String(REGISTRY),
            )
            if profile is not None:
                stream = pikepdf.Stream(pdf, profile.read_bytes())
                # Four components: the intent describes a CMYK press.
                stream.N = 4
                intent.DestOutputProfile = pdf.make_indirect(stream)
            pdf.Root.OutputIntents = pdf.make_indirect(pikepdf.Array([intent]))
            pdf.save(str(pdf_path))
    except Exception as exc:  # noqa: BLE001 - never fail a build over prepress metadata
        logger.warning("could not write the output intent onto %s: %s", pdf_path.name, exc)
        return False

    logger.info("print PDF declares output intent %s%s", condition,
                " with the supplied profile embedded" if profile else " by name")
    return True
