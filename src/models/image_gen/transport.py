"""One place that knows what a failed HTTP call looks like.

A generation died on this, and the message is worth keeping:

    HTTPSConnectionPool(host='api.cloudflare.com', port=443):
    Read timed out. (read timeout=120.0)
    1 step(s) failed: visual_enrichment

Nothing about that is unusual — an image model taking longer than two
minutes is a slow afternoon, not an outage. What made it fatal is that the
exception came from `requests`, so it was not an `ImageGenerationError`,
and every layer built to absorb a failed image let it through untouched:
the provider's own retry loop, the next model, the fallback to the second
vendor, and the agent's soft failure that exists precisely so a brochure
finishes without its generated images.

So transport failures get a type of their own (ImageGenerationTransportError,
a subclass, so existing handlers keep working) and `call()` is the only
place either provider talks to the network.

TIMEOUTS ARE TWO NUMBERS

`timeout=120` sets the connect timeout AND the read timeout to 120s, which
is the wrong shape for this: a vendor that is down should be given up on in
seconds, while a vendor that is generating an image deserves minutes. The
split is (CONNECT_TIMEOUT, read), and the read timeout stays the caller's.

AND THERE IS A BUDGET

Retries used to be counted, not timed. Four attempts across two models at
a 120s read timeout is sixteen minutes of a progress bar going nowhere,
and the person watching it has no idea whether to wait. `deadline()` caps
the whole provider call instead, so a slow vendor costs a bounded wait and
then the chain moves on to the next one.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from src.models.image_gen.base import ImageGenerationError, ImageGenerationTransportError

logger = logging.getLogger(__name__)

# Long enough for a TLS handshake on a bad line, short enough that a vendor
# that is simply down is noticed in seconds rather than minutes.
CONNECT_TIMEOUT = 10.0

# The whole of one provider's attempt — every model, every retry. Past this
# the chain is better off asking the other vendor.
DEFAULT_BUDGET_S = 210.0


def deadline(budget_s: float = DEFAULT_BUDGET_S) -> float:
    """A wall-clock instant after which a provider should stop trying."""
    return time.monotonic() + max(1.0, budget_s)


def expired(at: Optional[float]) -> bool:
    return at is not None and time.monotonic() >= at


def remaining(at: Optional[float], read_timeout: float) -> float:
    """The read timeout for the next attempt, capped by what is left.

    Without this the last attempt inside a budget can still block for the
    full read timeout, so a 210s budget turns into 330s."""
    if at is None:
        return read_timeout
    left = at - time.monotonic()
    return max(5.0, min(read_timeout, left))


def call(post: Callable[..., object], url: str, *, headers: dict, json: dict,
         read_timeout: float, at: Optional[float] = None):
    """POST, turning any transport failure into an ImageGenerationError.

    The exception classes are looked up through the installed `requests`
    rather than imported at module scope, because `post` is injectable for
    tests and a test double raises whatever it likes. Anything that is not
    already one of ours becomes a transport error, which is the safe
    reading: an unrecognised exception from an HTTP call is a failed HTTP
    call, and the cost of being wrong is one retry."""
    try:
        return post(url, headers=headers, json=json,
                    timeout=(CONNECT_TIMEOUT, remaining(at, read_timeout)))
    except ImageGenerationError:
        raise
    except Exception as exc:  # noqa: BLE001 - deliberately wide; see the docstring
        raise ImageGenerationTransportError(_describe(exc)) from exc


def _describe(exc: Exception) -> str:
    """A sentence an operator can act on, not a connection-pool dump."""
    name = type(exc).__name__
    if "Timeout" in name:
        # Stated as a fact, not a promise. The same string ends up in the
        # retry log (where retries are still to come) AND in the note the
        # person reads after everything has been tried, and "will be
        # retried" is wrong by the time they see it.
        return (f"the image service did not answer in time ({name}). Raise "
                "IMAGE_GEN_TIMEOUT_S or IMAGE_GEN_BUDGET_S if this is routine "
                "on your connection.")
    if "SSL" in name or "Certificate" in name:
        return f"the connection to the image service could not be secured ({name}: {exc})."
    if "Connection" in name or "DNS" in name or "Resolution" in name:
        return (f"the image service could not be reached ({name}). Check network access "
                "to the provider, or an outbound proxy.")
    return f"the image request failed in transport ({name}: {exc})."
