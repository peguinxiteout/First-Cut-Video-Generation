"""What a tool is, and the two rules every one of them follows.

A tool is one external capability an agent can call: a web search, a
geocode, a list of schools near a point, a driving distance. The agents
never name a vendor. They ask a TOOLBOX for a capability, and the toolbox
picks whichever provider is configured and working — which is the whole
reason this layer exists, because swapping DuckDuckGo for Tavily should be
a line of configuration and not an edit to the Research Agent.

RULE 1: A TOOL NEVER RAISES INTO AN AGENT.

Every one of these talks to something outside the process: a public API
with no SLA, a rate limit, a laptop on hotel wifi. If a research tool
could break a generation, the blog would be less reliable WITH research
than without it, and nobody would leave it turned on. So a tool returns a
ToolResult either way — `ok=False` and a plain-language `note` when it
could not answer — and the caller carries on with whatever it did get.
The note is written for the person reading the finished post, not for a
log: "DuckDuckGo did not respond" tells them why a section is thinner
than usual.

RULE 2: A TOOL IS HONEST ABOUT WHAT IT DID NOT FIND.

An empty result is a real answer and must not be dressed up as one. This
matters more here than in most places: the thing downstream of these
tools is a language model writing marketing copy about property, and a
model given a blank where it expected facts will fill the blank in. So
"no results" travels as `ok=True, data=[]` with a note, distinct from
"the search failed", and the writer's prompt is told which of the two
happened.

CACHING

Results are cached on disk by tool and arguments. Three reasons, in order
of how much they matter: the free APIs here have rate limits and a
revision loop would re-ask the same questions; a cached run is
reproducible, so a blog can be regenerated and compared; and it makes
developing against these tools free rather than a slow drip of live
calls.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Capability names. An agent asks for one of these, never for a vendor.
WEB_SEARCH = "web_search"
NEWS = "news"
GEOCODE = "geocode"
NEARBY = "nearby"
ROUTE = "route"
ENCYCLOPEDIA = "encyclopedia"
KEYWORDS = "keywords"
READ_URL = "read_url"

CAPABILITIES = (WEB_SEARCH, NEWS, GEOCODE, NEARBY, ROUTE, ENCYCLOPEDIA, KEYWORDS, READ_URL)


@dataclass
class ToolResult:
    """What a tool hands back. Never an exception."""

    tool: str
    capability: str
    ok: bool = True
    data: Any = field(default_factory=list)
    note: str = ""
    from_cache: bool = False

    @property
    def empty(self) -> bool:
        return not self.data

    @classmethod
    def failed(cls, tool: str, capability: str, note: str) -> "ToolResult":
        return cls(tool=tool, capability=capability, ok=False, data=[], note=note)

    def to_dict(self) -> dict:
        return {"tool": self.tool, "capability": self.capability, "ok": self.ok,
                "data": self.data, "note": self.note}


class Tool(ABC):
    """One provider of one capability.

    `requires` names the settings fields that must be non-empty for this
    tool to work. Most of the tools here require nothing at all — that is
    deliberate, and it is why the research layer is worth switching on by
    default: OpenStreetMap, Wikipedia, GDELT and Google News all answer
    without an account."""

    name: str = "tool"
    capability: str = ""
    requires: tuple[str, ...] = ()
    # Roughly how long an answer stays useful. News goes stale in hours;
    # where a place is does not.
    cache_ttl_s: float = 24 * 3600

    def __init__(self, settings=None, http=None, cache=None):
        if settings is None:
            from src.core.config import settings as _settings
            settings = _settings
        self._settings = settings
        self._http = http
        self._cache = cache

    def is_available(self) -> bool:
        return all(str(getattr(self._settings, field, "") or "").strip()
                   for field in self.requires)

    def why_unavailable(self) -> str:
        missing = [f.upper() for f in self.requires
                   if not str(getattr(self._settings, f, "") or "").strip()]
        return f"{self.name} needs {', '.join(missing)}" if missing else ""

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """Answer the call. Must not raise — return ToolResult.failed()."""


class ToolCache:
    """A small on-disk cache, keyed by tool name and arguments."""

    def __init__(self, directory: Path):
        self.directory = directory

    def _path(self, tool: str, params: dict) -> Path:
        payload = json.dumps({"tool": tool, "params": params}, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        return self.directory / f"{tool}-{digest}.json"

    def get(self, tool: str, params: dict, ttl_s: float) -> Optional[Any]:
        path = self._path(tool, params)
        try:
            if not path.exists():
                return None
            if ttl_s > 0 and (time.time() - path.stat().st_mtime) > ttl_s:
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A corrupt or unreadable cache entry is a cache miss, never an
            # error: the tool simply asks again.
            return None

    def put(self, tool: str, params: dict, value: Any) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._path(tool, params).write_text(
                json.dumps(value, default=str), encoding="utf-8")
        except (OSError, TypeError) as exc:
            logger.debug("Could not cache %s: %s", tool, exc)
