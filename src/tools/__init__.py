"""The tool registry, and the toolboxes agents are handed.

    box = toolbox_for("research")
    where = box.geocode(place="Shela, Ahmedabad")
    near  = box.nearby_places(lat=..., lon=..., kinds=("school", "metro_station"))

TWO IDEAS, AND THE SECOND IS THE IMPORTANT ONE.

First: an agent asks for a CAPABILITY, never a vendor. `search_web` is
served by Tavily when a key is configured and DuckDuckGo when it is not,
and the Research Agent contains no line of code that knows this. Swapping
one for the other, or adding Brave tomorrow, is a registration here.

Second: NOT EVERY AGENT GETS EVERY TOOL. A toolbox is built for one
agent's job. The Writer has no web tools at all — deliberately, and not
as an oversight. A writer that can search mid-sentence wanders: it
produces a different post every run, browses instead of writing, and
makes the result impossible to evaluate, because you can no longer tell
whether a weak post came from weak research or weak writing. Research
happens in the research step, is written down, and the writer consumes
that. Separating the two is what makes the pipeline reproducible.

EVERYTHING DEGRADES QUIETLY.

Every tool here can be missing — no key, no package, no network, the
kill switch off — and every capability call still returns a ToolResult
with `ok=False` and a plain-language reason. A blog generated with no
research at all is a blog written exactly as it was before this layer
existed, which is the floor this whole feature is built above.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

from src.tools.base import (
    CAPABILITIES, ENCYCLOPEDIA, GEOCODE, KEYWORDS, NEARBY, NEWS, READ_URL, ROUTE, WEB_SEARCH,
    Tool, ToolCache, ToolResult,
)
from src.tools.http import HttpClient
from src.tools.knowledge import WikipediaTool
from src.tools.news import GdeltNewsTool, GoogleNewsTool
from src.tools.places import AMENITY_KINDS, DEFAULT_KINDS, GeocodeTool, NearbyPlacesTool
from src.tools.routing import RouteTool
from src.tools.seo import AutocompleteKeywordTool
from src.tools.websearch import DuckDuckGoSearchTool, TavilySearchTool

logger = logging.getLogger(__name__)

__all__ = [
    "AMENITY_KINDS", "CAPABILITIES", "DEFAULT_KINDS", "ToolBox", "ToolResult",
    "available_tools", "toolbox_for", "TOOLBOXES",
]

# Providers per capability, in preference order. The first AVAILABLE one
# answers; the rest are tried only if it fails outright.
#
# Tavily before DuckDuckGo because its results arrive already extracted —
# but it needs a key, so on a fresh checkout DuckDuckGo is what runs.
# Google News before GDELT because it indexes local Indian outlets far
# better; GDELT is the independent second opinion.
PROVIDERS: dict[str, tuple[type[Tool], ...]] = {
    WEB_SEARCH:    (TavilySearchTool, DuckDuckGoSearchTool),
    NEWS:          (GoogleNewsTool, GdeltNewsTool),
    GEOCODE:       (GeocodeTool,),
    NEARBY:        (NearbyPlacesTool,),
    ROUTE:         (RouteTool,),
    ENCYCLOPEDIA:  (WikipediaTool,),
    KEYWORDS:      (AutocompleteKeywordTool,),
}

# Which agent gets which capabilities. The Writer is absent on purpose —
# see the module docstring.
TOOLBOXES: dict[str, tuple[str, ...]] = {
    # Location and market research for a blog: where the place is, what is
    # around it, how far the journeys are, what the area is, what is in
    # the news, and what competitors are covering.
    "research": (GEOCODE, NEARBY, ROUTE, ENCYCLOPEDIA, NEWS, WEB_SEARCH),
    # Keyword research from real search behaviour, plus the SERP shape.
    "seo": (KEYWORDS, WEB_SEARCH),
    # Checking a claim against something outside the model.
    "fact_check": (WEB_SEARCH, ENCYCLOPEDIA, NEARBY, ROUTE),
}


def _default_cache(settings) -> Optional[ToolCache]:
    """The on-disk cache, or none at all.

    A settings object without a data directory is a perfectly valid thing
    to construct a toolbox with — a test, a script, an embedding of this
    package somewhere else — and it should mean "do not cache", not
    AttributeError halfway through building a research run."""
    directory = getattr(settings, "data_dir_resolved", None)
    return ToolCache(directory / "tool_cache") if directory is not None else None


class ToolBox:
    """The capabilities one agent is allowed, as ordinary method calls."""

    def __init__(self, capabilities: Sequence[str], settings=None, http=None, cache=None):
        if settings is None:
            from src.core.config import settings as _settings
            settings = _settings
        self._settings = settings
        self._http = http or HttpClient(settings=settings)
        self._cache = cache if cache is not None else _default_cache(settings)
        self._capabilities = tuple(c for c in capabilities if c in PROVIDERS)
        self._built: dict[str, list[Tool]] = {}
        # Two logs, for two audiences.
        #
        # `notes` is what the PERSON reading the finished post should know:
        # a capability that could not answer at all, or answered with
        # nothing. It goes into the generation record.
        #
        # `diagnostics` is what an operator needs: every individual
        # provider that failed, including ones whose fallback then
        # succeeded. Those do not belong in `notes` — "DuckDuckGo was down
        # so we used Tavily" is not news to a reader — but silently
        # dropping them would hide a provider that is failing every run.
        self.notes: list[str] = []
        self.diagnostics: list[str] = []

    # -- plumbing -----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._http.enabled

    def tools_for(self, capability: str) -> list[Tool]:
        if capability not in self._capabilities:
            return []
        if capability not in self._built:
            self._built[capability] = [
                cls(settings=self._settings, http=self._http, cache=self._cache)
                for cls in PROVIDERS.get(capability, ())
            ]
        return self._built[capability]

    def _call(self, capability: str, **kwargs) -> ToolResult:
        """The first available provider that answers. A provider that is
        not configured is skipped silently; one that fails is recorded and
        the next is tried."""
        if capability not in self._capabilities:
            return ToolResult.failed("toolbox", capability,
                                     f"this agent has no {capability} tool")
        if not self.enabled:
            return ToolResult.failed("toolbox", capability,
                                     "external tools are turned off (TOOLS_ENABLED=false)")

        tried: list[str] = []
        for tool in self.tools_for(capability):
            if not tool.is_available():
                continue
            result = tool.run(**kwargs)
            if result.ok:
                if result.note:
                    self.notes.append(result.note)
                return result
            message = result.note or f"{tool.name} failed"
            tried.append(message)
            self.diagnostics.append(message)
            logger.info("Tool %s failed (%s); trying the next provider", tool.name, result.note)

        note = "; ".join(tried) or f"no {capability} tool is configured"
        self.notes.append(note)
        return ToolResult.failed("toolbox", capability, note)

    # -- the standard interface --------------------------------------------
    #
    # These names are the contract the agents are written against. A new
    # vendor changes PROVIDERS above; nothing below this line moves.

    def search_web(self, query: str, *, limit: int = 6) -> ToolResult:
        return self._call(WEB_SEARCH, query=query, limit=limit)

    def search_news(self, query: str, *, limit: int = 6) -> ToolResult:
        return self._call(NEWS, query=query, limit=limit)

    def geocode(self, place: str, *, country_code: str = "") -> ToolResult:
        return self._call(GEOCODE, place=place, country_code=country_code)

    def nearby_places(self, *, lat: float, lon: float,
                      kinds: Optional[Sequence[str]] = None,
                      radius_m: int = 3000) -> ToolResult:
        return self._call(NEARBY, lat=lat, lon=lon,
                          kinds=tuple(kinds or DEFAULT_KINDS), radius_m=radius_m)

    def travel_distance(self, *, from_lat: float, from_lon: float,
                        to_lat: float, to_lon: float, label: str = "") -> ToolResult:
        return self._call(ROUTE, from_lat=from_lat, from_lon=from_lon,
                          to_lat=to_lat, to_lon=to_lon, label=label)

    def encyclopedia(self, query: str) -> ToolResult:
        return self._call(ENCYCLOPEDIA, query=query)

    def keyword_suggestions(self, seed: str, *, limit: int = 20,
                            expand: bool = True) -> ToolResult:
        return self._call(KEYWORDS, seed=seed, limit=limit, expand=expand)


def toolbox_for(agent: str, settings=None, **kwargs) -> ToolBox:
    """The toolbox for a named agent. An unknown name gets an empty box
    rather than every tool in the project — a typo should quietly disable
    research, not hand an agent the whole internet."""
    capabilities = TOOLBOXES.get(agent, ())
    if not capabilities:
        logger.warning("No toolbox defined for agent %r; it will run without tools", agent)
    return ToolBox(capabilities, settings=settings, **kwargs)


def available_tools(settings=None) -> list[dict]:
    """Every registered tool and whether it can run — what the UI shows and
    what `scripts/preflight.py` reports."""
    if settings is None:
        from src.core.config import settings as _settings
        settings = _settings
    rows = []
    for capability, classes in PROVIDERS.items():
        for cls in classes:
            tool = cls(settings=settings)
            rows.append({"capability": capability, "tool": tool.name,
                         "available": tool.is_available(),
                         "reason": "" if tool.is_available() else tool.why_unavailable()})
    return rows
