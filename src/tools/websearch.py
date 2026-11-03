"""General web search: DuckDuckGo by default, Tavily when a key is set.

Two providers behind one capability, and the order matters. DuckDuckGo
needs no account and is the reason research works out of the box; Tavily
returns cleaner, already-extracted results but needs a key, so it is
preferred when one exists and simply absent when it does not.

A NOTE ON WHAT SEARCH IS FOR HERE

It is not for facts. Search results are titles and snippets from pages
nobody has vetted, and a property blog that repeats what a competitor's
landing page claims is worse than one that says less. What search is
genuinely good for is what the SERP SHAPE tells you: which questions
people are being served answers to, which subtopics every ranking page
covers, which angle nobody has taken. That is what the Research Agent
takes from it, and why the results travel to the writer as "topics
competitors cover" rather than as source material.
"""
from __future__ import annotations

import logging

from src.tools.base import WEB_SEARCH, Tool, ToolResult
from src.tools.http import HttpError

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
DDG_HTML = "https://html.duckduckgo.com/html/"


class DuckDuckGoSearchTool(Tool):
    """Keyless web search through the `ddgs` package when it is installed.

    The package is an optional dependency: without it this tool reports
    itself unavailable and the toolbox moves on, rather than the import
    failing at startup over a research feature."""

    name = "duckduckgo"
    capability = WEB_SEARCH
    cache_ttl_s = 3 * 24 * 3600

    def is_available(self) -> bool:
        if not super().is_available():
            return False
        return _ddgs_client() is not None

    def why_unavailable(self) -> str:
        if _ddgs_client() is None:
            return "duckduckgo needs the `ddgs` package (pip install ddgs)"
        return super().why_unavailable()

    def run(self, *, query: str, limit: int = 6, region: str = "in-en", **_) -> ToolResult:  # noqa: D102
        query = (query or "").strip()
        if not query:
            return ToolResult.failed(self.name, self.capability, "no query given")

        cache_key = {"query": query, "limit": limit, "region": region}
        cached = self._cache.get(self.name, cache_key, self.cache_ttl_s) if self._cache else None
        if cached is not None:
            return ToolResult(self.name, self.capability, data=cached, from_cache=True)

        if self._http is not None and not self._http.enabled:
            return ToolResult.failed(self.name, self.capability,
                                     "external tools are turned off (TOOLS_ENABLED=false)")

        client = _ddgs_client()
        if client is None:
            return ToolResult.failed(self.name, self.capability, self.why_unavailable())
        try:
            with client() as search:
                raw = list(search.text(query, region=region, max_results=limit))
        except Exception as exc:  # noqa: BLE001 - the package raises its own types
            return ToolResult.failed(self.name, self.capability, f"web search failed: {exc}")

        results = [{"title": r.get("title", ""), "url": r.get("href") or r.get("url", ""),
                    "snippet": r.get("body") or r.get("snippet", "")}
                   for r in raw if r.get("title")]
        if self._cache:
            self._cache.put(self.name, cache_key, results)
        return ToolResult(self.name, self.capability, data=results,
                          note="" if results else f"no web results for {query!r}")


def _ddgs_client():
    """The `ddgs` package, or its older name, or None."""
    try:
        from ddgs import DDGS
        return DDGS
    except ImportError:
        pass
    try:
        from duckduckgo_search import DDGS
        return DDGS
    except ImportError:
        return None


class TavilySearchTool(Tool):
    """Search built for agents: results arrive already extracted."""

    name = "tavily"
    capability = WEB_SEARCH
    requires = ("tavily_api_key",)
    cache_ttl_s = 3 * 24 * 3600

    def run(self, *, query: str, limit: int = 6, **_) -> ToolResult:  # noqa: D102
        query = (query or "").strip()
        if not query:
            return ToolResult.failed(self.name, self.capability, "no query given")
        if not self.is_available():
            return ToolResult.failed(self.name, self.capability, self.why_unavailable())

        cache_key = {"query": query, "limit": limit}
        cached = self._cache.get(self.name, cache_key, self.cache_ttl_s) if self._cache else None
        if cached is not None:
            return ToolResult(self.name, self.capability, data=cached, from_cache=True)

        try:
            payload = self._http.post_json(TAVILY_URL, json={
                "api_key": self._settings.tavily_api_key,
                "query": query, "max_results": limit,
                "search_depth": "basic", "include_answer": False})
        except HttpError as exc:
            return ToolResult.failed(self.name, self.capability, f"Tavily search failed: {exc}")

        results = [{"title": r.get("title", ""), "url": r.get("url", ""),
                    "snippet": r.get("content", "")}
                   for r in (payload.get("results") or []) if r.get("title")]
        if self._cache:
            self._cache.put(self.name, cache_key, results)
        return ToolResult(self.name, self.capability, data=results,
                          note="" if results else f"no web results for {query!r}")
