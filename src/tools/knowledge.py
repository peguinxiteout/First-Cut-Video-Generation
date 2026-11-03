"""Background on a place, from Wikipedia. No account.

Narrow on purpose. Wikipedia is very good at one thing this project
needs — a sourced, neutral paragraph on what an area IS, when it grew,
what it is known for — and poor at the things a property blog is tempted
to ask it: current prices, which builder is good, whether it is a smart
buy. So this returns a summary and nothing else, and the research pack
labels it as background rather than as fact to assert.

The value is smaller than the location tools but real: it is the
difference between "Shela is a developing area" (true of everywhere) and
a paragraph that knows Shela sits on the Sardar Patel Ring Road, which is
the kind of specific the reader came for.
"""
from __future__ import annotations

from urllib.parse import quote

from src.tools.base import ENCYCLOPEDIA, Tool, ToolResult
from src.tools.http import HttpError

WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary"
WIKI_SEARCH = "https://en.wikipedia.org/w/api.php"


class WikipediaTool(Tool):
    """A neutral summary of a place or subject."""

    name = "wikipedia"
    capability = ENCYCLOPEDIA
    cache_ttl_s = 30 * 24 * 3600

    def _best_title(self, query: str) -> str:
        """Wikipedia's own search, because a place name is rarely its
        article title: 'Shela, Ahmedabad' is an article, 'Shela' is a
        disambiguation page about somewhere else entirely."""
        payload = self._http.get_json(WIKI_SEARCH, params={
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": 1, "format": "json"})
        hits = ((payload.get("query") or {}).get("search") or [])
        return hits[0].get("title", "") if hits else ""

    def run(self, *, query: str, **_) -> ToolResult:  # noqa: D102
        query = (query or "").strip()
        if not query:
            return ToolResult.failed(self.name, self.capability, "no query given")

        cache_key = {"query": query}
        cached = self._cache.get(self.name, cache_key, self.cache_ttl_s) if self._cache else None
        if cached is not None:
            return ToolResult(self.name, self.capability, data=cached, from_cache=True)

        try:
            title = self._best_title(query)
            if not title:
                return ToolResult(self.name, self.capability, data={},
                                  note=f"Wikipedia has no article for {query!r}")
            payload = self._http.get_json(f"{WIKI_SUMMARY}/{quote(title.replace(' ', '_'))}")
        except HttpError as exc:
            return ToolResult.failed(self.name, self.capability, f"Wikipedia lookup failed: {exc}")

        extract = (payload.get("extract") or "").strip()
        if not extract:
            return ToolResult(self.name, self.capability, data={},
                              note=f"Wikipedia has no summary for {query!r}")
        data = {
            "title": payload.get("title", title),
            "summary": extract,
            "url": ((payload.get("content_urls") or {}).get("desktop") or {}).get("page", ""),
        }
        if self._cache:
            self._cache.put(self.name, cache_key, data)
        return ToolResult(self.name, self.capability, data=data)
