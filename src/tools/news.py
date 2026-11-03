"""Recent news, from two independent free sources.

What makes a property blog useful and dates it fastest is the same thing:
what is happening to the area right now. A new metro line, a road
widening, a policy change. A model trained months ago either does not
know or — worse — half-remembers something that has since changed.

Two providers, because they fail differently and cover differently:

    Google News RSS   broad, current, indexes local Indian outlets well
    GDELT             a research index; better for policy and
                      infrastructure announcements, worse for the last
                      48 hours

Neither needs an account. Both return headlines and links, and neither
returns the article body — the research pack carries headlines only, with
their source and date, and the writer is told these are signals to
mention rather than facts to assert. That is deliberate: a headline is
evidence that something was announced, not evidence that it happened.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from urllib.parse import quote_plus

from src.tools.base import NEWS, Tool, ToolResult
from src.tools.http import HttpError

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

_TAG = re.compile(r"<[^>]+>")
_ITEM = re.compile(r"<item>(.*?)</item>", re.S | re.I)


def _field(block: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", block, re.S | re.I)
    if not match:
        return ""
    text = match.group(1)
    text = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", text, flags=re.S)
    return _TAG.sub("", text).strip()


class GoogleNewsTool(Tool):
    """Headlines from Google News' public RSS search."""

    name = "google_news"
    capability = NEWS
    cache_ttl_s = 6 * 3600        # news goes stale fast

    def run(self, *, query: str, limit: int = 6, region: str = "IN",
            language: str = "en", **_) -> ToolResult:  # noqa: D102
        query = (query or "").strip()
        if not query:
            return ToolResult.failed(self.name, self.capability, "no query given")

        params = {"q": query, "hl": f"{language}-{region}", "gl": region,
                  "ceid": f"{region}:{language}"}
        cache_key = {**params, "limit": limit}
        cached = self._cache.get(self.name, cache_key, self.cache_ttl_s) if self._cache else None
        if cached is not None:
            return ToolResult(self.name, self.capability, data=cached, from_cache=True)

        try:
            xml = self._http.get_text(GOOGLE_NEWS_RSS, params=params)
        except HttpError as exc:
            return ToolResult.failed(self.name, self.capability, f"news search failed: {exc}")

        stories = []
        for block in _ITEM.findall(xml)[:limit]:
            title = _field(block, "title")
            if not title:
                continue
            stories.append({
                "title": title,
                "url": _field(block, "link"),
                "published": _field(block, "pubDate"),
                "source": _field(block, "source") or "Google News",
            })
        if self._cache:
            self._cache.put(self.name, cache_key, stories)
        return ToolResult(self.name, self.capability, data=stories,
                          note="" if stories else f"no recent news for {query!r}")


class GdeltNewsTool(Tool):
    """GDELT's document API — a second, independent news index."""

    name = "gdelt"
    capability = NEWS
    cache_ttl_s = 12 * 3600

    def run(self, *, query: str, limit: int = 6, months: int = 6, **_) -> ToolResult:  # noqa: D102
        query = (query or "").strip()
        if not query:
            return ToolResult.failed(self.name, self.capability, "no query given")

        params = {"query": query, "mode": "ArtList", "format": "json",
                  "maxrecords": max(1, min(limit, 20)),
                  "timespan": f"{max(1, months)}m", "sort": "DateDesc"}
        cached = self._cache.get(self.name, params, self.cache_ttl_s) if self._cache else None
        if cached is not None:
            return ToolResult(self.name, self.capability, data=cached, from_cache=True)

        try:
            payload = self._http.get_json(GDELT_DOC_API, params=params)
        except HttpError as exc:
            return ToolResult.failed(self.name, self.capability, f"GDELT search failed: {exc}")

        stories = []
        for article in (payload.get("articles") or [])[:limit]:
            title = (article.get("title") or "").strip()
            if not title:
                continue
            stories.append({
                "title": title,
                "url": article.get("url", ""),
                "published": _gdelt_date(article.get("seendate", "")),
                "source": article.get("domain", "") or "GDELT",
            })
        if self._cache:
            self._cache.put(self.name, params, stories)
        return ToolResult(self.name, self.capability, data=stories,
                          note="" if stories else f"GDELT has nothing recent for {query!r}")


def _gdelt_date(raw: str) -> str:
    """GDELT stamps are 20260930T104500Z. Render them as a plain date, and
    hand back whatever we were given if the format ever changes rather
    than losing the field."""
    try:
        return datetime.strptime(raw[:15], "%Y%m%dT%H%M%S").replace(
            tzinfo=timezone.utc).strftime("%d %b %Y")
    except (ValueError, TypeError):
        return raw or ""


def news_query(area: str, extra: str = "") -> str:
    """The search a property blog actually wants for an area."""
    terms = [t for t in (area, extra) if t and t.strip()]
    base = " ".join(terms).strip()
    return f"{base} real estate OR infrastructure OR metro OR property" if base else ""


def encode(value: str) -> str:
    return quote_plus(value)
