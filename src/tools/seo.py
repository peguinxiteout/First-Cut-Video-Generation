"""What people actually type. Search autocomplete, no account.

The SEO Agent used to write a meta description and a keyword list from
the finished post — which means the keywords were whatever the model
thought the article was about, not what anyone searches for. Those are
different things, and the gap is the entire point of keyword research.

Autocomplete closes it cheaply. The suggestions behind a search box are
built from real query volume, so asking for completions of "property in
Ahmedabad" returns the phrasings people really use, including the long
question-shaped ones that make good headings ("is it good to invest in
ahmedabad real estate"). It needs no key and no quota.

It is not a volume tool. It cannot say a phrase gets 2,400 searches a
month, and nothing here pretends otherwise: suggestions come back ranked
by the order the endpoint returned them, which correlates with popularity
but is not a number. Anything that claimed to be a number here would be
invented, and invented numbers are exactly what this whole layer exists
to stop.
"""
from __future__ import annotations

import json
import string

from src.tools.base import KEYWORDS, Tool, ToolResult
from src.tools.http import HttpError

# Google's public suggestion endpoint, the one the search box itself uses.
SUGGEST_URL = "https://suggestqueries.google.com/complete/search"
# DuckDuckGo's equivalent, as a second opinion that fails independently.
DDG_SUGGEST_URL = "https://duckduckgo.com/ac/"

# Question words that turn a seed into the queries people actually ask.
QUESTION_PREFIXES = ("how", "why", "is", "what", "which", "when", "best", "cost of")


class AutocompleteKeywordTool(Tool):
    """Real query phrasings for a seed term."""

    name = "autocomplete"
    capability = KEYWORDS
    cache_ttl_s = 7 * 24 * 3600

    def _suggest(self, query: str, region: str) -> list[str]:
        try:
            raw = self._http.get_text(SUGGEST_URL, params={
                "client": "firefox", "q": query, "hl": "en", "gl": region})
            payload = json.loads(raw)
            return [str(s) for s in (payload[1] if len(payload) > 1 else [])]
        except (HttpError, ValueError, IndexError, TypeError):
            return []

    def run(self, *, seed: str, region: str = "IN", limit: int = 20,
            expand: bool = True, **_) -> ToolResult:  # noqa: D102
        seed = (seed or "").strip()
        if not seed:
            return ToolResult.failed(self.name, self.capability, "no seed term given")

        cache_key = {"seed": seed, "region": region, "limit": limit, "expand": expand}
        cached = self._cache.get(self.name, cache_key, self.cache_ttl_s) if self._cache else None
        if cached is not None:
            return ToolResult(self.name, self.capability, data=cached, from_cache=True)

        if self._http is not None and not self._http.enabled:
            return ToolResult.failed(self.name, self.capability,
                                     "external tools are turned off (TOOLS_ENABLED=false)")

        # The seed alone, then the seed behind question words and letters.
        # One call each, and the alphabet walk is capped — a full a-z sweep
        # is 26 requests for diminishing returns and looks like scraping.
        queries = [seed]
        if expand:
            queries += [f"{word} {seed}" for word in QUESTION_PREFIXES]
            queries += [f"{seed} {letter}" for letter in string.ascii_lowercase[:6]]

        found: list[str] = []
        seen: set[str] = set()
        for query in queries:
            for suggestion in self._suggest(query, region):
                key = suggestion.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    found.append(suggestion.strip())
            if len(found) >= limit:
                break

        questions = [s for s in found if _looks_like_a_question(s)]
        data = {
            "seed": seed,
            "suggestions": found[:limit],
            # Split out because these make good H2s, and a heading a reader
            # already typed into a search box is a heading that gets clicked.
            "questions": questions[:10],
        }
        if not found:
            return ToolResult(self.name, self.capability, data={},
                              note=f"no search suggestions came back for {seed!r}")
        if self._cache:
            self._cache.put(self.name, cache_key, data)
        return ToolResult(self.name, self.capability, data=data)


def _looks_like_a_question(phrase: str) -> bool:
    first = phrase.strip().lower().split(" ", 1)[0]
    return first in {"how", "why", "is", "are", "what", "which", "when",
                     "where", "can", "should", "does", "do"}
