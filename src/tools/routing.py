"""Driving distance and time, via OSRM's public router. No account.

"18 km from the airport" is the single most common factual claim in an
Indian property blog, and the single easiest one to get wrong: a straight
line across a river is not a drive, and the gap is often 40%. A model
asked for the distance will give a plausible number with no idea which
one it is quoting.

This gives the real road distance, and — more useful to a buyer — the
real driving time. Both are labelled as free-flow estimates, because
that is what OSRM computes: it does not know about traffic, and an
Ahmedabad commute at 9am is not the number it returns. The research pack
carries that caveat through to the writer, so the post says "about 35
minutes in clear traffic" rather than stating a commute it cannot know.

OSRM's demo server is free, keyless and rate-limited, and it asks not to
be used for heavy traffic. The Research Agent calls it a handful of times
per post — for the airport, the station and the business district, not
for every school in a three-kilometre radius.
"""
from __future__ import annotations

from src.tools.base import ROUTE, Tool, ToolResult
from src.tools.http import HttpError

OSRM_URL = "https://router.project-osrm.org/route/v1/driving"


class RouteTool(Tool):
    """Road distance and free-flow driving time between two points."""

    name = "osrm"
    capability = ROUTE
    cache_ttl_s = 30 * 24 * 3600      # roads do not move often

    def run(self, *, from_lat: float, from_lon: float,
            to_lat: float, to_lon: float, label: str = "", **_) -> ToolResult:  # noqa: D102
        if not any((from_lat, from_lon)) or not any((to_lat, to_lon)):
            return ToolResult.failed(self.name, self.capability, "missing coordinates")

        cache_key = {"a": (round(from_lat, 4), round(from_lon, 4)),
                     "b": (round(to_lat, 4), round(to_lon, 4))}
        cached = self._cache.get(self.name, cache_key, self.cache_ttl_s) if self._cache else None
        if cached is not None:
            return ToolResult(self.name, self.capability,
                              data={**cached, "label": label}, from_cache=True)

        url = f"{OSRM_URL}/{from_lon},{from_lat};{to_lon},{to_lat}"
        try:
            payload = self._http.get_json(url, params={"overview": "false"})
        except HttpError as exc:
            return ToolResult.failed(self.name, self.capability,
                                     f"could not route to {label or 'destination'}: {exc}")

        routes = payload.get("routes") or []
        if not routes or payload.get("code") != "Ok":
            return ToolResult(self.name, self.capability, data={},
                              note=f"no road route found to {label or 'destination'}")
        route = routes[0]
        data = {
            "distance_km": round(float(route.get("distance", 0)) / 1000.0, 1),
            # OSRM's duration is free-flow: no traffic model at all. Naming
            # the field for what it is stops it being quoted as a commute.
            "free_flow_minutes": round(float(route.get("duration", 0)) / 60.0),
        }
        if self._cache:
            self._cache.put(self.name, cache_key, data)
        return ToolResult(self.name, self.capability, data={**data, "label": label})
