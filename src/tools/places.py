"""Where a place is, and what is around it. OpenStreetMap, no account.

This is the most valuable thing in the toolbox for a property blog, and
the reason is specific. A language model asked to write about "Shela,
Ahmedabad" will produce confident sentences about schools and metro
stations, and it has no idea whether any of them are there. The same
model given an actual list — DPS East, 2.1 km; Shela Lake Garden, 900 m —
writes the same paragraph with real names in it.

So this does not make the prose better by making the model better. It
makes it better by removing the reason the model was inventing.

WHAT IT USES

    Nominatim   turns "Shela, Ahmedabad" into coordinates
    Overpass    lists what OpenStreetMap knows within a radius

Both are free, need no key, and are run on donated hardware — which is
why everything here goes through HttpClient with an identifying
User-Agent, asks for one place at a time, and caches hard. Their usage
policies are not decorative; a project that ignores them gets its IP
blocked, and it takes everyone using the same office network with it.

WHAT IT DELIBERATELY DOES NOT DO

It does not decide what is "nearby" in the sense the reader cares about.
It returns distances as measured, in a straight line, and says so. A
straight line is not a drive — see routing.py, which the Research Agent
uses for the handful of journeys worth quoting properly.
"""
from __future__ import annotations

import math
from typing import Optional

from src.tools.base import GEOCODE, NEARBY, Tool, ToolResult
from src.tools.http import HttpError

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# What a property buyer actually asks about, mapped to OpenStreetMap tags.
# Kept short on purpose: a list of every amenity within 3 km is not
# research, it is noise, and it costs a much slower Overpass query.
AMENITY_KINDS: dict[str, tuple[str, ...]] = {
    "school":        ('node["amenity"="school"]', 'way["amenity"="school"]'),
    "hospital":      ('node["amenity"="hospital"]', 'way["amenity"="hospital"]'),
    "clinic":        ('node["amenity"="clinic"]',),
    "university":    ('node["amenity"="university"]', 'way["amenity"="university"]'),
    "metro_station": ('node["railway"="station"]', 'node["station"="subway"]'),
    "bus_station":   ('node["amenity"="bus_station"]',),
    "railway_station": ('node["railway"="station"]',),
    "airport":       ('way["aeroway"="aerodrome"]', 'node["aeroway"="aerodrome"]'),
    "mall":          ('node["shop"="mall"]', 'way["shop"="mall"]'),
    "supermarket":   ('node["shop"="supermarket"]', 'way["shop"="supermarket"]'),
    "park":          ('way["leisure"="park"]', 'node["leisure"="park"]'),
    "restaurant":    ('node["amenity"="restaurant"]',),
    "bank":          ('node["amenity"="bank"]',),
    "pharmacy":      ('node["amenity"="pharmacy"]',),
}

# The set a property blog almost always wants.
DEFAULT_KINDS = ("school", "hospital", "metro_station", "mall", "park", "supermarket")

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance in metres. Used for ordering and for the
    'about N km away' figures, which are labelled as straight-line
    precisely because they are not driving distances."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (math.sin(d_phi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


class GeocodeTool(Tool):
    """Place name -> coordinates, via Nominatim."""

    name = "nominatim"
    capability = GEOCODE
    cache_ttl_s = 30 * 24 * 3600      # where a suburb is does not move

    def run(self, *, place: str, country_code: str = "", **_) -> ToolResult:  # noqa: D102
        place = (place or "").strip()
        if not place:
            return ToolResult.failed(self.name, self.capability, "no place given")
        params = {"q": place, "format": "jsonv2", "limit": 1, "addressdetails": 1}
        if country_code:
            params["countrycodes"] = country_code
        cache_key = dict(params)
        cached = self._cache.get(self.name, cache_key, self.cache_ttl_s) if self._cache else None
        if cached is not None:
            return ToolResult(self.name, self.capability, data=cached, from_cache=True)

        try:
            payload = self._http.get_json(NOMINATIM_URL, params=params)
        except HttpError as exc:
            return ToolResult.failed(self.name, self.capability, f"could not look up {place}: {exc}")

        if not payload:
            return ToolResult(self.name, self.capability, data={},
                              note=f"OpenStreetMap has no entry for {place!r}")
        entry = payload[0]
        address = entry.get("address") or {}
        data = {
            "query": place,
            "display_name": entry.get("display_name", ""),
            "lat": float(entry.get("lat", 0) or 0),
            "lon": float(entry.get("lon", 0) or 0),
            "type": entry.get("type", ""),
            "city": address.get("city") or address.get("town") or address.get("village") or "",
            "state": address.get("state", ""),
            "country": address.get("country", ""),
        }
        if self._cache:
            self._cache.put(self.name, cache_key, data)
        return ToolResult(self.name, self.capability, data=data)


class NearbyPlacesTool(Tool):
    """What OpenStreetMap knows within a radius, by kind."""

    name = "overpass"
    capability = NEARBY
    cache_ttl_s = 14 * 24 * 3600

    def _query(self, lat: float, lon: float, kinds: tuple[str, ...], radius_m: int) -> str:
        clauses = []
        for kind in kinds:
            for selector in AMENITY_KINDS.get(kind, ()):
                clauses.append(f'{selector}(around:{radius_m},{lat},{lon});')
        return f"[out:json][timeout:25];({''.join(clauses)});out center tags {self._limit};"

    _limit = 60

    def run(self, *, lat: float, lon: float, kinds: Optional[tuple] = None,
            radius_m: int = 3000, **_) -> ToolResult:  # noqa: D102
        kinds = tuple(kinds or DEFAULT_KINDS)
        unknown = [k for k in kinds if k not in AMENITY_KINDS]
        kinds = tuple(k for k in kinds if k in AMENITY_KINDS)
        if not kinds:
            return ToolResult.failed(self.name, self.capability,
                                     f"no known amenity kinds in {unknown}")
        if not (lat or lon):
            return ToolResult.failed(self.name, self.capability, "no coordinates given")

        cache_key = {"lat": round(lat, 4), "lon": round(lon, 4),
                     "kinds": sorted(kinds), "radius": radius_m}
        cached = self._cache.get(self.name, cache_key, self.cache_ttl_s) if self._cache else None
        if cached is not None:
            return ToolResult(self.name, self.capability, data=cached, from_cache=True)

        try:
            payload = self._http.get_json(
                OVERPASS_URL, params={"data": self._query(lat, lon, kinds, radius_m)},
                timeout_s=30.0)
        except HttpError as exc:
            return ToolResult.failed(self.name, self.capability,
                                     f"could not list nearby places: {exc}")

        found: list[dict] = []
        for element in payload.get("elements", []) or []:
            tags = element.get("tags") or {}
            name = tags.get("name")
            if not name:
                continue        # an unnamed node is no use in a sentence
            centre = element.get("center") or element
            try:
                place_lat, place_lon = float(centre["lat"]), float(centre["lon"])
            except (KeyError, TypeError, ValueError):
                continue
            found.append({
                "name": name,
                "kind": _kind_of(tags),
                "distance_m": round(haversine_m(lat, lon, place_lat, place_lon)),
                "lat": place_lat,
                "lon": place_lon,
            })

        # Nearest first, and only one entry per name — Overpass returns the
        # same school as a node and as a way, and a blog that lists it twice
        # reads as though nobody checked.
        seen: set[str] = set()
        unique: list[dict] = []
        for place in sorted(found, key=lambda p: p["distance_m"]):
            key = place["name"].strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(place)

        if self._cache:
            self._cache.put(self.name, cache_key, unique)
        note = "" if unique else "OpenStreetMap lists nothing of these kinds within the radius"
        return ToolResult(self.name, self.capability, data=unique, note=note)


def _kind_of(tags: dict) -> str:
    for key in ("amenity", "shop", "leisure", "railway", "aeroway", "station"):
        if tags.get(key):
            return str(tags[key])
    return "place"
