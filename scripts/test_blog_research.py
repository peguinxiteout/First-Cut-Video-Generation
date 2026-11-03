"""Check the blog research tools actually reach their services.

    python scripts/test_blog_research.py
    python scripts/test_blog_research.py --place "Whitefield, Bengaluru, India"

Every tool here talks to a free public API, and the only way to know they
work from YOUR machine and YOUR network is to call them. This does that
once each and prints what came back, so a thin blog can be traced to a
blocked endpoint rather than guessed at.

Nothing here writes to the project. It is safe to run any time, and it
costs nothing — none of these services charge, and eight of the nine need
no account at all.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import settings  # noqa: E402

OK, FAIL, SKIP = "  OK  ", " FAIL ", " SKIP "


def line(status: str, name: str, detail: str = "") -> None:
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--place", default="Shela, Ahmedabad, India",
                        help="a place to research (default: an Ahmedabad suburb)")
    parser.add_argument("--topic", default="buying property in Ahmedabad",
                        help="a topic for the search and keyword tools")
    args = parser.parse_args()

    if not settings.tools_enabled:
        line(FAIL, "Research tools", "turned off — set TOOLS_ENABLED=true in .env")
        return 1

    from src.tools import available_tools, toolbox_for

    print("Blog research — checking each tool against its live service\n")
    for row in available_tools():
        if not row["available"]:
            line(SKIP, f"{row['capability']}/{row['tool']}", row["reason"])
    print()

    research, seo = toolbox_for("research"), toolbox_for("seo")
    failures = 0
    started = time.time()

    found = research.geocode(args.place)
    if found.ok and found.data:
        line(OK, "geocode (OpenStreetMap)", found.data.get("display_name", "")[:70])
        lat, lon = found.data["lat"], found.data["lon"]

        nearby = research.nearby_places(lat=lat, lon=lon, radius_m=3000)
        if nearby.ok and nearby.data:
            line(OK, "nearby places (Overpass)", f"{len(nearby.data)} found")
            for place in nearby.data[:5]:
                print(f"         · {place['name']} — {place['distance_m']} m ({place['kind']})")
        else:
            failures += 1
            line(FAIL, "nearby places (Overpass)", nearby.note)

        city = args.place.split(",")[-1].strip()
        airport = research.geocode(f"international airport {city}")
        if airport.ok and airport.data:
            route = research.travel_distance(
                from_lat=lat, from_lon=lon,
                to_lat=airport.data["lat"], to_lon=airport.data["lon"], label="airport")
            if route.ok and route.data:
                line(OK, "driving route (OSRM)",
                     f"{route.data['distance_km']} km, about "
                     f"{route.data['free_flow_minutes']} min in clear traffic")
            else:
                failures += 1
                line(FAIL, "driving route (OSRM)", route.note)
    else:
        failures += 1
        line(FAIL, "geocode (OpenStreetMap)", found.note)

    background = research.encyclopedia(args.place)
    if background.ok and background.data:
        line(OK, "background (Wikipedia)", background.data["summary"][:70] + "...")
    else:
        failures += 1
        line(FAIL, "background (Wikipedia)", background.note)

    news = research.search_news(f"{args.place} real estate infrastructure", limit=3)
    if news.ok and news.data:
        line(OK, "news", f"{len(news.data)} stories")
        for story in news.data[:3]:
            print(f"         · {story['title'][:70]}")
    else:
        failures += 1
        line(FAIL, "news", news.note)

    web = research.search_web(args.topic, limit=3)
    if web.ok and web.data:
        line(OK, "web search", f"{len(web.data)} results")
    else:
        failures += 1
        line(FAIL, "web search", web.note)

    keywords = seo.keyword_suggestions(args.topic, limit=10)
    if keywords.ok and keywords.data:
        suggestions = keywords.data.get("suggestions", [])
        line(OK, "keyword demand (autocomplete)", f"{len(suggestions)} phrasings")
        for phrase in suggestions[:5]:
            print(f"         · {phrase}")
    else:
        failures += 1
        line(FAIL, "keyword demand (autocomplete)", keywords.note)

    print(f"\nFinished in {time.time() - started:.1f}s.")
    if failures:
        print(f"{failures} tool(s) could not be reached. Blogs will still generate — those "
              "sections will simply be thinner, and the reason is shown with the post.")
    else:
        print("Every tool answered. Blog research is working.")
    return 0        # never a non-zero exit: a blocked tool is not a broken install


if __name__ == "__main__":
    raise SystemExit(main())
