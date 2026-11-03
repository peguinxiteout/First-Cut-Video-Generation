#!/usr/bin/env python
"""Create the three reference properties used to develop and test the
brochure templates — SOBHA Neopolis (Luxury), Brigade Citrine (Mid-Tier)
and Mahindra Happinest Kalyan (Affordable) — from the project brief
(Property_details.docx) and the supplied photographs.

    python scripts/seed_reference_properties.py "../Property Photos"

Writes data/properties/<id>/property.json, images/ and images/roles.json.
Each photograph's subject is recorded ("photo:pool") because the file names
the developers supplied (Picture7.jpg, ckebpvp0ap196fanp2vw.webp) say
nothing about what they show; with a Gemini key the Vision Agent reads
this itself.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROPERTIES = {
    "sobha-neopolis": {
        "folder": "sobha neopolis bangalore",
        "data": {
            "name": "SOBHA Neopolis", "developer": "SOBHA Limited",
            "location": {"city": "Bengaluru", "area": "Panathur, near Outer Ring Road (ORR)",
                         "address": "Panathur, near Outer Ring Road (ORR), Bengaluru"},
            "property_type": "Apartments", "category": "luxury", "tagline": "The Greek Paradise",
            "unit_types": [
                {"name": "1 BHK", "carpet_area": "660 sq.ft"},
                {"name": "3 BHK", "carpet_area": "1611 – 1915 sq.ft"},
                {"name": "3 BHK + Study", "carpet_area": "2150 – 2178 sq.ft"},
                {"name": "4 BHK", "carpet_area": "2333 – 2481 sq.ft"},
            ],
            "usps": [
                "Mainland Greek and Santorini-inspired classical architecture featuring arches, majestic columns, and bevelled pediments.",
                "Sprawling 25+ acre layout featuring 78% open spaces and vehicle-free peripheral roads.",
                "Three massive themed clubhouses spanning over 77,850 sq.ft.",
                "Proximity to major tech hubs along the Outer Ring Road and upcoming Metro Phase 2A lines.",
                "Built with SOBHA's backward-integrated signature construction quality, ensuring 1,456 quality checks.",
            ],
            "amenities": [
                "Recreational: 3 lavish clubhouses (Club Santorini, Club Syros, Club Athinios), Meandering Pool, and Floating Deck.",
                "Active Zone: Tennis Court, Basketball Court, Activity Field, Jogging Park, and Reflexology Walk.",
                "Park & Plaza: Entrance Plaza with Clock Tower & telescope, Olympus Plaza, and BBQ & Picnic Park.",
                "Kids' Zone: Aqua Park with water slides, Toddlers' Pad, and Skating Lane.",
                "Indoor: Mini Theatre, Co-working Space, Salon & Spa, Gym, and Guest Rooms.",
            ],
            "target_audience": "High-net-worth individuals, tech leaders working in the ORR/Whitefield IT corridor, and affluent families looking for a heavily-themed luxury Mediterranean-style community.",
            "contact": {"phone": "080 4646 4500", "website": "www.sobha.com"},
            "cta": "Visit the SOBHA Experience Center to explore the show residence and museum",
            "possession": "December 2028", "project_size": "25 Acres, 19 Towers, 1,875 Units",
            "landmarks": [
                {"name": "ORCHIDS The International School", "distance": "2.5 km", "category": "school"},
                {"name": "Salarpuria Sattva Eminence", "distance": "3.0 km", "category": "work"},
                {"name": "Prestige Tech Park / Cessna Business Park", "distance": "3.5 km", "category": "work"},
                {"name": "Sakra World Hospital", "distance": "4.5 km", "category": "hospital"},
                {"name": "RMZ Ecoworld / Ecospace", "distance": "5.0 km", "category": "work"},
            ],
        },
        "roles": {
            "center-view.jpg": "photo:exterior", "gallery-4.jpg": "photo:pool",
            "highlight.jpg": "photo:landscape", "location.jpg": "map", "logo.png": "logo",
            "Picture1.jpg": "photo:exterior", "Picture2.jpg": "photo:clubhouse",
            "Picture3.jpg": "masterplan", "Picture4.jpg": "photo:kids", "Picture5.jpg": "photo:clubhouse",
            "Picture6.jpg": "photo:clubhouse", "Picture7.jpg": "photo:landscape",
            "Picture8.jpg": "photo:arrival", "Picture9.jpg": "photo:landscape",
            "Picture10.jpg": "photo:sports", "Picture11.jpg": "photo:aerial",
            "Picture12.jpg": "photo:night", "Picture13.jpg": "photo:bedroom",
            "Picture14.jpg": "photo:living", "Picture15.jpg": "photo:balcony",
            "Picture16.jpg": "photo:bedroom", "Picture17.jpg": "photo:kitchen",
            "Picture18.jpg": "photo:bathroom", "Picture19.jpg": "photo:landscape",
        },
    },
    "mahindra-happinest-kalyan": {
        "folder": "mahindra happinest kalyan",
        "data": {
            "name": "Mahindra Happinest Kalyan", "developer": "Mahindra Lifespace Developers Ltd",
            "location": {"city": "Mumbai Metropolitan Region", "area": "Kalyan",
                         "address": "Kalyan, Mumbai Metropolitan Region"},
            "property_type": "Apartments", "category": "affordable", "price": "₹29.95 Lakh",
            "tagline": "#LiveSmileProsper", "brand_color": "#E31837",
            "unit_types": [
                {"name": "1 BHK-A", "carpet_area": "334.76 sq.ft", "price": "₹29.95 Lakh – ₹31.15 Lakh"},
                {"name": "1 BHK-B", "carpet_area": "378.03 sq.ft", "price": "₹34.50 Lakh – ₹35.90 Lakh"},
                {"name": "2 BHK-A", "carpet_area": "495.04 sq.ft", "price": "₹46.45 Lakh – ₹49.85 Lakh"},
                {"name": "2 BHK-B", "carpet_area": "547.46 sq.ft", "price": "₹51.90 Lakh – ₹55.70 Lakh"},
            ],
            "usps": [
                "Located on the Bhiwandi-Kalyan Corridor near the proposed Rajnouli Metro Station (Line 5).",
                "Pre-certified IGBC Gold project ensuring 45% water savings and 70% common area solar power.",
                "Zero-wastage layout with piped gas provision saving 7 sq.ft of carpet area.",
                "Multiplier Rebate Plan (MRP) offering assured early registration savings of ₹50,000+.",
                "PMAY subsidy benefits up to ₹2.67 Lakh for eligible buyers.",
            ],
            "amenities": [
                "Landscaped gardens featuring 1,200+ trees, fragrance garden, and picnic lawn.",
                "Jogging track, cycling track, outdoor gym, and reflexology path.",
                "Two cricket nets, multipurpose courts, badminton, and kabaddi courts.",
                "Clubhouse with multi-purpose hall, indoor games, gym, and terrace lounge.",
                "Primary health center by Apollo Clinic and daycare by Eager Beavers.",
                "24-hour security, CCTV monitoring, and automated boom barriers.",
            ],
            "target_audience": "Budget-conscious nuclear families, first-time homebuyers, and working professionals in the Kalyan-Bhiwandi logistics and business hubs seeking an affordable, community-driven lifestyle.",
            "contact": {"phone": "+91 91376 11940", "email": "info@propjunction.org"},
            "cta": "Register online and get assured savings of ₹50,000 and more",
            "rera_number": "P51700022981", "project_size": "9 Acres",
            "landmarks": [
                {"name": "Rajnouli Metro Station", "distance": "~500 m (2 min)", "category": "transit"},
                {"name": "VITS Upper Thane", "distance": "~1 km", "category": "hotel"},
                {"name": "Ved Hospital", "distance": "~3.5 km", "category": "hospital"},
                {"name": "St. Lawrence International School", "distance": "~3-4 km", "category": "school"},
                {"name": "DMart", "distance": "~4 km", "category": "retail"},
            ],
        },
        "roles": {
            "1.5 BHK.webp": "floorplan", "HK-2BHK-Regular- (1).webp": "floorplan",
            "HK-Floor Plan-Typical Tower 1-Template (1).webp": "floorplan",
            "HK-Floor Plan-Typical Tower 2 3 4-Template (1).webp": "floorplan",
            "HK-Floor Plan-Typical Tower 5-Template (1).webp": "floorplan",
            "HK-Floor Plan-Typical Tower 6 7-Template (1)_0.webp": "floorplan",
            "Master Layout Mahindra Happinest Kalyan - 1 (1) (1).webp": "masterplan",
            "logo.png": "logo", "ElevationPicture.webp": "photo:exterior",
            "ckebpvp0ap196fanp2vw.webp": "photo:landscape", "wwspatp3oroxl5xzgvlc.webp": "photo:convenience",
            "CampingDeck.webp": "photo:landscape", "ForestWalk.webp": "photo:landscape",
            "JoggingandCyclingTrack.webp": "photo:sports", "Multi-purposecourt.webp": "photo:sports",
        },
    },
    "brigade-citrine": {
        "folder": "brigade citrine bangalore",
        "data": {
            "name": "Brigade Citrine", "developer": "Brigade Enterprises Ltd",
            "location": {"city": "Bengaluru", "area": "Old Madras Road",
                         "address": "Old Madras Road, Bengaluru"},
            "property_type": "Apartments", "category": "mid", "price": "₹80 Lakh onwards",
            "tagline": "Building Positive Experiences — For Earth. For Life!",
            "unit_types": [
                {"name": "1 Bed + 1 Toilet", "carpet_area": "434.86 sq.ft"},
                {"name": "2 Bed + 2 Toilet", "carpet_area": "811.49 sq.ft"},
                {"name": "3 Bed + 2 Toilet", "carpet_area": "1017.62 sq.ft"},
                {"name": "3 Bed + 3 Toilet", "carpet_area": "1117.62 sq.ft"},
                {"name": "3 Bed + 3 Toilet + Study", "carpet_area": "1270.04 sq.ft"},
                {"name": "4 Bed + 4 Toilet", "carpet_area": "1540.22 sq.ft"},
            ],
            "usps": [
                "India's first completely Net Zero Residential Community (Net Zero Carbon, Energy, Water, and Waste).",
                "Strict privacy design: Across all 420 units, zero homes face each other.",
                "100% of common area lighting is powered directly by renewable solar energy.",
                "Eco-conscious architecture diverts up to 300 tonnes of waste from landfills annually (86% diversion rate).",
                "Strategic Old Madras Road location offering equidistant access to Whitefield, CBD, and Kempegowda Airport.",
            ],
            "amenities": [
                "Swimming Pool and dedicated Kids' Pool.",
                "Multi-play Court and Outdoor Fitness Area.",
                "Wellness Garden and Pet Park.",
                "Lavish Clubhouse with indoor spaces for meals and connections.",
                "100% DG backup for both common areas and residential units.",
                "Smart water management yielding up to a 46.5% reduction in external fresh water demand.",
            ],
            "target_audience": "Environmentally conscious professionals, tech workers traversing the Whitefield-Airport corridors, and families seeking a carbon-neutral, ultra-private luxury lifestyle.",
            "contact": {"phone": "+91 91376 11941", "website": "www.brigadegroup.com"},
            "cta": "Book a site visit",
            "possession": "June 2029", "project_size": "4.3 Acres, 420 Units",
            "landmarks": [
                {"name": "Orion Uptown Mall", "distance": "Proximity", "category": "retail"},
                {"name": "Holiday Inn Express Hotel", "distance": "Proximity", "category": "hotel"},
                {"name": "Decathlon", "distance": "Proximity", "category": "retail"},
                {"name": "VIBGYOR High School", "distance": "Proximity", "category": "school"},
                {"name": "East Point Hospital", "distance": "Proximity", "category": "hospital"},
                {"name": "ITPL / EPIP Zone Whitefield", "distance": "Proximity", "category": "work"},
            ],
        },
        "roles": {
            "aerial_view_revise.webp": "photo:aerial", "backside_elevation_view_01.webp": "photo:exterior",
            "bedroom.webp": "photo:bedroom", "brigade-logo_horizontal.png": "logo",
            "club_view_01.webp": "photo:clubhouse", "dining.webp": "photo:dining",
            "gym__kids_view_01.webp": "photo:kids", "kitchen.webp": "photo:kitchen",
            "living.webp": "photo:living", "location.webp": "map", "master_plan_2025x1432.webp": "masterplan",
            "multiple_court_view.webp": "photo:sports", "pool_to_elevation_view.webp": "photo:night",
            "tower_a_balcony_view.webp": "photo:balcony", "tower_b_dropoff_view_02.webp": "photo:arrival",
            "tower_elevation_view_02.webp": "photo:exterior",
        },
    },
}


def safe(name: str) -> str:
    stem, _, ext = name.rpartition(".")
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:60] + "." + ext.lower()


def main() -> int:
    source_root = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT.parent / "Property Photos")
    for pid, spec in PROPERTIES.items():
        src = source_root / spec["folder"]
        if not src.exists():
            print(f"skip {pid}: {src} not found")
            continue
        folder = ROOT / "data" / "properties" / pid
        images = folder / "images"
        if images.exists():
            shutil.rmtree(images)
        images.mkdir(parents=True)
        roles = {}
        for original, role in spec["roles"].items():
            path = src / original
            if not path.exists():
                print(f"  missing {original}")
                continue
            name = safe(original)
            shutil.copy2(path, images / name)
            roles[name] = role
        (images / "roles.json").write_text(json.dumps(roles, indent=1), encoding="utf-8")
        data = {"property_id": pid, **spec["data"]}
        (folder / "property.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{pid}: {len(roles)} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
