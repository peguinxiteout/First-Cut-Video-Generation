"""The markdown normaliser: every formatting defect found in generated
blog PDFs, fixed mechanically — and the things it must NOT touch."""
from __future__ import annotations

import markdown
import pytest

from src.agents.blog.normaliser import (
    find_placeholders, markup_leaks, normalise_markdown, remove_sentences, split_sentences,
)


def norm(md: str, **kw) -> str:
    return normalise_markdown(md, **kw).text


def html(md: str) -> str:
    return markdown.markdown(md, extensions=["tables", "sane_lists"])


# --- collapsed bullets ---------------------------------------------------------

def test_dash_bullets_collapsed_onto_one_line_are_split():
    out = norm("- Metro station 1.2 km away - Airport 18 km - Schools within 3 km")
    assert out.splitlines() == ["- Metro station 1.2 km away", "- Airport 18 km", "- Schools within 3 km"]
    assert html(out).count("<li>") == 3


def test_intro_followed_by_inline_bullets_becomes_paragraph_plus_list():
    out = norm("Key benefits: - Metro access - Schools nearby - Large parks")
    assert out.startswith("Key benefits:\n\n- Metro access")
    assert html(out).count("<li>") == 3


def test_bullet_characters_in_a_line_are_split():
    out = norm("Highlights • Lake views • Wide roads • Quiet streets")
    assert "- Lake views\n- Wide roads\n- Quiet streets" in out


def test_inline_numbered_list_is_split():
    out = norm("Follow these steps: 1. Check RERA 2. Visit the site 3. Compare offers")
    assert "1. Check RERA\n2. Visit the site\n3. Compare offers" in out


def test_list_typed_straight_under_a_paragraph_is_not_merged_into_it():
    """The commonest cause of 'list items merged into paragraphs': no
    blank line before the list, so markdown reads it as the paragraph."""
    out = norm("Here is what matters:\n- Budget\n- Location\nThat is all.")
    rendered = html(out)
    assert rendered.count("<li>") == 2
    assert "<p>That is all.</p>" in rendered


def test_a_price_range_and_a_dash_in_prose_are_not_mistaken_for_bullets():
    assert norm("- Prices ₹55 - 75 lakh for 2BHK homes").count("\n- ") == 0
    assert norm("The metro - when it finally opens - will change commutes.") == \
        "The metro - when it finally opens - will change commutes."


def test_loose_lists_are_tightened():
    out = norm("<ul><li>One</li><li>Two</li></ul>")
    assert "- One\n- Two" in out


# --- headings and stray markup --------------------------------------------------

def test_heading_levels_inside_a_section_are_demoted_and_cleaned():
    out = norm("## **Big heading:**\nText.\n####Small\nMore.")
    assert "### Big heading" in out and "#### Small" in out
    assert "**" not in out and "##Small" not in out


def test_bold_only_line_becomes_a_real_subheading():
    assert norm("**Why it matters**\nBecause it does.").startswith("### Why it matters")


def test_headings_are_removed_where_none_belong():
    out = norm("## Overview\nIntro text.", allow_headings=False)
    assert "#" not in out and "Overview" in out


def test_unbalanced_bold_and_stray_marks_do_not_reach_the_page():
    out = norm("Prices are **steady in the area. Also ## this.")
    assert "**" not in out and "##" not in out
    assert not markup_leaks(html(out))


def test_excessive_bold_is_capped_but_list_lead_ins_survive():
    out = norm("**This** is **very** **important** and **so is this long bold phrase that keeps going on**.\n\n"
               "- **Commute:** short and **easy** drive")
    paragraph, items = out.split("\n\n")
    assert paragraph.count("**") == 2
    assert items.startswith("- **Commute:**") and "**easy**" not in items


# --- tables ------------------------------------------------------------------

def test_table_missing_its_separator_row_is_repaired():
    out = norm("| Area | Price |\n| Shela | ₹55 lakh |\n| Bopal | ₹60 lakh |")
    assert out.splitlines()[1] == "| --- | --- |"
    assert html(out).count("<tr>") == 3


def test_ragged_table_rows_are_evened_out():
    out = norm("| A | B | C |\n|---|---|\n| 1 | 2 |\n| x | y | z | extra |")
    rows = out.splitlines()
    assert all(r.count("|") == 4 for r in rows)
    assert "| x | y | z extra |" in out


def test_separator_row_without_a_table_is_removed():
    out = norm("Some text\n|---|---|\nMore text.")
    assert "|" not in out and "More text." in out


def test_one_column_table_becomes_a_list():
    out = norm("| Item |\n|---|\n| One |\n| Two |")
    assert out == "- Item\n- One\n- Two"


# --- placeholders, links, contacts --------------------------------------------

@pytest.mark.parametrize("text", [
    "Visit https://example.com for details.", "Call +91 XXXXX XXXXX now.", "Read more [here](#).",
    "See [link] for more.", "Email info@example.com today.", "Go to www.yourwebsite.com.",
    "Lorem ipsum dolor.", "Details at https://... soon.",
])
def test_sentences_built_around_placeholders_are_removed(text):
    out = norm(text + " The lake is lovely.")
    assert out == "The lake is lovely."
    assert not find_placeholders(out)


def test_links_to_unverified_sites_lose_the_link_but_verified_ones_stay():
    out = norm("Check [GujRERA](https://gujrera.gujarat.gov.in/projects) first. Read [this guide](https://blog.test/x).",
               allowed_urls=["https://gujrera.gujarat.gov.in"])
    assert "[GujRERA](https://gujrera.gujarat.gov.in/projects)" in out
    assert "Read this guide." in out and "blog.test" not in out


def test_a_bare_url_nobody_supplied_takes_its_sentence_with_it():
    assert norm("See https://randomblog.com/post for more. Prices are stable.") == "Prices are stable."


def test_contact_details_from_the_brief_are_kept_and_invented_ones_removed():
    text = "Call +91 98100 12345 to book. Or ring +91 99999 88888 any time."
    assert norm(text, allowed_contacts=["+91 98100 12345"]) == "Call +91 98100 12345 to book."


# --- readability ------------------------------------------------------------------

def test_walls_of_text_are_split_into_short_paragraphs():
    wall = " ".join(f"This is sentence number {i} and it has a few words in it." for i in range(8))
    paragraphs = norm(wall).split("\n\n")
    assert len(paragraphs) == 3
    assert all(len(split_sentences(p)) <= 3 for p in paragraphs)


def test_sentence_split_survives_abbreviations_and_decimals():
    assert split_sentences("Prices are approx. ₹55 lakh, e.g. in Shela. The metro is 1.2 km away.") == \
        ["Prices are approx. ₹55 lakh, e.g. in Shela.", "The metro is 1.2 km away."]


def test_stock_phrasing_is_rewritten_or_dropped():
    out = norm("Nestled in greenery, the project boasts a pool. Look no further for your home. In conclusion, it works.")
    assert out == "Set in greenery, the project has a pool. It works."


def test_units_are_written_one_way():
    out = norm("Prices are Rs. 5,500 per sqft and 2 lacs extra, about 12kms or 25 mins away, 12 % growth.")
    assert out == "Prices are ₹5,500 per sq ft and 2 lakh extra, about 12 km or 25 min away, 12% growth."


def test_raw_html_and_code_fences_are_cleaned():
    out = norm("<p>Para one</p>```\ncode\n```<br>Next")
    assert "<" not in out and "```" not in out


# --- removal helper and leak detection ----------------------------------------------

def test_remove_sentences_handles_prose_items_and_table_rows():
    md = ("Keep this. Drop 40% here.\n\n- keep item\n- drop 40% item\n\n"
          "| A | B |\n| --- | --- |\n| keep | 1 |\n| drop | 40% |")
    out, removed = remove_sentences(md, lambda s: "40%" in s)
    assert "Drop" not in out and "drop 40% item" not in out and "| drop |" not in out
    assert "Keep this." in out and "- keep item" in out and "| keep | 1 |" in out
    assert len(removed) == 3


def test_markup_leaks_are_detected_in_rendered_text():
    assert markup_leaks("This **leaked** and |---| here") == ["bold markers", "table rule"]
    assert markup_leaks("Clean text with a - dash.") == []
