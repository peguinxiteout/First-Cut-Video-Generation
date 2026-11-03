"""Sections the person describes in their own words.

The catalogue in configs/blog/sections.yaml covers the shapes a property
post usually takes. It cannot cover "Why Ahmedabad is becoming a major
real estate investment hub", and asking someone to pick the nearest
preset instead is how a brief loses the thing the author actually cared
about.

So a custom section travels as the author's own sentence, all the way
through: into the brief the planner and writer see, and into the coverage
check that verifies it survived into the finished post. The last part is
what makes it a requirement rather than a suggestion.
"""
from __future__ import annotations

from src.agents.blog import brief
from src.data.schemas import BlogInput
from src.evaluation.blog_deterministic import uncovered_sections

CUSTOM = "Why Ahmedabad is becoming a major real estate investment hub"


def blog_input(**overrides):
    base = dict(blog_id="B", topic="Buying in Ahmedabad")
    base.update(overrides)
    return BlogInput(**base)


def test_a_custom_section_reaches_the_brief_marked_as_the_author_s_own():
    """Marked, because an unmarked one reads as another preset the model
    may merge away — which is how they were being dropped."""
    block = brief.requested_sections_block(blog_input(custom_sections=[CUSTOM]))
    assert CUSTOM in block
    assert "the author asked for this specifically" in block


def test_catalogue_and_custom_sections_coexist():
    block = brief.requested_sections_block(
        blog_input(sections=["strengths", "conclusion"], custom_sections=[CUSTOM]))
    assert "Strengths" in block and CUSTOM in block


def test_blank_and_duplicate_entries_are_cleaned_up():
    """A textarea produces blank lines and near-duplicates; neither should
    reach a prompt as a separate requirement."""
    result = brief.custom_sections(blog_input(custom_sections=[
        f"  {CUSTOM}  ", "", "   ", CUSTOM.upper(), "- Another one"]))
    assert result == [CUSTOM, "Another one"]


def test_a_post_that_covers_the_custom_section_passes_the_check():
    post = ("Ahmedabad is becoming a major investment hub: the new metro, the ring road "
            "and a widening industrial base are all pulling buyers in.")
    assert uncovered_sections(post, blog_input(custom_sections=[CUSTOM])) == []


def test_a_post_that_ignores_it_is_reported():
    """The whole point of carrying it into the evaluation: without this a
    dropped custom section is invisible until someone reads the post."""
    missing = uncovered_sections("A pleasant guide to buying a flat.",
                                 blog_input(custom_sections=[CUSTOM]))
    assert len(missing) == 1 and "your own section" in missing[0]


def test_the_check_tolerates_rewording():
    """The writer chooses its own headings, so requiring the exact
    sentence back would fail on every well-written post."""
    post = ("Ahmedabad has become a genuine investment hub for real estate over the past "
            "five years, and the reasons are structural.")
    assert uncovered_sections(post, blog_input(custom_sections=[CUSTOM])) == []


def test_the_check_is_not_satisfied_by_words_common_to_any_property_article():
    """'real', 'estate' and 'property' appear in every post on this site.
    A check they could satisfy would pass everything."""
    post = ("Real estate is a big decision. Property is expensive. This article is about "
            "real estate property for buyers.")
    assert uncovered_sections(post, blog_input(custom_sections=[CUSTOM]))


def test_a_section_with_nothing_specific_in_it_is_not_held_against_the_post():
    """'Please write about property' has no distinctive words to look for.
    Failing a post over an unverifiable requirement would be noise."""
    assert uncovered_sections("Anything at all.",
                              blog_input(custom_sections=["about the property"])) == []


def test_an_old_brief_without_the_field_still_loads():
    """blog_input.json files written before this existed must keep working."""
    parsed = BlogInput.model_validate({"blog_id": "B", "topic": "T", "sections": ["strengths"]})
    assert parsed.custom_sections == []
    assert brief.custom_sections(parsed) == []
