"""Slugs that a filesystem will actually accept.

A property name and a blog topic become directory names, so the reported
bug was not a naming problem, it was a crash: a 197-character blog topic
under a 75-character project folder is past the 260-character limit
Windows resolves a path against, and mkdir() fails before anything is
written. The error names a path that looks valid, which is why it reads
as a missing directory rather than as a name too long to create.
"""
from __future__ import annotations

from src.core.ids import SLUG_MAX_CHARS, slugify

# The topic from the report, verbatim.
REPORTED = ("Whitefield vs Sarjapur Road vs Electronic City Bengaluru: a practical "
            "comparison for homebuyers and property investors covering connectivity, "
            "pricing, infrastructure, lifestyle, rental potential and future development")
# The project folder it was running under, and the deepest path built inside
# a blog's directory.
WINDOWS_PREFIX = "D:\\Projects\\Xanadu Marketing Agent POC\\agentic-content-platform\\data\\blogs\\"
WINDOWS_LIMIT = 260


def test_the_topic_that_crashed_now_produces_a_path_windows_can_create():
    slug = slugify(REPORTED, "BLOG")
    full = len(WINDOWS_PREFIX) + len(slug) + len("\\blog_input.json")
    assert full < WINDOWS_LIMIT, f"{full} characters is still past the limit"


def test_no_slug_is_longer_than_the_limit_however_long_the_topic():
    for text in (REPORTED, "A" * 5000, " ".join(["locality"] * 400), "-" * 300 + "end"):
        assert len(slugify(text, "BLOG")) <= SLUG_MAX_CHARS + 9   # + "-" + digest


def test_two_long_topics_that_open_the_same_way_get_different_ids():
    """The reason a truncated slug carries a digest. Comparison posts about
    the same three localities open identically, and without this the second
    one would be written into the first one's directory and overwrite its
    brief — silently, since the directory already existing is the normal
    case for a regeneration."""
    first = slugify(REPORTED, "BLOG")
    second = slugify(REPORTED.replace("future development", "upcoming metro lines"), "BLOG")
    assert first != second
    assert first.split("-")[:6] == second.split("-")[:6], "they should still read alike"


def test_a_topic_in_a_script_the_slug_strips_is_still_unique():
    """Devanagari survives none of this, so every such topic used to be
    called BLOG and land in the same folder. This is an Indian
    property product; that is not a hypothetical."""
    one = slugify("व्हाइटफील्ड बनाम सरजापुर रोड", "BLOG")
    two = slugify("इलेक्ट्रॉनिक सिटी बेंगलुरु", "BLOG")
    assert one != two
    assert one.startswith("BLOG-") and two.startswith("BLOG-")


def test_an_ordinary_name_is_left_alone():
    """The common case must not get uglier to fix the rare one — and a
    person regenerating a property has to land in the folder it already
    has, so nothing random may enter a slug that fits."""
    assert slugify("Sky Villas at Estate 128", "PROP") == "SKY-VILLAS-AT-ESTATE-128"
    assert slugify("  Max Estates 128  ", "PROP") == "MAX-ESTATES-128"
    assert slugify("Sector 128, Noida", "PROP") == "SECTOR-128-NOIDA"


def test_the_same_input_always_gives_the_same_id():
    for text in (REPORTED, "Sky Villas at Estate 128", "व्हाइटफील्ड"):
        assert slugify(text, "BLOG") == slugify(text, "BLOG")


def test_nothing_at_all_falls_back_without_a_digest():
    """An empty topic is not a collision — there is only one of it — and a
    bare fallback is easier to recognise for what it is."""
    assert slugify("", "BLOG") == "BLOG"
    assert slugify("   ", "BLOG") == "BLOG"


def test_a_slug_contains_nothing_a_filesystem_objects_to():
    for text in (REPORTED, 'A "quoted" name: with / \\ * ? < > | characters',
                 "Trailing dots... and spaces   ", "₹11,000–13,000 per sq ft"):
        slug = slugify(text, "BLOG")
        assert slug == slug.strip()
        assert not slug.endswith((".", "-"))
        assert all(c.isalnum() or c == "-" for c in slug), slug


# --- ids that arrive from outside --------------------------------------------

def test_an_id_from_a_caller_cannot_point_outside_the_data_directory(tmp_path):
    """The HTTP API takes a property_id or a blog_id from the caller and
    joins it onto the data directory to find the brief. An id is something
    slugify produced — letters, digits and hyphens — but nothing was
    checking that, so '../../..' resolved outside the data root and the
    API would read a brief from wherever it landed."""
    import pytest

    from src.core.ids import brief_dir

    root = tmp_path / "blogs"
    root.mkdir()
    assert brief_dir(root, "BLOG-1234ABCD") == (root / "BLOG-1234ABCD").resolve()

    for hostile in ("../secrets", "..", ".", "a/b", "a\\b", "", "   ",
                    "BLOG/../..", "/etc/passwd", ".hidden"):
        with pytest.raises(ValueError):
            brief_dir(root, hostile)


def test_a_folder_someone_made_by_hand_is_still_usable(tmp_path):
    """The guard checks that an id names something directly inside the data
    directory — not that it looks like slugify output. A property folder
    put there by hand may well be called "Sky Villas 128" or
    "my_property", and those are perfectly good directories the person has
    every right to generate a brochure from. Rejecting them would trade a
    traversal bug for a broken feature."""
    from src.core.ids import brief_dir

    root = tmp_path / "properties"
    root.mkdir()
    for real in ("Sky Villas 128", "my_property", "max.estates", "e128", "a-b.c_d 1"):
        assert brief_dir(root, real) == (root / real).resolve()


def test_every_id_slugify_produces_survives_the_guard(tmp_path):
    """The guard and the generator have to agree, or the API rejects ids
    its own UI created."""
    from src.core.ids import brief_dir

    root = tmp_path / "blogs"
    root.mkdir()
    for text in (REPORTED, "Sky Villas at Estate 128", "व्हाइटफील्ड बनाम सरजापुर रोड",
                 "", "   ", "₹11,000–13,000 per sq ft", "A" * 400):
        brief_dir(root, slugify(text, "BLOG"))      # must not raise
