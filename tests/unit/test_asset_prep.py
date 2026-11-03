"""P2: source images are normalised before layout."""
from __future__ import annotations

from PIL import Image, ImageDraw

from src.data.asset_prep import (
    burned_in_text_warning, flatten_onto_white, normalise_image, print_dpi, uniform_border_box,
)


def _drawing_with_white_canvas(path, canvas=(2000, 3000), draw_box=(700, 900, 1300, 2100)):
    """Like the real master plan: the drawing fills ~30% of its own file."""
    img = Image.new("RGB", canvas, (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle(draw_box, outline=(40, 40, 40), width=6)
    d.rectangle((draw_box[0] + 60, draw_box[1] + 60, draw_box[2] - 60, draw_box[3] - 60), fill=(170, 196, 160))
    img.save(path)
    return img


def test_blank_canvas_around_a_drawing_is_trimmed(tmp_path):
    src = tmp_path / "masterplan.png"
    _drawing_with_white_canvas(src)
    out, issues = normalise_image(src, tmp_path / "norm", "I1", "floorplan")

    assert out != src
    w, h = Image.open(out).size
    assert (w, h) == (601, 1201)                      # the drawing, not the canvas
    assert any(i.kind == "trimmed" for i in issues)


def test_photographs_are_never_trimmed(tmp_path):
    """Trimming is only safe for drawings on a blank ground; a photo with a
    pale sky must not be cropped."""
    src = tmp_path / "exterior.png"
    img = Image.new("RGB", (2000, 1200), (245, 245, 245))
    ImageDraw.Draw(img).rectangle((0, 700, 2000, 1200), fill=(90, 70, 60))
    img.save(src)
    out, issues = normalise_image(src, tmp_path / "norm", "I2", "exterior")
    assert out == src and not any(i.kind == "trimmed" for i in issues)


def test_a_drawing_that_already_fills_its_file_is_left_alone(tmp_path):
    src = tmp_path / "plan.png"
    _drawing_with_white_canvas(src, canvas=(1000, 1400), draw_box=(10, 10, 990, 1390))
    out, _ = normalise_image(src, tmp_path / "norm", "I3", "floorplan")
    assert out == src


def test_transparency_is_flattened_onto_white(tmp_path):
    """A transparent plan can otherwise render as white lines on black."""
    src = tmp_path / "plan.png"
    img = Image.new("RGBA", (800, 1000), (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle((100, 100, 700, 900), outline=(20, 20, 20, 255), width=8)
    img.save(src)
    out, issues = normalise_image(src, tmp_path / "norm", "I4", "floorplan")
    assert Image.open(out).mode == "RGB"
    # (this plan is also trimmed, so check the flattening step directly)
    flat = flatten_onto_white(Image.open(src))
    assert flat.mode == "RGB" and flat.getpixel((5, 5)) == (255, 255, 255)
    assert any(i.kind == "flattened" for i in issues)


def test_flatten_is_a_no_op_for_opaque_images():
    img = Image.new("RGB", (10, 10), (1, 2, 3))
    assert flatten_onto_white(img).getpixel((0, 0)) == (1, 2, 3)


def test_low_print_resolution_is_reported(tmp_path):
    src = tmp_path / "garden.png"
    Image.new("RGB", (1472, 900), (100, 120, 90)).save(src)      # the real garden render
    _, issues = normalise_image(src, tmp_path / "norm", "I5", "amenity", placement_fraction=1.0)
    [issue] = [i for i in issues if i.kind == "low_resolution"]
    assert "178 dpi" in issue.message


def test_resolution_is_judged_against_the_size_it_is_placed_at():
    assert print_dpi(1472, 1.0) < 200          # full page width: too soft
    assert print_dpi(1472, 0.5) > 300          # half width: fine


def test_burned_in_legal_text_on_a_photo_is_flagged():
    """Photos supplied as screenshots of an existing brochure."""
    issue = burned_in_text_warning(
        "I6", "For Estate 128 U.P. RERA Registration No.: UPRERAPRJ446459 www.up-rera.in", "exterior")
    assert issue and issue.kind == "burned_in_text" and "RERA" in issue.message


def test_text_on_a_drawing_is_not_flagged():
    """Plans legitimately carry labels and notes."""
    assert burned_in_text_warning("I7", "Level 1 | Carpet area 5789 sq ft | not to scale", "floorplan") is None


def test_unreadable_file_returns_the_original(tmp_path):
    src = tmp_path / "broken.png"
    src.write_bytes(b"not an image")
    out, issues = normalise_image(src, tmp_path / "norm", "I8", "floorplan")
    assert out == src and issues == []
