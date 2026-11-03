"""Where to cut a photograph so it keeps its subject.

A fixed rule ("keep the top of architecture") fails on the very photographs
it was written for: a tall render of a gate with half its height in sky,
cut to a wide band, kept the sky and lost the building. The subject of an
architectural or interior render is where the DETAIL is — edges, texture,
windows, planting — and the parts a crop can lose are the flat ones: sky,
road, lawn, floor.

So the crop window is slid across the picture and placed where it holds the
most detail (the sum of local gradients, measured on a small greyscale
copy). Ties lean slightly above centre, which is where a building's crown
and a room's furniture usually sit. Deterministic, local, and free.
"""
from __future__ import annotations

from functools import lru_cache

SAMPLE = 192          # the long side of the measuring copy, in pixels


@lru_cache(maxsize=256)
def _energy(path: str):
    """Row and column detail profiles of the picture (numpy arrays), or None."""
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("L")
            im.thumbnail((SAMPLE, SAMPLE))
            a = np.asarray(im, dtype=np.float32)
    except Exception:  # noqa: BLE001 - an unreadable picture keeps the centre
        return None
    if a.shape[0] < 4 or a.shape[1] < 4:
        return None
    gx = np.abs(np.diff(a, axis=1))[:-1, :]
    gy = np.abs(np.diff(a, axis=0))[:, :-1]
    g = gx + gy
    return g.sum(axis=1), g.sum(axis=0)


def _best_offset(profile, window: int, bias: float) -> float:
    """The start of the window (as a share of the free travel) holding the
    most detail, nudged towards `bias` when windows score alike."""
    import numpy as np

    n = len(profile)
    if window >= n:
        return 0.5
    csum = np.concatenate([[0.0], np.cumsum(profile)])
    sums = csum[window:] - csum[:-window]
    travel = len(sums) - 1
    if travel <= 0:
        return 0.5
    peak = float(sums.max()) or 1.0
    positions = np.arange(len(sums)) / travel
    # within 4% of the best window, prefer the one nearest the bias
    score = sums / peak - 0.04 * np.abs(positions - bias)
    return float(positions[int(score.argmax())])


def focal_point(path: str, image_aspect: float, slot_aspect: float, topic: str = "") -> str:
    """`object-position` for a photograph cut to a slot: "50% 37%"."""
    if not image_aspect or not slot_aspect or abs(image_aspect - slot_aspect) / slot_aspect < 0.02:
        return "50% 50%"
    profiles = _energy(path) if path else None
    architecture = topic in ("exterior", "night", "arrival", "aerial", "clubhouse")
    if image_aspect < slot_aspect:                     # the picture is taller: choose a band of rows
        if profiles is None:
            return "50% 30%" if architecture else "50% 45%"
        rows = profiles[0]
        window = max(1, round(len(rows) * image_aspect / slot_aspect))
        y = _best_offset(rows, window, 0.35 if architecture else 0.45)
        return f"50% {round(y * 100)}%"
    if profiles is None:                               # the picture is wider: choose a band of columns
        return "50% 50%"
    cols = profiles[1]
    window = max(1, round(len(cols) * slot_aspect / image_aspect))
    x = _best_offset(cols, window, 0.5)
    return f"{round(x * 100)}% 50%"
