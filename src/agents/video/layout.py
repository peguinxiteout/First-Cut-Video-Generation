"""How a photograph fills a shot: fill (crop) or fit.

"fill" crops the photograph to the frame's shape around the focal point (the
largest window of that shape that fits), which for a wide picture in a tall
frame enlarges its pixels a long way: a 1568x1059 photograph in a 1080x1920
frame is a 596x1059 window shown 1.81 times larger, and the camera move
enlarges it further. Past `layout.fill_max_upscale` in configs/video/styles.yaml
that is soft, so the shot is laid out as "fit" instead (models/video_gen/fit.py):
the whole photograph across the frame width on a blurred copy of itself.

The peak upscale is the most the photograph's own pixels are enlarged at any
moment of the shot: the frame width over the window's width (fill) or over the
photograph's width (fit), times how close the camera gets at its closest.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.core.config import load_yaml_config
from src.models.video_gen import fit
from src.models.video_gen.kenburns import crop_box, peak_zoom


@dataclass(frozen=True)
class Layout:
    layout: str                  # fill | fit
    peak_upscale: float          # of the layout chosen
    fill_peak: float             # what fill would have been


def settings() -> dict:
    return load_yaml_config("video/styles.yaml")["layout"]


def fill_peak(src_w: int, src_h: int, out_w: int, out_h: int, focal: str, camera: str) -> float:
    left, top, right, bottom = crop_box(src_w, src_h, out_w / out_h, focal)
    return out_w / max(1, right - left) * peak_zoom(camera)


def choose(src_w: int, src_h: int, out_w: int, out_h: int, focal: str, camera: str,
           config: dict | None = None) -> Layout:
    config = config or settings()
    fill = fill_peak(src_w, src_h, out_w, out_h, focal, camera)
    if fill <= config["fill_max_upscale"]:
        return Layout("fill", round(fill, 3), round(fill, 3))
    return Layout("fit", round(fit.peak_upscale(src_w, src_h, out_w, out_h, camera, config["fit"]["zoom"]), 3), round(fill, 3))
