#!/usr/bin/env python3
"""Regenerate the comparison images in docs/.

    python tools/make_docs.py "Current Task" output

Every panel is rendered by actually running the pipeline, including the "without" halves --
those switch one fix off and run it again, so the figures cannot drift away from the code.
Needs the source posters, which are not in the repository.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import kabbik_batch as K  # noqa: E402

DOCS = ROOT / "docs"
BAR, PAD, GAP = 30, 10, 10
INK, BG = (235, 237, 240), (17, 19, 22)


def font(size=17):
    for p in ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def panels(items, width=520):
    """items: list of (caption, HxWx3 array). Lays them out side by side with captions."""
    f = font()
    tiles = []
    for cap, arr in items:
        im = Image.fromarray(arr)
        im = im.resize((width, max(1, round(width * im.height / im.width))), Image.LANCZOS)
        tiles.append((cap, im))
    h = max(t.height for _, t in tiles)
    out = Image.new("RGB", (len(tiles) * width + (len(tiles) - 1) * GAP + 2 * PAD,
                            h + BAR + 2 * PAD), BG)
    d = ImageDraw.Draw(out)
    for i, (cap, t) in enumerate(tiles):
        x = PAD + i * (width + GAP)
        d.text((x + 2, PAD + 6), cap, fill=INK, font=f)
        out.paste(t, (x, PAD + BAR))
    return out


def crop(img, box):
    x0, y0, x1, y1 = box
    return np.asarray(img)[y0:y1, x0:x1]


def diff_box(a, b, pad=70, aspect=16 / 10):
    """Where do these two renders disagree most? Crop both there, so a figure about a
    three-pixel speck actually shows the speck instead of me guessing coordinates."""
    import cv2
    d = np.abs(np.asarray(a).astype(int) - np.asarray(b).astype(int)).max(axis=2)
    m = (d > 18).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        h, w = d.shape
        return (0, 0, w, h)
    j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = st[j, :4]
    cx, cy = x + w / 2, y + h / 2
    half_h = max(h / 2 + pad, 60)
    half_w = max(half_h * aspect, w / 2 + pad)
    half_h = half_w / aspect
    H, W = d.shape
    x0 = int(max(0, min(W - 2 * half_w, cx - half_w)))
    y0 = int(max(0, min(H - 2 * half_h, cy - half_h)))
    return (x0, y0, int(min(W, x0 + 2 * half_w)), int(min(H, y0 + 2 * half_h)))


class Pipe:
    def __init__(self, src: Path, out: Path):
        self.src, self.out = src, out
        self.finder = K.LogoFinder()
        self.ink = K.Inpainter("lama")

    def original(self, name):
        return Image.open(self.src / f"{name}.png").convert("RGB")

    def processed(self, name):
        return Image.open(self.out / f"{name}.webp").convert("RGB")

    def hit(self, rgb):
        return self.finder.find(np.asarray(rgb))

    def run(self, name, *, refine=True, grow=True, windowed=True):
        """Inpaint one poster, optionally with a fix switched off."""
        rgb = np.asarray(self.original(name)).copy()
        hit = self.hit(rgb)
        if not refine:
            hit.wordmark = None
        keep = K._grow_into_leftovers
        if not grow:
            K._grow_into_leftovers = lambda m, r, reach: m
        try:
            mask = self.finder.logo_mask(rgb, hit, hit.has_text)
        finally:
            K._grow_into_leftovers = keep
        if windowed:
            return self.ink(rgb, mask), mask
        # Whole frame, as a naive call would. Despeckled like the real path, so the figure
        # isolates the windowing and nothing else.
        patch = K._despeckle(self.ink._lama(rgb, mask), mask)
        out = rgb.copy()
        sel = mask > 0
        out[sel] = patch[sel]
        return out, mask

    def overlay(self, name, *, refine=True):
        rgb = np.asarray(self.original(name)).copy()
        hit = self.hit(rgb)
        if not refine:
            hit.wordmark = None
        mask = self.finder.logo_mask(rgb, hit, hit.has_text)
        ov = rgb.copy()
        ov[mask > 0] = (0, 230, 90)
        return ov


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) < 2:
        print(__doc__)
        return 2
    src, out = Path(argv[0]), Path(argv[1])
    DOCS.mkdir(exist_ok=True)
    p = Pipe(src, out)

    # 1. The headline: logo in, logo gone.
    rows = []
    for name, box in [("bagh-singho-raja-ki-ojeyo-1024x1535", (0, 0, 520, 230)),
                      ("2_mathay_Jonmer_Itihas", (0, 0, 520, 230))]:
        rows.append(np.concatenate([crop(p.original(name), box),
                                    crop(p.processed(name), box)], axis=1))
    panels([("before  /  after", np.concatenate(rows, axis=0))], width=1040).save(DOCS / "hero.png")
    print("docs/hero.png")

    # 2. Detection: what the mask actually covers.
    tiles = [p.overlay(n)[:200, :460] for n in
             ("ek-ratei-sada-chul-1024x1535", "dotta-cinematic-1024x1535",
              "mukuter-niche-jontrona-1024x1535")]
    panels([("icon + wordmark", tiles[0]), ("disc only", tiles[1]), ("icon + wordmark", tiles[2])],
           width=340).save(DOCS / "detection.png")
    print("docs/detection.png")

    # 3. Two lockups: one rigid placement cannot sit on both ends.
    name, box = "prachin-romer-bondho-dorjar-opare-1024x1535", (20, 40, 340, 150)
    panels([("one fit, from the disc alone", crop(Image.fromarray(p.overlay(name, refine=False)), box)),
            ("disc fit + wordmark fit", crop(Image.fromarray(p.overlay(name)), box))],
           width=520).save(DOCS / "lockup.png")
    print("docs/lockup.png")

    # 4. The logo's last pixels, just outside the stamp.
    name = "536-chhayahin-dupur-1024x1535"
    without_, _ = p.run(name, grow=False)
    with_, _ = p.run(name)
    box = diff_box(without_, with_)
    panels([("mask alone", crop(Image.fromarray(without_), box)),
            ("mask + leftover sweep", crop(Image.fromarray(with_), box))],
           width=520).save(DOCS / "leftovers.png")
    print("docs/leftovers.png")

    # 5. What 100 KB costs an 8.3 MP banner.
    rel = ("Kabbik Gallery/Uploded Done/messir-football-jibon-cover-package/"
           "messir-football-jibon_website-home-hero-banner_3840x2160_16x9")
    a = Image.open(src / f"{rel}.png").convert("RGB")
    b = Image.open(out / f"{rel}.webp").convert("RGB")
    box = (1500, 700, 2260, 1120)
    panels([("3840x2160 source", crop(a, box)),
            ("same, forced under 100 KB", crop(b, box))], width=520).save(DOCS / "budget.png")
    print("docs/budget.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
