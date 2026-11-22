#!/usr/bin/env python3
"""Batch-process Kabbik posters: erase the Kabbik logo, then write WebP under a size budget.

    python kabbik_batch.py "Current Task" -o output

Stage 1 finds the logo by matching its magenta disc (scale-invariant), decides whether the
wordmark is present, and inpaints that rectangle. Images with no logo pass through untouched.
Stage 2 encodes WebP, picking the highest quality (then, if needed, the largest scale) that
still lands under the byte budget.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# --- Logo geometry -------------------------------------------------------------------
# assets/logo.png is the brand logo with its own alpha. It is cropped to the alpha bounding
# box and resized to CANON_W wide; every box below is in those canonical coordinates, and
# `scale` throughout maps canonical pixels to image pixels.
CANON_W = 400
SPLIT_X = 138          # the empty column band between the disc and the wordmark
TEXT_X0 = 143          # wordmark starts here
REF_CANVAS_W = 1024.0  # poster width the scale search is centred on

# Measured over the 192-poster set: every real logo scored <=0.071 on the masked SQDIFF and
# had >=0.68 magenta in the disc; every false positive scored >=0.236. Both gates must pass.
MAX_MATCH_SQDIFF = 0.15
MIN_DISC_MAGENTA = 0.40
MIN_TEXT_CORR = 0.14         # bgdiff correlation that says the wordmark is there
SCALE_LO, SCALE_HI, SCALE_STEP = 0.42, 1.30, 0.0125
MASK_GROW = 11               # canonical px added around the alpha, to catch the drop shadow
COLOUR_GROW = 6              # canonical px of reach when sweeping up logo-coloured leftovers
LAMA_WINDOW_PAD = 1.5        # window around the hole, as a multiple of the hole's size
LAMA_SIDE = 512              # and the longest side that window is scaled to for the model
BAD_DETAIL = 0.5             # below this, say out loud that the budget cost real quality
LEFTOVER_CONTRAST = 55       # local-background difference that marks a stray logo pixel
LOCKUP_SPREAD = 1.06         # small second stamp, after the refine fit

ASSETS = Path(__file__).resolve().parent / "assets"
LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
LAMA_PATH = ASSETS / "big-lama.pt"
SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _bgdiff(rgb: np.ndarray) -> np.ndarray:
    """How far each pixel sits from its smooth local background. Independent of whether the
    wordmark is drawn light or dark, which is why it can compare the two logo variants."""
    bg = cv2.medianBlur(rgb, 121)
    return np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).max(axis=2).astype(np.float32)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    return float((a * b).sum() / d) if d else 0.0


# (hue_lo, hue_hi, min saturation, min value) for the logo's magenta disc and orange bubbles.
# Floors sit below what the asset itself measures, because the logo's anti-aliased rim blends
# with the poster behind it and comes out darker and less saturated than the solid artwork.
# Only ever applied in a thin ring hugging the mask, so over-reaching costs a few pixels of
# background that were about to be repainted anyway, while under-reaching leaves a bright speck.
BRAND_COLOUR = ((128, 180, 100, 50), (0, 30, 110, 80))


def brand_coloured(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    out = np.zeros(h.shape, bool)
    for lo, hi, smin, vmin in BRAND_COLOUR:
        out |= (h > lo) & (h < hi) & (s > smin) & (v > vmin)
    return out


def _grow_into_leftovers(mask: np.ndarray, rgb: np.ndarray, reach: int) -> np.ndarray:
    """Pull the logo's last few pixels in from just outside the mask.

    The stamp is a rigid shape fitted to a logo that was scaled and re-saved by hand, so its
    outermost edge can sit a pixel or two beyond -- the rim of a bubble, the tail of a glyph.
    Two signatures catch those: the brand's own magenta and orange, and, for the white
    wordmark, anything standing far off its local background. Both are taken only inside a
    thin ring hugging the mask, and only as small blobs, so an adjacent leaf or wall is left
    alone while a three-pixel speck is not."""
    if reach < 1:
        return mask
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return mask
    H, W = mask.shape
    m = 4 * reach + 8                                 # work in a window around the mask
    x0, y0 = max(0, xs.min() - m), max(0, ys.min() - m)
    x1, y1 = min(W, xs.max() + 1 + m), min(H, ys.max() + 1 + m)
    sub, win = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1].copy()
    standout = _bgdiff(sub) > LEFTOVER_CONTRAST
    candidate = brand_coloured(sub) | standout
    kernel = np.ones((2 * reach + 1,) * 2, np.uint8)
    limit = max(24, int(0.004 * (mask > 0).sum()))
    for _ in range(4):          # repeat: absorbing a sliver can expose the next one past it
        ring = cv2.dilate(win, kernel) & ~win
        add = ((ring > 0) & candidate).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(add, 8)
        keep = np.zeros_like(add)
        for j in range(1, n):
            if st[j, cv2.CC_STAT_AREA] <= limit:      # a speck, not the background itself
                keep[lab == j] = 255
        if not keep.any():
            break
        win = np.maximum(win, cv2.dilate(keep, np.ones((5, 5), np.uint8)))
    out = mask.copy()
    out[y0:y1, x0:x1] = win
    return out


@dataclass
class Hit:
    scale: float      # canonical logo pixels -> image pixels, fitted on the disc
    ox: float         # canonical origin in image coords
    oy: float
    has_text: bool
    magenta: float
    # Second fit, for the wordmark: (scale, ox, oy). Posters vary in how the wordmark is set
    # against the disc, so one rigid placement cannot sit on both ends at once; the disc fit
    # above is exact for the icon, and this one is exact for the wordmark.
    wordmark: tuple[float, float, float] | None = None

    def box(self, b) -> tuple[int, int, int, int]:
        return (int(b[0] * self.scale + self.ox), int(b[1] * self.scale + self.oy),
                int(np.ceil(b[2] * self.scale + self.ox)), int(np.ceil(b[3] * self.scale + self.oy)))


class LogoFinder:
    def __init__(self, logo_path: Path = ASSETS / "logo.png"):
        rgba = np.array(Image.open(logo_path).convert("RGBA"))
        ys, xs = np.nonzero(rgba[..., 3] > 8)
        rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h = max(1, round(CANON_W * rgba.shape[0] / rgba.shape[1]))
        rgba = cv2.resize(rgba, (CANON_W, h), interpolation=cv2.INTER_AREA)
        self.rgb, self.alpha = rgba[..., :3].copy(), rgba[..., 3]
        self.rgb[self.alpha < 128] = 0
        self.size = (CANON_W, h)
        self.disc = self._disc_box()
        self.full_box = (0, 0, CANON_W, h)
        self.icon_box = (0, 0, SPLIT_X, h)
        self.text_box = (TEXT_X0, 0, CANON_W, h)
        # The wordmark reference is the logo's own alpha: it marks where contrast must appear
        # without saying whether the strokes are drawn light or dark.
        self.ref_shape = cv2.GaussianBlur(
            self.alpha[:, TEXT_X0:].astype(np.float32), (0, 0), 1.5)
        x0, y0, x1, y1 = self.disc
        tpl = self.rgb[y0:y1, x0:x1]
        th, tw = tpl.shape[:2]
        yy, xx = np.mgrid[0:th, 0:tw]
        circle = ((((xx - (tw - 1) / 2) / ((tw - 1) / 2)) ** 2
                   + ((yy - (th - 1) / 2) / ((th - 1) / 2)) ** 2) <= 0.90)
        # Drop the transparent pixels too (the bubble outlines cut through the disc): what
        # shows through there is poster, not logo, so it must not count towards the match.
        circle &= self.alpha[y0:y1, x0:x1] > 128
        self.tpl, self.tpl_mask = tpl, cv2.merge([circle.astype(np.uint8) * 255] * 3)

    def _disc_box(self) -> tuple[int, int, int, int]:
        """The play-button disc. The bubbles touch it, so a bounding box over the magenta
        blob would swallow them; the deepest point of the blob is the disc centre instead,
        and its distance to the edge is the radius."""
        raw = (self.alpha[:, :SPLIT_X] > 128).astype(np.uint8)
        m = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))  # bridge the
        m = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)      # bubble gaps
        d = cv2.distanceTransform(m, cv2.DIST_L2, 5)[1:-1, 1:-1]
        _, _, _, (cx, cy) = cv2.minMaxLoc(d)
        # Closing inflates the blob, so take the radius from the untouched alpha instead:
        # the unbroken opaque run down the disc's centre column. A scale fitted to an
        # oversized disc would place the wordmark short of where it actually sits.
        col = raw[:, cx]
        y0 = cy
        while y0 > 0 and col[y0 - 1]:
            y0 -= 1
        y1 = cy
        while y1 + 1 < len(col) and col[y1 + 1]:
            y1 += 1
        r = (y1 - y0) / 2.0
        ccy = (y0 + y1) / 2.0
        return (int(round(cx - r)), int(round(ccy - r)),
                int(round(cx + r)), int(round(ccy + r)))

    def logo_mask(self, rgb: np.ndarray, hit: "Hit", with_text: bool) -> np.ndarray:
        """The logo's own alpha, grown a little, placed where it sits in the image.

        A few posters use a lockup whose wordmark is set wider against the same disc, so the
        alpha is stamped twice -- at the fitted scale and at LOCKUP_SPREAD times it, both
        pivoting on the disc centre -- and the two are unioned. The disc lands in the same
        place either way; the extra reach only shows up out along the wordmark."""
        H, W = rgb.shape[:2]
        cx, cy = (self.disc[0] + self.disc[2]) / 2.0, (self.disc[1] + self.disc[3]) / 2.0
        mask = np.zeros((H, W), np.uint8)

        def stamp(src, s0, ox, oy):
            px, py = cx * s0 + ox, cy * s0 + oy          # pivot both stamps on the disc
            for f in (1.0, LOCKUP_SPREAD):
                s = s0 * f
                dw, dh = max(1, int(src.shape[1] * s)), max(1, int(src.shape[0] * s))
                g = max(1, int(round(MASK_GROW * s)))
                a = cv2.dilate((cv2.resize(src, (dw, dh), interpolation=cv2.INTER_LINEAR) > 8
                                ).astype(np.uint8) * 255,
                               np.ones((2 * g + 1, 2 * g + 1), np.uint8))
                x0, y0 = int(round(px - cx * s)), int(round(py - cy * s))
                sx, sy = max(0, x0), max(0, y0)
                ex, ey = min(W, x0 + dw), min(H, y0 + dh)
                if ex > sx and ey > sy:
                    np.maximum(mask[sy:ey, sx:ex], a[sy - y0:ey - y0, sx - x0:ex - x0],
                               out=mask[sy:ey, sx:ex])

        stamp(self.alpha[:, :SPLIT_X], hit.scale, hit.ox, hit.oy)      # icon: disc fit
        if with_text:
            s, ox, oy = hit.wordmark or (hit.scale, hit.ox, hit.oy)
            stamp(self.alpha, s, ox, oy)                               # wordmark: refined fit
        return _grow_into_leftovers(mask, rgb, int(round(COLOUR_GROW * hit.scale)))


    def find(self, rgb: np.ndarray) -> Hit | None:
        H, W = rgb.shape[:2]
        region = rgb[:max(40, int(H * 0.14)), :max(40, int(W * 0.22))]
        th, tw = self.tpl.shape[:2]
        k = W / REF_CANVAS_W
        best = None
        for s in np.arange(SCALE_LO * k, SCALE_HI * k, SCALE_STEP * k):
            dh, dw = int(th * s), int(tw * s)
            if dh < 25 or dw < 25 or dh > region.shape[0] or dw > region.shape[1]:
                continue
            t = cv2.resize(self.tpl, (dw, dh), interpolation=cv2.INTER_AREA)
            m = cv2.resize(self.tpl_mask, (dw, dh), interpolation=cv2.INTER_AREA)
            r = cv2.matchTemplate(region, t, cv2.TM_SQDIFF_NORMED, mask=m)
            r[~np.isfinite(r)] = 1.0
            score, _, loc, _ = cv2.minMaxLoc(r)
            if best is None or score < best[0]:
                best = (score, s, loc[0], loc[1])
        if best is None or best[0] > MAX_MATCH_SQDIFF:
            return None
        _, s, dx, dy = best
        hit = Hit(s, dx - self.disc[0] * s, dy - self.disc[1] * s, False, 0.0)
        hit.magenta = self._disc_magenta(rgb, hit)
        if hit.magenta < MIN_DISC_MAGENTA:
            return None
        hit.has_text = self._text_corr(rgb, hit) >= MIN_TEXT_CORR
        # Without a wordmark the logo *is* the disc, so the match above already fits it; a
        # re-fit there only has plain sky to work with and tends to drift.
        return self._refine(rgb, hit) if hit.has_text else hit

    def _refine(self, rgb: np.ndarray, hit: Hit) -> Hit:
        """Re-fit scale and position against the whole lockup.

        The disc alone pins the logo down well, but a scale read off a 100px circle is not
        precise enough at the far end of the wordmark, and some posters set the wordmark
        wider against the same disc. Matching the logo's silhouette to where the poster
        actually has graphic-like contrast catches both."""
        src = self.alpha if hit.has_text else self.alpha[:, :SPLIT_X]
        H, W = rgb.shape[:2]
        bw, bh = src.shape[1] * hit.scale, src.shape[0] * hit.scale
        x0, y0 = max(0, int(hit.ox - 0.15 * bw)), max(0, int(hit.oy - 0.30 * bh))
        x1, y1 = min(W, int(hit.ox + 1.35 * bw)), min(H, int(hit.oy + 1.40 * bh))
        crop = rgb[y0:y1, x0:x1]
        if crop.shape[0] < 24 or crop.shape[1] < 24:
            return hit
        d = _bgdiff(crop)
        best = None
        for fs in np.arange(0.90, 1.27, 0.02):
            s = hit.scale * fs
            dw, dh = int(src.shape[1] * s), int(src.shape[0] * s)
            if dw < 12 or dh < 12 or dw > crop.shape[1] or dh > crop.shape[0]:
                continue
            a = cv2.GaussianBlur(cv2.resize(src, (dw, dh), interpolation=cv2.INTER_AREA
                                            ).astype(np.float32), (0, 0), 2)
            r = cv2.matchTemplate(d, a, cv2.TM_CCOEFF_NORMED)
            r[~np.isfinite(r)] = -1
            _, mx, _, loc = cv2.minMaxLoc(r)
            if best is None or mx > best[0]:
                best = (mx, s, x0 + loc[0], y0 + loc[1])
        if best is None or best[0] < 0.20:        # weak fit -> keep the disc's answer
            return hit
        _, s, nx, ny = best
        hit.wordmark = (s, float(nx), float(ny))
        return hit

    @staticmethod
    def _crop(rgb: np.ndarray, b) -> np.ndarray:
        H, W = rgb.shape[:2]
        x0, y0 = max(0, b[0]), max(0, b[1])
        x1, y1 = min(W, b[2]), min(H, b[3])
        return rgb[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else np.zeros((0, 0, 3), np.uint8)

    def _disc_magenta(self, rgb: np.ndarray, hit: Hit) -> float:
        p = self._crop(rgb, hit.box(self.disc))
        if p.size == 0 or min(p.shape[:2]) < 10:
            return 0.0
        hsv = cv2.cvtColor(p, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
        return float(((h > 125) & (h < 178) & (s > 90) & (v > 50)).mean())

    def _text_corr(self, rgb: np.ndarray, hit: Hit) -> float:
        p = self._crop(rgb, hit.box(self.text_box))
        if p.shape[0] < 20 or p.shape[1] < 40:
            return 0.0
        # Resize to the reference size first: _bgdiff's blur radius is in pixels, so the two
        # sides have to be compared at the same scale.
        h, w = self.ref_shape.shape
        return _corr(_bgdiff(cv2.resize(p, (w, h), interpolation=cv2.INTER_AREA)), self.ref_shape)


# --- Inpainting ----------------------------------------------------------------------

class Inpainter:
    """LaMa (TorchScript) when the weights are present, otherwise OpenCV Telea.

    Telea is a visible fallback on detailed backgrounds; it exists so the pipeline still
    runs without torch, not because it looks good."""

    def __init__(self, backend: str = "auto"):
        self.model = None
        self.device = "cpu"
        if backend in ("auto", "lama"):
            self.model = self._load_lama(required=backend == "lama")
        self.name = "lama" if self.model is not None else "telea"

    def _load_lama(self, required: bool):
        try:
            import torch
        except ImportError:
            if required:
                raise SystemExit("--inpaint lama needs torch: pip install torch")
            return None
        if not LAMA_PATH.exists():
            self._download(LAMA_URL, LAMA_PATH)
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.jit.load(str(LAMA_PATH), map_location=self.device)
        model.eval()
        return model

    @staticmethod
    def _download(url: str, dest: Path) -> None:
        import urllib.request
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"downloading LaMa weights (~200 MB) -> {dest}", flush=True)
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(dest)

    def __call__(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.model is None:
            bgr = cv2.inpaint(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, 7, cv2.INPAINT_TELEA)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return rgb
        # Run on a window around the hole, scaled to about what LaMa was trained on. Handed a
        # 1535px poster at full size it leaves little invented bright blobs in the fill; with
        # the hole at a sane share of a ~512px frame it does not. Also far quicker.
        H, W = rgb.shape[:2]
        pad = int(max(64, LAMA_WINDOW_PAD * max(xs.max() - xs.min(), ys.max() - ys.min())))
        x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
        x1, y1 = min(W, xs.max() + 1 + pad), min(H, ys.max() + 1 + pad)
        sub, sm = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        side = max(sub.shape[:2])
        if side > LAMA_SIDE:
            f = LAMA_SIDE / side
            dw, dh = max(8, int(sub.shape[1] * f)), max(8, int(sub.shape[0] * f))
            patch = self._lama(cv2.resize(sub, (dw, dh), interpolation=cv2.INTER_AREA),
                               cv2.resize(sm, (dw, dh), interpolation=cv2.INTER_NEAREST))
            patch = cv2.resize(patch, (sub.shape[1], sub.shape[0]), interpolation=cv2.INTER_CUBIC)
        else:
            patch = self._lama(sub, sm)
        patch = _despeckle(patch, sm)
        out = rgb.copy()
        sel = sm > 0
        out[y0:y1, x0:x1][sel] = patch[sel]          # keep every untouched pixel bit-exact
        return out

    def _lama(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        torch = self.torch
        H, W = rgb.shape[:2]
        ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8          # LaMa needs multiples of 8
        img = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="edge")
        msk = np.pad(mask, ((0, ph), (0, pw)), mode="constant")
        ti = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(self.device)
        tm = torch.from_numpy(msk)[None, None].float().div(255).to(self.device)
        tm = (tm > 0).float()
        with torch.inference_mode():
            out = self.model(ti, tm)
        out = out[0].permute(1, 2, 0).mul(255).clamp(0, 255).to("cpu", torch.uint8).numpy()
        return out[:H, :W]


def _despeckle(patch: np.ndarray, mask: np.ndarray,
               deviation: int = 26, max_area_frac: float = 0.015) -> np.ndarray:
    """Flatten small bright/dark blobs the inpainter invented inside the hole.

    LaMa occasionally drops a speck into an otherwise plain fill -- a bright dot in a night
    sky, a pale smudge on a dark wall. Everything inside the mask is invented anyway, so a
    blob that disagrees sharply with its own surroundings is replaced by them. Bounded to
    small blobs, so larger reconstructed structure is left alone, and to the mask interior,
    so the seam with the real photo is untouched."""
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8))
    if not inner.any():
        return patch
    med = cv2.medianBlur(patch, 15)
    off = np.abs(patch.astype(np.int16) - med.astype(np.int16)).max(axis=2)
    bad = ((inner > 0) & (off > deviation)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bad, 8)
    limit = max(36, int(max_area_frac * (mask > 0).sum()))
    keep = np.zeros(bad.shape, np.uint8)
    for j in range(1, n):
        if st[j, cv2.CC_STAT_AREA] <= limit:
            keep[lab == j] = 1
    if not keep.any():
        return patch
    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8)) > 0
    out = patch.copy()
    out[keep] = med[keep]
    return out


def erase_logo(rgb: np.ndarray, hit: Hit, finder: LogoFinder, inpaint: Inpainter) -> np.ndarray:
    return inpaint(rgb, finder.logo_mask(rgb, hit, hit.has_text))


# --- WebP under a byte budget --------------------------------------------------------

def _encode(img: Image.Image, quality: int) -> bytes:
    buf = BytesIO()
    img.save(buf, format="WEBP", quality=quality, method=6)
    return buf.getvalue()


def fit_webp(img: Image.Image, budget: int, q_min: int = 0, q_max: int = 95
             ) -> tuple[bytes, int, float]:
    """Highest WebP quality that still encodes under `budget` bytes.

    Pixel dimensions are never changed -- the posters are cut to fixed sizes and have to stay
    that way. Quality is tried first. A 3840x2160 banner cannot reach 100 KB on quality alone
    (even q=1 lands around 250 KB), so past that point detail is dropped instead: the image is
    scaled down and back up to its original size, which softens it while keeping the
    dimensions. Returns the detail factor so the caller can report it."""
    def best_quality(im):
        best, lo, hi = None, q_min, q_max
        while lo <= hi:                                   # binary search on quality
            mid = (lo + hi) // 2
            data = _encode(im, mid)
            if len(data) < budget:
                best, lo = (data, mid), mid + 1
            else:
                hi = mid - 1
        return best

    detail, cur = 1.0, img
    for _ in range(22):
        got = best_quality(cur)
        if got is not None:
            return got[0], got[1], detail
        detail *= 0.85
        w, h = max(16, int(img.width * detail)), max(16, int(img.height * detail))
        cur = img.resize((w, h), Image.LANCZOS).resize(img.size, Image.LANCZOS)
    return _encode(cur, q_min), q_min, detail


# --- Driver --------------------------------------------------------------------------

def iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUFFIXES:
            yield p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="folder to scan (recursively)")
    ap.add_argument("-o", "--out", type=Path, default=Path("output"))
    ap.add_argument("--budget", type=int, default=100 * 1024, help="max bytes per file")
    ap.add_argument("--inpaint", choices=["auto", "lama", "telea"], default="auto")
    ap.add_argument("--keep-logo", action="store_true", help="skip stage 1, only re-encode")
    ap.add_argument("--dry-run", action="store_true", help="report detections, write nothing")
    a = ap.parse_args(argv)

    if not a.src.is_dir():
        print(f"not a folder: {a.src}", file=sys.stderr)
        return 2

    finder = LogoFinder()
    inpaint = None if (a.keep_logo or a.dry_run) else Inpainter(a.inpaint)
    if inpaint is not None:
        print(f"inpaint backend: {inpaint.name} ({inpaint.device})")

    files = list(iter_images(a.src))
    found = over = 0
    softened: list[tuple[str, float]] = []
    for i, src in enumerate(files, 1):
        rgb = np.array(Image.open(src).convert("RGB"))
        hit = None if a.keep_logo else finder.find(rgb)
        if hit:
            found += 1
            if not a.dry_run:
                rgb = erase_logo(rgb, hit, finder, inpaint)
        if a.dry_run:
            tag = (f"logo s={hit.scale:.2f} text={int(hit.has_text)}" if hit else "no logo")
            print(f"[{i}/{len(files)}] {tag}  {src.relative_to(a.src)}")
            continue

        data, q, detail = fit_webp(Image.fromarray(rgb), a.budget)
        dst = a.out / src.relative_to(a.src).with_suffix(".webp")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        if len(data) >= a.budget:
            over += 1
        if detail < 1.0:
            softened.append((str(dst.relative_to(a.out)), detail))
        print(f"[{i}/{len(files)}] {len(data)/1024:6.1f}KB q={q:2d} {rgb.shape[1]}x{rgb.shape[0]}"
              f" {'detail %3d%%' % round(detail * 100) if detail < 1.0 else '          '}"
              f" {'logo-removed' if hit else '           -'}  {dst.relative_to(a.out)}")

    if a.dry_run:
        print(f"\n{len(files)} files, logo found in {found}")
        return 0

    print(f"\n{len(files)} files, logo found in {found}, {over} over budget, "
          f"{len(softened)} needed detail dropped to fit")
    harsh = sorted((d, n) for n, d in softened if d < BAD_DETAIL)
    if harsh:
        print(f"\n{len(harsh)} file(s) could only reach {a.budget // 1024} KB by losing most of"
              f" their detail. At these pixel dimensions that budget is not really reachable --"
              f" raise --budget for them, or let them be smaller images:")
        for d, n in harsh:
            print(f"  detail {round(d * 100):3d}%  {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
