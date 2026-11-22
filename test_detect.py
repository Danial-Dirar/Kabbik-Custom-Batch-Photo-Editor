#!/usr/bin/env python3
"""Self-check for the logo finder: python test_detect.py

Composites the real logo onto synthetic backgrounds at known places and checks that the
finder puts it back where it was put, tells disc-only apart from the full lockup, and does
not fire on a poster that has no logo.
"""
import numpy as np
from PIL import Image

from kabbik_batch import ASSETS, SPLIT_X, LogoFinder


def background(w, h, seed):
    """Photo-ish texture: fine mid-tone detail, so the logo is the high-contrast thing in
    the frame the way it is on a real poster, not competing with huge bright blobs."""
    rng = np.random.default_rng(seed)
    small = rng.integers(70, 190, (max(2, h // 8), max(2, w // 8), 3), dtype=np.uint8)
    return np.array(Image.fromarray(small).resize((w, h), Image.BICUBIC))


def paste(bg, scale, x, y, icon_only=False):
    rgba = Image.open(ASSETS / "logo.png").convert("RGBA")
    a = np.array(rgba)
    ys, xs = np.nonzero(a[..., 3] > 8)
    rgba = Image.fromarray(a[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    w = max(1, int(400 * scale))
    rgba = rgba.resize((w, max(1, round(w * rgba.height / rgba.width))), Image.LANCZOS)
    if icon_only:
        rgba = rgba.crop((0, 0, max(1, int(SPLIT_X * scale)), rgba.height))
    out = Image.fromarray(bg.copy())
    out.paste(rgba, (x, y), rgba)
    return np.array(out)


def main():
    f = LogoFinder()
    failures = []

    def check(name, cond, detail=""):
        print(f"{'ok  ' if cond else 'FAIL'}  {name}{'' if cond else '  -> ' + detail}")
        if not cond:
            failures.append(name)

    for scale, x, y in [(1.0, 40, 30), (0.6, 25, 20), (1.4, 60, 45)]:
        img = paste(background(1400, 2100, 1), scale, x, y)
        hit = f.find(img)
        if hit is None:
            check(f"full lockup found (scale {scale})", False, "nothing detected")
            continue
        tol = max(8, 0.06 * 400 * scale)
        check(f"full lockup at scale {scale}",
              abs(hit.ox - x) < tol and abs(hit.oy - y) < tol
              and abs(hit.scale - scale) < 0.12 * scale and hit.has_text,
              f"got scale={hit.scale:.3f} at ({hit.ox:.0f},{hit.oy:.0f}) text={hit.has_text}")

    img = paste(background(1400, 2100, 2), 1.0, 50, 40, icon_only=True)
    hit = f.find(img)
    check("disc-only is not read as a full lockup",
          hit is not None and not hit.has_text,
          "not detected" if hit is None else f"has_text={hit.has_text}")

    # The mask has to cover every pixel the logo actually painted.
    img_bg = background(1400, 2100, 3)
    img = paste(img_bg, 1.0, 40, 30)
    hit = f.find(img)
    if hit is None:
        check("mask covers the logo", False, "nothing detected")
    else:
        mask = f.logo_mask(img, hit, hit.has_text)
        changed = (np.abs(img.astype(int) - img_bg.astype(int)).max(axis=2) > 10)
        missed = int((changed & (mask == 0)).sum())
        check("mask covers the logo", missed == 0, f"{missed} logo px left outside the mask")

    check("no logo -> no hit", f.find(background(1400, 2100, 4)) is None, "false positive")

    print("\n" + (f"{len(failures)} FAILED: {', '.join(failures)}" if failures else "all passed"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
