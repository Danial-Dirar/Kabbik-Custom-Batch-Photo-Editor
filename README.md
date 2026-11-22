# Custom Photo Resizer

Batch tool for Kabbik poster artwork. For every image in a folder tree it:

1. finds the Kabbik logo (if present) and paints it out,
2. writes a WebP that lands **just under** a byte budget (100 KB by default).

Images with no logo skip step 1 and go straight to the encoder, so you can point it at a
mixed folder and let it sort itself out.

```bash
pip install -r requirements.txt
python kabbik_batch.py "Current Task" -o output
```

## What it does

### Finding the logo

The logo is placed by hand, so its position and size vary from poster to poster, the wordmark
is drawn light on some and dark on others, and a few posters set the wordmark wider against
the same disc. `assets/logo.png` is the brand logo with its own alpha; everything is derived
from it, cropped to the alpha bounding box and scaled to 400px wide.

- **Locate** — multi-scale `TM_SQDIFF_NORMED` against the disc, masked to the disc's circle
  *and* its alpha so the transparent cut-outs between the bubble outlines (poster, not logo)
  do not count. Searched in the top-left corner over 0.42–1.30× the width-implied size.
- **Accept or reject** — a real logo scores ≤0.071 on that match and fills ≥68% of the disc box
  with magenta; false positives (deep-red corners, mostly) score ≥0.236. The gates sit in the
  gap at 0.15 / 0.40.
- **Wordmark present?** — some posters carry only the disc. Both wordmark variants are compared
  through a local-background difference map, which keys on *where* contrast is rather than its
  sign, so light and dark text score alike. Absent ≤0.07, present ≥0.21; the gate is 0.14.
- **Mask** — the logo's own alpha, grown by a few pixels to take the drop shadow with it. It is
  stamped twice: the icon at the disc fit, and the wordmark at a second fit that matches the
  logo silhouette against the poster's contrast map. No single rigid placement sits on both
  ends at once when the lockup differs, so each end gets the fit that is exact for it.

Only the disc is erased when the wordmark is absent, which matters on posters where artwork
sits immediately to the right of the logo.

### Painting it out

[LaMa](https://github.com/advimman/lama) via the TorchScript build published by
[simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting); the weights
(~200 MB) are fetched to `assets/big-lama.pt` on first run.

It runs on a window around the hole, scaled to ~512px, not on the whole poster. Handed a
1535px frame at full size the model leaves small invented bright blobs in the fill; with the
hole at a reasonable share of a 512px frame it does not. Everything outside the mask is copied
through bit-exact.

`--inpaint telea` falls back to OpenCV and needs no torch, but it smears visibly on detailed
backgrounds — it is there so the pipeline still runs, not because it looks good.

### Hitting the size budget

Pixel dimensions never change — the posters are cut to fixed sizes and stay that way.

Binary search on WebP quality (1–95) for the highest value that still fits. That is enough for
everything up to about 2 megapixels. A 3840×2160 hero banner cannot reach 100 KB on quality
alone (even q=1 lands near 250 KB), so past that point detail is dropped instead: the image is
scaled down and back up to its original size, in 20% steps, until a quality search fits. The
file keeps its dimensions and gets softer. The log prints `detail NN%` for those, and the
summary counts them — if too many show up, raise `--budget` rather than ship mush.

## Options

| flag | default | |
|---|---|---|
| `-o, --out` | `output` | output root; input folder structure is mirrored |
| `--budget` | `102400` | max bytes per file |
| `--inpaint` | `auto` | `auto`, `lama`, or `telea` |
| `--keep-logo` | off | skip detection, only re-encode |
| `--dry-run` | off | report what would be detected, write nothing |

`python test_detect.py` composites the real logo onto synthetic posters and checks the finder
puts it back where it was put, keeps disc-only apart from the full lockup, covers every logo
pixel with the mask, and stays quiet on an image with no logo.

## Known limits

- The logo is only looked for in the top-left corner (22% of the width, 14% of the height).
  A logo placed elsewhere will not be found.
- Whether the wordmark is present is decided from local contrast. On a background busy enough
  to rival the wordmark's own contrast the test can read low, and only the disc gets erased.
  `--dry-run` prints the verdict per file, so a wrong call is visible before anything is written.
- Torch on Python 3.15 is CPU-only at the time of writing, so LaMa runs on the CPU — correct,
  but slow. On a Python with CUDA wheels it picks the GPU up automatically.

## Retuning for a different logo

Replace `assets/logo.png` with the new logo on transparency and set `SPLIT_X` / `TEXT_X0` in
`kabbik_batch.py` to the empty column band between its icon and its wordmark, measured after
the canonical resize (alpha bbox, 400px wide). The disc anchor is found automatically, but it
assumes the icon's largest inscribed circle is the distinctive part. Run `--dry-run` and check
that the match scores still separate real hits from false ones.
