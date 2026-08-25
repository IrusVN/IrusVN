#!/usr/bin/env python3
"""
Portrait pipeline for the profile banner.

Turns one photo into the 300x340 1-bit dot grids the banner needs:
  * light mode keeps the background (dots draw the dark parts of the photo)
  * dark mode segments the background out (dots draw the lit subject)

A background-removed PNG (alpha channel) is accepted and preferred: its
silhouette is ground truth, which matters when the shirt is the same colour
as the wall and colour-distance segmentation bites into the shoulders.

Everything downstream reads the .npy files this writes -- those plus this
script are the source of truth, not the SVG.
"""
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from scipy import ndimage as ndi

GRID_W, GRID_H = 300, 340

# frame is 400x492 with the dots inset; scale(1.24, 1.4471) maps grid -> frame
DISP_ASPECT = 372.0 / 492.0

CONTRAST = 1.3          # 1.3x only -- 2.4x reads harsh and skull-like
UNSHARP = dict(radius=3, percent=140, threshold=2)
AUTOCONTRAST_CUTOFF = 1


def floyd_steinberg(gray):
    """1-bit Floyd-Steinberg dither, serpentine scan order.

    gray: float array 0..255. Returns bool array, True where ink goes.
    Serpentine (alternating scan direction) avoids the directional worm
    artefacts a fixed left-to-right pass leaves in flat gradients.
    """
    h, w = gray.shape
    e = gray.astype(np.float32).copy()
    ink = np.zeros((h, w), bool)
    for y in range(h):
        forward = (y % 2 == 0)
        xs = range(w) if forward else range(w - 1, -1, -1)
        row_next = e[y + 1] if y + 1 < h else None
        row = e[y]
        for x in xs:
            old = row[x]
            new = 255.0 if old >= 128.0 else 0.0
            ink[y, x] = (new == 0.0)
            err = old - new
            if forward:
                if x + 1 < w:
                    row[x + 1] += err * 7 / 16
                if row_next is not None:
                    if x > 0:
                        row_next[x - 1] += err * 3 / 16
                    row_next[x] += err * 5 / 16
                    if x + 1 < w:
                        row_next[x + 1] += err * 1 / 16
            else:
                if x - 1 >= 0:
                    row[x - 1] += err * 7 / 16
                if row_next is not None:
                    if x + 1 < w:
                        row_next[x + 1] += err * 3 / 16
                    row_next[x] += err * 5 / 16
                    if x - 1 >= 0:
                        row_next[x - 1] += err * 1 / 16
    return ink


def estimate_background(rgb):
    """Median colour of the frame border -- the backdrop, assuming flat bg."""
    h, w, _ = rgb.shape
    edges = np.concatenate([
        rgb[:max(1, int(h * 0.06))].reshape(-1, 3),
        rgb[:, :max(1, int(w * 0.05))].reshape(-1, 3),
        rgb[:, -max(1, int(w * 0.05)):].reshape(-1, 3),
    ])
    return np.median(edges, axis=0)


def subject_mask(rgb, thresh=40.0, alpha=None):
    """Subject silhouette, then clean up.

    If an alpha channel is supplied (background-removed PNG) it seeds the
    mask directly: a white shirt against a white wall is colour-identical
    to the backdrop and the colour test bites into the shoulder tops.
    Otherwise the mask comes from thresholding colour distance to the
    backdrop.

    Either way: closing -> fill holes -> keep largest component, which is
    the run the Master Prompt calls for. Without the largest-component
    step, JPEG noise in the corners survives as stray islands.
    """
    bg = estimate_background(rgb)
    if alpha is not None:
        m = alpha > 128
    else:
        dist = np.linalg.norm(rgb - bg, axis=2)
        m = dist > thresh
    m = ndi.binary_closing(m, np.ones((15, 15)))
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    return m, bg


def head_shoulders_crop(img, alpha=None):
    """Crop head + shoulders at the frame's aspect ratio.

    A tight face crop reads aggressive; the framing here keeps the top of the
    head near the frame edge and cuts around chest level.
    """
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    H, W, _ = rgb.shape
    m, _ = subject_mask(rgb, alpha=alpha)

    rows = [(y, np.nonzero(m[y])[0]) for y in range(H) if m[y].any()]
    if not rows:
        raise SystemExit("no subject found -- is the background flat?")
    top = rows[0][0]
    centres = [(x.min() + x.max()) // 2 for y, x in rows if y < int(H * 0.55)]
    cx = int(np.median(centres))

    # as tall as fits, so the crop keeps maximum resolution
    ch = H
    cw = int(round(ch * DISP_ASPECT))
    while cw > W and ch > 100:
        ch -= 10
        cw = int(round(ch * DISP_ASPECT))

    # a little headroom above the crown, then take the full height downward
    y0 = max(0, top - int(0.02 * ch))
    if y0 + ch > H:
        y0 = max(0, H - ch)
    x0 = cx - cw // 2
    x0 = max(0, min(x0, W - cw))
    return img.crop((x0, y0, x0 + cw, y0 + ch)), (x0, y0, cw, ch)


def _prep_gray(rgb_img):
    g = ImageOps.autocontrast(rgb_img.convert("L"), cutoff=AUTOCONTRAST_CUTOFF)
    g = ImageEnhance.Contrast(g).enhance(CONTRAST)
    return g.filter(ImageFilter.UnsharpMask(**UNSHARP))


def build_grids(path):
    """Return (light_ink, dark_ink) bool grids of shape (GRID_H, GRID_W)."""
    img = Image.open(path)
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)

    if has_alpha:
        alpha_full = np.asarray(img.split()[-1])
    else:
        alpha_full = None

    crop, box = head_shoulders_crop(img, alpha=alpha_full)
    x0, y0, cw, ch = box

    # flatten any transparency BEFORE the tone pipeline sees it: convert("RGB")
    # would otherwise leave fully-transparent pixels black and light mode
    # (which keeps the background) would dither the whole frame solid
    if has_alpha:
        wall = Image.new("RGB", crop.size, (255, 255, 255))
        wall.paste(crop, (0, 0), crop)
        crop = wall
    else:
        crop = crop.convert("RGB")
    small = crop.resize((GRID_W, GRID_H), Image.LANCZOS)

    gray = np.asarray(_prep_gray(small)).astype(np.float32)

    # Subject mask at grid resolution -- used by both modes below. For a
    # background-removed PNG the resized alpha channel IS the subject.
    rgb_small = np.asarray(small).astype(np.float32)
    alpha_small = None
    if alpha_full is not None:
        a_crop_img = Image.fromarray(alpha_full[y0:y0+ch, x0:x0+cw])
        alpha_small = np.asarray(
            a_crop_img.resize((GRID_W, GRID_H), Image.BILINEAR)).astype(np.float32)
    mask, _ = subject_mask(rgb_small, thresh=40.0, alpha=alpha_small)
    mask = ndi.binary_closing(mask, np.ones((5, 5)))
    mask = ndi.binary_fill_holes(mask)

    # light mode: background kept, dots draw the dark parts of the photo.
    # JPEG noise in the flat background gets amplified by autocontrast +
    # contrast + unsharp and dithers into sparse speckle outside the
    # silhouette. Suppress it by clearing small disconnected clusters of
    # ink outside the subject mask -- those are noise, not portrait detail.
    light = floyd_steinberg(gray)
    stray = light & ~mask
    stray_labelled, n_stray = ndi.label(stray)
    if n_stray > 0:
        stray_sizes = ndi.sum(stray, stray_labelled, range(1, n_stray + 1))
        # keep only components large enough to be real portrait edge detail;
        # everything smaller is JPEG-compression / sensor noise artefact
        keep_ids = np.flatnonzero(stray_sizes >= 25) + 1
        stray_keep = np.isin(stray_labelled, keep_ids)
        light &= ~(stray & ~stray_keep)

    # dark mode: dots draw the *lit* subject on the dark panel, so invert
    # tone inside the subject and drop the background entirely.
    inv = 255.0 - gray

    # Compress the inverted range before dithering. Without this, bright
    # fabric (white shirt) maps through inv to very low values that dither
    # as near-solid slabs, and shadow folds map to near-white and vanish.
    # A soft power curve pulls both extremes toward midtone so the FS
    # dither has room to lay down visible dot texture everywhere.
    inv_norm = np.clip(inv, 0.0, 255.0) / 255.0
    inv_compressed = np.power(inv_norm, 0.72) * 255.0
    inv_compressed = np.clip(inv_compressed, 18.0, 232.0)

    # push the (now bright) backdrop to white so it diffuses no error into
    # the subject, then hard-clear anything outside the mask afterwards
    inv_masked = np.where(mask, inv_compressed, 255.0)
    dark = floyd_steinberg(inv_masked)

    # hard-clear error-diffusion bleed at the mask edge: dither pushes error
    # sideways, so ink lands just outside the silhouette and reads as a halo
    eroded = ndi.binary_erosion(mask, np.ones((3, 3)))
    dark &= eroded

    return light, dark, box


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "logos/banner.jpeg"
    light, dark, box = build_grids(src)
    print(f"crop box (x,y,w,h) = {box}")
    print(f"light: {light.sum():6d} dots  ink {light.mean()*100:.1f}%")
    print(f"dark : {dark.sum():6d} dots  ink {dark.mean()*100:.1f}%")
