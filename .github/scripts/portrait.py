#!/usr/bin/env python3
"""
Portrait pipeline for the profile banner.

Turns one photo into the 300x340 1-bit dot grids the banner needs:
  * light mode keeps the background (dots draw the dark parts of the photo)
  * dark mode segments the background out (dots draw the lit subject)

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


def subject_mask(rgb, thresh=40.0):
    """Threshold on colour distance from the backdrop, then clean up.

    closing -> fill holes -> keep largest component, which is the run the
    Master Prompt calls for. Without the largest-component step, JPEG noise
    in the corners survives as stray islands.
    """
    bg = estimate_background(rgb)
    dist = np.linalg.norm(rgb - bg, axis=2)
    m = dist > thresh
    m = ndi.binary_closing(m, np.ones((15, 15)))
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    return m, bg


def head_shoulders_crop(img):
    """Crop head + shoulders at the frame's aspect ratio.

    A tight face crop reads aggressive; the framing here keeps the top of the
    head near the frame edge and cuts around chest level.
    """
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    H, W, _ = rgb.shape
    m, _ = subject_mask(rgb)

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
    img = Image.open(path).convert("RGB")
    crop, box = head_shoulders_crop(img)
    small = crop.resize((GRID_W, GRID_H), Image.LANCZOS)

    gray = np.asarray(_prep_gray(small)).astype(np.float32)

    # light mode: background kept, dots draw the dark parts of the photo
    light = floyd_steinberg(gray)

    # dark mode: dots draw the *lit* subject on the dark panel, so invert
    # tone inside the subject and drop the background entirely
    rgb_small = np.asarray(small).astype(np.float32)
    mask, _ = subject_mask(rgb_small, thresh=40.0)
    mask = ndi.binary_closing(mask, np.ones((5, 5)))
    mask = ndi.binary_fill_holes(mask)

    inv = 255.0 - gray
    # push the (now bright) backdrop to white so it diffuses no error into
    # the subject, then hard-clear anything outside the mask afterwards
    inv_masked = np.where(mask, inv, 255.0)
    dark = floyd_steinberg(inv_masked)

    # hard-clear error-diffusion bleed at the mask edge: dither pushes error
    # sideways, so ink lands just outside the silhouette and reads as a halo
    eroded = ndi.binary_erosion(mask, np.ones((3, 3)))
    dark &= eroded

    return light, dark, box


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "logos/banner.jpeg"
    light, dark, box = head_shoulders_crop and build_grids(src)
    np.save("/tmp/grid_light.npy", light)
    np.save("/tmp/grid_dark.npy", dark)
    print(f"crop box (x,y,w,h) = {box}")
    print(f"light: {light.sum():6d} dots  ink {light.mean()*100:.1f}%")
    print(f"dark : {dark.sum():6d} dots  ink {dark.mean()*100:.1f}%")
