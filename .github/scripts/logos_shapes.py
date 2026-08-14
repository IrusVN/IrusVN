#!/usr/bin/env python3
"""
Point clouds for the three logos the traveller dots morph between.

Each shape comes from exact geometry -- the Vue mark and the </> glyph from
their outlines, the Irus mark traced from logos/icon_white.png -- then sampled
to exactly N points. Hand-drawing a logo gives a shape that reads as
almost-right, which looks worse than a clean trace, so anything without a
reference or a precise construction is left out.

The three silhouettes are deliberately unalike: </> spreads horizontally in
thin strokes, Vue is a solid downward wedge, Irus is a wide diagonal mark. A
morph between two similar outlines just looks like a shape wobbling.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# Vue.js: outer V with the inner V notch cut out of it.
VUE_OUTER = [(0.0, 0.0), (0.195, 0.0), (0.5, 0.52), (0.805, 0.0), (1.0, 0.0), (0.5, 0.86)]
VUE_INNER = [(0.30, 0.0), (0.5, 0.34), (0.70, 0.0), (0.585, 0.0), (0.5, 0.145), (0.415, 0.0)]

# </> as three stroked polylines: left chevron, slash, right chevron.
CODE_STROKES = [
    [(0.30, 0.74), (0.09, 0.50), (0.30, 0.26)],
    [(0.415, 0.20), (0.585, 0.80)],
    [(0.70, 0.74), (0.91, 0.50), (0.70, 0.26)],
]
CODE_WIDTH = 0.085


def _poly_mask(poly, size, flip_y=True):
    """Rasterise a normalised polygon into a boolean mask of `size`x`size`."""
    n = size
    pts = np.array(poly, dtype=np.float64)
    xs = pts[:, 0] * (n - 1)
    ys = (1.0 - pts[:, 1]) * (n - 1) if flip_y else pts[:, 1] * (n - 1)

    yy, xx = np.mgrid[0:n, 0:n]
    px, py = xx + 0.5, yy + 0.5
    inside = np.zeros((n, n), bool)
    m = len(pts)
    # even-odd ray crossing test, vectorised over the whole raster
    for i in range(m):
        j = (i - 1) % m
        xi, yi, xj, yj = xs[i], ys[i], xs[j], ys[j]
        cond = ((yi > py) != (yj > py))
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xj - xi) * (py - yi) / np.where(yj - yi == 0, np.nan, yj - yi) + xi
        inside ^= np.nan_to_num(cond & (px < xint), nan=False).astype(bool)
    return inside


def vue_mask(size=256):
    return _poly_mask(VUE_OUTER, size) & ~_poly_mask(VUE_INNER, size)


def code_mask(size=256):
    """The </> glyph: distance-to-polyline thresholded, so joins stay clean.

    Stroking as polygons would need mitre maths at each corner; measuring
    distance to the segments instead gives round joins for free.
    """
    n = size
    yy, xx = np.mgrid[0:n, 0:n]
    px = (xx + 0.5) / n
    py = 1.0 - (yy + 0.5) / n
    best = np.full((n, n), np.inf)
    for stroke in CODE_STROKES:
        for (ax, ay), (bx, by) in zip(stroke, stroke[1:]):
            vx, vy = bx - ax, by - ay
            L2 = vx * vx + vy * vy
            t = ((px - ax) * vx + (py - ay) * vy) / L2
            t = np.clip(t, 0.0, 1.0)
            dx = px - (ax + t * vx)
            dy = py - (ay + t * vy)
            best = np.minimum(best, np.hypot(dx, dy))
    return best < (CODE_WIDTH / 2)


def _resolve(path):
    """Find a repo asset whether we are run from the root or from scripts/."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (path,
                 os.path.join(here, "..", "..", path),
                 os.path.join(here, path)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(path)


def image_mask(path, size=256, alpha_thresh=90):
    """Mask from a PNG's alpha channel, trimmed to content and squared off."""
    im = Image.open(_resolve(path)).convert("RGBA")
    a = np.asarray(im).astype(np.float32)
    al = a[..., 3]
    ys, xs = np.nonzero(al > alpha_thresh)
    im = im.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    w, h = im.size
    side = max(w, h)
    sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    sq.paste(im, ((side - w) // 2, (side - h) // 2))
    sq = sq.resize((size, size), Image.LANCZOS)
    m = np.asarray(sq).astype(np.float32)[..., 3] > alpha_thresh
    return ndi.binary_fill_holes(m)


def sample_mask(mask, n, box, seed=0):
    """Pick n points spread over a mask, mapped into box=(x0,y0,w,h).

    Uses a jittered grid rather than pure rejection sampling: pure random
    sampling clumps, and clumps read as blotches once the dots are 2.4x1.7.
    """
    rng = np.random.default_rng(seed)
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise ValueError("empty logo mask")

    # oversample candidate cells, then thin down to exactly n by farthest-ish
    # spacing so coverage stays even
    idx = rng.permutation(len(ys))
    take = idx[: max(n * 4, n)]
    cand = np.stack([xs[take], ys[take]], axis=1).astype(np.float64)
    cand += rng.uniform(-0.5, 0.5, cand.shape)

    chosen = _thin_to(cand, n, rng)

    x0, y0, bw, bh = box
    # normalise by the mask's own content bbox so the shape fills the box
    cx0, cx1 = xs.min(), xs.max()
    cy0, cy1 = ys.min(), ys.max()
    sw = max(cx1 - cx0, 1)
    sh = max(cy1 - cy0, 1)
    scale = min(bw / sw, bh / sh)
    out = np.empty_like(chosen)
    out[:, 0] = x0 + (bw - sw * scale) / 2 + (chosen[:, 0] - cx0) * scale
    out[:, 1] = y0 + (bh - sh * scale) / 2 + (chosen[:, 1] - cy0) * scale
    return out


def _thin_to(pts, n, rng):
    """Reduce pts to n by repeatedly dropping the closest-together pair."""
    if len(pts) <= n:
        reps = int(np.ceil(n / len(pts)))
        pts = np.tile(pts, (reps, 1))[: n]
        return pts
    # grid-bucket thinning: keep one point per occupied cell, growing the cell
    # size until the count lands at or below n, then top up
    lo, hi = 0.25, 64.0
    best = None
    for _ in range(40):
        cell = (lo + hi) / 2
        keys = np.floor(pts / cell).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        k = len(first)
        if k >= n:
            best = first
            lo = cell
        else:
            hi = cell
        if abs(k - n) <= max(2, n // 200):
            best = first if k >= n else best
            break
    if best is None:
        best = rng.permutation(len(pts))[:n]
    sel = pts[np.sort(best)]
    if len(sel) > n:
        keep = np.round(np.linspace(0, len(sel) - 1, n)).astype(int)
        sel = sel[keep]
    elif len(sel) < n:
        extra = pts[rng.permutation(len(pts))[: n - len(sel)]]
        sel = np.vstack([sel, extra])
    return sel


def logo_clouds(n, box, icon_path="logos/icon_white.png"):
    """The three logo point clouds, each exactly n points, in grid coords."""
    return [
        sample_mask(code_mask(), n, box, seed=11),
        sample_mask(vue_mask(), n, box, seed=22),
        sample_mask(image_mask(icon_path), n, box, seed=33),
    ]


if __name__ == "__main__":
    box = (40, 60, 220, 220)
    clouds = logo_clouds(900, box)
    ramp = " .:-=+*#%@"
    for name, P in zip(("code", "vue", "irus"), clouds):
        x0, x1 = P[:, 0].min(), P[:, 0].max()
        y0, y1 = P[:, 1].min(), P[:, 1].max()
        Wc, Hc = 58, 26
        acc = np.zeros((Hc, Wc))
        for x, y in P:
            cx = int((x - x0) / (x1 - x0 + 1e-9) * (Wc - 1))
            cy = int((y - y0) / (y1 - y0 + 1e-9) * (Hc - 1))
            acc[cy, cx] += 1
        print(f"\n=== {name}  n={len(P)}  bbox {x1-x0:.0f}x{y1-y0:.0f} ===")
        mx = max(acc.max(), 1e-9)
        for r in acc:
            print("  |" + "".join(ramp[min(9, int(v * 9.99 / mx))] if v else " " for v in r) + "|")
