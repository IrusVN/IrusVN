#!/usr/bin/env python3
"""
Build the animated profile banner: dark.svg and light.svg.

  python3 .github/scripts/banner.py logos/banner.jpeg .
      recompute the grids from the photo, save portrait_{light,dark}.npy,
      render the SVGs

  python3 .github/scripts/banner.py --from-npy [-o OUTDIR]
      render from portrait_{dark,light}.npy already on disk (e.g. exported
      from tools/dot-editor after hand-editing dots). The .npy files are read
      only -- they are never overwritten in this mode, which is the whole
      point: rebuilding from the photo would wipe the hand edits.

Animation config: if banner_config.json exists at the repo root (or is given
via --config PATH), it overrides the loop-layer behaviour. Keys (all
optional, unknown keys are reported and ignored):

  anim            drift | ripple | shimmer | static   (default drift)
  loop_dur        seconds per loop                    (default 13.9)
  drift_fraction  how far bands travel, 0..0.8        (default 0.42)
  band_noise      scatter added to band grouping      (default 4.0)
  n_bands         number of drift/twinkle bands       (default 94)
  dot_shape       square | circle | diamond | plus    (default square)

--dot-shape SHAPE overrides dot_shape from the config file.

Presets:
  drift    current behaviour -- bands translate toward the first logo's
           centroid while fading, then return
  ripple   bands expand radially OUTWARD from the portrait's ink centroid
           (the same machinery with a negative fraction against the centre)
  shimmer  bands never move; each twinkles on its own clock behind the
           shared fade-out, so the portrait breathes until the logos take over
  static   the portrait holds completely still; only the shared fade remains

tools/dot-editor has an Animation panel that writes this file. With no
config file present the output is byte-identical to the pre-config script.

Structure, per the design the two SVGs share:

  intro layer  60 interleaved random groups fade in over ~2s, then switch off
  loop layer   duplicate of the portrait, ~94 drift bands, runs for ever
  travellers   900 dots that morph between the three logos

The intro and loop layers have to be separate elements. One layer cannot both
fade in per-group and drift per-band, because a group is a set of scattered
dots and a band is a set of dots that move together -- different partitions of
the same 300x340 grid.

Verified by measurement, not by eye: cairosvg renders only the first SMIL
frame and mishandles additive transforms and textLength, so the checks at the
bottom of this file measure the point data instead. Look at the result in a
browser before shipping it.
"""
import argparse
import html
import json
import os

import numpy as np

from logos_shapes import logo_clouds
from portrait import GRID_H, GRID_W, build_grids

# ---------------------------------------------------------------- geometry
W, H = 1180, 610
FRAME = dict(x=36, y=84, w=400, h=492)
DOT_X, DOT_Y = 50, 86
SCALE_X, SCALE_Y = 1.2400, 1.4471
PANEL_X = 470

FONT = ("ui-monospace,SFMono-Regular,Menlo,Consolas,"
        "'Liberation Mono',monospace")

# ---------------------------------------------------------------- timing
INTRO_GROUPS = 60
INTRO_START = 0.20
INTRO_STEP = 1.0 / 30.0
INTRO_FADE = 0.9
INTRO_END = 3.2

LOOP_DUR = 13.9
N_BANDS = 94
N_TRAVELLERS = 900

# Uneven on purpose. Evenly spaced keyTimes force every phase to hold for the
# same length, which flattens the portrait's 3s pause into just another beat.
# portrait 2.7s | (1.3s move) logo 2.0s | logo 2.0s | logo 2.0s | back
KEYTIMES = [0.000, 0.194, 0.288, 0.432, 0.525, 0.669, 0.763, 0.906, 1.000]

DRIFT_FRACTION = 0.42
BAND_NOISE_SIGMA = 4.0

ANIM_PRESETS = ("drift", "ripple", "shimmer", "static")

# Dot shapes. square is the historical look and the default: with it the
# emitted SVG is byte-identical to pre-shape output (no pattern is written).
# The others tile a 1x1-cell <pattern>; path data stays run-rectangles either
# way, so a different shape costs ~200 bytes, not one element per dot.
DOT_SHAPES = ("square", "circle", "diamond", "plus")


def _pattern_tile(shape, color):
    """SVG body of one 1x1 pattern tile for a shape ('square' never reaches
    here -- it takes the no-pattern fast path)."""
    if shape == "circle":
        # inscribed circle; r=.48 leaves a hairline so neighbours read apart
        return f'<circle cx=".5" cy=".5" r=".48" fill="{color}"/>'
    if shape == "diamond":
        return f'<path d="M.5 0L1 .5L.5 1L0 .5z" fill="{color}"/>'
    if shape == "plus":
        # plus arms at half cell width keep single isolated dots visible
        return (f'<path d="M.375 0h.25v.375H1v.25H.625V1h-.25V.625H0v-.25h.375z" '
                f'fill="{color}"/>')
    raise ValueError(shape)


def dot_pattern_def(theme, shape):
    """<defs> entry + paint url for `shape`, or (None, None) for square."""
    if shape == "square":
        return None, None
    tid = f"dots{theme}"
    return (
        f'<pattern id="{tid}" width="1" height="1" '
        f'patternUnits="userSpaceOnUse">'
        f'{_pattern_tile(shape, THEMES[theme]["dots"])}'
        f'</pattern>',
        f'url(#{tid})',
    )

THEMES = {
    "dark": dict(
        page="#070B16", panel_top="#0A101F", panel_bot="#0C1426",
        bar="#0B1222", bar_line="rgba(255,255,255,0.10)",
        title="#94A3B8", label="#475569",
        frame_glow="#22D3EE", frame_fill="#0A101F",
        frame_stroke="rgba(34,211,238,0.35)", corner="#22D3EE",
        dots="#A78BFA",
        accent=("#7C3AED", "#22D3EE", "#10B981"),
        ascii_grad=("#60A5FA", "#A78BFA", "#22D3EE"),
        head="#22D3EE", rule="rgba(255,255,255,0.10)",
        live="#F87171", pill_bg="#4C1D95", pill_tx="#E9D5FF",
        key="#22D3EE", val="#F8FAFC", leader="rgba(148,163,184,0.35)",
        muted="#94A3B8", foot="#94A3B8", cursor="#22D3EE",
    ),
    "light": dict(
        page="#FFFFFF", panel_top="#F8FAFC", panel_bot="#EEF2F7",
        bar="#F1F5F9", bar_line="rgba(15,23,42,0.10)",
        title="#475569", label="#94A3B8",
        frame_glow="#06B6D4", frame_fill="#F8FAFC",
        frame_stroke="rgba(8,145,178,0.40)", corner="#06B6D4",
        dots="#7C3AED",
        accent=("#2563EB", "#06B6D4", "#10B981"),
        ascii_grad=("#1D4ED8", "#7C3AED", "#0891B2"),
        head="#0891B2", rule="rgba(15,23,42,0.10)",
        live="#DC2626", pill_bg="#DBEAFE", pill_tx="#1D4ED8",
        key="#0891B2", val="#0F172A", leader="rgba(15,23,42,0.30)",
        muted="#475569", foot="#475569", cursor="#06B6D4",
    ),
}

# ---------------------------------------------------------------- content
SUBJECT = "Mai Lê Huy Hoàng - Irus"
HANDLE = "hoangmai020603@gmail.com"

ROWS = [
    ("Subject", SUBJECT),
    ("Role", "Full-Stack Developer"),
    ("Origin", "Ho Chi Minh, Viet Nam"),
    ("Education", "Industrial University of Ho Chi Minh city"),
    ("Status", "Learning + Building project"),
    ("ToolChain", "VS Code, Git, Docker, Postman"),
    None,
    ("Core.Lang", "PHP, JavaScript, TypeScript, Python"),
    ("Core.Frontend", "Vue.js, Nuxt.js"),
    ("Core.Backend", "Laravel, CodeIgniter"),
    ("Core.Database", "PostgreSQL, MongoDB"),
    ("Core.Infra", "Docker, Git, Claude"),
    "Contact",
    ("Grid.Mail", "hoangmai020603@gmail.com"),
    ("Grid.Portfolio", "portfolio.irusgear.me"),
    ("Grid.LinkedIn", "Hoàng Mai"),
    ("Grid.GitHub", "@IrusVN"),
    ("Grid.Facebook", "Hoàng Mai"),
]

ROW_W = 655
ROW_CHARS = 78


def esc(s):
    return html.escape(str(s), quote=True)


# ---------------------------------------------------------------- dot paths
def runs_from_mask(ink):
    """Horizontal runs as (x, y, length) -- one rect each, not one per pixel."""
    out = []
    h, w = ink.shape
    for y in range(h):
        row = ink[y]
        if not row.any():
            continue
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        for x0, x1 in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
            out.append((int(x0), int(y), int(x1 - x0)))
    return out


def path_d(runs):
    """Runs -> one path. Rectangles as M/h/v/h/z, which is the most compact
    form that still renders crisply; font glyphs mush below about 2px."""
    parts = []
    for x, y, n in runs:
        parts.append(f"M{x} {y}h{n}v1h-{n}z")
    return "".join(parts)


# ---------------------------------------------------------------- intro layer
INTRO_MAX_RUN = 6


def _split_long_runs(runs, limit=INTRO_MAX_RUN):
    """Chop long runs into short pieces before grouping.

    A run is one path command, so a group that happens to draw a 180px run
    lights up a whole stripe of shirt at once. Light mode has large dark
    areas, so 43% of its ink sits in runs longer than 8px; splitting them
    keeps every group's coverage even. Purely a grouping concern -- the
    rendered result is identical, and the loop layer keeps its long runs.
    """
    out = []
    for x, y, n in runs:
        if n <= limit:
            out.append((x, y, n))
            continue
        for off in range(0, n, limit):
            out.append((x + off, y, min(limit, n - off)))
    return out


def intro_groups(runs, n_groups=INTRO_GROUPS, seed=7):
    """Split runs into n_groups scattered across the whole portrait.

    Each group must cover the entire frame, so dots appear everywhere at once
    and thicken together. Grouping by spatial region reveals the portrait
    patch by patch; a wipe is worse still. A plain random permutation is what
    gives every group global coverage.
    """
    pieces = _split_long_runs(runs)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pieces))
    return [[pieces[i] for i in chunk]
            for chunk in np.array_split(idx, n_groups)]


def _cell_hist(runs, shape, cells):
    h, w = shape
    acc = np.zeros((cells, cells))
    for x, y, n in runs:
        acc[min(cells - 1, y * cells // h), min(cells - 1, x * cells // w)] += n
    return acc


def evenness(groups, runs, shape, cells=4, seed=0):
    """How well each group mirrors the whole portrait. ~0.05 good, ~0.7 patchy.

    Per group, the total-variation distance between its dot distribution and
    the portrait's, minus the distance a same-sized random draw would show
    anyway. Without subtracting that noise floor the score mostly measures
    group size, and a perfectly scattered split still looks bad.
    """
    pieces = [r for g in groups for r in g]
    total = _cell_hist(pieces, shape, cells)
    p = total / max(total.sum(), 1)

    scores = []
    for g in groups:
        acc = _cell_hist(g, shape, cells)
        s = acc.sum()
        if s:
            scores.append(np.abs(acc / s - p).sum() / 2)
    if not scores:
        return 0.0

    # Noise floor: resample the same number of *runs* (the independent unit --
    # a run's pixels all land in one cell), weighted by their length, and see
    # how far a genuinely random split lands from the whole.
    rng = np.random.default_rng(seed)
    idx_cell = np.array([
        min(cells - 1, y * cells // shape[0]) * cells
        + min(cells - 1, x * cells // shape[1])
        for x, y, _ in pieces
    ])
    weights = np.array([n for _, _, n in pieces], float)
    per_group = max(1, len(pieces) // len(groups))
    floor = []
    for _ in range(40):
        pick = rng.choice(len(pieces), size=per_group, replace=False)
        acc = np.bincount(idx_cell[pick], weights=weights[pick],
                          minlength=cells * cells).reshape(cells, cells)
        s = acc.sum()
        if s:
            floor.append(np.abs(acc / s - p).sum() / 2)
    floor = float(np.mean(floor)) if floor else 0.0

    return float(max(0.0, np.mean(scores) - floor))


# ---------------------------------------------------------------- loop layer
def drift_bands(runs, target, n_bands=N_BANDS, sigma=BAND_NOISE_SIGMA, seed=13,
                fraction=None):
    """Group runs into bands that drift together toward `target`.

    The trap: drift is a linear function of position, so quantising it into
    bands recreates a square grid and the dissolve looks blocky. Adding
    per-dot noise before grouping breaks the alignment, so band boundaries
    follow no straight line.

    `fraction` overrides DRIFT_FRACTION (banner_config.json). Passing
    fraction=0 keeps the grouping but freezes every band's translation --
    that is how the shimmer preset reuses this machinery.
    """
    rng = np.random.default_rng(seed)
    pts = np.array([(x + n / 2.0, y) for x, y, n in runs], dtype=np.float64)
    tx, ty = target

    d = np.hypot(pts[:, 0] - tx, pts[:, 1] - ty)
    d = d + rng.normal(0.0, sigma, len(d))

    order = np.argsort(d)
    bands = []
    frac = DRIFT_FRACTION if fraction is None else fraction
    for chunk in np.array_split(order, n_bands):
        if len(chunk) == 0:
            continue
        sel = [runs[i] for i in chunk]
        c = pts[chunk].mean(axis=0)
        dx = (tx - c[0]) * frac
        dy = (ty - c[1]) * frac
        bands.append((sel, (dx, dy)))
    return bands


def _boundary_concentration(band_img):
    """How much band boundaries pile onto single rows/columns.

    Walks each row and column counting where the band id changes, then asks
    how much of that lands in the busiest 5% of lines. Axis-aligned tiling
    puts every change on a handful of lines; organic bands spread them out.
    """
    scores = []
    for A in (band_img, band_img.T):
        h, w = A.shape
        cnt = np.zeros(w)
        for y in range(h):
            r = A[y]
            idx = np.nonzero(r >= 0)[0]
            if len(idx) < 2:
                continue
            changed = r[idx[:-1]] != r[idx[1:]]
            cnt[idx[:-1][changed]] += 1
        tot = cnt.sum()
        if tot <= 0:
            scores.append(0.0)
            continue
        p = cnt / tot
        k = max(1, int(0.05 * w))
        scores.append(max(0.0, float(np.sort(p)[::-1][:k].sum()) - k / w))
    return float(np.mean(scores))


def straight_boundary(bands, shape, seed=0):
    """Are band boundaries straight lines? ~0.01 organic, ~0.17 means a grid.

    The trap this catches: drift is a linear function of position, so
    quantising it into groups partitions the frame into axis-aligned tiles and
    the dissolve looks blocky. Measured against a shuffled baseline, since
    even random bands produce some incidental alignment.
    """
    h, w = shape
    img = np.full((h, w), -1, dtype=np.int32)
    flat = []
    for bi, (sel, _) in enumerate(bands):
        for x, y, n in sel:
            img[y, x:x + n] = bi
            flat.append(bi)

    actual = _boundary_concentration(img)

    # baseline: same band sizes, dots assigned at random -- no spatial
    # structure at all, which is the floor this metric can reach
    rng = np.random.default_rng(seed)
    shuf = np.array(flat)
    rng.shuffle(shuf)
    base_img = np.full((h, w), -1, dtype=np.int32)
    i = 0
    for sel, _ in bands:
        for x, y, n in sel:
            base_img[y, x:x + n] = shuf[i]
            i += 1
    baseline = _boundary_concentration(base_img)

    return float(max(0.0, actual - baseline))


# ---------------------------------------------------------------- travellers
def match_clouds(A, B):
    """Reorder B so B[i] pairs with A[i] along short paths.

    An unmatched morph sends dots across the whole frame and past each other,
    which reads as noise rather than one shape becoming another. This is
    auction-style: sort by angle from each cloud's centroid for a good start,
    then run swap passes that only ever shorten the total distance.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    n = len(A)

    def angular_order(P):
        c = P.mean(axis=0)
        v = P - c
        return np.lexsort((np.hypot(v[:, 0], v[:, 1]), np.arctan2(v[:, 1], v[:, 0])))

    ia = angular_order(A)
    ib = angular_order(B)
    perm = np.empty(n, dtype=int)
    perm[ia] = ib

    Bp = B[perm]
    cost = np.hypot(*(A - Bp).T)

    rng = np.random.default_rng(5)
    for _ in range(12):
        i = rng.permutation(n)
        j = np.roll(i, 1)
        # swapping i and j is worth it when the crossed pairing is shorter
        cur = cost[i] + cost[j]
        alt = (np.hypot(*(A[i] - Bp[j]).T) + np.hypot(*(A[j] - Bp[i]).T))
        better = alt < cur - 1e-9
        if not better.any():
            break
        ii, jj = i[better], j[better]
        # apply in one pass; each index appears at most twice so collisions
        # are rare and merely leave a pair unswapped
        Bp[ii], Bp[jj] = Bp[jj].copy(), Bp[ii].copy()
        perm[ii], perm[jj] = perm[jj].copy(), perm[ii].copy()
        cost[ii] = np.hypot(*(A[ii] - Bp[ii]).T)
        cost[jj] = np.hypot(*(A[jj] - Bp[jj]).T)
    return perm


def traveller_paths(home, clouds):
    """Per-dot keyframe positions: home -> logo1 -> logo2 -> logo3 -> home."""
    stages = [np.asarray(home, float)]
    for c in clouds:
        prev = stages[-1]
        c = np.asarray(c, float)
        stages.append(c[match_clouds(prev, c)])
    return stages


def transport_cost(stages):
    """Mean per-dot travel between consecutive stages, in grid units."""
    tot = []
    for a, b in zip(stages, stages[1:]):
        tot.append(float(np.hypot(*(np.asarray(a) - np.asarray(b)).T).mean()))
    return tot


def seed_home(runs, n, seed=3):
    """Pick n dot positions from the portrait as the travellers' rest state."""
    rng = np.random.default_rng(seed)
    pts = np.array([(x + n2 / 2.0, y) for x, y, n2 in runs], dtype=np.float64)
    idx = rng.permutation(len(pts))[:n]
    return pts[idx]


# ---------------------------------------------------------------- info panel
def leader_row(label, value, y, begin, t):
    """One SYSTEM.INFO row: label, computed dotted leader, right-aligned value.

    The leader length is derived from the label and value, never hand-edited,
    and textLength + lengthAdjust pin the whole row to ROW_W so the value
    stays right-aligned whatever mono font the browser actually has.
    """
    fill = ROW_CHARS - len(label) - 1 - len(value) - 1
    dots = "." * max(fill, 2)
    return (
        f'<g opacity="0">'
        f'<animate attributeName="opacity" from="0" to="1" dur="0.4s" '
        f'begin="{begin:.2f}s" fill="freeze"/>'
        f'<animateTransform attributeName="transform" type="translate" '
        f'values="-8 0;0 0" dur="0.4s" begin="{begin:.2f}s" fill="freeze"/>'
        f'<text x="{PANEL_X}" y="{y}" font-size="14" textLength="{ROW_W}" '
        f'lengthAdjust="spacingAndGlyphs" xml:space="preserve">'
        f'<tspan fill="{t["key"]}">{esc(label)} </tspan>'
        f'<tspan fill="{t["leader"]}">{dots}</tspan>'
        f'<tspan fill="{t["val"]}"> {esc(value)}</tspan>'
        f'</text></g>'
    )


def section_row(title, y, begin, t):
    fill = ROW_CHARS - len(title) - 3
    return (
        f'<g opacity="0">'
        f'<animate attributeName="opacity" from="0" to="1" dur="0.4s" '
        f'begin="{begin:.2f}s" fill="freeze"/>'
        f'<text x="{PANEL_X}" y="{y}" font-size="14" textLength="{ROW_W}" '
        f'lengthAdjust="spacingAndGlyphs" xml:space="preserve">'
        f'<tspan fill="{t["muted"]}">- {esc(title)} </tspan>'
        f'<tspan fill="{t["leader"]}">{"-" * max(fill, 2)}</tspan>'
        f'</text></g>'
    )


def info_panel(t):
    s = []
    a = s.append
    a(f'<text x="{PANEL_X}" y="106" font-size="13" letter-spacing="2" '
      f'fill="{t["head"]}" filter="url(#txtGlow)">SYSTEM.INFO</text>')
    a(f'<line x1="566" y1="102" x2="1061" y2="102" stroke="{t["rule"]}"/>')
    a(f'<text x="1125" y="106" text-anchor="end" font-size="12" '
      f'fill="{t["live"]}" font-weight="700"><tspan>&#9679;</tspan> LIVE'
      f'<animate attributeName="opacity" values="1;0.25;1" dur="1.6s" '
      f'repeatCount="indefinite"/></text>')

    # handle pill
    a(f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" '
      f'dur="0.5s" begin="0.6s" fill="freeze"/>')
    a(f'<rect x="{PANEL_X}" y="122" width="245" height="20" rx="4" '
      f'fill="{t["pill_bg"]}"/>')
    a(f'<text x="{PANEL_X + 9}" y="136" font-size="14" font-weight="700" '
      f'fill="{t["pill_tx"]}">{esc(HANDLE)}</text>')
    a(f'<line x1="725" y1="130" x2="1125" y2="130" stroke="{t["rule"]}"/>')
    a('</g>')

    y = 162
    begin = 0.90
    for row in ROWS:
        if row is None:          # blank line between identity and stack
            y += 8
            begin += 0.10
            continue
        if isinstance(row, str):  # section divider
            y += 8
            a(section_row(row, y, begin, t))
        else:
            a(leader_row(row[0], row[1], y, begin, t))
        y += 23
        begin += 0.12

    a(f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" '
      f'dur="0.5s" begin="3.34s" fill="freeze"/>')
    a(f'<text x="{PANEL_X}" y="{y + 8}" font-size="14" fill="{t["foot"]}">'
      f'&#9656; More about me &amp; projects below in README &#8595; '
      f'<tspan fill="{t["cursor"]}">&#9608;'
      f'<animate attributeName="fill-opacity" values="1;0;1" dur="1s" '
      f'repeatCount="indefinite"/></tspan></text>')
    a('</g>')
    return "".join(s)


# ---------------------------------------------------------------- chrome
def _traveller_body(shape, color):
    """Geometry of one travelling dot, ~2.4x1.7 cells, origin top-left."""
    if shape == "circle":
        return f'<ellipse rx="1.2" ry=".85" fill="{color}"/>'
    if shape == "diamond":
        return f'<path d="M1.2 0L2.4 .85L1.2 1.7L0 .85z" fill="{color}"/>'
    if shape == "plus":
        # scale the 1x1-cell plus up to the traveller footprint
        return f'<g transform="scale(2.4,1.7)">{_pattern_tile("plus", color)}</g>'
    raise ValueError(shape)


def traveller_defs(shape):
    """Defs entries for the 900 morphing dots; square = historical rects."""
    if shape == "square":
        return ('<rect id="tvdark" width="2.4" height="1.7" fill="#A78BFA"/>',
                '<rect id="tvlight" width="2.4" height="1.7" fill="#7C3AED"/>')
    return tuple(
        f'<g id="tv{theme}">'
        f'{_traveller_body(shape, THEMES[theme]["dots"])}</g>'
        for theme in ("dark", "light")
    )


def chrome(t, theme, shape="square", pat_def=None):
    a1, a2, a3 = t["accent"]
    g1, g2, g3 = t["ascii_grad"]
    a1, a2, a3 = t["accent"]
    g1, g2, g3 = t["ascii_grad"]
    s = []
    a = s.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
      f'viewBox="0 0 {W} {H}" font-family="{FONT}" role="img" '
      f'aria-label="{esc(SUBJECT)} — profile.sh --live">')
    a('<defs>')
    if pat_def:
        a(pat_def)
    a('<linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">')
    for off, c, seq in ((0, a1, (a1, a2, a3, a1)),
                        (0.5, a2, (a2, a3, a1, a2)),
                        (1, a3, (a3, a1, a2, a3))):
        a(f'<stop offset="{off}" stop-color="{c}">'
          f'<animate attributeName="stop-color" values="{";".join(seq)}" '
          f'dur="10s" repeatCount="indefinite"/></stop>')
    a('</linearGradient>')
    a(f'<linearGradient id="asciiGrad" x1="0" y1="0" x2="0" y2="520" '
      f'gradientUnits="userSpaceOnUse">'
      f'<stop offset="0" stop-color="{g1}"/>'
      f'<stop offset="0.45" stop-color="{g2}"/>'
      f'<stop offset="1" stop-color="{g3}"/>'
      f'<animateTransform attributeName="gradientTransform" type="translate" '
      f'values="0 -120; 0 120; 0 -120" dur="9s" repeatCount="indefinite"/>'
      f'</linearGradient>')
    a(f'<linearGradient id="panelGrad" x1="0" y1="0" x2="0" y2="1">'
      f'<stop offset="0" stop-color="{t["panel_top"]}"/>'
      f'<stop offset="1" stop-color="{t["panel_bot"]}"/></linearGradient>')
    for fid, dev in (("glow8", 8), ("glow3", 3)):
        a(f'<filter id="{fid}" x="-60%" y="-60%" width="220%" height="220%">'
          f'<feGaussianBlur stdDeviation="{dev}"/></filter>')
    a('<filter id="txtGlow" x="-30%" y="-30%" width="160%" height="160%">'
      '<feGaussianBlur stdDeviation="0.9" result="b"/><feMerge>'
      '<feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
      '</filter>')
    a('<clipPath id="winClip">'
      '<rect x="2" y="2" width="1176" height="606" rx="18"/></clipPath>')
    # Traveller dot definitions for both themes
    for d_ in traveller_defs(shape):
        a(d_)
    a('</defs>')

    a(f'<rect x="2" y="2" width="1176" height="606" rx="18" '
      f'fill="{t["page"]}"/>')
    a('<g clip-path="url(#winClip)">')
    a('<rect x="2" y="2" width="1176" height="606" fill="url(#panelGrad)"/>')
    a(f'<rect x="2" y="2" width="1176" height="46" fill="{t["bar"]}"/>')
    a(f'<line x1="2" y1="48" x2="1178" y2="48" stroke="{t["bar_line"]}"/>')
    for cx, col in ((30, "#ff5f56"), (50, "#ffbd2e"), (70, "#27c93f")):
        a(f'<circle cx="{cx}" cy="25.0" r="5.5" fill="{col}"/>')
    a(f'<text x="590.0" y="29.0" text-anchor="middle" font-size="12" '
      f'fill="{t["title"]}">{esc(HANDLE)} - % ./profile.sh --live</text>')
    a(f'<text x="38" y="74" font-size="10" letter-spacing="3" '
      f'fill="{t["label"]}">VISUAL.MAP</text>')
    f = FRAME
    a(f'<rect x="{f["x"]}" y="{f["y"]}" width="{f["w"]}" height="{f["h"]}" '
      f'rx="10" fill="none" stroke="{t["frame_glow"]}" stroke-width="2" '
      f'opacity="0.45" filter="url(#glow3)"/>')
    a(f'<rect x="{f["x"]}" y="{f["y"]}" width="{f["w"]}" height="{f["h"]}" '
      f'rx="10" fill="{t["frame_fill"]}" stroke="{t["frame_stroke"]}"/>')
    return "".join(s)


def corner_ticks(t):
    f = FRAME
    x0, y0 = f["x"], f["y"]
    x1, y1 = x0 + f["w"], y0 + f["h"]
    c = t["corner"]
    out = []
    for (ax, ay, bx, by, cx, cy) in (
        (x0 + 14, y0, x0, y0, x0, y0 + 14),
        (x1 - 14, y0, x1, y0, x1, y0 + 14),
        (x0 + 14, y1, x0, y1, x0, y1 - 14),
        (x1 - 14, y1, x1, y1, x1, y1 - 14),
    ):
        out.append(f'<path d="M {ax} {ay} L {bx} {by} L {cx} {cy}" fill="none" '
                   f'stroke="{c}" stroke-width="2" opacity="0.8"/>')
    return "".join(out)


def kt():
    return ";".join(f"{v:.3f}" for v in KEYTIMES)


# ---------------------------------------------------------------- config
CONFIG_NAME = "banner_config.json"


def load_config(path=None):
    """Read banner_config.json (repo root unless --config gives a path).

    Returns (cfg dict, path-or-None). Missing file -> ({}, None): the output
    must stay byte-identical to the pre-config script. Unknown keys and bad
    types are reported loudly instead of being silently dropped.
    """
    p = path or os.path.join(ROOT, CONFIG_NAME)
    if not os.path.exists(p):
        return {}, None
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise SystemExit(f"{p}: expected a JSON object at the top level")
    known = {"anim", "loop_dur", "drift_fraction", "band_noise", "n_bands",
             "dot_shape"}
    for k in sorted(set(raw) - known):
        print(f"config warning: unknown key {k!r} ignored")
    cfg = {}
    if "dot_shape" in raw:
        v = str(raw["dot_shape"]).lower()
        if v not in DOT_SHAPES:
            raise SystemExit(f"{p}: dot_shape must be one of "
                             f"{'/'.join(DOT_SHAPES)}, got {raw['dot_shape']!r}")
        cfg["dot_shape"] = v
    if "anim" in raw:
        v = str(raw["anim"]).lower()
        if v not in ANIM_PRESETS:
            raise SystemExit(f"{p}: anim must be one of "
                             f"{'/'.join(ANIM_PRESETS)}, got {raw['anim']!r}")
        cfg["anim"] = v
    for k, lo, hi in (("loop_dur", 4.0, 60.0),
                      ("drift_fraction", 0.0, 0.8),
                      ("band_noise", 0.0, 40.0)):
        if k in raw:
            try:
                v = float(raw[k])
            except (TypeError, ValueError):
                raise SystemExit(f"{p}: {k} must be a number")
            if not lo <= v <= hi:
                raise SystemExit(f"{p}: {k}={v} out of range [{lo}, {hi}]")
            cfg[k] = v
    if "n_bands" in raw:
        try:
            n = int(raw["n_bands"])
        except (TypeError, ValueError):
            raise SystemExit(f"{p}: n_bands must be an integer")
        if not 10 <= n <= 400:
            raise SystemExit(f"{p}: n_bands={n} out of range [10, 400]")
        cfg["n_bands"] = n
    return cfg, p


class AnimParams:
    """Resolved animation settings for one build() call.

    drift_fraction/band_noise/n_bands apply to every preset (shimmer still
    needs band grouping; ripple uses its own fraction); loop_dur always does.
    """

    def __init__(self, cfg):
        self.anim = cfg.get("anim", "drift")
        self.loop_dur = cfg.get("loop_dur", LOOP_DUR)
        self.drift_fraction = cfg.get("drift_fraction", DRIFT_FRACTION)
        self.band_noise = cfg.get("band_noise", BAND_NOISE_SIGMA)
        self.n_bands = cfg.get("n_bands", N_BANDS)
        self.dot_shape = cfg.get("dot_shape", "square")


def build(ink, theme, anim=None):
    """Assemble one themed SVG from a 300x340 ink grid.

    `anim` is an AnimParams; None means defaults, which reproduce the exact
    pre-config output.
    """
    ap = anim or AnimParams({})
    t = THEMES[theme]
    runs = runs_from_mask(ink)
    shape = getattr(ap, "dot_shape", "square")
    pat_def, paint = dot_pattern_def(theme, shape)

    # square keeps the historical fill + crispEdges rendering exactly; the
    # pattern shapes drop crispEdges because anti-aliased curves are the point
    chrome_svg = chrome(t, theme, shape, pat_def)
    if pat_def:
        open_layer = (f'<g transform="translate({DOT_X},{DOT_Y}) '
                      f'scale({SCALE_X:.4f},{SCALE_Y:.4f})" fill="{paint}">')
    else:
        open_layer = (f'<g transform="translate({DOT_X},{DOT_Y}) '
                      f'scale({SCALE_X:.4f},{SCALE_Y:.4f})" fill="{t["dots"]}" '
                      f'shape-rendering="crispEdges">')

    s = []
    a = s.append
    a(chrome_svg)

    # ---- intro: 60 interleaved random groups, then hand off to the loop
    groups = intro_groups(runs)
    a(open_layer)
    a(f'<set attributeName="opacity" to="0" begin="{INTRO_END}s"/>')
    for i, g in enumerate(groups):
        b = INTRO_START + i * INTRO_STEP
        a(f'<g opacity="0"><animate attributeName="opacity" values="0;1" '
          f'dur="{INTRO_FADE}s" begin="{b:.2f}s" fill="freeze" '
          f'calcMode="spline" keyTimes="0;1" keySplines=".4 0 .2 1"/>'
          f'<path d="{path_d(g)}"/></g>')
    a('</g>')

    # ---- loop layer: duplicate portrait partitioned into bands. All presets
    # share this structure -- they differ only in each band's translate values
    # and whether the per-band twinkle animate element is emitted.
    logo_box = (int(GRID_W * 0.16), int(GRID_H * 0.16),
                int(GRID_W * 0.68), int(GRID_H * 0.62))
    clouds = logo_clouds(N_TRAVELLERS, logo_box)

    ink_pts = np.array([(x + n / 2.0, y) for x, y, n in runs],
                       dtype=np.float64)
    ink_centroid = (float(ink_pts[:, 0].mean()), float(ink_pts[:, 1].mean()))
    first_centroid = (float(clouds[0][:, 0].mean()), float(clouds[0][:, 1].mean()))

    if ap.anim == "ripple":
        # outward from the portrait's own centroid: negative fraction against
        # that centre makes every band travel away from it while fading
        target = ink_centroid
        frac = -abs(ap.drift_fraction)
    elif ap.anim == "drift":
        target = first_centroid
        frac = ap.drift_fraction
    else:
        # shimmer / static keep the grouping but never translate
        target = first_centroid
        frac = 0.0

    bands = drift_bands(runs, target, n_bands=ap.n_bands,
                        sigma=ap.band_noise, fraction=frac)

    twinkle = ap.anim == "shimmer"

    a(open_layer[:-1] + ' opacity="0">')
    a(f'<set attributeName="opacity" to="1" begin="{INTRO_END}s"/>')
    for bi, (sel, (dx, dy)) in enumerate(bands):
        # The shared fade lives on the outer <g>; a per-band twinkle must sit
        # on a nested one. Two <animate> elements on the same attribute
        # compose as a sandwich where the later-begun one wins -- with both
        # repeating indefinitely the twinkle would override the fade forever
        # and the handoff to the logos would break. Nested opacities multiply,
        # so the fade still lands.
        body = f'<path d="{path_d(sel)}"/>'
        if twinkle:
            # staggered clocks so bands breathe out of sync
            lo = 0.55 + 0.15 * ((bi * 7) % 3)
            phase = (bi % 8) / 8.0
            body = (f'<g><animate attributeName="opacity" '
                    f'values="1;{lo};1" '
                    f'dur="{(2.6 + (bi % 5) * 0.7):.1f}s" '
                    f'begin="{INTRO_END + phase:.2f}s" '
                    f'repeatCount="indefinite"/>{body}</g>')
        tr = ""
        if (dx, dy) != (0, 0):
            pos = f"0 0;0 0;{dx:.0f} {dy:.0f}"
            vals = ";".join([pos.split(";")[0], pos.split(";")[1]]
                            + [f"{dx:.0f} {dy:.0f}"] * 5 + ["0 0"])
            tr = (f'<animateTransform attributeName="transform" '
                  f'type="translate" values="{vals}" keyTimes="{kt()}" '
                  f'dur="{ap.loop_dur}s" begin="{INTRO_END}s" '
                  f'repeatCount="indefinite"/>')
        a(f'<g opacity="1">'
          f'<animate attributeName="opacity" values="1;1;0;0;0;0;0;0;1" '
          f'keyTimes="{kt()}" dur="{ap.loop_dur}s" begin="{INTRO_END}s" '
          f'repeatCount="indefinite"/>'
          + tr +
          f'{body}</g>')
    a('</g>')

    # ---- travellers: 900 dots morphing between the logos
    home = seed_home(runs, N_TRAVELLERS)
    stages = traveller_paths(home, clouds)
    tid = f"tv{theme}"
    a(f'<g transform="translate({DOT_X},{DOT_Y}) '
      f'scale({SCALE_X:.4f},{SCALE_Y:.4f})">')
    P0, P1, P2, P3 = stages
    for i in range(N_TRAVELLERS):
        seq = [P0[i], P0[i], P1[i], P1[i], P2[i], P2[i], P3[i], P3[i], P0[i]]
        vals = ";".join(f"{p[0]:.0f} {p[1]:.0f}" for p in seq)
        # hidden while the portrait holds -- their thicker dots would crowd
        # the fine dither
        a(f'<use href="#{tid}" opacity="0">'
          f'<animate attributeName="opacity" values="0;0;1;1;1;1;1;1;0" '
          f'keyTimes="{kt()}" dur="{ap.loop_dur}s" begin="{INTRO_END}s" '
          f'repeatCount="indefinite"/>'
          f'<animateTransform attributeName="transform" type="translate" '
          f'values="{vals}" keyTimes="{kt()}" dur="{ap.loop_dur}s" '
          f'begin="{INTRO_END}s" repeatCount="indefinite"/></use>')
    a('</g>')

    a(corner_ticks(t))
    a(info_panel(t))
    a('</g>')
    a(f'<rect x="3" y="3" width="1174" height="604" rx="17" fill="none" '
      f'stroke="url(#accent)" stroke-width="3" opacity="0.55" '
      f'filter="url(#glow8)"/>')
    a(f'<rect x="3" y="3" width="1174" height="604" rx="17" fill="none" '
      f'stroke="url(#accent)" stroke-width="1.6"/>')
    a('</svg>')

    stats = dict(
        runs=len(runs), dots=int(ink.sum()),
        evenness=evenness(groups, runs, ink.shape),
        straight=straight_boundary(bands, ink.shape),
        bands=len(bands), transport=transport_cost(stages),
    )
    return "\n".join(s), stats


def _load_npy_grid(path):
    """Read one bool grid saved by portrait.build_grids or the dot editor.

    The editor exports the same |b1 layout numpy does, but a stray file from
    elsewhere must not silently render wrong -- so shape and dtype are checked
    against the pipeline's constants instead of trusted.
    """
    g = np.load(path)
    if g.dtype != np.bool_:
        raise SystemExit(f"{path}: expected bool grid, got {g.dtype}")
    if g.shape != (GRID_H, GRID_W):
        raise SystemExit(f"{path}: expected shape ({GRID_H}, {GRID_W}), got {g.shape}")
    return g


# portrait_dark.npy / portrait_light.npy live at the repo root (this file sits
# in <root>/.github/scripts). Pinning them there instead of the CWD matters:
# the old code saved wherever you happened to run from, which is how a stale
# second pair ended up inside .github/scripts.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(
        description="Build the animated profile banner SVGs.")
    ap.add_argument("src", nargs="?", default=None,
                    help="photo to recompute grids from; omit with "
                         "--from-npy (default: logos/banner-cutout.png if "
                         "present, else logos/banner.jpeg)")
    ap.add_argument("outdir", nargs="?", default=None,
                    help="where to write dark.svg / light.svg (default .)")
    ap.add_argument("--from-npy", action="store_true",
                    help="render from <root>/portrait_dark.npy / "
                         "portrait_light.npy already on disk; never "
                         "overwrites them (use after editing in "
                         "tools/dot-editor)")
    ap.add_argument("-o", "--outdir", dest="outdir_opt",
                    help="output directory (same as positional outdir)")
    ap.add_argument("--config", default=None,
                    help="path to banner_config.json (default: "
                         "<repo-root>/banner_config.json if present)")
    ap.add_argument("--anim", choices=ANIM_PRESETS,
                    help="animation preset; overrides banner_config.json")
    ap.add_argument("--dot-shape", choices=DOT_SHAPES,
                    help="dot shape for portrait + travellers; overrides "
                         "banner_config.json")
    args = ap.parse_args()

    # --from-npy takes no input file, so a lone positional after it is really
    # an output dir ("banner.py --from-npy build" must not land in the photo
    # slot and silently write to ".")
    outdir = args.outdir_opt or args.outdir or (
        args.src if args.from_npy else None) or "."

    cfg, cfg_path = load_config(args.config)
    if args.anim:
        cfg["anim"] = args.anim
        print(f"anim: {cfg['anim']} (from --anim flag)")
    if args.dot_shape:
        cfg["dot_shape"] = args.dot_shape
        print(f"dot_shape: {args.dot_shape} (from --dot-shape flag)")

    if args.from_npy:
        grids = {}
        for theme in ("dark", "light"):
            p = os.path.join(ROOT, f"portrait_{theme}.npy")
            if not os.path.exists(p):
                raise SystemExit(
                    f"--from-npy: missing {p} (run without --from-npy once, "
                    f"or export from tools/dot-editor)")
            grids[theme] = _load_npy_grid(p)
            print(f"from-npy: loaded {p}")
    else:
        src = args.src
        if src is None:
            # banner-cutout.png (alpha-masked photo) is what the committed
            # grids were built from; banner.jpeg needs the background
            # segmentation pass and yields a different portrait
            cutout = os.path.join(ROOT, "logos", "banner-cutout.png")
            plain = os.path.join(ROOT, "logos", "banner.jpeg")
            src = cutout if os.path.exists(cutout) else plain
        light, dark, box = build_grids(src)
        # these saves are the point of this mode: the .npy pair at the repo
        # root is the source of truth the dot editor reads, so it must always
        # mirror the photo
        np.save(os.path.join(ROOT, "portrait_light.npy"), light)
        np.save(os.path.join(ROOT, "portrait_dark.npy"), dark)
        print(f"crop (x,y,w,h) = {box}   grid {GRID_W}x{GRID_H}")
        grids = {"dark": dark, "light": light}

    anim = AnimParams(cfg)
    if cfg:
        print(f"config: {cfg_path} -> anim={anim.anim} "
              f"loop_dur={anim.loop_dur}s drift={anim.drift_fraction} "
              f"noise={anim.band_noise} bands={anim.n_bands}")
    else:
        print("anim: drift (defaults — no banner_config.json)")
    if anim.dot_shape != "square":
        print(f"dot_shape: {anim.dot_shape}")

    for theme, ink in grids.items():
        svg, st = build(ink, theme, anim=anim)
        path = f"{outdir}/{theme}.svg"
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        kb = len(svg.encode()) / 1024
        print(f"\n{path}: {kb:.0f}KB")
        print(f"  dots {st['dots']}  runs {st['runs']}  bands {st['bands']}")
        print(f"  evenness         {st['evenness']:.3f}  (~0.05 good, ~0.7 patchy)")
        print(f"  straight-boundary {st['straight']:.3f}  (~0.01 organic, ~0.17 grid)")
        print(f"  transport (grid units/stage) "
              f"{', '.join(f'{v:.1f}' for v in st['transport'])}")


if __name__ == "__main__":
    main()

