#!/usr/bin/env python3
"""Tải bộ font "Pixel Terminal Brutalism" về thư mục này (chạy tay, cần mạng):

    python3 fetch_fonts.py

Sinh ra các file *.woff2 và fonts.css. Editor vẫn chạy bình thường nếu
chưa tải font (fallback sang system mono) — font chỉ là lớp hương vị.

Bộ font:
  VT323          — display kiểu CRT (có subset tiếng Việt)
  Press Start 2P — chip trang trí, CHỈ dùng cho chuỗi ASCII không dấu
  IBM Plex Mono  — body/UI (400 + 600, có subset tiếng Việt)

Chỉ tải subset vietnamese + latin cho gọn (~110 KB tổng).
"""
import re
import urllib.request
from pathlib import Path

CSS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=IBM+Plex+Mono:wght@400;600"
    "&family=Press+Start+2P"
    "&family=VT323&display=swap"
)
# UA hiện đại để Google trả về định dạng woff2 kèm comment subset
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
WANT_SUBSETS = {"vietnamese", "latin"}


def main():
    here = Path(__file__).resolve().parent
    req = urllib.request.Request(CSS_URL, headers={"User-Agent": UA})
    css = urllib.request.urlopen(req, timeout=30).read().decode()

    out_css, n = [], 0
    # mỗi khối trong CSS của Google: /* subset */ @font-face { ... }
    blocks = re.findall(r"/\*\s*([\w-]+)\s*\*/\s*@font-face\s*\{(.*?)\}", css, re.S)
    for subset, body in blocks:
        if subset not in WANT_SUBSETS:
            continue
        fam = re.search(r"font-family:\s*'([^']+)'", body).group(1)
        weight = re.search(r"font-weight:\s*(\d+)", body).group(1)
        url = re.search(r"url\((https:[^)]+)\)", body).group(1)
        rng_m = re.search(r"unicode-range:\s*([^;]+);", body)
        rng = rng_m.group(1).strip() if rng_m else ""
        fname = f"{fam.replace(' ', '')}-{weight}-{subset}.woff2"
        data = urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30
        ).read()
        (here / fname).write_bytes(data)
        out_css.append(
            "@font-face{font-family:'%s';font-style:normal;font-weight:%s;"
            "font-display:swap;src:url('%s') format('woff2');unicode-range:%s}"
            % (fam, weight, fname, rng)
        )
        n += 1
        print(f"  + {fname} ({len(data) / 1024:.0f} KB)")

    (here / "fonts.css").write_text("\n".join(out_css) + "\n", encoding="utf-8")
    print(f"Xong: {n} file woff2 + fonts.css.")


if __name__ == "__main__":
    main()
