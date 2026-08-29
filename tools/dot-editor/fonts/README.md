# fonts/ — bộ font của Dot Editor

Chạy một lần (cần mạng):

```bash
python3 fetch_fonts.py
```

Script tải 3 font từ Google Fonts (subset **vietnamese + latin**) và ghi
`fonts.css` + các file `.woff2` ngay tại đây. Sau đó editor tự dùng font
mới khi refresh — không cần sửa gì thêm.

| Font | Vai trò | Ghi chú |
|---|---|---|
| VT323 | Display/heading | Hỗ trợ dấu tiếng Việt |
| Press Start 2P | Chip trang trí | **Chỉ dùng cho chữ HOA không dấu** (DARK, LIGHT, v2…) — không có glyph tiếng Việt |
| IBM Plex Mono 400/600 | Body/UI | Hỗ trợ dấu tiếng Việt |

Không có font cũng không sao — CSS khai báo fallback `system mono`,
editor vẫn chạy đầy đủ tính năng (kể cả deploy lên GitHub Pages).
