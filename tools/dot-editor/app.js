/* IrusVN Dot Editor — sửa lưới bool 300×340 trong portrait_dark.npy / portrait_light.npy.
   Nguồn sự thật là 2 file .npy này; dark.svg / light.svg được banner.py tái sinh từ chúng. */
"use strict";

// ---------------------------------------------------------------- npy I/O
// Định dạng .npy của numpy: magic "\x93NUMPY", version, header-len, dict Python
// literal (descr '|b1', fortran_order False, shape [340,300]), rồi 102000 byte thô.
function parseNpy(buf) {
  const u8 = new Uint8Array(buf);
  const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // \x93NUMPY
  for (let i = 0; i < 6; i++) if (u8[i] !== magic[i]) throw new Error("Không phải file .npy");
  const major = u8[6];
  let hlen, off;
  if (major <= 1) { hlen = u8[8] | (u8[9] << 8); off = 10; }
  else { hlen = u8[8] + (u8[9] << 8) + (u8[10] << 16) + ((u8[11] << 24) >>> 0); off = 12; }
  let header = "";
  for (let i = 0; i < hlen; i++) header += String.fromCharCode(u8[off + i]);

  const descrM = header.match(/'descr':\s*'([^']+)'/);
  const shapeM = header.match(/'shape':\s*\(([^)]*)\)/);
  const fortM = header.match(/'fortran_order':\s*(True|False)/);
  if (!descrM || !shapeM) throw new Error("Header .npy không đọc được");
  const descr = descrM[1];
  if (descr !== "|b1") throw new Error(`Chỉ hỗ trợ mảng bool ('|b1'), nhận được '${descr}'`);
  const shape = shapeM[1].split(",").map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n));
  if (shape.length !== 2) throw new Error("Cần mảng 2 chiều");
  // numpy ghi hàng-chính (C order) khi fortran_order False; nếu True thì de-chuyển vị
  const dataOff = off + hlen;
  const H = shape[0], W = shape[1];
  let grid = new Uint8Array(H * W);
  grid.set(u8.subarray(dataOff, dataOff + H * W));
  if (fortM && fortM[1] === "True") {
    const t = new Uint8Array(H * W);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) t[y * W + x] = grid[x * H + y];
    grid = t;
  }
  return { grid, H, W };
}

function buildNpy(grid, H, W) {
  const head = `{'descr': '|b1', 'fortran_order': False, 'shape': (${H}, ${W}), }`;
  // numpy căn header tới biên 64 byte: 6 magic + 2 version + 2 hlen + nội dung + \n
  let hl = head.length + 1;
  const padding = (64 - (10 + hl) % 64) % 64;
  hl += padding;
  const hdr = "\x93NUMPY\x01\x00" +
    String.fromCharCode(hl & 0xff, (hl >> 8) & 0xff) +
    head + " ".repeat(padding) + "\n";
  const out = new Uint8Array(10 + hl + H * W);
  for (let i = 0; i < hdr.length; i++) out[i] = hdr.charCodeAt(i);
  out.set(grid, 10 + hl);
  return out;
}

// ---------------------------------------------------------------- state
const GW = 300, GH = 340;

const THEMES = {
  dark: {
    ink: "#a78bfa", bg: "#0a101f", canvasBg: "#070b14",
    label: "DARK", countEl: "cntDark", ph: "phDark",
    cv: "cvDark", ov: "ovDark", vp: "vpDark", stack: "stackDark",
    loadBtn: "loadDark", revertBtn: "revertDark", pw: "pwDark",
  },
  light: {
    ink: "#7c3aed", bg: "#f8fafc", canvasBg: "#e9edf3",
    label: "LIGHT", countEl: "cntLight", ph: "phLight",
    cv: "cvLight", ov: "ovLight", vp: "vpLight", stack: "stackLight",
    loadBtn: "loadLight", revertBtn: "revertLight", pw: "pwLight",
  },
};

function makePanel(theme) {
  const T = THEMES[theme];
  return {
    theme, T,
    loaded: false, dirty: false,
    H: GH, W: GW,
    grid: null,        // Uint8Array H*W, 1 = dot
    original: null,    // bản copy lúc tải, cho revert
    undoStack: [], redoStack: [],
    zoom: 3, panX: 12, panY: 12,
    // Các điều chỉnh hiển thị độc lập theo từng theme.
    densityBase: null, // lưới tại 100%; không bị autosave thay thế
    originalDensityBase: null,
    dotDensity: 100,
    originalDotDensity: 100,
    dotSize: 100,
    originalDotSize: 100,
  };
}
const panels = { dark: makePanel("dark"), light: makePanel("light") };

let tool = "erase";       // brush | erase | flood
let brushSize = 0;        // 0 | 1 | 2
let syncMode = false;
let showGrid = false;

// Hình dạng dot khi render (preview lẫn banner.py sinh SVG). "square" là
// bản gốc; các hình khác chỉ đổi lớp vẽ — dữ liệu lưới không đổi.
const DOT_SHAPES = ["square", "circle", "diamond", "plus"];
let dotShape = "square";

const $ = id => document.getElementById(id);

// ---------------------------------------------------------------- rendering
function renderZoom(p) {
  // Fit có thể đưa mỗi ô về 1 CSS pixel. Canvas nền giữ tối thiểu ba pixel
  // nội bộ/ô để thay đổi kích thước dot vẫn nhìn thấy rõ khi được thu nhỏ.
  return Math.max(3, p.zoom);
}

function layoutPanel(p) {
  const cv = $(p.T.cv), ov = $(p.T.ov), stack = $(p.T.stack);
  const w = p.W * p.zoom, h = p.H * p.zoom;
  const rw = p.W * renderZoom(p), rh = p.H * renderZoom(p);

  cv.width = rw; cv.height = rh;
  cv.style.width = w + "px"; cv.style.height = h + "px";
  ov.width = w; ov.height = h;
  ov.style.width = w + "px"; ov.style.height = h + "px";
  stack.style.width = w + "px";
  stack.style.height = h + "px";
}

// Sprite theo hình dạng/kích thước đã chọn. Tile cho pattern giữ đúng cỡ ô;
// sprite tự do có thể lớn hơn ô để dot >100% thực sự nở ra quanh tâm.
const SPRITE_CACHE = new Map();
function shapeSprite(shape, z, color, scale = 1, tile = false) {
  const key = `${shape}|${z}|${color}|${scale}|${tile}`;
  let c = SPRITE_CACHE.get(key);
  if (c) return c;

  const size = z * scale;
  const canvasSize = tile ? z : Math.ceil(size);
  const center = canvasSize / 2;
  const offset = center - size / 2;
  c = document.createElement("canvas");
  c.width = canvasSize; c.height = canvasSize;
  const g = c.getContext("2d");
  g.fillStyle = color;

  if (shape === "square") {
    g.fillRect(offset, offset, size, size);
  } else if (shape === "circle") {
    g.beginPath();
    g.arc(center, center, size * 0.48, 0, Math.PI * 2);
    g.fill();
  } else if (shape === "diamond") {
    g.beginPath();
    g.moveTo(center, offset); g.lineTo(offset + size, center);
    g.lineTo(center, offset + size); g.lineTo(offset, center);
    g.closePath(); g.fill();
  } else if (shape === "plus") {
    const arm = Math.max(1, size * 0.25);
    g.fillRect(center - arm / 2, offset, arm, size);
    g.fillRect(offset, center - arm / 2, size, arm);
  }

  SPRITE_CACHE.set(key, c);
  return c;
}

function shapePattern(ctx, shape, z, color, scale = 1) {
  return ctx.createPattern(shapeSprite(shape, z, color, scale, true), "repeat");
}

function drawOversizedDots(ctx, p) {
  const z = renderZoom(p);
  const sprite = shapeSprite(dotShape, z, p.T.ink, p.dotSize / 100);
  const offset = (z - sprite.width) / 2;
  ctx.imageSmoothingEnabled = false;
  for (let y = 0; y < p.H; y++) {
    const row = y * p.W;
    for (let x = 0; x < p.W; x++) {
      if (p.grid[row + x]) ctx.drawImage(sprite, x * z + offset, y * z + offset);
    }
  }
}

// Canvas offscreen tái sử dụng theo khoá — tránh cấp phát mỗi lần vẽ lại
const SCRATCH_POOL = new Map();
function scratch(key, w, h) {
  let c = SCRATCH_POOL.get(key);
  if (!c) { c = document.createElement("canvas"); SCRATCH_POOL.set(key, c); }
  if (c.width !== w) c.width = w;
  if (c.height !== h) c.height = h;
  return c;
}

function renderBase(p) {
  const cv = $(p.T.cv), ctx = cv.getContext("2d");
  ctx.fillStyle = p.T.bg;
  ctx.fillRect(0, 0, cv.width, cv.height);
  if (!p.loaded) return;

  if (p.dotSize > 100) {
    drawOversizedDots(ctx, p);
    return;
  }

  const mask = scratch(`mask:${p.theme}`, p.W, p.H);
  const mctx = mask.getContext("2d");
  const image = mctx.createImageData(p.W, p.H);
  for (let i = 0; i < p.grid.length; i++) image.data[i * 4 + 3] = p.grid[i] ? 255 : 0;
  mctx.putImageData(image, 0, 0);

  const layer = scratch(`dots:${p.theme}`, cv.width, cv.height);
  const lctx = layer.getContext("2d");
  lctx.clearRect(0, 0, layer.width, layer.height);
  lctx.fillStyle = shapePattern(lctx, dotShape, renderZoom(p), p.T.ink, p.dotSize / 100);
  lctx.fillRect(0, 0, layer.width, layer.height);
  lctx.globalCompositeOperation = "destination-in";
  lctx.imageSmoothingEnabled = false;
  lctx.drawImage(mask, 0, 0, layer.width, layer.height);
  lctx.globalCompositeOperation = "source-over";
  ctx.drawImage(layer, 0, 0);
}

function renderGridOverlay(p) {
  // vẽ lưới tham chiếu mỗi 10 ô lên chính overlay (cùng hover cursor)
  const ov = $(p.T.ov), ctx = ov.getContext("2d");
  ctx.clearRect(0, 0, ov.width, ov.height);
  if (!p.loaded || !showGrid) return;
  ctx.strokeStyle = "rgba(128,140,170,0.25)";
  ctx.lineWidth = 1;
  const z = p.zoom;
  ctx.beginPath();
  for (let x = 0; x <= p.W; x += 10) { ctx.moveTo(x * z + .5, 0); ctx.lineTo(x * z + .5, ov.height); }
  for (let y = 0; y <= p.H; y += 10) { ctx.moveTo(0, y * z + .5); ctx.lineTo(ov.width, y * z + .5); }
  ctx.stroke();
}

function updateCount(p) {
  const el = $(p.T.countEl);
  if (!p.loaded) { el.textContent = "— chưa tải"; return; }
  let n = 0;
  for (let i = 0; i < p.grid.length; i++) n += p.grid[i];
  el.textContent = `${n.toLocaleString("vi")} dots` + (p.dirty ? " ●" : "");
}

function refresh(p, { base = true } = {}) {
  if (base) renderBase(p);
  renderGridOverlay(p);
  updateCount(p);
  if (p === activePanel) updateDensitySizeLabels();
  $("btnExport").disabled = !(panels.dark.loaded || panels.light.loaded);
  $("btnSaveProj").disabled = !(panels.dark.loaded || panels.light.loaded);
  scheduleAutosave();
  $(p.T.revertBtn).disabled = !p.dirty;
  $("dirty").style.display =
    (panels.dark.dirty || panels.light.dirty) ? "inline" : "none";
}

// ---------------------------------------------------------------- undo/redo
function panelSnapshot(p) {
  return {
    grid: p.grid.slice(),
    densityBase: p.densityBase.slice(),
    dotDensity: p.dotDensity,
    dotSize: p.dotSize,
  };
}

function restoreSnapshot(p, snapshot) {
  p.grid = snapshot.grid.slice();
  p.densityBase = snapshot.densityBase.slice();
  p.dotDensity = snapshot.dotDensity;
  p.dotSize = snapshot.dotSize;
}

function pushUndo(p) {
  p.undoStack.push(panelSnapshot(p));
  if (p.undoStack.length > 60) p.undoStack.shift();
  p.redoStack.length = 0;
}

function applyEdit(target, fn) {
  const list = syncMode ? ["dark", "light"].map(k => panels[k]).filter(p => p.loaded)
                        : [target].filter(p => p.loaded);
  for (const p of list) {
    pushUndo(p);
    fn(p);
    p.densityBase = p.grid.slice();
    p.dotDensity = 100;
    p.dirty = true;
    refresh(p);
  }
}

function doUndo(p) {
  if (!p.undoStack.length) return;
  p.redoStack.push(panelSnapshot(p));
  restoreSnapshot(p, p.undoStack.pop());
  p.dirty = true;
  refresh(p);
}
function doRedo(p) {
  if (!p.redoStack.length) return;
  p.undoStack.push(panelSnapshot(p));
  restoreSnapshot(p, p.redoStack.pop());
  p.dirty = true;
  refresh(p);
}

// ---------------------------------------------------------------- tools
function brushMask(size) {
  // trả về danh sách offset [dx,dy] tương ứng cỡ 1/3/5
  if (size === 0) return [[0, 0]];
  if (size === 1) {
    const o = [];
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) o.push([dx, dy]);
    return o;
  }
  const o = [];
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++)
    if (dx * dx + dy * dy <= 6.5) o.push([dx, dy]);
  return o;
}

function paintAt(p, gx, gy, val) {
  const mask = brushMask(brushSize);
  for (const [dx, dy] of mask) {
    const x = gx + dx, y = gy + dy;
    if (x >= 0 && y >= 0 && x < p.W && y < p.H) p.grid[y * p.W + x] = val;
  }
}

function floodComponent(p, sx, sy) {
  // BFS 4-hướng: trả về mọi ô dot liền nhau tính từ (sx,sy); ô rỗng thì rỗng
  const idx0 = sy * p.W + sx;
  if (!p.grid[idx0]) return [];
  const seen = new Set([idx0]);
  const q = [[sx, sy]];
  const out = [];
  while (q.length) {
    const [x, y] = q.pop();
    out.push(y * p.W + x);
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= p.W || ny >= p.H) continue;
      const ni = ny * p.W + nx;
      if (p.grid[ni] && !seen.has(ni)) { seen.add(ni); q.push([nx, ny]); }
    }
  }
  return out;
}

function floodErase(p, gx, gy) {
  const comp = floodComponent(p, gx, gy);
  if (!comp.length) return;
  for (const i of comp) p.grid[i] = 0;
}

function speckClean(p, minSize) {
  // xoá mọi connected component có số dot < minSize
  const visited = new Uint8Array(p.W * p.H);
  let removed = 0, clusters = 0;
  for (let y = 0; y < p.H; y++) {
    for (let x = 0; x < p.W; x++) {
      const i = y * p.W + x;
      if (!p.grid[i] || visited[i]) continue;
      const comp = floodComponent(p, x, y);
      for (const ci of comp) visited[ci] = 1;
      if (comp.length < minSize) {
        for (const ci of comp) p.grid[ci] = 0;
        removed += comp.length; clusters++;
      }
    }
  }
  return { removed, clusters };
}

// ---------------------------------------------------------------- pointer
function cellFromEvent(e, p) {
  const cv = $(p.T.cv);
  const r = cv.getBoundingClientRect();
  const x = Math.floor((e.clientX - r.left) / p.zoom);
  const y = Math.floor((e.clientY - r.top) / p.zoom);
  if (x < 0 || y < 0 || x >= p.W || y >= p.H) return null;
  return [x, y];
}

// Vẽ một đoạn thẳng nối liền; undo chỉ push 1 lần cho cả nét (strokeTargets
// giữ các panel đã push của nét hiện tại, refresh khi nhả chuột).
let strokeTargets = null;

function beginStroke(target) {
  strokeTargets = syncMode ? ["dark", "light"].map(k => panels[k]).filter(q => q.loaded)
                           : [target].filter(q => q.loaded);
  for (const p of strokeTargets) pushUndo(p);
}

function paintLine(targets, lx, ly, cx2, cy2, val) {
  const steps = Math.max(Math.abs(cx2 - lx), Math.abs(cy2 - ly), 1);
  for (let s = 1; s <= steps; s++) {
    const ix = Math.round(lx + (cx2 - lx) * s / steps);
    const iy = Math.round(ly + (cy2 - ly) * s / steps);
    for (const p of targets) paintAt(p, ix, iy, val);
  }
  for (const p of targets) { p.dirty = true; renderBase(p); updateCount(p); }
}

function bindPanel(key) {
  const p = panels[key];
  const stack = $(p.T.stack);

  let painting = null;   // val đang vẽ (1/0) hoặc "flood"
  let panning = null;
  let lastCell = null;

  stack.addEventListener("contextmenu", e => e.preventDefault());

  stack.addEventListener("pointerdown", e => {
    if (!p.loaded) return;
    setActivePanel(p);
    try { stack.setPointerCapture(e.pointerId); } catch (err) {
      if (err.name !== "NotFoundError") throw err;
    }
    if (e.button === 1 || spaceDown || (e.button === 0 && e.altKey)) {
      panning = { x: e.clientX, y: e.clientY, px: p.panX, py: p.panY };
      stack.classList.add("panning");
      return;
    }
    const cell = cellFromEvent(e, p);
    if (!cell) return;
    const rightErase = e.button === 2;
    if (tool === "flood" && !rightErase) {
      painting = "flood";
      applyEdit(p, pp => floodErase(pp, cell[0], cell[1]));
    } else {
      const val = (rightErase || tool === "erase") ? 0 : 1;
      painting = val;
      beginStroke(p);
      paintLine(strokeTargets, cell[0], cell[1], cell[0], cell[1], val);
    }
    lastCell = cell;
    e.preventDefault();
  });

  stack.addEventListener("pointermove", e => {
    const cell = cellFromEvent(e, p);
    $("coords").textContent = cell ? `(${cell[0]}, ${cell[1]})` : "";
    drawHover(p, e);
    if (panning) {
      p.panX = panning.px + e.clientX - panning.x;
      p.panY = panning.py + e.clientY - panning.y;
      positionStack(p);
      return;
    }
    if (!painting || !cell || painting === "flood") return;
    // nối liền các ô giữa lần gọi trước để nét không đứt quãng
    const [lx, ly] = lastCell || cell;
    const steps = Math.max(Math.abs(cell[0] - lx), Math.abs(cell[1] - ly));
    const targets = syncMode ? ["dark", "light"].map(k => panels[k]).filter(q => q.loaded) : [p];
    paintLine(targets, lx, ly, cell[0], cell[1], painting);
    lastCell = cell;
  });

  window.addEventListener("pointerup", () => {
    if (strokeTargets) {
      for (const pp of strokeTargets) {
        pp.densityBase = pp.grid.slice();
        pp.dotDensity = 100;
        refresh(pp);
      }
      strokeTargets = null;
    }
    painting = null; panning = null;
    stack.classList.remove("panning");
  });

  stack.addEventListener("pointerleave", () => { $("coords").textContent = ""; });

  stack.addEventListener("wheel", e => {
    if (!p.loaded) return;
    e.preventDefault();
    const vp = $(p.T.vp);
    if (e.ctrlKey || e.metaKey) {
      zoomAt(p, e.clientX, e.clientY, e.deltaY < 0 ? 1.25 : 0.8);
    } else {
      vp.scrollLeft += e.deltaX;
      vp.scrollTop += e.deltaY;
    }
  }, { passive: false });

  $(p.T.loadBtn).addEventListener("click", () => pickFile(key));
  $(p.T.revertBtn).addEventListener("click", () => {
    if (!p.original) return;
    pushUndo(p);
    p.grid = p.original.slice();
    p.densityBase = p.originalDensityBase.slice();
    p.dotDensity = p.originalDotDensity;
    p.dotSize = p.originalDotSize;
    p.dirty = false;
    updateDensitySizeLabels();
    refresh(p);
  });
}

function drawHover(p, e) {
  const ov = $(p.T.ov), ctx = ov.getContext("2d");
  renderGridOverlay(p);
  if (!p.loaded || !e) return;
  const cell = cellFromEvent(e, p);
  if (!cell) return;
  const z = p.zoom;
  const drawZ = renderZoom(p);
  const size = tool === "flood" ? 0 : brushSize;
  ctx.strokeStyle = tool === "brush" && !syncMode ? "#67e8f9"
                  : tool === "brush" ? "#fbbf24" : "#f87171";
  ctx.lineWidth = 1.5;
  if (size === 0) {
    ctx.strokeRect(cell[0] * z - 1, cell[1] * z - 1, z + 2, z + 2);
  } else {
    const half = size === 1 ? 1.5 : 2.5;
    ctx.strokeRect((cell[0] - half) * z - 1, (cell[1] - half) * z - 1,
                   (half * 2 + 1) * z + 2, (half * 2 + 1) * z + 2);
  }
  // brush: hé trước hình dot sắp tô bằng sprite mờ đúng hình đang chọn
  if (tool === "brush" && dotShape !== "square") {
    const sp = shapeSprite(dotShape, drawZ, syncMode ? "#fbbf24" : "#67e8f9", p.dotSize / 100, true);
    ctx.save();
    ctx.globalAlpha = 0.55;
    for (const [dx, dy] of brushMask(brushSize))
      ctx.drawImage(sp, (cell[0] + dx) * z, (cell[1] + dy) * z, z, z);
    ctx.restore();
  }
}

// ---------------------------------------------------------------- zoom/pan
function positionStack(p) {
  $(p.T.stack).style.transform = `translate(${p.panX}px, ${p.panY}px)`;
}

function setZoom(p, z, cx, cy) {
  z = Math.max(1, Math.min(12, Math.round(z))); // 12x: canvas 3600x4080 — an toàn cho Safari
  if (z === p.zoom) return;
  const vp = $(p.T.vp);
  if (cx != null) {
    const r = vp.getBoundingClientRect();
    const vx = cx - r.left + vp.scrollLeft - p.panX;
    const vy = cy - r.top + vp.scrollTop - p.panY;
    const k = z / p.zoom;
    p.zoom = z;
    layoutPanel(p); renderBase(p); renderGridOverlay(p);
    p.panX = Math.round(cx - r.left - vx * k);
    p.panY = Math.round(cy - r.top - vy * k);
    positionStack(p);
  } else {
    p.zoom = z;
    layoutPanel(p); renderBase(p); renderGridOverlay(p);
    positionStack(p);
  }
  $("zoomLabel").textContent = z + "×";
  syncZoomAcrossPanels(p);
}

function syncZoomAcrossPanels(src) {
  for (const key of ["dark", "light"]) {
    const q = panels[key];
    if (q === src || !q.loaded) continue;
    q.zoom = src.zoom;
    layoutPanel(q); renderBase(q); renderGridOverlay(q); positionStack(q);
  }
}

function zoomAt(p, cx, cy, factor) {
  setZoom(p, p.zoom * factor, cx, cy);
}

function fitZoom() {
  for (const key of ["dark", "light"]) {
    const p = panels[key];
    const vp = $(p.T.vp);
    const z = Math.max(1, Math.min(12, Math.floor(Math.min(
      (vp.clientWidth - 24) / p.W, (vp.clientHeight - 24) / p.H))));
    p.zoom = z; p.panX = 12; p.panY = 12;
    layoutPanel(p); renderBase(p); renderGridOverlay(p); positionStack(p);
  }
  $("zoomLabel").textContent = panels.dark.zoom + "×";
}

// ---------------------------------------------------------------- loading
function pickFile(preferTheme) {
  const inp = $("fileAll");
  inp.dataset.prefer = preferTheme || "";
  inp.value = "";
  inp.click();
}

async function loadFiles(files) {
  for (const f of files) {
    const name = f.name.toLowerCase();
    const key = name.includes("dark") ? "dark"
              : name.includes("light") ? "light"
              : ($("fileAll").dataset.prefer || "dark");
    try {
      const { grid, H, W } = parseNpy(await f.arrayBuffer());
      const p = panels[key];
      p.grid = grid; p.H = H; p.W = W;
      p.original = grid.slice();
      p.densityBase = grid.slice();
      p.originalDensityBase = grid.slice();
      p.dotDensity = p.originalDotDensity = 100;
      p.dotSize = p.originalDotSize = 100;
      p.loaded = true;
      p.dirty = false; p.undoStack.length = 0; p.redoStack.length = 0;
      $(p.T.ph).style.display = "none";
      p.zoom = 3; p.panX = 12; p.panY = 12;
      layoutPanel(p); refresh(p); positionStack(p);
      setActivePanel(p);
      $("hint").textContent = `Đã nạp ${f.name} → ${key} (${W}×${H})`;
    } catch (err) {
      alert(`Lỗi nạp ${f.name}: ${err.message}`);
    }
  }
  fitZoom();
}

$("fileAll").addEventListener("change", e => loadFiles([...e.target.files]));

// drag & drop toàn trang
let dragDepth = 0;
window.addEventListener("dragenter", e => { e.preventDefault(); dragDepth++; document.body.classList.add("dragging"); });
window.addEventListener("dragleave", e => { e.preventDefault(); if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove("dragging"); } });
window.addEventListener("dragover", e => e.preventDefault());
window.addEventListener("drop", e => {
  e.preventDefault();
  dragDepth = 0; document.body.classList.remove("dragging");
  const files = [...(e.dataTransfer?.files || [])];
  if (!files.length) return;
  // file .json → coi là project; còn lại là .npy
  const proj = files.filter(f => f.name.toLowerCase().endsWith(".json"));
  if (proj.length) openProjectFiles(proj);
  const npys = files.filter(f => !f.name.toLowerCase().endsWith(".json"));
  if (npys.length) loadFiles(npys);
});

// ---------------------------------------------------------------- export
// Pháo giấy pixel khi export thành công (R4) — tự dọn sau khi chạy xong.
function confettiBurst() {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const colors = ["#22D3EE", "#A78BFA", "#FFD23F", "#FF5D5D", "#B8F135"];
  for (let i = 0; i < 28; i++) {
    const bit = document.createElement("span");
    bit.textContent = "▪";
    bit.style.cssText =
      `position:fixed;z-index:999;pointer-events:none;` +
      `font-size:${8 + Math.random() * 10}px;color:${colors[i % colors.length]};` +
      `left:50%;top:38%;`;
    document.body.appendChild(bit);
    const dx = (Math.random() - 0.5) * 560;
    const dy = -140 + Math.random() * 420;
    bit.animate(
      [
        { transform: "translate(0,0) rotate(0deg)", opacity: 1 },
        { transform: `translate(${dx}px, ${dy}px) rotate(${(Math.random() - 0.5) * 720}deg)`, opacity: 0 },
      ],
      { duration: 900 + Math.random() * 700, easing: "cubic-bezier(.2,.7,.3,1)" }
    ).onfinish = () => bit.remove();
  }
}

// Mỗi file một hộp thoại lưu riêng qua File System Access API, với tên gợi ý
// đúng cho từng file và có thể lưu thẳng vào thư mục repo. Không dùng hai
// download <a> liền kề: trên Chrome/macOS, download thứ hai bị xếp hàng sau
// hộp thoại lưu đang mở của file đầu và hộp thoại của nó nhận lại tên file
// thứ nhất — dữ liệu light bị ghi đè lên portrait_dark.npy.
async function saveGridFile(bytes, suggestedName) {
  if (typeof window.showSaveFilePicker === "function") {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName,
        types: [{ description: "NumPy bool grid", accept: { "application/octet-stream": [".npy"] } }],
      });
      const w = await handle.createWritable();
      await w.write(bytes);
      await w.close();
      return true;
    } catch (err) {
      if (err && err.name === "AbortError") return "cancelled"; // người dùng huỷ
      // lỗi khác (mất user activation, context không hợp lệ…) → dùng anchor
    }
  }
  downloadBlob(new Blob([bytes], { type: "application/octet-stream" }), suggestedName);
  return true;
}

$("btnExport").addEventListener("click", async () => {
  const exported = [];
  for (const key of ["dark", "light"]) {
    const p = panels[key];
    if (!p.loaded) continue;
    const saved = await saveGridFile(buildNpy(p.grid, p.H, p.W), `portrait_${key}.npy`);
    if (saved === "cancelled") break; // huỷ hộp thoại đầu = huỷ cả export
    if (saved) exported.push(key);
  }
  if (exported.length > 0) confettiBurst();
  const fileNames = exported.map((k) => `portrait_${k}.npy`).join(" / ");
  $("hint").textContent = exported.length
    ? `Đã lưu ${fileNames} → chạy: python3 .github/scripts/banner.py --from-npy .`
    : "Đã huỷ export — chưa lưu file nào.";
});

// ---------------------------------------------------------------- project
// File project gộp cả hai lưới + cấu hình animation thành 1 JSON để lưu/mở.
// Base64 cho dữ liệu lưới (102000 byte/lưới) để file gọn và đọc lại được.

function u8ToB64(u8) {
  let s = "";
  for (let i = 0; i < u8.length; i += 0x8000)
    s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
}
function b64ToU8(b64) {
  const s = atob(b64), u8 = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u8[i] = s.charCodeAt(i);
  return u8;
}

function buildProjectJSON() {
  const grids = {};
  for (const key of ["dark", "light"]) {
    const p = panels[key];
    if (!p.loaded) continue;
    grids[key] = {
      w: p.W, h: p.H, data: u8ToB64(p.grid),
      densityBase: u8ToB64(p.densityBase || p.grid),
      dotDensity: p.dotDensity,
      dotSize: p.dotSize,
    };
  }
  if (!Object.keys(grids).length) throw new Error("Chưa có lưới nào được nạp");
  return {
    format: "irusvn-dot-project", version: 1,
    saved: new Date().toISOString(),
    anim: { ...animState, shape: dotShape },
    grids,
  };
}

function installGrid(key, spec) {
  // Cài một lưới từ object {w,h,data(base64)} vào panel, không đụng undo.
  const grid = b64ToU8(spec.data);
  if (grid.length !== spec.w * spec.h) throw new Error("Dữ liệu lưới sai kích thước");
  const densityBase = spec.densityBase ? b64ToU8(spec.densityBase) : grid.slice();
  if (densityBase.length !== grid.length) throw new Error("Dữ liệu mật độ sai kích thước");

  const p = panels[key];
  p.grid = grid; p.W = spec.w; p.H = spec.h;
  p.original = grid.slice();
  p.densityBase = densityBase;
  p.originalDensityBase = densityBase.slice();
  p.dotDensity = p.originalDotDensity = clampPercent(spec.dotDensity, 100, 10, 200);
  p.dotSize = p.originalDotSize = clampPercent(spec.dotSize, 100, 50, 150);
  p.loaded = true;
  p.dirty = false;
  p.undoStack.length = 0; p.redoStack.length = 0;
  $(p.T.ph).style.display = "none";
  layoutPanel(p); refresh(p); positionStack(p);
}

function clampPercent(value, fallback, min, max) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(min, Math.min(max, n)) : fallback;
}

function downloadBlob(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 500);
}

$("btnSaveProj").addEventListener("click", () => {
  try {
    const json = JSON.stringify(buildProjectJSON(), null, 1);
    downloadBlob(new Blob([json], { type: "application/json" }), "dot-project.json");
    $("hint").textContent =
      "Đã tải dot-project.json (cả 2 lưới + tham số animation).";
    markSaved();
  } catch (err) {
    alert(err.message);
  }
});

// ---- tự lưu localStorage + khôi phục khi mở lại trang
const LS_KEY = "irusvn-dot-editor-session";

let autosaveTimer = null;
function scheduleAutosave() {
  clearTimeout(autosaveTimer);
  autosaveTimer = setTimeout(saveSessionLS, 800);
}

function saveSessionLS() {
  try {
    if (!(panels.dark.loaded || panels.light.loaded)) return;
    localStorage.setItem(LS_KEY, JSON.stringify(buildProjectJSON()));
    markSaved();
  } catch { /* localStorage đầy hoặc bị chặn — bỏ qua */ }
}

function markSaved() {
  for (const key of ["dark", "light"]) {
    const p = panels[key];
    if (!p.dirty || !p.loaded) continue;
    // đã ghi bản hiện tại vào nơi bền vững → mốc "chưa lưu" mới là từ giờ
    p.original = p.grid.slice();
    p.originalDensityBase = p.densityBase.slice();
    p.originalDotDensity = p.dotDensity;
    p.originalDotSize = p.dotSize;
    p.dirty = false;
    updateCount(p);
    $(p.T.revertBtn).disabled = true;
  }
  $("dirty").style.display = "none";
}

function restoreSession() {
  let raw = null;
  try { raw = localStorage.getItem(LS_KEY); } catch { return false; }
  if (!raw) return false;
  try {
    loadProjectJSON(JSON.parse(raw), "(khôi phục phiên trước)");
    return true;
  } catch (err) {
    console.warn("Không khôi phục được phiên:", err);
    return false;
  }
}

async function loadProjectJSON(obj, label) {
  if (!obj || obj.format !== "irusvn-dot-project")
    throw new Error("Không phải file dot-project của editor này");
  if (obj.anim) {
    // kẹp về đúng miền hợp lệ của banner.py để file sửa tay không phá preview
    const c = (v, lo, hi, def) => {
      const n = parseFloat(v);
      return isNaN(n) ? def : Math.max(lo, Math.min(hi, n));
    };
    const a = obj.anim;
    animState.preset = ["drift", "ripple", "shimmer", "static"].includes(a.preset)
      ? a.preset : "drift";
    animState.loopDur = c(a.loopDur, 4, 60, 13.9);
    animState.drift = c(a.drift, 0, 0.8, 0.42);
    animState.noise = c(a.noise, 0, 40, 4);
    animState.bands = Math.round(c(a.bands, 10, 400, 94));
    if (a.shape) setShape(DOT_SHAPES.includes(a.shape) ? a.shape : "square");
  }
  const order = ["dark", "light"];
  let n = 0;
  for (const key of order) {
    if (obj.grids?.[key]) { installGrid(key, obj.grids[key]); n++; }
  }
  if (!n) throw new Error("File không chứa lưới nào");
  setActivePanel(panels.dark.loaded ? panels.dark : panels.light);
  fitZoom();
  $("hint").textContent =
    `Đã mở ${label}: ${n}/2 lưới · anim=${animState.preset} · loop=${animState.loopDur}s`;
}

async function openProjectFiles(files) {
  for (const f of files) {
    try {
      await loadProjectJSON(JSON.parse(await f.text()), f.name);
    } catch (err) {
      alert(`Lỗi mở ${f.name}: ${err.message}`);
    }
  }
}

$("fileProj").addEventListener("change", e => {
  openProjectFiles([...e.target.files]);
  e.target.value = "";
});

$("btnOpenProj").addEventListener("click", () => $("fileProj").click());

// ---- ☁ Nạp .npy trực tiếp từ repo qua HTTP (khi chạy từ checkout/Pages)
async function fetchRepoGrids() {
  $("hint").textContent = "Đang tải portrait_dark.npy / portrait_light.npy từ repo…";
  let ok = 0;
  for (const key of ["dark", "light"]) {
    try {
      const res = await fetch(`../../portrait_${key}.npy`, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const { grid, H, W } = parseNpy(await res.arrayBuffer());
      installGrid(key, { w: W, h: H, data: u8ToB64(grid) });
      ok++;
    } catch (err) {
      alert(`Không tải được portrait_${key}.npy: ${err.message}\n` +
            `(phải mở editor qua HTTP server trong repo, không phải file://)`);
      break;
    }
  }
  if (ok) {
    fitZoom();
    $("hint").textContent =
      `Đã nạp ${ok}/2 lưới từ repo. Lưu ý: đây là bản trên đĩa, ` +
      `không phải sửa đổi chưa export của bạn.`;
  } else {
    $("hint").textContent = "Không nạp được lưới nào từ repo.";
  }
}
$("btnFetchRepo").addEventListener("click", fetchRepoGrids);

// ---------------------------------------------------------------- toolbar
function setTool(t) {
  tool = t;
  for (const b of document.querySelectorAll("#tools button"))
    b.classList.toggle("active", b.dataset.tool === t);
  $("hint").textContent =
    t === "brush" ? "Vẽ: thêm dot. Chuột phải vẫn xoá."
  : t === "erase" ? "Xoá: xoá dot. Chuột phải cũng xoá."
  : "Xoá cụm: click một dot sẽ xoá cả vùng liền nhau (nhiễu rời).";
}
document.querySelectorAll("#tools button").forEach(b =>
  b.addEventListener("click", () => setTool(b.dataset.tool)));

function setSize(s) {
  brushSize = s;
  for (const b of document.querySelectorAll("#sizes button"))
    b.classList.toggle("active", +b.dataset.size === s);
}
document.querySelectorAll("#sizes button").forEach(b =>
  b.addEventListener("click", () => setSize(+b.dataset.size)));

// ---- hình dạng dot: đổi chỉ ảnh hưởng lớp vẽ + file config xuất ra
function setShape(shape) {
  if (!DOT_SHAPES.includes(shape)) return;
  dotShape = shape;
  for (const b of document.querySelectorAll("#shapes button"))
    b.classList.toggle("active", b.dataset.shape === shape);
  // vẽ lại cả hai panel vì sprite thay đổi
  renderBase(panels.dark); renderBase(panels.light);
  if (panels.dark.loaded || panels.light.loaded) scheduleAutosave();
}
document.querySelectorAll("#shapes button").forEach(b =>
  b.addEventListener("click", () => setShape(b.dataset.shape)));

$("chkSync").addEventListener("change", e => {
  syncMode = e.target.checked;
  $("hint").textContent = syncMode
    ? "SYNC: mọi thao tác áp lên cả hai lưới (dot cùng toạ độ)."
    : "Sync tắt — mỗi khung chỉnh độc lập.";
});
$("chkGrid").addEventListener("change", e => {
  showGrid = e.target.checked;
  renderGridOverlay(panels.dark); renderGridOverlay(panels.light);
});

$("zIn").addEventListener("click", () => setZoom(panels.dark, panels.dark.zoom + 1));
$("zOut").addEventListener("click", () => setZoom(panels.dark, panels.dark.zoom - 1));
$("zFit").addEventListener("click", fitZoom);

// ---- mật độ và kích thước dot (độc lập cho mỗi panel)
let activePanel = null; // panel nào đang được focus để áp dụng điều chỉnh

function setActivePanel(p) {
  if (!p || !p.loaded) return;
  activePanel = p;
  for (const key of ["dark", "light"])
    $(panels[key].T.pw).classList.toggle("active-panel", panels[key] === p);
  updateDensitySizeLabels();
}

function updateDensitySizeLabels() {
  if (!activePanel) {
    $("densityLabel").textContent = "—";
    $("sizeLabel").textContent = "—";
    return;
  }
  $("densityLabel").textContent = activePanel.dotDensity + "%";
  $("sizeLabel").textContent = activePanel.dotSize + "%";
}

function applyDensityChange(p, delta) {
  if (!p || !p.loaded) return;
  const next = Math.max(10, Math.min(200, p.dotDensity + delta));
  if (next === p.dotDensity) return;
  pushUndo(p);
  p.dotDensity = next;
  applyDensityToGrid(p);
  p.dirty = true;
  updateDensitySizeLabels();
  refresh(p);
}

function applySizeChange(p, delta) {
  if (!p || !p.loaded) return;
  const next = Math.max(50, Math.min(150, p.dotSize + delta));
  if (next === p.dotSize) return;
  pushUndo(p);
  p.dotSize = next;
  p.dirty = true;
  updateDensitySizeLabels();
  refresh(p);
}

function applyDensityToGrid(p) {
  if (!p.densityBase) return;
  const ratio = p.dotDensity / 100;
  p.grid.set(p.densityBase);
  if (ratio === 1) return;

  const rng = seededRandom(p.theme === "dark" ? 12345 : 67890);
  if (ratio < 1) {
    for (let i = 0; i < p.grid.length; i++) {
      if (p.grid[i] && rng() > ratio) p.grid[i] = 0;
    }
    return;
  }

  const sourceDots = countDots(p.densityBase);
  const extraDots = Math.min(p.grid.length - sourceDots,
    Math.round(sourceDots * (ratio - 1)));
  const candidates = [];
  for (let y = 0; y < p.H; y++) {
    for (let x = 0; x < p.W; x++) {
      const i = y * p.W + x;
      if (p.grid[i]) continue;
      const neighbors = countInkNeighbors(p.densityBase, p.W, p.H, x, y);
      if (neighbors) candidates.push({ i, neighbors, tie: rng() });
    }
  }
  candidates.sort((a, b) => b.neighbors - a.neighbors || a.tie - b.tie);
  for (let i = 0; i < Math.min(extraDots, candidates.length); i++) p.grid[candidates[i].i] = 1;
}

function countDots(grid) {
  let total = 0;
  for (const dot of grid) total += dot;
  return total;
}

function countInkNeighbors(grid, width, height, x, y) {
  let total = 0;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      if (!dx && !dy) continue;
      const nx = x + dx, ny = y + dy;
      if (nx >= 0 && ny >= 0 && nx < width && ny < height) total += grid[ny * width + nx];
    }
  }
  return total;
}

function seededRandom(seed) {
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

$("densityDown").addEventListener("click", () => applyDensityChange(activePanel, -10));
$("densityUp").addEventListener("click", () => applyDensityChange(activePanel, 10));
$("sizeDown").addEventListener("click", () => applySizeChange(activePanel, -10));
$("sizeUp").addEventListener("click", () => applySizeChange(activePanel, 10));

// Focus panel khi click phần viewport, kể cả khi click vùng trống ngoài canvas.
for (const key of ["dark", "light"]) {
  const p = panels[key];
  $(p.T.vp).addEventListener("pointerdown", () => setActivePanel(p));
}

$("btnUndo").addEventListener("click", () => { doUndo(panels.dark); doUndo(panels.light); });
$("btnRedo").addEventListener("click", () => { doRedo(panels.dark); doRedo(panels.light); });

$("btnSpeck").addEventListener("click", () => {
  const n = Math.max(1, parseInt($("speckSize").value, 10) || 5);
  const targets = syncMode ? ["dark", "light"] : ["dark", "light"].filter(k => panels[k].loaded);
  const report = [];
  for (const key of targets) {
    const p = panels[key];
    if (!p.loaded) continue;
    pushUndo(p);
    const { removed, clusters } = speckClean(p, n);
    p.densityBase = p.grid.slice();
    p.dotDensity = 100;
    p.dirty = true;
    refresh(p);
    report.push(`${key}: -${removed} dots / ${clusters} cụm`);
  }
  $("hint").textContent = report.length
    ? "Dọn nhiễu — " + report.join(" · ")
    : "Chưa tải lưới nào.";
});

$("btnLoadAll").addEventListener("click", () => pickFile(""));

// ---------------------------------------------------------------- keyboard
let spaceDown = false;
window.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.code === "Space") { spaceDown = true; e.preventDefault(); }
  const k = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && k === "z") {
    e.preventDefault();
    if (e.shiftKey) { doRedo(panels.dark); doRedo(panels.light); }
    else { doUndo(panels.dark); doUndo(panels.light); }
    return;
  }
  if ((e.ctrlKey || e.metaKey) && k === "y") {
    e.preventDefault(); doRedo(panels.dark); doRedo(panels.light); return;
  }
  if (e.ctrlKey || e.metaKey) return;
  if (k === "b") setTool("brush");
  else if (k === "e") setTool("erase");
  else if (k === "f") setTool("flood");
  else if (k === "1") setSize(0);
  else if (k === "2") setSize(1);
  else if (k === "3") setSize(2);
  else if (k === "g") { $("chkGrid").checked = !$("chkGrid").checked; $("chkGrid").dispatchEvent(new Event("change")); }
  else if (k === "+" || k === "=") setZoom(panels.dark, panels.dark.zoom + 1);
  else if (k === "-") setZoom(panels.dark, panels.dark.zoom - 1);
  else if (k === "0") fitZoom();
});
window.addEventListener("keyup", e => { if (e.code === "Space") spaceDown = false; });

// ---------------------------------------------------------------- animation
// Mô phỏng lớp loop của banner trên canvas: mỗi band là một nhóm dot dịch
// chuyển/twinkle theo preset. Đây chỉ là preview nhanh (band hóa xấp xỉ bằng
// cách chia theo khoảng cách tới tâm, có noise giống banner.py), không thay
// thế việc render SVG thật.
const animState = { preset: "drift", loopDur: 13.9, drift: 0.42, noise: 4, bands: 94 };

const ANIM_HINTS = {
  drift: "Band tan rồi trôi về phía centroid của logo </> rồi quay về.",
  ripple: "Band lan tỏa ra ngoài từ trọng tâm chân dung.",
  shimmer: "Chân dung đứng yên nhưng từng band lấp lánh lệch pha nhau.",
  static: "Chân dung đứng yên hoàn toàn; chỉ còn fade sang logo.",
};

function bandify(grid, W, H, nBands, noise) {
  // Chia các run ngang thành n nhóm theo khoảng-cách+noise tới (cx,cy),
  // giống drift_bands() trong banner.py. Trả về [{runs:[{x,y,n}], dx, dy}].
  const runs = [];
  for (let y = 0; y < H; y++) {
    let x = 0;
    while (x < W) {
      if (!grid[y * W + x]) { x++; continue; }
      let n = 0;
      while (x + n < W && grid[y * W + x + n]) n++;
      runs.push({ x, y, n });
      x += n;
    }
  }
  // trọng tâm mực để ripple
  let sx = 0, sy = 0, tot = 0;
  for (const r of runs) { sx += (r.x + r.n / 2) * r.n; sy += r.y * r.n; tot += r.n; }
  const cx = sx / Math.max(tot, 1), cy = sy / Math.max(tot, 1);

  // PRNG seed cố định (mulberry32) cho band ổn định giữa các lần mở dialog
  let s = 13 >>> 0;
  const rnd = () => {
    s |= 0; s = s + 0x6D2B79F5 | 0;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };

  const pts = runs.map(r => {
    const d = Math.hypot(r.x + r.n / 2 - cx, r.y - cy);
    return { r, key: d + (rnd() - 0.5) * 2 * noise };
  });
  pts.sort((a, b) => a.key - b.key);
  const per = Math.ceil(pts.length / nBands);
  const bands = [];
  for (let i = 0; i < pts.length; i += per) {
    const chunk = pts.slice(i, i + per);
    let bx = 0, by = 0;
    for (const q of chunk) { bx += q.r.x + q.r.n / 2; by += q.r.y; }
    bands.push({ runs: chunk.map(q => q.r), cx: bx / chunk.length, cy: by / chunk.length, i: bands.length });
  }
  return { bands, inkCx: cx, inkCy: cy };
}

function bandOffsets(bandInfo, preset, drift) {
  // dx,dy đích của từng band theo preset — cùng ngữ nghĩa với build() của banner.py:
  // drift hướng về target (centroid logo ≈ giữa khung logo), ripple hướng ra ngoài,
  // shimmer/static đứng yên.
  const tx = 150, ty = 120; // centroid logo </> trong hệ toạ độ lưới
  const out = new Array(bandInfo.bands.length);
  for (let bi = 0; bi < bandInfo.bands.length; bi++) {
    const b = bandInfo.bands[bi];
    let dx = 0, dy = 0;
    if (preset === "drift") {
      dx = (tx - b.cx) * drift; dy = (ty - b.cy) * drift;
    } else if (preset === "ripple") {
      dx = (b.cx - bandInfo.inkCx) * drift;
      dy = (b.cy - bandInfo.inkCy) * drift;
    }
    out[bi] = [dx, dy];
  }
  return out;
}

// Mặt nạ từng band (vị trí đứng yên, không đổi) dựng 1 lần sau khi lưới/preset
// đổi; lớp màu đúng hình được ghép lại mỗi khung qua một canvas offscreen dùng
// chung — nhờ đó preview đạt 60fps mà vẫn đúng hình dot đang chọn.
let bandMasks = null;
const BAND_STAMP_K = 4; // px mỗi ô lúc đóng dấu hình

function bandMask(band) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const r of band.runs) {
    if (r.x < minX) minX = r.x;
    if (r.x + r.n > maxX) maxX = r.x + r.n;
    if (r.y < minY) minY = r.y;
    if (r.y > maxY) maxY = r.y;
  }
  const bw = maxX - minX, bh = maxY - minY;
  const cv = document.createElement("canvas");
  cv.width = bw; cv.height = bh;
  const ctx = cv.getContext("2d");
  const img = ctx.createImageData(bw, bh);
  const d = img.data;
  for (const r of band.runs) {
    const base = ((r.y - minY) * bw + (r.x - minX)) * 4;
    for (let i = 0; i < r.n; i++) d[base + i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return { cv, x: minX, y: minY, w: bw, h: bh };
}

function stampBand(ctx, p, mk, dx, dy) {
  // một canvas cố định cỡ lưới×K dùng chung: đóng dấu band vào vùng của nó
  // rồi chép đúng vùng đó ra — không phải resize canvas từng band
  const K = BAND_STAMP_K;
  const layer = scratch("anim-band", p.W * K, p.H * K);
  const lc = layer.getContext("2d");
  const x0 = mk.x * K, y0 = mk.y * K, w = mk.w * K, h = mk.h * K;
  lc.imageSmoothingEnabled = false;
  lc.globalCompositeOperation = "source-over";
  // xoá vùng này trước: bbox các band có thể chồng nhau, nếu không sẽ hở
  // mực sót của band trước qua khoảng trong suốt của pattern
  lc.clearRect(x0, y0, w, h);
  lc.fillStyle = shapePattern(lc, dotShape, K, p.T.ink, p.dotSize / 100);
  lc.fillRect(x0, y0, w, h);
  // destination-in đục hết vùng ngoài mặt nạ — vô hại vì ta chỉ chép vùng này
  lc.globalCompositeOperation = "destination-in";
  lc.drawImage(mk.cv, x0, y0, w, h);
  lc.globalCompositeOperation = "source-over";
  ctx.drawImage(layer, x0, y0, w, h, mk.x + dx, mk.y + dy, mk.w, mk.h);
}

function drawPreviewFrame(ctx, p, bandInfo, offs, t01) {
  // t01: 0..1 pha trong vòng lặp (khớp KEYTIMES thô của banner.py:
  // portrait giữ đến ~0.194, tan dần, về nhà lúc ~0.906)
  const T = p.T;
  ctx.fillStyle = T.bg === "#f8fafc" ? "#f8fafc" : "#070b14";
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  // đúng hình học lớp dot của banner.py: translate(50,86) scale(1.24,1.4471)
  const ox = 50, oy = 86;

  // pha: 0–0.194 đứng, 0.194–0.432 tan ra và về, 0.432–0.906 ẩn, 0.906–1 về
  let move = 0, alpha = 1;
  if (t01 < 0.194) { move = 0; alpha = 1; }
  else if (t01 < 0.288) { const k = (t01 - 0.194) / 0.094; move = k * k * (3 - 2 * k); alpha = 1; }
  else if (t01 < 0.432) { const k = (t01 - 0.288) / 0.144; move = k; alpha = 1 - k; }
  else if (t01 < 0.906) { move = 1; alpha = 0; }
  else { const k = (t01 - 0.906) / 0.094; move = 1 - k; alpha = k; }

  const twinkle = animState.preset === "shimmer";
  // mặt nạ band dựng 1 lần; chỉ cần khi hình ≠ vuông
  if (!bandMasks && dotShape !== "square" && p.dotSize <= 100) {
    bandMasks = bandInfo.bands.map(bandMask);
  }
  ctx.save();
  ctx.translate(ox, oy); ctx.scale(1.24, 1.4471);
  for (let bi = 0; bi < bandInfo.bands.length; bi++) {
    const b = bandInfo.bands[bi];
    const [dx0, dy0] = offs[bi];
    let a = alpha, dx = dx0 * move, dy = dy0 * move;
    if (twinkle && a > 0.5) {
      const lo = 0.55 + 0.15 * ((bi * 7) % 3);
      const dur = 2.6 + (bi % 5) * 0.7;
      const w = Math.sin(((t01 * animState.loopDur - (bi % 8) / 8) / dur) * 2 * Math.PI);
      a *= 1 - (1 - lo) * (0.5 + 0.5 * w);
    }
    ctx.globalAlpha = a;
    const mk = bandMasks && bandMasks[bi];
    if (mk) { stampBand(ctx, p, mk, dx, dy); continue; }
    const sprite = shapeSprite(dotShape, BAND_STAMP_K, T.ink, p.dotSize / 100);
    const offset = (BAND_STAMP_K - sprite.width) / (2 * BAND_STAMP_K);
    const spriteSize = sprite.width / BAND_STAMP_K;
    for (const r of b.runs) {
      for (let x = 0; x < r.n; x++)
        ctx.drawImage(sprite, r.x + x + dx + offset, r.y + dy + offset, spriteSize, spriteSize);
    }
  }
  ctx.restore();
}

let previewRAF = null;

function openAnimDialog() {
  const dlg = $("animDlg");
  $("inLoopDur").value = animState.loopDur;
  $("inDrift").value = animState.drift;
  $("inNoise").value = animState.noise;
  $("inBands").value = animState.bands;
  $("animSel").value = animState.preset;
  if (!dlg.open) { $("animStatus").textContent = ""; dlg.showModal(); }

  const cv = $("animCv"), ctx = cv.getContext("2d");
  const srcP = panels.dark.loaded ? panels.dark : panels.light;
  if (!srcP.loaded) {
    ctx.fillStyle = "#070b14"; ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = "#8a94aa"; ctx.font = "16px monospace"; ctx.textAlign = "center";
    ctx.fillText("Nạp .npy trước đã", cv.width / 2, cv.height / 2);
    return;
  }
  const bandInfo = bandify(srcP.grid, srcP.W, srcP.H, animState.bands, animState.noise);
  bandMasks = null; // lưới/preset/hình đổi → dựng lại mặt nạ
  const offs = bandOffsets(bandInfo, animState.preset, animState.drift);
  const start = performance.now();
  if (previewRAF) cancelAnimationFrame(previewRAF);
  const frame = now => {
    const t01 = ((now - start) / 1000 / animState.loopDur) % 1;
    drawPreviewFrame(ctx, srcP, bandInfo, offs, t01);
    previewRAF = requestAnimationFrame(frame);
  };
  previewRAF = requestAnimationFrame(frame);
}

function closeAnimDialog() {
  if (previewRAF) { cancelAnimationFrame(previewRAF); previewRAF = null; }
  $("animDlg").close();
}

function readAnimInputs() {
  const clamp = (v, lo, hi, def) => {
    const n = parseFloat(v);
    return isNaN(n) ? def : Math.max(lo, Math.min(hi, n));
  };
  animState.loopDur = clamp($("inLoopDur").value, 4, 60, 13.9);
  animState.drift = clamp($("inDrift").value, 0, 0.8, 0.42);
  animState.noise = clamp($("inNoise").value, 0, 40, 4);
  animState.bands = Math.round(clamp($("inBands").value, 10, 400, 94));
  const sel = $("animSel").value;
  if (sel !== animState.preset) {
    animState.preset = sel;
    $("animStatus").textContent = ANIM_HINTS[sel] || "";
  }
}

$("btnAnim").addEventListener("click", () => { readAnimInputs(); openAnimDialog(); });
$("animSel").addEventListener("change", () => {
  readAnimInputs();
  $("hint").textContent = ANIM_HINTS[animState.preset];
  // đổi preset cần tính lại offset band
  if ($("animDlg").open) openAnimDialog();
});
["inLoopDur", "inDrift", "inNoise", "inBands"].forEach(id =>
  $(id).addEventListener("change", () => { readAnimInputs(); openAnimDialog(); }));
$("animClose").addEventListener("click", closeAnimDialog);
$("animDlg").addEventListener("close", () => {
  if (previewRAF) { cancelAnimationFrame(previewRAF); previewRAF = null; }
});

$("btnApplyAnim").addEventListener("click", () => {
  readAnimInputs();
  // banner_config.json đúng schema banner.py đọc
  const cfg = { anim: animState.preset };
  if (animState.preset === "drift" && animState.drift !== 0.42) cfg.drift_fraction = animState.drift;
  if (animState.loopDur !== 13.9) cfg.loop_dur = animState.loopDur;
  if (animState.noise !== 4) cfg.band_noise = animState.noise;
  if (animState.bands !== 94) cfg.n_bands = animState.bands;
  if (dotShape !== "square") cfg.dot_shape = dotShape;
  const blob = new Blob([JSON.stringify(cfg, null, 2) + "\n"], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "banner_config.json";
  a.click();
  URL.revokeObjectURL(a.href);
  $("hint").textContent =
    `Đã tải banner_config.json (${cfg.anim}) → chép vào repo root, chạy: python3 .github/scripts/banner.py --from-npy`;
});

$("btnCopyCmd").addEventListener("click", async () => {
  readAnimInputs();
  const cmd = "python3 .github/scripts/banner.py --from-npy";
  try {
    await navigator.clipboard.writeText(cmd);
    $("animStatus").textContent = "Đã copy vào clipboard ✓";
  } catch {
    $("animStatus").textContent = cmd;
  }
});

// ---------------------------------------------------------------- banner thật
// Xem dark.svg / light.svg ĐỘNG (SMIL chạy trong <img>) ngay trong editor.
// SVG được lấy từ repo root qua ../../dark.svg — tức là bản trên đĩa, chỉ mới
// sau khi đã export .npy và chạy banner.py --from-npy.

function openBannerDialog() {
  const dlg = $("bannerDlg");
  // cảnh báo khi lưới trong editor khác với bản trên đĩa
  const edited = (panels.dark.loaded && panels.dark.dirty) ||
                 (panels.light.loaded && panels.light.dirty);
  $("bannerWarn").style.display = edited ? "" : "none";
  const stamp = `v=${Date.now()}`; // phá cache để thấy SVG vừa sinh lại
  $("imgBannerDark").src = `../../dark.svg?${stamp}`;
  $("imgBannerLight").src = `../../light.svg?${stamp}`;
  $("bannerStatus").textContent =
    (panels.dark.loaded || panels.light.loaded)
      ? "SVG lấy từ đĩa (repo root). Export .npy + chạy banner.py --from-npy để cập nhật."
      : "Chưa nạp lưới — đang xem bản SVG trong repo.";
  if (!dlg.open) dlg.showModal();
}

$("btnBanner").addEventListener("click", openBannerDialog);
$("bannerClose").addEventListener("click", () => $("bannerDlg").close());

for (const id of ["imgBannerDark", "imgBannerLight"]) {
  $(id).addEventListener("error", () => {
    $(id).title = "Không tìm thấy SVG — hãy chạy banner.py trước";
    $("bannerStatus").textContent =
      "Không tải được SVG từ repo root (chạy editor qua HTTP server trong repo).";
  });
}

// ---------------------------------------------------------------- boot
bindPanel("dark"); bindPanel("light");
for (const key of ["dark", "light"]) {
  const p = panels[key];
  layoutPanel(p); positionStack(p);
}
setTool("erase");
setSize(0);
setShape("square");
if (!restoreSession()) {
  $("hint").textContent =
    "Nạp 2 file .npy từ repo (portrait_dark.npy, portrait_light.npy) rồi bắt đầu chỉnh.";
}
