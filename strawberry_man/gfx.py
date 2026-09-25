"""Tiny 8-bit graphics toolkit: 320x180 canvas, bitmap font, sprites, dithering."""
import numpy as np

W, H = 320, 180

def hexc(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)

PAL = {k: hexc(v) for k, v in {
    'K': '#1a1025', 'R': '#e8283c', 'r': '#a3122e', 'P': '#ff8fa0', 'Y': '#ffd84a',
    'G': '#3cc24a', 'g': '#1d7a35', 'W': '#ffffff', 'S': '#f5c49a', 'M': '#3b1f5c',
    'B': '#ffc21f', 'L': '#c8cad8', 'D': '#6e7090', 'O': '#ff9a2e', 'T': '#efd08a',
    't': '#c0924e', 'C': '#f6eed4', 'c': '#ff7fa6', 'Q': '#3fb0a8', 'q': '#23777a',
    'u': '#a3542a', 'w': '#9fd8f5', 'y': '#ffe070', 'Z': '#4a4a58', 'z': '#8d8da0',
    'N': '#2a2a3a', 'E': '#ff5a1f', 'b': '#3a78d8',
}.items()}

# ---------------------------------------------------------------- font 5x7
_F = {
 'A': ".###.|#...#|#...#|#####|#...#|#...#|#...#", 'B': "####.|#...#|#...#|####.|#...#|#...#|####.",
 'C': ".###.|#...#|#....|#....|#....|#...#|.###.", 'D': "####.|#...#|#...#|#...#|#...#|#...#|####.",
 'E': "#####|#....|#....|####.|#....|#....|#####", 'F': "#####|#....|#....|####.|#....|#....|#....",
 'G': ".###.|#...#|#....|#.###|#...#|#...#|.####", 'H': "#...#|#...#|#...#|#####|#...#|#...#|#...#",
 'I': ".###.|..#..|..#..|..#..|..#..|..#..|.###.", 'J': "..###|...#.|...#.|...#.|#..#.|#..#.|.##..",
 'K': "#...#|#..#.|#.#..|##...|#.#..|#..#.|#...#", 'L': "#....|#....|#....|#....|#....|#....|#####",
 'M': "#...#|##.##|#.#.#|#.#.#|#...#|#...#|#...#", 'N': "#...#|#...#|##..#|#.#.#|#..##|#...#|#...#",
 'O': ".###.|#...#|#...#|#...#|#...#|#...#|.###.", 'P': "####.|#...#|#...#|####.|#....|#....|#....",
 'Q': ".###.|#...#|#...#|#...#|#.#.#|#..#.|.##.#", 'R': "####.|#...#|#...#|####.|#.#..|#..#.|#...#",
 'S': ".####|#....|#....|.###.|....#|....#|####.", 'T': "#####|..#..|..#..|..#..|..#..|..#..|..#..",
 'U': "#...#|#...#|#...#|#...#|#...#|#...#|.###.", 'V': "#...#|#...#|#...#|#...#|#...#|.#.#.|..#..",
 'W': "#...#|#...#|#...#|#.#.#|#.#.#|#.#.#|.#.#.", 'X': "#...#|#...#|.#.#.|..#..|.#.#.|#...#|#...#",
 'Y': "#...#|#...#|.#.#.|..#..|..#..|..#..|..#..", 'Z': "#####|....#|...#.|..#..|.#...|#....|#####",
 '0': ".###.|#...#|#..##|#.#.#|##..#|#...#|.###.", '1': "..#..|.##..|..#..|..#..|..#..|..#..|.###.",
 '2': ".###.|#...#|....#|...#.|..#..|.#...|#####", '3': "####.|....#|....#|.###.|....#|....#|####.",
 '4': "...#.|..##.|.#.#.|#..#.|#####|...#.|...#.", '5': "#####|#....|####.|....#|....#|#...#|.###.",
 '6': "..##.|.#...|#....|####.|#...#|#...#|.###.", '7': "#####|....#|...#.|..#..|.#...|.#...|.#...",
 '8': ".###.|#...#|#...#|.###.|#...#|#...#|.###.", '9': ".###.|#...#|#...#|.####|....#|...#.|.##..",
 '.': ".....|.....|.....|.....|.....|.##..|.##..", ',': ".....|.....|.....|.....|.##..|..#..|.#...",
 '!': "..#..|..#..|..#..|..#..|..#..|.....|..#..", '?': ".###.|#...#|....#|...#.|..#..|.....|..#..",
 "'": "..#..|..#..|.#...|.....|.....|.....|.....", '"': ".#.#.|.#.#.|.....|.....|.....|.....|.....",
 ':': ".....|.##..|.##..|.....|.##..|.##..|.....", '-': ".....|.....|.....|.###.|.....|.....|.....",
 '(': "...#.|..#..|.#...|.#...|.#...|..#..|...#.", ')': ".#...|..#..|...#.|...#.|...#.|..#..|.#...",
 '/': "....#|....#|...#.|..#..|.#...|#....|#....", '&': ".##..|#..#.|#.#..|.#...|#.#.#|#..#.|.##.#",
 '$': "..#..|.####|#.#..|.###.|..#.#|####.|..#..", '+': ".....|..#..|..#..|#####|..#..|..#..|.....",
 '=': ".....|.....|#####|.....|#####|.....|.....", '>': "#....|##...|###..|####.|###..|##...|#....",
 '%': "##..#|##.#.|...#.|..#..|.#...|.#.##|#..##", '@': "#####|#####|#####|#####|#####|#####|#####",
 '~': "#####|#...#|#...#|#...#|#...#|#...#|#####", '*': ".....|.#.#.|#####|#####|.###.|..#..|.....",
 'x': ".....|.....|#...#|.#.#.|..#..|.#.#.|#...#", '#': ".#.#.|#####|.#.#.|.#.#.|.#.#.|#####|.#.#.",
 '^': ".###.|#...#|#.#.#|##..#|#.#.#|#...#|.###.", ' ': ".....|.....|.....|.....|.....|.....|.....",
}
FONT = {k: np.array([[c == '#' for c in row] for row in v.split('|')], dtype=bool) for k, v in _F.items()}
CW = 6  # advance per char

def text_width(s, scale=1):
    return len(s) * CW * scale - scale

def text_mask(s, scale=1):
    m = np.zeros((7 * scale, max(1, len(s) * CW * scale)), dtype=bool)
    for i, ch in enumerate(s.upper() if ch_upper(s) else s):
        g = FONT.get(ch, FONT.get(ch.upper(), FONT['?']))
        if scale > 1:
            g = g.repeat(scale, 0).repeat(scale, 1)
        m[:, i * CW * scale:i * CW * scale + 5 * scale] = g
    return m

def ch_upper(s):
    return 'x' not in s

def dilate(m, r=1):
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= shift(m, dy, dx)
    return out

def shift(m, dy, dx):
    o = np.zeros_like(m)
    h, w = m.shape[:2]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    o[yd, xd] = m[ys, xs]
    return o

def put_mask(cv, m, x, y, color):
    """Paint boolean mask m onto canvas at (x,y) with color (clipped)."""
    x, y = int(round(x)), int(round(y))
    h, w = m.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(cv.shape[1], x + w), min(cv.shape[0], y + h)
    if x0 >= x1 or y0 >= y1:
        return
    sub = m[y0 - y:y1 - y, x0 - x:x1 - x]
    region = cv[y0:y1, x0:x1]
    col = np.asarray(color, dtype=np.uint8)
    if col.ndim == 1:
        region[sub] = col
    else:  # per-row colors (gradient)
        cg = col[y0 - y:y1 - y]
        region[sub] = np.broadcast_to(cg[:, None, :], region.shape)[sub]

def text(cv, s, x, y, color=PAL['W'], scale=1, shadow=None, outline=None, grad=None):
    m = text_mask(s, scale)
    if outline is not None:
        put_mask(cv, dilate(np.pad(m, 1), 1), x - 1, y - 1, outline)
    if shadow is not None:
        put_mask(cv, m, x + scale, y + scale, shadow)
    put_mask(cv, m, x, y, grad if grad is not None else color)

def text_c(cv, s, cx, y, **kw):
    text(cv, s, cx - text_width(s, kw.get('scale', 1)) // 2, y, **kw)

# ---------------------------------------------------------------- sprites
def sprite(rows, pal=PAL):
    rows = [r for r in rows]
    w = max(len(r) for r in rows)
    for i, r in enumerate(rows):
        assert len(r) == w, f"row {i} len {len(r)} != {w}: {r}"
    img = np.zeros((len(rows), w, 4), dtype=np.uint8)
    for yy, r in enumerate(rows):
        for xx, c in enumerate(r):
            if c != '.':
                img[yy, xx, :3] = pal[c]
                img[yy, xx, 3] = 255
    return img

def outline(img, color=PAL['K']):
    a = img[..., 3] > 0
    ring = dilate(np.pad(a, 1), 1) & ~np.pad(a, 1)
    out = np.pad(img, ((1, 1), (1, 1), (0, 0)))
    out[ring, :3] = color
    out[ring, 3] = 255
    return out

def flip(img):
    return img[:, ::-1]

def scale(img, s):
    return img.repeat(s, 0).repeat(s, 1)

def blit(cv, img, x, y, alpha=1.0, tint=None):
    x, y = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(cv.shape[1], x + w), min(cv.shape[0], y + h)
    if x0 >= x1 or y0 >= y1:
        return
    src = img[y0 - y:y1 - y, x0 - x:x1 - x]
    m = src[..., 3] > 0
    rgb = src[..., :3]
    if tint is not None:
        rgb = np.broadcast_to(np.asarray(tint, np.uint8), rgb.shape)
    region = cv[y0:y1, x0:x1]
    if alpha >= 1.0:
        region[m] = rgb[m]
    else:
        region[m] = (region[m] * (1 - alpha) + rgb[m] * alpha).astype(np.uint8)

def rect(cv, x, y, w, h, color):
    x, y = int(x), int(y)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(cv.shape[1], x + int(w)), min(cv.shape[0], y + int(h))
    if x0 < x1 and y0 < y1:
        cv[y0:y1, x0:x1] = color

def darken(cv, x, y, w, h, f):
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(cv.shape[1], int(x + w)), min(cv.shape[0], int(y + h))
    cv[y0:y1, x0:x1] = (cv[y0:y1, x0:x1] * f).astype(np.uint8)

BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16.0

def dither_gradient(h, w, stops, y0=0):
    """Vertical gradient quantized to the given color stops using 4x4 Bayer dithering."""
    cols = np.array([s[1] for s in stops], dtype=np.float32)
    pos = np.array([s[0] for s in stops], dtype=np.float32)
    ys = np.arange(h, dtype=np.float32) + y0
    t = np.interp(ys, pos, np.arange(len(stops)))
    idx = np.floor(t).astype(int)
    frac = t - idx
    thr = np.tile(BAYER, (h // 4 + 2, w // 4 + 2))[:h, :w]
    pick = idx[:, None] + (frac[:, None] > thr)
    pick = np.clip(pick, 0, len(stops) - 1)
    return cols[pick].astype(np.uint8)

def disc(r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx ** 2 + yy ** 2) <= r * r + r * 0.8

def box(cv, x, y, w, h, fill=PAL['K'], border=PAL['W'], border2=None):
    """RPG-style window with a double border and rounded corners."""
    rect(cv, x, y, w, h, fill)
    b2 = border2 if border2 is not None else PAL['D']
    rect(cv, x + 1, y, w - 2, 1, border); rect(cv, x + 1, y + h - 1, w - 2, 1, border)
    rect(cv, x, y + 1, 1, h - 2, border); rect(cv, x + w - 1, y + 1, 1, h - 2, border)
    rect(cv, x + 2, y + 2, w - 4, 1, b2); rect(cv, x + 2, y + h - 3, w - 4, 1, b2)
    rect(cv, x + 2, y + 2, 1, h - 4, b2); rect(cv, x + w - 3, y + 2, 1, h - 4, b2)

def ease_out(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3

def ease_in_out(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)

def back_out(t, s=1.9):
    t = min(max(t, 0.0), 1.0) - 1
    return t * t * ((s + 1) * t + s) + 1

def clamp01(t):
    return min(max(t, 0.0), 1.0)
