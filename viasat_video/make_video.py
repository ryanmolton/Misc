#!/usr/bin/env python3
"""
VIASAT: THE QUEST FOR CONNECTION
An 8-bit history of Viasat, told by Mark Dankberg.

Everything in the video is generated procedurally by this one script:
pixel art, sprites, animation, era-specific graphics pipelines (Game Boy ->
NES -> SNES -> HD), the chiptune soundtrack and every sound effect.
No image, audio or video assets are used, apart from two open-license
pixel fonts in ./fonts.

Usage:  python3 make_video.py            -> viasat_8bit_story.mp4
        python3 make_video.py --preview  -> key frames as PNG in ./preview
"""
import math, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile
from scipy.signal import fftconvolve, lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, SCALE, FPS, SR = 320, 180, 6, 30, 44100

F8 = ImageFont.truetype(os.path.join(HERE, "fonts/PressStart2P.ttf"), 8)
F16 = ImageFont.truetype(os.path.join(HERE, "fonts/PressStart2P.ttf"), 16)
F24 = ImageFont.truetype(os.path.join(HERE, "fonts/PressStart2P.ttf"), 24)
F32 = ImageFont.truetype(os.path.join(HERE, "fonts/PressStart2P.ttf"), 32)
FS = ImageFont.truetype(os.path.join(HERE, "fonts/Silkscreen.ttf"), 8)


def C(h, a=None):
    h = h.lstrip("#")
    c = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return c if a is None else c + (a,)


def lerp(a, b, t):
    return a + (b - a) * t


def clamp01(x):
    return max(0.0, min(1.0, x))


def ease(t):
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def mix(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def rng(seed):
    return np.random.RandomState(seed)


# ════════════════════════════════════════════════════════════════════
#  TIMELINE
# ════════════════════════════════════════════════════════════════════
def bars(n, bpm):
    return n * 4 * 60.0 / bpm


SCENES = [
    dict(name="title", bpm=120, dur=bars(2, 120), era="nes"),
    dict(name="w1", bpm=100, dur=bars(5, 100), era="gb", card=("WORLD 1", "1986 · CARLSBAD, CA", "1986")),
    dict(name="w2", bpm=120, dur=bars(6, 120), era="nes", card=("WORLD 2", "THE EARLY YEARS", "1990s")),
    dict(name="w3", bpm=128, dur=bars(7, 128), era="snes", card=("WORLD 3", "2011 · OUR OWN SATELLITE", "bunny"),
         badge="★ 16-BIT UPGRADE ★"),
    dict(name="w4", bpm=140, dur=bars(7, 140), era="snes", card=("WORLD 4", "EVERYWHERE", "pilot")),
    dict(name="w5", bpm=90, dur=bars(4, 90), era="hd", card=("WORLD 5", "TODAY · 2026", "today"),
         badge="★ HD UPGRADE ★"),
    dict(name="w6", bpm=100, dur=bars(6, 100), era="hd", card=("WORLD ∞", "WHAT'S NEXT", "astro")),
    dict(name="end", bpm=100, dur=bars(4, 100), era="hd"),
]
t0 = 0.0
for s in SCENES:
    s["start"] = t0
    t0 += s["dur"]
TOTAL = t0
CARD_T = 1.25  # length of world title cards

DIALOG = {
    "w1": [(1.5, "Hi, I'm Mark! In 1986, two friends and I started a little company in my spare bedroom.", "1986"),
           (6.9, "Our budget: $25,000 of our own money, one desk, and a LOT of coffee.", "1986")],
    "w2": [(1.4, "Our first big customer: the U.S. military. Our modems let many users share one satellite.", "1990s"),
           (6.0, "1996: we went public! And the spare bedroom went back to being a spare bedroom.", "1990s")],
    "w3": [(1.4, "Then we asked a crazy question: what if we built our OWN satellite?", "bunny"),
           (5.8, "ViaSat-1 launched in 2011 as the highest-capacity satellite on Earth. Well... above it.", "2011")],
    "w4": [(1.4, "Soon we were connecting homes... airplanes... ships at sea... and those who serve.", None),
           (7.3, "In 2023, Inmarsat joined our family, and together we cover the whole planet.", "today")],
    "w5": [(1.4, "Today, all three ViaSat-3 satellites are live. Each one adds over a terabit per second!", "today"),
           (7.0, "That's a LOT of cat videos.", "today")],
    "w6": [(1.4, "Next: quantum-safe encryption from tiny satellites, cyber defense, self-flying aircraft!", "astro"),
           (8.1, "And a network on the Moon. Yes... the MOON!", "astro")],
    "end": [(0.2, "It all started in a spare bedroom. Just imagine where we'll go next.", "today")],
}


def char_times(text, start, cps=31.0):
    ts, t = [], start
    for i, ch in enumerate(text):
        ts.append(t)
        t += 1.0 / cps
        nxt = text[i + 1] if i + 1 < len(text) else " "
        if ch in ".!?" and nxt == " ":
            t += 0.28
        elif ch == "." and nxt == ".":
            t += 0.10
        elif ch in ",:" and nxt == " ":
            t += 0.14
    return ts, t


def wrap(text, maxc=31):
    words, lines, cur = text.split(" "), [], ""
    for w_ in words:
        if len(cur) + len(w_) + (1 if cur else 0) <= maxc:
            cur = (cur + " " + w_) if cur else w_
        else:
            lines.append(cur)
            cur = w_
    lines.append(cur)
    return lines


def word_time(scene, line_idx, word):
    st, text, _ = DIALOG[scene][line_idx]
    ts, _ = char_times(text, st)
    return ts[text.index(word)]


# ════════════════════════════════════════════════════════════════════
#  SPRITES
# ════════════════════════════════════════════════════════════════════
BASE = [
    "....oooooooo....",
    "...ohhhhhhhho...",
    "..ohhhhhhhhhho..",
    "..ohhhhhhhhhho..",
    "..ohhsssssshho..",
    "..ohssssssssho..",
    "..ossessssesso..",
    "..ossessssesso..",
    "..orssssssssro..",
    "..ossssmmsssso..",
    "...ossssssssо...".replace("о", "o"),
    "....oooooooo....",
    "...oSSSccSSSo...",
    "..oSSSSSSSSSSo..",
    ".oSSSSSSSSSSSSo.",
    ".oSSoSSSSSSoSSo.",
    ".ossoSSSSSSosso.",
    "..oooPPPPPPooo..",
    "....oPPPPPPo....",
    "....oPPooPPo....",
    "....oPPooPPo....",
    "...oBBBooBBBo...",
    "...oooo..oooo...",
]
OX, OY, SW, SH = 2, 4, 20, 28

VARIANTS = {
    "1986": dict(h="#3b2414", S="#d9534f", P="#3a5a9a", B="#4a2e1a", x=["bighair"]),
    "1990s": dict(h="#4a2e1a", S="#f4f4f4", c="#f4f4f4", P="#5e6272", B="#222228", x=["tie"]),
    "bunny": dict(h="#8f8a86", S="#f4f6fa", c="#f4f6fa", P="#f4f6fa", B="#c8d4e0", x=["bunny"]),
    "2011": dict(h="#8f8a86", S="#2f6fb5", P="#3a3a48", B="#222228", x=["collar"]),
    "home": dict(h="#c4c4c4", S="#9a5a3c", P="#4a4a58", B="#3a2a20", x=["cardigan", "mug"]),
    "pilot": dict(h="#c4c4c4", S="#f4f4f4", c="#f4f4f4", P="#1f2a48", B="#111118", x=["tie", "pilotcap"]),
    "captain": dict(h="#c4c4c4", S="#f4f4f4", c="#f4f4f4", P="#f4f4f4", B="#111118", x=["captainhat"]),
    "defense": dict(h="#c4c4c4", S="#3d4a5e", P="#3a3a48", B="#111118", x=["headset"]),
    "today": dict(h="#e8e8e8", S="#1f3a68", P="#2a2a38", B="#111118", x=["blazer"]),
    "astro": dict(h="#eeeeee", S="#eef0f4", c="#eef0f4", P="#eef0f4", B="#8a929e", x=["astro"]),
    "friend1": dict(h="#1a1a1a", S="#2f9a9a", P="#3a5a9a", B="#222228", x=[]),
    "friend2": dict(h="#8a5a2a", S="#5a9a3a", P="#6a5a4a", B="#3a2a20", x=["stache", "bighair"]),
}
COMMON = dict(o="#1a1020", s="#f2c29a", e="#1a1020", r="#f09a8a", m="#8a3030",
              t="#c0282c", w="#bcd8f4", k="#1a1a22", N="#1f2a48", G="#f4c430", W="#f8f8f8",
              M="#8cc8f0", L="#b8c0cc", R="#e04040", U="#5a3a2a", Y="#6a7a8a")
_sprite_cache = {}


def mark_sprite(var="1986", blink=False, talk=False, scale=1):
    key = (var, blink, talk, scale)
    if key in _sprite_cache:
        return _sprite_cache[key]
    v = VARIANTS[var]
    grid = [["."] * SW for _ in range(SH)]
    for r, row in enumerate(BASE):
        for c, ch in enumerate(row):
            grid[r + OY][c + OX] = ch

    def S(c, r, ch):
        if 0 <= r + OY < SH and 0 <= c + OX < SW:
            grid[r + OY][c + OX] = ch

    def G_(c, r):
        return grid[r + OY][c + OX]

    x = v["x"]
    if blink:
        S(5, 6, "s"); S(10, 6, "s")
    if talk:
        S(7, 9, "m"); S(8, 9, "m"); S(7, 10, "m"); S(8, 10, "m")
    if "bighair" in x:
        for c in range(4, 12):
            S(c, -2, "o"); S(c, -1, "h"); S(c, 0, "h")
        S(3, -1, "o"); S(12, -1, "o"); S(3, 0, "h"); S(12, 0, "h"); S(2, 0, "o"); S(13, 0, "o")
        for r in range(2, 8):
            S(1, r, "h"); S(14, r, "h"); S(0, r, "o"); S(15, r, "o")
        S(1, 1, "o"); S(14, 1, "o"); S(1, 8, "o"); S(14, 8, "o")
    if "stache" in x:
        for c in range(5, 11):
            S(c, 9, "h")
    if "tie" in x:
        for r in range(12, 17):
            S(7, r, "t"); S(8, r, "t")
    if "collar" in x:
        S(6, 12, "W"); S(9, 12, "W")
    if "blazer" in x:
        for c in range(6, 10):
            S(c, 12, "w")
        for r in (13, 14):
            S(7, r, "w"); S(8, r, "w")
    if "cardigan" in x:
        for r in range(12, 17):
            S(7, r, "W"); S(8, r, "W")
        S(6, 14, "o"); S(6, 16, "o")
    if "mug" in x:
        for (c, r) in [(13, 13), (14, 13), (13, 14), (14, 14), (13, 15), (14, 15)]:
            S(c, r, "W")
        S(15, 14, "W"); S(12, 13, "o"); S(15, 13, "o"); S(13, 12, "o"); S(14, 12, "o")
        S(13, 16, "o"); S(14, 16, "o"); S(16, 14, "o"); S(15, 15, "o")
    if "pilotcap" in x or "captainhat" in x:
        crown = "N" if "pilotcap" in x else "W"
        for c in range(3, 13):
            S(c, -3, "o")
        for r in (-2, -1, 0, 1):
            for c in range(2, 14):
                S(c, r, crown)
            S(1, r, "o"); S(14, r, "o")
        if "captainhat" in x:
            for c in range(2, 14):
                S(c, 1, "k")
        for c in range(1, 15):
            S(c, 2, "k")
        S(7, -1, "G"); S(8, -1, "G"); S(7, 0, "G"); S(8, 0, "G")
    if "headset" in x:
        for c in range(3, 13):
            S(c, 0, "k")
        for r in range(4, 8):
            S(1, r, "k"); S(2, r, "k"); S(13, r, "k"); S(14, r, "k")
        S(2, 1, "k"); S(13, 1, "k"); S(2, 2, "k"); S(13, 2, "k"); S(2, 3, "k"); S(13, 3, "k")
        S(3, 8, "k"); S(4, 9, "k"); S(5, 9, "R")
    if "bunny" in x:
        for r in range(0, 6):
            for c in range(16):
                if G_(c, r) == "h":
                    S(c, r, "W")
        for r in range(4, 11):
            S(2, r, "W"); S(13, r, "W"); S(1, r, "o"); S(14, r, "o")
        S(2, 3, "o"); S(13, 3, "o")
        for r in (8, 9, 10):
            for c in range(3, 13):
                if G_(c, r) in "srm":
                    S(c, r, "M")
    if "astro" in x:
        S(6, 14, "R"); S(7, 14, "G"); S(8, 14, "M"); S(9, 14, "R")
    pal = dict(COMMON)
    pal.update({k: v[k] for k in v if len(k) == 1})
    pal.setdefault("c", COMMON["s"])
    arr = np.zeros((SH, SW, 4), np.uint8)
    for r in range(SH):
        for c in range(SW):
            ch = grid[r][c]
            if ch != ".":
                arr[r, c] = C(pal[ch], 255)
    if "astro" in x:  # glass helmet bubble
        cx, cy, R = OX + 7.5, OY + 4.5, 8.6
        for r in range(SH):
            for c in range(SW):
                d = math.hypot(c - cx, r - cy)
                if R - 1.0 <= d < R and r < OY + 12:
                    arr[r, c] = C("#c8d0dc", 255)
                elif d < R - 1.0 and r < OY + 12:
                    if arr[r, c, 3] == 0:
                        arr[r, c] = C("#9fd8ff", 90)
                    else:
                        arr[r, c, :3] = (arr[r, c, :3] * 0.8 + np.array(C("#9fd8ff")) * 0.2).astype(np.uint8)
        for (c, r) in [(OX + 2, OY + 1), (OX + 3, OY + 0), (OX + 2, OY + 2), (OX + 4, OY - 1)]:
            arr[r, c] = C("#ffffff", 230)
    img = Image.fromarray(arr, "RGBA")
    if scale != 1:
        img = img.resize((SW * scale, SH * scale), Image.NEAREST)
    _sprite_cache[key] = img
    return img


def paste(img, spr, x, y):
    img.paste(spr, (int(round(x)), int(round(y))), spr)


def put_mark(img, var, x, feet_y, t, scale=2, bob=True, talk=False, phase=0.0, hop=0.0):
    """Places Mark so that his feet are at feet_y, centred on x."""
    blink = (math.fmod(t + phase * 1.7, 3.1) < 0.12)
    by = 0
    if bob:
        by = -scale * (1 if math.sin((t + phase) * 2 * math.pi * 1.2) > 0.6 else 0)
    spr = mark_sprite(var, blink, talk, scale)
    yy = feet_y - (OY + 23) * scale + by - hop
    xx = x - (SW * scale) // 2
    if scale == 2:
        xx, yy = int(xx) // 2 * 2, int(yy) // 2 * 2
    paste(img, spr, xx, yy)
    return xx, yy


# ── other pixel sprites from ascii ──────────────────────────────────
def ascii_sprite(rows, pal, scale=1):
    h, w = len(rows), max(len(r) for r in rows)
    arr = np.zeros((h, w, 4), np.uint8)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch in pal:
                arr[r, c] = C(pal[ch], 255)
    img = Image.fromarray(arr, "RGBA")
    return img.resize((w * scale, h * scale), Image.NEAREST) if scale != 1 else img


CAT = ascii_sprite([
    "o.....o..",
    "oo...oo..",
    "oyyyyyo..",
    "oyeyeyo.o",
    "oyypyyo.o",
    ".oyyyyooo",
    ".oyyyyyo.",
    ".oyoyoyo.",
    ".o.o.o.o.",
], dict(o="#2a1a10", y="#f0a040", e="#1a1010", p="#f08090"), 1)

LOCK = ascii_sprite([
    ".ooo.",
    "o...o",
    "o...o",
    "yyyyy",
    "yyoyy",
    "yyoyy",
    "yyyyy",
], dict(o="#d8e0ea", y="#f4c430"), 1)

HEART = ascii_sprite([
    ".rr.rr.",
    "rrrrrrr",
    "rrrrrrr",
    ".rrrrr.",
    "..rrr..",
    "...r...",
], dict(r="#ff5070"), 1)


# ════════════════════════════════════════════════════════════════════
#  COLOUR PIPELINES  (the graphics "upgrade" through the eras)
# ════════════════════════════════════════════════════════════════════
GB_PAL = np.array([C("#0f380f"), C("#306230"), C("#8bac0f"), C("#9bbc0f")], np.float32)
NES_HEX = ("7C7C7C 0000FC 0000BC 4428BC 940084 A80020 A81000 881400 503000 007800 006800 005800 004058 000000 "
           "BCBCBC 0078F8 0058F8 6844FC D800CC E40058 F83800 E45C10 AC7C00 00B800 00A800 00A844 008888 "
           "F8F8F8 3CBCFC 6888FC 9878F8 F878F8 F85898 F87858 FCA044 F8B800 B8F818 58D854 58F898 00E8D8 787878 "
           "FCFCFC A4E4FC B8B8F8 D8B8F8 F8B8F8 F8A4C0 F0D0B0 FCE0A8 F8D878 D8F878 B8F8B8 B8F8D8 00FCFC F8D8F8").split()
NES_PAL = np.array([C(h) for h in NES_HEX], np.float32)


def era_colour(arr, era):
    """arr: HxWx3 uint8 -> era-styled uint8"""
    if era == "gb":
        y = arr[..., 0] * 0.30 + arr[..., 1] * 0.59 + arr[..., 2] * 0.11
        idx = np.digitize(y, [62, 118, 178])
        return GB_PAL[idx].astype(np.uint8)
    if era == "nes":
        a = arr.reshape(-1, 1, 3).astype(np.float32)
        # perceptual-ish weighting
        d = ((a - NES_PAL[None]) ** 2).sum(-1)
        return NES_PAL[d.argmin(1)].reshape(arr.shape).astype(np.uint8)
    if era == "snes":
        return (arr & 0xF8) | (arr >> 5)  # 15-bit colour
    return arr


def post(img, era):
    arr = np.asarray(img.convert("RGB"))
    if era == "gb":  # chunky 160x90 Game Boy resolution
        small = arr[::2, ::2]
        small = era_colour(small, "gb")
        arr = small.repeat(2, 0).repeat(2, 1)
    else:
        arr = era_colour(arr, era)
    return Image.fromarray(np.ascontiguousarray(arr), "RGB")


def era_img(img, era):
    return Image.fromarray(np.ascontiguousarray(era_colour(np.asarray(img.convert("RGB")), era)), "RGB")


# ════════════════════════════════════════════════════════════════════
#  PROCEDURAL WORLD PIECES
# ════════════════════════════════════════════════════════════════════
def value_noise(w, h, cells, seed, octaves=4):
    r = rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        cw, ch = cells * 2 ** o, max(2, cells * 2 ** o // 2)
        g = r.rand(ch + 1, cw + 1).astype(np.float32)
        g[:, -1] = g[:, 0]
        img = Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        out += np.asarray(img, np.float32) / 255 * amp
        tot += amp
        amp *= 0.5
    return out / tot


TEX_W, TEX_H = 256, 128
_land = value_noise(TEX_W, TEX_H, 4, 7)
_dry = value_noise(TEX_W, TEX_H, 6, 11)
_cloud = value_noise(TEX_W, TEX_H, 12, 23, 4)
_lat = np.linspace(90, -90, TEX_H)[:, None] * np.ones((1, TEX_W))
LAND = _land > 0.52
ICE = np.abs(_lat) > 72
CITY = LAND & (rng(5).rand(TEX_H, TEX_W) > 0.93)
_moonn = value_noise(TEX_W, TEX_H, 5, 31)


def earth_tex():
    tex = np.zeros((TEX_H, TEX_W, 3), np.float32)
    tex[:] = C("#1c5cc0")
    tex[_land < 0.47] = C("#174a9e")
    tex[LAND] = C("#3f9a45")
    tex[LAND & (_dry > 0.55)] = C("#c8b060")
    tex[LAND & (_land > 0.62)] = C("#2f7a35")
    tex[ICE] = C("#eef4f8")
    return tex


EARTH_TEX = earth_tex()
MOON_TEX = np.stack([_moonn * 90 + 110] * 3, -1)
_cr = rng(99)
for _ in range(40):
    cx_, cy_, rr_ = _cr.randint(0, TEX_W), _cr.randint(10, TEX_H - 10), _cr.randint(2, 8)
    yy_, xx_ = np.ogrid[:TEX_H, :TEX_W]
    d_ = np.hypot((xx_ - cx_), (yy_ - cy_))
    MOON_TEX[(d_ < rr_)] *= 0.78
    MOON_TEX[(d_ >= rr_) & (d_ < rr_ + 1)] *= 1.12


def draw_sphere(img, cx, cy, r, rot, kind="earth", light=(-0.6, -0.35, 0.72), night_lights=True,
                clouds=True, cloud_rot=0.0, atmo=True):
    arr = np.array(img.convert("RGB"), np.float32)
    x0, x1 = max(0, int(cx - r - 3)), min(W, int(cx + r + 4))
    y0, y1 = max(0, int(cy - r - 3)), min(H, int(cy + r + 4))
    if x0 >= x1 or y0 >= y1:
        return img
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    nx, ny = (xs + 0.5 - cx) / r, (ys + 0.5 - cy) / r
    d2 = nx * nx + ny * ny
    inside = d2 < 1
    nz = np.sqrt(np.clip(1 - d2, 0, 1))
    lat = np.arcsin(np.clip(-ny, -1, 1))
    lon = np.arctan2(nx, nz) + rot
    u = ((lon / (2 * np.pi)) % 1 * TEX_W).astype(int) % TEX_W
    v = np.clip(((0.5 - lat / np.pi) * TEX_H).astype(int), 0, TEX_H - 1)
    tex = EARTH_TEX if kind == "earth" else MOON_TEX
    col = tex[v, u].copy()
    L = np.array(light, np.float32)
    L /= np.linalg.norm(L)
    lam = nx * L[0] + ny * L[1] + nz * L[2]
    if kind == "earth" and clouds:
        uc = ((((lon + cloud_rot) / (2 * np.pi)) % 1) * TEX_W).astype(int) % TEX_W
        cl = _cloud[v, uc] > 0.62
        col[cl] = col[cl] * 0.25 + np.array(C("#ffffff")) * 0.75
    shade = np.clip(lam * 1.1 + 0.25, 0.0, 1.0)[..., None]
    night = np.array(C("#0a1430") if kind == "earth" else C("#1a1a22"), np.float32)
    out = col * shade + night * (1 - shade)
    if kind == "earth" and night_lights:
        cit = CITY[v, u] & (lam < 0.0)
        out[cit] = C("#ffd860")
    if atmo and kind == "earth":
        ring = (d2 >= 1) & (d2 < (1 + 2.6 / r) ** 2)
        a = arr[y0:y1, x0:x1]
        a[ring] = a[ring] * 0.35 + np.array(C("#6fb8ff")) * 0.65
        rim = inside & (d2 > (1 - 2.0 / r) ** 2) & (lam > -0.2)
        out[rim] = out[rim] * 0.5 + np.array(C("#9fd4ff")) * 0.5
    sub = arr[y0:y1, x0:x1]
    sub[inside] = out[inside]
    arr[y0:y1, x0:x1] = sub
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


STARS = [(rng(3).randint(0, W), rng(3 + i).randint(0, H), rng(9 + i).rand(), rng(17 + i).rand() * 6.28)
         for i in range(0)]
_sr = rng(3)
STARS = [(_sr.randint(0, W), _sr.randint(0, H), _sr.rand(), _sr.rand() * 6.28) for _ in range(140)]


def stars(d, t, ymax=H, dens=1.0, big=True):
    for i, (x, y, b, ph) in enumerate(STARS):
        if y >= ymax or i > len(STARS) * dens:
            continue
        tw = 0.55 + 0.45 * math.sin(t * (1.5 + b * 3) + ph)
        v = int(120 + 135 * tw * (0.4 + 0.6 * b))
        c = (v, v, min(255, v + 25))
        d.point((x, y), fill=c)
        if big and b > 0.93 and tw > 0.7:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                d.point((x + dx, y + dy), fill=(v // 2, v // 2, v // 2 + 20))


def vgrad(d, y0, y1, c0, c1, x0=0, x1=W, steps=None):
    n = y1 - y0
    for i in range(n):
        tt = i / max(1, n - 1)
        if steps:
            tt = round(tt * steps) / steps
        d.line([(x0, y0 + i), (x1 - 1, y0 + i)], fill=mix(c0, c1, tt))


def small(d, xy, s, fill, anchor="la"):
    d.text(xy, s, font=FS, fill=fill, anchor=anchor)


def text_shadow(d, xy, s, font, fill, shadow=(0, 0, 0), anchor="la", off=1):
    x, y = xy
    d.text((x + off, y + off), s, font=font, fill=shadow, anchor=anchor)
    d.text((x, y), s, font=font, fill=fill, anchor=anchor)


def draw_satellite(d, x, y, kind="modern", s=1.0, lit=1.0, open_=1.0, t=0.0):
    """Satellite centred on x,y. kind: uhf | vs1 | modern | cube"""
    x, y = int(x), int(y)
    if kind == "uhf":
        d.rectangle([x - 4, y - 5, x + 4, y + 5], fill=C("#c8c8d0"), outline=C("#505060"))
        d.rectangle([x - 16, y - 2, x - 6, y + 2], fill=C("#2a3a8a"), outline=C("#101830"))
        d.rectangle([x + 6, y - 2, x + 16, y + 2], fill=C("#2a3a8a"), outline=C("#101830"))
        d.line([(x - 6, y), (x - 4, y)], fill=C("#505060"))
        d.line([(x + 4, y), (x + 6, y)], fill=C("#505060"))
        d.line([(x, y - 5), (x, y - 11)], fill=C("#e0e0e0"))
        d.line([(x - 3, y - 11), (x + 3, y - 11)], fill=C("#e0e0e0"))
        return
    if kind == "cube":
        d.rectangle([x - 3, y - 3, x + 3, y + 3], fill=C("#d8b040"), outline=C("#503010"))
        d.rectangle([x - 11, y - 2, x - 5, y + 2], fill=C("#3050c0"), outline=C("#101830"))
        d.rectangle([x + 5, y - 2, x + 11, y + 2], fill=C("#3050c0"), outline=C("#101830"))
        return
    # vs1 / modern: gold body, big wings, reflectors
    bw, bh = int(7 * s), int(9 * s)
    wing = int((24 if kind == "modern" else 18) * s * open_)
    wh = int(4 * s)
    if wing > 0:
        for sgn in (-1, 1):
            xa = x + sgn * (bw + 1)
            xb = x + sgn * (bw + 1 + wing)
            d.line([(x + sgn * bw, y), (xa, y)], fill=C("#a0a0a8"))
            d.rectangle([min(xa, xb), y - wh, max(xa, xb), y + wh], fill=C("#233c8c"), outline=C("#0c1640"))
            step = max(3, int(5 * s))
            for k in range(min(xa, xb) + step, max(xa, xb), step):
                d.line([(k, y - wh), (k, y + wh)], fill=C("#0c1640"))
            if lit > 0.5:
                d.line([(min(xa, xb) + 1, y - wh + 1), (max(xa, xb) - 1, y - wh + 1)], fill=C("#5a7ae0"))
    d.rectangle([x - bw, y - bh, x + bw, y + bh], fill=C("#e0a830"), outline=C("#6a4010"))
    for k in range(y - bh + 2, y + bh, 3):
        d.line([(x - bw + 1, k), (x + bw - 1, k)], fill=C("#f4c850"))
    # reflectors
    rr = int(6 * s)
    d.ellipse([x - bw - rr, y + bh - rr - 2, x - bw + rr - 4, y + bh + rr - 2], fill=C("#f0f0f4"), outline=C("#8a8a96"))
    d.ellipse([x + bw - rr + 4, y + bh - rr - 2, x + bw + rr, y + bh + rr - 2], fill=C("#f0f0f4"), outline=C("#8a8a96"))
    if kind == "modern":
        d.ellipse([x - rr + 1, y - bh - rr - 3, x + rr - 1, y - bh + 2], fill=C("#f8f8fc"), outline=C("#8a8a96"))
    if lit > 0:
        blink = (math.sin(t * 8) > 0)
        d.point((x, y - bh - 1), fill=C("#ff4040") if blink else C("#601010"))


def beam(d, x0, y0, x1, y1, t, col=(120, 220, 255), speed=60, gap=6, dash=3):
    L = math.hypot(x1 - x0, y1 - y0)
    if L < 1:
        return
    off = (t * speed) % gap
    s = off
    while s < L:
        e = min(L, s + dash)
        a, b = s / L, e / L
        d.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)], fill=col)
        s += gap


def wifi(d, x, y, t, col, n=3):
    k = int(t * 4) % (n + 1)
    for i in range(1, n + 1):
        if i <= k or k == 0:
            r = 3 * i
            d.arc([x - r, y - r, x + r, y + r], 225, 315, fill=col)
    d.point((x, y), fill=col)


def iris(img, t, dur, cx=W / 2, cy=H / 2 - 20, tin=0.3, tout=0.35, start_t=0.0):
    """classic circle wipe in after start_t, out at the end"""
    r = None
    if start_t <= t < start_t + tin:
        r = ease((t - start_t) / tin) * 200
    elif t > dur - tout:
        r = (1 - ease((t - (dur - tout)) / tout)) * 200
    if r is None:
        return img
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    black = Image.new("RGB", (W, H), (0, 0, 0))
    return Image.composite(img, black, mask)


# ════════════════════════════════════════════════════════════════════
#  UI: world cards + dialog box
# ════════════════════════════════════════════════════════════════════
def draw_card(scene, t):
    era = scene["era"]
    title, sub, var = scene["card"]
    bg, fg, accent = (0, 0, 0), (248, 248, 248), (252, 160, 68)
    if era == "gb":
        bg, fg, accent = tuple(GB_PAL[0].astype(int)), tuple(GB_PAL[3].astype(int)), tuple(GB_PAL[2].astype(int))
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    d.fontmode = "1"
    if era == "hd":
        vgrad(d, 0, H, C("#050818"), C("#101a3a"))
        stars(d, t, dens=0.5, big=False)
    d.text((W // 2, 52), title, font=F16, fill=fg, anchor="mm")
    d.text((W // 2, 76), sub, font=F8, fill=accent, anchor="mm")
    spr = mark_sprite(var, blink=(0.5 < t < 0.6), scale=2)
    if era in ("gb", "nes"):
        spr = spr.copy()
        a = np.array(spr)
        a[..., :3] = era_colour(a[..., :3], era)
        spr = Image.fromarray(a, "RGBA")
    paste(img, spr, W // 2 - 44, 88)
    lbl = {"1986": "× 3 FOUNDERS", "1990s": "× 1 BIG IDEA", "bunny": "× 1 SATELLITE", "pilot": "× 4 MARKETS",
           "today": "× 3 VIASAT-3", "astro": "× ∞ IDEAS"}[var]
    d.text((W // 2 + 2, 115), lbl, font=F8, fill=fg, anchor="lm")
    if scene.get("badge") and t > 0.25:
        if int(t * 8) % 2 == 0 or t > 0.8:
            text_shadow(d, (W // 2, 150), scene["badge"], F8, (255, 230, 90) if era != "gb" else fg, anchor="mm")
    return img


BOX = (4, 131, 316, 176)


def draw_dialog(img, era, text, n_vis, var, talking, t, future=False):
    d = ImageDraw.Draw(img, "RGBA")
    d.fontmode = "1"
    x0, y0, x1, y1 = BOX
    if era == "gb":
        dk, lt, md = [tuple(GB_PAL[i].astype(int)) for i in (0, 3, 1)]
        d.rectangle(BOX, fill=lt, outline=dk)
        d.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], outline=dk)
        txt, sh, name_bg = dk, None, dk
    elif era == "nes":
        d.rectangle(BOX, fill=(0, 0, 0), outline=(252, 252, 252))
        d.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], outline=(188, 188, 188))
        txt, sh, name_bg = (252, 252, 252), None, (0, 0, 0)
    elif era == "snes":
        for i in range(y1 - y0):
            d.line([(x0, y0 + i), (x1, y0 + i)], fill=mix(C("#3050d0"), C("#101860"), i / (y1 - y0)))
        d.rectangle(BOX, outline=(230, 230, 240))
        d.rectangle([x0 + 1, y0 + 1, x1 - 1, y1 - 1], outline=(140, 150, 190))
        txt, sh, name_bg = (255, 255, 255), (20, 20, 60), C("#3050d0")
    else:
        acc = C("#c080ff") if future else C("#40d0ff")
        d.rounded_rectangle(BOX, 4, fill=(8, 14, 36, 215), outline=acc + (255,))
        d.rounded_rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], 3, outline=acc + (80,))
        txt, sh, name_bg = (255, 255, 255), (0, 0, 20), acc
    # portrait
    px, py = x0 + 5, y0 + 5
    spr = mark_sprite(var, blink=(math.fmod(t, 2.7) < 0.12), talk=talking, scale=2)
    head = spr.crop((0, 2, 40, 2 + 36))
    frame_bg = {"gb": tuple(GB_PAL[2].astype(int)), "nes": (0, 88, 248), "snes": C("#6a8ae8")}.get(era, C("#1a3a7a"))
    pimg = Image.new("RGB", (40, 36), frame_bg)
    pimg.paste(head, (0, 0), head)
    if era in ("gb", "nes"):
        pimg = era_img(pimg, era)
    img.paste(pimg, (px, py))
    d.rectangle([px - 1, py - 1, px + 40, py + 36], outline=txt if era != "gb" else tuple(GB_PAL[0].astype(int)))
    # name tab
    nm = "MARK"
    d.rectangle([x0 + 4, y0 - 8, x0 + 42, y0], fill=name_bg,
                outline=(252, 252, 252) if era != "gb" else tuple(GB_PAL[0].astype(int)))
    d.text((x0 + 8, y0 - 7), nm, font=F8, fill=(252, 252, 252) if era != "gb" else tuple(GB_PAL[3].astype(int)))
    # text
    lines = wrap(text)
    left = n_vis
    tx, ty = x0 + 51, y0 + 7
    for i, ln in enumerate(lines):
        if left <= 0:
            break
        seg = ln[:left]
        left -= len(ln) + 1
        if sh:
            d.text((tx + 1, ty + i * 12 + 1), seg, font=F8, fill=sh)
        d.text((tx, ty + i * 12), seg, font=F8, fill=txt)
    if n_vis >= len(text) and int(t * 3) % 2 == 0:
        d.text((x1 - 12, y1 - 11), "▼", font=F8, fill=txt)


# ════════════════════════════════════════════════════════════════════
#  SCENES
# ════════════════════════════════════════════════════════════════════
def sc_title(t, dur):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    d.fontmode = "1"
    vgrad(d, 0, H, C("#000020"), C("#1830a0"), steps=6)
    stars(d, t)
    img = draw_sphere(img, 160, 362, 205, 1.2 + t * 0.05, light=(0.0, -0.7, 0.7), night_lights=False)
    d = ImageDraw.Draw(img)
    d.fontmode = "1"
    sx = -30 + t * 55
    draw_satellite(d, sx, 100, "uhf", t=t)
    # title
    wob = [0, -1, -2, -1, 0, 1]
    for i, ch in enumerate("VIASAT"):
        yy = 34 + (wob[(int(t * 10) + i) % 6] if t < 3.2 else 0)
        xx = 160 - 96 + i * 32 + 16
        d.text((xx + 2, yy + 3), ch, font=F32, fill=(0, 0, 60), anchor="mm")
        cyc = [C("#fcfcfc"), C("#a4e4fc"), C("#3cbcfc"), C("#a4e4fc")]
        d.text((xx, yy), ch, font=F32, fill=cyc[(int(t * 8) - i) % 4], anchor="mm")
    d.text((160, 64), "THE QUEST FOR CONNECTION", font=F8, fill=C("#fca044"), anchor="mm")
    d.text((160, 80), "AS TOLD BY MARK DANKBERG", font=F8, fill=C("#fcfcfc"), anchor="mm")
    blink = (int(t * 2.5) % 2 == 0) if t < 3.1 else (int(t * 14) % 2 == 0)
    if blink:
        d.text((160, 122), "PRESS START", font=F8, fill=C("#fcfcfc"), anchor="mm")
    small(d, (160, 140), "EST. 1986  ·  CARLSBAD, CALIFORNIA", C("#bcbcbc"), anchor="mm")
    return img


def sc_w1(t, dur):
    img = Image.new("RGB", (W, H), C("#e8d8b8"))
    d = ImageDraw.Draw(img)
    # wallpaper stripes
    for x in range(0, W, 16):
        d.rectangle([x, 0, x + 7, 112], fill=C("#dccaa6"))
    d.rectangle([0, 112, W, H], fill=C("#8a5a34"))
    for x in range(0, W, 24):
        d.line([(x, 112), (x - 30, H)], fill=C("#6a4020"))
    d.line([(0, 112), (W, 112)], fill=C("#4a2a10"))
    # window with Carlsbad palm + ocean
    wx0, wy0, wx1, wy1 = 16, 16, 76, 64
    d.rectangle([wx0 - 3, wy0 - 3, wx1 + 3, wy1 + 3], fill=C("#f4f4f4"))
    vgrad(d, wy0, wy1, C("#8ad0ff"), C("#e0f4ff"), wx0, wx1 + 1)
    d.rectangle([wx0, 52, wx1, wy1], fill=C("#2a6ab0"))
    d.ellipse([54, 22, 66, 34], fill=C("#fff4a0"))
    d.line([(34, 64), (36, 34)], fill=C("#5a3a1a"), width=2)
    for ang in (-150, -110, -60, -20, 20):
        a = math.radians(ang + 4 * math.sin(t * 2))
        d.line([(36, 34), (36 + 13 * math.cos(a), 34 + 7 * math.sin(a) + 5)], fill=C("#2a7a2a"), width=2)
    d.line([(46, wy0), (46, wy1)], fill=C("#f4f4f4"), width=2)
    d.line([(wx0, 40), (wx1, 40)], fill=C("#f4f4f4"), width=2)
    # whiteboard w/ doodle of the big idea
    d.rectangle([108, 12, 192, 58], fill=C("#fafafa"), outline=C("#6a6a6a"))
    d.rectangle([110, 57, 190, 60], fill=C("#8a8a8a"))
    d.rectangle([140, 20, 150, 28], outline=C("#303030"))
    d.line([(130, 24), (140, 24)], fill=C("#303030"))
    d.line([(150, 24), (160, 24)], fill=C("#303030"))
    d.arc([118, 42, 132, 56], 180, 360, fill=C("#303030"))
    d.arc([168, 42, 182, 56], 180, 360, fill=C("#303030"))
    for k in range(0, 5):
        if (t * 3 + k) % 5 < 3.5:
            d.point((133 + k * 2, 42 - k * 3), fill=C("#303030"))
            d.point((167 - k * 2, 42 - k * 3), fill=C("#303030"))
    small(d, (150, 16), "BIG IDEA!", C("#303030"), anchor="mt")
    # calendar
    d.rectangle([204, 18, 232, 50], fill=C("#fafafa"), outline=C("#404040"))
    d.rectangle([204, 18, 232, 26], fill=C("#c02020"))
    small(d, (218, 30), "1986", C("#202020"), anchor="mt")
    for gy in range(40, 48, 3):
        for gx in range(208, 230, 4):
            d.point((gx, gy), fill=C("#808080"))
    # the spare bed
    d.rectangle([244, 70, 250, 118], fill=C("#6a3a1a"))
    d.rectangle([250, 92, 316, 112], fill=C("#f0f0f0"))
    d.rectangle([258, 88, 316, 110], fill=C("#5a80c8"))
    d.rectangle([252, 84, 268, 94], fill=C("#fafafa"), outline=C("#a0a0a0"))
    d.rectangle([250, 112, 254, 120], fill=C("#6a3a1a"))
    d.rectangle([310, 112, 314, 120], fill=C("#6a3a1a"))
    # boxes on the bed (office supplies!)
    d.rectangle([278, 70, 300, 88], fill=C("#c09058"), outline=C("#6a4a20"))
    d.line([(278, 76), (300, 76)], fill=C("#6a4a20"))
    d.rectangle([284, 58, 298, 70], fill=C("#c09058"), outline=C("#6a4a20"))
    # desk + computer
    d.rectangle([98, 90, 210, 95], fill=C("#7a4a24"), outline=C("#3a2008"))
    d.rectangle([102, 95, 106, 124], fill=C("#5a3414"))
    d.rectangle([202, 95, 206, 124], fill=C("#5a3414"))
    d.rectangle([126, 60, 162, 88], fill=C("#e0d8c0"), outline=C("#5a5448"))
    d.rectangle([130, 63, 158, 81], fill=C("#103010"))
    for i, ln in enumerate(["C:\\>", "DAMA.BAS"]):
        small(d, (132, 64 + i * 7), ln, C("#50f050"))
    if int(t * 2.5) % 2 == 0:
        d.rectangle([150, 72, 153, 77], fill=C("#50f050"))
    d.rectangle([124, 88, 164, 90], fill=C("#c8c0a8"))
    d.rectangle([118, 84, 124, 90], fill=C("#303030"))  # floppy
    # coffee mugs (they multiply during the "LOT of coffee" line)
    mugs = 1
    L2 = DIALOG["w1"][1][0]
    if t > L2:
        mugs = 1 + min(6, int((t - L2 - 1.4) * 3.5)) if t > L2 + 1.4 else 1
    for m in range(mugs):
        mx = 172 + (m % 4) * 8
        my = 82 - (m // 4) * 7
        d.rectangle([mx, my, mx + 5, my + 7], fill=C("#fafafa"), outline=C("#303030"))
        d.point((mx + 6, my + 3), fill=C("#303030"))
        if (int(t * 4) + m) % 2 == 0:
            d.point((mx + 2, my - 2), fill=C("#b0b0b0"))
            d.point((mx + 3, my - 4), fill=C("#b0b0b0"))
    # three founders
    put_mark(img, "friend1", 84, 126, t, phase=0.4)
    put_mark(img, "friend2", 226, 126, t, phase=1.1)
    tlk = any(st <= t < char_times(tx, st)[1] for st, tx, _ in DIALOG["w1"])
    put_mark(img, "1986", 152, 126, t, talk=tlk and int(t * 12) % 2 == 0)
    return img


def w1_labels(img, t):
    d = ImageDraw.Draw(img)
    d.fontmode = "1"
    dk, lt = tuple(GB_PAL[0].astype(int)), tuple(GB_PAL[3].astype(int))
    if t > 1.4:
        for x, nm in ((84, "MARK M."), (152, "MARK D."), (226, "STEVE H.")):
            w_ = int(d.textlength(nm, font=FS))
            d.rectangle([x - w_ // 2 - 2, 62, x + w_ // 2 + 1, 70], fill=dk)
            d.text((x, 62), nm, font=FS, fill=lt, anchor="mt")


def sc_w2(t, dur):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    vgrad(d, 0, 96, C("#3cbcfc"), C("#a4e4fc"))
    for i, (cx, cy, sp) in enumerate([(40, 20, 6), (180, 34, 4), (280, 14, 5)]):
        x = (cx + t * sp) % (W + 60) - 30
        for dx, dy, r in ((0, 0, 7), (8, -3, 8), (16, 0, 6)):
            d.ellipse([x + dx - r, cy + dy - r // 2, x + dx + r, cy + dy + r // 2 + 2], fill=C("#fcfcfc"))
    # sea
    d.rectangle([0, 96, W, H], fill=C("#0058f8"))
    for k in range(40):
        wx = (k * 37 + int(t * 20)) % W
        wy = 100 + (k * 13) % 30
        d.line([(wx, wy), (wx + 4, wy)], fill=C("#3cbcfc"))
    # navy ship
    bob = 1 if math.sin(t * 3) > 0 else 0
    sx, sy = 196, 92 + bob
    d.polygon([(sx, sy), (sx + 110, sy), (sx + 100, sy + 12), (sx + 8, sy + 12)], fill=C("#7c7c7c"), outline=C("#303030"))
    d.rectangle([sx + 30, sy - 14, sx + 70, sy], fill=C("#bcbcbc"), outline=C("#303030"))
    d.rectangle([sx + 44, sy - 24, sx + 58, sy - 14], fill=C("#bcbcbc"), outline=C("#303030"))
    for k in range(4):
        d.rectangle([sx + 34 + k * 9, sy - 10, sx + 38 + k * 9, sy - 7], fill=C("#203050"))
    d.line([(sx + 51, sy - 24), (sx + 51, sy - 38)], fill=C("#303030"))
    d.ellipse([sx + 60, sy - 22, sx + 70, sy - 14], fill=C("#f8f8f8"), outline=C("#303030"))  # radome
    d.rectangle([sx + 52, sy - 38, sx + 60, sy - 34], fill=C("#e40058"))
    # UHF satellite
    satx, saty = 150 + 6 * math.sin(t * 0.7), 26
    # jet
    jx = -40 + ((t - 2) * 70) % (W + 120)
    d.polygon([(jx, 52), (jx + 16, 50), (jx + 20, 52), (jx + 16, 54)], fill=C("#7c7c7c"))
    d.polygon([(jx + 6, 52), (jx + 10, 46), (jx + 12, 52)], fill=C("#505050"))
    d.polygon([(jx + 6, 52), (jx + 10, 58), (jx + 12, 52)], fill=C("#505050"))
    # dock + DAMA terminal
    d.rectangle([0, 104, 128, H], fill=C("#bcbcbc"))
    d.line([(0, 104), (128, 104)], fill=C("#7c7c7c"))
    for x in range(0, 128, 16):
        d.line([(x, 104), (x, H)], fill=C("#a0a0a0"))
    d.rectangle([78, 58, 110, 104], fill=C("#505060"), outline=C("#101018"))
    for r_ in range(4):
        yy = 62 + r_ * 10
        d.rectangle([81, yy, 107, yy + 7], fill=C("#303038"))
        for k in range(4):
            on = (int(t * 6) + k * 3 + r_ * 5) % 7 < 4
            d.point((84 + k * 3, yy + 3), fill=C("#58f898") if on else C("#105020"))
        d.rectangle([98, yy + 2, 104, yy + 5], fill=C("#f8b800") if r_ == 1 else C("#7c7c7c"))
    small(d, (94, 51), "DAMA", C("#fcfcfc"), anchor="mt")
    # dish on dock
    d.line([(118, 104), (118, 84)], fill=C("#505060"), width=2)
    d.chord([106, 70, 130, 90], 150, 330, fill=C("#f8f8f8"), outline=C("#505060"))
    d = ImageDraw.Draw(img)
    beam(d, satx, saty + 6, 118, 78, t, col=C("#f8f878"))
    beam(d, satx, saty + 6, sx + 65, sy - 20, t, col=C("#f8f878"))
    beam(d, satx, saty + 6, jx + 10, 50, t, col=C("#f8f878"), dash=2, gap=7)
    draw_satellite(d, satx, saty, "uhf", t=t)
    # IPO
    L2 = DIALOG["w2"][1][0]
    hop = 0
    if t > L2:
        k = t - L2
        yb = int(min(0, -40 + k * 160))
        d.rectangle([8, 8 + yb, 140, 30 + yb], fill=C("#000000"), outline=C("#f8b800"))
        msg = "  NASDAQ: VSAT ▲  IPO 1996 ▲  NASDAQ: VSAT ▲  IPO 1996 ▲"
        off = int(k * 40) % 136
        tick = Image.new("RGB", (128, 10), (0, 0, 0))
        td = ImageDraw.Draw(tick)
        td.fontmode = "1"
        td.text((-off, 1), msg, font=F8, fill=C("#58f898"))
        img.paste(tick, (10, 14 + yb))
        for i in range(50):
            r_ = rng(i + 400)
            cx = r_.randint(0, W)
            cy = -10 + (k * (40 + r_.randint(0, 50)) + r_.randint(0, 40)) % 150
            cx += int(4 * math.sin(k * 5 + i))
            col = [C("#f83800"), C("#f8b800"), C("#58d854"), C("#3cbcfc"), C("#f878f8")][i % 5]
            d.rectangle([cx, cy, cx + 1, cy + 1], fill=col)
        if k < 0.8:
            hop = int(math.sin(k / 0.8 * math.pi) * 14)
    tlk = any(st <= t < char_times(tx, st)[1] for st, tx, _ in DIALOG["w2"])
    put_mark(img, "1990s", 46, 124, t, talk=tlk and int(t * 12) % 2 == 0, hop=hop)
    return img


def sc_w3(t, dur):
    liftoff = bars(3, 128)
    orbit_t = liftoff + 3.8
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img, "RGBA")
    d.fontmode = "1"
    if t < liftoff:
        # clean room
        vgrad(d, 0, 110, C("#dce6f0"), C("#f4f8fc"))
        for x in range(0, W, 40):
            d.line([(x, 0), (x, 110)], fill=C("#c4d0dc"))
        for x in range(20, W, 80):
            d.rectangle([x, 2, x + 40, 5], fill=C("#ffffff"), outline=C("#b0bcc8"))
        d.rectangle([0, 110, W, H], fill=C("#c8d2dc"))
        for x in range(0, W, 20):
            d.line([(x, 110), (x - 20, H)], fill=C("#b4c0cc"))
        # scaffold
        d.rectangle([168, 30, 172, 118], fill=C("#f0c020"))
        d.rectangle([268, 30, 272, 118], fill=C("#f0c020"))
        d.rectangle([168, 30, 272, 34], fill=C("#f0c020"))
        # the satellite, being built (sparkle as it "completes")
        prog = clamp01((t - 1.2) / 3.5)
        cx, cy = 220, 72
        d.rectangle([cx - 24, cy - 30, cx + 24, cy + 34], fill=C("#e0a830"), outline=C("#6a4010"))
        for k in range(cy - 28, cy + 34, 4):
            d.line([(cx - 23, k), (cx + 23, k)], fill=C("#f4c850"))
        d.rectangle([cx - 30, cy - 26, cx - 25, cy + 30], fill=C("#233c8c"), outline=C("#0c1640"))
        d.rectangle([cx + 25, cy - 26, cx + 30, cy + 30], fill=C("#233c8c"), outline=C("#0c1640"))
        d.ellipse([cx - 22, cy - 50, cx + 2, cy - 26], fill=C("#f0f0f4"), outline=C("#8a8a96"))
        if prog > 0.5:
            d.ellipse([cx - 2, cy - 48, cx + 20, cy - 28], fill=C("#f8f8fc"), outline=C("#8a8a96"))
        d.rectangle([cx - 10, cy + 34, cx + 10, cy + 40], fill=C("#909090"))
        for i in range(5):
            ph = (t * 1.3 + i * 0.37) % 1
            if ph < 0.35:
                sx_ = cx - 22 + (i * 23) % 44
                sy_ = cy - 20 + (i * 31) % 50
                sz = int(3 * math.sin(ph / 0.35 * math.pi))
                d.line([(sx_ - sz, sy_), (sx_ + sz, sy_)], fill=(255, 255, 255))
                d.line([(sx_, sy_ - sz), (sx_, sy_ + sz)], fill=(255, 255, 255))
        d.rectangle([236, 116, 300, 128], fill=C("#203050"))
        small(d, (268, 118), "VIASAT-1", C("#ffffff"), anchor="mt")
        tlk = any(st <= t < char_times(tx, st)[1] for st, tx, _ in DIALOG["w3"])
        put_mark(img, "bunny", 104, 126, t, talk=False)
        # countdown during bar 2
        b = 60 / 128
        cd = [(liftoff - 4 * b, "3"), (liftoff - 3 * b, "2"), (liftoff - 2 * b, "1"), (liftoff - b, "GO!")]
        for (ts_, s_) in cd:
            if ts_ <= t < ts_ + b:
                sz_ = F24 if s_ != "GO!" else F16
                text_shadow(d, (104, 40), s_, sz_, C("#f83800"), (255, 255, 255), anchor="mm", off=2)
    elif t < orbit_t:
        k = t - liftoff
        vgrad(d, 0, H, C("#1a1040"), C("#f08040"))
        stars(d, t, ymax=60, dens=0.4, big=False)
        # ocean horizon + pad
        d.rectangle([0, 118, W, H], fill=C("#3a2a3a"))
        d.rectangle([124, 104, 200, 118], fill=C("#505060"))
        d.rectangle([176, 20, 184, 104], fill=C("#8a3030"))
        for yy in range(24, 104, 8):
            d.line([(176, yy), (184, yy + 8)], fill=C("#5a1a1a"))
        rise = max(0.0, k - 0.3) ** 2 * 28
        shake = int(2 * math.sin(k * 60)) if k < 2.5 else 0
        rx, ry = 160 + shake, 30 - rise
        # smoke puffs (stateless)
        for i in range(70):
            r_ = rng(i + 900)
            born = i * 0.035
            if k < born:
                continue
            age = k - born
            ang = r_.rand() * math.pi
            spd = 30 + r_.rand() * 40
            px_ = 162 + math.cos(ang) * spd * min(age, 1.6) * (1.6 if r_.rand() > 0.5 else -1.6) * 0.6
            py_ = 110 + min(8, age * 10) - age * 2
            rad = 5 + age * 7
            g = int(200 + 40 * r_.rand())
            d.ellipse([px_ - rad, py_ - rad * 0.6, px_ + rad, py_ + rad * 0.6], fill=(g, g, g, 200))
        # rocket
        if ry > -80:
            fl = 6 + int(4 * abs(math.sin(k * 40)))
            d.polygon([(rx - 5, ry + 70), (rx + 5, ry + 70), (rx, ry + 70 + fl * 2)], fill=C("#fff080"))
            d.polygon([(rx - 3, ry + 70), (rx + 3, ry + 70), (rx, ry + 70 + fl)], fill=C("#ffffff"))
            d.rectangle([rx - 6, ry + 8, rx + 6, ry + 70], fill=C("#f0f0f4"), outline=C("#505060"))
            d.polygon([(rx - 6, ry + 8), (rx + 6, ry + 8), (rx, ry - 6)], fill=C("#e04030"), outline=C("#505060"))
            d.polygon([(rx - 6, ry + 56), (rx - 12, ry + 72), (rx - 6, ry + 70)], fill=C("#e04030"))
            d.polygon([(rx + 6, ry + 56), (rx + 12, ry + 72), (rx + 6, ry + 70)], fill=C("#e04030"))
            d.rectangle([rx - 6, ry + 30, rx + 6, ry + 33], fill=C("#303848"))
        # Mark cheering (regular clothes now)
        hop_ = int(abs(math.sin(k * 5)) * 8)
        put_mark(img, "2011", 56, 128, t, hop=hop_)
    else:
        k = t - orbit_t
        vgrad(d, 0, H, C("#02030a"), C("#0a1030"))
        stars(d, t)
        img = draw_sphere(img, 160, 300, 190, 2.2 + t * 0.03, light=(-0.4, -0.8, 0.5))
        d = ImageDraw.Draw(img, "RGBA")
        op = ease(k / 1.2)
        sx_, sy_ = 160, 50 + (1 - ease(k / 0.8)) * -40
        if op > 0.95:
            for bx in (90, 140, 190, 240):
                beam(d, sx_, sy_ + 10, bx, 124, t, col=C("#80e0ff"), dash=2, gap=5)
        draw_satellite(d, sx_, sy_, "vs1", s=1.4, open_=op, t=t)
        if op > 0.95:
            text_shadow(d, (160, 12), "HIGHEST CAPACITY IN THE WORLD!", F8, C("#f8d878"), anchor="mt")
            small(d, (160, 24), "(AT LAUNCH, 2011)", C("#bcbcbc"), anchor="mt")
    return img


def panel_home(p, t):
    d = ImageDraw.Draw(p)
    vgrad(d, 0, 48, C("#101a48"), C("#3a3a80"))
    for i in range(10):
        d.point(((i * 37) % 150, (i * 11) % 22), fill=(255, 255, 220))
    d.rectangle([0, 40, 150, 48], fill=C("#2a5a2a"))
    d.polygon([(70, 22), (96, 8), (122, 22)], fill=C("#a03030"))
    d.rectangle([74, 22, 118, 42], fill=C("#e8d8b0"))
    fl = 180 + int(60 * abs(math.sin(t * 7)))
    d.rectangle([80, 27, 92, 36], fill=(fl, fl, 255))
    d.rectangle([100, 30, 108, 42], fill=C("#6a3a1a"))
    wifi(d, 96, 6, t, C("#80e0ff"))
    put_mark(p, "home", 36, 45, t, scale=1)


def panel_air(p, t):
    d = ImageDraw.Draw(p)
    vgrad(d, 0, 48, C("#4aa8f0"), C("#b8e4ff"))
    for i in range(5):
        x = (i * 45 - t * 70) % 190 - 20
        y = 8 + (i * 17) % 34
        d.ellipse([x, y, x + 20, y + 6], fill=(255, 255, 255))
    y = 24 + int(2 * math.sin(t * 3))
    d.ellipse([70, y - 4, 136, y + 4], fill=C("#f8f8f8"), outline=C("#6a7a8a"))
    d.polygon([(124, y - 3), (134, y - 14), (138, y - 14), (134, y - 2)], fill=C("#1f4aa0"))
    d.polygon([(94, y), (112, y + 12), (116, y + 12), (106, y)], fill=C("#b0bcc8"))
    for k in range(6):
        d.point((82 + k * 6, y - 1), fill=C("#3a5a8a"))
    d.polygon([(70, y), (76, y - 3), (76, y + 2)], fill=C("#3a5a8a"))
    wifi(d, 104, y - 8, t, C("#1f4aa0"))
    put_mark(p, "pilot", 30, 46, t, scale=1, phase=0.3)


def panel_sea(p, t):
    d = ImageDraw.Draw(p)
    vgrad(d, 0, 30, C("#80c8f8"), C("#d8f0ff"))
    d.rectangle([0, 30, 150, 48], fill=C("#1a5ab0"))
    for k in range(14):
        x = (k * 23 + int(t * 15)) % 150
        d.line([(x, 34 + (k * 5) % 12), (x + 3, 34 + (k * 5) % 12)], fill=C("#80c8f8"))
    b = 1 if math.sin(t * 2.5) > 0 else 0
    d.polygon([(72, 26 + b), (140, 26 + b), (132, 36 + b), (78, 36 + b)], fill=C("#c03030"))
    d.rectangle([80, 18 + b, 130, 26 + b], fill=C("#f8f8f8"))
    for k in range(5):
        d.rectangle([84 + k * 9, 20 + b, 88 + k * 9, 23 + b], fill=C("#3a6aa0"))
    d.rectangle([118, 10 + b, 124, 18 + b], fill=C("#303030"))
    d.ellipse([104, 11 + b, 112, 18 + b], fill=C("#f8f8f8"), outline=C("#808080"))
    wifi(d, 108, 8 + b, t, C("#1f4aa0"))
    put_mark(p, "captain", 32, 46, t, scale=1, phase=0.7)


def panel_def(p, t):
    d = ImageDraw.Draw(p)
    vgrad(d, 0, 34, C("#f0c890"), C("#f8e8c8"))
    d.rectangle([0, 34, 150, 48], fill=C("#b89060"))
    d.polygon([(0, 34), (40, 24), (90, 34)], fill=C("#c8a070"))
    d.rectangle([78, 28, 124, 40], fill=C("#5a6a3a"), outline=C("#2a3018"))
    d.rectangle([108, 22, 124, 28], fill=C("#5a6a3a"), outline=C("#2a3018"))
    d.ellipse([82, 36, 92, 46], fill=C("#202020"))
    d.ellipse([110, 36, 120, 46], fill=C("#202020"))
    d.chord([86, 12, 104, 30], 150, 330, fill=C("#e8e8e8"), outline=C("#505050"))
    d.line([(95, 22), (95, 28)], fill=C("#505050"))
    jx = (t * 60) % 200 - 30
    d.polygon([(jx, 8), (jx + 12, 7), (jx + 15, 8), (jx + 12, 9)], fill=C("#606878"))
    d.polygon([(jx + 4, 8), (jx + 8, 3), (jx + 9, 8)], fill=C("#404858"))
    wifi(d, 95, 10, t, C("#304020"))
    put_mark(p, "defense", 32, 46, t, scale=1, phase=0.2)


PANELS = [("HOMES", panel_home, "homes", "home", (8, 20)),
          ("AIRPLANES", panel_air, "airplanes", "pilot", (162, 20)),
          ("SHIPS", panel_sea, "ships", "captain", (8, 76)),
          ("THOSE WHO SERVE", panel_def, "those", "defense", (162, 76))]


def w4_reveals():
    return [word_time("w4", 0, w_) for (_, _, w_, _, _) in PANELS]


def sc_w4(t, dur):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img, "RGBA")
    d.fontmode = "1"
    vgrad(d, 0, H, C("#060a24"), C("#182a60"))
    stars(d, t, dens=0.8)
    reveals = w4_reveals()
    satx, saty = 160, 12
    for (lbl, fn, _, _, (px_, py_)), rt in zip(PANELS, reveals):
        if t < rt:
            continue
        k = t - rt
        s = ease(k / 0.25)
        p = Image.new("RGB", (150, 48))
        fn(p, t)
        pw, ph_ = max(1, int(150 * s)), max(1, int(48 * s))
        if s < 1:
            p = p.resize((pw, ph_), Image.NEAREST)
        cx, cy = px_ + 75, py_ + 24
        img.paste(p, (cx - pw // 2, cy - ph_ // 2))
        d = ImageDraw.Draw(img, "RGBA")
        d.fontmode = "1"
        d.rectangle([cx - pw // 2 - 1, cy - ph_ // 2 - 1, cx + pw // 2, cy + ph_ // 2], outline=C("#f8f8f8"))
        if s >= 1:
            tw = int(d.textlength(lbl, font=FS))
            d.rectangle([px_, py_ - 1, px_ + tw + 4, py_ + 7], fill=(0, 0, 0, 200))
            d.text((px_ + 2, py_ - 4), lbl, font=FS, fill=C("#f8f8f8"))
            beam(d, satx, saty + 6, cx, py_ - 1 if py_ < 60 else py_ - 1, t, col=C("#80e0ff"), dash=2, gap=6)
    draw_satellite(d, satx, saty, "modern", s=0.7, t=t)
    # Inmarsat joins the party
    L2 = DIALOG["w4"][1][0]
    if t > L2 + 0.8:
        k = t - L2 - 0.8
        x = int(W - k * 260) if k < 0.9 else int(W - 0.9 * 260)
        x = max(x, 24)
        d.rectangle([x, 126 - 14, x + 272, 126 - 2], fill=(20, 10, 60, 230), outline=C("#f8d878"))
        d.text((x + 6, 126 - 12), "★ NEW PARTY MEMBER: INMARSAT ★", font=F8, fill=C("#f8d878"))
    return img


def w4_portrait(t):
    var = "today"
    for (lbl, fn, w_, v, _), rt in zip(PANELS, w4_reveals()):
        if t >= rt:
            var = v
    if t >= DIALOG["w4"][1][0]:
        var = "today"
    return var


def w5_times():
    b = 60 / 90
    return [3 * b, 5 * b, 7 * b]


def sc_w5(t, dur):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img, "RGBA")
    d.fontmode = "1"
    # deep space w/ nebula
    arr = np.zeros((H, W, 3), np.float32)
    neb = NEBULA
    arr[:] = np.array(C("#03050f"), np.float32)
    arr += neb[..., None] * np.array([60, 30, 110], np.float32)
    arr += (NEBULA2[..., None]) * np.array([10, 60, 90], np.float32)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img, "RGBA")
    stars(d, t)
    ex, ey, er = 160, 64, 46
    img = draw_sphere(img, ex, ey, er, 3.0 + t * 0.18, light=(-0.7, -0.3, 0.65), cloud_rot=t * 0.05)
    d = ImageDraw.Draw(img, "RGBA")
    d.fontmode = "1"
    # orbit ring
    for i in range(120):
        a = i / 120 * 2 * math.pi + t * 0.1
        x = ex + math.cos(a) * 112
        y = ey + math.sin(a) * 38
        if i % 3 == 0 and not (abs(x - ex) < er and y > ey - 10 and math.sin(a) < 0):
            d.point((x, y), fill=(120, 170, 255, 120))
    acts = w5_times()
    sats = [(48, 58, "F1"), (160, 12, "F2"), (272, 58, "F3")]
    lit_n = 0
    for (sx_, sy_, nm), at in zip(sats, acts):
        on = t >= at
        if on:
            lit_n += 1
            k = t - at
            glow = max(0.0, 1 - k * 1.5)
            # coverage cone
            tx_ = ex + (sx_ - ex) * 0.55
            ty_ = ey + (sy_ - ey) * 0.55
            a = int(70 + 30 * math.sin(t * 3))
            d.polygon([(sx_, sy_ + 4), (tx_ - 16, ty_ + 6), (tx_ + 16, ty_ + 6)], fill=(80, 220, 255, a // 2))
            beam(d, sx_, sy_ + 4, tx_, ty_ + 4, t, col=(160, 240, 255), dash=2, gap=5, speed=40)
            if glow > 0:
                rr = int(20 * (1 - glow) + 6)
                d.ellipse([sx_ - rr, sy_ - rr, sx_ + rr, sy_ + rr], outline=(255, 255, 255, int(255 * glow)))
        draw_satellite(d, sx_, sy_, "modern", s=0.8, lit=1.0 if on else 0.0, t=t)
        text_shadow(d, (sx_, sy_ + 14), "VS3 " + nm, FS, C("#ffffff") if on else C("#5a6a8a"), anchor="mt")
    # HUD capacity meter
    d.rectangle([6, 4, 100, 22], fill=(0, 0, 0, 160), outline=(64, 208, 255, 255))
    small(d, (10, 3), "NETWORK CAPACITY", C("#40d0ff"))
    target = lit_n / 3
    for i in range(3):
        on = i < lit_n
        d.rectangle([10 + i * 30, 14, 10 + i * 30 + 26, 19], fill=C("#40d0ff") if on else (30, 40, 70))
    if lit_n == 3:
        small(d, (104, 8), "3+ TBPS!", C("#f8d878"))
    # control room console foreground + Mark
    d.rectangle([0, 112, W, H], fill=C("#0c1020"))
    d.polygon([(0, 112), (W, 112), (W, 118), (0, 118)], fill=C("#1a2440"))
    for k in range(20):
        on = (int(t * 5) + k * 7) % 9 < 5
        d.point((12 + k * 15, 115), fill=(80, 255, 160) if on else (20, 60, 40))
    L2 = DIALOG["w5"][1][0]
    tlk = any(st <= t < char_times(tx, st)[1] for st, tx, _ in DIALOG["w5"])
    put_mark(img, "today", 30, 128, t, talk=tlk and int(t * 12) % 2 == 0)
    if t > L2 + 0.3:
        k = t - L2 - 0.3
        cy = 104 - int(ease(k / 0.3) * 6)
        paste(img, CAT.resize((18, 18), Image.NEAREST), 262, cy - 10)
        if int(t * 3) % 2 == 0:
            paste(img, HEART, 284, cy - 16)
    return img


NEBULA = value_noise(W, H, 3, 77, 4)
NEBULA = np.clip((NEBULA - 0.45) * 2.2, 0, 1) ** 1.5
NEBULA2 = np.clip((value_noise(W, H, 4, 78, 4) - 0.5) * 2.2, 0, 1) ** 2


def sc_w6(t, dur):
    arr = np.zeros((H, W, 3), np.float32)
    arr[:] = np.array(C("#020308"), np.float32)
    arr += NEBULA[..., None] * np.array([70, 20, 90], np.float32) * 0.7
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img, "RGBA")
    d.fontmode = "1"
    stars(d, t)
    ex, ey = 250, 40
    img = draw_sphere(img, ex, ey, 24, 1.0 + t * 0.2, light=(-0.8, 0.1, 0.6), cloud_rot=t * 0.05)
    d = ImageDraw.Draw(img, "RGBA")
    d.fontmode = "1"
    L1, txt1, _ = DIALOG["w6"][0]
    tq = word_time("w6", 0, "quantum")
    tc = word_time("w6", 0, "cyber")
    ta = word_time("w6", 0, "self-flying")
    tm = DIALOG["w6"][1][0] + 1.0
    # quantum-safe cubesats with entangled link
    if t > tq:
        k = t - tq
        base_x = -40 + k * 34
        pts = []
        for i in range(3):
            x = base_x + i * 26
            y = 30 + i * 8 + 3 * math.sin(t * 2 + i)
            pts.append((x, y))
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            n = 16
            for j in range(n):
                a = j / n
                xx = x1 + (x2 - x1) * a
                yy = y1 + (y2 - y1) * a + 3 * math.sin(a * math.pi * 4 + t * 10)
                d.point((xx, yy), fill=(200, 140, 255))
        for (x, y) in pts:
            draw_satellite(d, x, y, "cube", t=t)
            paste(img, LOCK, x - 2, y - 13)
        d = ImageDraw.Draw(img, "RGBA")
        d.fontmode = "1"
        if k < 3.2:
            small(d, (pts[1][0], pts[1][1] + 8), "QUANTUM-SAFE", C("#d8b0ff"), anchor="mt")
    # cyber shield around earth
    if t > tc:
        k = t - tc
        s = ease(k / 0.4)
        r = 24 + 7 * s
        hexp = [(ex + r * math.cos(math.radians(60 * i + 30)), ey + r * math.sin(math.radians(60 * i + 30))) for i in range(6)]
        a = int(160 + 90 * math.sin(t * 5))
        d.polygon(hexp, outline=(80, 255, 180, a))
        if k < 3.0:
            small(d, (ex, ey + 34), "CYBER SHIELD", C("#80ffc0"), anchor="mt")
    # autonomous air taxi
    if t > ta:
        k = t - ta
        x = -30 + k * 60
        y = 70 + 4 * math.sin(k * 3)
        d.rectangle([x - 8, y - 2, x + 8, y + 3], fill=C("#f0f0f4"), outline=C("#505a6a"))
        d.rectangle([x - 3, y - 1, x + 4, y + 1], fill=C("#60c0ff"))
        for rx in (x - 12, x + 12):
            d.line([(rx, y - 4), (rx, y - 1)], fill=C("#a0a8b8"))
            sp = int(6 * abs(math.sin(t * 30)))
            d.line([(rx - sp, y - 4), (rx + sp, y - 4)], fill=C("#e0e8f0"))
        d.line([(x - 12, y - 1), (x + 12, y - 1)], fill=C("#a0a8b8"))
        if k < 3.0:
            small(d, (x, y + 7), "SELF-FLYING", C("#a0e0ff"), anchor="mt")
    # moon surface
    d.pieslice([-200, 96, 520, 420], 180, 360, fill=C("#9a9aa4"))
    for (cx, cy, rr) in [(40, 118, 8), (120, 108, 5), (200, 124, 10), (290, 114, 6), (160, 128, 4), (70, 128, 5)]:
        d.ellipse([cx - rr, cy - rr // 2, cx + rr, cy + rr // 2], fill=C("#7a7a86"))
        d.arc([cx - rr, cy - rr // 2, cx + rr, cy + rr // 2], 0, 180, fill=C("#b4b4be"))
    # lunar base
    bx, by = 236, 104
    d.pieslice([bx - 16, by - 14, bx + 16, by + 14], 180, 360, fill=C("#e0e4ec"), outline=C("#707888"))
    d.rectangle([bx + 14, by - 6, bx + 30, by], fill=C("#d0d4dc"), outline=C("#707888"))
    online = t > tm
    for k in range(3):
        d.point((bx - 8 + k * 6, by - 4), fill=(255, 220, 90) if online else (80, 80, 90))
    d.line([(bx + 24, by - 6), (bx + 24, by - 18)], fill=C("#707888"))
    d.chord([bx + 16, by - 28, bx + 32, by - 14], 150, 330, fill=C("#f0f0f4"), outline=C("#707888"))
    if online:
        beam(d, bx + 24, by - 22, ex - 10, ey + 18, t, col=(255, 220, 120), dash=2, gap=5)
        # lunar relay satellite orbiting
        a = t * 0.9
        rx_, ry_ = 150 + 110 * math.cos(a), 80 + 20 * math.sin(a)
        if math.sin(a) < 0.6:
            draw_satellite(d, rx_, ry_, "cube", t=t)
            beam(d, rx_, ry_, bx - 4, by - 12, t, col=(255, 220, 120), dash=2, gap=6)
        # moon signal bars HUD
        d.rectangle([6, 6, 84, 20], fill=(0, 0, 0, 160), outline=(192, 128, 255, 255))
        small(d, (9, 7), "MOON", C("#e0c0ff"))
        for i in range(4):
            if t - tm > i * 0.25:
                d.rectangle([42 + i * 9, 16 - i * 2 - 3, 47 + i * 9, 16], fill=C("#c080ff"))
    # astronaut Mark, low-gravity hopping, plants a wi-fi flag
    hop = abs(math.sin(t * 1.6)) * 14
    mx = 70
    if online:
        fx = 108
        d.line([(fx, 118), (fx, 94)], fill=C("#c0c0c8"))
        d.rectangle([fx + 1, 94, fx + 17, 105], fill=C("#1f3a68"))
        wifi(d, fx + 9, 103, t, C("#f8f8f8"))
    tlk = any(st <= t < char_times(tx, st)[1] for st, tx, _ in DIALOG["w6"])
    put_mark(img, "astro", mx, 124, t, bob=False, hop=hop)
    return img


def sc_end(t, dur):
    fin = bars(2, 100)
    if t < fin:
        img = Image.new("RGB", (W, H))
        d = ImageDraw.Draw(img, "RGBA")
        d.fontmode = "1"
        vgrad(d, 0, H, C("#050818"), C("#1a2250"))
        stars(d, t)
        img = draw_sphere(img, 60, 34, 14, 0.5, kind="moon", light=(0.6, -0.2, 0.7))
        d = ImageDraw.Draw(img, "RGBA")
        d.fontmode = "1"
        if int(t * 3) % 2 == 0:
            d.point((56, 30), fill=(255, 220, 90))
        for i in range(7):
            a = (t * (0.08 + i * 0.01) + i * 0.9) % 1
            x = -20 + a * (W + 40)
            y = 20 + i * 11 + 6 * math.sin(a * 3)
            draw_satellite(d, x, y, "cube" if i % 2 else "uhf", t=t)
        # ocean + hills + palm (Carlsbad at night)
        d.rectangle([0, 104, W, H], fill=C("#0a1838"))
        for k in range(30):
            x = (k * 29 + int(t * 8)) % W
            d.line([(x, 106 + (k * 7) % 20), (x + 3, 106 + (k * 7) % 20)], fill=(60, 90, 150))
        d.polygon([(90, 104), (180, 90), (320, 96), (320, 132), (90, 132)], fill=C("#101a14"))
        # the house, spare bedroom window glowing
        hx, hy = 190, 70
        d.polygon([(hx - 4, hy + 16), (hx + 30, hy - 4), (hx + 64, hy + 16)], fill=C("#3a1a1a"))
        d.rectangle([hx, hy + 16, hx + 60, hy + 50], fill=C("#2a2a38"))
        d.rectangle([hx + 40, hy + 30, hx + 50, hy + 50], fill=C("#1a1a24"))
        # window shows a tiny Game-Boy-green 1986 scene: Mark at the desk
        win = Image.new("RGB", (22, 16), C("#e8d8b8"))
        wd = ImageDraw.Draw(win)
        wd.rectangle([0, 11, 22, 16], fill=C("#8a5a34"))
        wd.rectangle([12, 4, 20, 10], fill=C("#e0d8c0"))
        spr = mark_sprite("1986", scale=1).crop((0, 0, 20, 22)).resize((10, 11), Image.NEAREST)
        win.paste(spr, (1, 4), spr)
        win = Image.fromarray(np.ascontiguousarray(era_colour(np.asarray(win), "gb")))
        img.paste(win, (hx + 8, hy + 22))
        d.rectangle([hx + 7, hy + 21, hx + 30, hy + 38], outline=C("#f8e090"))
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.ellipse([hx - 6, hy + 10, hx + 44, hy + 50], fill=(255, 220, 120, 26))
        img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
        d = ImageDraw.Draw(img)
        d.line([(160, 104), (158, 72)], fill=C("#2a1a10"), width=2)
        for ang in (-150, -110, -60, -20, 20):
            a = math.radians(ang + 4 * math.sin(t * 2))
            d.line([(158, 72), (158 + 13 * math.cos(a), 72 + 7 * math.sin(a) + 5)], fill=C("#1a3a1a"), width=2)
        return img
    k = t - fin
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img, "RGBA")
    d.fontmode = "1"
    vgrad(d, 0, H, C("#02030a"), C("#0c1438"))
    stars(d, t)
    img = draw_sphere(img, 160, 372, 208, 1.2 + t * 0.05, light=(0.0, -1.0, 0.4))
    d = ImageDraw.Draw(img, "RGBA")
    d.fontmode = "1"
    for i, ch in enumerate("VIASAT"):
        xx = 160 - 96 + i * 32 + 16
        s_ = ease((k - i * 0.06) / 0.3)
        yy = int(lerp(-30, 40, s_))
        d.text((xx + 2, yy + 3), ch, font=F32, fill=(0, 0, 60), anchor="mm")
        d.text((xx, yy), ch, font=F32, fill=C("#ffffff"), anchor="mm")
    if k > 0.6:
        text_shadow(d, (160, 72), "1986 → ∞", F8, C("#f8d878"), anchor="mm")
    if k > 1.2:
        text_shadow(d, (160, 96), "CONTINUE?", F8, C("#ffffff"), anchor="mm")
        sel = k > 2.4
        yes_c = C("#f8d878") if (not sel or int(k * 12) % 2 == 0) else C("#ffffff")
        text_shadow(d, (136, 112), "YES", F8, yes_c, anchor="lm")
        text_shadow(d, (176, 112), "NO", F8, C("#8a8aa0"), anchor="lm")
        if int(k * 4) % 2 == 0 or sel:
            d.text((124, 112), "▶", font=F8, fill=C("#ffffff"), anchor="lm")
    if k > 2.6:
        text_shadow(d, (160, 132), "THANKS FOR PLAYING!", F8, C("#80e0ff"), anchor="mm")
        small(d, (160, 146), "CREATED WITH CLAUDE", C("#8a9ac0"), anchor="mm")
        small(d, (160, 155), "EVERY PIXEL AND NOTE GENERATED IN CODE", C("#6a7aa0"), anchor="mm")
    return img


SCENE_FN = dict(title=sc_title, w1=sc_w1, w2=sc_w2, w3=sc_w3, w4=sc_w4, w5=sc_w5, w6=sc_w6, end=sc_end)


def portrait_for(scene, line_var, t):
    if scene == "w4":
        return w4_portrait(t)
    if scene == "w3" and line_var == "2011" and t < bars(3, 128):
        return "bunny"
    return line_var


def render_frame(T):
    for sc in SCENES:
        if sc["start"] <= T < sc["start"] + sc["dur"] or sc is SCENES[-1]:
            break
    t = T - sc["start"]
    name, era, dur = sc["name"], sc["era"], sc["dur"]
    if "card" in sc and t < CARD_T:
        img = draw_card(sc, t)
    else:
        img = SCENE_FN[name](t, dur)
        img = post(img, era)
        if name == "w1":
            w1_labels(img, t)
        # dialog
        active = None
        for i, (st, tx, var) in enumerate(DIALOG.get(name, [])):
            if t >= st:
                active = (st, tx, var)
        if name == "end" and t >= bars(2, 100):
            active = None
        if active:
            st, tx, var = active
            ts, tend = char_times(tx, st)
            n = sum(1 for x in ts if x <= t)
            talking = t < tend and int(t * 12) % 2 == 0
            draw_dialog(img, era, tx, n, portrait_for(name, var, t), talking, t, future=(name == "w6"))
        cs = CARD_T if "card" in sc else 0.0
        if name == "end":
            if t > dur - 1.0:
                img = Image.blend(img, Image.new("RGB", (W, H)), ease((t - (dur - 1.0)) / 1.0))
            elif t < 0.3:
                img = iris(img, t, dur, start_t=0.0)
        else:
            img = iris(img, t, dur, start_t=cs)
    return img


# ════════════════════════════════════════════════════════════════════
#  AUDIO: a tiny chiptune synth + tracker
# ════════════════════════════════════════════════════════════════════
def mf(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def adsr(n, a=0.005, d=0.08, s=0.6, r=0.05, sr=SR):
    env = np.ones(n, np.float32) * s
    na, nd, nr = int(a * sr), int(d * sr), int(r * sr)
    na = min(na, n)
    env[:na] = np.linspace(0, 1, na, endpoint=False)
    e2 = min(n, na + nd)
    env[na:e2] = np.linspace(1, s, e2 - na, endpoint=False)
    if nr > 0 and n > nr:
        env[-nr:] *= np.linspace(1, 0, nr)
    return env


def onepole(x, a):
    return lfilter([a], [1, a - 1], x).astype(np.float32)


def osc(kind, f, n, t0=0.0, vib=0.0, duty=0.5):
    t = np.arange(n, dtype=np.float32) / SR
    if vib:
        f = f * (1 + vib * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.12) * 6, 0, 1))
        ph = np.cumsum(f / SR)
    else:
        ph = f * t
    ph = ph % 1.0
    if kind == "pulse":
        return np.where(ph < duty, 1.0, -1.0).astype(np.float32)
    if kind == "tri":
        tri = 4 * np.abs(ph - 0.5) - 1
        return (np.round(tri * 7.5) / 7.5).astype(np.float32)  # NES 4-bit stepping
    if kind == "saw":
        return (2 * ph - 1).astype(np.float32)
    if kind == "sine":
        return np.sin(2 * np.pi * ph).astype(np.float32)
    raise ValueError(kind)


def inst(name, m, dur):
    n = max(1, int(dur * SR))
    f = mf(m)
    t = np.arange(n, dtype=np.float32) / SR
    if name == "gb_lead":
        return osc("pulse", f, n, vib=0.004, duty=0.125) * adsr(n, 0.004, 0.1, 0.55, 0.04) * 0.9
    if name == "gb_bass":
        return osc("pulse", f, n, duty=0.5) * adsr(n, 0.003, 0.05, 0.5, 0.02) * 0.6
    if name == "gb_wave":
        return osc("tri", f, n) * adsr(n, 0.003, 0.2, 0.3, 0.03)
    if name == "nes_lead":
        return osc("pulse", f, n, vib=0.005, duty=0.25) * adsr(n, 0.003, 0.12, 0.6, 0.03)
    if name == "nes_harm":
        return osc("pulse", f, n, duty=0.5) * adsr(n, 0.003, 0.12, 0.45, 0.03) * 0.7
    if name == "nes_tri":
        return osc("tri", f, n) * adsr(n, 0.002, 0.02, 1.0, 0.01)
    if name == "snes_lead":
        x = osc("saw", f, n, vib=0.006) * 0.6 + osc("pulse", f * 1.003, n, duty=0.3) * 0.4
        return onepole(x, 0.25) * adsr(n, 0.01, 0.15, 0.7, 0.06) * 1.4
    if name == "brass":
        x = osc("saw", f, n, vib=0.005) + osc("saw", f * 1.005, n) * 0.7 + osc("saw", f * 0.995, n) * 0.7
        cut = np.clip(0.05 + 0.3 * np.minimum(t / 0.12, 1), 0, 1)
        y = np.zeros(n, np.float32)
        s = 0.0
        # time-varying one-pole (swell)
        for i in range(0, n, 256):
            seg = x[i:i + 256]
            a = float(cut[i])
            out = lfilter([a], [1, a - 1], seg, zi=[s * (1 - a)])[0]
            y[i:i + 256] = out
            s = out[-1] if len(out) else s
        return y * adsr(n, 0.05, 0.2, 0.8, 0.12) * 0.7
    if name == "pad":
        x = sum(osc("saw", f * dt, n) for dt in (1.0, 1.006, 0.994, 2.003)) / 4
        return onepole(x, 0.06) * adsr(n, 0.35, 0.3, 0.8, 0.4) * 1.6
    if name == "pluck":
        x = osc("pulse", f, n, duty=0.25) * 0.5 + osc("saw", f, n) * 0.5
        return onepole(x, 0.35) * np.exp(-t * 9).astype(np.float32)
    if name == "bell":
        mod = np.sin(2 * np.pi * f * 3.5 * t) * 2.2 * np.exp(-t * 3)
        return (np.sin(2 * np.pi * f * t + mod) * np.exp(-t * 2.2)).astype(np.float32) * 0.8
    if name == "musicbox":
        return (np.sin(2 * np.pi * f * t) * 0.7 + np.sin(2 * np.pi * f * 2 * t) * 0.3).astype(np.float32) * np.exp(-t * 4)
    if name == "sine_lead":
        return osc("sine", f, n, vib=0.006) * adsr(n, 0.03, 0.2, 0.7, 0.1) + osc("tri", f * 2, n) * 0.08 * adsr(n, 0.03, 0.2, 0.7, 0.1)
    if name == "bass":
        x = osc("sine", f, n) * 0.8 + osc("pulse", f, n, duty=0.5) * 0.25
        return onepole(x, 0.2) * adsr(n, 0.004, 0.1, 0.7, 0.03) * 1.3
    raise ValueError(name)


def drum(kind, sr=SR):
    r = rng(hash(kind) % 1000)
    if kind == "kick":
        n = int(0.25 * sr)
        t = np.arange(n) / sr
        f = 50 + 120 * np.exp(-t * 30)
        return (np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 12)).astype(np.float32) * 1.2
    if kind == "snare":
        n = int(0.18 * sr)
        t = np.arange(n) / sr
        return ((r.rand(n) * 2 - 1) * 0.7 * np.exp(-t * 20) + np.sin(2 * np.pi * 190 * t) * 0.4 * np.exp(-t * 30)).astype(np.float32)
    if kind == "hat":
        n = int(0.05 * sr)
        t = np.arange(n) / sr
        x = r.rand(n) * 2 - 1
        x = x - onepole(x.astype(np.float32), 0.3)
        return (x * np.exp(-t * 70)).astype(np.float32) * 0.5
    if kind == "nnoise":  # NES-ish snare: stepped noise
        n = int(0.12 * sr)
        t = np.arange(n) / sr
        x = np.repeat(r.rand(n // 8 + 1) * 2 - 1, 8)[:n]
        return (x * np.exp(-t * 28)).astype(np.float32) * 0.8
    if kind == "nhat":
        n = int(0.04 * sr)
        t = np.arange(n) / sr
        x = np.repeat(r.rand(n // 2 + 1) * 2 - 1, 2)[:n]
        return (x * np.exp(-t * 90)).astype(np.float32) * 0.5
    if kind == "timp":
        n = int(1.2 * sr)
        t = np.arange(n) / sr
        f = 65 * (1 + 0.1 * np.exp(-t * 8))
        return (np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 3) + (r.rand(n) * 2 - 1) * 0.15 * np.exp(-t * 25)).astype(np.float32)
    if kind == "crash":
        n = int(2.0 * sr)
        t = np.arange(n) / sr
        x = (r.rand(n) * 2 - 1).astype(np.float32)
        x = x - onepole(x, 0.2)
        return x * np.exp(-t * 2.2).astype(np.float32) * 0.45
    raise ValueError(kind)


class Sec:
    def __init__(self, dur, bpm):
        self.bpm, self.dur = bpm, dur
        self.beat = 60.0 / bpm
        self.buf = np.zeros(int((dur + 3.0) * SR), np.float32)

    def add(self, x, t, vol=1.0):
        i = int(t * SR)
        if i < 0 or i >= len(self.buf):
            return
        j = min(len(self.buf), i + len(x))
        self.buf[i:j] += x[:j - i] * vol

    def note(self, name, m, beat, beats, vol=0.2, gate=0.92):
        if beat * self.beat >= self.dur:
            return
        self.add(inst(name, m, beats * self.beat * gate), beat * self.beat, vol)

    def hit(self, kind, beat, vol=0.3):
        if beat * self.beat < self.dur:
            self.add(drum(kind), beat * self.beat, vol)


# The Viasat theme: 4 bars of (midi, eighths)
THEME = [
    [(76, 2), (79, 2), (84, 2), (83, 1), (79, 1)],
    [(81, 3), (76, 1), (72, 2), (76, 2)],
    [(77, 2), (81, 2), (84, 2), (86, 1), (84, 1)],
    [(83, 3), (79, 1), (74, 2), (79, 2)],
]
THEME_END = [(83, 1), (84, 1), (86, 2), (84, 4)]
CH = dict(C=[60, 64, 67], Am=[57, 60, 64], F=[53, 57, 60], G=[55, 59, 62], Cmaj7=[60, 64, 67, 71],
          DC=[60, 62, 66, 69], Am9=[57, 60, 64, 67, 71], Fmaj7=[53, 57, 60, 64], Gsus=[55, 60, 62])
PROG = ["C", "Am", "F", "G"]


def melody(sec, bar_notes, bar, name, vol, tr=0, harm=None, hvol=0.0, hname=None, chord=None):
    b = bar * 4
    for (m, l) in bar_notes:
        sec.note(name, m + tr, b, l / 2, vol)
        if harm and chord:
            cands = [c + 12 * o for c in CH[chord] for o in range(-1, 3) if c + 12 * o < m - 2]
            if cands:
                sec.note(hname, max(cands) + tr, b, l / 2, hvol)
        b += l / 2


def music():
    out = np.zeros(int((TOTAL + 4) * SR), np.float32)

    def place(sec, start, wet=0.0, echo=0.0, ir_len=1.2, gain=1.0):
        x = sec.buf * gain
        if echo:
            d = int(sec.beat * 0.75 * SR)
            y = x.copy()
            y[d:] += x[:-d] * echo
            y[2 * d:] += x[:-2 * d] * echo * 0.5
            x = y
        if wet:
            x = x + reverb(x, ir_len) * wet
        # fade out the tail past the section so sections hand over cleanly
        n_end = int(sec.dur * SR)
        tail = len(x) - n_end
        x = x.copy()
        x[n_end:] *= np.exp(-np.arange(tail) / (0.35 * SR))
        i = int(start * SR)
        j = min(len(out), i + len(x))
        out[i:j] += x[:j - i]

    S = {s["name"]: s for s in SCENES}

    # ── TITLE: NES fanfare ─────────────────────────────────────────
    s = Sec(S["title"]["dur"], 120)
    arp = [60, 64, 67, 72, 64, 67, 72, 76, 67, 72, 76, 79, 72, 76, 79, 84]
    for i, m in enumerate(arp):
        s.note("nes_lead", m, i * 0.25, 0.25, 0.16)
    for i, (m, l) in enumerate([(79, 0.5), (79, 0.5), (79, 0.5), (84, 2.5)]):
        st = 4 + sum(x[1] for x in [(79, 0.5), (79, 0.5), (79, 0.5), (84, 2.5)][:i])
        s.note("nes_lead", m, st, l, 0.2)
        s.note("nes_harm", m - 5 if m == 84 else m - 3, st, l, 0.1)
    for b, m in [(0, 48), (2, 48), (4, 43), (5, 43), (6, 48)]:
        s.note("nes_tri", m, b, 1 if b < 6 else 2, 0.35)
    for i in range(8):
        s.hit("nnoise", 3 + i * 0.125, 0.05 + i * 0.02)
    s.hit("nnoise", 4, 0.25)
    place(s, S["title"]["start"], gain=0.7)

    # ── WORLD 1 (1986): Game Boy, cosy 2-pulse + wave ──────────────
    s = Sec(S["w1"]["dur"], 100)
    for bar in range(5):
        ch = PROG[(bar - 1) % 4] if bar > 0 else "C"
        root = CH[ch][0] - 24
        for e in range(8):
            s.note("gb_bass", root + (12 if e % 2 else 0), bar * 4 + e * 0.5, 0.5, 0.09)
        if bar > 0:
            for q in range(4):
                s.note("gb_wave", CH[ch][q % 3] + 12, bar * 4 + q, 1, 0.08)
            melody(s, THEME[bar - 1], bar, "gb_lead", 0.13)
        for q in range(4):
            s.hit("nhat", bar * 4 + q + 0.5, 0.06)
    place(s, S["w1"]["start"], gain=1.9)

    # ── WORLD 2 (1990s): NES march ────────────────────────────────
    s = Sec(S["w2"]["dur"], 120)
    for i in range(16):
        s.hit("nnoise", 2 + i * 0.125, 0.04 + i * 0.012)
    for bar in range(1, 6):
        ch = PROG[(bar - 1) % 4]
        root = CH[ch][0] - 12
        for q, off in enumerate([0, 7, 0, 7]):
            s.note("nes_tri", root + off - 12 if off else root - 12, bar * 4 + q, 1, 0.33)
        for pos in [0, 1, 1.5, 2, 3, 3.25, 3.5, 3.75]:
            s.hit("nnoise", bar * 4 + pos, 0.12 if pos in (1, 3) else 0.07)
        if bar <= 4:
            melody(s, THEME[bar - 1], bar, "nes_lead", 0.15, harm=bar >= 3, hvol=0.07, hname="nes_harm", chord=ch)
        else:
            melody(s, THEME_END, bar, "nes_lead", 0.16, harm=True, hvol=0.07, hname="nes_harm", chord="C")
    s.note("nes_tri", 36, 0, 1, 0.3)
    s.note("nes_tri", 43, 3, 1, 0.3)
    place(s, S["w2"]["start"], gain=0.72)

    # ── WORLD 3 (2011): SNES, anticipation -> liftoff anthem ───────
    s = Sec(S["w3"]["dur"], 128)
    for bar, ch in enumerate(["Am", "F", "G"]):
        s.note("pad", CH[ch][0] - 12, bar * 4, 4, 0.10)
        for v in CH[ch]:
            s.note("pad", v, bar * 4, 4, 0.06)
        for e in range(8):
            s.note("pluck", CH[ch][e % 3] + 12 * (1 + (e // 3) % 2), bar * 4 + e * 0.5, 0.5, 0.07)
        for sx in range(16):
            s.hit("hat", bar * 4 + sx * 0.25, 0.03 + 0.03 * (bar / 2))
    for i in range(16):
        s.hit("snare", 8 + i * 0.25, 0.05 + i * 0.015)
    for bar in range(3, 7):
        ch = PROG[(bar - 3) % 4]
        melody(s, THEME[bar - 3], bar, "snes_lead", 0.12)
        melody(s, THEME[bar - 3], bar, "snes_lead", 0.04, tr=-12)
        for v in CH[ch]:
            s.note("pad", v, bar * 4, 4, 0.05)
        root = CH[ch][0] - 24
        for e in range(8):
            s.note("bass", root + (12 if e in (3, 7) else 0), bar * 4 + e * 0.5, 0.5, 0.16)
        for q in range(4):
            s.hit("kick", bar * 4 + q, 0.22 if q % 2 == 0 else 0.12)
            if q % 2:
                s.hit("snare", bar * 4 + q, 0.18)
            s.hit("hat", bar * 4 + q + 0.5, 0.07)
    s.hit("crash", 12, 0.3)
    s.hit("timp", 12, 0.35)
    place(s, S["w3"]["start"], wet=0.18, echo=0.25, gain=1.1)

    # ── WORLD 4: bouncy montage ───────────────────────────────────
    s = Sec(S["w4"]["dur"], 140)
    seq = [None, 0, 1, 2, 3, 2, "end"]
    for bar in range(7):
        ch = PROG[(bar - 1) % 4] if bar > 0 else "C"
        if bar == 6:
            ch = "G"
        root = CH[ch][0] - 24
        for e in range(8):
            s.note("bass", root + (12 if e % 2 else 0), bar * 4 + e * 0.5, 0.5, 0.15)
        for q in range(4):
            s.hit("kick", bar * 4 + q, 0.18)
            s.hit("hat", bar * 4 + q + 0.5, 0.07)
            if q % 2:
                s.hit("snare", bar * 4 + q, 0.13)
            for v in CH[ch]:
                s.note("pluck", v + 12, bar * 4 + q + 0.5, 0.4, 0.035)
        if seq[bar] is not None:
            notes = THEME_END if seq[bar] == "end" else THEME[seq[bar]]
            melody(s, notes, bar, "snes_lead", 0.12, harm=bar >= 5, hvol=0.06, hname="snes_lead", chord=ch)
        for sx in range(16):
            s.note("pluck", CH[ch][sx % 3] + 24, bar * 4 + sx * 0.25, 0.25, 0.02)
    place(s, S["w4"]["start"], wet=0.12, echo=0.12)

    # ── WORLD 5: HD epic ──────────────────────────────────────────
    s = Sec(S["w5"]["dur"], 90)
    for i in range(12):
        s.hit("timp", i * 0.25, 0.05 + i * 0.015)
    s.note("pad", 48, 0, 4, 0.12)
    for bar in range(1, 4):
        ch = PROG[(bar - 1) % 4]
        melody(s, THEME[bar - 1], bar, "brass", 0.14)
        melody(s, THEME[bar - 1], bar, "brass", 0.07, tr=-12)
        for v in CH[ch] + [CH[ch][0] + 12]:
            s.note("pad", v, bar * 4, 4, 0.06)
        s.note("bass", CH[ch][0] - 24, bar * 4, 4, 0.22)
        s.hit("timp", bar * 4, 0.4)
        s.hit("kick", bar * 4 + 2, 0.15)
        s.hit("snare", bar * 4 + 3.5, 0.06)
    s.hit("crash", 4, 0.35)
    place(s, S["w5"]["start"], wet=0.35, ir_len=2.2)

    # ── WORLD ∞: dreamy lydian bells ──────────────────────────────
    s = Sec(S["w6"]["dur"], 100)
    chords = ["Cmaj7", "DC", "Cmaj7", "Am9", "Fmaj7", "Gsus"]
    for bar, ch in enumerate(chords):
        for v in CH[ch]:
            s.note("pad", v, bar * 4, 4, 0.045)
        s.note("bass", CH[ch][0] - 24, bar * 4, 4, 0.13)
        tones = CH[ch]
        for e in range(8):
            s.note("bell", tones[e % len(tones)] + 12 + (12 if e >= 4 else 0), bar * 4 + e * 0.5, 0.5, 0.06)
        if bar >= 2:
            melody(s, THEME[bar - 2], bar, "sine_lead", 0.12, tr=0)
            s.hit("kick", bar * 4, 0.12)
            s.hit("hat", bar * 4 + 2, 0.03)
    place(s, S["w6"]["start"], wet=0.45, echo=0.35, ir_len=2.5, gain=1.15)

    # ── END: music-box callback -> grand finale ───────────────────
    s = Sec(S["end"]["dur"], 100)
    for bar in range(2):
        ch = PROG[bar]
        melody(s, THEME[bar], bar, "gb_lead", 0.08)
        melody(s, THEME[bar], bar, "musicbox", 0.07, tr=12)
        for v in CH[ch]:
            s.note("pad", v, bar * 4, 4, 0.04)
    # finale bar 2: theme bar 3 w/ full band; bar 3: huge C chord
    melody(s, THEME[2], 2, "brass", 0.15)
    melody(s, THEME[2], 2, "snes_lead", 0.07, tr=-12)
    for v in CH["F"]:
        s.note("pad", v, 8, 2, 0.07)
    for v in CH["G"]:
        s.note("pad", v, 10, 2, 0.07)
    s.note("bass", 41, 8, 2, 0.22)
    s.note("bass", 43, 10, 2, 0.22)
    for q in range(4):
        s.hit("kick", 8 + q, 0.18)
        s.hit("snare", 8 + q + 0.5 if q == 3 else 8 + q, 0.0)
    for i in range(8):
        s.hit("snare", 11 + i * 0.125, 0.05 + i * 0.02)
    s.note("brass", 84, 12, 3.5, 0.17)
    s.note("brass", 72, 12, 3.5, 0.09)
    for v in [48, 55, 60, 64, 67, 72]:
        s.note("pad", v, 12, 4, 0.06)
    for i, m in enumerate([72, 76, 79, 84, 88, 91, 96]):
        s.note("bell", m, 12 + i * 0.25, 1.5, 0.06)
    s.note("bass", 36, 12, 4, 0.25)
    s.hit("timp", 12, 0.45)
    s.hit("crash", 12, 0.4)
    place(s, S["end"]["start"], wet=0.35, ir_len=2.5)
    return out


_ir_cache = {}


def reverb(x, L):
    if L not in _ir_cache:
        n = int(L * SR)
        t = np.arange(n) / SR
        r = rng(42)
        ir = (r.rand(n) * 2 - 1) * np.exp(-t * 6.0 / L)
        ir = onepole(ir.astype(np.float32), 0.35)
        ir /= np.sqrt((ir ** 2).sum())
        _ir_cache[L] = ir.astype(np.float32)
    return fftconvolve(x, _ir_cache[L])[:len(x)].astype(np.float32) * 0.6


def sfx():
    out = np.zeros(int((TOTAL + 4) * SR), np.float32)

    def add(x, T, vol):
        i = int(T * SR)
        j = min(len(out), i + len(x))
        out[i:j] += x[:j - i] * vol

    def tone(kind, f, dur, duty=0.5, dec=20):
        n = int(dur * SR)
        t = np.arange(n) / SR
        return osc(kind, f, n, duty=duty) * np.exp(-t * dec).astype(np.float32)

    blip_voice = dict(gb=("pulse", 0.125, 1046), nes=("pulse", 0.5, 784), snes=("pulse", 0.25, 622),
                      hd=("sine", 0.5, 587))
    for sc in SCENES:
        st0 = sc["start"]
        kind, duty, base = blip_voice[sc["era"]]
        # card chime (coin-ish)
        if "card" in sc:
            add(np.concatenate([tone("pulse", 988, 0.07, 0.5, 5), tone("pulse", 1319, 0.35, 0.5, 8)]), st0 + 0.05, 0.10)
        for i, (st, tx, _) in enumerate(DIALOG.get(sc["name"], [])):
            ts, _ = char_times(tx, st)
            r = rng(len(tx) + i)
            for j, (ch, tt) in enumerate(zip(tx, ts)):
                if ch.isalnum() and j % 2 == 0:
                    f = base * (2 ** (r.choice([0, 2, 4, 5, 7]) / 12))
                    b = tone(kind, f, 0.035, duty, 45)
                    add(b, st0 + tt, 0.045 if kind == "sine" else 0.028)
    S = {s["name"]: s for s in SCENES}
    # title: press start confirm
    add(np.concatenate([tone("pulse", 1568, 0.05, 0.25, 5), tone("pulse", 2093, 0.2, 0.25, 12)]), S["title"]["start"] + 3.1, 0.07)
    # 1-UP for the IPO
    up = np.concatenate([tone("pulse", mf(m), 0.07, 0.5, 3) for m in (76, 79, 88, 84, 86, 91)])
    add(up, S["w2"]["start"] + DIALOG["w2"][1][0] + 0.2, 0.07)
    # rocket rumble
    n = int(4.0 * SR)
    t = np.arange(n) / SR
    rum = onepole((rng(5).rand(n) * 2 - 1).astype(np.float32), 0.03) * np.minimum(t / 0.3, 1) * np.exp(-np.maximum(t - 1.5, 0) * 1.2)
    add(rum * 4.0, S["w3"]["start"] + bars(3, 128), 0.5)
    # w4 panel pops
    for rt in w4_reveals():
        add(np.concatenate([tone("pulse", 523, 0.04, 0.5, 5), tone("pulse", 1046, 0.1, 0.5, 20)]), S["w4"]["start"] + rt, 0.06)
    add(np.concatenate([tone("pulse", mf(m), 0.08, 0.5, 4) for m in (72, 76, 79, 84, 79, 84)]),
        S["w4"]["start"] + DIALOG["w4"][1][0] + 0.8, 0.06)
    # w5 satellites power up
    for i, at in enumerate(w5_times()):
        arp_ = np.concatenate([tone("sine", mf(m), 0.05, dec=6) for m in (72 + i * 2, 79 + i * 2, 84 + i * 2, 91 + i * 2)])
        add(arp_, S["w5"]["start"] + at, 0.10)
    add(np.concatenate([tone("pulse", 1320, 0.06, 0.25, 10), tone("pulse", 1760, 0.12, 0.25, 12)]),
        S["w5"]["start"] + DIALOG["w5"][1][0] + 0.35, 0.05)  # meow-ish ping
    # w6 twinkles
    for w_ in ("quantum", "cyber", "self-flying"):
        tt = word_time("w6", 0, w_)
        add(np.concatenate([tone("sine", mf(m), 0.06, dec=8) for m in (91, 96, 103)]), S["w6"]["start"] + tt, 0.07)
    add(np.concatenate([tone("sine", mf(m), 0.08, dec=5) for m in (84, 88, 91, 96, 100)]),
        S["w6"]["start"] + DIALOG["w6"][1][0] + 1.0, 0.09)
    # end: continue -> yes
    add(np.concatenate([tone("pulse", 1568, 0.05, 0.25, 5), tone("pulse", 2093, 0.25, 0.25, 10)]),
        S["end"]["start"] + bars(2, 100) + 2.4, 0.07)
    return out


def build_audio(path):
    a = music() + sfx()
    a = a[:int((TOTAL + 0.2) * SR)]
    a = np.tanh(a * 1.6) / np.tanh(1.6)
    a /= max(1e-6, np.abs(a).max()) / 0.89
    # gentle stereo widening via tiny delay
    d = int(0.012 * SR)
    L = a
    R = np.concatenate([a[:d] * 0, a[:-d]]) * 0.25 + a * 0.75
    st = np.stack([L, R], 1)
    wavfile.write(path, SR, (st * 32767).astype(np.int16))


# ════════════════════════════════════════════════════════════════════
def main():
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    if "--preview" in sys.argv:
        pdir = os.environ.get("PREVIEW_DIR", os.path.join(HERE, "preview"))
        os.makedirs(pdir, exist_ok=True)
        times = [float(x) for x in sys.argv[sys.argv.index("--preview") + 1:]] or \
            [2.0, 5.5, 9.0, 14.0, 20.0, 24.0, 30.0, 34.5, 38.0, 45.0, 50.0, 53.0, 58.0, 62.0, 68.0, 74.0, 80.0, 86.0]
        for T in times:
            render_frame(T).resize((W * 3, H * 3), Image.NEAREST).save(os.path.join(pdir, f"f_{T:06.2f}.png"))
        print("preview written")
        return
    wav = os.path.join(HERE, "soundtrack.wav")
    print(f"total length {TOTAL:.2f}s; building audio…")
    build_audio(wav)
    out = os.path.join(HERE, "viasat_8bit_story.mp4")
    if "--audio-only" in sys.argv:  # re-mux a new soundtrack onto the existing video
        tmp = out + ".tmp.mp4"
        subprocess.run([ff, "-y", "-i", out, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                        "-b:a", "256k", "-shortest", "-movflags", "+faststart", tmp], check=True, stderr=subprocess.DEVNULL)
        os.replace(tmp, out)
        print("remuxed", out)
        return
    cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W * SCALE}x{H * SCALE}", "-r", str(FPS),
           "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-tune", "animation",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    nf = int(TOTAL * FPS)
    for i in range(nf):
        fr = render_frame(i / FPS).resize((W * SCALE, H * SCALE), Image.NEAREST)
        p.stdin.write(fr.tobytes())
        if i % 150 == 0:
            print(f"frame {i}/{nf}", flush=True)
    p.stdin.close()
    p.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
