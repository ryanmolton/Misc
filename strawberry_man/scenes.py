"""Scene renderers. Every function draws one 320x180 frame for absolute time t."""
import math
import numpy as np
from gfx import *
from art import *

# ------------------------------------------------------------------ timeline
T_BOOT, T_TITLE, T_BEACH, T_PROF, T_BATTLE, T_TALLY, T_END, T_TOTAL = 0.0, 1.3, 3.6, 9.2, 15.2, 28.7, 35.1, 41.6
B = T_BATTLE
# battle cues (absolute)
BT_SCENE = B + 0.5         # battle screen appears
BT_POSE = B + 7.0          # full-screen POSE cut-in starts
BT_POSE_END = B + 8.9      # back to the battle
BT_TROMBONE = B + 9.0
BT_FLEE = B + 11.35        # seagull takes off
E = T_END
END_LIGHT_OFF = E + 2.9
END_FADE = E + 3.6
END_FIN = E + 3.9

# typed dialog: (start, end, lines, chars/sec)
DIALOG = {
    'beach1': (3.85, 6.2, ["CARPINTERIA, CALIFORNIA.", "\"THE WORLD'S SAFEST BEACH.\""], 34),
    'beach2': (T_BEACH + 3.35, T_PROF, ["(NOT BECAUSE OF THIS GUY.)"], 42),
    'b1': (B + 0.6, B + 3.2, ["A WILD SEAGULL APPEARS!", "IT HAS YOUR BREAKFAST BURRITO."], 45),
    'b2': (B + 4.3, B + 6.0, ["YOU DON'T HAVE ANY POWERS."], 48),
    'b3': (BT_POSE + 0.1, BT_POSE_END, ["STRAWBERRY MAN", "STRIKES A POSE!"], 40),
    'b4': (BT_POSE_END + 0.05, B + 11.2, ["IT'S NOT VERY EFFECTIVE..."], 30),
    'b5': (B + 11.2, T_TALLY, ["THE SEAGULL FLED!", "(WITH YOUR BURRITO.)"], 45),
    'end1': (E + 0.45, T_TOTAL, ["WILL RETURN."], 22),
    'end2': (E + 1.3, T_TOTAL, ["...TO THE VAN."], 22),
    'fin': (END_FIN + 0.05, T_TOTAL, ["NO POWERS WERE USED IN", "THE MAKING OF THIS FILM."], 60),
}
# (start, end, [(time, cursor index)], select time)
MENU = [(B + 3.2, B + 4.3, [(B + 3.2, 0), (B + 3.8, 1)], B + 4.3), (B + 6.0, BT_POSE, [(B + 6.0, 1), (B + 6.5, 2)], B + 6.9)]
PROFILE_LINES_T0, PROFILE_DT = T_PROF + 0.35, 0.55
TALLY = {k: T_TALLY + v for k, v in dict(clear=0.15, l0=1.0, l1=1.6, l2=2.2, l2end=3.0, l3=3.4, label=4.0, stamp=4.4).items()}


def typed(key, t):
    t0, t1, lines, cps = DIALOG[key]
    n = int(max(0.0, t - t0) * cps)
    out, k = [], 0
    for ln in lines:
        out.append(ln[:max(0, n - k)])
        k += len(ln)
    return out, n >= k


def typed_blip_times(key):
    t0, t1, lines, cps = DIALOG[key]
    s = ''.join(lines)
    return [t0 + i / cps for i, ch in enumerate(s) if ch not in ' .' and i % 2 == 0]


def dialog_box(cv, key, t, x=4, y=132, w=312, h=44, color=PAL['W']):
    box(cv, x, y, w, h, fill=hexc('#10102a'), border=PAL['W'], border2=hexc('#5060a0'))
    lines, done = typed(key, t)
    for i, ln in enumerate(lines):
        text(cv, ln, x + 10, y + 10 + i * 14, color=color, shadow=hexc('#303060'))
    if done and int(t * 3) % 2 == 0:
        text(cv, '>', x + w - 14, y + h - 12, color=PAL['Y'])


def fade(cv, f):
    """NES-style stepped fade. f=1 full bright, 0 black."""
    f = round(clamp01(f) * 4) / 4
    if f < 1:
        cv[:] = (cv * f).astype(np.uint8)


def iris(cv, cx, cy, r):
    yy, xx = np.ogrid[:H, :W]
    cv[((xx - cx) ** 2 + (yy - cy) ** 2) > r * r] = 0


def star4(cv, x, y, s, color=PAL['W']):
    x, y = int(x), int(y)
    rect(cv, x - s, y, 2 * s + 1, 1, color)
    rect(cv, x, y - s, 1, 2 * s + 1, color)
    if s >= 2:
        rect(cv, x - 1, y - 1, 3, 3, color)


# ------------------------------------------------------------------ BOOT + TITLE
_rng = np.random.default_rng(3)
STARS = [(_rng.uniform(0, W), _rng.uniform(0, H), _rng.integers(1, 4), _rng.uniform(0, 6.28)) for _ in range(110)]


def starfield(cv, t, speed=1.0, color_top='#0b0b2a', color_bot='#2a1650'):
    cv[:] = dither_gradient(H, W, [(0, hexc(color_top)), (H, hexc(color_bot))])
    for (x, y, layer, ph) in STARS:
        xx = (x - t * speed * 8 * layer) % W
        b = 0.55 + 0.45 * math.sin(t * 5 + ph)
        c = (np.array([255, 255, 255]) * b * (0.4 + 0.2 * layer)).astype(np.uint8)
        cv[int(y), int(xx)] = c
        if layer == 3 and b > 0.9:
            star4(cv, xx, y, 1, c)


def render_boot(cv, t):
    cv[:] = hexc('#0c0c14')
    y = -18 + (78 + 18) * clamp01(t / 0.85)
    text_c(cv, 'VANTENDO', 160, int(y), scale=2, color=hexc('#b8b8d8'))
    if t > 0.85:
        text(cv, '(R)', 160 + 48 + 2, int(y) - 2, color=hexc('#b8b8d8'))
    if t > 1.15:
        f = clamp01((t - 1.15) / 0.15)
        cv[:] = (cv * (1 - f) + 255 * f).astype(np.uint8)


def title_grad(h, top, bot):
    return np.array([np.array(hexc(top)) * (1 - i / (h - 1)) + np.array(hexc(bot)) * (i / (h - 1)) for i in range(h)]).astype(np.uint8)


GRAD_RED = title_grad(21, '#ffb0b8', '#c0102c')
GRAD_RED[:3] = hexc('#ffe0e4')
GRAD_GOLD = title_grad(28, '#fff6b0', '#ff8a10')


def render_title(cv, t):
    lt = t - T_TITLE
    starfield(cv, lt)
    # title words drop in
    y1 = -30 + (18 + 30) * back_out(lt / 0.45, 1.6)
    y2 = 190 - (190 - 44) * back_out((lt - 0.3) / 0.4, 1.3)
    for (word, sc, y, grad) in (('STRAWBERRY', 3, y1, GRAD_RED), ('MAN', 4, y2, GRAD_GOLD)):
        x = 160 - text_width(word, sc) // 2
        m = text_mask(word, sc)
        put_mask(cv, dilate(np.pad(m, 2), 2), x - 2, int(y) - 2 + 2, PAL['K'])  # drop shadow
        put_mask(cv, dilate(np.pad(m, 1), 1), x - 1, int(y) - 1, hexc('#3a0a18'))
        put_mask(cv, m, x, int(y), grad)
        # shine sweep
        s = (lt - 0.75) * 420 - 40
        if -40 < s < 360:
            yy, xx = np.nonzero(m)
            sel = ((xx + x + (yy)) > s) & ((xx + x + yy) < s + 5)
            for a, b in zip(yy[sel], xx[sel]):
                if 0 <= int(y) + a < H and 0 <= x + b < W:
                    cv[int(y) + a, x + b] = 255
    # bouncing strawberry icons
    if lt > 0.55:
        ic = scale(STRAWBERRY_ICON, 3)
        for i, cx in enumerate((92, 204)):
            by = 46 + int(abs(math.sin(lt * 6 + i * 1.5)) * -6)
            blit(cv, ic, cx, by)
    if lt > 0.7:
        tag = 'THE HERO NOBODY ASKED FOR'
        n = int((lt - 0.7) * 50)
        text_c(cv, tag[:n].ljust(len(tag)), 160, 86, color=PAL['Y'], shadow=PAL['K'])
    # little hero running across the bottom
    hx = -30 + (lt - 0.2) * 150
    legs = 'runA' if int(lt * 10) % 2 else 'runB'
    blit(cv, hero('stand', legs, lt * 3, 0.6, 0.5), hx, 132 + (1 if int(lt * 10) % 2 else 0))
    pressed = lt > 1.85
    blink = (int(lt * 20) % 2 == 0) if pressed else (int(lt * 2.5) % 2 == 0)
    if lt > 0.9 and blink:
        text_c(cv, 'PRESS START', 160, 112, color=PAL['W'], shadow=hexc('#e8283c'))
    text_c(cv, '(C)1987 VANTENDO   LICENSED BY NOBODY', 160, 168, color=hexc('#8080b0'))
    if lt < 0.12:
        f = 1 - lt / 0.12
        cv[:] = (cv * (1 - f) + 255 * f).astype(np.uint8)
    if lt > 2.05:
        iris(cv, 160, 96, max(0, 200 * (1 - (lt - 2.05) / 0.25)))


# ------------------------------------------------------------------ BEACH WORLD
WORLD_W = 640
VAN_WX, VAN_Y, GROUND = 330, 137, 172

PALETTES = {
    'dawn': dict(sky=[(0, '#34479a'), (30, '#6570c2'), (60, '#c48bc4'), (84, '#f8a8a4'), (104, '#ffd6a8')],
                 island='#9a86c0', ocean=[(104, '#a8a0d8'), (110, '#6a86d0'), (128, '#3aa0d0')],
                 hl='#e8e0ff', sand='#f2d7a0', sand2='#d9b87e', wet='#c9a778', road='#5a5a6c', road2='#6a6a7e',
                 curb='#b8b8c8', palm='#2f9a44', palm2='#1d6a34', trunk='#9a6a3a', trunk2='#7a4a2a', tint=None),
    'sunset': dict(sky=[(0, '#241248'), (26, '#4f2474'), (54, '#b23a78'), (80, '#f06a48'), (104, '#ffc048')],
                   island='#5a2a64', ocean=[(104, '#f08850'), (112, '#a04878'), (128, '#4a2a6a')],
                   hl='#ffd070', sand='#c88a6a', sand2='#a8685a', wet='#8a5060', road='#3a2a4a', road2='#4a3858',
                   curb='#8a6a8a', palm='#2a1238', palm2='#1a0a28', trunk='#2a1238', trunk2='#1a0a28',
                   tint=(1.0, 0.72, 0.68)),
    'night': dict(sky=[(0, '#050818'), (50, '#0e1540'), (104, '#28306e')],
                  island='#161c44', ocean=[(104, '#1e2860'), (128, '#0c1234')],
                  hl='#b8c8ff', sand='#3a3a5e', sand2='#303050', wet='#26264a', road='#1c1c2c', road2='#24243a',
                  curb='#44445e', palm='#0a0c20', palm2='#060816', trunk='#0a0c20', trunk2='#060816',
                  tint=(0.42, 0.48, 0.78)),
}

_layers = {}


def palm_sprite(p, h=78, lean=0.25, seed=0):
    img = np.zeros((h + 30, 90, 4), np.uint8)
    cx0 = 45
    top = None
    for y in range(h):
        k = y / h
        x = cx0 + int(lean * (y ** 1.6) / h ** 0.6) - int(lean * h * 0.35)
        yy = h + 28 - y
        c = hexc(p['trunk']) if (y // 3) % 2 == 0 else hexc(p['trunk2'])
        wdt = 3 if k > 0.3 else 4
        img[yy, x:x + wdt, :3] = c
        img[yy, x:x + wdt, 3] = 255
        top = (x + 1, yy)
    tx, ty = top
    rng = np.random.default_rng(seed)
    for ang in np.linspace(-2.8, -0.35, 7):
        L = 26 + rng.integers(-3, 5)
        for s in np.linspace(0, 1, 60):
            x = tx + math.cos(ang) * L * s
            y = ty + math.sin(ang) * L * s * 0.55 + (s ** 2) * 16
            for d in (0, 1):
                xi, yi = int(x), int(y) + d
                if 0 <= xi < 90 and 0 <= yi < img.shape[0]:
                    img[yi, xi, :3] = hexc(p['palm']) if d == 0 else hexc(p['palm2'])
                    img[yi, xi, 3] = 255
            if s > 0.2 and int(s * 60) % 4 == 0:  # leaflets
                for dd in range(1, 4):
                    xi, yi = int(x - dd * 0.5), int(y) + 1 + dd
                    if 0 <= xi < 90 and 0 <= yi < img.shape[0]:
                        img[yi, xi, :3] = hexc(p['palm2']); img[yi, xi, 3] = 255
    for (dx, dy) in ((-2, 2), (1, 3), (3, 1)):
        img[ty + dy:ty + dy + 2, tx + dx:tx + dx + 2, :3] = hexc(p['trunk2']); img[ty + dy:ty + dy + 2, tx + dx:tx + dx + 2, 3] = 255
    return img, (h + 28)


def lifeguard_tower(p, night=False):
    rows = [
        "....RRRRRRRRRRRRRR....",
        "...RRRRRRRRRRRRRRRR...",
        "..RRRRRRRRRRRRRRRRRR..",
        "...CCCCCCCCCCCCCCCC...",
        "...CwwwwwwCCwwwwwwC...",
        "...CwwwwwwCCwwwwwwC...",
        "...CCCCCCCCCCCCCCCC...",
        "...QQQQQQQQQQQQQQQQ...",
        "...CCCCCCCCCCCCCCCC...",
        "tttttttttttttttttttttt",
        "..t....t......t....t..",
        "..t...t.t....t.t...t..",
        "..t..t...t..t...t..t..",
        "..t.t.....tt.....t.t..",
        "..tt......tt......tt..",
        "..t.t....t..t....t.t..",
        "..t..t..t....t..t..t..",
        "..t...tt......tt...t..",
        "..t...tt......tt...t..",
        "..t..t..t....t..t..t..",
    ]
    return outline(sprite(rows))


def no_parking_sign():
    rows = [
        "..WWWWWWW..",
        ".WRRRRRRRW.",
        "WRRWWWWWRRW",
        "WRWKKKWRWRW",
        "WRWKWWKRWRW",
        "WRWKKKRWWRW",
        "WRWKWRWWWRW",
        "WRWKRWWWWRW",
        "WRRRWWWWRRW",
        ".WRRRRRRRW.",
        "..WWWWWWW..",
        ".....z.....",
        ".....z.....",
        ".....z.....",
        ".....z.....",
        ".....z.....",
        ".....z.....",
        ".....z.....",
        ".....z.....",
        ".....z.....",
        "....zzz....",
    ]
    return outline(sprite(rows))


def cloud(r_list):
    w = 60
    m = np.zeros((24, w), bool)
    for (cx, cy, r) in r_list:
        d = disc(r)
        m[cy - r:cy + r + 1, cx - r:cx + r + 1] |= d
    img = np.zeros((24, w, 4), np.uint8)
    img[m, :3] = PAL['W']; img[m, 3] = 255
    under = m & ~shift(m, 2, 0)
    img[under, :3] = hexc('#c8d0f0')
    return img


CLOUDS = [cloud([(12, 14, 6), (22, 10, 8), (34, 12, 7), (44, 15, 5)]), cloud([(10, 15, 5), (20, 12, 7), (30, 15, 5)])]


def tint_img(img, tint, keep=()):
    if tint is None:
        return img
    out = img.copy()
    keepm = np.zeros(img.shape[:2], bool)
    for c in keep:
        keepm |= np.all(img[..., :3] == c, axis=-1)
    f = np.array(tint)
    out[..., :3] = np.where(keepm[..., None], img[..., :3], np.clip(img[..., :3] * f, 0, 255)).astype(np.uint8)
    return out


def get_layers(name):
    if name in _layers:
        return _layers[name]
    p = PALETTES[name]
    L = {}
    L['sky'] = dither_gradient(104, W, [(y, hexc(c)) for y, c in p['sky']])
    # islands (Channel Islands) far horizon, world-wide strip
    isl = np.zeros((12, WORLD_W, 4), np.uint8)
    xs = np.arange(WORLD_W)
    hgt = (np.maximum(0, 6 * np.sin(xs / 37.0) + 4 * np.sin(xs / 13.0 + 1) + 2) * (np.sin(xs / 90.0) > -0.2)).astype(int)
    for x in xs:
        if hgt[x] > 0:
            isl[12 - hgt[x]:, x, :3] = hexc(p['island']); isl[12 - hgt[x]:, x, 3] = 255
    L['isl'] = isl
    L['ocean'] = dither_gradient(24, WORLD_W, [(y - 104, hexc(c)) for y, c in p['ocean']])
    g = np.zeros((180 - 128, WORLD_W, 3), np.uint8)
    rng = np.random.default_rng(11)
    g[:] = hexc(p['sand'])
    g[0:4] = hexc(p['wet'])
    spk = rng.random((180 - 128, WORLD_W)) < 0.06
    g[spk] = hexc(p['sand2'])
    g[146 - 128:148 - 128] = hexc(p['curb'])
    g[148 - 128:] = hexc(p['road'])
    spk2 = (rng.random((180 - 128, WORLD_W)) < 0.08)
    spk2[:148 - 128] = False
    g[spk2] = hexc(p['road2'])
    for x0 in range(0, WORLD_W, 28):  # dashed lane line
        g[176 - 128:178 - 128, x0:x0 + 14] = hexc('#e8c040') if name == 'dawn' else hexc(p['curb'])
    L['ground'] = g
    L['palms'] = [(wx, *palm_sprite(p, h, lean, i)) for i, (wx, h, lean) in enumerate([(30, 80, 0.3), (230, 70, -0.25), (470, 84, 0.2), (560, 66, -0.3)])]
    L['tower'] = tint_img(lifeguard_tower(p), p['tint'])
    L['sign'] = tint_img(no_parking_sign(), p['tint'])
    L['clouds'] = [tint_img(c, p['tint']) for c in CLOUDS]
    _layers[name] = L
    return L


def draw_world(cv, name, cam, t, van_img=None, van_dy=0, sun=None):
    p = PALETTES[name]
    L = get_layers(name)
    cv[:104] = L['sky']
    if name == 'night':
        for (x, y, layer, ph) in STARS:
            if y < 100:
                b = 0.5 + 0.5 * math.sin(t * 3 + ph)
                cv[int(y), int(x)] = (np.array([255, 255, 230]) * b * (0.35 + 0.2 * layer)).astype(np.uint8)
        m = np.pad(disc(9), 4)
        crescent = m & ~shift(m, -3, 4)
        put_mask(cv, crescent, 246, 14, hexc('#fff4c8'))
    if sun is not None:
        sx, sy, sr = sun
        yy, xx = np.ogrid[:104, :W]
        d = (xx - sx) ** 2 + (yy - sy) ** 2 <= sr * sr
        band = (yy - (sy - sr)) / (2 * sr)
        col = np.where(band < 0.5, 1, 0)
        stripes = ((yy > sy - 6) & ((yy - sy) % 5 < (yy - sy + 12) // 5)) & (yy < sy + sr)
        m = d & ~stripes
        cv[:104][m & (yy < sy - 8)] = hexc('#fff4a0')
        cv[:104][m & (yy >= sy - 8)] = hexc('#ffb040')
    if name != 'night':
        for i, c in enumerate(L['clouds']):
            cx = (40 + i * 170 - cam * 0.08 + t * 3) % (W + 80) - 60
            blit(cv, c, cx, 14 + i * 22)
    # islands
    ix = int(cam * 0.1)
    sub = L['isl'][:, ix:ix + W]
    m = sub[..., 3] > 0
    cv[92:104][m] = sub[..., :3][m]
    # ocean
    ox = int(cam * 0.5)
    cv[104:128] = L['ocean'][:, ox:ox + W]
    hl = hexc(p['hl'])
    rng = np.random.default_rng(int(t * 4))
    for i in range(26):
        yy = 105 + (i * 7) % 22
        xx = int((i * 53 + t * (6 + i % 3 * 4) - cam * 0.5) % W)
        if (i + int(t * 4)) % 3:
            rect(cv, xx, yy, 3 + i % 4, 1, hl)
    if sun is not None:  # sun reflection column
        sx = sun[0]
        for k, yy in enumerate(range(105, 128, 2)):
            wdt = 4 + k * 2 + int(3 * math.sin(t * 6 + k))
            rect(cv, sx - wdt // 2 + int(2 * math.sin(t * 4 + k * 1.7)), yy, wdt, 1, hexc('#ffd860'))
    if name == 'night':
        for k, yy in enumerate(range(106, 128, 2)):
            wdt = 2 + k + int(2 * math.sin(t * 5 + k))
            rect(cv, 250 - wdt // 2 + 4 - int(cam * 0.5 - 101), yy, wdt, 1, hexc('#8890c8'))
    # ground
    gx = int(cam)
    cv[128:180] = L['ground'][:, gx:gx + W]
    # animated foam line
    fy = 128 + int(round(1.5 + 1.5 * math.sin(t * 2.0)))
    for x in range(W):
        if (x + gx + int(t * 10)) % 11 < 8:
            cv[fy, x] = hexc('#ffffff') if name == 'dawn' else hl
    cv[127, :] = hl
    # props
    blit(cv, L['tower'], 196 - cam, 146 - L['tower'].shape[0] + 2)
    for (wx, img, baseh) in L['palms']:
        blit(cv, img, wx - cam - 45, 148 - baseh)
    blit(cv, L['sign'], 412 - cam, GROUND - L['sign'].shape[0] + 1)
    # distant gulls
    if name != 'night':
        for i in range(3):
            gx2 = (i * 120 + t * (14 + i * 5) - cam * 0.3) % (W + 40) - 20
            gy2 = 40 + i * 12 + 3 * math.sin(t * 3 + i)
            flap = int(t * 6 + i) % 2
            c = hexc('#303050') if name == 'dawn' else hexc('#2a1238')
            if flap:
                for (dx, dy) in ((0, 0), (1, 1), (2, 1), (3, 0), (4, 1), (5, 1), (6, 0)):
                    rect(cv, gx2 + dx, gy2 + dy, 1, 1, c)
            else:
                for (dx, dy) in ((0, 1), (1, 0), (2, 1), (3, 1), (4, 1), (5, 0), (6, 1)):
                    rect(cv, gx2 + dx, gy2 + dy, 1, 1, c)
    if van_img is not None:
        blit(cv, van_img, VAN_WX - cam, VAN_Y + van_dy)


def hud(cv, t, score=0, time_left=400):
    items = [(10, 'STRAWBERRY MAN', f'{score:06d}'), (190, 'WORLD', ' 1-1'), (258, 'TIME', f' {time_left:3d}')]
    for x, a, b in items:
        text(cv, a, x, 5, shadow=PAL['K'])
        text(cv, b, x, 14, shadow=PAL['K'])
    blit(cv, STRAWBERRY_ICON, 118, 13)
    text(cv, 'x03', 128, 14, shadow=PAL['K'])


# ------------------------------------------------------------------ BEACH
def render_beach(cv, t):
    lt = t - T_BEACH
    cam = 160 * ease_in_out(lt / 1.7)
    shaking = 1.7 < lt < 2.45
    door = clamp01((lt - 2.45) / 0.15)
    van_img = van(door_open=door)
    draw_world(cv, 'dawn', cam, t, van_img, van_dy=(-1 if shaking and int(lt * 14) % 2 else 0))
    # snoring Zs
    for i in range(4):
        zt = lt - 1.6 - i * 0.2
        if 0 < zt < 0.9 and lt < 2.5:
            zx = VAN_WX - cam + 40 + zt * 14 + 3 * math.sin(zt * 8)
            zy = VAN_Y + 4 - zt * 30
            text(cv, 'Z', zx, zy, color=PAL['W'], shadow=PAL['M'], scale=1 + (i % 2))
    # the hero leaps from the sliding door
    jx0, jx1 = VAN_WX - cam + 12, 118
    if lt >= 2.6:
        k = clamp01((lt - 2.6) / 0.5)
        hx = jx0 + (jx1 - jx0) * k
        hy = (GROUND - 31) - 10 - 46 * math.sin(math.pi * k) * (1 - 0.2 * k) + 10 * k
        if k < 1:
            img = hero('stand', 'jump', lt * 2, 0.8, 1.0)
        else:
            img = hero('hips', None, lt * 1.5, 0.35 + 0.15 * math.sin(lt * 5), 0.8)
            hy = GROUND - 31
            if lt < 3.25:  # landing squash: shake camera
                cv[:] = np.roll(cv, 1 if int(lt * 30) % 2 else -1, axis=0)
        blit(cv, img, hx, hy)
        if 3.1 < lt < 3.6:  # sparkle burst
            for i in range(6):
                a = i / 6 * 6.283 + lt * 2
                r = 10 + (lt - 3.1) * 50
                star4(cv, hx + 15 + math.cos(a) * r, hy + 14 + math.sin(a) * r * 0.8, 2 if i % 2 else 1, PAL['Y'])
    hud(cv, t, score=0, time_left=400 - int(lt * 2.5))
    if lt < 3.35:
        dialog_box(cv, 'beach1', t, y=26)
    else:
        dialog_box(cv, 'beach2', t, y=26, h=30)
    if lt < 0.3:
        iris(cv, 170, 150, 330 * ease_out(lt / 0.3))
    if t > T_PROF - 0.15:  # blinds wipe to next scene
        k = (t - (T_PROF - 0.15)) / 0.15
        for x0 in range(0, W, 16):
            rect(cv, x0, 0, int(16 * k), H, PAL['K'])


# ------------------------------------------------------------------ PROFILE
def stripes_bg(cv, t, c1='#5a0f22', c2='#7a1830', speed=30):
    yy, xx = np.mgrid[0:H, 0:W]
    m = (((xx + yy + t * speed) // 14) % 2).astype(bool)
    cv[:] = hexc(c1)
    cv[m] = hexc(c2)


def bar(cv, x, y, n_full, n_total, color, empty=True):
    for i in range(n_total):
        g = '@' if i < n_full else ('~' if empty else ' ')
        text(cv, g, x + i * 6, y, color=color if i < n_full else PAL['D'])


def render_profile(cv, t):
    lt = t - T_PROF
    stripes_bg(cv, lt)
    shake = 0
    # spotlight + hero
    put_mask(cv, np.ones((1, 1), bool), 0, 0, cv[0, 0])
    yy, xx = np.ogrid[:H, :W]
    spot = ((xx - 62) / 50.0) ** 2 + ((yy - 150) / 9.0) ** 2 <= 1
    cv[spot] = hexc('#ffb0c0')
    cone = (np.abs(xx - 62) < (8 + (yy) * 0.28)) & (yy < 150)
    cv[cone] = (cv[cone] * 0.7 + np.array([255, 220, 230]) * 0.3).astype(np.uint8)
    big = scale(hero('hips', None, lt * 1.6, 0.4 + 0.2 * math.sin(lt * 3), 0.9), 3)
    blit(cv, big, 62 - big.shape[1] // 2, 150 - 92 + int(math.sin(lt * 4) > 0.9))
    # header
    text_c(cv, '- PLAYER 1 -', 62, 8, color=PAL['Y'], shadow=PAL['K'])
    box(cv, 120, 24, 194, 134, fill=hexc('#14102a'), border=PAL['W'], border2=hexc('#b03050'))
    rows = [('NAME', 'STRAWBERRY MAN', PAL['W']), ('POWERS', 'NONE', PAL['R']), ('ORIGIN', 'UNCLEAR', PAL['Y']),
            ('HOME', 'A VAN (2002)', PAL['W']), ('STRENGTH', None, 2), ('SPEED', None, 1), ('VITAMIN C', None, 99),
            ('WEAKNESS', 'WHIPPED CREAM', PAL['P'])]
    for i, (lab, val, extra) in enumerate(rows):
        ti = PROFILE_LINES_T0 + i * PROFILE_DT
        if t < ti:
            continue
        y = 34 + i * 15
        text(cv, lab, 130, y, color=PAL['L'])
        n = int((t - ti) * 60)
        if val is not None:
            text(cv, val[:n], 190, y, color=extra, shadow=PAL['K'])
            if val == 'NONE' and t - ti > 0.4 and int(t * 4) % 2:
                text(cv, '!', 190 + 30, y, color=PAL['R'])
        elif extra == 99:
            fill_n = int((t - ti) * 26)
            for k in range(min(fill_n, 40)):
                c = [PAL['R'], PAL['O'], PAL['Y'], PAL['G'], PAL['b'], PAL['P']][(k + int(t * 12)) % 6] if k >= 6 else PAL['G']
                text(cv, '@', 190 + k * 6, y, color=c)
            if fill_n > 20:
                shake = 1
                if int(t * 8) % 2:
                    text(cv, 'MAX!!', 256, y - 10, color=PAL['Y'], outline=PAL['K'])
        else:
            bar(cv, 190, y, min(extra, int((t - ti) * 20)), 6, PAL['G'])
    if shake:
        cv[:] = np.roll(cv, int(t * 40) % 3 - 1, axis=1)
    if lt < 0.15:
        k = 1 - lt / 0.15
        for x0 in range(0, W, 16):
            rect(cv, x0 + int(16 * (1 - k)), 0, int(16 * k) + 1, H, PAL['K'])
    # battle transition
    if t > T_BATTLE - 0.02:
        pass


# ------------------------------------------------------------------ BATTLE
_battle_bg = None


def battle_bg():
    global _battle_bg
    if _battle_bg is None:
        bg = np.zeros((H, W, 3), np.uint8)
        bg[:60] = dither_gradient(60, W, [(0, hexc('#7cc4ff')), (60, hexc('#dff2ff'))])
        bg[52:66] = dither_gradient(14, W, [(0, hexc('#58b0e8')), (14, hexc('#2e86c8'))])
        bg[66:68] = hexc('#ffffff')
        bg[68:] = dither_gradient(H - 68, W, [(0, hexc('#f6e2a8')), (112, hexc('#e8c888'))])
        for (cx, cy, rx, ry) in ((236, 90, 50, 11), (80, 140, 70, 16)):
            yy, xx = np.ogrid[:H, :W]
            e = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
            bg[e <= 1] = hexc('#d4a868')
            bg[e <= 0.7] = hexc('#e8c080')
            bg[(e <= 1) & (e > 0.85) & (yy < cy)] = hexc('#b88850')
        _battle_bg = bg
    return _battle_bg


WING_UP = sprite([
    "......KK",
    ".....KLK",
    "....KLLK",
    "...KLLDK",
    "..KLLDK.",
    ".KLDDK..",
    "KKKKK...",
])
BURSTA = None


def hp_bar(cv, x, y, w, frac, blink=False):
    rect(cv, x - 1, y - 1, w + 2, 5, PAL['K'])
    rect(cv, x, y, w, 3, hexc('#304030'))
    c = hexc('#48d860') if frac > 0.5 else hexc('#f0c030')
    if not blink:
        rect(cv, x, y, int(w * frac), 3, c)
        rect(cv, x, y, int(w * frac), 1, hexc('#a0f8a8'))


def render_battle(cv, t):
    lt = t - T_BATTLE
    if lt < 0.5:  # encounter transition drawn over the profile screen
        render_profile(cv, t)
        if int(lt * 16) % 2 == 0 and lt < 0.25:
            cv[:] = 255 - cv
        if lt >= 0.2:
            k = (lt - 0.2) / 0.3
            for i, y0 in enumerate(range(0, H, 12)):
                wdt = int(W * clamp01(k * 1.4 - i * 0.03))
                if i % 2:
                    rect(cv, 0, y0, wdt, 12, PAL['K'])
                else:
                    rect(cv, W - wdt, y0, wdt, 12, PAL['K'])
        return
    if BT_POSE <= t < BT_POSE_END:
        return render_pose(cv, t)
    cv[:] = battle_bg()
    et = t - BT_SCENE
    # enemy
    gx = -80 + (204 + 80) * ease_out(et / 0.45)
    gy = 46 + (1 if int(t * 4) % 2 else 0)
    if BT_POSE_END + 0.2 < t < BT_POSE_END + 2.0:
        gy -= int(abs(math.sin((t - BT_POSE_END - 0.2) * 9)) * 5)  # smug hops
    flee = t >= BT_FLEE
    if flee:
        ft = t - BT_FLEE
        gx += ft * 125
        gy -= ft * 70 + 6 * math.sin(ft * 30)
    gimg = scale(GULL_ANGRY if (t > BT_POSE_END) else GULL, 3)
    blit(cv, gimg, gx, gy)
    blit(cv, scale(BURRITO, 3), gx - 21, gy + 6)
    if flee and int(t * 12) % 2:
        blit(cv, scale(WING_UP, 3), gx + 33, gy - 12)
    if BT_POSE_END + 0.05 < t < BT_POSE_END + 0.9:
        text(cv, '-0', gx + 24, gy - 8 - (t - BT_POSE_END - 0.05) * 20, color=PAL['W'], outline=PAL['K'])
    # hero back sprite
    hx = 320 - (320 - 36) * ease_out(et / 0.45)
    blit(cv, scale(HERO_BACK, 3), hx, 64 + (1 if int(t * 3) % 2 else 0))
    # info boxes
    if et > 0.45:
        box(cv, 12, 12, 132, 32, fill=hexc('#fff8e8'), border=PAL['K'], border2=hexc('#c0b090'))
        text(cv, 'SEAGULL', 22, 18, color=PAL['K'])
        text(cv, 'LV99', 110, 18, color=PAL['K'])
        text(cv, 'HP', 22, 30, color=hexc('#e06020'))
        hp_bar(cv, 38, 32, 96, 1.0, blink=(BT_POSE_END + 0.05 < t < BT_POSE_END + 0.6 and int(t * 16) % 2))
        box(cv, 172, 92, 142, 36, fill=hexc('#fff8e8'), border=PAL['K'], border2=hexc('#c0b090'))
        text(cv, 'STRAWBERRY MAN', 180, 98, color=PAL['K'])
        text(cv, 'LV1', 290, 98, color=PAL['K'])
        text(cv, 'HP', 180, 111, color=hexc('#e06020'))
        hp_bar(cv, 196, 113, 110, 1.0)
    # dialog / menu
    menu = None
    for (m0, m1, cur, sel) in MENU:
        if m0 <= t < m1:
            menu = (cur, sel)
    if menu:
        cur, sel = menu
        idx = [i for (tt, i) in cur if t >= tt][-1]
        box(cv, 4, 132, 170, 44, fill=hexc('#10102a'), border=PAL['W'], border2=hexc('#5060a0'))
        text(cv, 'WHAT WILL', 14, 142)
        text(cv, 'STRAWBERRY MAN DO?', 14, 156)
        box(cv, 172, 132, 144, 44, fill=hexc('#fff8e8'), border=PAL['K'], border2=hexc('#c0b090'))
        opts = [('FIGHT', 192, 142), ('POWERS', 252, 142), ('POSE', 192, 156), ('RUN', 252, 156)]
        for i, (o, x, y) in enumerate(opts):
            text(cv, o, x, y, color=hexc('#a0a0a0') if (o == 'POWERS' and t > DIALOG['b2'][0] + 0.5) else PAL['K'])
        o, x, y = opts[idx]
        flash = t > sel and int(t * 20) % 2
        if not flash:
            text(cv, '>', x - 9, y, color=PAL['R'])
    else:
        for key in ('b1', 'b2', 'b4', 'b5'):
            t0, t1 = DIALOG[key][:2]
            if t0 <= t < t1:
                dialog_box(cv, key, t)
    if t < BT_SCENE + 0.1:
        f = clamp01((BT_SCENE + 0.1 - t) / 0.1)
        cv[:] = (cv * (1 - f) + 255 * f).astype(np.uint8)
    if BT_POSE_END <= t < BT_POSE_END + 0.08:
        cv[:] = 255
    if t > T_TALLY - 0.2:
        fade(cv, (T_TALLY - t) / 0.2)


_ang = None


def render_pose(cv, t):
    global _ang
    lt = t - BT_POSE
    if _ang is None:
        yy, xx = np.mgrid[0:H, 0:W]
        _ang = np.arctan2(yy - 80, xx - 90)
    wedge = (((_ang + lt * 1.8) / (2 * np.pi / 18)).astype(int) % 2).astype(bool)
    cv[:] = hexc('#ff304a')
    cv[wedge] = hexc('#ffb428')
    rng = np.random.default_rng(int(lt * 30))
    for i in range(14):  # speed lines
        y = rng.integers(0, H)
        x = rng.integers(-40, W)
        rect(cv, x, y, rng.integers(20, 60), 1, PAL['W'])
    k = back_out(lt / 0.3, 1.4)
    big = scale(hero('fist', None, lt * 3, -0.9, 2.0), 4)
    hx, hy = 30, 190 - 186 * k
    blit(cv, outline(outline(big, PAL['W']), PAL['K']), hx - 2, hy - 2)
    for i in range(7):
        a = i * 0.9 + lt * 3
        r = 30 + 12 * math.sin(lt * 10 + i)
        s = 3 if int(lt * 12 + i) % 3 == 0 else 2
        star4(cv, hx + 98 + math.cos(a) * r, hy + 18 + math.sin(a) * r, s, PAL['W'])
    if lt > 0.15:
        sc = max(3, 7 - int((lt - 0.15) * 40))
        word = 'POSE!'
        jig = int(math.sin(lt * 40) * 1.5)
        x = 232 - text_width(word, sc) // 2
        m = text_mask(word, sc)
        put_mask(cv, dilate(np.pad(m, 3), 3), x - 3 + 2, 50 - 3 + 2 + jig, PAL['K'])
        put_mask(cv, dilate(np.pad(m, 2), 2), x - 2, 50 - 2 + jig, PAL['W'])
        put_mask(cv, m, x, 50 + jig, title_grad(m.shape[0], '#fff080', '#ff4020'))
    dialog_box(cv, 'b3', t)
    if lt < 0.1:
        cv[:] = 255
    else:
        cv[:] = np.roll(cv, (int(lt * 60) % 3 - 1) * (lt < 0.6), axis=0)


# ------------------------------------------------------------------ TALLY (sunset)
def render_tally(cv, t):
    lt = t - T_TALLY
    cam = 316
    vimg = tint_img(van(), PALETTES['sunset']['tint'])
    draw_world(cv, 'sunset', cam, t, vimg, sun=(122, 100, 20))
    hx = VAN_WX - cam + 18
    himg = tint_img(hero('sit', 'sit', lt * 1.3, 0.9 + 0.2 * math.sin(lt * 2), 0.4), (1.0, 0.8, 0.78))
    blit(cv, himg, hx, VAN_Y - 21 + (1 if int(lt * 1.5) % 2 else 0))
    # panel
    px, py, pw, ph = 150, 22, 164, 136
    darken(cv, px, py, pw, ph, 0.45)
    rect(cv, px, py, pw, 1, PAL['Y']); rect(cv, px, py + ph - 1, pw, 1, PAL['Y'])
    rect(cv, px, py, 1, ph, PAL['Y']); rect(cv, px + pw - 1, py, 1, ph, PAL['Y'])
    if t >= TALLY['clear']:
        k = back_out((t - TALLY['clear']) / 0.35, 2.2)
        y = int(-20 + (32 + 20) * k)
        text_c(cv, 'STAGE CLEAR!', px + pw // 2, y, scale=2, grad=title_grad(14, '#fff8b0', '#ff9020'), outline=PAL['K'])
    rows = [('l0', 'CRIMES STOPPED', '0'), ('l1', 'BURRITOS LOST', '1'), ('l2', 'POSES STRUCK', None), ('l3', 'EXP GAINED', '0')]
    for i, (key, lab, val) in enumerate(rows):
        if t < TALLY[key]:
            continue
        y = 60 + i * 13
        text(cv, lab, px + 10, y, color=PAL['W'], shadow=PAL['K'])
        if val is None:
            n = int(47 * clamp01((t - TALLY['l2']) / (TALLY['l2end'] - TALLY['l2'])))
            val = str(n)
        dots = '.' * (22 - len(lab) - len(val))
        text(cv, dots, px + 10 + len(lab) * 6 + 2, y, color=hexc('#a080a0'))
        text(cv, val, px + 10 + 24 * 6 - len(val) * 6 - 6, y, color=PAL['Y'], shadow=PAL['K'])
    if t >= TALLY['label']:
        text_c(cv, 'HERO STATUS:', px + pw // 2, 116, color=PAL['L'], shadow=PAL['K'])
    if t >= TALLY['stamp']:
        st = t - TALLY['stamp']
        sc = max(1, 4 - int(st * 30))
        word = 'UNCONFIRMED'
        m = text_mask(word, sc)
        w = m.shape[1]
        cx, cy = px + pw // 2, 134
        x0, y0 = cx - w // 2, cy - m.shape[0] // 2
        rect(cv, x0 - 5, y0 - 4, w + 10, m.shape[0] + 8, hexc('#ff3050'))
        rect(cv, x0 - 4, y0 - 3, w + 8, m.shape[0] + 6, hexc('#2a0a1a'))
        put_mask(cv, m, x0, y0, hexc('#ff3050'))
        if st < 0.15:
            cv[:] = np.roll(cv, 2 if int(st * 60) % 2 else -2, axis=0)
    if lt < 0.2:
        fade(cv, lt / 0.2)
    if t > T_END - 0.25:
        fade(cv, (T_END - t) / 0.25)


# ------------------------------------------------------------------ END (night)
def render_end(cv, t):
    lt = t - T_END
    cam = 203
    light = t < END_LIGHT_OFF
    vimg = tint_img(van(night=True, light=light), PALETTES['night']['tint'], keep=(PAL['y'], PAL['O']))
    draw_world(cv, 'night', cam, t, vimg)
    if light:  # warm light spilling on the ground
        vx = VAN_WX - cam
        for yy in range(GROUND - 2, GROUND + 6):
            for xx in range(vx + 4, vx + 48):
                if (xx + yy) % 2 == 0 and ((xx - vx - 26) / 24) ** 2 + ((yy - GROUND - 1) / 5) ** 2 < 1:
                    if 0 <= xx < W and 0 <= yy < H:
                        cv[yy, xx] = np.minimum(255, cv[yy, xx].astype(int) + [70, 50, 10])
    else:
        for i in range(3):
            zt = t - END_LIGHT_OFF - 0.1 - i * 0.25
            if zt > 0:
                text(cv, 'Z', VAN_WX - cam + 44 + zt * 10, VAN_Y + 2 - zt * 22, color=hexc('#c8d0ff'), scale=1 + i % 2)
    # shooting star
    if E + 0.5 < t < E + 1.1:
        k = (t - E - 0.5) / 0.6
        sx, sy = 40 + 180 * k, 20 + 40 * k
        for j in range(8):
            rect(cv, sx - j * 3, sy - j * 0.66, 2, 1, (np.array([255, 255, 255]) * (1 - j / 8)).astype(np.uint8))
    # titles
    if lt > 0.15:
        k = ease_out((lt - 0.15) / 0.4)
        a = int(clamp01(k) * 1)
        text_c(cv, 'STRAWBERRY MAN', 160, int(22 - 10 * (1 - k)), scale=2, grad=GRAD_RED[::2][:14] if len(GRAD_RED[::2]) >= 14 else title_grad(14, '#ffb0b8', '#c0102c'), outline=PAL['K'])
    for key, y in (('end1', 46), ('end2', 60)):
        lines, _ = typed(key, t)
        text_c(cv, lines[0].ljust(len(DIALOG[key][2][0])), 160, y, color=PAL['W'], shadow=PAL['K'])
    if lt < 0.25:
        fade(cv, lt / 0.25)
    if t >= END_FADE:
        fade(cv, (END_FIN - t) / 0.3)
    if t >= END_FIN:
        cv[:] = 0
        lines, _ = typed('fin', t)
        for i, ln in enumerate(lines):
            text_c(cv, ln.ljust(len(DIALOG['fin'][2][i])), 160, 78 + i * 12, color=hexc('#c8c8d8'))
        if t > END_FIN + 0.9:
            blit(cv, scale(STRAWBERRY_ICON, 2), 152, 108)


def render(t):
    cv = np.zeros((H, W, 3), np.uint8)
    if t < T_TITLE:
        render_boot(cv, t)
    elif t < T_BEACH:
        render_title(cv, t)
    elif t < T_PROF:
        render_beach(cv, t)
    elif t < T_BATTLE:
        render_profile(cv, t)
    elif t < T_TALLY:
        render_battle(cv, t)
    elif t < T_END:
        render_tally(cv, t)
    else:
        render_end(cv, t)
    return cv
