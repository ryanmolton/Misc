"""Sprites and pixel-art props for Strawberry Man."""
import numpy as np
from gfx import PAL, sprite, outline, dilate, shift, hexc

HEAD = [
    ".........Gg.........",
    "....g..gGGGGg..g....",
    "....GggGGggGGggG....",
    "...KKGGGgKKgGGGKK...",
    "..KRRKKgKRRKgKKRRK..",
    ".KRRPRRKRRRRKRRYRRK.",
    ".KRPRRYRRRRRRRRRRRK.",
    "KRRRRRRRRRYRRRRRRRRK",
    "KRYRRKKKKKKKKKKRRYRK",
    "KRRRKSSSSSSSSSSKRRRK",
    "KRRRKMMMMMMMMMMKRRRK",
    "KRRYKMWWKMMWWKMKRRRK",
    "KRRRKSSSSSSSSSSKRYRK",
    ".KRRKSSKSSSSKSSKRRK.",
    ".KRRRKSSKKKKSSKRRRK.",
    "..KRRYKKSSSSKKRRRK..",
    "...KRRRRKKKKRRYRK...",
    "....KKRRRRRYRRKK....",
    "......KKRRRRKK......",
    "........KKKK........",
]
TORSO = [
    "......KRRRRRRK......",
    ".....KRRRYYRRRK.....",
    "....KRKRRYYRRKRK....",
    "....KRKRRRRRRKRK....",
    "....KWKBBBBBBKWK....",
    ".....KKRRRRRRKK.....",
]
TORSO_HIPS = [
    "......KRRRRRRK......",
    ".....KRRRYYRRRK.....",
    "...KRRKRRYYRRKRRK...",
    "..KRK.KRRRRRRK.KRK..",
    "...KWKKBBBBBBKKWK...",
    ".....KKRRRRRRKK.....",
]
TORSO_FIST = [  # right arm removed; drawn raised separately
    "......KRRRRRRK......",
    ".....KRRRYYRRRKK....",
    "....KRKRRYYRRRK.....",
    "....KRKRRRRRRK......",
    "....KWKBBBBBBK......",
    ".....KKRRRRRRKK.....",
]
LEGS = {
    'stand': ["......KRRKKRRK......", "......KRRKKRRK......", "......KGGKKGGK......", ".....KKGGKKGGKK....."],
    'runA': ["......KRRKKRRK......", ".....KRRK..KRRK.....", "....KGGK....KGGK....", "....KKK......KKK...."],
    'runB': ["......KRRKKRRK......", ".......KRRRRK.......", ".......KGGGGK.......", "......KKGKKGKK......"],
    'jump': [".....KRRK..KRRK.....", "....KRRK....KRRK....", "....KGGK....KGGK....", "....KKKK....KKKK...."],
    'sit':  [".....KRRRRRRRRK.....", ".....KRRK..KRRK.....", ".....KGGK..KGGK.....", ".....KKKK..KKKK....."],
}
PAD = 4  # side padding so raised arm and cape fit
HW, HH = 20 + 2 * PAD, 30


def _grid(rows):
    return [list('.' * PAD + r + '.' * PAD) for r in rows]


def _cape(phase, wind, flare):
    """Procedural cape (behind body) as RGBA, 28x31."""
    img = np.zeros((HH + 1, HW, 4), np.uint8)
    for y in range(20, HH + 1):
        k = (y - 20) / 10.0
        sway = np.sin(phase * 2 * np.pi + k * 3.0) * 1.2 * k
        l = 10 - (3 + flare) * k + wind * 6 * k * k + sway
        r = 17 + (3 + flare) * k + wind * 6 * k * k + sway
        l, r = int(round(l)), int(round(r))
        for x in range(max(0, l), min(HW, r + 1)):
            edge = x in (l, l + 1, r - 1, r) or (y == HH and (x + int(phase * 8)) % 3 == 0)
            c = PAL['g'] if edge else PAL['G']
            img[y, x, :3] = c
            img[y, x, 3] = 255
    return img


_cache = {}


def hero(pose='stand', legs=None, phase=0.0, wind=0.0, flare=0.0):
    key = (pose, legs, round(phase * 8) % 8, round(wind, 1), round(flare, 1))
    if key in _cache:
        return _cache[key]
    torso = {'stand': TORSO, 'hips': TORSO_HIPS, 'fist': TORSO_FIST}.get(pose, TORSO)
    legs = legs or ('stand' if pose != 'sit' else 'sit')
    body = sprite([''.join(r) for r in _grid(HEAD + torso + LEGS[legs])])
    if pose == 'fist':
        # raised right arm: upper arm out, forearm straight up, glove on top
        for (x, y, c) in ([(18, 21, 'R'), (19, 21, 'R'), (20, 20, 'R'), (21, 19, 'R'), (21, 20, 'R'), (22, 19, 'R')] +
                          [(xx, yy, 'R') for yy in range(7, 19) for xx in (23, 24)] +
                          [(xx, yy, 'W') for yy in range(3, 7) for xx in (22, 23, 24, 25)]):
            body[y, x, :3] = PAL[c]; body[y, x, 3] = 255
        body[3, 22, 3] = 0; body[3, 25, 3] = 0
        body = outline(body)[1:-1, 1:-1]
    img = np.zeros((HH + 3, HW + 2, 4), np.uint8)
    cape = outline(_cape(phase, wind, flare))
    m = cape[..., 3] > 0
    img[:cape.shape[0], :cape.shape[1]][m] = cape[m]
    b = np.pad(body, ((1, 1), (1, 1), (0, 0)))
    mb = b[..., 3] > 0
    img[:b.shape[0], :b.shape[1]][mb] = b[mb]
    _cache[key] = img
    return img


HERO_BACK = sprite([
    ".........Gg.........",
    "....g..gGGGGg..g....",
    "....GggGGggGGggG....",
    "...KKGGGgKKgGGGKK...",
    "..KRRKKgKRRKgKKRRK..",
    ".KRRYRRKRRRRKRRYRRK.",
    ".KRPRRRRRRYRRRRRRRK.",
    "KRRRRRYRRRRRRRYRRRRK",
    "KRYRRRRRRRRRRRRRRYRK",
    "KRRRRRRYRRRRYRRRRRRK",
    "KRPRRRRRRRRRRRRRRRRK",
    "KRRYRRRRRRRRRRYRRRRK",
    "KRRRRRRRYRRRRRRRRYRK",
    ".KRRRRRRRRRRRRRRRRK.",
    ".KRRRYRRRRRRRYRRRRK.",
    "..KRRRRRRYRRRRRRRK..",
    "...KRRRRRRRRRRRRK...",
    "....KKRRRYRRRRKK....",
    "...KgKKKRRRRKKKgK...",
    "..KgGGGGKKKKGGGGgK..",
    ".KWgGGGGGGGGGGGGgWK.",
    ".KKgGGGGGGGGGGGGgKK.",
    "..KgGGGGGGGGGGGGgK..",
    "..KgGGGGgGGgGGGGgK..",
    ".KggGGGGgGGgGGGGggK.",
    ".KgggggggggggggggggK",
])

GULL = sprite([
    "....KKKK..............",
    "...KWWWWK.............",
    "..KWWKWWWK............",
    "OOOKWWWWWK............",
    ".OOKWWWWWWK...........",
    "...KWWWWWWWKKKKK......",
    "...KWWWWWWWLLLLLKKK...",
    "...KWWWWWWLLLLLLLLLKK.",
    "....KWWWWWWLLDDDDDDDLK",
    "....KWWWWWWWLLDDDDDDDK",
    ".....KWWWWWWWLLDKDKDKK",
    "......KWWWWWWWWLLLKK..",
    ".......KKKKKKKKKKK....",
    "........KO...KO.......",
    ".......KOOK.KOOK......",
])
GULL_ANGRY = GULL.copy()
GULL_ANGRY[1, 4:8, :3] = PAL['K']  # furrowed brow

BURRITO = sprite([
    ".KKKKKKKKK.",
    "KTTtTTtTTGK",
    "KTtTTtTTTRK",
    "KTTTtTTtTYK",
    ".KKKKKKKKK.",
])

STRAWBERRY_ICON = sprite([
    "...Gg...",
    ".gGGGGg.",
    "KRRRRRRK",
    "KRYRRYRK",
    "KRRRRRRK",
    ".KRYRRK.",
    "..KRRK..",
    "...KK...",
])

HEART = sprite([
    ".KK.KK.",
    "KRRKRPK",
    "KRRRRRK",
    ".KRRRK.",
    "..KRK..",
    "...K...",
])

TICKET = sprite([
    "KKKKKKK",
    "KWWWWWK",
    "KWRRRWK",
    "KWWWWWK",
    "KWDDDWK",
    "KWWWWWK",
    "KKKKKKK",
])

SEAL = sprite([
    "......KKKK.....",
    ".....KDDDDK....",
    "....KDKDDDDK...",
    "...KDDDDDDZK...",
    "KKKDDDDDDDDDKKK",
    "KZDDDDDDDDDDDZK",
    ".KKKKKKKKKKKKK.",
])


def van(night=False, door_open=0.0, light=True):
    """Beat-up camper van, facing right. ~66x36 RGBA."""
    Wv, Hv = 64, 34
    c = np.zeros((Hv, Wv), dtype='<U1'); c[:] = '.'
    def fill(x0, y0, x1, y1, ch):
        c[y0:y1, x0:x1] = ch
    # surfboard on roof
    fill(6, 1, 56, 3, 'Y'); fill(8, 1, 54, 2, 'y'); fill(4, 2, 6, 3, 'Y'); fill(56, 2, 58, 3, 'Y')
    fill(14, 1, 17, 3, 'E'); fill(40, 1, 43, 3, 'E')
    fill(12, 3, 14, 5, 'Z'); fill(48, 3, 50, 5, 'Z')  # roof rack posts
    # body
    fill(2, 5, 60, 17, 'C')      # cream top
    fill(1, 17, 62, 27, 'Q')     # teal bottom
    fill(1, 17, 62, 18, 'q')
    fill(1, 26, 62, 27, 'q')
    fill(58, 9, 62, 17, 'C')     # nose
    fill(62, 19, 63, 25, 'Z')    # bumper
    fill(0, 19, 1, 25, 'Z')
    # windows
    wc = 'y' if (night and light) else ('N' if night else 'w')
    for x0 in (6, 20, 33):
        fill(x0, 8, x0 + 10, 15, wc)
    fill(47, 8, 58, 15, 'N' if night else 'w')  # windshield side window
    if not night:
        for x0 in (6, 20, 33):   # pink curtains
            fill(x0, 8, x0 + 3, 15, 'c'); fill(x0 + 7, 8, x0 + 10, 15, 'c')
        fill(48, 9, 50, 11, 'W')   # glare
    elif light:
        for x0 in (6, 20, 33):
            fill(x0, 8, x0 + 2, 15, 'O'); fill(x0 + 8, 8, x0 + 10, 15, 'O')
    # sliding door (between 19 and 32)
    fill(19, 6, 20, 27, 'z'); fill(32, 6, 33, 27, 'z')
    fill(29, 20, 31, 21, 'Z')  # handle
    # rust + dents + sticker
    for (x, y) in [(3, 24), (4, 25), (5, 24), (40, 25), (41, 25), (41, 24), (55, 18), (56, 19), (10, 22), (60, 23)]:
        c[y, x] = 'u'
    fill(44, 21, 52, 24, 'W'); c[22, 45:51] = 'R'  # bumper sticker
    fill(59, 20, 61, 22, 'y')  # headlight
    fill(1, 20, 2, 22, 'R')    # taillight
    # wheels
    img = sprite([''.join(r) for r in c], PAL)
    for cx in (13, 50):
        yy, xx = np.mgrid[0:Hv, 0:Wv]
        d = (xx - cx) ** 2 + (yy - 27) ** 2
        tire = d <= 30
        hub = d <= 6
        img[tire, :3] = PAL['N']; img[tire, 3] = 255
        img[hub, :3] = PAL['z']
        img[(d <= 1), :3] = PAL['L']
        img[(d > 30) & (yy < 27) & (abs(xx - cx) < 7) & (yy > 20), :3] = PAL['K']  # wheel arch
        img[(d > 30) & (yy < 27) & (abs(xx - cx) < 7) & (yy > 20), 3] = 255
    if door_open > 0:
        w = int(round(13 * door_open))
        img[6:26, 20:20 + w, :3] = PAL['K']
        img[6:26, 20:20 + w, 3] = 255
        if w > 2:
            img[20:26, 21:20 + w - 1, :3] = PAL['u']  # interior floor/rug
            img[8:12, 22:min(31, 20 + w), :3] = PAL['Y'] if not night else PAL['O']  # string lights
            img[8:12:2, 22:min(31, 20 + w):2, :3] = PAL['c']
    return outline(img)
