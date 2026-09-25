"""NES-flavoured chiptune synth: 2 pulse channels, stepped triangle, noise drums, SFX."""
import numpy as np

SR = 44100
NOTE_IDX = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}


def midi(n):
    name, octv = n[:-1], int(n[-1])
    return 12 * (octv + 1) + NOTE_IDX[name]


def freq(n):
    m = midi(n) if isinstance(n, str) else n
    return 440.0 * 2 ** ((m - 69) / 12)


def parse(seq):
    """'C5:2 R:1 E5:4' -> [(start_step, dur_steps, note|None)]"""
    out, t = [], 0
    for tok in seq.split():
        n, d = tok.split(':')
        d = float(d)
        out.append((t, d, None if n == 'R' else n))
        t += d
    return out


class Mix:
    def __init__(self, dur):
        self.n = int(dur * SR)
        self.L = np.zeros(self.n)
        self.R = np.zeros(self.n)

    def add(self, t, sig, vol=1.0, pan=0.0):
        i = int(t * SR)
        if i >= self.n or i + len(sig) <= 0:
            return
        j0 = max(0, -i)
        sig = sig[j0:]
        i = max(0, i)
        k = min(len(sig), self.n - i)
        gl, gr = vol * np.sqrt(0.5 * (1 - pan)), vol * np.sqrt(0.5 * (1 + pan))
        self.L[i:i + k] += sig[:k] * gl
        self.R[i:i + k] += sig[:k] * gr


def env(n, a=0.004, d=0.08, s=0.7, r=0.03, sus_len=None):
    t = np.arange(n) / SR
    e = np.where(t < a, t / a, s + (1 - s) * np.exp(-(t - a) / d))
    rel = int(r * SR)
    if rel > 0 and n > rel:
        e[-rel:] *= np.linspace(1, 0, rel)
    return e


def pulse(f, dur, duty=0.5, vib=0.0, vib_delay=0.12, slide=0.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    fr = f * (1 + slide * t)
    if vib:
        fr = fr * (1 + vib * np.sin(2 * np.pi * 6.0 * t) * np.clip((t - vib_delay) * 8, 0, 1))
    ph = np.cumsum(fr) / SR
    return np.where((ph % 1.0) < duty, 1.0, -1.0)


def tri(f, dur):
    n = int(dur * SR)
    ph = (np.arange(n) * f / SR) % 1.0
    v = 1 - 4 * np.abs(ph - 0.5)          # -1..1
    return np.round(v * 7.5) / 7.5        # 4-bit stepped like the NES


_rng = np.random.default_rng(7)
NOISE = _rng.uniform(-1, 1, SR * 2)


def noise(dur, decay=0.05, hp=False, tone=1):
    n = int(dur * SR)
    src = NOISE[:n * tone:tone] if tone > 1 else NOISE[:n]
    src = np.repeat(src, tone)[:n] if tone > 1 else src
    if hp:
        src = np.diff(np.concatenate([[0], src]))
    return src * np.exp(-np.arange(n) / SR / decay)


def kick(dur=0.14):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 55 + 120 * np.exp(-t / 0.025)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.07)


def snare():
    return noise(0.14, 0.05, tone=2) * 0.9 + tri(190, 0.14) * np.exp(-np.arange(int(0.14 * SR)) / SR / 0.03) * 0.5


def hat():
    return noise(0.04, 0.012, hp=True) * 0.8


def crash():
    return noise(0.9, 0.3, hp=True) * 0.6


def lowpass(x, a=0.35):
    y = np.empty_like(x)
    acc = 0.0
    # vectorised one-pole via lfilter-free trick (chunked)
    from itertools import accumulate
    y = np.fromiter(accumulate(x, lambda p, v: p + a * (v - p)), dtype=float, count=len(x))
    return y


CHORDS = {'C': [0, 4, 7], 'Am': [9, 12, 16], 'F': [5, 9, 12], 'G': [7, 11, 14], 'E': [4, 8, 11],
          'Dm': [2, 5, 9], 'G7': [7, 11, 17], 'Em': [4, 7, 11]}


def play_melody(mix, t0, seq, step, duty=0.25, vol=0.2, pan=-0.15, vib=0.006, octave=0, cut=None, legato=0.9):
    for s, d, n in parse(seq):
        if n is None:
            continue
        t = t0 + s * step
        if cut is not None and t >= cut:
            break
        dur = d * step * legato
        if cut is not None:
            dur = min(dur, cut - t)
        m = midi(n) + 12 * octave
        sig = pulse(freq(m), dur, duty, vib=vib if d * step > 0.25 else 0) * env(int(dur * SR), d=0.18, s=0.55)
        mix.add(t, sig, vol, pan)


def play_arp(mix, t0, chords, step, root=60, vol=0.07, pan=0.35, duty=0.125, cut=None, pattern=(0, 1, 2, 3)):
    """chords: list of (steps, name). 16th-note arpeggio."""
    s = 0
    for nsteps, name in chords:
        iv = CHORDS[name]
        tones = [iv[0], iv[1], iv[2], iv[0] + 12]
        for k in range(nsteps):
            t = t0 + (s + k) * step
            if cut is not None and t >= cut:
                return
            m = root + tones[pattern[k % len(pattern)]]
            dur = step * 0.85
            mix.add(t, pulse(freq(m), dur, duty) * env(int(dur * SR), d=0.05, s=0.3), vol, pan)
        s += nsteps


def play_bass(mix, t0, chords, step, root=36, vol=0.32, cut=None, every=2, octave_bounce=True):
    s = 0
    for nsteps, name in chords:
        r = root + CHORDS[name][0] % 12
        for k in range(0, nsteps, every):
            t = t0 + (s + k) * step
            if cut is not None and t >= cut:
                return
            m = r + (12 if (octave_bounce and (k // every) % 2) else 0)
            dur = step * every * 0.9
            mix.add(t, tri(freq(m), dur) * env(int(dur * SR), a=0.002, d=0.3, s=0.8, r=0.01), vol, 0)
        s += nsteps


def play_drums(mix, t0, pattern, step, bars, vol=0.3, cut=None):
    """pattern: string per 16 steps, k=kick s=snare h=hat c=crash x=kick+hat o=snare+hat"""
    for b in range(bars):
        for i, ch in enumerate(pattern):
            t = t0 + (b * len(pattern) + i) * step
            if cut is not None and t >= cut:
                return
            if ch in 'kx':
                mix.add(t, kick(), vol * 1.2)
            if ch in 'so':
                mix.add(t, snare(), vol * 0.7)
            if ch in 'hxo':
                mix.add(t, hat(), vol * 0.35, 0.2)
            if ch == 'c':
                mix.add(t, crash(), vol * 0.5)
                mix.add(t, kick(), vol * 1.2)


def plus(*sigs):
    n = max(len(x) for x in sigs)
    out = np.zeros(n)
    for x in sigs:
        out[:len(x)] += x
    return out


# ------------------------------------------------------------------ SFX
def sfx_blip(pitch=880, dur=0.025):
    return pulse(pitch, dur, 0.5) * env(int(dur * SR), a=0.001, d=0.02, s=0.2, r=0.005)


def sfx_select():
    a = pulse(freq('B5'), 0.06, 0.5)
    b = pulse(freq('E6'), 0.22, 0.5) * env(int(0.22 * SR), d=0.08, s=0.2)
    return np.concatenate([a, b])


def sfx_ding():  # console boot "ba-ding"
    a = pulse(freq('C6'), 0.07, 0.5) * 0.8
    b = pulse(freq('C7'), 0.6, 0.5) * env(int(0.6 * SR), a=0.002, d=0.25, s=0.0, r=0.05)
    return np.concatenate([a, b])


def sfx_buzz():
    return pulse(95, 0.35, 0.5) * 0.8 * env(int(0.35 * SR), d=1, s=1, r=0.03) + pulse(101, 0.35, 0.3) * 0.5


def sfx_jump():
    n = int(0.22 * SR)
    t = np.arange(n) / SR
    f = 300 * (1 + 3.5 * t / 0.22)
    return np.where((np.cumsum(f) / SR) % 1 < 0.25, 1.0, -1.0) * np.exp(-t / 0.15)


def sfx_land():
    return plus(kick(0.12), noise(0.08, 0.02, tone=4) * 0.5)


def sfx_door():
    n = int(0.35 * SR)
    return noise(0.35, 1.0, tone=6) * np.sin(np.linspace(0, np.pi, n)) * 0.6


def sfx_snore(dur=0.5):
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = noise(dur, 10, tone=12) * (0.5 + 0.5 * np.sin(2 * np.pi * 38 * t)) * np.sin(np.pi * t / dur)
    return s


def sfx_sparkle():
    out = []
    for m in (84, 88, 91, 96, 100, 103):
        out.append(pulse(freq(m), 0.035, 0.125) * env(int(0.035 * SR), d=0.03, s=0.5))
    return np.concatenate(out)


def sfx_powerup():
    out = []
    for i, m in enumerate([60, 64, 67, 72, 76, 79, 84, 88, 91, 96]):
        out.append(pulse(freq(m), 0.035, 0.25 if i % 2 else 0.5))
    return np.concatenate(out) * env(int(0.35 * SR) + 1, d=1, s=1, r=0.02)[:len(np.concatenate(out))]


def sfx_sweep(dur=0.5):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 200 + 1800 * (0.5 + 0.5 * np.sin(2 * np.pi * 9 * t)) * (1 - t / dur)
    return np.where((np.cumsum(f) / SR) % 1 < 0.5, 1.0, -1.0) * (1 - t / dur) * 0.7


def sfx_sad_trombone():
    parts = []
    for n, d in (('D#4', 0.28), ('D4', 0.28), ('C#4', 0.28), ('C4', 1.0)):
        k = int(d * SR)
        t = np.arange(k) / SR
        f = freq(n) * (1 - 0.03 * t / d) * (1 + (0.025 * np.sin(2 * np.pi * 7 * t) if d > 0.5 else 0))
        s = np.where((np.cumsum(f) / SR) % 1 < 0.5, 1.0, -1.0) * 0.6
        s += np.where((np.cumsum(f * 0.5) / SR) % 1 < 0.5, 1.0, -1.0) * 0.3
        e = np.minimum(1, t / 0.03) * np.minimum(1, (d - t) / 0.06)
        parts.append(s * e)
    return lowpass(np.concatenate(parts), 0.18)


def sfx_squawk():
    n = int(0.28 * SR)
    t = np.arange(n) / SR
    f = 1150 + 500 * np.sin(2 * np.pi * 14 * t) - 900 * t
    s = np.where((np.cumsum(f) / SR) % 1 < 0.3, 1.0, -1.0) * 0.5 + noise(0.28, 1, tone=3) * 0.35
    return s * np.sin(np.pi * t / 0.28) ** 0.5


def sfx_flap():
    return noise(0.07, 0.03, tone=5) * 0.7


def sfx_tick():
    return sfx_blip(1760, 0.018)


def sfx_coin():
    a = pulse(freq('B5'), 0.05, 0.5)
    b = pulse(freq('E6'), 0.3, 0.5) * env(int(0.3 * SR), d=0.1, s=0.1)
    return np.concatenate([a, b])


def sfx_stamp():
    return plus(kick(0.25) * 1.3, noise(0.2, 0.05, tone=3) * 0.8)


def sfx_click():
    return plus(noise(0.02, 0.004) * 1.2, pulse(1200, 0.01, 0.5) * 0.3)


def sfx_twinkle():
    return pulse(freq('E7'), 0.12, 0.125) * env(int(0.12 * SR), d=0.04, s=0.0)


def waves(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    src = _rng.uniform(-1, 1, n)
    lp = lowpass(src, 0.04)
    swell = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 3.2 - 1.2)) ** 2
    return lp * swell * 3.0
