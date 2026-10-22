#!/usr/bin/env python3
"""
"Dance of the Sugar Plum Fairy" (Tchaikovsky, 1892 - public domain)
re-scored for a bank of clean, modern chip-synth "gadgets".

One score (EVENTS) drives both outputs:
  * a rendered mix  -> sugar_plum_gadgets.wav / .mp3
  * per-part stems  -> stems/*.wav  (drag straight into GarageBand)
  * a multitrack MIDI file -> sugar_plum_gadgets.mid (GarageBand: File > Open)

Requires: numpy, mido, ffmpeg (for the MP3).
"""
import os
import subprocess
import numpy as np
import mido

SR = 44100
BPM = 80                 # quarter-note pulse; piece is in 2/4
BAR = 2.0                # beats per bar
SPB = 60.0 / BPM         # seconds per beat
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- score ----

NOTE_BASE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def n(name):
    """'A#5' -> MIDI number."""
    pc = NOTE_BASE[name[0]]
    i = 1
    while name[i] in '#b':
        pc += 1 if name[i] == '#' else -1
        i += 1
    return pc + 12 * (int(name[i:]) + 1)


EVENTS = []  # (part, start_beat, dur_beat, midi, velocity)


def add(part, start, dur, note, vel=100):
    EVENTS.append((part, start, dur, note if isinstance(note, int) else n(note), vel))


def eighths(part, bar, notes, vel=100, length=0.5):
    """Lay a bar of eighth notes ('-' = rest) starting at `bar`."""
    for i, nm in enumerate(notes):
        if nm != '-':
            add(part, bar * BAR + i * 0.5, length, nm, vel)


CHORDS = {
    'Em': ['E3', 'G3', 'B3'], 'G': ['D3', 'G3', 'B3'], 'Am': ['E3', 'A3', 'C4'],
    'B7': ['D#3', 'F#3', 'A3'], 'C': ['E3', 'G3', 'C4'], 'D': ['D3', 'F#3', 'A3'],
    'E': ['E3', 'G#3', 'B3'],
}
ROOTS = {'Em': ('E2', 'B2'), 'G': ('G2', 'D2'), 'Am': ('A2', 'E2'), 'B7': ('B1', 'F#2'),
         'C': ('C2', 'G2'), 'D': ('D2', 'A2'), 'E': ('E2', 'B2')}


def pizz(bar, chord, vel=100):
    """Tchaikovsky's pizzicato pulse: bass on the beat, chord on the off-beat."""
    root, fifth = ROOTS[chord]
    b = bar * BAR
    add('bass', b, 0.4, root, vel)
    add('bass', b + 1.0, 0.4, fifth, int(vel * 0.85))
    for tone in CHORDS[chord]:
        add('chords', b + 0.5, 0.2, tone, int(vel * 0.7))
        add('chords', b + 1.5, 0.2, tone, int(vel * 0.6))


# The celesta theme (bars 5ff. of the original), taken up an octave like the celesta.
THEME_A = [
    ['B5', 'G5', 'B5', 'A#5'],
    ['F#5', 'G5', '-', '-'],
    ['E5', 'E5', 'E5', 'D#5'],
    ['D5', 'D5', 'D5', 'C#5'],
    ['C5', 'C5', 'C5', 'B4'],
    ['-', '-', '-', '-'],          # (bass-clarinet swoop -> gadget sweep, below)
    ['B5', 'G5', 'B5', 'A#5'],
    ['F#5', 'G5', 'E5', '-'],
]
THEME_A2 = THEME_A[:5] + [
    ['A4', 'B4', 'C5', 'D#5'],
    ['E5', 'G5', 'F#5', 'D#5'],
    ['E5', '-', '-', '-'],
]
HARM_A = ['Em', 'Em', 'Em', 'G', 'Am', 'B7', 'Em', 'Em']
HARM_A2 = ['Em', 'Em', 'Em', 'G', 'Am', 'B7', 'Em', 'Em']


def theme(bar0, phrase, harm, harmony_line=False, hats=False):
    for i, notes in enumerate(phrase):
        eighths('lead', bar0 + i, notes, 100, 0.28)       # staccato, celesta-like
        if harmony_line:                                  # a second gadget a 3rd/6th below
            eighths('harmony', bar0 + i, [lower_third(x) for x in notes], 70, 0.28)
        pizz(bar0 + i, harm[i], 92)
        if hats:
            for k in range(4):
                add('perc', (bar0 + i) * BAR + k * 0.5 + 0.25, 0.1, 42, 45 if k % 2 else 30)
            add('perc', (bar0 + i) * BAR, 0.2, 36, 55)
    if phrase is THEME_A:
        # the bass-clarinet run, played as an upward data sweep
        sweep = ['E4', 'G4', 'B4', 'E5', 'G5', 'B5', 'E6']
        for k, nm in enumerate(sweep):
            add('sparkle', (bar0 + 5) * BAR + k * 0.25, 0.22, nm, 60 + k * 6)


E_MINOR_DOWN3 = {  # diatonic third below, E harmonic minor, for the duet line
    'B5': 'G5', 'G5': 'E5', 'A#5': 'F#5', 'F#5': 'D#5', 'E5': 'C5', 'D#5': 'B4',
    'D5': 'B4', 'C#5': 'A#4', 'C5': 'A4', 'B4': 'G4', 'A4': 'F#4',
}


def lower_third(nm):
    return '-' if nm == '-' else E_MINOR_DOWN3.get(nm, '-')


def arp_bar(bar, chord, top_note=None):
    """Bright middle section: 16th-note celesta-style arpeggios over major-leaning chords."""
    shapes = {
        'G': ['G4', 'B4', 'D5', 'G5', 'B5', 'G5', 'D5', 'B4'],
        'C': ['C5', 'E5', 'G5', 'C6', 'E6', 'C6', 'G5', 'E5'],
        'D': ['D5', 'F#5', 'A5', 'D6', 'F#6', 'D6', 'A5', 'F#5'],
        'Em': ['E5', 'G5', 'B5', 'E6', 'G6', 'E6', 'B5', 'G5'],
        'Am': ['A4', 'C5', 'E5', 'A5', 'C6', 'A5', 'E5', 'C5'],
        'B7': ['B4', 'D#5', 'F#5', 'A5', 'B5', 'A5', 'F#5', 'D#5'],
    }
    for k, nm in enumerate(shapes[chord]):
        add('sparkle', bar * BAR + k * 0.25, 0.2, nm, 78 if k == 0 else 64)
    if top_note:
        add('lead', bar * BAR, 1.8, top_note, 70)
    root, fifth = ROOTS[chord]
    add('bass', bar * BAR, 0.9, root, 90)
    add('bass', bar * BAR + 1.0, 0.9, fifth, 75)
    for tone in CHORDS[chord]:
        add('chords', bar * BAR + 0.5, 0.25, tone, 50)
        add('chords', bar * BAR + 1.5, 0.25, tone, 45)


def build_score():
    bar = 0
    # --- bars 0-2: the CEO flips the switch (FX), system boots in E major
    add('fx', 0.25, 0.1, 'C7', 110)                       # switch click
    for k, nm in enumerate(['E3', 'B3', 'E4', 'G#4', 'B4', 'E5', 'G#5', 'B5', 'E6']):
        add('fx', 0.75 + k * 0.125, 0.12, nm, 60 + k * 5)  # power-up ripple
    for k, nm in enumerate(['E5', 'G#5', 'B5', 'E6']):
        add('fx', 2.25 + k * 0.25, 1.5 - k * 0.25, nm, 85)  # "system ready" chime
    for k, nm in enumerate(['B6', 'E7', 'B6', 'G#6', 'E6', 'B5']):
        add('fx', 4.25 + k * 0.25, 0.12, nm, 40)           # data handshake blips
    bar = 3
    # --- intro: four bars of pizzicato pulse
    for i in range(4):
        pizz(bar + i, 'Em', 80 + i * 5)
    bar += 4
    theme(bar, THEME_A, HARM_A); bar += 8
    theme(bar, THEME_A2, HARM_A2); bar += 8
    # --- middle: the celesta's cascading arpeggios, brightened toward G major
    mid = [('G', 'B5'), ('C', 'C6'), ('D', 'A5'), ('G', 'B5'),
           ('Em', 'G5'), ('Am', 'A5'), ('B7', 'F#5'), ('B7', 'D#5')]
    for i, (ch, top) in enumerate(mid):
        arp_bar(bar + i, ch, top)
    bar += 8
    # --- reprise: full gadget band, duet line + light rhythm
    theme(bar, THEME_A, HARM_A, harmony_line=True, hats=True); bar += 8
    theme(bar, THEME_A2[:7], HARM_A2[:7], harmony_line=True, hats=True)
    bar += 7
    # --- ending: resolve to E MAJOR (bright Picardy third, "signal locked")
    for k, nm in enumerate(['E5', 'G#5', 'B5', 'E6', 'G#6', 'B6', 'E7']):
        add('sparkle', bar * BAR + k * 0.25, 0.25, nm, 70 + k * 4)
    add('lead', bar * BAR, 3.5, 'E5', 95)
    add('harmony', bar * BAR, 3.5, 'G#4', 70)
    add('bass', bar * BAR, 3.5, 'E2', 95)
    for tone in CHORDS['E']:
        add('chords', bar * BAR, 3.5, tone, 60)
    add('fx', bar * BAR + 2.0, 1.5, 'E7', 45)
    return bar * BAR + 4.0


# ---------------------------------------------------------------- synth ----

def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def polyblep(p, dt):
    out = np.zeros_like(p)
    dt = np.broadcast_to(dt, p.shape)
    lo = p < dt
    t = p[lo] / dt[lo]
    out[lo] = 2 * t - t * t - 1
    hi = p > 1 - dt
    t = (p[hi] - 1) / dt[hi]
    out[hi] = t * t + 2 * t + 1
    return out


def phase_of(freq):
    return np.cumsum(freq / SR) % 1.0


def pulse(freq, duty):
    """Band-limited pulse - the clean-chip core sound (no aliasing hash)."""
    p = phase_of(freq)
    dt = freq / SR
    saw1 = 2 * p - 1 - polyblep(p, dt)
    p2 = (p + duty) % 1.0
    saw2 = 2 * p2 - 1 - polyblep(p2, dt)
    return 0.5 * (saw1 - saw2)


def triangle(freq):
    p = phase_of(freq)
    out = np.zeros_like(p)
    for k in range(1, 12, 2):                              # band-limited, smooth
        out += ((-1) ** ((k - 1) // 2)) * np.sin(2 * np.pi * k * p) / (k * k)
    return out * 8 / np.pi ** 2


def env_adsr(n_samples, gate, a, d, s, r):
    t = np.arange(n_samples) / SR
    e = np.where(t < a, t / max(a, 1e-6),
                 s + (1 - s) * np.exp(-(t - a) / max(d, 1e-6)))
    rel = t >= gate
    e_gate = np.interp(gate, t, e) if gate < t[-1] else e[-1]
    e[rel] = e_gate * np.exp(-(t[rel] - gate) / max(r, 1e-6))
    return e


def voice(part, note, vel, dur_s):
    f0 = mtof(note)
    amp = (vel / 127.0) ** 1.4
    if part in ('lead', 'harmony'):
        tail = 0.35
        N = int((dur_s + tail) * SR)
        t = np.arange(N) / SR
        vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.15) / 0.2, 0, 1)
        f = f0 * vib
        duty = 0.25 if part == 'lead' else 0.5
        tone = pulse(f, duty) * 0.55 + np.sin(2 * np.pi * phase_of(f)) * 0.45
        bell = np.sin(2 * np.pi * phase_of(2 * f)) * np.exp(-t / 0.12) * 0.35  # celesta "ting"
        e = env_adsr(N, dur_s, 0.004, 0.18, 0.45, 0.09)
        return (tone + bell) * e * amp * (0.5 if part == 'lead' else 0.26)
    if part == 'sparkle':
        N = int((dur_s + 0.4) * SR)
        t = np.arange(N) / SR
        tone = pulse(np.full(N, f0), 0.5) * 0.35 + np.sin(2 * np.pi * f0 * t) * 0.65
        tone += np.sin(2 * np.pi * 3 * f0 * t) * np.exp(-t / 0.05) * 0.2
        e = np.minimum(t / 0.002, 1) * np.exp(-t / 0.16)
        return tone * e * amp * 0.3
    if part == 'bass':
        N = int((dur_s + 0.25) * SR)
        e = env_adsr(N, dur_s, 0.003, 0.22, 0.25, 0.06)
        return triangle(np.full(N, f0)) * e * amp * 0.26
    if part == 'chords':
        N = int((dur_s + 0.12) * SR)
        t = np.arange(N) / SR
        tone = pulse(np.full(N, f0), 0.5) * 0.5 + np.sin(2 * np.pi * f0 * t) * 0.5
        e = env_adsr(N, dur_s, 0.003, 0.08, 0.3, 0.05)
        return tone * e * amp * 0.055
    if part == 'perc':
        rng = np.random.default_rng(note * 1000 + vel)
        if note == 42:                                   # tiny clean "data tick"
            N = int(0.06 * SR)
            t = np.arange(N) / SR
            x = rng.standard_normal(N)
            x = np.diff(x, prepend=0)                    # brighten
            return x * np.exp(-t / 0.012) * amp * 0.06
        N = int(0.25 * SR)                               # soft sine kick
        t = np.arange(N) / SR
        f = 45 + 90 * np.exp(-t / 0.03)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.09) * amp * 0.35
    if part == 'fx':
        if note == n('C7'):                              # the big switch: click + clunk
            N = int(0.3 * SR)
            t = np.arange(N) / SR
            rng = np.random.default_rng(7)
            click = rng.standard_normal(N) * np.exp(-t / 0.004)
            clunk = np.sin(2 * np.pi * 70 * t) * np.exp(-t / 0.07)
            return (click * 0.2 + clunk * 0.4) * amp
        N = int((dur_s + 0.5) * SR)
        t = np.arange(N) / SR
        tone = pulse(np.full(N, f0), 0.25) * 0.4 + np.sin(2 * np.pi * f0 * t) * 0.6
        e = env_adsr(N, dur_s, 0.003, 0.3, 0.5, 0.25)
        return tone * e * amp * 0.25
    raise ValueError(part)


def lowpass_fft(x, fc):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X /= np.sqrt(1 + (f / fc) ** 4)                      # gentle 2-pole roll-off
    return np.fft.irfft(X, len(x))


def highpass_fft(x, fc):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / np.sqrt(1 + (fc / np.maximum(f, 1e-3)) ** 4)
    return np.fft.irfft(X, len(x))


def fft_convolve(x, h):
    L = len(x) + len(h) - 1
    nfft = 1 << (L - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(h, nfft), nfft)
    return y[:len(x)]


def make_reverb_ir(seconds=1.4, seed=0):
    rng = np.random.default_rng(seed)
    N = int(seconds * SR)
    t = np.arange(N) / SR
    ir = rng.standard_normal(N) * np.exp(-t / 0.35)
    ir = lowpass_fft(ir, 5000)
    ir[: int(0.015 * SR)] = 0                            # pre-delay
    return ir / np.sqrt(np.sum(ir ** 2))


def delay(x, seconds, gain):
    d = int(seconds * SR)
    y = np.zeros_like(x)
    y[d:] = x[:-d] * gain
    return y


PANS = {'lead': 0.0, 'harmony': -0.25, 'sparkle': 0.3, 'bass': 0.0,
        'chords': -0.3, 'perc': 0.15, 'fx': 0.0}
LP = {'lead': 7000, 'harmony': 5500, 'sparkle': 9000, 'bass': 2500,
      'chords': 3500, 'perc': 14000, 'fx': 9000}
STEM_GROUPS = {
    '1_melody': ['lead', 'harmony'],
    '2_sparkle_arps': ['sparkle'],
    '3_bass_and_chords': ['bass', 'chords'],
    '4_rhythm_ticks': ['perc'],
    '5_power_up_fx': ['fx'],
}


def render(total_beats):
    total = int((total_beats * SPB + 2.5) * SR)
    dry = {p: np.zeros(total) for p in PANS}
    for part, start, dur, note, vel in EVENTS:
        s = int(start * SPB * SR)
        v = voice(part, note, vel, dur * SPB)
        e = min(total, s + len(v))
        dry[part][s:e] += v[: e - s]

    ir_l, ir_r = make_reverb_ir(seed=1), make_reverb_ir(seed=2)
    send = {'lead': 0.22, 'harmony': 0.22, 'sparkle': 0.3, 'bass': 0.06,
            'chords': 0.15, 'perc': 0.08, 'fx': 0.3}
    stems = {}
    for part, x in dry.items():
        x = lowpass_fft(x, LP[part])
        if part != 'bass' and part != 'perc':
            x = highpass_fft(x, 120)
        pan = PANS[part]
        L = x * np.sqrt((1 - pan) / 2)
        R = x * np.sqrt((1 + pan) / 2)
        if part in ('lead', 'sparkle'):                  # dotted-8th ping-pong echo
            echo = 0.75 * SPB
            L += delay(x, echo, 0.22)
            R += delay(x, 2 * echo, 0.14)
        L = L + fft_convolve(L, ir_l) * send[part]
        R = R + fft_convolve(R, ir_r) * send[part]
        stems[part] = np.stack([L, R], axis=1)
    return stems


def write_wav(path, stereo):
    import wave
    data = np.clip(stereo, -1, 1)
    pcm = (data * 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ----------------------------------------------------------------- MIDI ----

MIDI_TRACKS = [  # part, track name, channel, GM program (0-based)
    ('lead', 'Melody (Square Lead)', 0, 80),
    ('harmony', 'Duet Harmony', 1, 80),
    ('sparkle', 'Sparkle Arps (Celesta)', 2, 8),
    ('bass', 'Pizz Bass (Synth Bass)', 3, 38),
    ('chords', 'Off-beat Chords', 4, 81),
    ('fx', 'Power-Up FX', 5, 98),
    ('perc', 'Data Ticks (Drums)', 9, 0),
]


def write_midi(path):
    tpb = 480
    mid = mido.MidiFile(ticks_per_beat=tpb, type=1)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage('track_name', name='Sugar Plum Gadgets', time=0))
    meta.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(BPM), time=0))
    meta.append(mido.MetaMessage('time_signature', numerator=2, denominator=4, time=0))
    meta.append(mido.MetaMessage('key_signature', key='Em', time=0))
    mid.tracks.append(meta)
    for part, name, ch, prog in MIDI_TRACKS:
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage('track_name', name=name, time=0))
        if ch != 9:
            tr.append(mido.Message('program_change', program=prog, channel=ch, time=0))
        msgs = []
        for p, start, dur, note, vel in EVENTS:
            if p != part:
                continue
            on = int(round(start * tpb))
            off = int(round((start + dur) * tpb))
            msgs.append((on, 1, mido.Message('note_on', note=note, velocity=vel, channel=ch)))
            msgs.append((off, 0, mido.Message('note_off', note=note, velocity=0, channel=ch)))
        msgs.sort(key=lambda m: (m[0], m[1]))
        now = 0
        for tick, _, m in msgs:
            tr.append(m.copy(time=tick - now))
            now = tick
        mid.tracks.append(tr)
    mid.save(path)


# ----------------------------------------------------------------- main ----

def main():
    total_beats = build_score()
    stems = render(total_beats)
    mix = sum(stems.values())
    peak = np.max(np.abs(mix))
    gain = 0.89 / peak                                   # -1 dBFS, no limiting/clipping
    os.makedirs(os.path.join(OUT, 'stems'), exist_ok=True)
    for stem_name, parts in STEM_GROUPS.items():
        write_wav(os.path.join(OUT, 'stems', f'{stem_name}.wav'),
                  sum(stems[p] for p in parts) * gain)
    wav = os.path.join(OUT, 'sugar_plum_gadgets.wav')
    write_wav(wav, mix * gain)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', wav, '-codec:a', 'libmp3lame',
                    '-b:a', '256k', '-metadata', 'title=Dance of the Sugar Plum Fairy (Gadget Mix)',
                    '-metadata', 'composer=P. I. Tchaikovsky', os.path.join(OUT, 'sugar_plum_gadgets.mp3')],
                   check=True)
    write_midi(os.path.join(OUT, 'sugar_plum_gadgets.mid'))
    print(f'duration {len(mix) / SR:.1f}s, {len(EVENTS)} notes, peak gain {gain:.2f}')


if __name__ == '__main__':
    main()
