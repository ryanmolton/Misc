"""Render STRAWBERRY MAN (30s, 1080p30) -> strawberry_man.mp4"""
import subprocess, sys, wave
import numpy as np
import imageio_ffmpeg
from music import *
from scenes import *

FPS, SCALE = 30, 6
OUT = sys.argv[1] if len(sys.argv) > 1 else 'strawberry_man.mp4'


def build_audio():
    mx = Mix(T_TOTAL)
    add = mx.add
    # --- boot + title
    add(0.92, sfx_ding(), 0.28)
    st = 0.095
    t0 = T_TITLE + 0.02
    play_melody(mx, t0, "C5:1 C5:1 C5:1 G5:3 E5:1 G5:1 C6:6 R:1 D6:1 E6:1 D6:1 E6:1 G6:5", st, duty=0.25, vol=0.2)
    play_melody(mx, t0, "E4:1 E4:1 E4:1 E4:3 C4:1 E4:1 G4:6 R:1 B4:1 C5:1 B4:1 C5:1 E5:5", st, duty=0.5, vol=0.08, pan=0.4)
    play_bass(mx, t0, [(12, 'C'), (6, 'G'), (6, 'C')], st, every=1)
    play_drums(mx, t0, "sss c.h.x.h.o.h.x.h.o.h.x.h", st, 1)
    add(T_TITLE + 1.85, sfx_select(), 0.22)
    # --- main theme: beach + profile
    step = 0.1
    t0 = T_BEACH
    lead = ("E5:2 G5:2 C6:3 G5:1 A5:2 G5:2 E5:4 "
            "C5:2 E5:2 A5:3 G5:1 E5:2 C5:2 A4:4 "
            "F5:2 A5:2 C6:3 D6:1 C6:2 A5:2 F5:4 "
            "G5:2 B5:2 D6:3 C6:1 B5:2 A5:2 G5:4 "
            "A5:2 C6:2 F6:2 E6:2 D6:2 B5:2 G5:2 D6:2 "
            "C6:2 G5:1 E5:1 C6:2 E6:2 G6:8")
    chords = [(16, 'C'), (16, 'Am'), (16, 'F'), (16, 'G'), (8, 'F'), (8, 'G'), (16, 'C')]
    cut = T_BATTLE + 0.02
    play_melody(mx, t0, lead, step, duty=0.25, vol=0.16, cut=cut)
    play_melody(mx, t0 + 0.3, lead, step, duty=0.125, vol=0.045, pan=0.6, cut=cut)  # echo
    play_arp(mx, t0, chords, step, cut=cut)
    play_bass(mx, t0, chords, step, cut=cut)
    play_drums(mx, t0, "x.h.o.h.x.hxo.h.", step, 6, vol=0.26, cut=cut)
    add(T_BEACH, waves(T_PROF - T_BEACH + 0.3), 0.10)
    # beach SFX
    for i in range(3):
        add(T_BEACH + 1.65 + i * 0.28, sfx_snore(0.24), 0.25, 0.3)
    add(T_BEACH + 2.45, sfx_door(), 0.3, 0.3)
    add(T_BEACH + 2.6, sfx_jump(), 0.18)
    add(T_BEACH + 3.1, sfx_land(), 0.5)
    add(T_BEACH + 3.15, sfx_sparkle(), 0.14)
    for key in ('beach1', 'beach2', 'b1', 'b2', 'b3', 'b4', 'b5'):
        pitch = 1320 if key.startswith('beach') else 990
        for tt in typed_blip_times(key):
            add(tt, sfx_blip(pitch), 0.05)
    # profile
    add(T_PROF, sfx_sweep(0.15), 0.08)
    for i in range(8):
        add(PROFILE_LINES_T0 + i * PROFILE_DT, sfx_coin() if i == 1 else sfx_blip(1760, 0.04), 0.12 if i != 1 else 0.14)
    add(PROFILE_LINES_T0 + 6 * PROFILE_DT, np.concatenate([sfx_powerup(), sfx_powerup()]), 0.13)
    # --- battle
    add(T_BATTLE, sfx_sweep(0.5), 0.2)
    bt0, bstep = 12.9, 0.085
    bcut = 16.6
    blead = ("A5:1 R:1 A5:1 C6:1 E6:2 A5:2 G5:1 A5:1 C6:2 B5:2 A5:2 "
             "F5:1 R:1 F5:1 A5:1 C6:2 F5:2 E5:1 F5:1 A5:2 C6:2 F6:2 "
             "G5:1 R:1 G5:1 B5:1 D6:2 G6:2 F6:1 E6:1 D6:2 B5:2 G5:2 "
             "E5:1 R:1 G#5:1 B5:1 E6:2 D6:2 C6:2 B5:2 G#5:4 "
             "A5:1 R:1 A5:1 C6:1 E6:2 A5:2 G5:1 A5:1 C6:2 B5:2 A5:2 "
             "F5:1 R:1 F5:1 A5:1 C6:2 F5:2 E5:1 F5:1 A5:2 C6:2 F6:2 ")
    CHORDS['E'] = [4, 8, 11]
    bchords = [(16, 'Am'), (16, 'F'), (16, 'G'), (16, 'E'), (16, 'Am'), (16, 'F')]
    play_melody(mx, bt0, blead, bstep, duty=0.5, vol=0.13, cut=bcut, legato=0.8)
    play_arp(mx, bt0, bchords, bstep, root=57, vol=0.06, cut=bcut)
    play_bass(mx, bt0, bchords, bstep, root=33, every=1, vol=0.3, cut=bcut)
    play_drums(mx, bt0, "x.hox.hox.hoxoho", bstep, 6, vol=0.26, cut=bcut)
    for (m0, m1, cur, sel) in MENU:
        add(m0, sfx_blip(1500, 0.03), 0.1)
        for tt, i in cur[1:]:
            add(tt, sfx_blip(1200, 0.03), 0.12)
        add(sel, sfx_select(), 0.2)
    add(15.33, sfx_buzz(), 0.22)
    # POSE!
    add(16.6, crash(), 0.35)
    add(16.6, sfx_powerup(), 0.16)
    add(16.95, sfx_sparkle(), 0.14)
    for m, v in ((60, 0.12), (64, 0.08), (67, 0.08), (72, 0.1)):
        s = pulse(freq(m), 1.2, 0.5, vib=0.01) * env(int(1.2 * SR), d=0.6, s=0.4, r=0.2)
        add(16.75, s, v)
    add(16.75, tri(freq(36), 1.2) * env(int(1.2 * SR), d=0.6, s=0.5, r=0.2), 0.35)
    for k in range(3):
        add(16.75 + k * 0.1, snare(), 0.25)
    add(18.1, sfx_sad_trombone(), 0.3)
    add(19.6, sfx_squawk(), 0.3, -0.2)
    for k in range(12):
        add(19.75 + k * 0.1, sfx_flap(), 0.18, min(0.8, k * 0.08))
    add(20.3, sfx_squawk()[::-1] * 0.5, 0.2, 0.6)
    # --- tally (sunset)
    add(T_TALLY, waves(T_END - T_TALLY + 0.5), 0.07)
    fs = 0.08
    play_melody(mx, TALLY['clear'], "G5:1 C6:1 E6:1 G6:3 E6:1 G6:6", fs, duty=0.25, vol=0.17, legato=0.95)
    play_melody(mx, TALLY['clear'], "E5:1 G5:1 C6:1 E6:3 C6:1 E6:6", fs, duty=0.5, vol=0.07, pan=0.4, legato=0.95)
    play_bass(mx, TALLY['clear'], [(3, 'C'), (3, 'C'), (7, 'C')], fs, every=3, octave_bounce=False)
    mt0, mst = 22.4, 0.12
    mch = [(16, 'F'), (16, 'C')]
    play_arp(mx, mt0, mch, mst, root=60, vol=0.05, pattern=(0, 1, 2, 3, 2, 1))
    play_bass(mx, mt0, mch, mst, every=4, vol=0.22, octave_bounce=False)
    play_melody(mx, mt0, "A5:4 C6:4 A5:2 G5:2 F5:4 E5:4 G5:4 C6:8", mst, duty=0.125, vol=0.09, legato=0.95)
    for key in ('l0', 'l1', 'l3'):
        add(TALLY[key], sfx_coin(), 0.13)
    n = 47
    for i in range(n):
        add(TALLY['l2'] + (TALLY['l2end'] - TALLY['l2']) * i / n, sfx_tick(), 0.07)
    add(TALLY['l2end'], sfx_coin(), 0.13)
    add(TALLY['label'], sfx_blip(990, 0.05), 0.1)
    add(TALLY['stamp'], sfx_stamp(), 0.5)
    # --- end (night)
    et0, est = 26.1, 0.13
    endlead = "E5:2 G5:2 C6:3 G5:1 A5:2 G5:2 E5:4 D5:2 F5:2 B5:2 D6:2 C6:10"
    play_melody(mx, et0, endlead, est, duty=0.125, vol=0.12, vib=0.01, legato=0.95)
    play_melody(mx, et0 + 0.26, endlead, est, duty=0.125, vol=0.04, pan=0.6, vib=0.01)
    play_arp(mx, et0, [(16, 'C'), (8, 'G'), (10, 'C')], est, root=48, vol=0.05, pattern=(0, 2, 1, 3))
    play_bass(mx, et0, [(16, 'C'), (8, 'G'), (10, 'C')], est, every=8, vol=0.22, octave_bounce=False)
    add(26.5, sfx_twinkle(), 0.08, 0.5)
    add(27.9, sfx_click(), 0.4)
    for i in range(3):
        add(28.0 + i * 0.25, sfx_snore(0.22), 0.12)
    for tt in typed_blip_times('end1') + typed_blip_times('end2'):
        add(tt, sfx_blip(1100), 0.04)
    L, R = lowpass(mx.L, 0.7), lowpass(mx.R, 0.7)  # tame naive-square aliasing
    # master: gentle soft clip + fade out
    peak = max(np.abs(L).max(), np.abs(R).max())
    g = 0.95 / peak if peak > 0.95 else 1.0
    L, R = np.tanh(L * g * 1.2) / np.tanh(1.2), np.tanh(R * g * 1.2) / np.tanh(1.2)
    fo = int(0.3 * SR)
    L[-fo:] *= np.linspace(1, 0, fo); R[-fo:] *= np.linspace(1, 0, fo)
    return np.stack([L, R], 1)


def write_wav(path, stereo):
    data = (np.clip(stereo, -1, 1) * 32000).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(data.tobytes())


def crt_mask():
    h, w = H * SCALE, W * SCALE
    m = np.ones((h, w), np.float32)
    m[SCALE - 1::SCALE, :] = 0.80          # scanline gap
    m[SCALE - 2::SCALE, :] *= 0.94
    yy, xx = np.mgrid[0:h, 0:w]
    v = 1 - 0.28 * (((xx - w / 2) / (w / 2)) ** 2 * 0.6 + ((yy - h / 2) / (h / 2)) ** 2 * 0.4) ** 1.4
    return (m * v * 256).astype(np.uint16)[..., None]


def main():
    audio_path = OUT.replace('.mp4', '.wav')
    print('synthesizing audio...', flush=True)
    write_wav(audio_path, build_audio())
    mask = crt_mask()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W * SCALE}x{H * SCALE}',
           '-r', str(FPS), '-i', '-', '-i', audio_path, '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
           '-tune', 'animation', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-shortest',
           '-movflags', '+faststart', OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(T_TOTAL * FPS)
    for i in range(n):
        f = render(i / FPS)
        big = f.repeat(SCALE, 0).repeat(SCALE, 1).astype(np.uint16)
        big = ((big * mask) >> 8).astype(np.uint8)
        p.stdin.write(big.tobytes())
        if i % 60 == 0:
            print(f'frame {i}/{n}', flush=True)
    p.stdin.close()
    p.wait()
    print('done ->', OUT)


if __name__ == '__main__':
    main()
