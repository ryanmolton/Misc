# Dance of the Sugar Plum Fairy — Gadget Mix

Tchaikovsky's *Dance of the Sugar Plum Fairy* (1892, public domain), re-scored for a bank of clean, modern chip-synth "gadgets". 2/4, E minor, 80 BPM, about 1:14.

## Files

| File | Use |
|---|---|
| `sugar_plum_gadgets.mp3` | Finished mix, ready to drop on the animatic |
| `sugar_plum_gadgets.wav` | Same mix as uncompressed audio, for the final edit |
| `sugar_plum_gadgets.mid` | Multitrack MIDI. In GarageBand use **File → Open** and pick your own instruments |
| `stems/*.wav` | The mix split by part. Drag them into GarageBand and they line up from 0:00 |
| `make_sugar_plum.py` | The generator. Edit the score or sounds and re-run `python3 make_sugar_plum.py` |

## Cue sheet (for syncing the animation)

| Bars | Time | What happens |
|---|---|---|
| 1–3 | 0:00 | **Switch flip** (click and clunk at 0:00.2), then a power-up ripple and an E-major "system ready" chime and data blips |
| 4–7 | 0:04.5 | Pizzicato pulse intro, played by the synth bass and off-beat chords |
| 8–15 | 0:10.5 | Main celesta theme. The bass-clarinet run in bar 13 becomes an upward data sweep |
| 16–23 | 0:22.5 | Theme again, with a cadence |
| 24–31 | 0:34.5 | Bright middle: cascading arpeggios over G – C – D – G major chords |
| 32–46 | 0:46.5 | Reprise with the full gadget band: duet harmony line and soft rhythm ticks |
| 47–48 | 1:10.5 | Ends on a bright **E major** chord ("signal locked") instead of the original's minor ending |

Each bar is 1.5 s long, so bar *N* starts at (N − 1) × 1.5 s.

## GarageBand tips

* MIDI: GarageBand ignores the General MIDI instrument names. Good matches are Synthesizer → Lead → *Square Lead* / *Retro Synth* for the melody, *Celesta* or a bell patch for the sparkle arps, and *Analog Mono* / *Retro Synth* for the bass.
* Tempo (80) and time signature (2/4) come in with the MIDI file. If you only use the audio stems, set the project to 80 BPM and 2/4 so the grid matches.
