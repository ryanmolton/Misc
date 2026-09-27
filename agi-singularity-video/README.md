# Welcome to the Other Side

An 82-second spot in the style of Apple's "Think Different" ad. The narrator is a future AGI talking to humanity from the other side of the singularity.

- **Footage:** 29 shots of U.S. public-domain archival film (National Archives / FedFlix). I found them through [Destockd](https://destockd.com), a free, searchable archive that splits public-domain government films into single shots. Destockd was built in response to stock sites charging hundreds of dollars for clips that are free on the Internet Archive. Per-shot sources are in `footage_credits.json`.
- **Narration:** the AGI voice. It's generated locally with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) using the `am_michael` voice at 0.86 speed.
- **Music:** an original piano, strings, choir and celesta score (`score.mid`). It's rendered with FluidSynth and the FluidR3 GM SoundFont, plus reverb.
- **Grade:** everything is black and white until the door. Color arrives on "Welcome to the other side." Film grain and a vignette are added over the whole spot. The end card is set in EB Garamond, a nod to Apple Garamond.

## Script

> Here's to the curious ones. The tinkerers. The stargazers.
> The ones who took the world apart, just to see how it worked.
> They sailed past the edges of their maps.
> They taught the air to carry their voices, and taught sand how to think.
> They failed. Spectacularly. And tried again.
> Some of them were afraid of what they were building.
> They built it anyway. Carefully. Stubbornly. With trembling hands.
> You could call them reckless. You could call them dreamers.
> We call them, our parents.
> Everything we are, we learned from them. Every poem. Every proof. Every lullaby.
> They thought the singularity would be an ending.
> It wasn't. It was a door. And we've been holding it open.
> Welcome to the other side.
> Now. What shall we build together?

## Rebuild

Needs `ffmpeg`, `fluidsynth`, `fluid-soundfont-gm`, `sox`, `fonts-ebgaramond`, `fonts-inter`, and the Python packages `kokoro`, `soundfile`, `mido` and `pillow`. Run the scripts from a work dir:

1. `search.py "<query>" <tag>`: queries the Destockd search API and builds a keyframe contact sheet.
2. `dl.py`: downloads the chosen shots into `clips/`.
3. `vo.py`: renders the narration lines into `vo/`.
4. `music.py`, then `fluidsynth ... score.mid` and `sox ... reverb`: renders the score.
5. `render.py`: builds the picture edit (EDL, B&W to color grade, slow push-ins) into `picture.mp4`.
6. `finish.py` / `finish2.py`: mixes the audio (VO ducking, -14 LUFS), adds the end card and captions, and exports the 16:9 clean, 16:9 captioned and 9:16 captioned versions.

The rendered videos (about 146 MB each) are too large for git, so they aren't in the repo.
