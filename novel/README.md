# The Long Exposure

A novel. ~41,500 words (about 140 book pages), 22 chapters plus a historical
afterword.

Zermatt, 1897. A young stereograph photographer from the San Bernardino
Valley is sent to the Alps to finish the catalogue a vanished legend left
half-made — and to find out what the old man was really photographing, eleven
times, on a worthless stretch of glacier.

## Files

- **`The-Long-Exposure.epub`** — ready-made ebook. This is the easiest path:
  just drag it into Calibre (or send it straight to a Kindle). It has a cover,
  a full table of contents, and one file per chapter, so chapter navigation
  works out of the box.
- `the-long-exposure.html` — the whole book as a single HTML file, as a
  fallback Calibre input (chapters are `<h1>` headings, which Calibre's
  default chapter detection picks up) or for reading in a browser.
- `manuscript/` — the source text, one Markdown file per chapter.
- `build/build_epub.py` — rebuilds the EPUB and HTML from `manuscript/`
  (`python3 build/build_epub.py`; needs Pillow for the cover).

## Kindle via Calibre

1. Add `The-Long-Exposure.epub` to Calibre.
2. Convert to AZW3/KFX, or use *Send to device* — no special settings needed;
   the TOC and chapter breaks are already in the file.
3. The author field is set to "Hollis Joiner," the novel's narrator. Edit the
   metadata in Calibre if you'd rather file it under something else.
