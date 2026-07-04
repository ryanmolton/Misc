#!/usr/bin/env python3
"""Build The Long Exposure as a valid EPUB 3 (with NCX fallback) and a
single-file HTML edition, from the markdown chapters in ../manuscript.

Usage: python3 build_epub.py
Outputs: ../The-Long-Exposure.epub and ../the-long-exposure.html
"""
import re, os, zipfile, html, math

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'manuscript')
OUT_EPUB = os.path.join(HERE, '..', 'The-Long-Exposure.epub')
OUT_HTML = os.path.join(HERE, '..', 'the-long-exposure.html')
COVER = os.path.join(HERE, 'cover.png')

TITLE = 'The Long Exposure'
AUTHOR = 'Hollis Joiner'
LANG = 'en'
UID = 'urn:uuid:7e1f3a52-8c1b-4d5a-9f0e-2a6b1897zmtt'.replace('zmtt', '4c21')

EPIGRAPHS = [
    ("Climb if you will, but remember that courage and strength are nought "
     "without prudence, and that a momentary negligence may destroy the "
     "happiness of a lifetime. Do nothing in haste; look well to each step; "
     "and from the beginning think what may be the end.",
     "Edward Whymper, <i>Scrambles Amongst the Alps</i>, 1871"),
    ("Every view in this set is a Sun Sculpture, modelled by light itself, "
     "and will bear examination under the strongest lens.",
     "Stroud Brothers, <i>Catalogue of Stereoscopic Views</i>, 1896"),
]


def ital(s):
    s = html.escape(s)
    return re.sub(r'\*([^*]+)\*', r'<i>\1</i>', s)


def load_chapters():
    chapters = []
    for fn in sorted(os.listdir(SRC)):
        if not re.match(r'\d\d-', fn) or fn.startswith('00-'):
            continue
        raw = open(os.path.join(SRC, fn), encoding='utf-8').read().strip()
        lines = raw.split('\n')
        heading = lines[0].lstrip('# ').strip()
        body = '\n'.join(lines[1:]).strip()
        blocks, first_after_break = [], True
        for block in re.split(r'\n\s*\n', body):
            block = block.strip()
            if not block:
                continue
            if block == '* * *':
                blocks.append('<p class="brk">&#8258;</p>')
                first_after_break = True
                continue
            cls = ' class="noind"' if first_after_break else ''
            blocks.append('<p%s>%s</p>' % (cls, ital(block).replace('\n', ' ')))
            first_after_break = False
        chapters.append((heading, '\n'.join(blocks)))
    return chapters


CSS = '''
@page { margin: 1em; }
body { font-family: "Georgia", "Charter", serif; line-height: 1.45; margin: 0 4%; }
p { text-indent: 1.35em; margin: 0; text-align: justify; }
p.noind { text-indent: 0; }
p.brk { text-indent: 0; text-align: center; margin: 1.1em 0; color: #666; }
h1.chap { font-weight: normal; text-align: center; margin: 2.5em 0 0.2em 0;
          font-size: 0.95em; letter-spacing: 0.35em; color: #777; }
h2.chaptitle { font-weight: normal; font-style: italic; text-align: center;
               margin: 0 0 2.2em 0; font-size: 1.55em; }
.titlepage { text-align: center; margin-top: 28%; }
.titlepage h1 { font-size: 2em; letter-spacing: 0.12em; font-weight: normal; margin: 0; }
.titlepage .rule { width: 5em; border-bottom: 1px solid #999; margin: 1.4em auto; }
.titlepage p { text-indent: 0; text-align: center; font-style: italic; color: #555; }
.epi { margin: 18% 8% 0 8%; }
.epi p { text-indent: 0; text-align: left; font-style: italic; margin-bottom: 0.6em; }
.epi .attr { font-style: normal; font-size: 0.9em; color: #555; text-align: right;
             margin-bottom: 2.5em; }
'''

XHTML_HEAD = ('<?xml version="1.0" encoding="utf-8"?>\n'
              '<!DOCTYPE html>\n'
              '<html xmlns="http://www.w3.org/1999/xhtml" '
              'xmlns:epub="http://www.idpf.org/2007/ops" lang="en" xml:lang="en">\n'
              '<head><title>%s</title>'
              '<link rel="stylesheet" type="text/css" href="style.css"/></head>\n'
              '<body>\n')


def chapter_xhtml(idx, heading, body_html):
    m = re.match(r'(\d+)\.\s+(.*)', heading)
    if m:
        num, name = m.groups()
        head = ('<h1 class="chap">CHAPTER %s</h1>\n<h2 class="chaptitle">%s</h2>\n'
                % (num, html.escape(name)))
        title = '%s. %s' % (num, name)
    else:
        head = '<h2 class="chaptitle">%s</h2>\n' % html.escape(heading)
        title = heading
    return title, XHTML_HEAD % html.escape(title) + head + body_html + '\n</body>\n</html>\n'


def make_cover():
    from PIL import Image, ImageDraw, ImageFont
    W, H = 1600, 2560
    bg = (29, 42, 54)          # deep slate-ink blue
    gold = (198, 166, 106)     # old gilt
    ivory = (233, 226, 210)
    img = Image.new('RGB', (W, H), bg)
    d = ImageDraw.Draw(img)

    def font(path, size):
        return ImageFont.truetype(path, size)
    serif_b = '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf'
    serif_r = '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf'
    serif_i = '/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf'

    # double rule frame
    for inset, wd in ((70, 6), (92, 2)):
        d.rectangle([inset, inset, W - inset, H - inset], outline=gold, width=wd)

    def center(text, f, y, fill, tracking=0):
        if tracking:
            text = (' ' * 1).join(text)  # simple letterspacing via spaces
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, font=f, fill=fill)
        return y

    center('T H E', font(serif_r, 72), 330, ivory)
    center('LONG', font(serif_b, 230), 440, ivory)
    center('EXPOSURE', font(serif_b, 230), 700, ivory)
    center('A  N O V E L', font(serif_r, 58), 1030, gold)

    # twin stereo frames with offset mountain (parallax)
    fw, fh = 470, 360
    gap = 60
    x0 = (W - (2 * fw + gap)) // 2
    y0 = 1330
    for k in range(2):
        x = x0 + k * (fw + gap)
        d.rounded_rectangle([x, y0, x + fw, y0 + fh], radius=26, outline=gold, width=5)
        # horizon
        hy = y0 + fh - 96
        d.line([x + 36, hy, x + fw - 36, hy], fill=gold, width=3)
        # matterhorn-ish silhouette, offset slightly between frames
        off = -14 + k * 28
        cx = x + fw // 2 + off
        peak = [(cx - 150, hy), (cx - 60, hy - 130), (cx - 22, hy - 108),
                (cx + 30, hy - 232), (cx + 58, hy - 150), (cx + 96, hy - 170),
                (cx + 168, hy)]
        d.polygon(peak, fill=gold)
        # sun disc
        d.ellipse([cx - 208, y0 + 52, cx - 168, y0 + 92], outline=gold, width=4)
    cap = 'S U N   S C U L P T U R E'
    f = font(serif_i, 44)
    w = d.textlength(cap, font=f)
    d.text(((W - w) / 2, y0 + fh + 54), cap, font=f, fill=(140, 150, 158))

    center('H O L L I S   J O I N E R', font(serif_r, 64), 2260, ivory)
    img.save(COVER, 'PNG')


def build_epub():
    chapters = load_chapters()
    make_cover()

    spine_items, manifest_items, nav_lis, ncx_points = [], [], [], []
    files = {}

    files['cover.xhtml'] = (XHTML_HEAD % 'Cover' +
        '<div style="text-align:center"><img src="cover.png" alt="The Long Exposure" '
        'style="max-width:100%%;height:auto"/></div>\n</body>\n</html>\n')

    files['title.xhtml'] = (XHTML_HEAD % TITLE +
        '<div class="titlepage"><h1>THE LONG<br/>EXPOSURE</h1>'
        '<div class="rule"></div><p>a novel</p></div>\n</body>\n</html>\n')

    epi = ['<div class="epi">']
    for quote, attr in EPIGRAPHS:
        epi.append('<p>%s</p><p class="attr">&#8212; %s</p>' % (quote, attr))
    epi.append('</div>')
    files['epigraph.xhtml'] = XHTML_HEAD % 'Epigraph' + '\n'.join(epi) + '\n</body>\n</html>\n'

    toc = [('title.xhtml', 'Title Page'), ('epigraph.xhtml', 'Epigraph')]
    for i, (heading, body) in enumerate(chapters, 1):
        fn = 'ch%02d.xhtml' % i
        title, doc = chapter_xhtml(i, heading, body)
        files[fn] = doc
        toc.append((fn, title))

    manifest = ['<item id="css" href="style.css" media-type="text/css"/>',
                '<item id="cover-img" href="cover.png" media-type="image/png" properties="cover-image"/>',
                '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
                '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
                '<item id="cover" href="cover.xhtml" media-type="application/xhtml+xml"/>']
    spine = ['<itemref idref="cover" linear="yes"/>']
    for fn, _t in toc:
        iid = fn.split('.')[0]
        manifest.append('<item id="%s" href="%s" media-type="application/xhtml+xml"/>' % (iid, fn))
        spine.append('<itemref idref="%s"/>' % iid)

    nav_entries = '\n'.join('<li><a href="%s">%s</a></li>' % (fn, html.escape(t))
                            for fn, t in toc)
    files['nav.xhtml'] = (XHTML_HEAD % 'Contents' +
        '<nav epub:type="toc" id="toc"><h2 class="chaptitle">Contents</h2>\n<ol>\n'
        + nav_entries + '\n</ol>\n</nav>\n</body>\n</html>\n')

    ncx_points = '\n'.join(
        '<navPoint id="np%d" playOrder="%d"><navLabel><text>%s</text></navLabel>'
        '<content src="%s"/></navPoint>' % (i, i, html.escape(t), fn)
        for i, (fn, t) in enumerate(toc, 1))
    files['toc.ncx'] = ('<?xml version="1.0" encoding="utf-8"?>\n'
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">\n'
        '<head><meta name="dtb:uid" content="%s"/></head>\n'
        '<docTitle><text>%s</text></docTitle>\n<navMap>%s</navMap>\n</ncx>\n'
        % (UID, TITLE, ncx_points))

    files['style.css'] = CSS

    files['content.opf'] = ('<?xml version="1.0" encoding="utf-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">\n'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        '<dc:identifier id="uid">%s</dc:identifier>\n'
        '<dc:title>%s</dc:title>\n'
        '<dc:creator id="creator">%s</dc:creator>\n'
        '<dc:language>%s</dc:language>\n'
        '<dc:date>1897-08-22</dc:date>\n'
        '<meta property="dcterms:modified">2026-07-04T00:00:00Z</meta>\n'
        '<meta name="cover" content="cover-img"/>\n'
        '</metadata>\n<manifest>\n%s\n</manifest>\n<spine toc="ncx">\n%s\n</spine>\n'
        '</package>\n' % (UID, TITLE, AUTHOR, LANG,
                          '\n'.join(manifest), '\n'.join(spine)))

    with zipfile.ZipFile(OUT_EPUB, 'w') as z:
        z.writestr('mimetype', 'application/epub+zip', compress_type=zipfile.ZIP_STORED)
        z.writestr('META-INF/container.xml',
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" '
            'media-type="application/oebps-package+xml"/></rootfiles>\n</container>\n')
        for name, content in files.items():
            z.writestr('OEBPS/' + name, content, compress_type=zipfile.ZIP_DEFLATED)
        z.write(COVER, 'OEBPS/cover.png', compress_type=zipfile.ZIP_DEFLATED)
    print('wrote', OUT_EPUB)


def build_html():
    chapters = load_chapters()
    parts = ['<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"/>'
             '<title>%s</title><style>%s\nbody{max-width:38em;margin:0 auto;padding:2em 1em;}'
             'h1{font-weight:normal;text-align:center;margin-top:3em;}</style></head><body>' % (TITLE, CSS)]
    parts.append('<div class="titlepage"><h1>THE LONG EXPOSURE</h1>'
                 '<div class="rule"></div><p>a novel</p></div>')
    parts.append('<div class="epi">')
    for quote, attr in EPIGRAPHS:
        parts.append('<p>%s</p><p class="attr">&#8212; %s</p>' % (quote, attr))
    parts.append('</div>')
    for i, (heading, body) in enumerate(chapters, 1):
        parts.append('<h1>%s</h1>' % html.escape(heading))
        parts.append(body)
    parts.append('</body></html>')
    open(OUT_HTML, 'w', encoding='utf-8').write('\n'.join(parts))
    print('wrote', OUT_HTML)


if __name__ == '__main__':
    build_epub()
    build_html()
