#!/usr/bin/env python3
"""Extract the Fig. 4 shuttle decals from the TOMS paper.

Renders page 9 of levtoms.pdf at 300 dpi, crops the figure, masks the
annotation colors (navy and red labels, arrows) and the white background
to magenta, then writes square pursuer and evader crops for MuJoCo.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

PAGE, DPI = 9, 300
FIG_BOX = (1080, 1460, 1870, 2190)          # Fig. 4 at 300 dpi
PURSUER_BOX = (400, 215, 560, 405)
EVADER_BOX = (85, 350, 200, 560)
KEY = (255, 0, 255)


def render_figure(pdf, workdir):
    import subprocess
    prefix = Path(workdir) / 'hi'
    subprocess.run(['pdftoppm', '-f', str(PAGE), '-l', str(PAGE), '-r', str(DPI),
                    '-png', str(pdf), str(prefix)], check=True)
    page = sorted(Path(workdir).glob('hi-*.png'))[0]
    return Image.open(page).convert('RGB').crop(FIG_BOX)


def mask(fig):
    a = np.asarray(fig).astype(int)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    annotations = ((B - R > 40) & (B - G > 30)) | ((R - B > 70) & (R - G > 70))
    background = (R + G + B) > 735
    out = a.copy()
    out[annotations | background] = KEY
    return Image.fromarray(out.astype(np.uint8))


def square_crop(img, box):
    im = img.crop(box)
    s = max(im.size)
    sq = Image.new('RGB', (s, s), KEY)
    sq.paste(im, ((s - im.size[0]) // 2, (s - im.size[1]) // 2))
    return sq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf', type=Path, required=True, help='levtoms.pdf')
    ap.add_argument('--out', type=Path, default=Path(__file__).parent / 'assets')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        fig = mask(render_figure(a.pdf, tmp))
    square_crop(fig, PURSUER_BOX).save(a.out / 'rocket_pursuer_key.png')
    square_crop(fig, EVADER_BOX).save(a.out / 'rocket_evader_key.png')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
