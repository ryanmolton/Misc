"""Scanned-print restoration helpers: dust/scratch detection, hole filling,
LaMa inpainting and tonal repair. All operate on the grey luminance image."""

import math

import torch
import torch.nn.functional as F


def _erode(x, k):
    return -F.max_pool2d(-x, k, 1, k // 2)


def _dilate(x, k):
    return F.max_pool2d(x, k, 1, k // 2)


def _robust_threshold(r, z):
    flat = r.flatten()
    if flat.numel() > 2_000_000:
        idx = torch.randint(0, flat.numel(), (2_000_000,), device=flat.device)
        flat = flat[idx]
    med = flat.median()
    mad = (flat - med).abs().median() * 1.4826
    return med + z * mad.clamp(min=1e-4)


def detect_defects(Y, sensitivity=0.5, max_width=4, polarity="both", min_contrast=0.06, grow=1):
    """Y: (1,1,H,W) luminance 0..1. Returns (1,1,H,W) mask of dust/scratch
    pixels: thin structures (narrower than ~max_width px) that stand out
    strongly from their surroundings (morphological top-hat), robustly
    thresholded against the image's own noise/grain level."""
    k = 2 * int(max_width) + 1
    r = torch.zeros_like(Y)
    if polarity in ("both", "bright (white specks & scratches)"):
        opened = _dilate(_erode(Y, k), k)
        r = torch.maximum(r, Y - opened)
    if polarity in ("both", "dark (black specks & hairs)"):
        closed = _erode(_dilate(Y, k), k)
        r = torch.maximum(r, closed - Y)
    z = 12.0 - 9.0 * float(sensitivity)
    t = torch.maximum(_robust_threshold(r, z), torch.tensor(min_contrast, device=Y.device))
    # Real texture (hair, foliage, fabric, lettering) is dense: many strong
    # top-hat responses close together. Dust and scratches are sparse. Keep
    # only responses that dominate their neighbourhood.
    # (the grain/noise floor is subtracted first so grain does not count as texture)
    floor = _robust_threshold(r, 3.0)
    excess = (r - floor).clamp(min=0)
    win = 4 * k + 1
    local = F.avg_pool2d(excess, win, 1, win // 2, count_include_pad=False)
    ratio = 3.0 + 7.0 * (1.0 - float(sensitivity))
    m = ((r > t) & (excess > ratio * local)).float()
    # drop isolated single pixels (film grain), keep specks and lines
    m = m * (F.avg_pool2d(m, 3, 1, 1) * 9.0 >= 2.5).float()
    if grow > 0:
        m = _dilate(m, 2 * int(grow) + 1)
    return m


def fast_fill(img, mask):
    """Push-pull pyramid fill: masked pixels are replaced with a smooth
    interpolation from their surroundings. Good for thin scratches and dust;
    use LaMa for large tears or missing corners."""
    known = (mask < 0.5).float()
    if known.min() > 0.5:
        return img
    pyr = []
    x, w = img * known, known
    while min(x.shape[-2:]) > 2:
        pyr.append((x, w))
        x = F.avg_pool2d(x, 2, ceil_mode=True)
        w = F.avg_pool2d(w, 2, ceil_mode=True)
    filled = x / w.clamp(min=1e-6)
    filled = torch.where(w > 0, filled, filled.mean(dim=(-2, -1), keepdim=True).expand_as(filled))
    for x, w in reversed(pyr):
        up = F.interpolate(filled, size=x.shape[-2:], mode="bilinear", align_corners=False)
        val = x / w.clamp(min=1e-6)
        filled = torch.where(w > 0.999, val, w * val + (1 - w) * up)
    return img * known + filled * (1 - known)


def lama_fill(model, img, mask, tile=1536, overlap=192):
    """Run a spandrel LaMa model on img/mask, tiling large images and only
    touching tiles that contain masked pixels. img (1,3,H,W), mask (1,1,H,W)."""
    _, _, H, W = img.shape
    hard = (mask > 0.5).float()
    out = img.clone()
    if hard.max() == 0:
        return out
    if H * W <= tile * tile:
        res = model(img, hard)
        return img * (1 - hard) + res * hard
    step = tile - overlap
    ny = max(1, math.ceil((H - overlap) / step))
    nx = max(1, math.ceil((W - overlap) / step))
    acc = torch.zeros_like(img)
    wsum = torch.zeros_like(mask)
    for iy in range(ny):
        for ix in range(nx):
            y0 = min(iy * step, max(0, H - tile))
            x0 = min(ix * step, max(0, W - tile))
            y1, x1 = min(H, y0 + tile), min(W, x0 + tile)
            m = hard[..., y0:y1, x0:x1]
            if m.max() == 0:
                continue
            res = model(img[..., y0:y1, x0:x1], m)
            # feathered weight so overlapping tiles blend
            wy = torch.ones(y1 - y0, device=img.device)
            wx = torch.ones(x1 - x0, device=img.device)
            ramp = torch.linspace(0.05, 1, overlap // 2, device=img.device)
            if y0 > 0: wy[: ramp.numel()] = ramp
            if y1 < H: wy[-ramp.numel():] = ramp.flip(0)
            if x0 > 0: wx[: ramp.numel()] = ramp
            if x1 < W: wx[-ramp.numel():] = ramp.flip(0)
            wt = (wy[:, None] * wx[None, :]).view(1, 1, y1 - y0, x1 - x0)
            acc[..., y0:y1, x0:x1] += res * wt
            wsum[..., y0:y1, x0:x1] += wt
    filled = acc / wsum.clamp(min=1e-6)
    use = hard * (wsum > 0).float()
    return img * (1 - use) + filled * use


def tone_curve(L, black_clip=0.1, white_clip=0.1, strength=1.0, midtone=1.0, contrast=0.0):
    """Global, monotonic tone repair on L (0..100): percentile-based black and
    white points (restores the range of a faded print), midtone gamma and a
    gentle S-curve. Being a single global curve, it cannot move edges or alter
    detail; it only redistributes brightness levels."""
    x = L / 100.0
    flat = x.flatten()
    if flat.numel() > 4_000_000:
        flat = flat[torch.randint(0, flat.numel(), (4_000_000,), device=flat.device)]
    lo = torch.quantile(flat.float(), black_clip / 100.0)
    hi = torch.quantile(flat.float(), 1.0 - white_clip / 100.0)
    if hi - lo > 0.05:
        stretched = ((x - lo) / (hi - lo)).clamp(0, 1)
        x = x + (stretched - x) * strength
    if midtone != 1.0:
        x = x.clamp(0, 1) ** (1.0 / midtone)
    if contrast != 0.0:
        # smooth S-curve about 0.5; contrast in -1..1
        k = 1.0 + 4.0 * abs(contrast)
        s = torch.sigmoid(k * (x - 0.5) * 2.0)
        s0, s1 = torch.sigmoid(torch.tensor(-k)), torch.sigmoid(torch.tensor(k))
        s = (s - s0) / (s1 - s0)
        x = x + (s - x) * (1.0 if contrast > 0 else -0.5)
    return (x.clamp(0, 1) * 100.0)
