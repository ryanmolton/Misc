"""Colour-space maths (sRGB <-> CIE Lab, D65) and helpers, all in torch.

Tensors are channels-first: (B, 3, H, W) for images, (B, 1, H, W) for L.
sRGB values are 0..1, L is 0..100, a/b are roughly -128..127.
"""

import torch
import torch.nn.functional as F

_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0
_WHITE = (0.95047, 1.0, 1.08883)

_RGB2XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_XYZ2RGB = torch.linalg.inv(_RGB2XYZ)

# Rows processed per chunk for full-resolution maths, to keep memory bounded
# on 50+ megapixel scans.
CHUNK_PIXELS = 4_000_000


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp(min=0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp(min=0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


def _mat(x, m):
    return torch.einsum("ij,bjhw->bihw", m.to(x), x)


def _f(t):
    return torch.where(t > _EPS, t.clamp(min=1e-12) ** (1.0 / 3.0), (_KAPPA * t + 16.0) / 116.0)


def _finv(t):
    t3 = t ** 3
    return torch.where(t3 > _EPS, t3, (116.0 * t - 16.0) / _KAPPA)


def rgb_to_lab(rgb):
    return linear_to_lab(srgb_to_linear(rgb))


def linear_to_lab(lin):
    xyz = _mat(lin, _RGB2XYZ)
    white = torch.tensor(_WHITE).to(lin).view(1, 3, 1, 1)
    f = _f(xyz / white)
    L = 116.0 * f[:, 1:2] - 16.0
    a = 500.0 * (f[:, 0:1] - f[:, 1:2])
    b = 200.0 * (f[:, 1:2] - f[:, 2:3])
    return torch.cat([L, a, b], 1)


def lab_to_linear(lab):
    L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    white = torch.tensor(_WHITE).to(lab).view(1, 3, 1, 1)
    xyz = torch.cat([_finv(fx), _finv(fy), _finv(fz)], 1) * white
    return _mat(xyz, _XYZ2RGB)


def lab_to_rgb(lab):
    return linear_to_srgb(lab_to_linear(lab)).clamp(0, 1)


def luminance_Y(rgb):
    """Relative luminance (linear) of an sRGB image, (B,1,H,W)."""
    lin = srgb_to_linear(rgb)
    w = _RGB2XYZ[1].to(rgb).view(1, 3, 1, 1)
    return (lin * w).sum(1, keepdim=True)


def rgb_to_L(rgb):
    return (116.0 * _f(luminance_Y(rgb)) - 16.0).clamp(0, 100)


def L_to_gray_rgb(L):
    """Neutral grey sRGB whose CIE L* equals L."""
    y = _finv((L + 16.0) / 116.0).clamp(0, 1)
    return linear_to_srgb(y).clamp(0, 1).expand(-1, 3, -1, -1)


def gamut_map(L, ab, iters=10):
    """Lab -> sRGB keeping L exactly: pixels whose colour falls outside sRGB
    have their chroma scaled down (hue kept) until they fit, instead of being
    clipped per channel (which would change their brightness)."""
    lin = lab_to_linear(torch.cat([L, ab], 1))
    bad = ((lin < -1e-4) | (lin > 1.0 + 1e-4)).any(1, keepdim=True)
    if bad.any():
        lo = torch.zeros_like(L)
        hi = torch.ones_like(L)
        for _ in range(iters):
            mid = (lo + hi) * 0.5
            t = lab_to_linear(torch.cat([L, ab * mid], 1))
            ok = ((t >= -1e-4) & (t <= 1.0 + 1e-4)).all(1, keepdim=True)
            lo = torch.where(ok, mid, lo)
            hi = torch.where(ok, hi, mid)
        scale = torch.where(bad, lo, torch.ones_like(L))
        lin = lab_to_linear(torch.cat([L, ab * scale], 1))
    return linear_to_srgb(lin.clamp(0, 1)).clamp(0, 1)


def row_chunks(h, w):
    rows = max(1, CHUNK_PIXELS // max(1, w))
    for y0 in range(0, h, rows):
        yield y0, min(h, y0 + rows)


def resize(x, h, w, mode="bilinear"):
    if x.shape[-2:] == (h, w):
        return x
    if mode == "area" or (mode == "bilinear" and (x.shape[-2] > 2 * h or x.shape[-1] > 2 * w)):
        return F.interpolate(x, size=(h, w), mode="area")
    return F.interpolate(x, size=(h, w), mode=mode, align_corners=False)


def box(x, r):
    """Mean filter with radius r (window 2r+1), edge-normalised."""
    if r <= 0:
        return x
    k = 2 * r + 1
    ones = torch.ones_like(x[:, :1])
    s = F.avg_pool2d(F.pad(x, (r, r, r, r), mode="constant"), k, 1)
    n = F.avg_pool2d(F.pad(ones, (r, r, r, r), mode="constant"), k, 1)
    return s / n.clamp(min=1e-6)


def guided_coeffs(I, p, r, eps):
    """Guided filter (He et al.) coefficients for grey guide I and inputs p.
    Returns A, B with q = A*I + B. Computed at the resolution of I/p; they can
    be upsampled and applied to a higher-resolution guide (fast guided
    filter / joint upsampling)."""
    mI = box(I, r)
    mp = box(p, r)
    cov = box(I * p, r) - mI * mp
    var = box(I * I, r) - mI * mI
    A = cov / (var + eps)
    B = mp - A * mI
    return box(A, r), box(B, r)


def gaussian_blur(x, sigma):
    if sigma <= 0:
        return x
    r = max(1, int(3 * sigma))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, 1, -1)
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.expand(c, 1, 1, -1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x


def hex_to_rgb01(s):
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) < 6:
        raise ValueError(f"Colour must be #RRGGBB, got '{s}'")
    return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def to_bchw(img):
    """ComfyUI IMAGE (B,H,W,C) -> (B,3,H,W) float32, alpha dropped."""
    return img[..., :3].movedim(-1, 1).float()


def to_bhwc(x):
    return x.movedim(1, -1).clamp(0, 1)
