"""Sub-pixel registration of the model's colour output to the original photo.

Edit models occasionally shift or zoom the picture by a few pixels. We fit a
global affine (or shift+scale) transform that maps output-grid coordinates to
colour-image coordinates, by minimising the difference between locally
contrast-normalised luminance images over a small image pyramid. Normalising
first makes the fit insensitive to the model having changed brightness or
contrast.
"""

import torch
import torch.nn.functional as F

from .color import gaussian_blur, resize


def _normalise(L):
    x = L / 100.0
    mu = gaussian_blur(x, 4.0)
    d = x - mu
    sd = gaussian_blur(d * d, 4.0).sqrt()
    return d / (sd + 0.02)


def _theta(params, mode):
    if mode == "shift+scale":
        sx, sy, tx, ty = params
        z = torch.zeros_like(sx)
        return torch.stack([torch.stack([1 + sx, z, tx]), torch.stack([z, 1 + sy, ty])]).unsqueeze(0)
    return (torch.eye(2, 3, dtype=params.dtype, device=params.device) + params.view(2, 3)).unsqueeze(0)


def _loss(ref, mov, theta):
    grid = F.affine_grid(theta, list(ref.shape), align_corners=False)
    warped = F.grid_sample(mov, grid, mode="bilinear", padding_mode="border", align_corners=False)
    # ignore a border band where padding dominates
    m = max(2, ref.shape[-1] // 32)
    return (warped - ref)[..., m:-m, m:-m].abs().mean()


def estimate(ref_L, mov_L, mode="affine", max_side=512, iters=80):
    """ref_L, mov_L: (1,1,H,W) L channels at the same resolution.
    Returns (theta (1,2,3), improvement ratio). theta is identity if the fit
    does not clearly improve on no alignment."""
    ref_L = ref_L.float()
    mov_L = mov_L.float()
    h, w = ref_L.shape[-2:]
    s = min(1.0, max_side / max(h, w))
    H, W = max(16, round(h * s)), max(16, round(w * s))
    ref = _normalise(resize(ref_L, H, W, "area"))
    mov = _normalise(resize(mov_L, H, W, "area"))

    n = 4 if mode == "shift+scale" else 6
    params = torch.zeros(n, device=ref.device, requires_grad=True)
    identity = _theta(torch.zeros(n, device=ref.device), mode).detach()

    with torch.enable_grad():
        for level in (4, 2, 1):
            r = F.avg_pool2d(ref, level) if level > 1 else ref
            m = F.avg_pool2d(mov, level) if level > 1 else mov
            opt = torch.optim.Adam([params], lr=2e-3 * level)
            for _ in range(iters):
                opt.zero_grad()
                loss = _loss(r, m, _theta(params, mode))
                loss.backward()
                opt.step()

    with torch.no_grad():
        theta = _theta(params.detach(), mode)
        base = _loss(ref, mov, identity).item()
        fitted = _loss(ref, mov, theta).item()
    if fitted < base * 0.98:
        return theta, base / max(fitted, 1e-8)
    return identity, 1.0


def warp(img, theta, out_h=None, out_w=None):
    """Resample img (B,C,h,w) with theta onto an (out_h,out_w) grid."""
    b, c, h, w = img.shape
    out_h = out_h or h
    out_w = out_w or w
    grid = F.affine_grid(theta.to(img).expand(b, -1, -1), [b, c, out_h, out_w], align_corners=False)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
