"""Built-in monocular depth via Hugging Face ``transformers`` (fallback route).

The recommended route is ComfyUI's native MoGe / Depth Anything 3 nodes; this
module exists so the pack also works on ComfyUI builds without them, and adds
a multi-scale "detail" pass for Depth Anything V2 that recovers thin
structures on very large images.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

from .core import gaussian_blur

MODELS = {
    "Depth-Anything-V2-Large": "depth-anything/Depth-Anything-V2-Large-hf",
    "Depth-Anything-V2-Base": "depth-anything/Depth-Anything-V2-Base-hf",
    "Depth-Anything-V2-Small": "depth-anything/Depth-Anything-V2-Small-hf",
    "Depth-Pro": "apple/DepthPro-hf",
}

_CACHE: dict = {}
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _cache_dir() -> str | None:
    try:
        import folder_paths
        d = os.path.join(folder_paths.models_dir, "stereoforge")
        os.makedirs(d, exist_ok=True)
        return d
    except Exception:
        return None


def load(name: str, device: torch.device, dtype: torch.dtype):
    key = (name, str(device), dtype)
    if key in _CACHE:
        return _CACHE[key]
    from transformers import AutoModelForDepthEstimation
    repo = MODELS[name]
    model = AutoModelForDepthEstimation.from_pretrained(repo, cache_dir=_cache_dir(), dtype=dtype)
    model = model.to(device).eval()
    _CACHE.clear()
    _CACHE[key] = model
    return model


def _resize_to(img_bchw: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return F.interpolate(img_bchw, size=(h, w), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)


def _fit14(h: int, w: int, long_side: int) -> tuple[int, int]:
    s = long_side / max(h, w)
    return max(14, int(round(h * s / 14)) * 14), max(14, int(round(w * s / 14)) * 14)


@torch.no_grad()
def _dav2(model, img_bchw: torch.Tensor, h: int, w: int) -> torch.Tensor:
    x = _resize_to(img_bchw, h, w)
    x = (x - _MEAN.to(x)) / _STD.to(x)
    p = next(model.parameters())
    out = model(pixel_values=x.to(p.device, p.dtype)).predicted_depth
    return out.float().reshape(h, w)


@torch.no_grad()
def _depth_pro(model, img_bchw: torch.Tensor) -> torch.Tensor:
    x = _resize_to(img_bchw, 1536, 1536)
    x = (x - 0.5) / 0.5
    p = next(model.parameters())
    out = model(pixel_values=x.to(p.device, p.dtype)).predicted_depth.float().reshape(1, 1, 1536, 1536)
    return out


def _hann2d(h: int, w: int, device) -> torch.Tensor:
    wy = torch.hann_window(h + 2, periodic=False, device=device)[1:-1]
    wx = torch.hann_window(w + 2, periodic=False, device=device)[1:-1]
    return (wy[:, None] * wx[None, :]).clamp_min(1e-4)


def estimate_disparity(image_hwc: torch.Tensor, name: str, process_res: int, detail_pass: bool,
                       device: torch.device, dtype: torch.dtype, pbar=None) -> torch.Tensor:
    """Return a relative disparity map (H, W) at the image's full resolution (larger = nearer)."""
    H, W, _ = image_hwc.shape
    img = image_hwc[..., :3].permute(2, 0, 1).unsqueeze(0).float()
    model = load(name, device, dtype)

    if name == "Depth-Pro":
        depth = _depth_pro(model, img.to(device))
        depth = F.interpolate(depth, size=(H, W), mode="bicubic", align_corners=False)[0, 0]
        disp = 1.0 / depth.clamp_min(1e-4)
        return disp.cpu()

    gh, gw = _fit14(H, W, process_res)
    img_d = img.to(device)
    glob = _dav2(model, img_d, gh, gw)

    scale = 2.0
    if not detail_pass or max(H, W) < 1.5 * process_res:
        out = F.interpolate(glob[None, None], size=(H, W), mode="bicubic", align_corners=False)[0, 0]
        return out.cpu()

    # Detail pass: tiles at 2x the global resolution, each aligned (scale/shift)
    # to the global prediction, blended with Hann windows; the global map
    # supplies the low frequencies (consistent geometry), the tiles only the
    # high-frequency detail (thin structures, crisp boundaries).
    hh, ww = int(round(gh * scale)), int(round(gw * scale))
    big = _resize_to(img_d, hh, ww)
    glob_up = F.interpolate(glob[None, None], size=(hh, ww), mode="bicubic", align_corners=False)[0, 0]
    th, tw = min(gh, hh), min(gw, ww)
    th, tw = th // 14 * 14, tw // 14 * 14
    ny = max(1, math.ceil((hh - th) / (th / 2)) + 1)
    nx = max(1, math.ceil((ww - tw) / (tw / 2)) + 1)
    ys = [int(round(v)) for v in torch.linspace(0, hh - th, ny).tolist()]
    xs = [int(round(v)) for v in torch.linspace(0, ww - tw, nx).tolist()]
    acc = torch.zeros(hh, ww, device=device)
    wacc = torch.zeros(hh, ww, device=device)
    win = _hann2d(th, tw, device)
    for y0 in ys:
        for x0 in xs:
            t = _dav2(model, big[:, :, y0:y0 + th, x0:x0 + tw], th, tw)
            g = glob_up[y0:y0 + th, x0:x0 + tw]
            A = torch.stack([t.reshape(-1), torch.ones_like(t).reshape(-1)], 1).double()
            sol = torch.linalg.lstsq(A, g.reshape(-1, 1).double()).solution.float().reshape(-1)
            if sol[0] <= 0:  # degenerate tile (no structure): trust the global map
                t_al = g
            else:
                t_al = t * sol[0] + sol[1]
            acc[y0:y0 + th, x0:x0 + tw] += t_al * win
            wacc[y0:y0 + th, x0:x0 + tw] += win
            if pbar is not None:
                pbar()
    tiles = acc / wacc.clamp_min(1e-6)
    sigma = max(th, tw) / 16.0
    fused = gaussian_blur(glob_up, sigma) + (tiles - gaussian_blur(tiles, sigma))
    out = F.interpolate(fused[None, None], size=(H, W), mode="bicubic", align_corners=False)[0, 0]
    return out.cpu()
