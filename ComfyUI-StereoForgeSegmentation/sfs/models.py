"""Matting (ViTMatte) and inpainting (LaMa) backends, run at native resolution in crops."""

from __future__ import annotations

import os
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

from . import core

LAMA_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
VITMATTE = {
    "ViTMatte-Base": "hustvl/vitmatte-base-composition-1k",
    "ViTMatte-Small": "hustvl/vitmatte-small-composition-1k",
}
_CACHE: dict = {}


def _models_dir(sub: str) -> str:
    try:
        import folder_paths
        base = folder_paths.models_dir
    except Exception:
        base = os.path.join(os.path.dirname(__file__), "..", "models")
    d = os.path.join(base, sub)
    os.makedirs(d, exist_ok=True)
    return d


def _crops(mask: torch.Tensor, crop: int, ctx: int):
    """Yield (core box, context box) for every cell of a grid that contains mask."""
    H, W = mask.shape
    step = max(64, crop - 2 * ctx)
    for y in range(0, H, step):
        for x in range(0, W, step):
            cy0, cy1, cx0, cx1 = y, min(H, y + step), x, min(W, x + step)
            if not bool(mask[cy0:cy1, cx0:cx1].any()):
                continue
            yield (cy0, cy1, cx0, cx1), (max(0, cy0 - ctx), min(H, cy1 + ctx), max(0, cx0 - ctx), min(W, cx1 + ctx))


# --------------------------------------------------------------------------- #
# ViTMatte
# --------------------------------------------------------------------------- #

def load_vitmatte(name: str, device, dtype=torch.float32):
    key = ("vitmatte", name, str(device), dtype)
    if key not in _CACHE:
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor
        repo = VITMATTE[name]
        cache = _models_dir("vitmatte")
        proc = VitMatteImageProcessor.from_pretrained(repo, cache_dir=cache)
        model = VitMatteForImageMatting.from_pretrained(repo, cache_dir=cache).to(device=device, dtype=dtype).eval()
        _CACHE[key] = (model, proc)
    return _CACHE[key]


@torch.no_grad()
def vitmatte(model_proc, image: torch.Tensor, trimap: torch.Tensor, crop: int = 1024) -> torch.Tensor:
    """Alpha matte: trimap 0 / 0.5 / 1 (bg / unknown / fg); unknown resolved by ViTMatte."""
    model, proc = model_proc
    alpha = (trimap > 0.75).float()
    unk = trimap == 0.5
    if not bool(unk.any()):
        return alpha
    p = next(model.parameters())
    for (cy0, cy1, cx0, cx1), (y0, y1, x0, x1) in _crops(unk, crop, 96):
        img = (image[y0:y1, x0:x1, :3].cpu().numpy() * 255 + 0.5).astype(np.uint8)
        tri = (trimap[y0:y1, x0:x1].cpu().numpy() * 255 + 0.5).astype(np.uint8)
        inp = proc(images=img, trimaps=tri, return_tensors="pt")
        pv = inp["pixel_values"].to(device=p.device, dtype=p.dtype)
        out = model(pixel_values=pv).alphas[0, 0, : y1 - y0, : x1 - x0].float().to(alpha.device)
        sub = unk[cy0:cy1, cx0:cx1]
        region = alpha[cy0:cy1, cx0:cx1]
        region[sub] = out[cy0 - y0: cy1 - y0, cx0 - x0: cx1 - x0][sub]
    return alpha.clamp(0, 1)


# --------------------------------------------------------------------------- #
# LaMa
# --------------------------------------------------------------------------- #

def load_lama(device):
    key = ("lama", str(device))
    if key not in _CACHE:
        path = os.path.join(_models_dir("inpaint"), "big-lama.pt")
        if not os.path.exists(path):
            tmp = path + ".part"
            urllib.request.urlretrieve(LAMA_URL, tmp)
            os.replace(tmp, path)
        _CACHE[key] = torch.jit.load(path, map_location=device).eval()
    return _CACHE[key]


@torch.no_grad()
def lama_fill(model, image: torch.Tensor, fill: torch.Tensor, hide: torch.Tensor, guard: int,
              crop: int = 1024) -> torch.Tensor:
    """Inpaint ``fill`` at native resolution; ``hide`` (within ``guard`` px of the fill)
    is masked too so the inpainter cannot see - and extend - the occluding object."""
    out = image.clone()
    if not bool(fill.any()):
        return out
    guard_m = hide & core.max_filter(fill.float(), guard).bool()
    rem = fill.clone()
    dev = next(model.parameters()).device
    for (cy0, cy1, cx0, cx1), (y0, y1, x0, x1) in _crops(fill, crop, 128):
        # already-filled pixels from earlier crops act as context, not holes
        m = (rem | guard_m)[y0:y1, x0:x1].float()[None, None]
        img = out[y0:y1, x0:x1, :3].permute(2, 0, 1)[None].float()
        h, w = img.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8
        img_p = F.pad(img, (0, pw, 0, ph), mode="replicate")
        m_p = F.pad(m, (0, pw, 0, ph))
        res = model(img_p.to(dev), m_p.to(dev))[0, :, :h, :w].permute(1, 2, 0).float().to(out.device)
        sub = rem[cy0:cy1, cx0:cx1]
        region = out[cy0:cy1, cx0:cx1, :3]
        region[sub] = res[cy0 - y0: cy1 - y0, cx0 - x0: cx1 - x0][sub]
        rem[cy0:cy1, cx0:cx1] = False
    return out.clamp(0, 1)
