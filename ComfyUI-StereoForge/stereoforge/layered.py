"""Layered (matte + background plate) stereo rendering.

Edges are defined by an alpha matte from a trimap-based matting network, not
by the depth map. Around every occlusion edge the scene is split into two
layers, the way manual 2D-to-3D conversion does it with roto and clean plates:

* front layer: the near surface, with the matte's alpha and de-contaminated
  colour F, at the near surface's depth;
* back layer (clean plate): the far surface continued *behind* the near one
  (filled once, in the source view, by an inpainter), at the far depth.

Each layer is a smooth surface, so warping it produces no stretching and no
holes at the edge; the front layer is composited "over" the back one in the
target view. The only remaining holes are the frame-edge strips.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from . import core


@dataclass
class Layers:
    zone: torch.Tensor       # (H, W) bool, neighbourhood of occlusion edges
    near: torch.Tensor       # (H, W) bool, near side of the local edge
    unknown: torch.Tensor    # (H, W) bool, trimap unknown band
    trimap: torch.Tensor     # (H, W) float 0 / 0.5 / 1
    D: torch.Tensor          # (H, W) signed pixel disparity (edge-sharpened)
    plate_mask: torch.Tensor  # (H, W) bool, where the back layer must be synthesised
    plate_hide: torch.Tensor  # (H, W) bool, near-side pixels hidden from the inpainter (superset of plate_mask)


def build_trimap(D: torch.Tensor, jump_px: float, band: int, reach: int | None = None) -> Layers:
    """Occlusion-edge trimap from an edge-sharpened disparity map.

    Occlusion edges are real disparity *steps* between neighbouring pixels
    (bigger than ~3 px of parallax). Each step stores its mid level and
    height; every pixel is then classified against its nearest step: near if
    its disparity is above that step's mid level. The unknown band (``band``
    px either side) is what the matting network resolves; it absorbs the
    misalignment between depth edges and true image edges. The back layer is
    needed on the near side for as far as the step will move it.
    """
    thr = max(3.0, 2 * jump_px)
    H, W = D.shape
    hi = torch.zeros_like(D)
    lo = torch.full_like(D, float("inf"))
    edge = torch.zeros_like(D, dtype=torch.bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = D[max(0, -dy) + 0: H - dy if dy > 0 else H, max(0, -dx): W - dx if dx > 0 else W]
        ys = slice(dy, H) if dy >= 0 else slice(0, H + dy)
        xs = slice(dx, W) if dx >= 0 else slice(0, W + dx)
        ya = slice(0, H - dy) if dy >= 0 else slice(-dy, H)
        xa = slice(0, W - dx) if dx >= 0 else slice(-dx, W)
        A, B = D[ya, xa], D[ys, xs]
        step = (A - B).abs() > thr
        # mark the near pixel of each step, remember the far level it steps down to
        a_near = step & (A > B)
        b_near = step & (B > A)
        edge[ya, xa] |= a_near
        edge[ys, xs] |= b_near
        lo[ya, xa] = torch.where(a_near, torch.minimum(lo[ya, xa], B), lo[ya, xa])
        lo[ys, xs] = torch.where(b_near, torch.minimum(lo[ys, xs], A), lo[ys, xs])
    if not bool(edge.any()):
        z = torch.zeros_like(edge)
        return Layers(z, torch.ones_like(edge), z, torch.ones_like(D), D, z, z)
    mid_e = torch.where(edge, 0.5 * (D + lo), torch.zeros_like(D))
    h_e = torch.where(edge, D - lo, torch.zeros_like(D))
    dist_np, idx = ndimage.distance_transform_edt(~edge.cpu().numpy(), return_indices=True)
    dist = torch.from_numpy(dist_np).to(D)
    iy, ix = torch.from_numpy(idx[0]).to(D.device), torch.from_numpy(idx[1]).to(D.device)
    mid = mid_e[iy, ix]
    height = h_e[iy, ix]
    zone = dist <= height + band + 2
    near = (D >= mid) | ~zone
    unknown = zone & (dist <= band)
    tri = torch.ones_like(D)
    tri = torch.where(zone & ~near, torch.zeros_like(D), tri)
    tri = torch.where(unknown, torch.full_like(D, 0.5), tri)
    plate = (zone & near & (dist <= height + band + 2)) | unknown
    # the inpainter must not see the near object next to the plate region, or it
    # continues the object instead of the background behind it
    guard = torch.clamp(2 * height, min=float(4 * band))
    hide = ((D >= mid) & (dist <= height + band + 2 + guard)) | plate
    return Layers(zone, near, unknown, tri, D, plate, hide)


def nearest_fill(values: torch.Tensor, known: torch.Tensor, where: torch.Tensor) -> torch.Tensor:
    """Replace ``values`` at ``where`` by the value of the nearest ``known`` pixel."""
    if not bool(where.any()) or not bool(known.any()):
        return values
    _, idx = ndimage.distance_transform_edt(~known.cpu().numpy(), return_indices=True)
    iy = torch.from_numpy(idx[0]).to(values.device)
    ix = torch.from_numpy(idx[1]).to(values.device)
    filled = values[iy, ix]
    return torch.where(where, filled, values)


@torch.no_grad()
def render_layered(image: torch.Tensor, alpha_near: torch.Tensor, plate: torch.Tensor, L: Layers,
                   sign: float, jump_px: float = 1.5, max_stretch: float = 2.0):
    """Render the opposite eye from the two layers.

    image: (H, W, 3) source; alpha_near: (H, W) matte of the near surface
    (1 outside edge zones); plate: (H, W, 3) source with the background
    synthesised under ``L.plate_mask``.
    Returns (view, holes) where holes are target pixels neither layer covers.
    """
    D = L.D
    a = alpha_near.clamp(0, 1)
    # --- depths of each layer: extend each side across the edge band ---------
    far_known = L.zone & ~L.near & ~L.unknown
    near_known = L.zone & L.near & ~L.unknown
    D_back = nearest_fill(D, far_known, L.plate_mask)
    D_front = nearest_fill(D, near_known, L.unknown & ~L.near)
    # --- front colour: de-contaminate the matte's mixed pixels ---------------
    a3 = a.unsqueeze(-1)
    Fg = ((image - (1 - a3) * plate) / a3.clamp_min(0.05)).clamp(0, 1)
    Fg = torch.where((a3 > 0.98) | ~L.unknown.unsqueeze(-1), image, Fg)
    # --- back layer ----------------------------------------------------------
    back = torch.where(L.plate_mask.unsqueeze(-1), plate, image)
    sx_b, z_b = core.forward_map(D_back, sign, max_stretch)
    hole_b = torch.isnan(sx_b)
    view_b = core.sample_rows(back, sx_b, z_b, D_back, tol_px=jump_px)
    # --- front layer (premultiplied colour + alpha) -------------------------
    keep = a > 0.01
    sx_f, z_f = core.forward_map(D_front, sign, max_stretch, keep=keep)
    pre = torch.cat([Fg * a3, a3], -1)
    view_f = core.sample_rows(pre, sx_f, z_f, D_front, tol_px=jump_px)
    af = torch.where(torch.isnan(sx_f), torch.zeros_like(a), view_f[..., 3]).clamp(0, 1)
    cf = view_f[..., :3]
    # the front layer may only cover the back layer where it is nearer (or equal)
    vis = torch.isnan(sx_b) | (z_f >= z_b - jump_px)
    af = torch.where(vis, af, torch.zeros_like(af))
    cf = torch.where(vis.unsqueeze(-1), cf, torch.zeros_like(cf))
    out = cf + (1 - af.unsqueeze(-1)) * view_b
    holes = hole_b & (af < 0.99)
    return out.clamp(0, 1), holes, af


# --------------------------------------------------------------------------- #
# Model wrappers (ViTMatte via transformers, LaMa TorchScript)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def vitmatte(model, processor, image: torch.Tensor, trimap: torch.Tensor, crop: int = 1024,
             device=None) -> torch.Tensor:
    """Alpha for the trimap's unknown region, processed in native-res crops around it."""
    H, W, _ = image.shape
    alpha = (trimap > 0.75).float()
    unk = trimap == 0.5
    if not bool(unk.any()):
        return alpha
    dev = device or next(model.parameters()).device
    ctx = 64
    step = crop - 2 * ctx
    for y in range(0, H, step):
        for x in range(0, W, step):
            cy0, cy1, cx0, cx1 = y, min(H, y + step), x, min(W, x + step)
            if not bool(unk[cy0:cy1, cx0:cx1].any()):
                continue
            y0, y1 = max(0, cy0 - ctx), min(H, cy1 + ctx)
            x0, x1 = max(0, cx0 - ctx), min(W, cx1 + ctx)
            img = (image[y0:y1, x0:x1].cpu().numpy() * 255).astype(np.uint8)
            tri = (trimap[y0:y1, x0:x1].cpu().numpy() * 255).astype(np.uint8)
            inp = processor(images=img, trimaps=tri, return_tensors="pt").to(dev)
            p = next(model.parameters())
            inp["pixel_values"] = inp["pixel_values"].to(p.dtype)
            out = model(**inp).alphas[0, 0, : y1 - y0, : x1 - x0].float().cpu()
            sub = unk[cy0:cy1, cx0:cx1]
            region = alpha[cy0:cy1, cx0:cx1]
            region[sub] = out[cy0 - y0: cy1 - y0, cx0 - x0: cx1 - x0][sub]
    return alpha


@torch.no_grad()
def lama(model, image: torch.Tensor, mask: torch.Tensor, crop: int = 1024) -> torch.Tensor:
    """Inpaint ``mask`` with LaMa at native resolution (crops with context)."""
    H, W, _ = image.shape
    out = image.clone()
    if not bool(mask.any()):
        return out
    ctx = 128
    step = crop - 2 * ctx
    dev = next(model.parameters()).device
    for y in range(0, H, step):
        for x in range(0, W, step):
            cy0, cy1, cx0, cx1 = y, min(H, y + step), x, min(W, x + step)
            if not bool(mask[cy0:cy1, cx0:cx1].any()):
                continue
            y0, y1 = max(0, cy0 - ctx), min(H, cy1 + ctx)
            x0, x1 = max(0, cx0 - ctx), min(W, cx1 + ctx)
            img = out[y0:y1, x0:x1].permute(2, 0, 1)[None]
            m = mask[y0:y1, x0:x1].float()[None, None]
            h, w = img.shape[-2:]
            ph, pw = (-h) % 8, (-w) % 8
            img_p = F.pad(img, (0, pw, 0, ph), mode="reflect")
            m_p = F.pad(m, (0, pw, 0, ph))
            res = model(img_p.to(dev), m_p.to(dev))[0, :, :h, :w].permute(1, 2, 0).float().cpu()
            sub = mask[cy0:cy1, cx0:cx1]
            region = out[cy0:cy1, cx0:cx1]
            region[sub] = res[cy0 - y0: cy1 - y0, cx0 - x0: cx1 - x0][sub]
    return out.clamp(0, 1)
