"""Object-layer stereo: mattes + depth-ordered layers + reviewable clean plates.

The scene is decomposed the way manual 2D-to-3D conversion does it:

* one layer per object mask (people, gifts, furniture...), each with an alpha
  matte, de-contaminated colour and its own depth surface;
* a background layer (everything that is not an object).

Edges come from the object mattes, never from the depth map: the depth map only
decides how far each layer moves. Wherever a layer will be revealed in the new
eye behind a nearer one, the hidden content is invented once, in the source
view ("clean plate"), and exposed for review and editing. Rendering is then a
deterministic back-to-front composite of smooth layers, so if the mattes and
plates look right as flat images, the stereo pair is clean by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from . import core

OBJ_STRETCH = 8.0  # object layers are smooth surfaces: stretch rather than tear

FillFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
MatteFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def dilate(m: torch.Tensor, r: int) -> torch.Tensor:
    return core.max_filter(m.float(), r) > 0.5 if r > 0 else m.bool()


def erode(m: torch.Tensor, r: int) -> torch.Tensor:
    return ~dilate(~m.bool(), r) if r > 0 else m.bool()


def nearest_fill(values: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    """Every unknown pixel takes the value of the nearest known pixel."""
    if bool(known.all()) or not bool(known.any()):
        return values
    _, idx = ndimage.distance_transform_edt(~known.cpu().numpy(), return_indices=True)
    iy = torch.from_numpy(idx[0]).to(values.device)
    ix = torch.from_numpy(idx[1]).to(values.device)
    return torch.where(known if values.ndim == 2 else known.unsqueeze(-1), values, values[iy, ix])


def pushpull_fill(values: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    """Smooth interpolation of unknown pixels (multi-scale push-pull).

    Used for the background depth under objects, so a floor or wall keeps its
    slope behind a person instead of becoming a flat terrace.
    """
    if bool(known.all()) or not bool(known.any()):
        return values
    w = known.float()[None, None]
    v = torch.where(known, values, torch.zeros_like(values)).float()[None, None] * w
    pyr = [(v, w)]
    while min(v.shape[-2:]) > 2:
        v = F.avg_pool2d(v, 2, ceil_mode=True)
        w = F.avg_pool2d(w, 2, ceil_mode=True)
        pyr.append((v, w))
    est = pyr[-1][0] / pyr[-1][1].clamp_min(1e-8)
    for v, w in reversed(pyr[:-1]):
        up = F.interpolate(est, size=v.shape[-2:], mode="bilinear", align_corners=False)
        own = v / w.clamp_min(1e-8)
        c = w.clamp(0, 1)
        est = c * own + (1 - c) * up
    return torch.where(known, values, est[0, 0])


def sample_at(target_map: torch.Tensor, x_src: torch.Tensor) -> torch.Tensor:
    """Linear sample of a target-view map (H, W) at per-pixel x positions (same row)."""
    H, W = target_map.shape
    x = x_src.clamp(0, W - 1)
    x0 = torch.floor(x).long().clamp(0, W - 1)
    x1 = (x0 + 1).clamp(0, W - 1)
    t = x - x0.float()
    return target_map.gather(1, x0) * (1 - t) + target_map.gather(1, x1) * t


def render_layer(color: torch.Tensor, alpha: torch.Tensor, D: torch.Tensor, sign: float,
                 max_stretch: float = 2.0, jump_px: float = 1.5):
    # (object layers pass a larger max_stretch: a smooth object never tears)
    """Forward-render one layer (any number of channels).

    Returns (premultiplied channels, alpha, hole) in the target view."""
    keep = alpha > 0.004
    sx, z = core.forward_map(D, sign, max_stretch, keep=keep)
    a3 = alpha.unsqueeze(-1)
    C = color.shape[-1]
    pre = torch.cat([color * a3, a3], -1)
    out = core.sample_rows(pre, sx, z, D, tol_px=jump_px)
    hole = torch.isnan(sx)
    a_t = torch.where(hole, torch.zeros_like(alpha), out[..., C]).clamp(0, 1)
    c_t = torch.where(hole.unsqueeze(-1), torch.zeros_like(color), out[..., :C])
    return c_t, a_t, hole


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

@dataclass
class Layer:
    name: str
    alpha: torch.Tensor          # (H, W) visible alpha in the source view
    color: torch.Tensor          # (H, W, 3) de-contaminated colour + invented content
    D: torch.Tensor              # (H, W) pixel disparity, defined over the whole frame
    invented: torch.Tensor       # (H, W) bool, pixels synthesised for this layer (clean plate)
    hide: torch.Tensor           # (H, W) bool, nearer objects hidden from the inpainter for this layer
    order_depth: float = 0.0     # median disparity (for ordering)

    def render_alpha(self) -> torch.Tensor:
        return torch.where(self.invented, torch.ones_like(self.alpha), self.alpha)


@dataclass
class Stack:
    image: torch.Tensor          # (H, W, 3) source
    layers: list                 # [background, far object ... near object]
    sign: float                  # -1: render right eye from left source, +1: left eye from right source
    source_eye: str
    zero_plane: float
    params: dict = field(default_factory=dict)
    edge_warn: torch.Tensor | None = None  # background depth steps not covered by any object mask

    def clone(self) -> "Stack":
        layers = [Layer(l.name, l.alpha, l.color.clone(), l.D, l.invented.clone(), l.hide, l.order_depth)
                  for l in self.layers]
        return Stack(self.image, layers, self.sign, self.source_eye, self.zero_plane, dict(self.params),
                     self.edge_warn)


# --------------------------------------------------------------------------- #
# building the stack
# --------------------------------------------------------------------------- #

def trimap_from_mask(mask: torch.Tensor, band: int) -> torch.Tensor:
    m = mask > 0.5
    fg = erode(m, band)
    bg = ~dilate(m, band)
    tri = torch.full(mask.shape, 0.5, device=mask.device)
    tri = torch.where(fg, torch.ones_like(tri), tri)
    tri = torch.where(bg, torch.zeros_like(tri), tri)
    return tri


def build_stack(image: torch.Tensor, disp: torch.Tensor, masks: list, p: core.StereoParams,
                matte_fn: MatteFn | None, fill_fn: FillFn, edge_band: int = 0,
                names: list | None = None, log=print, obj_smooth: float = 0.0) -> Stack:
    H, W, _ = image.shape
    md = max(H, W)
    image = image.float()
    band = edge_band or max(3, int(round(md / 300)))
    sign = -1.0 if p.source_eye == "left" else 1.0

    disp = core.refine_disparity_edges(disp.float(), image, core.auto_px(md, 1000, 2))
    D, z0 = core.to_pixel_disparity(disp, p)

    # ---- mattes ------------------------------------------------------------
    alphas = []
    for i, m in enumerate(masks):
        m = m.float().to(image.device)
        if m.shape != (H, W):
            m = F.interpolate(m[None, None], size=(H, W), mode="bilinear")[0, 0]
        if not bool((m > 0.5).any()):
            continue
        if matte_fn is not None:
            a = matte_fn(image, trimap_from_mask(m, band))
        else:
            a = core.gaussian_blur((m > 0.5).float(), 0.7)
        alphas.append(a.clamp(0, 1))
    names = names or [f"object {i + 1}" for i in range(len(alphas))]

    # ---- depth per object, depth order ------------------------------------
    r_d = max(2, int(round(md / 300)))
    obj_sigma = obj_smooth if obj_smooth > 0 else max(2.0, md / 100)
    objs = []
    for i, a in enumerate(alphas):
        core_px = erode(a > 0.98, r_d)
        if core_px.sum() < 16:
            core_px = a > 0.98
        if core_px.sum() < 4:
            core_px = a > 0.5
        med = float(D[core_px].median()) if bool(core_px.any()) else float(D.median())
        # One object = one smooth surface: internal relief (an arm in front of the
        # chest) is kept as a gentle slope that stretches by a pixel or two instead
        # of a step that would tear the object open.
        Di = core.gaussian_blur(nearest_fill(D, core_px), obj_sigma)
        objs.append(dict(alpha=a, D=Di, med=med, name=names[i] if i < len(names) else f"object {i + 1}"))
    objs.sort(key=lambda o: o["med"])  # far -> near

    # visible alpha: nearer objects occlude farther ones
    K = len(objs)
    for j in range(K):
        v = objs[j]["alpha"]
        for k in range(j + 1, K):
            v = v * (1 - objs[k]["alpha"])
        objs[j]["vis"] = v

    any_obj = torch.zeros(H, W, dtype=torch.bool, device=image.device)
    for o in objs:
        any_obj |= o["alpha"] > 0.01
    # Depth right next to an object is unreliable: monocular depth edges often sit
    # a few px outside the true silhouette, so those background pixels carry the
    # object's depth. Ignore a band around every object and interpolate across it.
    r_bg = 2 * band + r_d
    bg_known = erode(~any_obj, r_bg) if bool(any_obj.any()) else torch.ones_like(any_obj)
    if not bool(bg_known.any()):
        bg_known = ~any_obj
    D0 = pushpull_fill(D, bg_known)
    D0 = core.sharpen_discontinuities(D0, 0, p.jump_px, p.grad_thr)

    # ---- which hidden pixels each layer must supply (nearest first) -------
    reach = int(math.ceil(float(D.max() - D.min()))) + 2 * band + 2
    margin = max(2, band // 2)
    A_t = torch.zeros(H, W, device=image.device)
    xs = torch.arange(W, device=image.device, dtype=torch.float32).expand(H, W)
    soft_any = torch.zeros_like(any_obj)
    for o in objs:
        soft_any |= (o["alpha"] > 0.01) & (o["alpha"] < 0.99)
    fills = {}
    for j in range(K - 1, -1, -1):
        o = objs[j]
        nearer = torch.zeros_like(any_obj)
        for k in range(j + 1, K):
            nearer |= objs[k]["alpha"] > 0.01
        inv = torch.zeros_like(any_obj)
        if bool(nearer.any()):
            cand = nearer & dilate(o["vis"] > 0.5, reach)
            covered = sample_at(A_t, xs + sign * o["D"])
            rev = cand & (covered < 0.98)
            inv = (dilate(rev, margin) & cand) | (cand & soft_any & dilate(o["vis"] > 0.5, band + 2))
        o["invented"], o["hide"] = inv, nearer
        ra = torch.where(inv, torch.ones_like(o["vis"]), o["vis"])
        _, a_t, _ = render_layer(ra.unsqueeze(-1), ra, o["D"], sign, OBJ_STRETCH)
        A_t = a_t + (1 - a_t) * A_t
    covered = sample_at(A_t, xs + sign * D0)
    rev0 = any_obj & (covered < 0.98)
    inv0 = (dilate(rev0, margin) & any_obj) | soft_any

    # ---- clean plates (source view), far to near context ------------------
    log(f"[StereoForgeSeg] {K} object layer(s); inventing background for "
        f"{int(inv0.sum())} px and object extensions for {sum(int(o['invented'].sum()) for o in objs)} px")
    plate0 = fill_fn(image, inv0, any_obj)
    layers = [Layer("background", torch.ones(H, W, device=image.device), torch.where(inv0.unsqueeze(-1), plate0, image),
                    D0, inv0, any_obj, float(D0.median()))]
    for o in objs:
        if bool(o["invented"].any()):
            pj = fill_fn(image, o["invented"], o["hide"])
            col = torch.where(o["invented"].unsqueeze(-1), pj, image)
        else:
            col = image.clone()
        layers.append(Layer(o["name"], o["vis"], col, o["D"], o["invented"], o["hide"], o["med"]))

    stack = Stack(image, layers, sign, p.source_eye, z0, dict(budget_pct=p.budget_pct, band=band))
    decontaminate(stack)
    # unmasked depth edges in the background (will be auto-filled in the new eye)
    _, _, hole0 = render_layer(image, torch.ones(H, W, device=image.device), D0, sign)
    stack.edge_warn = hole0
    return stack


def decontaminate(stack: Stack) -> None:
    """Recover each object's own colour in its soft edge: I = a*F + (1-a)*B.

    B is what lies behind the object (the composite of all farther layers,
    including their clean plates). Very low alphas take the object's interior
    colour instead of the unstable division, which removes speckle."""
    img = stack.image
    comp = stack.layers[0].color.clone()
    for L in stack.layers[1:]:
        a = L.alpha
        soft = (a > 0.004) & (a < 0.98)
        if bool(soft.any()):
            a3 = a.unsqueeze(-1)
            Fr = ((img - (1 - a3) * comp) / a3.clamp_min(0.1)).clamp(0, 1)
            Fin = nearest_fill(img, a > 0.98)
            w = ((a - 0.05) / 0.25).clamp(0, 1).unsqueeze(-1)
            Fd = w * Fr + (1 - w) * Fin
            L.color = torch.where(soft.unsqueeze(-1) & ~L.invented.unsqueeze(-1), Fd, L.color)
        ra = L.render_alpha().unsqueeze(-1)
        comp = L.color * ra + comp * (1 - ra)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def render_stack(stack: Stack, hole_fill: FillFn | None = None):
    """Composite all layers into the opposite eye.

    Returns (eye, invented_visible, autofilled, unfilled):
    * eye: (H, W, 3) final generated eye;
    * invented_visible / autofilled: review masks in the target view (pixels
      showing clean-plate content / pixels no layer covered - unmasked depth
      edges and the frame-edge strip - which were auto-filled);
    * unfilled: (H, W, 4) the same eye with only photo-derived content, alpha 0
      wherever content was invented or auto-filled (for inpainting elsewhere).
    """
    img = stack.image
    H, W, _ = img.shape
    ones = torch.ones(H, W, device=img.device)

    def ext_of(L):
        inv = L.invented.float().unsqueeze(-1)
        return torch.cat([L.color, inv, L.color * (1 - inv), 1 - inv], -1)

    L0 = stack.layers[0]
    c, a, hole0 = render_layer(ext_of(L0), ones, L0.D, stack.sign)
    out, inv, cov = c[..., :3], c[..., 3], a
    kc, ka = c[..., 4:7], c[..., 7]
    for L in stack.layers[1:]:
        c, a, _ = render_layer(ext_of(L), L.render_alpha(), L.D, stack.sign, OBJ_STRETCH)
        a1 = (1 - a).unsqueeze(-1)
        out = c[..., :3] + a1 * out
        inv = c[..., 3] + (1 - a) * inv
        cov = a + (1 - a) * cov
        # invented content of a nearer layer still hides what is behind it
        kc = c[..., 4:7] + a1 * kc
        ka = c[..., 7] + (1 - a) * ka
    holes = cov < 0.5
    norm = out / cov.clamp_min(1e-3).unsqueeze(-1)
    norm = torch.where(holes.unsqueeze(-1), torch.zeros_like(norm), norm).clamp(0, 1)
    if bool(holes.any()):
        if hole_fill is None:
            filled = core.prefill_holes(norm, holes, torch.zeros(H, W, device=img.device), 1.0)[0]
        else:
            filled = hole_fill(norm, holes, torch.zeros_like(holes))
        norm = torch.where(holes.unsqueeze(-1), filled, norm)
    inv_vis = (inv / cov.clamp_min(1e-3) > 0.5) & ~holes
    ka = torch.where(holes, torch.zeros_like(ka), ka).clamp(0, 1)
    krgb = torch.where((ka > 1e-3).unsqueeze(-1), kc / ka.clamp_min(1e-3).unsqueeze(-1), torch.zeros_like(kc))
    unfilled = torch.cat([krgb.clamp(0, 1), ka.unsqueeze(-1)], -1)
    return norm.clamp(0, 1), inv_vis, holes, unfilled
