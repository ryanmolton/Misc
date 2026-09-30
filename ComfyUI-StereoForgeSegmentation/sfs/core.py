"""StereoForge core: full-resolution stereo view synthesis primitives.

Everything here is plain PyTorch (no ComfyUI imports) so it can be unit-tested
and reused. Conventions:

* images are float tensors (H, W, C) in [0, 1]
* ``disp`` is a *normalised disparity* map (H, W) in [0, 1], 1 = nearest
* ``D`` is a *signed pixel disparity* map (H, W): positive = in front of the
  screen plane (crossed), negative = behind it (uncrossed)
* a target view is produced by moving every source pixel horizontally to
  ``x + sign * D(x)``; ``sign = -1`` renders the right eye from a left source,
  ``sign = +1`` renders the left eye from a right source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Small image-processing helpers
# --------------------------------------------------------------------------- #

def _as4d(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(1, 1, x.shape[-2], x.shape[-1])


def box_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    """Mean filter with a (2r+1)^2 window and replicate borders. x: (H, W)."""
    if r <= 0:
        return x
    y = F.pad(_as4d(x), (r, r, r, r), mode="replicate")
    return F.avg_pool2d(y, 2 * r + 1, stride=1)[0, 0]


def max_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return x
    y = F.pad(_as4d(x), (r, r, r, r), mode="replicate")
    return F.max_pool2d(y, 2 * r + 1, stride=1)[0, 0]


def min_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    return -max_filter(-x, r)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of an (H, W) map."""
    if sigma <= 0:
        return x
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    y = F.pad(_as4d(x), (r, r, 0, 0), mode="replicate")
    y = F.conv2d(y, k.view(1, 1, 1, -1))
    y = F.pad(y, (0, 0, r, r), mode="replicate")
    y = F.conv2d(y, k.view(1, 1, -1, 1))
    return y[0, 0]


def luminance(img: torch.Tensor) -> torch.Tensor:
    if img.shape[-1] == 1:
        return img[..., 0]
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114


def guided_filter(guide: torch.Tensor, src: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    """He et al. grey-guide guided filter. guide/src: (H, W)."""
    if r <= 0:
        return src
    mean_i = box_filter(guide, r)
    mean_p = box_filter(src, r)
    cov_ip = box_filter(guide * src, r) - mean_i * mean_p
    var_i = box_filter(guide * guide, r) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box_filter(a, r) * guide + box_filter(b, r)


def robust_quantile(x: torch.Tensor, q: float, max_samples: int = 2_000_000) -> float:
    x = x.reshape(-1)
    if x.numel() > max_samples:
        step = x.numel() // max_samples + 1
        x = x[::step]
    return float(torch.quantile(x.float(), q))


def auto_px(max_dim: int, per_px: float, lo: int) -> int:
    """Resolution-aware default radius: one unit per ``per_px`` pixels of the long side."""
    return max(lo, int(round(max_dim / per_px)))


# --------------------------------------------------------------------------- #
# Disparity preparation
# --------------------------------------------------------------------------- #

def normalize_disparity(raw: torch.Tensor, invalid_is_far: bool = True,
                        q_lo: float = 0.001, q_hi: float = 0.999) -> torch.Tensor:
    """Map an arbitrary-scale disparity (larger = nearer) to [0, 1] robustly."""
    raw = raw.float()
    finite = torch.isfinite(raw)
    if not finite.any():
        return torch.zeros_like(raw)
    vals = raw[finite]
    lo = robust_quantile(vals, q_lo)
    hi = robust_quantile(vals, q_hi)
    if hi - lo < 1e-8:
        out = torch.full_like(raw, 0.5)
    else:
        out = ((raw - lo) / (hi - lo)).clamp(0.0, 1.0)
    fill = 0.0 if invalid_is_far else 0.5
    return torch.where(finite, out, torch.full_like(out, fill))


def depth_to_disparity(depth: torch.Tensor) -> torch.Tensor:
    """1/z with non-finite / non-positive depth mapped to 0 (= infinitely far)."""
    depth = depth.float()
    ok = torch.isfinite(depth) & (depth > 0)
    return torch.where(ok, 1.0 / depth.clamp_min(1e-8), torch.zeros_like(depth))


def refine_disparity_edges(disp: torch.Tensor, image: torch.Tensor, radius: int,
                           eps: float = 1e-4) -> torch.Tensor:
    """Edge-aware refinement of an upsampled disparity map with the full-res image.

    The guided filter aligns the soft ramps produced by bilinear upsampling of a
    low-res prediction with the actual colour edges of the full-res image.
    """
    if radius <= 0:
        return disp
    g = luminance(image).to(disp.dtype)
    out = guided_filter(g, disp, radius, eps)
    return out.clamp(float(disp.min()), float(disp.max()))


def fit_planes(disp: torch.Tensor, mask: torch.Tensor, ring: int) -> torch.Tensor:
    """Replace each masked region by a least-squares plane fitted to the ring
    of disparity just outside it (flat pictures, screens, signs, mirrors...)."""
    out = disp.clone()
    m = mask > 0.5
    if not m.any():
        return out
    labels = connected_components(m)
    H, W = disp.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=disp.device, dtype=disp.dtype),
                            torch.arange(W, device=disp.device, dtype=disp.dtype), indexing="ij")
    for lab in torch.unique(labels):
        if lab == 0:
            continue
        region = labels == lab
        around = (max_filter(region.float(), ring) > 0) & ~m
        if around.sum() < 3:
            continue
        A = torch.stack([xx[around], yy[around], torch.ones_like(xx[around])], 1)
        sol = torch.linalg.lstsq(A.double(), disp[around].double().unsqueeze(1)).solution.squeeze(1).to(disp.dtype)
        out[region] = sol[0] * xx[region] + sol[1] * yy[region] + sol[2]
    return out.clamp(0.0, 1.0)


def connected_components(m: torch.Tensor) -> torch.Tensor:
    """8-connected labelling (scipy)."""
    from scipy import ndimage
    lab, _ = ndimage.label(m.cpu().numpy(), structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    return torch.from_numpy(lab).to(m.device)


@dataclass
class StereoParams:
    budget_pct: float = 3.0          # total disparity range, % of image width
    convergence: float = 0.25        # zero-parallax plane, 0 = farthest, 1 = nearest
    stereo_window: bool = True       # push zero plane back so nothing at the L/R borders pops out
    source_eye: str = "left"         # which eye the source image is
    grad_thr: float = 0.35           # horizontal disparity slope (px/px) treated as an occlusion edge
    fg_dilate: int = 0               # extra px of edge pixels that move fully opaque with the foreground
    soft_band: int = -1              # soft-boundary (hair/fur) band width, -1 = auto, 0 = off
    fg_guard: int = -1               # foreground band hidden from the inpainter, -1 = auto, 0 = off
    min_fill_px: int = -1            # gaps narrower than this keep the geometric fill (no diffusion), -1 = auto
    edge_band: int = 0               # 0 = auto
    jump_px: float = 1.5             # disparity steps below this are rendered as stretch, not holes
    max_stretch: float = 2.0         # max target span of one source pixel before it counts as a hole
    black_borders: bool = False      # floating-window style: leave out-of-frame strips black


def zero_plane(disp: torch.Tensor, p: StereoParams) -> float:
    z0 = p.convergence
    if p.stereo_window:
        H, W = disp.shape
        s = max(2, int(round(W * 0.02)))
        border = torch.cat([disp[:, :s].reshape(-1), disp[:, -s:].reshape(-1)])
        z0 = max(z0, robust_quantile(border, 0.98))
    return float(min(max(z0, 0.0), 1.0))


def to_pixel_disparity(disp: torch.Tensor, p: StereoParams) -> tuple[torch.Tensor, float]:
    W = disp.shape[1]
    budget = p.budget_pct / 100.0 * W
    z0 = zero_plane(disp, p)
    return budget * (disp - z0), z0


def hmax_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    """Horizontal-only max filter of an (H, W) map."""
    if r <= 0:
        return x
    y = F.pad(x.unsqueeze(0), (r, r), mode="replicate")
    return F.max_pool1d(y, 2 * r + 1, stride=1)[0]


def sharpen_discontinuities(D: torch.Tensor, fg_r: int, jump_px: float,
                            grad_thr: float = 0.35) -> torch.Tensor:
    """Turn blurred depth *discontinuities* into clean steps, leave slopes alone.

    Monocular predictors (and any upsampling) spread an occlusion boundary over
    several pixels. Warped naively, that ramp becomes a rubber sheet of
    stretched, duplicated "flying" pixels. Horizontal parallax only depends on
    the horizontal disparity gradient, and a surface whose disparity changes by
    more than ``grad_thr`` px per px would be visibly stretched anyway, so such
    runs are treated as occlusion boundaries: every pixel of the run is
    snapped to the nearer of the two plateau values bounding it (scale
    independent: works for 3 px and 30 px ramps alike). The foreground is then
    dilated by ``fg_r`` pixels so the mixed-colour fringe travels with it (it
    is later alpha-matted over the regenerated background instead of being
    left behind as a ghost outline).
    """
    H, W = D.shape
    g = torch.zeros_like(D)
    g[:, 1:-1] = (D[:, 2:] - D[:, :-2]).abs() * 0.5
    g[:, 0] = (D[:, 1] - D[:, 0]).abs()
    g[:, -1] = (D[:, -1] - D[:, -2]).abs()
    ramp = g > grad_thr
    if ramp.any():
        left, right = hole_neighbours(ramp)
        has_l, has_r = left >= 0, right < W
        dl = D.gather(1, left.clamp(0, W - 1))
        dr = D.gather(1, right.clamp(0, W - 1))
        dl = torch.where(has_l, dl, dr)
        dr = torch.where(has_r, dr, dl)
        big = (dl - dr).abs() > jump_px
        snapped = torch.where((D - dl).abs() <= (D - dr).abs(), dl, dr)
        D = torch.where(ramp & big & (has_l | has_r), snapped, D)
    if fg_r > 0:
        dil = hmax_filter(D, fg_r)
        D = torch.where(dil - D > jump_px, dil, D)
    return D


# --------------------------------------------------------------------------- #
# Forward mapping (row-wise mesh rasterisation with z-buffer)
# --------------------------------------------------------------------------- #

def forward_map(D: torch.Tensor, sign: float, max_stretch: float = 2.0, keep: torch.Tensor | None = None,
                rows_per_chunk: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """For every target pixel find the source x-coordinate that lands on it.

    Each row is treated as a 1-D triangle strip: consecutive source pixels form
    a segment that is rasterised into the target if it keeps its orientation
    and is not stretched beyond ``max_stretch`` pixels (otherwise it is a
    disocclusion and left as a hole). Every source pixel is additionally
    point-splatted so one-pixel-wide structures (hair, wires) survive. A
    z-buffer on disparity resolves occlusions exactly.

    ``keep`` (bool, H x W) restricts the mapping to a subset of source pixels
    (used to render the soft-boundary fringe as a separate layer).

    Returns (src_x, zbuf): float maps of shape (H, W); holes have src_x = NaN
    and zbuf = -inf.
    """
    H, W = D.shape
    dev, dt = D.device, torch.float32
    D = D.to(dt)
    K = int(math.ceil(max_stretch)) + 1
    src_out = torch.full((H, W), float("nan"), device=dev, dtype=dt)
    z_out = torch.full((H, W), float("-inf"), device=dev, dtype=dt)
    xs = torch.arange(W, device=dev, dtype=dt)
    for r0 in range(0, H, rows_per_chunk):
        r1 = min(H, r0 + rows_per_chunk)
        h = r1 - r0
        d = D[r0:r1]
        xt = xs + sign * d
        rows = torch.arange(h, device=dev).unsqueeze(1)
        kp = torch.ones_like(d, dtype=torch.bool) if keep is None else keep[r0:r1]

        pos_l, src_l, z_l, row_l = [], [], [], []
        # point splats
        p = torch.round(xt)
        pos_l.append(torch.where(kp, p, torch.full_like(p, -1)))
        src_l.append(xs + (p - xt))
        z_l.append(d)
        row_l.append(rows.expand(h, W))
        # segment rasterisation
        xa, xb = xt[:, :-1], xt[:, 1:]
        span = xb - xa
        seg_ok = (span > 1e-6) & (span <= max_stretch) & kp[:, :-1] & kp[:, 1:]
        da, db = d[:, :-1], d[:, 1:]
        start = torch.ceil(xa)
        for k in range(K):
            pk = start + k
            ok = seg_ok & (pk <= xb)
            t = ((pk - xa) / span.clamp_min(1e-6)).clamp(0, 1)
            pos_l.append(torch.where(ok, pk, torch.full_like(pk, -1)))
            src_l.append(xs[:-1] + t)
            z_l.append(da + t * (db - da))
            row_l.append(rows.expand(h, W - 1))

        pos = torch.cat([a.reshape(-1) for a in pos_l])
        src = torch.cat([a.reshape(-1) for a in src_l])
        z = torch.cat([a.reshape(-1) for a in z_l])
        row = torch.cat([a.reshape(-1) for a in row_l])
        valid = (pos >= 0) & (pos <= W - 1) & (src >= -0.5) & (src <= W - 0.5)
        pos, src, z, row = pos[valid].long(), src[valid], z[valid], row[valid]
        idx = row * W + pos

        zbuf = torch.full((h * W,), float("-inf"), device=dev, dtype=dt)
        zbuf.scatter_reduce_(0, idx, z, reduce="amax", include_self=True)
        win = z >= zbuf[idx] - 1e-4
        sbuf = torch.full((h * W,), float("nan"), device=dev, dtype=dt)
        sbuf[idx[win]] = src[win]
        src_out[r0:r1] = sbuf.view(h, W)
        z_out[r0:r1] = zbuf.view(h, W)
    return src_out, z_out


# --------------------------------------------------------------------------- #
# Depth-aware high-quality resampling
# --------------------------------------------------------------------------- #

def _lanczos(x: torch.Tensor, a: int) -> torch.Tensor:
    ax = x.abs()
    px = math.pi * x
    out = torch.where(ax < 1e-6, torch.ones_like(x),
                      a * torch.sin(px) * torch.sin(px / a) / (px * px).clamp_min(1e-12))
    return torch.where(ax < a, out, torch.zeros_like(x))


def sample_rows(img: torch.Tensor, src_x: torch.Tensor, z: torch.Tensor, D_src: torch.Tensor,
                tol_px: float, a: int = 3, rows_per_chunk: int = 256) -> torch.Tensor:
    """Horizontal Lanczos-``a`` resampling of ``img`` at ``src_x`` (per row).

    Only taps whose source disparity matches the target sample's disparity
    contribute, so foreground and background colours never bleed into each
    other across an occlusion boundary. Integer shifts reproduce source pixels
    exactly; fractional shifts keep full sharpness (no bilinear softening), so
    the synthesised eye matches the source eye's acuity. Holes (NaN) -> 0.
    """
    H, W, C = img.shape
    out = torch.zeros_like(img)
    for r0 in range(0, H, rows_per_chunk):
        r1 = min(H, r0 + rows_per_chunk)
        sx = src_x[r0:r1]
        hole = torch.isnan(sx)
        sx = torch.where(hole, torch.zeros_like(sx), sx)
        zz = torch.where(hole, torch.zeros_like(sx), z[r0:r1])
        im = img[r0:r1]
        ds = D_src[r0:r1]
        x0 = torch.floor(sx)
        frac = sx - x0
        x0 = x0.long()
        # local slope of the source disparity -> tolerance for slanted surfaces
        xi0 = x0.clamp(0, W - 1)
        xi1 = (x0 + 1).clamp(0, W - 1)
        g = (ds.gather(1, xi1) - ds.gather(1, xi0)).abs()
        tol = tol_px + (a + 1) * g
        acc = torch.zeros_like(im)
        wsum = torch.zeros_like(sx)
        lo = torch.full_like(im, float("inf"))
        hi = torch.full_like(im, float("-inf"))
        best_w = torch.full_like(sx, -1.0)
        best_c = torch.zeros_like(im)
        for j in range(-a + 1, a + 1):
            xi = (x0 + j).clamp(0, W - 1)
            dist = frac - j
            w = _lanczos(dist, a)
            ok = (ds.gather(1, xi) - zz).abs() <= tol
            w = torch.where(ok, w, torch.zeros_like(w))
            c = im.gather(1, xi.unsqueeze(-1).expand(-1, -1, C))
            acc += w.unsqueeze(-1) * c
            wsum += w
            if j == 0:
                c_exact, ok_exact = c, ok
            if -1 <= j <= 2:
                okc = ok.unsqueeze(-1)
                lo = torch.where(okc, torch.minimum(lo, c), lo)
                hi = torch.where(okc, torch.maximum(hi, c), hi)
            prox = torch.where(ok, 1.0 - dist.abs() / a, torch.full_like(w, -1.0))
            better = prox > best_w
            best_w = torch.where(better, prox, best_w)
            best_c = torch.where(better.unsqueeze(-1), c, best_c)
        res = acc / wsum.clamp_min(1e-6).unsqueeze(-1)
        # anti-ringing: clamp to the range of the four nearest same-layer taps
        res = torch.maximum(torch.minimum(res, hi), lo)
        weak = (wsum < 0.25).unsqueeze(-1)
        res = torch.where(weak, best_c, res)
        # integer sample positions reproduce the source pixel bit-exactly
        exact = ((frac < 1e-4) & ok_exact).unsqueeze(-1)
        res = torch.where(exact, c_exact, res)
        res = torch.where(hole.unsqueeze(-1), torch.zeros_like(res), res)
        out[r0:r1] = res.clamp(0.0, 1.0)
    return out


# --------------------------------------------------------------------------- #
# Hole analysis, pre-fill and soft compositing masks
# --------------------------------------------------------------------------- #

def hole_neighbours(hole: torch.Tensor):
    """Index of nearest valid pixel to the left / right of every pixel (-1 / W if none)."""
    H, W = hole.shape
    ar = torch.arange(W, device=hole.device).expand(H, W)
    left = torch.where(~hole, ar, torch.full_like(ar, -1)).cummax(dim=1).values
    rv = torch.where(~hole, ar, torch.full_like(ar, W)).flip(1).cummin(dim=1).values.flip(1)
    return left, rv


def prefill_holes(view: torch.Tensor, hole: torch.Tensor, z: torch.Tensor, tol_px: float):
    """Fill disocclusions from the *background* side by mirroring.

    For each hole pixel the neighbour with the smaller disparity (farther) is
    the side that is being revealed; its texture is mirrored into the hole. The
    result is a plausible, structure-free initialisation for diffusion
    inpainting and a reasonable fallback when no generative fill is run.
    Returns (filled_view, bg_index, border_hole) where border_hole marks holes
    that touch the left/right image border (out-of-frame content).
    """
    H, W, C = view.shape
    left, right = hole_neighbours(hole)
    has_l, has_r = left >= 0, right < W
    zl = torch.where(has_l, z.gather(1, left.clamp(0, W - 1)), torch.full_like(z, float("inf")))
    zr = torch.where(has_r, z.gather(1, right.clamp(0, W - 1)), torch.full_like(z, float("inf")))
    use_r = zr <= zl
    bg = torch.where(use_r, right, left).clamp(0, W - 1)
    ar = torch.arange(W, device=view.device).expand(H, W)
    mirror = torch.where(use_r, 2 * right - ar - 1, 2 * left - ar + 1).clamp(0, W - 1)
    zbg = z.gather(1, bg)
    ok = (~hole.gather(1, mirror)) & ((z.gather(1, mirror) - zbg) <= tol_px)
    src = torch.where(ok, mirror, bg)
    filled = view.gather(1, src.unsqueeze(-1).expand(-1, -1, C))
    out = torch.where(hole.unsqueeze(-1), filled, view)
    border = hole & (~has_l | ~has_r)
    none = hole & ~has_l & ~has_r
    out = torch.where(none.unsqueeze(-1), torch.full_like(out, 0.5), out)
    return out, bg, border


def soft_alpha(view_filled: torch.Tensor, hole: torch.Tensor, band: int, eps: float = 2e-3) -> torch.Tensor:
    """Matting-style soft coverage around disocclusions.

    A guided filter of the binary coverage mask, guided by the rendered image,
    yields an alpha that follows real colour edges (hair, fur, foliage) on the
    foreground side of a hole instead of a hard cut, so the fringe can be
    blended over the regenerated background without a halo.
    """
    cov = (~hole).float()
    if band <= 0:
        return cov
    a = guided_filter(luminance(view_filled), cov, band, eps).clamp(0.0, 1.0)
    near = max_filter(hole.float(), band) > 0
    a = torch.where(near, a, torch.ones_like(a))
    a = torch.where(hole, torch.zeros_like(a), a)
    return a


# --------------------------------------------------------------------------- #
# Soft boundaries (hair, fur, whiskers, defocus): two-layer decomposition
# --------------------------------------------------------------------------- #

def _norm_conv(img: torch.Tensor, w: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted local mean of an (H, W, C) image over a (2r+1)^2 window."""
    wf = w.float()
    den = box_filter(wf, r)
    num = torch.stack([box_filter(img[..., c] * wf, r) for c in range(img.shape[-1])], -1)
    return num / den.clamp_min(1e-6).unsqueeze(-1), den


@dataclass
class SoftBoundary:
    fringe: torch.Tensor   # (H, W) bool: source pixels that belong (partly) to a nearer surface
    alpha: torch.Tensor    # (H, W) foreground coverage of fringe pixels
    color: torch.Tensor    # (H, W, C) de-contaminated foreground colour of fringe pixels
    D_fg: torch.Tensor     # (H, W) disparity of the surface the fringe belongs to


def soft_boundary(img: torch.Tensor, D: torch.Tensor, band: int, jump_px: float,
                  min_contrast: float = 0.06, min_alpha: float = 0.12) -> SoftBoundary:
    """Estimate alpha / foreground colour / foreground depth for pixels just
    outside occlusion edges.

    Depth maps put the boundary at the solid silhouette, but hair, fur,
    whiskers and defocus blur extend past it and are mixed with the background
    (I = a*F + (1-a)*B). Left at background depth those strands stay behind
    when the foreground shifts and read as a ghost outline. For every
    background-side pixel within ``band`` px of a nearer surface we estimate the
    local foreground colour F (mean of nearby foreground-edge pixels) and
    background colour B (mean of nearby clean background), solve for alpha by
    projecting I onto the F-B line, and move the de-contaminated strand with the
    foreground disparity as a matted layer. Where F and B are too similar the
    decomposition is unreliable - but then a left-behind fringe is also
    invisible, so nothing is lost by skipping it.
    """
    H, W, C = img.shape
    thr = max(3.0, 2.0 * jump_px)  # parallax below ~3 px leaves no visible ghost
    D_near = max_filter(D, band)
    bandm = (D_near - D) > thr                          # fringe candidates (far side, close to the edge)
    far_side = (max_filter(D, 3 * band) - D) > thr      # far side of an edge, wider neighbourhood
    fg_edge = ((D - min_filter(D, band)) > thr) & ~bandm
    bg_clean = far_side & ~bandm                         # depth-consistent background just beyond the fringe
    F_loc, wf = _norm_conv(img, fg_edge, band)
    B_loc, wb = _norm_conv(img, bg_clean, 3 * band)
    diff = F_loc - B_loc
    c2 = (diff * diff).sum(-1)
    a = ((img - B_loc) * diff).sum(-1) / c2.clamp_min(1e-6)
    a = a.clamp(0.0, 1.0)
    ok = bandm & (wf > 1e-3) & (wb > 1e-3) & (c2 > min_contrast ** 2) & (a > min_alpha)
    a = torch.where(ok, a, torch.zeros_like(a))
    col = (B_loc + (img - B_loc) / a.clamp_min(0.3).unsqueeze(-1)).clamp(0.0, 1.0)
    col = torch.where(ok.unsqueeze(-1), col, img)
    return SoftBoundary(ok, a, col, D_near)


def render_fringe(sb: SoftBoundary, sign: float, z_base: torch.Tensor, jump_px: float):
    """Forward-map the fringe layer; returns (premultiplied colour, alpha) in the target view."""
    H, W, C = sb.color.shape
    src_x, z = forward_map(sb.D_fg, sign, max_stretch=2.0, keep=sb.fringe)
    valid = ~torch.isnan(src_x)
    sx = torch.where(valid, src_x, torch.zeros_like(src_x))
    x0 = torch.floor(sx).long().clamp(0, W - 1)
    x1 = (x0 + 1).clamp(0, W - 1)
    t = (sx - torch.floor(sx)).clamp(0, 1)
    fr = sb.fringe.float()
    w0 = (1 - t) * fr.gather(1, x0)
    w1 = t * fr.gather(1, x1)
    ws = (w0 + w1).clamp_min(1e-6)
    a = (w0 * sb.alpha.gather(1, x0) + w1 * sb.alpha.gather(1, x1)) / ws
    pm = sb.color * sb.alpha.unsqueeze(-1)
    g = lambda im, ix: im.gather(1, ix.unsqueeze(-1).expand(-1, -1, C))
    col = (w0.unsqueeze(-1) * g(pm, x0) + w1.unsqueeze(-1) * g(pm, x1)) / ws.unsqueeze(-1)
    vis = valid & (z >= z_base - jump_px)
    a = torch.where(vis, a, torch.zeros_like(a))
    col = torch.where(vis.unsqueeze(-1), col, torch.zeros_like(col))
    return col, a


# --------------------------------------------------------------------------- #
# Full render
# --------------------------------------------------------------------------- #

@dataclass
class RenderResult:
    view: torch.Tensor          # (H, W, C) rendered opposite eye (base layer), holes pre-filled
    hole: torch.Tensor          # (H, W) bool, disocclusions + out-of-frame + vacated fringe
    inpaint_mask: torch.Tensor  # (H, W) float, region to regenerate
    alpha: torch.Tensor         # (H, W) weight of the rendered base pixels vs. the inpainted ones
    fringe_color: torch.Tensor  # (H, W, C) premultiplied soft-boundary layer (target view)
    fringe_alpha: torch.Tensor  # (H, W) its coverage
    D_src: torch.Tensor         # (H, W) signed pixel disparity (source view, after edge processing)
    zero_plane: float
    black: torch.Tensor         # (H, W) bool, pixels forced black (floating window)
    fg_ref: torch.Tensor        # (H, W, C) local foreground colour next to holes
    bg_ref: torch.Tensor        # (H, W, C) local background colour next to holes

    def preview(self) -> torch.Tensor:
        """Base + fringe without generative fill (geometric pre-fill in holes)."""
        a = self.fringe_alpha.unsqueeze(-1)
        return (self.fringe_color + (1 - a) * self.view).clamp(0, 1)


def wide_holes(hole: torch.Tensor, min_width: int) -> torch.Tensor:
    """Hole regions that are at least ``min_width`` px thick somewhere.

    Thickness is the diameter of the largest disc that fits inside the region
    (from a Euclidean distance transform), so long thin slivers along
    near-horizontal edges are not mistaken for wide gaps.
    """
    if min_width <= 1 or not bool(hole.any()):
        return hole
    from scipy import ndimage
    h_np = hole.cpu().numpy()
    thick = 2.0 * ndimage.distance_transform_edt(h_np)
    lab, n = ndimage.label(h_np, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    if n == 0:
        return hole
    mx = ndimage.maximum(thick, lab, index=range(1, n + 1))
    keep = torch.zeros(n + 1, dtype=torch.bool)
    keep[1:] = torch.from_numpy(mx >= min_width)
    return keep[torch.from_numpy(lab).long()].to(hole.device)


def plan_crop(owned: torch.Tensor, cell: tuple[int, int, int, int], H: int, W: int, max_h: int, max_w: int,
              ctx: int, min_dim: int, mult: int = 16) -> tuple[int, int, int, int] | None:
    """Crop (y0, y1, x0, x1) covering the owned mask pixels of one cell plus ``ctx`` px of context.

    ``owned`` is the (H, W) bool mask of pixels this cell still has to fill.
    The crop is at most (max_h, max_w), at least ``min_dim`` per side, a
    multiple of ``mult``, and lies fully inside the image.
    """
    cy0, cy1, cx0, cx1 = cell
    sub = owned[cy0:cy1, cx0:cx1]
    if not bool(sub.any()):
        return None
    ys = torch.nonzero(sub.any(1)).flatten()
    xs = torch.nonzero(sub.any(0)).flatten()
    by0, by1 = cy0 + int(ys[0]), cy0 + int(ys[-1]) + 1
    bx0, bx1 = cx0 + int(xs[0]), cx0 + int(xs[-1]) + 1

    def span(b0, b1, size, cap):
        want = (b1 - b0) + 2 * ctx
        want = max(want, min(min_dim, cap))
        want = min(cap, ((want + mult - 1) // mult) * mult)
        c = (b0 + b1) // 2
        a = min(max(0, c - want // 2), size - want)
        return a, a + want

    y0, y1 = span(by0, by1, H, max_h)
    x0, x1 = span(bx0, bx1, W, max_w)
    return y0, y1, x0, x1


def render_opposite_eye(image: torch.Tensor, disp: torch.Tensor, p: StereoParams) -> RenderResult:
    H, W, C = image.shape
    md = max(H, W)
    fg_r = p.fg_dilate
    band = p.edge_band or auto_px(md, 1000, 2)
    soft = p.soft_band if p.soft_band >= 0 else auto_px(md, 300, 3)
    sign = -1.0 if p.source_eye == "left" else 1.0
    image = image.float()

    D, z0 = to_pixel_disparity(disp.float(), p)
    D = sharpen_discontinuities(D, fg_r, p.jump_px, p.grad_thr)

    if soft > 0:
        sb = soft_boundary(image, D, soft, p.jump_px)
        keep = ~sb.fringe
    else:
        sb, keep = None, None
    src_x, z = forward_map(D, sign, p.max_stretch, keep=keep)
    hole = torch.isnan(src_x)
    view = sample_rows(image, src_x, z, D, tol_px=p.jump_px)
    filled, bg, border = prefill_holes(view, hole, z, p.jump_px)

    if sb is not None and bool(sb.fringe.any()):
        z_base = torch.where(hole, torch.full_like(z, float("-inf")), z)
        f_col, f_a = render_fringe(sb, sign, z_base, p.jump_px)
    else:
        f_col, f_a = torch.zeros_like(image), torch.zeros(H, W, device=image.device)

    black = border & p.black_borders
    alpha = soft_alpha(filled, hole & ~black, band)
    alpha = torch.where(black, torch.ones_like(alpha), alpha)
    filled = torch.where(black.unsqueeze(-1), torch.zeros_like(filled), filled)
    f_a = torch.where(black, torch.zeros_like(f_a), f_a)
    f_col = torch.where(black.unsqueeze(-1), torch.zeros_like(f_col), f_col)
    # Only gaps wide enough to need new content go to the (expensive) generative
    # fill; slivers behind hair, whiskers and small edges keep the background
    # mirror pre-fill, which is indistinguishable at that width.
    min_fill = p.min_fill_px if p.min_fill_px >= 0 else max(6, int(round(W / 400)))
    gen = wide_holes(hole & ~black, min_fill)
    inpaint = max_filter(gen.float(), band + 2)
    guard_px = p.fg_guard if p.fg_guard >= 0 else auto_px(md, 60, 16)
    R = max(guard_px, 4 * band)
    thr = max(3.0, 2 * p.jump_px)
    z_t = torch.where(hole, z.gather(1, bg), z)
    hz = torch.where(hole & ~black, z_t, torch.full_like(z_t, float("-inf")))
    zh = max_filter(hz, R)
    near = torch.isfinite(zh) & ~hole
    fg_side = near & (z_t > zh + thr)
    if guard_px > 0:
        # Hide the foreground next to each disocclusion from the inpainter, so it
        # continues the *background* instead of extending the foreground into the
        # hole. Those pixels are restored from the exact render (alpha = 1) at
        # composite time; only their soft edge blends with the new background.
        hz_gen = torch.where(gen, z_t, torch.full_like(z_t, float("-inf")))
        guard = fg_side & (max_filter(hz_gen, guard_px) > float("-inf"))
        inpaint = torch.maximum(inpaint, guard.float())
    # colour references for the post-fill "foreground bleed" check
    fg_ref, _ = _norm_conv(filled, fg_side, R)
    bg_ref, _ = _norm_conv(filled, near & ~fg_side, R)
    return RenderResult(filled, hole, inpaint, alpha, f_col, f_a, D, z0, black, fg_ref, bg_ref)


def reject_foreground_bleed(inpainted: torch.Tensor, prefill: torch.Tensor, alpha: torch.Tensor,
                            fg_ref: torch.Tensor, bg_ref: torch.Tensor, min_contrast: float = 0.1) -> torch.Tensor:
    """Undo generative fill that extends the foreground into a disocclusion.

    A disocclusion can only reveal *background*. Where the inpainted colour
    projects onto the local foreground colour rather than the local background
    colour (and the two are distinguishable), fall back to the
    background-mirrored geometric pre-fill. Model-independent safety net.
    """
    diff = fg_ref - bg_ref
    c2 = (diff * diff).sum(-1)
    a = ((inpainted - bg_ref) * diff).sum(-1) / c2.clamp_min(1e-6)
    w = ((a - 0.45) / 0.35).clamp(0, 1)
    w = w * w * (3 - 2 * w)
    w = w * (c2 > min_contrast ** 2).float() * (1 - alpha)
    return inpainted + w.unsqueeze(-1) * (prefill - inpainted)


def composite(rendered: torch.Tensor, inpainted: torch.Tensor | None, alpha: torch.Tensor,
              fringe_color: torch.Tensor | None = None, fringe_alpha: torch.Tensor | None = None) -> torch.Tensor:
    """Final opposite eye: base render over the generative fill, then the soft-boundary layer on top."""
    out = _blend_fill(rendered, inpainted, alpha)
    if fringe_color is not None and fringe_alpha is not None:
        out = fringe_color + (1 - fringe_alpha.unsqueeze(-1)) * out
    return out.clamp(0.0, 1.0)


def _blend_fill(rendered: torch.Tensor, inpainted: torch.Tensor | None, alpha: torch.Tensor) -> torch.Tensor:
    if inpainted is None:
        return rendered
    if inpainted.shape != rendered.shape:
        inpainted = F.interpolate(inpainted.permute(2, 0, 1).unsqueeze(0), size=rendered.shape[:2],
                                  mode="bicubic", align_corners=False)[0].permute(1, 2, 0)
    a = alpha.unsqueeze(-1)
    return (rendered * a + inpainted * (1 - a)).clamp(0.0, 1.0)


def anaglyph_dubois(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Red/cyan Dubois anaglyph (for quick depth checks without a viewer)."""
    ml = torch.tensor([[0.456, 0.500, 0.176], [-0.040, -0.038, -0.016], [-0.015, -0.021, -0.005]],
                      device=left.device, dtype=left.dtype)
    mr = torch.tensor([[-0.043, -0.088, -0.002], [0.378, 0.734, -0.018], [-0.072, -0.113, 1.226]],
                      device=left.device, dtype=left.dtype)
    lin = lambda x: torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    out = lin(left[..., :3]) @ ml.T + lin(right[..., :3]) @ mr.T
    out = out.clamp(0, 1)
    return torch.where(out <= 0.0031308, out * 12.92, 1.055 * out.clamp_min(1e-8) ** (1 / 2.4) - 0.055)


def disparity_preview(D: torch.Tensor) -> torch.Tensor:
    """Diverging preview: red = in front of the screen, blue = behind, white = screen plane."""
    m = float(D.abs().max().clamp_min(1e-6))
    v = (D / m).clamp(-1, 1)
    pos, neg = v.clamp_min(0), (-v).clamp_min(0)
    r = 1 - neg
    g = 1 - pos - neg
    b = 1 - pos
    return torch.stack([r, g.clamp(0, 1), b], -1)
