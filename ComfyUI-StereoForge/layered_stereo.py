"""
Layered depth-image-based stereo (StereoForge v2).

Pipeline (canonical: source = LEFT eye, synthesise RIGHT eye; parallel rig):

  1. Monocular metric geometry (Depth Anything 3) at ~1 MP -> depth Z, intrinsics K.
  2. Reflective-plane module:
       * fit the dominant ground/floor plane from the DA3 point cloud (RANSAC)
       * mirror the visible scene across that plane and render it into the camera:
         this predicts what a mirror on that plane would show (colour M) and at
         which *virtual* depth (Zv)
       * wherever the photo actually matches that prediction on the plane, the
         pixel shows a reflection: it gets the virtual depth instead of the
         surface depth DA3 gave it (DA3 paints puddle reflections onto the ground)
  3. Full-resolution rendering: edge-snapped disparity upsampling, z-buffered
     splatting, bicubic resampling of the ORIGINAL pixels (from stereo_core).
  4. Disocclusion fill (background-only context).

Every module is confidence-gated: when it is unsure it leaves the DA3 depth alone,
so the worst case is the plain DA3 warp.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from . import stereo_core as sc

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def backproject(Z: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = Z.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x = (u - K[0, 2]) / K[0, 0] * Z
    y = (v - K[1, 2]) / K[1, 1] * Z
    return np.stack([x, y, Z], -1)


def fit_ground_plane(P: np.ndarray, valid: np.ndarray, iters: int = 400, seed: int = 0):
    """
    RANSAC plane among plausible floor points: lower 65% of the frame, normal within
    75 deg of camera 'up' (-y).  Returns (n, d) with n.X + d = 0, n pointing towards
    the camera side (so height = n.X + d > 0 above the floor), inlier mask, score.
    """
    h, w, _ = P.shape
    rng = np.random.default_rng(seed)
    rows = np.arange(h)[:, None].repeat(w, 1)
    cand = valid & (rows > 0.35 * h)
    pts = P[cand]
    if len(pts) < 500:
        return None
    sub = pts[rng.choice(len(pts), min(len(pts), 60000), replace=False)]
    best = None
    for _ in range(iters):
        a, b, c = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        if n[1] > 0:  # make it point 'up' (camera y is down)
            n = -n
        if -n[1] < math.cos(math.radians(75)):
            continue
        d = -n @ a
        # scale-aware threshold: 1.5% of distance
        dist = np.abs(sub @ n + d)
        thr = 0.015 * np.linalg.norm(sub, axis=1)
        score = (dist < thr).sum()
        if best is None or score > best[0]:
            best = (score, n, d)
    if best is None:
        return None
    _, n, d = best
    # least-squares refine on inliers
    for _ in range(2):
        dist = np.abs(pts @ n + d)
        inl = pts[dist < 0.015 * np.linalg.norm(pts, axis=1)]
        c = inl.mean(0)
        _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
        n = vt[2]
        if n[1] > 0:
            n = -n
        d = -n @ c
    height = P @ n + d
    inlier = valid & (np.abs(height) < 0.02 * np.linalg.norm(P, axis=-1))
    # camera must be above the plane (origin height = d > 0)
    if d <= 0:
        return None
    return dict(n=n, d=d, inlier=inlier, frac=float(inlier.mean()))


def render_mirror(P: np.ndarray, img: np.ndarray, K: np.ndarray, plane, valid: np.ndarray,
                  min_height_frac: float = 0.02):
    """
    Mirror every scene point that lies above the plane and splat it into the camera.
    Returns virtual depth Zv (z of the mirrored point, i.e. the depth a stereo pair sees
    for the reflection), mirrored colour M, and a hit mask.  Nearest mirrored point wins.
    """
    h, w, _ = P.shape
    n, d = plane["n"], plane["d"]
    height = P @ n + d
    above = valid & (height > min_height_frac * np.linalg.norm(P, axis=-1))
    X = P[above]
    col = img[above]
    Xm = X - 2 * (X @ n + d)[:, None] * n[None, :]
    z = Xm[:, 2]
    ok = z > 1e-3
    Xm, z, col = Xm[ok], z[ok], col[ok]
    u = K[0, 0] * Xm[:, 0] / z + K[0, 2]
    v = K[1, 1] * Xm[:, 1] / z + K[1, 2]
    ui, vi = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    ins = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui, vi, z, col = ui[ins], vi[ins], z[ins], col[ins]
    # z-buffer: sort far->near so nearest is written last
    order = np.argsort(-z)
    ui, vi, z, col = ui[order], vi[order], z[order], col[order]
    Zv = np.full((h, w), np.inf, np.float32)
    M = np.zeros((h, w, 3), np.float32)
    Zv[vi, ui] = z
    M[vi, ui] = col
    hit = np.isfinite(Zv)
    return Zv, M, hit


def local_ncc(a: np.ndarray, b: np.ndarray, r: int):
    ta = torch.from_numpy(a.astype(np.float32))[None, None]
    tb = torch.from_numpy(b.astype(np.float32))[None, None]
    ma, mb = sc.box(ta, r), sc.box(tb, r)
    va = (sc.box(ta * ta, r) - ma * ma).clamp_min(0)
    vb = (sc.box(tb * tb, r) - mb * mb).clamp_min(0)
    cov = sc.box(ta * tb, r) - ma * mb
    ncc = cov / (torch.sqrt(va * vb) + 1e-4)
    return ncc[0, 0].numpy(), torch.sqrt(torch.minimum(va, vb))[0, 0].numpy()


# ----------------------------------------------------------------------------
# reflection analysis
# ----------------------------------------------------------------------------
def _t(a):
    a = torch.from_numpy(np.ascontiguousarray(a).astype(np.float32))
    return a[None, None] if a.dim() == 2 else a.permute(2, 0, 1)[None]


def _n(t):
    t = t[0].detach().cpu()
    return t[0].numpy() if t.shape[0] == 1 else t.permute(1, 2, 0).numpy()


def luma_np(x):
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def reflection_layers(img, Z, K, region, R_sep=None):
    """
    All inputs on the depth grid (h x w):
      img     HxWx3 [0,1]
      Z       surface depth (DA3)
      K       intrinsics
      region  soft reflector mask (semantic, e.g. GroundingDINO+SAM2+GrabCut), or None
      R_sep   reflection layer from a single-image reflection-separation net (RDNet), or None
    Returns dict: rho (reflection fraction of each pixel), dv_scale (1/Zv, 0 = infinity),
    region (gated), plane, and debug maps.  rho == 0 everywhere means "plain DA3 warp".
    """
    h, w = Z.shape
    valid = np.isfinite(Z) & (Z > 0)
    P = backproject(Z, K)
    plane = fit_ground_plane(P, valid)
    out = dict(plane=plane, rho=np.zeros((h, w), np.float32), invZv=1.0 / np.maximum(Z, 1e-6),
               region=np.zeros((h, w), np.float32))
    if plane is None or region is None:
        return out
    n, d = plane["n"], plane["d"]
    dist = np.linalg.norm(P, axis=-1)
    height = P @ n + d
    # a floor reflection can only live on (or, if DA3 followed the reflection, below) the floor plane
    on_floor = valid & (height < 0.05 * dist)
    on_floor_t = sc.gaussian_blur(sc.dilate(_t(on_floor), 2), 2.0)
    reg = np.clip(region, 0, 1) * _n(on_floor_t)
    if cv2 is not None:
        # reflectors are (nearly) simply connected: fill holes left by segmentation/gating
        rb = (reg > 0.5).astype(np.uint8)
        ff = rb.copy()
        cnts, _ = cv2.findContours(rb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ff, cnts, -1, 1, thickness=cv2.FILLED)
        reg = np.maximum(reg, ff.astype(np.float32))
    if reg.max() < 0.5:
        return out

    # --- predicted mirror image and virtual depth (in-frame reflected content) ---
    Zv, M, hit = render_mirror(P, img, K, plane, valid)
    inv_hit = np.where(hit, 1.0 / Zv, 0.0)
    # close splat gaps (nearest wins -> max of inverse depth), only a few px
    inv_c = _n(sc.dilate(_t(inv_hit), 2))
    hit_c = _n(sc.dilate(_t(hit.astype(np.float32)), 2)) > 0.5
    Mf = _n(sc._push_pull_fill(_t(M) * _t(hit.astype(np.float32)), _t(hit.astype(np.float32))))
    # reflected content that is not in frame (sky, overhead branches): at infinity
    invZv = np.where(hit_c, inv_c, 0.0)
    invZv = np.minimum(invZv, 1.0 / np.maximum(Z, 1e-6))  # virtual image is never in front of the surface

    # --- how much of each pixel's light is reflected: Fresnel + brightness budget ---
    # I = F * Env + (1 - F) * T_wet.  F (water, n=1.33) depends only on the angle between
    # the viewing ray and the floor normal: ~1 at grazing angles, ~0.02 looking straight
    # down (that is why bricks show through shallow water).  T_wet is estimated from the
    # dry floor at the same distance, darkened by wetting.
    yI = luma_np(img)
    rays = P / np.maximum(dist[..., None], 1e-9)
    cos_i = np.clip(np.abs(rays @ n), 0, 1)
    F0 = 0.02
    Fr = F0 + (1 - F0) * (1 - cos_i) ** 5
    floor_px = on_floor & (reg < 0.2)
    band = max(8, h // 24)
    Lrow = np.full(h, np.nan, np.float32)
    for r0 in range(0, h, band):
        sel = floor_px[r0:r0 + band]
        if sel.sum() > 50:
            Lrow[r0:r0 + band] = np.median(yI[r0:r0 + band][sel])
    if np.isnan(Lrow).all():
        Lrow[:] = np.percentile(yI[on_floor], 50) if on_floor.any() else np.percentile(yI, 50)
    idx = np.arange(h)
    ok = ~np.isnan(Lrow)
    Lrow = np.interp(idx, idx[ok], Lrow[ok])[:, None]
    T_wet = 0.6 * Lrow
    refl = yI - (1 - Fr) * T_wet
    rho_f = np.clip(refl / np.maximum(yI, 0.03), 0, 1)
    # Fresnel cap: even an environment ENV_RATIO times brighter than the wet floor can
    # only claim rho_max of the pixel at this angle (keeps submerged texture on the floor)
    ENV_RATIO = 30.0
    rho_max = Fr * ENV_RATIO / (Fr * ENV_RATIO + (1 - Fr))
    rho_f = np.minimum(rho_f, rho_max)
    # reflected objects that are visible in frame (dark ones too) via mirror consistency
    ncc_s, _ = local_ncc(yI, luma_np(Mf), r=max(3, int(round(min(h, w) / 120))))
    ncc_l, _ = local_ncc(yI, luma_np(Mf), r=max(6, int(round(min(h, w) / 50))))
    e_mirror = np.clip((np.maximum(ncc_s, ncc_l) - 0.35) / 0.4, 0, 1) * hit_c
    e_mirror = _n(sc.gaussian_blur(sc.dilate(_t(e_mirror), 2), 2.0))
    rho_s = np.maximum(rho_f, e_mirror)
    rho_s = _n(sc.gaussian_blur(_t(rho_s), 1.5))
    rho_s = (np.clip(rho_s, 0, 1) * (reg > 0.5)).astype(np.float32)
    dens, e_bright, e_smooth = rho_f, Fr, cos_i

    # --- partial (thin film / glossy) reflections: take the separation net's layer ---
    R_part = None
    if R_sep is not None:
        R_part = (R_sep * (reg > 0.5)[..., None] * (1 - rho_s)[..., None]).astype(np.float32)
    out.update(rho=rho_s, R_part=R_part, invZv=invZv.astype(np.float32),
               region=(reg > 0.5).astype(np.float32), M=Mf, hit=hit, e_mirror=e_mirror, e_bright=e_bright, dens=dens,
               e_smooth=e_smooth)
    return out


def extend_horizontal_mirror(img: torch.Tensor, mask: torch.Tensor, max_px: int):
    """
    Extend `img` beyond `mask` (1 = valid) horizontally by mirroring the valid texture
    about the nearest boundary in the same row.  Used for the reflection layer, which
    only ever slides horizontally under the reflector outline: mirrored texture keeps
    sharpness and statistics where a push-pull fill would smear.
    """
    m = (mask[0, 0] > 0.5).cpu().numpy()
    h, w = m.shape
    xs = np.arange(w)[None, :].repeat(h, 0)
    li = np.where(m, xs, -1)
    li = np.maximum.accumulate(li, axis=1)  # nearest valid at or left of x
    ri = np.where(m, xs, w)
    ri = np.minimum.accumulate(ri[:, ::-1], axis=1)[:, ::-1]  # nearest valid at or right of x
    dl = np.where(li >= 0, xs - li, 10 ** 9)
    dr = np.where(ri < w, ri - xs, 10 ** 9)
    use_l = dl <= dr
    b = np.where(use_l, li, ri)
    dist = np.where(use_l, dl, dr)
    src_x = np.where(use_l, 2 * b - xs, 2 * b - xs)  # mirror about the boundary pixel
    src_x = np.clip(src_x, 0, w - 1)
    # the mirrored sample must itself be valid; otherwise fall back to the boundary pixel
    rows = np.arange(h)[:, None].repeat(w, 1)
    ok = m[rows, src_x]
    src_x = np.where(ok, src_x, np.clip(b, 0, w - 1))
    fill = (~m) & (dist <= max_px) & (b >= 0) & (b < w)
    idx = torch.from_numpy(rows * w + src_x).to(img.device)
    flat = img.reshape(img.shape[0], img.shape[1], -1)
    ext = torch.gather(flat, 2, idx.reshape(1, 1, -1).expand(img.shape[0], img.shape[1], -1)).reshape(img.shape)
    fillt = torch.from_numpy(fill).to(img)[None, None]
    return img * (1 - fillt) + ext * fillt


# ----------------------------------------------------------------------------
# full-resolution layered rendering
# ----------------------------------------------------------------------------
def _splat_render(img, d_src, mode="bicubic"):
    """
    Returns the warped image, target-grid disparity and a 'covered' mask.  Pixels the
    new eye sees beyond the photo's frame edge are filled by mirroring the image across
    that edge (sharp, texture-consistent; such a strip is monocular, like the view past
    a window frame) and count as covered; only interior disocclusions remain holes.
    """
    d_tgt, hit = sc.splat_disparity(d_src)
    out, inframe = sc.backward_sample(img, d_tgt, mode=mode, padding="reflection")
    covered = torch.maximum(hit * inframe, 1 - inframe)
    return out, d_tgt, covered


def render_right_eye(src, Z, K, layers=None, baseline_scale=1.0, inpaint=None, hole_dilate=2):
    """
    src    [1,3,H,W] full-res source (left eye)
    Z, K   DA3 surface depth + intrinsics on the depth grid (h x w)
    layers output of reflection_layers() (or None for the plain DA3 warp)
    baseline_scale: disparity (px, full res) = baseline_scale * fx_full / Z
    Returns right eye [1,3,H,W], debug dict.
    """
    _, _, H, W = src.shape
    h, w = Z.shape
    sx = W / w
    fx = K[0, 0] * sx
    inv = _t(1.0 / np.maximum(Z, 1e-6))
    d_lr = inv * (baseline_scale * fx / sx)  # disparity in low-res px
    d_s = sc.upsample_disparity(d_lr, src, sx)  # full-res px, edge snapped
    dbg = {"d_s": d_s}

    has_refl = layers is not None and (float(layers["rho"].max()) > 1e-3 or
                                          (layers.get("R_part") is not None and float(layers["R_part"].max()) > 1e-3))
    if not has_refl:
        right, d_t, hit = _splat_render(src, d_s)
        holes = sc.dilate(1 - hit, hole_dilate)
    else:
        up = lambda a, m="bilinear": F.interpolate(_t(a), size=(H, W), mode=m, align_corners=False)
        g = sc.luma(src)
        region = sc.guided_filter(g, up(layers["region"]), r=max(2, int(sx * 2)), eps=1e-3).clamp(0, 1)
        region = (region > 0.5).float()
        # extend rho past the low-res outline first, so the only edge is the sharp
        # full-res outline (a soft rho fade at the rim separates into a seam)
        reg_lr = _t(layers["region"])
        rho_lr = sc._push_pull_fill(_t(layers["rho"]) * reg_lr, reg_lr)
        rho = sc.guided_filter(g, F.interpolate(rho_lr, size=(H, W), mode="bilinear", align_corners=False),
                               r=max(2, int(sx * 2)), eps=1e-3).clamp(0, 1) * region
        # reflection layer at full resolution: low frequencies split by the reflected
        # fraction, fine detail goes to the reflection only where it clearly dominates
        # (a fractional split of texture would ghost when the layers move differently)
        sig = max(1.5, 1.5 * sx)
        low = sc.gaussian_blur(src, sig)
        R = rho * low + sc.smoothstep(rho, 0.45, 0.75) * (src - low)
        if layers.get("R_part") is not None:
            Rp = sc._push_pull_fill(_t(layers["R_part"]) * reg_lr, reg_lr)
            Rp = F.interpolate(Rp, size=(H, W), mode="bicubic", align_corners=False).clamp(0, 1)
            R = R + Rp * (1 - rho) * region
        R = torch.minimum(R, src)
        T = src - R
        # surface layer and the reflector outline move with the surface
        T_w, d_t, hit = _splat_render(T, d_s)
        reg_w, _ = sc.backward_sample(region.repeat(1, 3, 1, 1), d_t, mode="bilinear")
        reg_w = reg_w[:, :1].clamp(0, 1)
        # the reflected image moves with the virtual depth; extend it past the outline
        # so sliding under the rim never reveals black
        d_v = F.interpolate(_t(layers["invZv"]) * (baseline_scale * fx / sx), size=(H, W), mode="bilinear",
                            align_corners=False) * sx
        slide = int(math.ceil(float((d_s - d_v).abs().max()))) + 4
        # rim pixels are anti-aliased cobble/water mixtures: don't use them as reflection texture
        core = sc.erode(region, 2)
        R_ext = extend_horizontal_mirror(R * core, core, slide + 2)
        R_ext = sc._push_pull_fill(R_ext, torch.maximum(region, (R_ext.sum(1, keepdim=True) > 0).float()))
        R_w, d_vt, hit_v = _splat_render(R_ext, d_v)
        right = (T_w + R_w * reg_w).clamp(0, 1)
        holes = sc.dilate(1 - hit, hole_dilate)
        dbg.update(rho=rho, d_v=d_v, T=T, R=R)
    dbg["holes"] = holes
    if inpaint is not None and float(holes.sum()) > 0:
        right = inpaint(right, holes)
    return right.clamp(0, 1), dbg


def lama_inpaint_fn():
    from simple_lama_inpainting import SimpleLama
    from PIL import Image

    lama = SimpleLama()

    def run(img, mask):
        im = Image.fromarray((img[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
        mk = Image.fromarray(((mask[0, 0].cpu().numpy() > 0.5) * 255).astype(np.uint8))
        o = np.asarray(lama(im, mk)).astype(np.float32)[: img.shape[2], : img.shape[3]] / 255.0
        o = torch.from_numpy(o).permute(2, 0, 1)[None].to(img)
        m = mask.to(img)
        return img * (1 - m) + o * m

    return run


def choose_baseline_scale(Z, K, W_full, w_lr, target_frac=0.03, pct=98):
    """Scale so the pct-percentile surface disparity equals target_frac of the width."""
    fx = K[0, 0] * W_full / w_lr
    inv = 1.0 / np.maximum(Z, 1e-6)
    return target_frac * W_full / (fx * np.percentile(inv, pct))
