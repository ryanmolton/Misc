"""
Full-resolution stereo "lifting" core for StereoForge.

Every function works in the canonical configuration

    source = LEFT eye, target = RIGHT eye   (target pixel x  <-  source pixel x + d)

A left-eye target is handled by the caller by mirroring all inputs horizontally,
running the canonical path and mirroring the result back (pure geometry, so the
mirror is exact).

Tensors are float32, layout [B, C, H, W], values in [0, 1] unless noted.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# ----------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------
def to_bchw(img_bhwc: torch.Tensor) -> torch.Tensor:
    return img_bhwc[..., :3].permute(0, 3, 1, 2).contiguous()


def to_bhwc(img_bchw: torch.Tensor) -> torch.Tensor:
    return img_bchw.permute(0, 2, 3, 1).contiguous()


def luma(x: torch.Tensor) -> torch.Tensor:
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def fit_processing_size(h: int, w: int, megapixels: float, multiple: int = 64):
    """Aspect-preserving size with area ~= megapixels, both sides multiples of `multiple`."""
    target = megapixels * 1e6
    s = math.sqrt(target / float(h * w))
    ph = max(multiple, int(round(h * s / multiple)) * multiple)
    pw = max(multiple, int(round(w * s / multiple)) * multiple)
    return ph, pw


def resize(x: torch.Tensor, h: int, w: int, mode: str = "bicubic") -> torch.Tensor:
    if x.shape[-2:] == (h, w):
        return x
    down = h < x.shape[-2] or w < x.shape[-1]
    if mode == "area" or (down and mode == "bicubic"):
        return F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
    if mode == "nearest":
        return F.interpolate(x, size=(h, w), mode="nearest")
    return F.interpolate(x, size=(h, w), mode=mode, align_corners=False)


def box(x: torch.Tensor, r: int) -> torch.Tensor:
    """Mean filter of radius r (window 2r+1) via cumulative sums, O(1) per pixel."""
    if r <= 0:
        return x
    H, W = x.shape[-2:]
    r = min(r, (min(H, W) - 1) // 2) if min(H, W) > 2 else 0
    if r <= 0:
        return x
    xp = F.pad(x, (r + 1, r, r + 1, r), mode="replicate")
    c = xp.cumsum(-1)
    c = c[..., 2 * r + 1:] - c[..., : -(2 * r + 1)]
    c = c.cumsum(-2)
    c = c[..., 2 * r + 1:, :] - c[..., : -(2 * r + 1), :]
    return c / float((2 * r + 1) ** 2)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    k = int(math.ceil(3 * sigma)) * 2 + 1
    t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-0.5 * (t / sigma) ** 2)
    g = g / g.sum()
    C = x.shape[1]
    pad = k // 2
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect" if x.shape[-1] > pad else "replicate")
    x = F.conv2d(x, g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    x = F.pad(x, (0, 0, pad, pad), mode="reflect" if x.shape[-2] > pad else "replicate")
    x = F.conv2d(x, g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)
    return x


def guided_filter(guide: torch.Tensor, src: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    """Grey-guide guided filter (He et al. 2010)."""
    mI = box(guide, r)
    mp = box(src, r)
    cov = box(guide * src, r) - mI * mp
    var = box(guide * guide, r) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return box(a, r) * guide + box(b, r)


def dilate(mask: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return mask
    return F.max_pool2d(mask, 2 * r + 1, stride=1, padding=r)


def erode(mask: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return mask
    return 1.0 - F.max_pool2d(1.0 - mask, 2 * r + 1, stride=1, padding=r)


def smoothstep(x, lo, hi):
    t = ((x - lo) / max(hi - lo, 1e-6)).clamp(0, 1)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------
# disparity estimation on the low-resolution generated pair
# ----------------------------------------------------------------------------
def _sgbm(left_u8: np.ndarray, right_u8: np.ndarray, min_disp: int, num_disp: int, block: int):
    m = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=-1,
        uniquenessRatio=5,
        speckleWindowSize=0,
        speckleRange=0,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_HH,
    )
    return m.compute(left_u8, right_u8).astype(np.float32) / 16.0


_RAFT = {}


def _raft(device):
    key = str(device)
    if key not in _RAFT:
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        _RAFT.clear()
        _RAFT[key] = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False).eval().to(device)
    return _RAFT[key]


def _warp_x(t: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor = None):
    B, C, H, W = t.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=t.device, dtype=t.dtype),
                            torch.arange(W, device=t.device, dtype=t.dtype), indexing="ij")
    gx = (xs[None] + dx[:, 0] + 0.5) / W * 2 - 1
    gy = (ys[None] + (dy[:, 0] if dy is not None else 0) + 0.5) / H * 2 - 1
    return F.grid_sample(t, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border", align_corners=False)


@torch.no_grad()
def estimate_disparity_lr(src_lr: torch.Tensor, gen_lr: torch.Tensor, device=None, iters: int = 24):
    """
    Binocular (not monocular!) correspondence between the low-res source (left) and
    the generated view (right) with RAFT optical flow (torchvision, BSD), measured on
    source pixels.  Forward/backward consistency, in-frame and epipolar (vertical
    flow) checks give the reliability mask.
    Returns  disp  [1,1,h,w] (dense, low-res px, target x = x - disp)
             valid [1,1,h,w]
    """
    device = device or src_lr.device
    try:
        model = _raft(device)
    except Exception as e:  # no weights / offline -> classical fallback
        print(f"[StereoForge] RAFT unavailable ({e}); falling back to OpenCV SGBM")
        return _estimate_disparity_sgbm(src_lr, gen_lr)
    h, w = src_lr.shape[-2:]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    a = F.pad(src_lr.to(device) * 2 - 1, (0, pw, 0, ph), mode="replicate")
    b = F.pad(gen_lr.to(device) * 2 - 1, (0, pw, 0, ph), mode="replicate")
    fw = model(a, b, num_flow_updates=iters)[-1][..., :h, :w].float()
    bw = model(b, a, num_flow_updates=iters)[-1][..., :h, :w].float()
    fx, fy = fw[:, 0:1], fw[:, 1:2]
    bw_at = _warp_x(bw, fx, fy)
    err = torch.sqrt((fx + bw_at[:, 0:1]) ** 2 + (fy + bw_at[:, 1:2]) ** 2)
    mag = torch.sqrt(fx ** 2 + fy ** 2)
    xs = torch.arange(w, device=fx.device, dtype=fx.dtype).view(1, 1, 1, w)
    valid = (err < 0.75 + 0.05 * mag) & (fy.abs() < 1.0) & ((xs + fx) >= 0) & ((xs + fx) <= w - 1)
    disp = (-fx).clamp_min(-2.0)
    return disp.cpu(), valid.float().cpu()


def _estimate_disparity_sgbm(src_lr: torch.Tensor, gen_lr: torch.Tensor, max_disp_frac: float = 0.15,
                             lrc_thresh: float = 1.0):
    if cv2 is None:
        raise RuntimeError("opencv-python is required (pip install opencv-python-headless)")
    h, w = src_lr.shape[-2:]
    L = (src_lr[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)[..., ::-1]
    R = (gen_lr[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)[..., ::-1]
    min_d = -8
    num = max(32, int(math.ceil((max_disp_frac * w - min_d) / 16.0)) * 16)
    pad = num + 16  # OpenCV leaves the first minDisp+numDisp columns unmatched: pad them away
    padl = lambda im: cv2.copyMakeBorder(np.ascontiguousarray(im), 0, 0, pad, 0, cv2.BORDER_REPLICATE)
    dL = _sgbm(padl(L), padl(R), min_d, num, 5)[:, pad:]
    dR = _sgbm(padl(R[:, ::-1]), padl(L[:, ::-1]), min_d, num, 5)[:, pad:][:, ::-1]
    bad = dL <= (min_d - 0.5)
    badR = dR <= (min_d - 0.5)
    xs = np.arange(w, dtype=np.float32)[None, :].repeat(h, 0)
    xr = np.clip(np.round(xs - dL).astype(np.int64), 0, w - 1)
    dR_at = np.take_along_axis(np.where(badR, np.nan, dR), xr, axis=1)
    lrc = np.abs(dL - dR_at) <= lrc_thresh
    valid = (~bad) & lrc & np.isfinite(dR_at) & ((xs - dL) >= 0)
    d = torch.from_numpy(np.where(valid, dL, 0).astype(np.float32))[None, None]
    v = torch.from_numpy(valid.astype(np.float32))[None, None]
    return _push_pull_fill(d, v), v


def _row_background_fill(disp: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """For every invalid pixel: min(nearest valid to the left, nearest valid to the right)."""
    h, w = disp.shape
    big = 1e9
    idx = np.where(valid, np.arange(w)[None, :], -1)
    left_idx = np.maximum.accumulate(idx, axis=1)
    idx_r = np.where(valid, np.arange(w)[None, :], w)
    right_idx = np.minimum.accumulate(idx_r[:, ::-1], axis=1)[:, ::-1]
    rows = np.arange(h)[:, None]
    lv = np.where(left_idx >= 0, disp[rows, np.clip(left_idx, 0, w - 1)], big)
    rv = np.where(right_idx < w, disp[rows, np.clip(right_idx, 0, w - 1)], big)
    fill = np.minimum(lv, rv)
    fill = np.where(fill >= big, 0.0, fill)
    return np.where(valid, disp, fill)


def _push_pull_fill(d: torch.Tensor, v: torch.Tensor, levels: int = 10) -> torch.Tensor:
    """Normalised-convolution pyramid fill of invalid pixels (smooth, colour-agnostic)."""
    pyr = []
    cd, cv = d * v, v
    for _ in range(levels):
        pyr.append((cd, cv))
        if min(cd.shape[-2:]) < 4:
            break
        cd = F.avg_pool2d(cd, 2, ceil_mode=True)
        cv = F.avg_pool2d(cv, 2, ceil_mode=True)
    est = cd / cv.clamp_min(1e-6)
    for cd, cv in reversed(pyr[:-1]):
        up = F.interpolate(est, size=cd.shape[-2:], mode="bilinear", align_corners=False)
        wgt = cv.clamp(0, 1)
        est = (cd / cv.clamp_min(1e-6)) * wgt + up * (1 - wgt)
    return torch.where(v > 0.5, d, est)


def complete_disparity(disp: torch.Tensor, valid: torch.Tensor, guide: torch.Tensor):
    """
    Fill invalid low-res disparities: pixels that are occluded in the target view get
    background disparity (so they never ride along with a foreground edge), other
    unmatched pixels (textureless, specular) get a smooth push-pull estimate; then an
    edge-aware guided filter restricted to the unreliable pixels.
    """
    d_np = disp[0, 0].cpu().numpy()
    v_np = valid[0, 0].cpu().numpy() > 0.5
    bg = torch.from_numpy(_row_background_fill(d_np, v_np))[None, None]
    pp = disp  # dense estimate (RAFT, or push-pull filled SGBM)
    # occlusion test for the right view: a pixel is hidden if some pixel to its right
    # lands on or left of it with larger disparity:  max_k (d(x+k) - k) > d(x) + 0.5
    h, w = d_np.shape
    ppn = pp[0, 0]
    occ = torch.zeros_like(ppn, dtype=torch.bool)
    run = torch.full_like(ppn, -1e9)
    kmax = int(max(4, min(w - 1, math.ceil(float(ppn.max()) - float(ppn.min()) + 2))))
    for k in range(1, kmax + 1):
        shifted = torch.full_like(ppn, -1e9)
        shifted[:, :-k] = ppn[:, k:] - k
        run = torch.maximum(run, shifted)
    occ = run > ppn + 0.5
    filled = torch.where(occ[None, None], bg, pp)
    filled = torch.where(valid > 0.5, disp, filled)
    g = luma(guide)
    sm = guided_filter(g, filled, r=4, eps=1e-3)
    keep = erode(valid, 1)
    out = torch.where(keep > 0.5, filled, torch.minimum(sm, filled.max()))
    return out, occ[None, None].float()


# ----------------------------------------------------------------------------
# full resolution disparity + forward splatting
# ----------------------------------------------------------------------------
def upsample_disparity(d_lr: torch.Tensor, guide_hr: torch.Tensor, scale_x: float):
    """
    Edge-snapped upsampling: guided filter with the full-res source, then near
    depth discontinuities every pixel is snapped to either the local foreground or
    the local background value, so no pixel gets an in-between disparity (the cause
    of stretched / smeared edges in naive DIBR).
    """
    H, W = guide_hr.shape[-2:]
    h, w = d_lr.shape[-2:]
    up = F.interpolate(d_lr, size=(H, W), mode="bilinear", align_corners=False) * scale_x
    s = max(H / h, W / w)
    r = max(2, int(round(s * 1.5)))
    g = luma(guide_hr)
    gf = guided_filter(g, up, r=r, eps=2e-4)
    # discontinuity band from the low-res map (> 0.75 lr px jump between neighbours)
    gx = (d_lr[..., :, 1:] - d_lr[..., :, :-1]).abs()
    gy = (d_lr[..., 1:, :] - d_lr[..., :-1, :]).abs()
    e = torch.zeros_like(d_lr)
    e[..., :, 1:] = torch.maximum(e[..., :, 1:], gx)
    e[..., :, :-1] = torch.maximum(e[..., :, :-1], gx)
    e[..., 1:, :] = torch.maximum(e[..., 1:, :], gy)
    e[..., :-1, :] = torch.maximum(e[..., :-1, :], gy)
    band = dilate((e > 0.75).float(), 1)
    band = F.interpolate(band, size=(H, W), mode="nearest")
    k = 2 * r + 1
    lmax = F.max_pool2d(up, k, stride=1, padding=r)
    lmin = -F.max_pool2d(-up, k, stride=1, padding=r)
    snapped = torch.where((gf - lmin).abs() <= (lmax - gf).abs(), lmin, lmax)
    return torch.where(band > 0.5, snapped, gf)


def splat_disparity(d_src: torch.Tensor):
    """
    Z-buffered forward splat of source-grid disparity to the target (right) grid.
    Each source pixel covers [x-d-0.5, x-d+0.5] -> both neighbouring integer pixels.
    The largest disparity (nearest surface) wins.  Returns target-grid disparity and
    a hit mask (0 = disoccluded).
    """
    B, _, H, W = d_src.shape
    assert B == 1
    d = d_src[0, 0]
    xs = torch.arange(W, device=d.device, dtype=d.dtype)[None, :].expand(H, W)
    xt = xs - d
    base = torch.floor(xt)
    rows = torch.arange(H, device=d.device)[:, None].expand(H, W) * W
    out = torch.full((H * W,), -1e9, device=d.device, dtype=d.dtype)
    for off in (0, 1):
        xi = base.long() + off
        ok = (xi >= 0) & (xi < W) & ((xt - xi.to(d.dtype)).abs() < 1.0)
        idx = (rows + xi.clamp(0, W - 1))[ok]
        out.scatter_reduce_(0, idx, d[ok], reduce="amax", include_self=True)
    out = out.view(H, W)
    hit = out > -1e8
    # hole disparity = background side (min of nearest hits left/right), per row
    dn = out.cpu().numpy()
    filled = _row_background_fill(np.where(hit.cpu().numpy(), dn, 0), hit.cpu().numpy())
    filled = torch.from_numpy(filled).to(d.device, d.dtype)
    return filled[None, None], hit[None, None].float()


def backward_sample(img: torch.Tensor, d_tgt: torch.Tensor, mode: str = "bicubic"):
    """target(x) = source(x + d)."""
    B, C, H, W = img.shape
    ys, xs = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=img.dtype),
        torch.arange(W, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    xsrc = xs[None] + d_tgt[:, 0]
    gx = (xsrc + 0.5) / W * 2 - 1
    gy = (ys[None].expand_as(gx) + 0.5) / H * 2 - 1
    grid = torch.stack([gx, gy], -1)
    out = F.grid_sample(img, grid, mode=mode, padding_mode="border", align_corners=False)
    inframe = ((xsrc >= -0.5) & (xsrc <= W - 0.5)).float()[:, None]
    return out.clamp(0, 1), inframe


# ----------------------------------------------------------------------------
# appearance transfer / blending
# ----------------------------------------------------------------------------
def fit_color(gen: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor, iters: int = 3):
    """Per-channel robust affine map gen -> ref on reliable pixels (removes VAE colour drift)."""
    out = gen.clone()
    m = mask[0, 0] > 0.5
    if m.sum() < 64:
        return out
    for c in range(3):
        x = gen[0, c][m].double()
        y = ref[0, c][m].double()
        keep = torch.ones_like(x, dtype=torch.bool)
        a, b = 1.0, 0.0
        for _ in range(iters):
            xk, yk = x[keep], y[keep]
            vx = xk.var()
            if vx < 1e-8:
                break
            a = float(((xk - xk.mean()) * (yk - yk.mean())).mean() / vx)
            a = min(max(a, 0.7), 1.3)
            b = float(yk.mean() - a * xk.mean())
            res = (a * x + b - y).abs()
            keep = res < max(float(res.quantile(0.8)) * 2.0, 2.0 / 255)
        out[0, c] = (a * gen[0, c] + b)
    return out.clamp(0, 1)


def local_structure_similarity(a: torch.Tensor, b: torch.Tensor, r: int = 3, c2: float = 2e-4):
    """SSIM structure term (correlation) on luma; ~1 where both share local structure."""
    a, b = luma(a), luma(b)
    ma, mb = box(a, r), box(b, r)
    va = (box(a * a, r) - ma * ma).clamp_min(0)
    vb = (box(b * b, r) - mb * mb).clamp_min(0)
    cov = box(a * b, r) - ma * mb
    s = (cov + c2 / 2) / (torch.sqrt(va * vb) + c2 / 2)
    activity = torch.sqrt(torch.maximum(va, vb))
    return s, activity


def laplacian_blend(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor, levels: int = 6):
    """Multi-band blend: m=1 -> b, m=0 -> a.  Seam-free compositing."""
    levels = max(1, min(levels, int(math.log2(max(2, min(a.shape[-2:])))) - 2))
    ga, gb, gm = [a], [b], [m]
    for _ in range(levels):
        ga.append(F.avg_pool2d(ga[-1], 2, ceil_mode=True))
        gb.append(F.avg_pool2d(gb[-1], 2, ceil_mode=True))
        gm.append(F.avg_pool2d(gm[-1], 2, ceil_mode=True))

    def up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    out = ga[-1] * (1 - gm[-1]) + gb[-1] * gm[-1]
    for i in range(levels - 1, -1, -1):
        la = ga[i] - up(ga[i + 1], ga[i])
        lb = gb[i] - up(gb[i + 1], gb[i])
        out = up(out, ga[i]) + la * (1 - gm[i]) + lb * gm[i]
    return out.clamp(0, 1)


def hf_energy(x: torch.Tensor, mask: torch.Tensor, sigma: float = 1.0):
    y = luma(x)
    hp = y - gaussian_blur(y, sigma)
    m = mask > 0.5
    if m.sum() < 64:
        return None
    return float(hp[m].abs().mean())


def match_sharpness(warped: torch.Tensor, source: torch.Tensor, valid: torch.Tensor):
    """Compensate the resampling softness of the warp so both eyes have equal acutance."""
    e_src = hf_energy(source, torch.ones_like(source[:, :1]))
    e_w = hf_energy(warped, valid)
    if e_src is None or e_w is None or e_w <= 1e-6:
        return warped, 1.0
    ratio = e_src / e_w
    amount = float(np.clip(ratio - 1.0, 0.0, 0.6))
    if amount < 0.02:
        return warped, ratio
    blur = gaussian_blur(warped, 0.8)
    return (warped + amount * (warped - blur)).clamp(0, 1), ratio


def noise_sigma(x: torch.Tensor, mask=None):
    """Robust per-image noise estimate (MAD of fine-scale Laplacian in flat areas)."""
    y = luma(x)
    k = torch.tensor([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3)
    r = F.conv2d(F.pad(y, (1, 1, 1, 1), mode="replicate"), k)
    act = box((y - box(y, 2)).abs(), 2)
    sel = act < act.quantile(0.3) if act.numel() < 16_000_000 else act < act.mean()
    if mask is not None:
        sel = sel & (mask > 0.5)
    if sel.sum() < 64:
        return 0.0
    v = r[sel].abs()
    return float(v.median() * math.sqrt(math.pi / 2) / 6.0)


# ----------------------------------------------------------------------------
# main entry: lift a low-res generated view to the full-resolution target eye
# ----------------------------------------------------------------------------
@torch.no_grad()
def lift_to_full_resolution(
    source_hr: torch.Tensor,  # [1,3,H,W] full-res source (canonical left)
    source_lr: torch.Tensor,  # [1,3,h,w] exact image fed to the generator
    gen_lr: torch.Tensor,  # [1,3,h,w] generated right view
    gen_hr_hint: torch.Tensor = None,  # optional [1,3,*,*] e.g. ESRGAN-upscaled generated view
    sensitivity: float = 1.0,
    sharpen_match: bool = True,
    hole_dilate_px: int = 2,
    device: torch.device = torch.device("cpu"),
):
    H, W = source_hr.shape[-2:]
    h, w = source_lr.shape[-2:]
    sx = W / float(w)
    dbg = {}

    source_hr = source_hr.to(device)
    source_lr = source_lr.to(device)
    gen_lr = gen_lr.to(device)

    # 1) binocular disparity from the generated pair (captures what the generator
    #    actually rendered, incl. mirror / glass virtual depths)
    g0 = fit_color(gen_lr, source_lr, torch.ones_like(gen_lr[:, :1]), iters=2)
    d_lr, v_lr = estimate_disparity_lr(source_lr, g0, device=device)
    d_lr, v_lr = d_lr.to(device), v_lr.to(device)
    d_lr_full, occ_lr = complete_disparity(d_lr, v_lr, source_lr)
    dbg["disp_lr"] = d_lr_full
    dbg["valid_lr"] = v_lr

    # 2) full-res source-grid disparity with snapped discontinuities
    d_hr = upsample_disparity(d_lr_full, source_hr, sx)
    # 3) z-buffered splat -> target-grid disparity; backward bicubic sample of the
    #    untouched full-res source pixels
    d_tgt, hit = splat_disparity(d_hr)
    warped, inframe = backward_sample(source_hr, d_tgt)
    # confidence carried from source pixels that were binocularly verified
    v_hr_src = F.interpolate(erode(v_lr, 1), size=(H, W), mode="nearest")
    v_tgt, _ = backward_sample(v_hr_src.repeat(1, 3, 1, 1), d_tgt, mode="nearest")
    v_tgt = v_tgt[:, :1]
    holes = (1 - hit) * 1.0
    holes = torch.maximum(holes, 1 - inframe)
    holes = dilate(holes, hole_dilate_px)
    valid_hr = (1 - holes) * v_tgt

    if sharpen_match:
        warped, ratio = match_sharpness(warped, source_hr, valid_hr)
        dbg["sharpness_ratio"] = ratio

    # 4) compare the warped full-res view with the generated view at generator scale
    warped_lr = resize(warped, h, w, "area")
    valid_lr_t = resize(valid_hr, h, w, "area")
    v_lr_t_src = resize(v_tgt, h, w, "area")  # RAFT-validated source pixels, seen from target
    g = fit_color(gen_lr, warped_lr, (valid_lr_t > 0.95).float())
    blur_s = 1.0
    a = gaussian_blur(warped_lr, blur_s)
    b = gaussian_blur(g, blur_s)
    struct, activity = local_structure_similarity(a, b, r=3)
    diff = gaussian_blur((a - b).abs().mean(1, keepdim=True), 1.5)
    sens = max(sensitivity, 1e-3)
    # structural disagreement only counts where there is structure to compare
    struct_bad = smoothstep(0.55 - struct, 0.0, 0.35) * smoothstep(activity, 0.015 / sens, 0.05 / sens)
    color_bad = smoothstep(diff, 0.06 / sens, 0.16 / sens)
    replace_lr = torch.maximum(struct_bad * smoothstep(diff, 0.03 / sens, 0.08 / sens), color_bad)
    # morphological opening: drop isolated specks (generator noise) that would only
    # punch soft patches into the sharp warped view
    replace_lr = dilate(erode(replace_lr, 1), 1)
    # Only trust a disagreeing generator where its pixels are binocularly explainable
    # (forward/backward-consistent, epipolar-aligned correspondence).  Content with no
    # valid horizontal match is a generator hallucination, not a view-dependent effect:
    # keep the original pixels there.  Disocclusions are handled separately below.
    explainable = erode((valid_lr_t > 0.5).float() * (v_lr_t_src > 0.5).float(), 1)
    dbg["hallucination"] = F.interpolate(((replace_lr > 0.3).float() * (1 - explainable)),
                                         size=(H, W), mode="nearest")
    replace_lr = replace_lr * explainable
    replace_lr = gaussian_blur(dilate(replace_lr, 1), 1.0).clamp(0, 1)
    # mild, smooth appearance differences (moving highlights, reflections' tint,
    # transparency) -> transfer as a low-frequency residual on top of full-res detail
    resid = gaussian_blur(g - warped_lr, 2.0)
    tint = smoothstep(gaussian_blur((g - warped_lr).abs().mean(1, keepdim=True), 2.0), 0.02 / sens, 0.06 / sens)
    tint = tint * gaussian_blur(explainable, 2.0)

    up = lambda t: F.interpolate(t, size=(H, W), mode="bicubic", align_corners=False)
    replace_hr = up(replace_lr).clamp(0, 1)
    replace_hr = gaussian_blur(replace_hr, max(1.0, sx * 0.75)).clamp(0, 1)
    replace_hr = torch.maximum(replace_hr, holes)
    tinted = (warped + up(resid) * up(tint).clamp(0, 1) * (1 - holes)).clamp(0, 1)

    # 5) generated view at full resolution (optionally super-resolved upstream)
    if gen_hr_hint is not None:
        g_hr = resize(gen_hr_hint.to(device), H, W, "bicubic")
        g_hr = fit_color(g_hr, tinted, (replace_hr < 0.05).float())
    else:
        g_hr = up(g).clamp(0, 1)

    out = laplacian_blend(tinted, g_hr, replace_hr)

    # refine mask: 1.0 where content is purely generated (holes / out of frame),
    # 0.65 where view-dependent re-synthesis replaced the warp, feathered
    refine = torch.maximum(holes, replace_hr * 0.65)
    refine = gaussian_blur(dilate(refine, max(2, int(sx))), max(1.5, sx * 0.5)).clamp(0, 1)
    dbg.update(
        disp_tgt=d_tgt, holes=holes, replace=replace_hr, tint=up(tint), warped=warped,
        gen_up=g_hr, disp_src=d_hr,
    )
    return out, refine, dbg


def disparity_stats(d_lr: torch.Tensor, valid: torch.Tensor, width: int):
    m = valid > 0.5
    if m.sum() < 32:
        return None
    v = d_lr[m]
    return dict(
        p02=float(v.quantile(0.02)) / width,
        p50=float(v.quantile(0.5)) / width,
        p98=float(v.quantile(0.98)) / width,
    )


def colorize_disparity(d: torch.Tensor) -> torch.Tensor:
    lo, hi = float(d.quantile(0.01)), float(d.quantile(0.99))
    t = ((d - lo) / max(hi - lo, 1e-6)).clamp(0, 1)
    # simple perceptual-ish ramp (dark blue -> yellow)
    r = (1.5 * t - 0.2).clamp(0, 1)
    g = (1.2 * t).clamp(0, 1) * 0.9
    b = (1.0 - 1.4 * t).clamp(0, 1) * 0.8 + 0.2 * t
    return torch.cat([r, g, b], 1)


def anaglyph(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Dubois-style optimised red/cyan anaglyph for quick previews."""
    Lm = torch.tensor([[0.456, 0.500, 0.176], [-0.040, -0.038, -0.016], [-0.015, -0.021, -0.005]])
    Rm = torch.tensor([[-0.043, -0.088, -0.002], [0.378, 0.734, -0.018], [-0.072, -0.113, 1.226]])
    Lm, Rm = Lm.to(left), Rm.to(left)
    out = torch.einsum("ij,bjhw->bihw", Lm, left) + torch.einsum("ij,bjhw->bihw", Rm, right)
    return out.clamp(0, 1)
