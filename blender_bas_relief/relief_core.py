"""
Pure-numpy bas-relief solver (no Blender dependency, unit-testable on its own).

Pipeline (gradient-domain relief, after Weyrich et al. 2007 "Digital Bas-Relief
from 3D Scenes" and Kerber et al. 2009 "Feature Sensitive Bas Relief
Generation"):

 1. Per-pixel surface slopes are taken from the (smooth, interpolated) surface
    normals seen by the camera - i.e. a normal map - rather than from the
    depth map, so smooth shading and fine surface detail survive.
 2. Depth discontinuities (silhouettes, one object in front of another) are
    detected from jumps in the depth buffer and their gradients are removed,
    so nothing "extrudes" straight towards the viewer.
 3. Remaining slopes are compressed with a logarithmic curve: steep slopes are
    flattened a lot, gentle slopes barely at all. This is what keeps detail
    readable while the total depth shrinks to a few millimetres.
 4. Optional unsharp-masking in the gradient domain boosts fine detail.
 5. The gradient field is integrated back into a height field with an exact
    least-squares Poisson solve (DCT, Neumann boundaries).
 6. A small, soft amount of compressed depth ("layering") is blended back in so
    objects still separate from the background and from each other.
"""

import numpy as np


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------

def gaussian_blur(img, sigma):
    """Gaussian blur with reflected borders, done in the frequency domain so
    that large radii cost the same as small ones."""
    if sigma <= 1e-6:
        return img.copy()
    h, w = img.shape
    pad = int(min(max(h, w), np.ceil(3.0 * sigma)))
    p = np.pad(img, pad, mode="reflect") if pad > 0 else img
    ph, pw = p.shape
    fy = np.fft.fftfreq(ph)[:, None]
    fx = np.fft.rfftfreq(pw)[None, :]
    kern = np.exp(-2.0 * (np.pi * sigma) ** 2 * (fx * fx + fy * fy))
    out = np.fft.irfft2(np.fft.rfft2(p) * kern, s=p.shape)
    return out[pad:pad + h, pad:pad + w]


def masked_blur(img, weight, sigma, eps=1e-8):
    """Normalized convolution: blur only using samples where weight > 0."""
    num = gaussian_blur(img * weight, sigma)
    den = gaussian_blur(weight.astype(np.float64), sigma)
    return num / np.maximum(den, eps)


# --------------------------------------------------------------------------
# Poisson integration
# --------------------------------------------------------------------------

def poisson_integrate(gx, gy):
    """Least-squares height field h with h[:,j+1]-h[:,j] ~ gx[:,j] and
    h[i+1,:]-h[i,:] ~ gy[i,:].  gx[:, -1] and gy[-1, :] are ignored.
    Solved exactly with a DCT (via a mirrored FFT) - Neumann boundaries."""
    H, W = gx.shape
    gx = gx.copy()
    gy = gy.copy()
    gx[:, -1] = 0.0
    gy[-1, :] = 0.0
    div = gx.copy()
    div[:, 1:] -= gx[:, :-1]
    div += gy
    div[1:, :] -= gy[:-1, :]

    ext = np.concatenate([div, div[:, ::-1]], axis=1)
    ext = np.concatenate([ext, ext[::-1, :]], axis=0)
    F = np.fft.rfft2(ext)
    ky = np.arange(2 * H)[:, None]
    kx = np.arange(F.shape[1])[None, :]
    denom = (2.0 * np.cos(np.pi * kx / W) - 2.0) + (2.0 * np.cos(np.pi * ky / H) - 2.0)
    denom[0, 0] = 1.0
    F /= denom
    F[0, 0] = 0.0
    h = np.fft.irfft2(F, s=ext.shape)[:H, :W]
    return h


# --------------------------------------------------------------------------
# Relief
# --------------------------------------------------------------------------

DEFAULTS = dict(
    compression=4.0,         # log compression strength of slopes (0 = none)
    detail_boost=1.5,        # >1 exaggerates fine detail (unsharp in gradient domain)
    detail_radius=6.0,       # px, what counts as "fine" detail
    layering=0.25,           # 0 = pure normal-based relief, 1 = pure depth map
    background_lift=0.25,    # how far the farthest object still stands off the background (in layering)
    edge_softness=2.0,       # px, softens silhouette steps (also better for printing)
    edge_threshold=3.0,      # slope (dz/dx) above which a depth jump is a discontinuity
    smoothing=0.0,           # px, final gaussian smoothing
    use_normals=True,        # slopes from smooth normals (True) or depth differences (False)
    flatten_scale=0.08,      # fraction of image size; removes low-frequency drift/halo
)


def _compress(s, alpha):
    if alpha <= 1e-6:
        return s
    return np.sign(s) * np.log1p(alpha * np.abs(s)) / alpha


def compute_relief(depth, normals, mask, pix, params=None):
    """
    depth   : (H,W) distance in front of the camera (positive), row 0 = top
    normals : (H,W,3) camera-space unit normals facing the camera (+z towards viewer, +y up)
    mask    : (H,W) bool, True where geometry was hit
    pix     : (H,W) world-space size of one pixel at that depth
    returns : (H,W) float64 height in [0,1]  (1 = nearest to viewer)
    """
    p = dict(DEFAULTS)
    if params:
        p.update(params)

    H, W = depth.shape
    mask = mask.astype(bool)
    out = np.zeros((H, W))
    if not mask.any():
        return out

    pix = np.where(mask, pix, 0.0)
    pix_ref = float(np.median(pix[mask]))
    if pix_ref <= 0:
        pix_ref = 1.0
    z = np.where(mask, -depth, 0.0)          # height towards the viewer, world units
    max_slope = max(p["edge_threshold"], 1e-3)

    # ---- per-pixel slopes (height per lateral unit) from normals ----
    if p["use_normals"]:
        nz = np.maximum(normals[..., 2], 1e-4)
        sx = np.clip(-normals[..., 0] / nz, -max_slope, max_slope)
        sy = np.clip(normals[..., 1] / nz, -max_slope, max_slope)   # image rows go down: negate dz/dy
        sx = np.where(mask, sx, 0.0)
        sy = np.where(mask, sy, 0.0)

    # ---- staggered gradients + discontinuity detection ----
    def axis_grad(axis):
        if axis == 1:
            a, b = (slice(None), slice(0, -1)), (slice(None), slice(1, None))
        else:
            a, b = (slice(0, -1), slice(None)), (slice(1, None), slice(None))
        both = mask[a] & mask[b]
        pav = 0.5 * (pix[a] + pix[b])
        dz = z[b] - z[a]
        edge = np.abs(dz) > max_slope * np.maximum(pav, 1e-12) * 1.5
        valid = both & ~edge
        if p["use_normals"]:
            s = sx if axis == 1 else sy
            slope = 0.5 * (s[a] + s[b])
        else:
            slope = np.clip(dz / np.maximum(pav, 1e-12), -max_slope, max_slope)
        # compress slope, then convert to "reference pixel" height steps
        g = _compress(slope, p["compression"]) * (pav / pix_ref)
        g = np.where(valid, g, 0.0)
        full = np.zeros((H, W))
        vfull = np.zeros((H, W))
        full[a] = g
        vfull[a] = valid
        return full, vfull

    gx, vx = axis_grad(1)
    gy, vy = axis_grad(0)

    # ---- detail boost (unsharp mask of gradients) ----
    boost = p["detail_boost"]
    if abs(boost - 1.0) > 1e-6 and p["detail_radius"] > 0:
        r = p["detail_radius"]
        lx = masked_blur(gx, vx, r)
        ly = masked_blur(gy, vy, r)
        gx = np.where(vx > 0, lx + boost * (gx - lx), 0.0)
        gy = np.where(vy > 0, ly + boost * (gy - ly), 0.0)

    # ---- integrate ----
    R = poisson_integrate(gx, gy)

    # ---- remove low-frequency drift so the background is flat and edges sit
    #      flush (layering below is what lifts objects off the background) ----
    bg = ~mask
    flat_sigma = max(2.0, p["flatten_scale"] * max(H, W))
    if bg.any():
        R = R - masked_blur(R, bg.astype(np.float64), flat_sigma, eps=1e-12)
        R[bg] = 0.0
    else:
        R = R - R.min()
    scale = float(np.percentile(np.abs(R[mask]), 99.8)) if mask.any() else 1.0
    Rn = R / scale if scale > 1e-12 else R * 0.0

    # ---- layering: compressed, softened depth order ----
    zf = z[mask]
    lo, hi = np.percentile(zf, 0.5), np.percentile(zf, 99.5)
    if hi - lo < 1e-12:
        Lraw = np.ones((H, W))
    else:
        Lraw = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    lift = p["background_lift"]
    L = np.where(mask, lift + (1.0 - lift) * Lraw, 0.0)
    if p["edge_softness"] > 0:
        L = gaussian_blur(L, p["edge_softness"])

    mix = float(np.clip(p["layering"], 0.0, 1.0))
    Hf = mix * L + (1.0 - mix) * Rn
    if p["smoothing"] > 0:
        Hf = gaussian_blur(Hf, p["smoothing"])

    Hf = Hf - min(float(Hf.min()), 0.0)
    m = float(Hf.max())
    if m > 1e-12:
        Hf /= m
    return Hf


def write_png16(path, height01):
    """Write a 16-bit grayscale PNG (row 0 = top). No dependencies."""
    import struct
    import zlib

    h, w = height01.shape
    data = np.clip(np.round(height01 * 65535.0), 0, 65535).astype(">u2")
    raw = b"".join(b"\x00" + data[i].tobytes() for i in range(h))

    def chunk(tag, payload):
        c = struct.pack(">I", len(payload)) + tag + payload
        return c + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)
