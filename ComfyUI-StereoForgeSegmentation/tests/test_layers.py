"""CPU tests for the object-layer engine (no models needed: simple matte/fill stand-ins).

    python tests/test_layers.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sfs import core, layers as L  # noqa: E402


def _tex(H, W, seed):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(H, W, 3, generator=g)
    return torch.stack([core.gaussian_blur(t[..., c], 1.0) for c in range(3)], -1)


def _scene(H=240, W=360):
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    disc = ((yy - 120) ** 2 + (xx - 180) ** 2) < 60 ** 2
    fg = torch.tensor([0.9, 0.2, 0.2]).expand(H, W, 3)
    bg = _tex(H, W, 1) * 0.5
    img = torch.where(disc.unsqueeze(-1), fg, bg)
    # depth edge deliberately 3 px *outside* the true silhouette (typical monocular offset)
    disp = (((yy - 120) ** 2 + (xx - 180) ** 2) < 63 ** 2).float()
    return img, disp, disc


def _fill_const(im, fill, hide):
    """Stand-in inpainter: paints the fill region with the mean of the visible background."""
    vis = ~(fill | hide)
    mean = im[vis].mean(0)
    return torch.where(fill.unsqueeze(-1), mean.expand_as(im), im)


def _params(W, px):
    return core.StereoParams(budget_pct=px / W * 100, convergence=0.0, stereo_window=False)


def test_pushpull_is_smooth_and_exact_on_known():
    v = torch.linspace(0, 10, 100).expand(50, 100).clone()
    known = torch.ones_like(v, dtype=torch.bool)
    known[:, 40:60] = False
    out = L.pushpull_fill(v, known)
    assert torch.equal(out[known], v[known])
    assert (out[:, 40:60] - v[:, 40:60]).abs().max() < 1.5


def test_object_moves_with_its_matte_not_the_depth_edge():
    img, disp, disc = _scene()
    H, W = disp.shape
    st = L.build_stack(img, disp, [disc.float()], _params(W, 20), None, _fill_const, log=lambda *a: None)
    eye, inv_vis, auto = L.render_stack(st)
    # right eye: disc (disparity 20 px in front) moves 20 px left; background stays
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    moved = ((yy - 120) ** 2 + (xx - 160) ** 2) < 55 ** 2
    assert (eye[moved] - torch.tensor([0.9, 0.2, 0.2])).abs().max() < 0.05
    # no red ghost where the disc's rim used to be (the 3 px depth-edge offset must not drag background)
    ring = (((yy - 120) ** 2 + (xx - 180) ** 2) > 61 ** 2) & (((yy - 120) ** 2 + (xx - 180) ** 2) < 70 ** 2) & (xx > 200)
    assert (eye[ring][:, 0] - eye[ring][:, 1]).max() < 0.35
    # the revealed crescent is invented (plate) content and is reported as such
    assert bool(inv_vis[120, 223:238].all())
    # nothing but the frame-edge strip needs auto-filling
    assert not bool(auto[:, 30:W - 30].any())


def test_invented_only_where_revealed():
    img, disp, disc = _scene()
    W = disp.shape[1]
    st = L.build_stack(img, disp, [disc.float()], _params(W, 20), None, _fill_const, log=lambda *a: None)
    inv = st.layers[0].invented
    H = disp.shape[0]
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    r = ((yy - 120) ** 2 + (xx - 180) ** 2).float().sqrt()
    # right eye reveals background on the right side of the object only; elsewhere
    # only the 1-2 px soft rim is reconstructed (needed to clean edge colours)
    assert bool(inv[120, 225:238].all())
    assert not bool((inv & (xx < 150) & (r < 57)).any())


def test_edit_only_touches_invented_pixels():
    img, disp, disc = _scene()
    W = disp.shape[1]
    st = L.build_stack(img, disp, [disc.float()], _params(W, 20), None, _fill_const, log=lambda *a: None)
    before = st.layers[0].color.clone()
    edited = torch.zeros_like(before)
    st2 = st.clone()
    Ly = st2.layers[0]
    Ly.color = torch.where(Ly.invented.unsqueeze(-1), edited, Ly.color)
    changed = (Ly.color != before).any(-1)
    assert bool((changed <= Ly.invented).all())
    assert torch.equal(st.layers[0].color, before)  # clone is independent


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
