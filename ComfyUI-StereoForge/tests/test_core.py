"""Geometry tests for the StereoForge core (CPU, no ComfyUI needed).

    python -m pytest tests            # or: python tests/test_core.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stereoforge import core  # noqa: E402


def _texture(H, W, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(H, W, 3, generator=g)
    return torch.stack([core.gaussian_blur(t[..., c], 1.0) for c in range(3)], -1)


def _disc_scene(H=300, W=480, r=60, blur=3.0):
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    m = ((yy - H // 2) ** 2 + (xx - W // 2) ** 2) < r ** 2
    img = torch.where(m.unsqueeze(-1), _texture(H, W, 1), _texture(H, W, 2))
    disp = core.gaussian_blur(m.float(), blur)
    return img, disp, m


def test_flat_scene_is_identity():
    img = _texture(128, 200)
    disp = torch.full((128, 200), 0.5)
    r = core.render_opposite_eye(img, disp, core.StereoParams(convergence=0.5, stereo_window=False))
    assert not r.hole.any()
    assert torch.equal(r.view, img)


def test_integer_shift_is_exact():
    img = _texture(64, 160)
    disp = torch.ones(64, 160)
    # budget chosen so that the whole plane shifts by exactly 8 px
    p = core.StereoParams(budget_pct=8 / 160 * 100, convergence=0.0, stereo_window=False)
    r = core.render_opposite_eye(img, disp, p)
    assert torch.allclose(r.view[:, : 160 - 8], img[:, 8:], atol=1e-6)
    assert r.hole[:, 160 - 8:].all() and not r.hole[:, : 160 - 8].any()


def test_disocclusion_is_one_clean_gap_on_the_correct_side():
    img, disp, m = _disc_scene(blur=4.0)
    H, W = disp.shape
    budget = 30
    p = core.StereoParams(budget_pct=budget / W * 100, convergence=0.0, stereo_window=False, soft_band=0)
    r = core.render_opposite_eye(img, disp, p)
    row = r.hole[H // 2].nonzero().flatten()
    # right eye: the disc moves left, the gap opens at its right edge (x ~ 300-30 .. 300)
    assert len(row) > 0
    assert row.max() - row.min() + 1 == len(row), "hole must be contiguous (no flying pixels)"
    assert abs(len(row) - budget) <= 3
    assert row.min() >= W // 2 + 60 - budget - 3 and row.max() <= W // 2 + 60 + 3


def test_left_eye_from_right_source_mirrors():
    img, disp, m = _disc_scene()
    W = disp.shape[1]
    p = core.StereoParams(budget_pct=20 / W * 100, convergence=0.0, stereo_window=False, source_eye="right",
                          soft_band=0)
    r = core.render_opposite_eye(img, disp, p)
    row = r.hole[disp.shape[0] // 2].nonzero().flatten()
    assert row.max() < W // 2  # gap opens on the left of the disc


def test_slopes_do_not_tear():
    W = 400
    disp = torch.linspace(0, 1, W).expand(100, W).clone()
    img = _texture(100, W)
    r = core.render_opposite_eye(img, disp, core.StereoParams(budget_pct=5, convergence=0.5, stereo_window=False))
    interior = r.hole[:, 12:-12]
    assert not interior.any()


def test_background_pixels_unchanged_at_zero_parallax():
    img, disp, m = _disc_scene()
    r = core.render_opposite_eye(img, disp, core.StereoParams(budget_pct=5, convergence=0.0, stereo_window=False))
    H, W = disp.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    far = ((yy - H // 2) ** 2 + (xx - W // 2) ** 2) > 110 ** 2
    assert (r.view[far] - img[far]).abs().max() < 1e-5


def test_stereo_window_pushes_border_content_behind_screen():
    disp = torch.zeros(100, 200)
    disp[:, :10] = 0.8  # near object touching the left frame edge
    D, z0 = core.to_pixel_disparity(disp, core.StereoParams(convergence=0.2, stereo_window=True))
    assert z0 >= 0.8 - 1e-6 and D[:, :10].max() <= 1e-6


def test_soft_fringe_moves_with_foreground():
    # a white disc with a semi-transparent halo that the depth map does not cover
    H, W = 200, 320
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    rr = ((yy - 100) ** 2 + (xx - 160) ** 2).float().sqrt()
    alpha = (1 - (rr - 50) / 6).clamp(0, 1)          # 6 px soft edge outside r=50
    bg = torch.tensor([0.1, 0.3, 0.1]).expand(H, W, 3)
    img = alpha.unsqueeze(-1) * 1.0 + (1 - alpha.unsqueeze(-1)) * bg
    disp = (rr < 50).float()
    p = core.StereoParams(budget_pct=20 / W * 100, convergence=0.0, stereo_window=False)
    r = core.render_opposite_eye(img, disp, p)
    final = core.composite(r.view, None, r.alpha, r.fringe_color, r.fringe_alpha)
    # at the source silhouette + halo location (now background, right of the moved disc)
    # there must be no bright ghost ring left behind outside the regenerated region
    ghost_zone = (rr > 51) & (rr < 56) & (xx > 160) & (r.inpaint_mask < 0.5)
    if ghost_zone.any():
        assert final[ghost_zone].mean() < 0.35


def test_narrow_gaps_skip_generative_fill():
    img, disp, m = _disc_scene(blur=2.0)
    W = disp.shape[1]
    # 4 px of parallax -> 4 px gaps: below the threshold, no diffusion needed
    p = core.StereoParams(budget_pct=4 / W * 100, convergence=0.0, stereo_window=False, min_fill_px=8)
    r = core.render_opposite_eye(img, disp, p)
    assert r.hole.any() and not (r.inpaint_mask > 0.5).any()
    # 30 px gaps: inpainted
    p = core.StereoParams(budget_pct=30 / W * 100, convergence=0.0, stereo_window=False, min_fill_px=8)
    r = core.render_opposite_eye(img, disp, p)
    assert ((r.inpaint_mask > 0.5) & r.hole).sum() >= 0.9 * r.hole.sum()


def test_crop_planner_covers_every_gap_pixel():
    g = torch.Generator().manual_seed(3)
    H, W = 1300, 2100
    mask = torch.zeros(H, W, dtype=torch.bool)
    for _ in range(40):  # thin vertical slivers like disocclusions
        y, x = int(torch.randint(0, H - 200, (1,), generator=g)), int(torch.randint(0, W - 30, (1,), generator=g))
        mask[y:y + 200, x:x + 12] = True
    mask[:, -40:] = True  # frame-edge strip
    tile, ctx = 1024, 64
    max_h, max_w = min(tile, H // 16 * 16), min(tile, W // 16 * 16)
    ch, cw = max_h - 2 * ctx, max_w - 2 * ctx
    rem, area = mask.clone(), 0
    for y in range(0, H, ch):
        for x in range(0, W, cw):
            cell = (y, min(H, y + ch), x, min(W, x + cw))
            owned = torch.zeros_like(rem)
            owned[cell[0]:cell[1], cell[2]:cell[3]] = rem[cell[0]:cell[1], cell[2]:cell[3]]
            c = core.plan_crop(owned, cell, H, W, max_h, max_w, ctx, 384)
            if c is None:
                continue
            y0, y1, x0, x1 = c
            assert 0 <= y0 < y1 <= H and 0 <= x0 < x1 <= W
            assert (y1 - y0) % 16 == 0 and (x1 - x0) % 16 == 0
            assert y1 - y0 <= max_h and x1 - x0 <= max_w
            assert owned[y0:y1, x0:x1].sum() == owned.sum()
            rem[y0:y1, x0:x1] &= ~owned[y0:y1, x0:x1]
            area += (y1 - y0) * (x1 - x0)
    assert not rem.any()
    # the previous fixed grid (1024 px tiles, 256 overlap) processed 3 x 2 full tiles here
    assert area < 0.8 * 6 * max_h * max_w


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
