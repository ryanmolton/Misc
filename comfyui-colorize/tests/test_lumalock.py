"""Numerical tests for the LumaLock nodes (run outside ComfyUI, CPU only).

A real colour photo is the ground truth. We make it black-and-white, then
simulate what an edit model returns: lower resolution, shifted and zoomed by
a few pixels, brighter, with blurry colour. The merge must give back the
original's luminance exactly and colours close to the ground truth."""

import sys, os, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ComfyUI-LumaLock"))
from lumalock import nodes as N
from lumalock.color import rgb_to_lab, rgb_to_L, to_bchw, to_bhwc


def photo(name, size):
    import skimage.data as d
    img = torch.from_numpy(getattr(d, name)()).float().div(255).movedim(-1, 0).unsqueeze(0)
    h, w = img.shape[-2:]
    s = size / max(h, w)
    img = F.interpolate(img, size=(round(h * s), round(w * s)), mode="bicubic", align_corners=False).clamp(0, 1)
    return to_bhwc(img)  # (1,H,W,3)


def simulate_model(gt, shift=(3.0, -2.0), zoom=1.012, mp=1.0, gamma=0.85, chroma_blur=2.0):
    x = to_bchw(gt)
    h, w = x.shape[-2:]
    s = math.sqrt(mp * 1024 * 1024 / (h * w))
    nh, nw = round(h * s / 32) * 32, round(w * s / 32) * 32
    x = F.interpolate(x, size=(nh, nw), mode="area")
    theta = torch.tensor([[[1 / zoom, 0, 2 * shift[0] / nw], [0, 1 / zoom, 2 * shift[1] / nh]]])
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    lab = rgb_to_lab(x)
    k = int(chroma_blur * 3) * 2 + 1
    ab = F.avg_pool2d(lab[:, 1:], k, 1, k // 2, count_include_pad=False)
    L = 100 * (lab[:, :1] / 100) ** gamma  # model brightened the picture
    from lumalock.color import lab_to_rgb
    rgb = lab_to_rgb(torch.cat([L, ab], 1))
    return to_bhwc(torch.cat([rgb, torch.ones_like(rgb[:, :1])], 1))  # RGBA like Qwen-Image 2.1


def deltaE_ab(a, b):
    la, lb = rgb_to_lab(to_bchw(a)), rgb_to_lab(to_bchw(b))
    return (la[:, 1:] - lb[:, 1:]).pow(2).sum(1).sqrt()


def main():
    torch.manual_seed(0)
    # ComfyUI executes nodes under inference mode; the aligner must work there
    with torch.inference_mode(), torch.no_grad():
        g0 = photo("chelsea", 600)
        out = N.LumaLockMerge().run(N.LumaLockToGray().run(g0)[0], simulate_model(g0), "affine", 0.6, 2, 0.001, 1.0)[0]
        assert out.shape == g0.shape
    print("merge with alignment works under torch.inference_mode()")
    results = []
    for name, size in [("astronaut", 2400), ("coffee", 3000), ("chelsea", 1800)]:
        gt = photo(name, size)
        grey = N.LumaLockToGray().run(gt)[0]
        L_grey = rgb_to_L(to_bchw(grey))
        # ToGray keeps lightness
        assert (L_grey - rgb_to_L(to_bchw(gt))).abs().max() < 0.05, "ToGray changed lightness"
        sim = simulate_model(gt)
        rows = []
        for label, kw in [("plain resize (no align, no snap)", dict(alignment="off", edge_snap=0.0)),
                          ("align only", dict(alignment="affine", edge_snap=0.0)),
                          ("align + edge snap (default)", dict(alignment="affine", edge_snap=0.6))]:
            out = N.LumaLockMerge().run(grey, sim, snap_radius=2, snap_eps=0.001, saturation=1.0, **kw)[0]
            dL = (rgb_to_L(to_bchw(out)) - L_grey).abs()
            dE = deltaE_ab(out, gt)
            assert out.shape == gt.shape
            rows.append((label, float(dL.max()), float(dL.mean()), float(dE.mean()), float(dE.quantile(0.99) if dE.numel() < 16e6 else dE.flatten()[::7].quantile(0.99))))
        results.append((name, tuple(gt.shape[1:3]), rows))

        # the finishing nodes must never change lightness
        out = N.LumaLockMerge().run(grey, sim, "affine", 0.6, 2, 0.001, 1.0)[0]
        warm = to_bhwc(torch.cat([to_bchw(out)[:, :1] * 1.08, to_bchw(out)[:, 1:2], to_bchw(out)[:, 2:] * 0.85], 1))
        warm = N.LumaLockMerge().run(grey, warm, "off", 0.0, 3, 0.002, 1.0)[0]  # sepia-ish cast, exact L
        fin = N.LumaLockColorFinish().run(warm, 0.7, 1.0, 0.15, 70.0)[0]
        mood = N.LumaLockReferenceMood().run(fin, 0.5, "palette + saturation", reference=photo("coffee", 600))[0]
        mask = torch.zeros(1, 64, 64); mask[:, 20:40, 20:40] = 1
        hint = N.LumaLockRegionHint().run(mood, mask, "#b01c2a", 1.0, "colour", 2.0, True)[0]
        for tag, im in [("finish", fin), ("mood", mood), ("hint", hint)]:
            d = (rgb_to_L(to_bchw(im)) - L_grey).abs().max()
            assert d < 0.35, f"{tag} changed lightness by {d}"
        cast_before = rgb_to_lab(to_bchw(warm))[:, 1:].mean((2, 3))
        cast_after = rgb_to_lab(to_bchw(fin))[:, 1:].mean((2, 3))
        print(f"{name}: mean a/b cast before WB {cast_before.tolist()}, after {cast_after.tolist()}")
        if name == "astronaut":
            import PIL.Image as I
            outdir = os.environ.get("LUMALOCK_TEST_OUT")
            if outdir:
                for tag, im in [("gt", gt), ("grey", grey), ("sim_model_output", sim[..., :3]), ("merged", out), ("warm_cast", warm), ("finished", fin), ("mood", mood), ("hint", hint)]:
                    I.fromarray((im[0].numpy() * 255).round().astype(np.uint8)).save(os.path.join(outdir, f"{name}_{tag}.png"))

    print()
    print(f"{'image':10} {'size':12} {'method':34} {'max dL*':>8} {'mean dL*':>9} {'mean dE_ab':>10} {'p99 dE_ab':>9}")
    for name, sz, rows in results:
        for label, mx, mn, de, p99 in rows:
            print(f"{name:10} {str(sz):12} {label:34} {mx:8.4f} {mn:9.5f} {de:10.3f} {p99:9.3f}")
            assert mx < 0.35, "luminance not preserved"
        assert rows[2][3] < rows[0][3], "alignment + snapping should beat plain resize"

    # restoration: synthetic dust and scratches on a clean photo
    gt = N.LumaLockToGray().run(photo("astronaut", 1600))[0]
    x = to_bchw(gt).clone()
    H, W = x.shape[-2:]
    damage = torch.zeros(1, 1, H, W)
    for _ in range(60):
        cy, cx = np.random.randint(0, H), np.random.randint(0, W)
        damage[..., max(0, cy - 2):cy + 2, max(0, cx - 2):cx + 2] = 1
    for _ in range(4):
        cx = np.random.randint(50, W - 50)
        for yy in range(100, H - 100):
            damage[..., yy, cx + int(8 * math.sin(yy / 90)):cx + int(8 * math.sin(yy / 90)) + 2] = 1
    x = (x + torch.randn(1, 1, H, W) * 0.02).clamp(0, 1)  # film grain
    damaged = x * (1 - damage) + torch.where(torch.rand(1, 1, H, W) > 0.5, 0.95, 0.05) * damage
    dimg = to_bhwc(damaged)
    m, _ = N.LumaLockDefectMask().run(dimg, 0.3, 3, "both", 1)
    det = m.unsqueeze(1)
    recall = float((det * damage).sum() / damage.sum())
    clean = 1 - F.max_pool2d(damage, 3, 1, 1)  # exclude the intended 1 px safety margin (grow_px=1)
    false_pos = float((det * clean).sum() / clean.sum())
    filled = N.LumaLockFillDefects().run(dimg, m, N.FAST_FILL)[0]
    err_before = float((damaged - x).abs()[damage.bool().expand_as(x)].mean())
    err_after = float((to_bchw(filled) - x).abs()[damage.bool().expand_as(x)].mean())
    print(f"\nrestoration: detected {recall*100:.1f}% of damaged pixels, false positives on {false_pos*100:.2f}% of clean pixels; "
          f"error in damaged pixels {err_before:.3f} -> {err_after:.3f}")
    assert recall > 0.85 and false_pos < 0.01 and err_after < err_before * 0.3
    tone = N.LumaLockTone().run(to_bhwc(0.3 + 0.4 * to_bchw(gt)), 0.1, 0.1, 1.0, 1.0, 0.15)[0]
    L_t = rgb_to_L(to_bchw(tone))
    print(f"tone repair: faded L range {float(rgb_to_L(0.3 + 0.4 * to_bchw(gt)).min()):.1f}-{float(rgb_to_L(0.3 + 0.4 * to_bchw(gt)).max()):.1f} -> {float(L_t.min()):.1f}-{float(L_t.max()):.1f}")
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
