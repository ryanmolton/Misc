"""ComfyUI node definitions for StereoForge."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils
import node_helpers
import nodes as comfy_nodes

from .stereoforge import core

CATEGORY = "StereoForge"
STEREO = "SF_STEREO"


def _device():
    try:
        return mm.get_torch_device()
    except Exception:
        return torch.device("cpu")


def _preview_disp(disp: torch.Tensor) -> torch.Tensor:
    return disp.unsqueeze(-1).expand(*disp.shape, 3).contiguous().float()


# --------------------------------------------------------------------------- #
# 1. Optional downscale (off by default -> input passes through bit-exact)
# --------------------------------------------------------------------------- #

class SF_OptionalDownscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "enabled": ("BOOLEAN", {"default": False, "tooltip": "Off: the image passes through untouched (bit-exact), so the source eye is preserved pixel-for-pixel."}),
            "max_long_side": ("INT", {"default": 3840, "min": 256, "max": 16384, "step": 8,
                                      "tooltip": "When enabled, images whose long side exceeds this are downscaled (Lanczos). Smaller images are never upscaled."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, enabled, max_long_side):
        if not enabled:
            return (image,)
        B, H, W, C = image.shape
        s = max_long_side / max(H, W)
        if s >= 1.0:
            return (image,)
        nh, nw = max(1, round(H * s)), max(1, round(W * s))
        x = comfy.utils.common_upscale(image.movedim(-1, 1), nw, nh, "lanczos", "disabled")
        return (x.movedim(1, -1).clamp(0, 1),)


# --------------------------------------------------------------------------- #
# 2. Disparity sources
# --------------------------------------------------------------------------- #

class SF_DisparityFromGeometry:
    """Convert native ComfyUI MoGe (recommended) or Depth Anything 3 output into
    a normalised disparity map (MASK, 1 = nearest). Both give depth that is
    correct up to scale, so 1/z is the physically correct stereo disparity."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {},
                "optional": {
                    "moge_geometry": ("MOGE_GEOMETRY",),
                    "da3_geometry": ("DA3_GEOMETRY",),
                }}

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("disparity", "preview")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, moge_geometry=None, da3_geometry=None):
        geo = moge_geometry if moge_geometry is not None else da3_geometry
        if geo is None:
            raise ValueError("Connect a MoGe or Depth Anything 3 geometry output.")
        depth = geo["depth"].float()
        outs = []
        for i in range(depth.shape[0]):
            d = depth[i].clone()
            if da3_geometry is not None and moge_geometry is None and "sky" in geo:
                d = torch.where(geo["sky"][i] >= 0.5, torch.full_like(d, float("inf")), d)
            disp = core.depth_to_disparity(d)
            disp = torch.where(torch.isfinite(d) & (d > 0), disp, torch.zeros_like(disp))
            outs.append(core.normalize_disparity(disp))
        disp = torch.stack(outs)
        return (disp, _preview_disp(disp))


class SF_DisparityFromDepthImage:
    """Use any depth/disparity map produced by other nodes (e.g. Depth Anything V2
    preprocessors, Marigold, DepthCrafter...)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "depth_image": ("IMAGE",),
            "encoding": (["disparity (near = bright)", "depth (far = bright)", "depth (near = bright, linear)"],
                         {"default": "disparity (near = bright)",
                          "tooltip": "Depth Anything V1/V2 and MiDaS output disparity with near = bright. "
                                     "Metric/linear depth maps need to be inverted to disparity (1/z)."}),
        }}

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("disparity", "preview")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, depth_image, encoding):
        x = depth_image[..., :3].float().mean(-1)
        outs = []
        for i in range(x.shape[0]):
            v = x[i]
            if encoding.startswith("disparity"):
                disp = v
            elif encoding.startswith("depth (far"):
                disp = core.depth_to_disparity(v + 1e-3)
            else:
                disp = core.depth_to_disparity(1.0 - v + 1e-3)
            outs.append(core.normalize_disparity(disp))
        disp = torch.stack(outs)
        return (disp, _preview_disp(disp))


class SF_DepthEstimate:
    """Self-contained depth (downloads from Hugging Face on first use). Use this
    if your ComfyUI has no native MoGe / Depth Anything 3 nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        from .stereoforge.depth_hf import MODELS
        return {"required": {
            "image": ("IMAGE",),
            "model": (list(MODELS.keys()), {"default": "Depth-Anything-V2-Large"}),
            "process_res": ("INT", {"default": 1036, "min": 280, "max": 2044, "step": 14,
                                    "tooltip": "Long side for the global depth pass (Depth Anything V2). Depth Pro always runs at 1536."}),
            "detail_pass": ("BOOLEAN", {"default": True,
                                        "tooltip": "Depth Anything V2 only: fuse an extra tiled pass at 2x resolution for thin structures on large images (skipped automatically for small images)."}),
            "precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
        }}

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("disparity", "preview")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, model, process_res, detail_pass, precision):
        from .stereoforge import depth_hf
        dev = _device()
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
        if dev.type == "cpu":
            dtype = torch.float32
        outs = []
        for i in range(image.shape[0]):
            raw = depth_hf.estimate_disparity(image[i], model, process_res, detail_pass, dev, dtype)
            outs.append(core.normalize_disparity(raw))
        disp = torch.stack(outs)
        return (disp, _preview_disp(disp))


class SF_DisparityPlaneFit:
    """Flatten user-masked regions (photos/screens/posters, mirrors, glass panes,
    signage) to the plane of their surroundings. Monocular depth often 'sees'
    a depicted scene inside a picture; flattening makes such regions behave
    like the physical surface they are."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "disparity": ("MASK",),
            "regions": ("MASK",),
            "ring_px": ("INT", {"default": 8, "min": 1, "max": 256,
                                "tooltip": "Width of the border ring around each region used to fit its plane."}),
        }}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("disparity",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, disparity, regions, ring_px):
        out = []
        for i in range(disparity.shape[0]):
            m = regions[min(i, regions.shape[0] - 1)]
            if m.shape != disparity[i].shape:
                m = F.interpolate(m[None, None].float(), size=disparity[i].shape, mode="nearest")[0, 0]
            out.append(core.fit_planes(disparity[i].float(), m, ring_px))
        return (torch.stack(out),)


# --------------------------------------------------------------------------- #
# 3. Render the opposite eye at full resolution
# --------------------------------------------------------------------------- #

class SF_StereoRender:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "disparity": ("MASK", {"tooltip": "Normalised disparity, 1 = nearest (from the StereoForge disparity nodes)."}),
            "source_eye": (["left", "right"], {"default": "left",
                                               "tooltip": "Which eye the input image is. The other eye is synthesised."}),
            "depth_budget_pct": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 10.0, "step": 0.1,
                                           "tooltip": "Total parallax range (nearest to farthest) as % of image width. "
                                                      "~2-3% is comfortable on monitors/TVs, 3-5% for VR/headsets and small prints."}),
            "convergence": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "Screen plane position in the disparity range: 0 = farthest point on screen "
                                                 "(everything pops out), 1 = nearest point on screen (everything behind)."}),
            "stereo_window": ("BOOLEAN", {"default": True,
                                          "tooltip": "Automatically push the screen plane back so nothing touching the left/right "
                                                     "frame edges is in front of the screen (avoids window violations)."}),
            "border_fill": (["inpaint", "black (floating window)"], {"default": "inpaint",
                                                                     "tooltip": "Out-of-frame strips at the image sides: regenerate, or leave black."}),
        },
            "optional": {
                "edge_refine_px": ("INT", {"default": -1, "min": -1, "max": 64,
                                           "tooltip": "Guided-filter radius to snap depth edges to image edges. -1 = auto, 0 = off."}),
                "fg_dilate_px": ("INT", {"default": 0, "min": 0, "max": 32,
                                         "tooltip": "Extra edge pixels that move fully opaque with the foreground (0 = off; the soft-edge matte normally handles fringes)."}),
                "edge_band_px": ("INT", {"default": 0, "min": 0, "max": 64,
                                         "tooltip": "Width of the soft matte blended over the regenerated background. 0 = auto."}),
                "edge_slope": ("FLOAT", {"default": 0.35, "min": 0.1, "max": 2.0, "step": 0.05,
                                         "tooltip": "Horizontal disparity slope (px per px) treated as an occlusion edge."}),
                "soft_edge_px": ("INT", {"default": -1, "min": -1, "max": 128,
                                         "tooltip": "Width of the soft-boundary zone (hair, fur, whiskers, defocus) that is matted and moved "
                                                    "with the foreground. -1 = auto (~0.33% of the long side), 0 = off."}),
                "fg_guard_px": ("INT", {"default": -1, "min": -1, "max": 512,
                                        "tooltip": "Foreground pixels next to each disocclusion that are hidden from the inpainter so it "
                                                   "continues the background instead of extending the foreground. -1 = auto, 0 = off."}),
            }}

    RETURN_TYPES = (STEREO, "IMAGE", "MASK", "IMAGE", "MASK")
    RETURN_NAMES = ("stereo", "rendered_view", "inpaint_mask", "disparity_preview", "hole_mask")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, disparity, source_eye, depth_budget_pct, convergence, stereo_window, border_fill,
            edge_refine_px=-1, fg_dilate_px=0, edge_band_px=0, edge_slope=0.35, soft_edge_px=-1, fg_guard_px=-1):
        dev = _device()
        p = core.StereoParams(budget_pct=depth_budget_pct, convergence=convergence, stereo_window=stereo_window,
                              source_eye=source_eye, fg_dilate=fg_dilate_px, edge_band=edge_band_px,
                              grad_thr=edge_slope, soft_band=soft_edge_px, fg_guard=fg_guard_px, black_borders=border_fill.startswith("black"))
        views, masks, alphas, prevs, holes, planes, blacks, fcols, fas, frefs, brefs = [], [], [], [], [], [], [], [], [], [], []
        pbar = comfy.utils.ProgressBar(image.shape[0])
        for i in range(image.shape[0]):
            img = image[i].to(dev).float()
            H, W = img.shape[:2]
            d = disparity[min(i, disparity.shape[0] - 1)].to(dev).float()
            if d.shape != (H, W):
                d = F.interpolate(d[None, None], size=(H, W), mode="bicubic", align_corners=False)[0, 0].clamp(0, 1)
            r = edge_refine_px if edge_refine_px >= 0 else core.auto_px(max(H, W), 1000, 2)
            d = core.refine_disparity_edges(d, img, r)
            res = core.render_opposite_eye(img, d, p)
            views.append(res.view.cpu())
            masks.append(res.inpaint_mask.cpu())
            alphas.append(res.alpha.cpu())
            prevs.append(core.disparity_preview(res.D_src).cpu())
            holes.append(res.hole.float().cpu())
            blacks.append(res.black.cpu())
            fcols.append(res.fringe_color.cpu())
            fas.append(res.fringe_alpha.cpu())
            frefs.append(res.fg_ref.cpu())
            brefs.append(res.bg_ref.cpu())
            planes.append(res.zero_plane)
            pbar.update(1)
        stereo = {"source_eye": source_eye, "alpha": torch.stack(alphas), "rendered": torch.stack(views),
                  "black": torch.stack(blacks), "zero_plane": planes,
                  "fringe_color": torch.stack(fcols), "fringe_alpha": torch.stack(fas),
                  "fg_ref": torch.stack(frefs), "bg_ref": torch.stack(brefs)}
        return (stereo, torch.stack(views), torch.stack(masks), torch.stack(prevs), torch.stack(holes))


# --------------------------------------------------------------------------- #
# 4. Native-resolution tiled inpainting (any inpaint model, FLUX.1 Fill recommended)
# --------------------------------------------------------------------------- #

def _tile_starts(size: int, tile: int, overlap: int) -> list[int]:
    if size <= tile:
        return [0]
    n = math.ceil((size - overlap) / (tile - overlap))
    return sorted({int(round(v)) for v in torch.linspace(0, size - tile, n).tolist()})


class SF_TiledInpaint:
    """Regenerate masked regions at 1:1 pixel scale in overlapping tiles.

    Disocclusions are thin slivers spread over the whole frame, so cropping to
    the mask's bounding box would force a downscale; instead every tile that
    contains mask is inpainted at native resolution (so the new pixels have the
    same acuity as the rest of the eye). Tiles run in raster order and each
    masked pixel is owned by exactly one tile; later tiles see earlier results
    as fixed context, which keeps fills coherent across tile boundaries.
    Works with FLUX.1 Fill (recommended), SD/SDXL inpainting checkpoints, or
    any model (plain noise-mask inpainting)."""

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE",),
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "steps": ("INT", {"default": 28, "min": 1, "max": 200}),
            "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "tile_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64,
                                  "tooltip": "Tile edge in pixels, processed at 1:1 scale. ~1024 for FLUX/SDXL, 512 for SD1.5."}),
            "tile_overlap": ("INT", {"default": 256, "min": 0, "max": 1024, "step": 16}),
            "mask_grow_px": ("INT", {"default": 4, "min": 0, "max": 64,
                                     "tooltip": "Extra pixels the sampler may repaint around the mask to blend seams."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    @staticmethod
    def _decode(vae, samples):
        img = vae.decode(samples)
        if img.ndim == 5:
            img = img.reshape(-1, *img.shape[-3:])
        return img[..., :3].float().cpu()

    def run(self, model, positive, negative, vae, image, mask, seed, steps, cfg, sampler_name, scheduler,
            denoise, tile_size, tile_overlap, mask_grow_px):
        B, H, W, C = image.shape
        if min(H, W) < 64:
            raise ValueError("Image too small for tiled inpainting.")
        out = image.clone().float().cpu()
        th = min(tile_size, H // 16 * 16)
        tw = min(tile_size, W // 16 * 16)
        ov = min(tile_overlap, max(0, min(th, tw) - 16))
        ys, xs = _tile_starts(H, th, ov), _tile_starts(W, tw, ov)
        # ownership: each pixel belongs to the tile whose centre is nearest (per axis)
        cy = torch.tensor([y + th / 2 for y in ys])
        cx = torch.tensor([x + tw / 2 for x in xs])
        own_y = (torch.arange(H)[:, None].float() - cy[None]).abs().argmin(1)
        own_x = (torch.arange(W)[:, None].float() - cx[None]).abs().argmin(1)

        work = []
        for b in range(B):
            m = mask[min(b, mask.shape[0] - 1)].float().cpu()
            if m.shape != (H, W):
                m = F.interpolate(m[None, None], size=(H, W), mode="bilinear")[0, 0]
            rem = m > 0.5
            for iy, y0 in enumerate(ys):
                for ix, x0 in enumerate(xs):
                    if rem[y0:y0 + th, x0:x0 + tw].any():
                        work.append((b, iy, ix, y0, x0))
        pbar = comfy.utils.ProgressBar(max(1, len(work)))
        rems = {}
        for n, (b, iy, ix, y0, x0) in enumerate(work):
            mm.throw_exception_if_processing_interrupted()
            if b not in rems:
                m = mask[min(b, mask.shape[0] - 1)].float().cpu()
                if m.shape != (H, W):
                    m = F.interpolate(m[None, None], size=(H, W), mode="bilinear")[0, 0]
                rems[b] = m > 0.5
            rem = rems[b]
            sub = rem[y0:y0 + th, x0:x0 + tw]
            if not sub.any():
                pbar.update(1)
                continue
            own = (own_y[y0:y0 + th, None] == iy) & (own_x[None, x0:x0 + tw] == ix)
            samp = core.max_filter(sub.float(), mask_grow_px)
            pix = out[b:b + 1, y0:y0 + th, x0:x0 + tw, :3]
            smask = samp[None, None]
            masked = (pix - 0.5) * (1.0 - smask.round()).squeeze(1).unsqueeze(-1) + 0.5
            concat = vae.encode(masked)
            orig = vae.encode(pix)
            pos = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat, "concat_mask": smask})
            neg = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat, "concat_mask": smask})
            latent = {"samples": orig, "noise_mask": smask}
            res = comfy_nodes.common_ksampler(model, seed + n, steps, cfg, sampler_name, scheduler, pos, neg,
                                              latent, denoise=denoise)[0]
            dec = self._decode(vae, res["samples"])[0]
            if dec.shape[:2] != (th, tw):
                dec = F.interpolate(dec.permute(2, 0, 1)[None], size=(th, tw), mode="bicubic")[0].permute(1, 2, 0)
            # paste: owned masked pixels; unowned pixels that later tiles will
            # redo also get the result now so they act as coherent context
            paste = sub
            region = out[b, y0:y0 + th, x0:x0 + tw, :3]
            region[paste] = dec[paste].clamp(0, 1)
            done = sub & own
            rem[y0:y0 + th, x0:x0 + tw] = sub & ~done
            pbar.update(1)
        return (out,)


# --------------------------------------------------------------------------- #
# 5. Compose the side-by-side stereograph
# --------------------------------------------------------------------------- #

class SF_StereoCompose:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "stereo": (STEREO,),
            "source_image": ("IMAGE",),
            "layout": (["parallel (left | right)", "cross-eyed (right | left)"], {"default": "parallel (left | right)"}),
        },
            "optional": {
                "inpainted": ("IMAGE", {"tooltip": "Inpainted version of rendered_view. If omitted, the geometric pre-fill is used."}),
                "reject_foreground_bleed": ("BOOLEAN", {"default": True,
                                                        "tooltip": "Replace generated pixels inside disocclusions that look like the adjacent "
                                                                   "foreground (the inpainter extending an object into the revealed background) "
                                                                   "with the background-side pre-fill."}),
            }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("side_by_side", "generated_eye", "anaglyph_preview")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, stereo, source_image, layout, inpainted=None, reject_foreground_bleed=True):
        rendered, alpha, black = stereo["rendered"], stereo["alpha"], stereo["black"]
        B, H, W, C = source_image.shape
        if rendered.shape[1:3] != (H, W):
            raise ValueError(f"source_image is {W}x{H} but the stereo render is "
                             f"{rendered.shape[2]}x{rendered.shape[1]}; connect the same image used for rendering.")
        gens, sbss, anas = [], [], []
        for i in range(B):
            ri = min(i, rendered.shape[0] - 1)
            inp = inpainted[min(i, inpainted.shape[0] - 1)][..., :3].float() if inpainted is not None else None
            if inp is not None and reject_foreground_bleed and "fg_ref" in stereo:
                inp = core.reject_foreground_bleed(inp, rendered[ri].float(), alpha[ri].float(),
                                                   stereo["fg_ref"][ri].float(), stereo["bg_ref"][ri].float())
            gen = core.composite(rendered[ri].float(), inp[..., :3] if inp is not None else None, alpha[ri].float(),
                                 stereo["fringe_color"][ri].float(), stereo["fringe_alpha"][ri].float())
            gen = torch.where(black[ri].unsqueeze(-1), torch.zeros_like(gen), gen)
            src = source_image[i]
            if src.shape[-1] != gen.shape[-1]:
                gen = torch.cat([gen, torch.ones_like(gen[..., :1]).expand(*gen.shape[:2], src.shape[-1] - 3)], -1)
            if stereo["source_eye"] == "left":
                left, right = src, gen.to(src.dtype)
            else:
                left, right = gen.to(src.dtype), src
            pair = (left, right) if layout.startswith("parallel") else (right, left)
            sbss.append(torch.cat(pair, dim=1))
            gens.append(gen.to(src.dtype))
            anas.append(core.anaglyph_dubois(left[..., :3].float(), right[..., :3].float()))
        return (torch.stack(sbss), torch.stack(gens), torch.stack(anas))


NODE_CLASS_MAPPINGS = {
    "SF_OptionalDownscale": SF_OptionalDownscale,
    "SF_DisparityFromGeometry": SF_DisparityFromGeometry,
    "SF_DisparityFromDepthImage": SF_DisparityFromDepthImage,
    "SF_DepthEstimate": SF_DepthEstimate,
    "SF_DisparityPlaneFit": SF_DisparityPlaneFit,
    "SF_StereoRender": SF_StereoRender,
    "SF_TiledInpaint": SF_TiledInpaint,
    "SF_StereoCompose": SF_StereoCompose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SF_OptionalDownscale": "StereoForge: Optional Downscale",
    "SF_DisparityFromGeometry": "StereoForge: Disparity from MoGe / DA3",
    "SF_DisparityFromDepthImage": "StereoForge: Disparity from Depth Image",
    "SF_DepthEstimate": "StereoForge: Depth Estimate (built-in)",
    "SF_DisparityPlaneFit": "StereoForge: Flatten Regions to Plane",
    "SF_StereoRender": "StereoForge: Render Opposite Eye",
    "SF_TiledInpaint": "StereoForge: Tiled Inpaint (native res)",
    "SF_StereoCompose": "StereoForge: Compose Side-by-Side",
}
