"""ComfyUI nodes for StereoForge Segmentation (object layers + reviewable clean plates)."""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils
import node_helpers
import nodes as comfy_nodes

from .sfs import core, layers as L, models

CATEGORY = "StereoForge Segmentation"
LAYERS = "SFS_LAYERS"
PALETTE = [(0.95, 0.35, 0.35), (0.35, 0.75, 0.95), (0.55, 0.9, 0.4), (0.95, 0.8, 0.3), (0.8, 0.45, 0.95),
           (0.3, 0.9, 0.8), (0.95, 0.55, 0.2), (0.6, 0.6, 1.0)]


def _dev():
    try:
        return mm.get_torch_device()
    except Exception:
        return torch.device("cpu")


def _label(img: torch.Tensor, items):
    """Draw text labels [(x, y, text)] onto an (H, W, 3) tensor."""
    from PIL import Image, ImageDraw
    arr = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    size = max(12, img.shape[1] // 60)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=size)
    except Exception:
        font = None
    for x, y, t in items:
        d.text((x, y), t, fill=(255, 255, 255), stroke_width=max(1, size // 8), stroke_fill=(0, 0, 0), font=font,
               anchor="mm")
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)


def _outline(mask: torch.Tensor, r: int = 1) -> torch.Tensor:
    m = mask.bool()
    return L.dilate(m, r) & ~L.erode(m, r)


# --------------------------------------------------------------------------- #
# input helpers (same behaviour as the original StereoForge pack)
# --------------------------------------------------------------------------- #

class SFS_OptionalDownscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "enabled": ("BOOLEAN", {"default": False, "tooltip": "Off: the image passes through bit-exact."}),
            "max_long_side": ("INT", {"default": 3840, "min": 256, "max": 16384, "step": 8}),
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
        x = comfy.utils.common_upscale(image.movedim(-1, 1), max(1, round(W * s)), max(1, round(H * s)),
                                       "lanczos", "disabled")
        return (x.movedim(1, -1).clamp(0, 1),)


class SFS_DisparityFromGeometry:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"moge_geometry": ("MOGE_GEOMETRY",), "da3_geometry": ("DA3_GEOMETRY",)}}

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
            if moge_geometry is None and "sky" in geo:
                d = torch.where(geo["sky"][i] >= 0.5, torch.full_like(d, float("inf")), d)
            outs.append(core.normalize_disparity(core.depth_to_disparity(d)))
        disp = torch.stack(outs)
        return (disp, disp.unsqueeze(-1).expand(*disp.shape, 3).contiguous())


# --------------------------------------------------------------------------- #
# objects
# --------------------------------------------------------------------------- #

class SFS_CollectObjects:
    """Gather object masks: SAM 3 detections and/or masks you paint yourself.

    Each mask becomes one depth layer with its own matte. Everything not
    covered by a mask is background. Give every object whose outline matters a
    mask (people, pets, furniture edges, gifts...)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "drop": ("STRING", {"default": "", "tooltip": "Comma-separated numbers (from the preview) of detected masks to discard, e.g. '2,5'."}),
            "min_area_pct": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 50.0, "step": 0.01,
                                       "tooltip": "Discard masks smaller than this % of the image."}),
        }, "optional": {
            "detected": ("MASK", {"tooltip": "Per-object masks, e.g. SAM3 Detect with individual_masks on."}),
            "mask_1": ("MASK", {"tooltip": "Extra object you painted (Load Image > mask editor). Replaces nothing; it is added."}),
            "mask_2": ("MASK",),
            "mask_3": ("MASK",),
            "mask_4": ("MASK",),
        }}

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("objects", "preview")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, drop, min_area_pct, detected=None, mask_1=None, mask_2=None, mask_3=None, mask_4=None):
        img = image[0, ..., :3].float()
        H, W, _ = img.shape
        cands = []
        if detected is not None:
            det = detected if detected.ndim == 3 else detected[None]
            cands += [det[i] for i in range(det.shape[0])]
        for m in (mask_1, mask_2, mask_3, mask_4):
            if m is not None:
                cands.append(m[0] if m.ndim == 3 else m)
        dropped = {int(t) for t in drop.replace(" ", "").split(",") if t.strip().isdigit()}
        keep = []
        for i, m in enumerate(cands, start=1):
            m = m.float()
            if m.shape != (H, W):
                m = F.interpolate(m[None, None], size=(H, W), mode="bilinear")[0, 0]
            if i in dropped or (m > 0.5).float().mean() * 100 < min_area_pct:
                continue
            keep.append((i, m.clamp(0, 1)))
        prev = img * 0.45
        labels = []
        for n, (i, m) in enumerate(keep):
            col = torch.tensor(PALETTE[n % len(PALETTE)])
            a = (m > 0.5).float().unsqueeze(-1)
            prev = prev * (1 - 0.55 * a) + (0.55 * a) * (img * 0.4 + col * 0.6)
            edge = _outline(m > 0.5, 1).unsqueeze(-1)
            prev = torch.where(edge, col.expand_as(prev), prev)
            ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
            labels.append((float(xs.float().mean()), float(ys.float().mean()), str(i)))
        prev = _label(prev, labels)
        objs = torch.stack([m for _, m in keep]) if keep else torch.zeros(0, H, W)
        return (objs, prev[None])


# --------------------------------------------------------------------------- #
# build layers
# --------------------------------------------------------------------------- #

class SFS_BuildLayers:
    """Mattes, per-object depth and clean plates. Everything the new eye will
    show that the photo does not contain is created here, in the source view,
    where you can review and fix it (see plate_review)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "disparity": ("MASK",),
            "objects": ("MASK",),
            "source_eye": (["left", "right"], {"default": "left",
                           "tooltip": "Which eye the photo is. Tip: generating the other side moves all invented "
                                      "background to the other side of every object - pick the side with simpler backgrounds."}),
            "depth_budget_pct": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            "convergence": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            "stereo_window": ("BOOLEAN", {"default": True}),
            "matting": (["ViTMatte-Base", "ViTMatte-Small", "none (hard edges)"], {"default": "ViTMatte-Base"}),
            "edge_band_px": ("INT", {"default": 0, "min": 0, "max": 128,
                                     "tooltip": "How far (px) either side of each mask edge the matting may move it. "
                                                "0 = auto (~0.33% of the long side). Raise for loose masks or fluffy hair."}),
            "object_smoothing_px": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 200.0, "step": 0.5,
                                              "tooltip": "Blur radius for each object's own depth. Higher = no stretching or tearing "
                                                         "where parts of one object overlap (arm across chest), but flatter objects. 0 = auto (~1% of the long side)."}),
        }}

    RETURN_TYPES = (LAYERS, "IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("layers", "matte_preview", "plate_review", "plates", "plate_masks")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, disparity, objects, source_eye, depth_budget_pct, convergence, stereo_window, matting,
            edge_band_px, object_smoothing_px=0.0):
        dev = _dev()
        img = image[0, ..., :3].to(dev).float()
        H, W, _ = img.shape
        d = disparity[0].to(dev).float()
        if d.shape != (H, W):
            d = F.interpolate(d[None, None], size=(H, W), mode="bicubic", align_corners=False)[0, 0].clamp(0, 1)
        p = core.StereoParams(budget_pct=depth_budget_pct, convergence=convergence, stereo_window=stereo_window,
                              source_eye=source_eye)
        matte_fn = None
        if not matting.startswith("none"):
            mp = models.load_vitmatte(matting, dev, torch.float16 if dev.type == "cuda" else torch.float32)
            matte_fn = lambda im, tri: models.vitmatte(mp, im, tri)
        lama = models.load_lama(dev)
        md = max(H, W)
        guard = max(24, int(round(md / 40)))
        fill_fn = lambda im, fill, hide: models.lama_fill(lama, im, fill, hide, guard)
        masks = [objects[i].to(dev) for i in range(objects.shape[0])] if objects is not None else []
        stack = L.build_stack(img, d, masks, p, matte_fn, fill_fn, edge_band_px, log=logging.info,
                              obj_smooth=object_smoothing_px)
        stack.params["guard"] = guard
        return (stack, *self.previews(stack))

    @staticmethod
    def previews(stack):
        img = stack.image
        H, W, _ = img.shape
        prev = img * 0.5
        labels = []
        for n, Ly in enumerate(stack.layers[1:], start=1):
            col = torch.tensor(PALETTE[(n - 1) % len(PALETTE)], device=img.device)
            a = Ly.alpha.unsqueeze(-1)
            prev = prev * (1 - a) + a * img
            edge = _outline(Ly.alpha > 0.5, 1).unsqueeze(-1)
            prev = torch.where(edge, col.expand_as(prev), prev)
            ys, xs = torch.nonzero(Ly.alpha > 0.5, as_tuple=True)
            if len(xs):
                labels.append((float(xs.float().mean()), float(ys.float().mean()), f"layer {n}"))
        matte_prev = _label(prev.cpu(), labels)[None]
        reviews, plates, pmasks = [], [], []
        for n, Ly in enumerate(stack.layers):
            plate = Ly.color
            inv = Ly.invented
            rv = plate.clone()
            # dim everything that is not part of this layer, outline invented content
            if n > 0:
                own = (Ly.alpha > 0.5) | inv
                rv = torch.where(own.unsqueeze(-1), rv, rv * 0.3)
            edge = _outline(inv, max(1, W // 1500)).unsqueeze(-1)
            rv = torch.where(edge, torch.tensor([1.0, 0.1, 0.9], device=img.device).expand_as(rv), rv)
            reviews.append(_label(rv.cpu(), [(W * 0.5, max(16, H // 30),
                                              f"layer {n} ({Ly.name}): outlined = invented, check it looks natural")]))
            plates.append(plate.cpu())
            pmasks.append(inv.float().cpu())
        return matte_prev, torch.stack(reviews), torch.stack(plates), torch.stack(pmasks)


# --------------------------------------------------------------------------- #
# manual / diffusion fixes to the clean plates
# --------------------------------------------------------------------------- #

class SFS_EditPlate:
    """Apply your own fix to one layer's clean plate.

    Save the 'plates' output, edit that layer's image in any editor (clone,
    heal, paint), load it with Load Image and connect it here. Only the
    invented pixels of that layer are taken from your edit - everything else
    stays exactly as in the photo."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "layers": (LAYERS,),
            "layer": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "0 = background, 1.. = object layers (see matte_preview)."}),
            "edited_plate": ("IMAGE",),
        }, "optional": {
            "only_here": ("MASK", {"tooltip": "Optional: restrict the edit to where you painted."}),
        }}

    RETURN_TYPES = (LAYERS,)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, layers, layer, edited_plate, only_here=None):
        st = layers.clone()
        if layer >= len(st.layers):
            raise ValueError(f"There are only {len(st.layers)} layers (0..{len(st.layers) - 1}).")
        Ly = st.layers[layer]
        H, W = Ly.alpha.shape
        ed = edited_plate[0, ..., :3].float().to(Ly.color.device)
        if ed.shape[:2] != (H, W):
            raise ValueError(f"Edited plate is {ed.shape[1]}x{ed.shape[0]}, expected {W}x{H}.")
        m = Ly.invented
        if only_here is not None:
            oh = only_here[0].to(m.device) if only_here.ndim == 3 else only_here.to(m.device)
            m = m & (oh > 0.5)
        Ly.color = torch.where(m.unsqueeze(-1), ed, Ly.color)
        L.decontaminate(st)
        return (st,)


class SFS_RefinePlates:
    """Regenerate clean-plate content with a diffusion inpainting model
    (FLUX.1 Fill recommended) instead of LaMa. Only the invented pixels are
    replaced, at 1:1 pixel scale, with nearer objects hidden from the model.
    Use 'only_here' to redo just the spots you are unhappy with."""

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {"required": {
            "layers": (LAYERS,),
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE",),
            "layer": ("INT", {"default": -1, "min": -1, "max": 64, "tooltip": "-1 = all layers, 0 = background, 1.. = object layers."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
            "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
            "context_px": ("INT", {"default": 96, "min": 16, "max": 512, "step": 16}),
        }, "optional": {
            "only_here": ("MASK",),
        }}

    RETURN_TYPES = (LAYERS,)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, layers, model, positive, negative, vae, layer, seed, steps, cfg, sampler_name, scheduler,
            context_px, only_here=None):
        st = layers.clone()
        idx = range(len(st.layers)) if layer < 0 else [layer]
        guard = st.params.get("guard", 32)
        n = 0
        for i in idx:
            Ly = st.layers[i]
            fill = Ly.invented.cpu()
            if only_here is not None:
                oh = (only_here[0] if only_here.ndim == 3 else only_here).cpu()
                fill = fill & (oh > 0.5)
            if not bool(fill.any()):
                continue
            hide = Ly.hide.cpu() & core.max_filter(fill.float(), guard).bool()
            base = Ly.color.cpu()
            res = diffusion_fill(model, positive, negative, vae, base, fill, hide, seed + 1000 * n, steps, cfg,
                                 sampler_name, scheduler, context_px)
            Ly.color = torch.where(fill.unsqueeze(-1).to(Ly.color.device), res.to(Ly.color.device), Ly.color)
            n += 1
        L.decontaminate(st)
        return (st,)


def diffusion_fill(model, positive, negative, vae, image, fill, hide, seed, steps, cfg, sampler_name, scheduler,
                   ctx, tile=1024):
    """Tiled 1:1 inpainting of ``fill`` (hide is masked but not kept)."""
    H, W, _ = image.shape
    out = image.clone().float()
    max_h, max_w = min(tile, H // 16 * 16), min(tile, W // 16 * 16)
    ctx = max(0, min(ctx, (min(max_h, max_w) - 64) // 2))
    ch, cw = max(16, max_h - 2 * ctx), max(16, max_w - 2 * ctx)
    rem = fill.clone()
    cells = [(y, min(H, y + ch), x, min(W, x + cw)) for y in range(0, H, ch) for x in range(0, W, cw)]
    work = [c for c in cells if bool(rem[c[0]:c[1], c[2]:c[3]].any())]
    pbar = comfy.utils.ProgressBar(max(1, len(work)))
    for n, cell in enumerate(work):
        mm.throw_exception_if_processing_interrupted()
        owned = torch.zeros_like(rem)
        owned[cell[0]:cell[1], cell[2]:cell[3]] = rem[cell[0]:cell[1], cell[2]:cell[3]]
        crop = core.plan_crop(owned, cell, H, W, max_h, max_w, ctx, min(384, max_h, max_w))
        if crop is None:
            pbar.update(1)
            continue
        y0, y1, x0, x1 = crop
        logging.info(f"[StereoForgeSeg] diffusion crop {n + 1}/{len(work)}: {x1 - x0}x{y1 - y0}")
        sub = rem[y0:y1, x0:x1]
        samp = core.max_filter((sub | hide[y0:y1, x0:x1]).float(), 4)
        pix = out[None, y0:y1, x0:x1, :3]
        smask = samp[None, None]
        masked = (pix - 0.5) * (1.0 - smask.round()).squeeze(1).unsqueeze(-1) + 0.5
        concat = vae.encode(masked)
        orig = vae.encode(pix)
        pos = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat, "concat_mask": smask})
        neg = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat, "concat_mask": smask})
        res = comfy_nodes.common_ksampler(model, seed + n, steps, cfg, sampler_name, scheduler, pos, neg,
                                          {"samples": orig, "noise_mask": smask}, denoise=1.0)[0]
        dec = vae.decode(res["samples"])
        if dec.ndim == 5:
            dec = dec.reshape(-1, *dec.shape[-3:])
        dec = dec[0, ..., :3].float().cpu()
        if dec.shape[:2] != (y1 - y0, x1 - x0):
            dec = F.interpolate(dec.permute(2, 0, 1)[None], size=(y1 - y0, x1 - x0), mode="bicubic")[0].permute(1, 2, 0)
        region = out[y0:y1, x0:x1, :3]
        region[sub] = dec[sub].clamp(0, 1)
        rem[y0:y1, x0:x1] = sub & ~owned[y0:y1, x0:x1]
        pbar.update(1)
    return out


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #

class SFS_Render:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "layers": (LAYERS,),
            "source_image": ("IMAGE", {"tooltip": "The same image used to build the layers (copied bit-exactly as the source eye)."}),
            "layout": (["parallel (left | right)", "cross-eyed (right | left)"], {"default": "parallel (left | right)"}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("side_by_side", "generated_eye", "review", "anaglyph")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, layers, source_image, layout):
        st = layers
        src = source_image[0]
        H, W = src.shape[:2]
        if (H, W) != tuple(st.image.shape[:2]):
            raise ValueError("source_image does not match the image the layers were built from.")
        lama = models.load_lama(st.image.device)
        hf = lambda im, m, h: models.lama_fill(lama, im, m, h, 8)
        eye, inv_vis, auto = L.render_stack(st, hf)
        eye = eye.cpu()
        review = eye * 0.85
        review = torch.where(_outline(inv_vis.cpu(), 1).unsqueeze(-1), torch.tensor([0.1, 1.0, 1.0]).expand_as(review), review)
        review = torch.where(auto.cpu().unsqueeze(-1), review * 0.5 + torch.tensor([1.0, 0.1, 0.9]) * 0.5, review)
        review = _label(review, [(W * 0.5, max(16, H // 30), "cyan outline = invented (from plates)   magenta = auto-filled: add a mask there if it matters")])
        gen = eye.to(src.dtype)
        if src.shape[-1] > 3:
            gen = torch.cat([gen, torch.ones_like(gen[..., :1]).expand(H, W, src.shape[-1] - 3)], -1)
        left, right = (src, gen) if st.source_eye == "left" else (gen, src)
        pair = (left, right) if layout.startswith("parallel") else (right, left)
        sbs = torch.cat(pair, dim=1)
        ana = core.anaglyph_dubois(left[..., :3].float(), right[..., :3].float())
        return (sbs[None], gen[None], review[None], ana[None])


NODE_CLASS_MAPPINGS = {
    "SFS_OptionalDownscale": SFS_OptionalDownscale,
    "SFS_DisparityFromGeometry": SFS_DisparityFromGeometry,
    "SFS_CollectObjects": SFS_CollectObjects,
    "SFS_BuildLayers": SFS_BuildLayers,
    "SFS_EditPlate": SFS_EditPlate,
    "SFS_RefinePlates": SFS_RefinePlates,
    "SFS_Render": SFS_Render,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SFS_OptionalDownscale": "StereoForge Seg: Optional Downscale",
    "SFS_DisparityFromGeometry": "StereoForge Seg: Disparity from MoGe / DA3",
    "SFS_CollectObjects": "StereoForge Seg: Collect Object Masks",
    "SFS_BuildLayers": "StereoForge Seg: Build Layers (mattes + clean plates)",
    "SFS_EditPlate": "StereoForge Seg: Apply Plate Edit",
    "SFS_RefinePlates": "StereoForge Seg: Refine Plates (diffusion)",
    "SFS_Render": "StereoForge Seg: Render Stereo Pair",
}
