import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils
import folder_paths

from . import stereo_core as sc

MODEL_FOLDER = "stereospace"
_default_dir = os.path.join(folder_paths.models_dir, MODEL_FOLDER)
if MODEL_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(MODEL_FOLDER, _default_dir, is_default=True)

_PIPE_CACHE = {}


def _log(msg):
    print(f"[StereoForge] {msg}")


def _pick_dtype(precision, device):
    if precision == "fp32" or device.type == "cpu":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _mirror(x):
    return torch.flip(x, dims=[-1])


# ============================================================================
class StereoForgeLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "precision": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("STEREOSPACE",)
    RETURN_NAMES = ("stereo_model",)
    FUNCTION = "load"
    CATEGORY = "StereoForge"
    DESCRIPTION = "Loads StereoSpace (prs-eth/stereospace-v1-0, MIT). Downloads ~11 GB into models/stereospace on first use."

    def load(self, precision):
        from .stereospace_vendor.pipeline import StereoSpacePipeline, ensure_weights

        base = folder_paths.get_folder_paths(MODEL_FOLDER)[0]
        model_dir = os.path.join(base, "stereospace-v1-0")
        ensure_weights(model_dir)
        device = mm.get_torch_device()
        dtype = _pick_dtype(precision, device)
        key = (model_dir, str(dtype))
        if key not in _PIPE_CACHE:
            _PIPE_CACHE.clear()
            _log(f"loading StereoSpace ({dtype}) from {model_dir}")
            _PIPE_CACHE[key] = StereoSpacePipeline(model_dir, torch.device("cpu"), dtype)
        return (_PIPE_CACHE[key],)


# ============================================================================
class StereoForgePrepareInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "downscale": ("BOOLEAN", {"default": False, "tooltip": "Off = the source eye is kept pixel-for-pixel."}),
                "max_long_side": ("INT", {"default": 4096, "min": 256, "max": 32768, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("working_image",)
    FUNCTION = "run"
    CATEGORY = "StereoForge"

    def run(self, image, downscale, max_long_side):
        image = image[..., :3]
        if not downscale:
            return (image,)
        _, H, W, _ = image.shape
        s = max_long_side / float(max(H, W))
        if s >= 1.0:
            return (image,)
        h, w = max(1, int(round(H * s))), max(1, int(round(W * s)))
        x = sc.to_bchw(image)
        x = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        return (sc.to_bhwc(x),)


# ============================================================================
class StereoForgeGenerateView:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stereo_model": ("STEREOSPACE",),
                "image": ("IMAGE",),
                "generate_eye": (["right", "left"], {"default": "right",
                                  "tooltip": "Which eye to synthesise. The input is kept as the other eye."}),
                "baseline_mode": (["auto_comfort", "fixed"], {"default": "auto_comfort"}),
                "baseline_m": ("FLOAT", {"default": 0.065, "min": 0.005, "max": 1.0, "step": 0.005,
                                         "tooltip": "Camera separation in metres (fixed mode, and first probe in auto mode). Human IPD ~0.065."}),
                "target_max_disparity_pct": ("FLOAT", {"default": 2.5, "min": 0.3, "max": 8.0, "step": 0.1,
                                                       "tooltip": "auto_comfort: nearest content gets this much parallax (percent of image width)."}),
                "fov_deg": ("FLOAT", {"default": 55.0, "min": 10.0, "max": 120.0, "step": 0.5,
                                      "tooltip": "Assumed field of view across the long side of the image."}),
                "processing_megapixels": ("FLOAT", {"default": 0.59, "min": 0.15, "max": 1.6, "step": 0.01,
                                                    "tooltip": "Generator resolution. The model was trained at 768x768 (0.59 MP)."}),
                "steps": ("INT", {"default": 50, "min": 4, "max": 200}),
                "guidance": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 5.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "offload_after": ("BOOLEAN", {"default": True, "tooltip": "Move StereoSpace back to CPU afterwards to free VRAM for the refiner."}),
            }
        }

    RETURN_TYPES = ("STEREO_GEN", "IMAGE", "STRING")
    RETURN_NAMES = ("stereo_gen", "generated_lowres", "info")
    FUNCTION = "run"
    CATEGORY = "StereoForge"

    def _gen(self, pipe, src_lr, baseline, focal, steps, guidance, seed, pbar, done, total):
        def cb(i, n):
            pbar.update_absolute(done + i, total)
            mm.throw_exception_if_processing_interrupted()

        return pipe.generate(src_lr, baseline, focal, steps=steps, guidance=guidance, seed=seed, callback=cb)

    def run(self, stereo_model, image, generate_eye, baseline_mode, baseline_m, target_max_disparity_pct,
            fov_deg, processing_megapixels, steps, guidance, seed, offload_after):
        pipe = stereo_model
        image = image[:1, ..., :3]
        _, H, W, _ = image.shape
        ph, pw = sc.fit_processing_size(H, W, processing_megapixels)
        src_hr = sc.to_bchw(image)
        src_lr = sc.resize(src_hr, ph, pw, "area").clamp(0, 1)
        focal = 0.5 * max(ph, pw) / math.tan(math.radians(fov_deg) / 2)
        sign = 1.0 if generate_eye == "right" else -1.0

        device = mm.get_torch_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        pipe.to(device)
        info = [f"input {W}x{H}, generator {pw}x{ph}, f={focal:.1f}px"]
        try:
            probe_steps = min(steps, 20) if baseline_mode == "auto_comfort" else steps
            total = probe_steps + (steps if baseline_mode == "auto_comfort" else 0)
            pbar = comfy.utils.ProgressBar(total)
            b = baseline_m
            gen = self._gen(pipe, src_lr, sign * b, focal, probe_steps, guidance, seed, pbar, 0, total)
            if baseline_mode == "auto_comfort":
                s_c, g_c = (src_lr, gen) if sign > 0 else (_mirror(src_lr), _mirror(gen))
                d, v = sc.estimate_disparity_lr(s_c.to(device), sc.fit_color(g_c.float(), s_c, torch.ones_like(s_c[:, :1])).to(device), device=device)
                st = sc.disparity_stats(d, v, pw)
                if st is None or st["p98"] <= 1e-4:
                    info.append("auto_comfort: disparity not measurable, keeping baseline")
                    new_b = b
                else:
                    new_b = float(np.clip(b * (target_max_disparity_pct / 100.0) / st["p98"], 0.01, 0.6))
                    info.append(
                        f"probe baseline {b:.3f} m -> max parallax {100*st['p98']:.2f}% ; "
                        f"new baseline {new_b:.3f} m"
                    )
                gen = self._gen(pipe, src_lr, sign * new_b, focal, steps, guidance, seed, pbar, probe_steps, total)
                b = new_b
        finally:
            if offload_after:
                pipe.to("cpu")
                mm.soft_empty_cache()
        gen = gen.float().cpu()
        info.append(f"final baseline {b:.3f} m ({generate_eye} eye)")
        payload = dict(src_lr=src_lr.cpu(), gen_lr=gen, eye=generate_eye, baseline=b, fov=fov_deg,
                       size=(H, W))
        text = "\n".join(info)
        _log(text)
        return (payload, sc.to_bhwc(gen), text)


# ============================================================================
class StereoForgeLiftToFullRes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The full-resolution working image (same one given to Generate View)."}),
                "stereo_gen": ("STEREO_GEN",),
                "view_dependent_sensitivity": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05,
                                                         "tooltip": "Higher = trust the generated view more where it disagrees with the warped source (reflections, glass, speculars)."}),
                "match_sharpness": ("BOOLEAN", {"default": True}),
                "hole_dilate_px": ("INT", {"default": 2, "min": 0, "max": 16}),
            },
            "optional": {
                "generated_upscaled": ("IMAGE", {"tooltip": "Optional: the low-res generated view after a SR model (e.g. 4x ESRGAN)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK")
    RETURN_NAMES = ("target_eye", "refine_mask", "disparity_vis", "resynth_mask")
    FUNCTION = "run"
    CATEGORY = "StereoForge"

    def run(self, image, stereo_gen, view_dependent_sensitivity, match_sharpness, hole_dilate_px, generated_upscaled=None):
        g = stereo_gen
        src_hr = sc.to_bchw(image[:1])
        H, W = src_hr.shape[-2:]
        if (H, W) != tuple(g["size"]):
            raise ValueError(f"image {W}x{H} differs from the one used in Generate View {g['size'][1]}x{g['size'][0]}")
        left_target = g["eye"] == "left"
        src_lr, gen_lr = g["src_lr"], g["gen_lr"]
        hint = sc.to_bchw(generated_upscaled[:1]) if generated_upscaled is not None else None
        if left_target:
            src_hr, src_lr, gen_lr = _mirror(src_hr), _mirror(src_lr), _mirror(gen_lr)
            hint = _mirror(hint) if hint is not None else None

        def go(dev):
            return sc.lift_to_full_resolution(
                src_hr, src_lr, gen_lr, gen_hr_hint=hint, sensitivity=view_dependent_sensitivity,
                sharpen_match=match_sharpness, hole_dilate_px=hole_dilate_px, device=dev,
            )

        dev = mm.get_torch_device()
        try:
            out, refine, dbg = go(dev)
        except torch.cuda.OutOfMemoryError:
            _log("GPU out of memory in lift, retrying on CPU")
            mm.soft_empty_cache()
            out, refine, dbg = go(torch.device("cpu"))
        resynth = dbg["replace"]
        vis = sc.colorize_disparity(dbg["disp_tgt"])
        if left_target:
            out, refine, resynth, vis = map(_mirror, (out, refine, resynth, vis))
        out = out.float().cpu()
        return (sc.to_bhwc(out), refine[:, 0].float().cpu(), sc.to_bhwc(vis.float().cpu()), resynth[:, 0].float().cpu())


# ============================================================================
def _tile_starts(size, tile, overlap):
    if size <= tile:
        return [0]
    step = tile - overlap
    n = int(math.ceil((size - tile) / step)) + 1
    return [min(i * step, size - tile) for i in range(n)]


class StereoForgeTiledMaskedRefine:
    """Native-resolution, mask-limited img2img detail pass (any ComfyUI model)."""

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 24, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tile_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 1024, "step": 16}),
                "mask_threshold": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "StereoForge"

    def run(self, image, mask, model, positive, negative, vae, seed, steps, cfg, sampler_name, scheduler,
            denoise, tile_size, overlap, mask_threshold):
        import nodes as comfy_nodes

        img = image[:1, ..., :3].clone()
        _, H, W, _ = img.shape
        m = mask[:1]
        if m.shape[-2:] != (H, W):
            m = F.interpolate(m[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
        m = m.clamp(0, 1)
        if float(m.max()) < mask_threshold or denoise <= 0:
            return (img,)
        th = min(tile_size, int(math.ceil(H / 16)) * 16)
        tw = min(tile_size, int(math.ceil(W / 16)) * 16)
        ys, xs = _tile_starts(H, th, overlap), _tile_starts(W, tw, overlap)
        tiles = [(y, x) for y in ys for x in xs if float(m[:, y:y + th, x:x + tw].max()) >= mask_threshold]
        pbar = comfy.utils.ProgressBar(len(tiles))
        out = img.clone()
        for i, (y, x) in enumerate(tiles):
            crop = out[:, y:y + th, x:x + tw, :]
            mc = m[:, y:y + th, x:x + tw]
            ch, cw = crop.shape[1:3]
            ph, pw = th - ch, tw - cw  # only when the image is smaller than a tile
            if ph or pw:
                crop = F.pad(crop.permute(0, 3, 1, 2), (0, pw, 0, ph), mode="replicate").permute(0, 2, 3, 1)
                mc = F.pad(mc[:, None], (0, pw, 0, ph), mode="constant", value=0)[:, 0]
            lat = vae.encode(crop)
            latent = {"samples": lat, "noise_mask": mc.reshape(1, 1, th, tw)}
            res = comfy_nodes.common_ksampler(model, seed + i, steps, cfg, sampler_name, scheduler, positive,
                                              negative, latent, denoise=denoise)[0]
            dec = vae.decode(res["samples"])
            if dec.dim() == 5:
                dec = dec.reshape(-1, *dec.shape[-3:])
            dec = dec[:1, :ch, :cw, :3].to(out)
            if dec.shape[1:3] != (ch, cw):
                dec = F.interpolate(dec.permute(0, 3, 1, 2), size=(ch, cw), mode="bicubic",
                                    align_corners=False).permute(0, 2, 3, 1)
            # feather tile borders that touch neighbouring tiles
            fy = torch.ones(ch, device=out.device)
            fx = torch.ones(cw, device=out.device)
            ov = max(1, overlap)
            ramp = torch.linspace(0, 1, ov + 2, device=out.device)[1:-1]
            if y > 0:
                fy[:ov] = torch.minimum(fy[:ov], ramp[: min(ov, ch)])
            if y + ch < H:
                fy[-ov:] = torch.minimum(fy[-ov:], ramp.flip(0)[-min(ov, ch):])
            if x > 0:
                fx[:ov] = torch.minimum(fx[:ov], ramp[: min(ov, cw)])
            if x + cw < W:
                fx[-ov:] = torch.minimum(fx[-ov:], ramp.flip(0)[-min(ov, cw):])
            wgt = (m[:, y:y + ch, x:x + cw].to(out) * fy[:, None] * fx[None, :])[..., None]
            out[:, y:y + ch, x:x + cw, :] = out[:, y:y + ch, x:x + cw, :] * (1 - wgt) + dec.clamp(0, 1) * wgt
            pbar.update_absolute(i + 1, len(tiles))
        return (out,)


# ============================================================================
class StereoForgeCompose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE", {"tooltip": "The working image; copied into its eye bit-exactly."}),
                "target_eye": ("IMAGE",),
                "stereo_gen": ("STEREO_GEN",),
                "layout": (["parallel (L|R)", "cross-eyed (R|L)"], {"default": "parallel (L|R)"}),
                "grain_match": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
            },
            "optional": {"mask": ("MASK", {"tooltip": "Regions that were synthesised (grain is matched there)."})},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("side_by_side", "left_eye", "right_eye", "anaglyph_preview", "generated_eye")
    FUNCTION = "run"
    CATEGORY = "StereoForge"

    def run(self, source, target_eye, stereo_gen, layout, grain_match, seed, mask=None):
        src = source[:1, ..., :3]
        tgt = target_eye[:1, ..., :3].to(src)
        _, H, W, _ = src.shape
        if tgt.shape[1:3] != (H, W):
            _log("target eye size mismatch; resizing to source size")
            tgt = sc.to_bhwc(sc.resize(sc.to_bchw(tgt), H, W, "bicubic"))
        if grain_match and mask is not None:
            mk = mask[:1].to(src)
            if mk.shape[-2:] != (H, W):
                mk = F.interpolate(mk[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
            s_src = sc.noise_sigma(sc.to_bchw(src))
            s_tgt = sc.noise_sigma(sc.to_bchw(tgt), (mk > 0.5)[:, None])
            add = math.sqrt(max(0.0, s_src ** 2 - s_tgt ** 2))
            if add > 1e-4:
                g = torch.Generator().manual_seed(seed)
                n = torch.randn((1, H, W, 1), generator=g).to(src) * add
                tgt = (tgt + n * mk[..., None]).clamp(0, 1)
                _log(f"grain match: source sigma {s_src:.4f}, synthesized {s_tgt:.4f}, added {add:.4f}")
        # quantise the synthesised eye exactly like the saved source will be
        tgt = torch.round(tgt * 255.0) / 255.0
        if stereo_gen["eye"] == "right":
            left, right = src, tgt
        else:
            left, right = tgt, src
        if layout.startswith("parallel"):
            sbs = torch.cat([left, right], dim=2)
        else:
            sbs = torch.cat([right, left], dim=2)
        ana = sc.to_bhwc(sc.anaglyph(sc.to_bchw(left), sc.to_bchw(right)))
        return (sbs, left, right, ana, tgt)


NODE_CLASS_MAPPINGS = {
    "StereoForgeLoadModel": StereoForgeLoadModel,
    "StereoForgePrepareInput": StereoForgePrepareInput,
    "StereoForgeGenerateView": StereoForgeGenerateView,
    "StereoForgeLiftToFullRes": StereoForgeLiftToFullRes,
    "StereoForgeTiledMaskedRefine": StereoForgeTiledMaskedRefine,
    "StereoForgeCompose": StereoForgeCompose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StereoForgeLoadModel": "StereoForge: Load StereoSpace",
    "StereoForgePrepareInput": "StereoForge: Prepare Input (optional downscale)",
    "StereoForgeGenerateView": "StereoForge: Generate Opposite Eye (StereoSpace)",
    "StereoForgeLiftToFullRes": "StereoForge: Lift to Full Resolution",
    "StereoForgeTiledMaskedRefine": "StereoForge: Tiled Masked Refine",
    "StereoForgeCompose": "StereoForge: Compose Side-by-Side",
}
