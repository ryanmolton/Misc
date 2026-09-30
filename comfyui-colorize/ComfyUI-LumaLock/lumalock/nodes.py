import logging
import math
import os

import torch
import torch.nn.functional as F

from . import align as _align
from .color import (
    L_to_gray_rgb, gamut_map, guided_coeffs, gaussian_blur, hex_to_rgb01, lab_to_linear,
    linear_to_lab, resize, rgb_to_L, rgb_to_lab, row_chunks, srgb_to_linear, to_bchw, to_bhwc,
)
from . import restore as _restore

log = logging.getLogger("LumaLock")

try:
    import comfy.model_management as mm
    import folder_paths
except ImportError:  # running outside ComfyUI (tests)
    mm = None
    folder_paths = None

CATEGORY = "LumaLock colourise"
FAST_FILL = "fast fill (no model)"

if folder_paths is not None and "inpaint" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["inpaint"] = (
        [os.path.join(folder_paths.models_dir, "inpaint")], folder_paths.supported_pt_extensions)


def _device():
    return mm.get_torch_device() if mm is not None else torch.device("cpu")


def _interp_rows(low, H, W, y0, y1):
    """Rows y0..y1 of F.interpolate(low, (H, W), bilinear, align_corners=False),
    without materialising the full-resolution tensor."""
    dev = low.device
    xs = (torch.arange(W, device=dev, dtype=low.dtype) + 0.5) / W * 2 - 1
    ys = (torch.arange(y0, y1, device=dev, dtype=low.dtype) + 0.5) / H * 2 - 1
    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij")[::-1], -1).unsqueeze(0)
    return F.grid_sample(low, grid.expand(low.shape[0], -1, -1, -1), mode="bilinear",
                         padding_mode="border", align_corners=False)


def _mask_bchw(mask, h, w, dev):
    if mask is None:
        return None
    m = mask
    if m.dim() == 2:
        m = m.unsqueeze(0)
    m = m[:1].unsqueeze(1).float().to(dev)
    return resize(m, h, w, "bilinear").clamp(0, 1)


def _stats_image(x, max_pixels=1_000_000):
    h, w = x.shape[-2:]
    s = min(1.0, math.sqrt(max_pixels / (h * w)))
    return resize(x, max(8, round(h * s)), max(8, round(w * s)), "area")


# ---------------------------------------------------------------- preparation

class LumaLockToGray:
    """Convert any scan (sepia, cast, RGB-scanned B&W) to a neutral grey image
    with exactly the same perceptual lightness (CIE L*)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grey",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image):
        dev = _device()
        out = []
        for i in range(image.shape[0]):
            x = to_bchw(image[i:i + 1])
            H, W = x.shape[-2:]
            res = torch.empty(1, 3, H, W)
            for y0, y1 in row_chunks(H, W):
                res[..., y0:y1, :] = L_to_gray_rgb(rgb_to_L(x[..., y0:y1, :].to(dev))).cpu()
            out.append(to_bhwc(res))
        return (torch.cat(out, 0),)


class LumaLockTone:
    """Restoration: fix faded contrast with one global, monotonic tone curve
    (black/white points, midtones, S-curve). Cannot move edges or change detail."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "black_clip_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.05,
                                             "tooltip": "Percent of darkest pixels allowed to become pure black."}),
            "white_clip_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.05,
                                             "tooltip": "Percent of brightest pixels allowed to become pure white."}),
            "levels_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "How much of the black/white point stretch to apply."}),
            "midtones": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01,
                                   "tooltip": ">1 brightens midtones, <1 darkens them."}),
            "contrast": ("FLOAT", {"default": 0.15, "min": -1.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "Gentle S-curve. 0 = off."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grey",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, black_clip_percent, white_clip_percent, levels_strength, midtones, contrast):
        dev = _device()
        out = []
        for i in range(image.shape[0]):
            L = rgb_to_L(to_bchw(image[i:i + 1]).to(dev))
            L = _restore.tone_curve(L, black_clip_percent, white_clip_percent, levels_strength, midtones, contrast)
            out.append(to_bhwc(L_to_gray_rgb(L).cpu()))
        return (torch.cat(out, 0),)


class LumaLockScaleForEdit:
    """Resize to a pixel budget keeping the aspect ratio, with sides rounded to
    a multiple. Matches what Qwen-Image-Edit's encoder does internally, so the
    output canvas lines up with the reference and the edit does not drift."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
            "multiple_of": ("INT", {"default": 8, "min": 1, "max": 64}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, megapixels, multiple_of):
        x = to_bchw(image)
        h, w = x.shape[-2:]
        s = math.sqrt(megapixels * 1024 * 1024 / (h * w))
        nw = max(multiple_of, round(w * s / multiple_of) * multiple_of)
        nh = max(multiple_of, round(h * s / multiple_of) * multiple_of)
        if nh < h:
            y = F.interpolate(x, size=(nh, nw), mode="area")
        else:
            y = F.interpolate(x, size=(nh, nw), mode="bicubic", align_corners=False)
        return (to_bhwc(y),)


class LumaLockColorizePrompt:
    """Builds the edit instruction for the colourisation model."""

    LOOKS = {
        "modern DSLR, neutral daylight": "neutral daylight white balance, like a photo taken today on a modern full-frame digital camera",
        "modern DSLR, warm golden hour": "warm late-afternoon sunlight, like a photo taken today on a modern full-frame digital camera",
        "modern DSLR, overcast soft light": "soft overcast daylight with neutral white balance, like a photo taken today on a modern full-frame digital camera",
        "modern DSLR, indoor available light": "natural indoor light with correctly balanced white balance, like a photo taken today on a modern full-frame digital camera",
        "true-to-period colour film": "the natural colour rendering of good quality colour film of the period, without fading or colour casts",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_family": (["Qwen-Image 2.1", "Qwen-Image-Edit 2509/2511"],
                             {"tooltip": "How the prompt refers to the input images: <image1> for Qwen-Image 2.1, 'Picture 1' for Qwen-Image-Edit."}),
            "look": (list(cls.LOOKS.keys()),),
            "colour_hints": ("STRING", {"multiline": True, "default": "",
                                        "tooltip": "Optional. One hint per line or comma-separated, e.g. 'the woman's dress is deep red', 'the car is pale blue'."}),
            "scene_notes": ("STRING", {"multiline": False, "default": "",
                                       "tooltip": "Optional context, e.g. '1950s seaside holiday, England'."}),
        }, "optional": {
            "reference": ("IMAGE", {"tooltip": "Connect the same colour-reference photo that goes into image_2 of the encoder."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model_family, look, colour_hints, scene_notes, reference=None):
        qwen21 = model_family == "Qwen-Image 2.1"
        img1, img2 = ("<image1>", "<image2>") if qwen21 else ("Picture 1", "Picture 2")
        tag = f"the {img1}" if reference is not None else "this"
        p = (f"Colorize {tag} black-and-white photograph into a realistic full-colour photograph with "
             f"{self.LOOKS[look]}. Give every object its own natural, plausible and varied colour; "
             "realistic, healthy skin tones with natural variation between people; true whites and neutral greys; "
             "clean colour separation between neighbouring objects with no colour bleeding; natural, "
             "moderate saturation. Keep the composition, framing, every face, expression, texture, "
             "text and fine detail exactly as they are; change only the colours. "
             "No sepia, no yellow or brown tint, no colour cast, no hand-tinted look, no oversaturation.")
        if scene_notes.strip():
            p += f" Context: {scene_notes.strip()}."
        hints = [h.strip(" .") for h in colour_hints.replace("\n", ",").split(",") if h.strip(" .")]
        if hints:
            p += " Specific colours: " + "; ".join(hints) + "."
        if reference is not None:
            p += (f" Use {img2} only as a reference for the colour palette, white balance and mood; "
                  f"do not copy any content, objects or composition from {img2}.")
        return (p,)


# ---------------------------------------------------------------- the core merge

class LumaLockMerge:
    """Take ONLY the colour (CIE a*b* chroma) from the model's output and put it
    under the original photo's lightness at full resolution. Faces, text and
    texture come from the original pixels; the model cannot alter them.

    Steps: register the model output to the original (fixes small shifts or
    zoom), snap the low-resolution chroma to the original's edges with a
    joint guided filter (prevents colour bleeding), then recombine with the
    exact original L* and gamut-map by reducing chroma, never brightness."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original": ("IMAGE", {"tooltip": "Full-resolution photo whose brightness is kept (after optional restoration)."}),
            "colour_source": ("IMAGE", {"tooltip": "Colourised image from the model, any resolution."}),
            "alignment": (["affine", "shift+scale", "off"], {"default": "affine",
                          "tooltip": "Correct small shifts/zoom the model introduced before taking its colour."}),
            "edge_snap": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "Snap colour boundaries to edges in the original. 0 = plain upscale of the colour."}),
            "snap_radius": ("INT", {"default": 2, "min": 1, "max": 16,
                                    "tooltip": "Guided-filter radius in model-output pixels. Larger = smoother colour, stronger snapping."}),
            "snap_eps": ("FLOAT", {"default": 0.001, "min": 0.00001, "max": 0.1, "step": 0.0005,
                                   "tooltip": "Edge sensitivity. Smaller follows fainter edges."}),
            "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "aligned_colour_preview")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, original, colour_source, alignment, edge_snap, snap_radius, snap_eps, saturation):
        dev = _device()
        outs, previews = [], []
        for i in range(original.shape[0]):
            src = to_bchw(original[i:i + 1])
            H, W = src.shape[-2:]
            col = to_bchw(colour_source[min(i, colour_source.shape[0] - 1)].unsqueeze(0)).to(dev)
            h, w = col.shape[-2:]

            L_full = torch.empty(1, 1, H, W)
            for y0, y1 in row_chunks(H, W):
                L_full[..., y0:y1, :] = rgb_to_L(src[..., y0:y1, :].to(dev)).cpu()
            L_low = resize(L_full.to(dev), h, w, "area")

            lab_c = rgb_to_lab(col)
            if alignment != "off":
                theta, gain = _align.estimate(L_low, lab_c[:, :1], mode=alignment)
                lab_c = _align.warp(lab_c, theta)
                t = theta[0].tolist()
                log.info("LumaLock alignment: shift %.2f/%.2f px, scale %.4f/%.4f (fit %.2fx better)",
                         t[0][2] * w / 2, t[1][2] * h / 2, t[0][0], t[1][1], gain)
            ab = lab_c[:, 1:3] * saturation
            previews.append(to_bhwc(gamut_map(lab_c[:, :1], lab_c[:, 1:3]).cpu()))

            A, B = guided_coeffs(L_low / 100.0, ab, int(snap_radius), float(snap_eps))
            res = torch.empty(1, 3, H, W)
            for y0, y1 in row_chunks(H, W):
                Lc = L_full[..., y0:y1, :].to(dev)
                ab_up = _interp_rows(ab, H, W, y0, y1)
                if edge_snap > 0:
                    q = _interp_rows(A, H, W, y0, y1) * (Lc / 100.0) + _interp_rows(B, H, W, y0, y1)
                    ab_up = ab_up + (q - ab_up) * edge_snap
                res[..., y0:y1, :] = gamut_map(Lc, ab_up).cpu()
            outs.append(to_bhwc(res))
        return (torch.cat(outs, 0), torch.cat(previews, 0) if len({p.shape for p in previews}) == 1 else previews[0])


# ---------------------------------------------------------------- colour finishing

def _apply_lab_op(image, fn, dev):
    """Run fn(L, ab, lin) -> ab over row chunks of image, keeping L exact."""
    x = to_bchw(image)
    H, W = x.shape[-2:]
    res = torch.empty(1, 3, H, W)
    for y0, y1 in row_chunks(H, W):
        c = x[..., y0:y1, :].to(dev)
        lin = srgb_to_linear(c)
        lab = linear_to_lab(lin)
        L = lab[:, :1]
        res[..., y0:y1, :] = gamut_map(L, fn(L, lab[:, 1:3], lin)).cpu()
    return to_bhwc(res)


class LumaLockColorFinish:
    """White balance, saturation and a soft chroma ceiling, all without
    touching lightness. Removes leftover sepia/yellow casts and stops
    oversaturated patches."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "white_balance": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "Neutralise the cast of near-grey areas (walls, clothes, paper, sky haze). 0 = off."}),
            "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "vibrance": ("FLOAT", {"default": 0.15, "min": -1.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "Boosts muted colours more than already-strong ones."}),
            "max_chroma": ("FLOAT", {"default": 70.0, "min": 20.0, "max": 130.0, "step": 1.0,
                                     "tooltip": "Soft ceiling on CIE chroma. ~70 keeps things photographic; 130 = off."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, white_balance, saturation, vibrance, max_chroma):
        dev = _device()
        out = []
        for i in range(image.shape[0]):
            img = image[i:i + 1]
            gains = torch.ones(1, 3, 1, 1, device=dev)
            if white_balance > 0:
                small = _stats_image(to_bchw(img).to(dev))
                lin_s = srgb_to_linear(small)
                lab_s = linear_to_lab(lin_s)
                C = torch.hypot(lab_s[:, 1], lab_s[:, 2])
                L = lab_s[:, 0]
                valid = (L > 15) & (L < 97)
                if valid.sum() > 500:
                    thr = torch.quantile(C[valid].float(), 0.35)
                    sel = (valid & (C <= thr)).unsqueeze(1).float()
                    m = (lin_s * sel).sum((2, 3), keepdim=True) / sel.sum().clamp(min=1)
                    Y = (m * torch.tensor([0.2126729, 0.7151522, 0.0721750], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
                    g = Y / m.clamp(min=1e-6)
                    gains = (1 + (g - 1) * white_balance).clamp(0.5, 2.0)
                    log.info("LumaLock white balance gains: %s", [round(v, 3) for v in gains.flatten().tolist()])

            ceil = float(max_chroma)
            knee = 0.6 * ceil

            def op(L, ab, lin):
                if white_balance > 0:
                    ab = linear_to_lab(lin * gains)[:, 1:3]
                C = torch.sqrt((ab ** 2).sum(1, keepdim=True) + 1e-12)
                Cn = C * saturation
                if vibrance != 0:
                    Cn = Cn * (1 + vibrance * (1 - (Cn / 60.0).clamp(0, 1)))
                if ceil < 130:
                    over = (Cn - knee).clamp(min=0)
                    Cn = torch.where(Cn > knee, knee + (ceil - knee) * torch.tanh(over / (ceil - knee)), Cn)
                return ab * (Cn / C)

            out.append(_apply_lab_op(img, op, dev))
        return (torch.cat(out, 0),)


class LumaLockReferenceMood:
    """Transfer the colour mood of a reference photo (palette and balance in
    shadows, midtones and highlights separately). Lightness is untouched.
    With no reference connected the image passes through unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "match": (["palette + saturation", "palette only"],),
        }, "optional": {"reference": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    @staticmethod
    def _weights(L):
        sh = torch.sigmoid((35 - L) / 6)
        hi = torch.sigmoid((L - 70) / 6)
        mid = (1 - sh - hi).clamp(min=0)
        return [sh, mid, hi]

    def _stats(self, lab):
        L, ab = lab[:, :1], lab[:, 1:3]
        res = []
        for wgt in self._weights(L):
            s = wgt.sum().clamp(min=1e-3)
            mu = (ab * wgt).sum((2, 3), keepdim=True) / s
            sd = (((ab - mu) ** 2) * wgt).sum((2, 3), keepdim=True).div(s).sqrt()
            res.append((mu, sd.clamp(min=0.5)))
        return res

    def run(self, image, strength, match, reference=None):
        if reference is None or strength <= 0:
            return (image,)
        dev = _device()
        ref_stats = self._stats(rgb_to_lab(_stats_image(to_bchw(reference[:1]).to(dev), 500_000)))
        out = []
        for i in range(image.shape[0]):
            img = image[i:i + 1]
            tgt_stats = self._stats(rgb_to_lab(_stats_image(to_bchw(img).to(dev), 500_000)))

            def op(L, ab, lin):
                mapped = torch.zeros_like(ab)
                for wgt, (mt, st), (mr, sr) in zip(self._weights(L), tgt_stats, ref_stats):
                    ratio = (sr / st).clamp(0.5, 2.0) if match == "palette + saturation" else 1.0
                    mapped = mapped + wgt * ((ab - mt) * ratio + mr)
                return ab + (mapped - ab) * strength

            out.append(_apply_lab_op(img, op, dev))
        return (torch.cat(out, 0),)


class LumaLockRegionHint:
    """Force a colour inside a painted mask (e.g. paint the dress in the mask
    editor and pick red). The mask is snapped to the photo's edges, and only
    colour changes: shading, folds and texture stay from the original.
    An empty mask passes the image through unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "colour": ("STRING", {"default": "#9e1b24", "tooltip": "Target colour as #RRGGBB."}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "mode": (["colour", "hue only (keep model's saturation)"],),
            "feather": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.5}),
            "snap_to_edges": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, mask, colour, strength, mode, feather, snap_to_edges):
        if mask is None or float(mask.max()) < 1e-3 or strength <= 0:
            return (image,)
        dev = _device()
        rgb = torch.tensor(hex_to_rgb01(colour), device=dev).view(1, 3, 1, 1)
        t_ab = rgb_to_lab(rgb)[:, 1:3]
        out = []
        for i in range(image.shape[0]):
            x = to_bchw(image[i:i + 1])
            H, W = x.shape[-2:]
            m = _mask_bchw(mask[min(i, mask.shape[0] - 1)] if mask.dim() == 3 else mask, H, W, "cpu")
            if feather > 0:
                small = _stats_image(m, 4_000_000)
                sc = small.shape[-1] / W
                m = resize(gaussian_blur(small.to(dev), feather * sc), H, W).cpu()
            if snap_to_edges:
                small_img = _stats_image(x, 2_000_000).to(dev)
                sm = resize(m.to(dev), *small_img.shape[-2:], "area")
                Ls = rgb_to_L(small_img) / 100.0
                r = max(2, round(max(small_img.shape[-2:]) / 200))
                A, B = guided_coeffs(Ls, sm, r, 1e-3)
            res = torch.empty(1, 3, H, W)
            for y0, y1 in row_chunks(H, W):
                c = x[..., y0:y1, :].to(dev)
                lab = rgb_to_lab(c)
                L, ab = lab[:, :1], lab[:, 1:3]
                mm_ = m[..., y0:y1, :].to(dev)
                if snap_to_edges:
                    snapped = _interp_rows(A, H, W, y0, y1) * (L / 100.0) + _interp_rows(B, H, W, y0, y1)
                    mm_ = torch.minimum(snapped.clamp(0, 1), gaussian_blur(mm_, 1.0) * 1.5).clamp(0, 1)
                if mode == "colour":
                    target = t_ab.expand_as(ab)
                else:
                    C = torch.sqrt((ab ** 2).sum(1, keepdim=True) + 1e-12)
                    target = t_ab / torch.sqrt((t_ab ** 2).sum(1, keepdim=True) + 1e-12) * C
                ab = ab + (target - ab) * (mm_ * strength)
                res[..., y0:y1, :] = gamut_map(L, ab).cpu()
            out.append(to_bhwc(res))
        return (torch.cat(out, 0),)


# ---------------------------------------------------------------- restoration

class LumaLockDefectMask:
    """Restoration: find dust specks, hairs and thin scratches automatically.
    Check the preview before filling: raise max width for bigger scratches,
    lower sensitivity if real detail (catchlights, lettering) is picked up,
    or paint areas to protect into protect_mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "sensitivity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
            "max_width_px": ("INT", {"default": 3, "min": 1, "max": 30,
                                     "tooltip": "Widest defect to detect, in pixels of this image."}),
            "polarity": (["both", "bright (white specks & scratches)", "dark (black specks & hairs)"],),
            "grow_px": ("INT", {"default": 1, "min": 0, "max": 10}),
            "min_contrast": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01,
                                       "tooltip": "Ignore specks fainter than this (0..1 brightness difference)."}),
        }, "optional": {
            "extra_mask": ("MASK", {"tooltip": "Hand-painted damage to add (tears, stains)."}),
            "protect_mask": ("MASK", {"tooltip": "Areas that must never be touched (eyes, lettering)."}),
        }}

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, sensitivity, max_width_px, polarity, grow_px, min_contrast=0.1, extra_mask=None, protect_mask=None):
        dev = _device()
        masks, previews = [], []
        for i in range(image.shape[0]):
            x = to_bchw(image[i:i + 1]).to(dev)
            H, W = x.shape[-2:]
            Y = rgb_to_L(x) / 100.0
            m = _restore.detect_defects(Y, sensitivity, max_width_px, polarity, min_contrast=min_contrast, grow=grow_px)
            if extra_mask is not None and float(extra_mask.max()) > 0:
                m = torch.maximum(m, (_mask_bchw(extra_mask, H, W, dev) > 0.5).float())
            if protect_mask is not None and float(protect_mask.max()) > 0:
                m = m * (_mask_bchw(protect_mask, H, W, dev) < 0.5).float()
            red = torch.tensor([1.0, 0.1, 0.1], device=dev).view(1, 3, 1, 1)
            previews.append(to_bhwc((x * (1 - m) + red * m).cpu()))
            masks.append(m[:, 0].cpu())
            log.info("LumaLock defect mask covers %.3f%% of the image", 100 * float(m.mean()))
        return (torch.cat(masks, 0), torch.cat(previews, 0))


_lama_cache = {}


class LumaLockFillDefects:
    """Restoration: fill masked defects. 'fast fill' needs no model and is
    right for dust and thin scratches; choose big-lama for tears and missing
    areas (put big-lama.pt in models/inpaint)."""

    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("inpaint") if folder_paths is not None else []
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "method": ([FAST_FILL] + list(models),),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def _lama(self, name, dev):
        from spandrel import ModelLoader
        if name not in _lama_cache:
            _lama_cache.clear()
            _lama_cache[name] = ModelLoader().load_from_file(folder_paths.get_full_path_or_raise("inpaint", name)).eval()
        return _lama_cache[name].to(dev)

    def run(self, image, mask, method):
        if float(mask.max()) < 0.5:
            return (image,)
        dev = _device()
        model = None if method == FAST_FILL else self._lama(method, dev)
        out = []
        for i in range(image.shape[0]):
            x = to_bchw(image[i:i + 1]).to(dev)
            H, W = x.shape[-2:]
            m = _mask_bchw(mask[min(i, mask.shape[0] - 1)] if mask.dim() == 3 else mask, H, W, dev)
            with torch.no_grad():
                y = _restore.fast_fill(x, m) if model is None else _restore.lama_fill(model, x, m)
            out.append(to_bhwc(y.cpu()))
        if model is not None:
            model.to("cpu")
        return (torch.cat(out, 0),)


NODE_CLASS_MAPPINGS = {
    "LumaLockToGray": LumaLockToGray,
    "LumaLockTone": LumaLockTone,
    "LumaLockScaleForEdit": LumaLockScaleForEdit,
    "LumaLockColorizePrompt": LumaLockColorizePrompt,
    "LumaLockMerge": LumaLockMerge,
    "LumaLockColorFinish": LumaLockColorFinish,
    "LumaLockReferenceMood": LumaLockReferenceMood,
    "LumaLockRegionHint": LumaLockRegionHint,
    "LumaLockDefectMask": LumaLockDefectMask,
    "LumaLockFillDefects": LumaLockFillDefects,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LumaLockToGray": "LumaLock · Neutral Grey (remove sepia/cast)",
    "LumaLockTone": "LumaLock · Restore Tone (faded contrast)",
    "LumaLockScaleForEdit": "LumaLock · Scale For Edit Model",
    "LumaLockColorizePrompt": "LumaLock · Colourise Prompt",
    "LumaLockMerge": "LumaLock · Merge Colour Onto Original Luminance",
    "LumaLockColorFinish": "LumaLock · Colour Finish (white balance, saturation)",
    "LumaLockReferenceMood": "LumaLock · Reference Colour Mood",
    "LumaLockRegionHint": "LumaLock · Region Colour Hint (mask)",
    "LumaLockDefectMask": "LumaLock · Detect Dust & Scratches",
    "LumaLockFillDefects": "LumaLock · Fill Dust & Scratches",
}
