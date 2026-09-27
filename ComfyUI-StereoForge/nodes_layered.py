"""
StereoForge v2 nodes: layered depth-image-based stereo with physically based
floor-reflection handling.  Parallel side-by-side output.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils

from . import layered_stereo as ls
from . import stereo_core as sc

_CACHE = {}

DA3_MODELS = [
    "DA3NESTED-GIANT-LARGE-1.1",  # metric depth + intrinsics, best quality (CC BY-NC 4.0)
    "DA3-GIANT-1.1",
    "DA3-LARGE-1.1",
    "DA3METRIC-LARGE",  # Apache 2.0
]


def _log(msg):
    print(f"[StereoForge] {msg}")


def _to_pil(img_bhwc):
    from PIL import Image

    a = (img_bhwc[0, ..., :3].cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(a)


# ============================================================================
class StereoForgeDA3Depth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (DA3_MODELS, {"default": DA3_MODELS[0]}),
                "process_res": ("INT", {"default": 1008, "min": 504, "max": 2016, "step": 14,
                                        "tooltip": "Long side DA3 works at. The warp itself is always full resolution."}),
            }
        }

    RETURN_TYPES = ("STEREO_DEPTH", "IMAGE")
    RETURN_NAMES = ("stereo_depth", "depth_preview")
    FUNCTION = "run"
    CATEGORY = "StereoForge"

    def run(self, image, model, process_res):
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as e:
            raise RuntimeError("Depth Anything 3 is not installed - see the StereoForge README") from e
        dev = mm.get_torch_device()
        key = ("da3", model)
        if key not in _CACHE:
            for k in [k for k in _CACHE if k[0] == "da3"]:
                del _CACHE[k]
            _log(f"loading depth-anything/{model}")
            _CACHE[key] = DepthAnything3.from_pretrained(f"depth-anything/{model}").eval()
        net = _CACHE[key].to(dev)
        with torch.no_grad():
            pred = net.inference([_to_pil(image)], process_res=process_res)
        net.to("cpu")
        mm.soft_empty_cache()
        Z = np.asarray(pred.depth[0], np.float32)
        K = np.asarray(pred.intrinsics[0], np.float64)
        inv = 1.0 / np.maximum(Z, 1e-6)
        lo, hi = np.percentile(inv, 1), np.percentile(inv, 99)
        vis = np.clip((inv - lo) / max(hi - lo, 1e-9), 0, 1)
        prev = torch.from_numpy(np.repeat(vis[..., None], 3, -1))[None].float()
        return (dict(Z=Z, K=K, size=tuple(image.shape[1:3])), prev)


# ============================================================================
class StereoForgeReflectorMask:
    """Open-vocabulary reflector segmentation: Grounding DINO boxes -> SAM 2.1 masks -> GrabCut snap."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "puddle. sky reflected in water. mirror.", "multiline": False}),
                "box_threshold": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 0.9, "step": 0.01}),
                "grabcut_refine": ("BOOLEAN", {"default": True}),
            },
            "optional": {"extra_mask": ("MASK", {"tooltip": "Union with this mask (e.g. from SAM 3 or hand-painted)."})},
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("reflector_mask",)
    FUNCTION = "run"
    CATEGORY = "StereoForge"

    def run(self, image, prompt, box_threshold, grabcut_refine, extra_mask=None):
        from transformers import AutoProcessor, GroundingDinoForObjectDetection, Sam2Model, Sam2Processor

        dev = mm.get_torch_device()
        if "gdino" not in _CACHE:
            _CACHE["gdino"] = (AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base"),
                               GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").eval())
            _CACHE["sam2"] = (Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large"),
                              Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").eval())
        gp, gm = _CACHE["gdino"]
        spp, smm = _CACHE["sam2"]
        im = _to_pil(image)
        W, H = im.size
        boxes, scores = [], []
        gm.to(dev)
        for text in [t.strip() + "." for t in prompt.split(".") if t.strip()]:
            inp = gp(images=im, text=text, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = gm(**inp)
            r = gp.post_process_grounded_object_detection(out, inp.input_ids, threshold=box_threshold,
                                                          text_threshold=0.2, target_sizes=[(H, W)])[0]
            for s_, b in zip(r["scores"].tolist(), r["boxes"].tolist()):
                b = [max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])]
                if (b[2] - b[0]) * (b[3] - b[1]) > 0.9 * W * H:
                    continue  # a whole-frame box carries no localisation
                boxes.append(b)
                scores.append(s_)
        gm.to("cpu")
        mask = np.zeros((H, W), np.float32)
        if boxes:
            smm.to(dev)
            si = spp(images=im, input_boxes=[boxes], return_tensors="pt").to(dev)
            with torch.no_grad():
                so = smm(**si, multimask_output=False)
            ms = spp.post_process_masks(so.pred_masks.cpu(), si["original_sizes"].cpu())[0]
            smm.to("cpu")
            for k in range(len(boxes)):
                mask = np.maximum(mask, ms[k, 0].float().numpy())
        _log(f"reflector boxes: {len(boxes)}, coverage {100 * (mask > 0.5).mean():.1f}%")
        if grabcut_refine and mask.max() > 0.5 and ls.cv2 is not None:
            cv2 = ls.cv2
            img = np.asarray(im)
            gc = np.full((H, W), cv2.GC_PR_BGD, np.uint8)
            gc[mask > 0.5] = cv2.GC_PR_FGD
            k = max(3, int(round(min(H, W) / 40)) | 1)
            gc[cv2.dilate((mask > 0.5).astype(np.uint8), np.ones((k, k), np.uint8)) == 0] = cv2.GC_BGD
            bg, fg = np.zeros((1, 65)), np.zeros((1, 65))
            cv2.grabCut(np.ascontiguousarray(img[..., ::-1]), gc, None, bg, fg, 5, cv2.GC_INIT_WITH_MASK)
            mask = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.float32)
        mm.soft_empty_cache()
        m = torch.from_numpy(mask)[None]
        if extra_mask is not None:
            em = extra_mask[:1].float()
            if em.shape[-2:] != (H, W):
                em = F.interpolate(em[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
            m = torch.maximum(m, em.cpu())
        return (m,)


# ============================================================================
class StereoForgeLayeredRender:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "stereo_depth": ("STEREO_DEPTH",),
                "source_eye": (["left", "right"], {"default": "left",
                               "tooltip": "Which eye the input image is. The other eye is synthesised."}),
                "parallax_mode": (["percent_of_width", "metric_ipd"], {"default": "percent_of_width"}),
                "parallax": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 150.0, "step": 0.1,
                                       "tooltip": "percent_of_width: parallax of the nearest 2% of the scene, in % of image width. "
                                                  "metric_ipd: camera separation in mm (65 = human eyes; needs a metric DA3 model)."}),
                "reflections": ("BOOLEAN", {"default": True,
                                            "tooltip": "Layered rendering of floor reflections (puddles, wet floors) at their mirrored depth."}),
                "inpaint": (["lama", "none"], {"default": "lama"}),
                "hole_dilate_px": ("INT", {"default": 2, "min": 0, "max": 16}),
            },
            "optional": {
                "reflector_mask": ("MASK",),
                "reflection_layer": ("IMAGE", {"tooltip": "Optional reflection layer from a reflection-separation model."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("side_by_side", "generated_eye", "disoccluded", "reflection_preview", "anaglyph_preview")
    FUNCTION = "run"
    CATEGORY = "StereoForge"

    def run(self, image, stereo_depth, source_eye, parallax_mode, parallax, reflections, inpaint, hole_dilate_px,
            reflector_mask=None, reflection_layer=None):
        src = image[:1, ..., :3]
        _, H, W, _ = src.shape
        if tuple(stereo_depth["size"]) != (H, W):
            raise ValueError("stereo_depth was computed for a different image size")
        Z = stereo_depth["Z"]
        K = stereo_depth["K"].copy()
        h, w = Z.shape
        x = sc.to_bchw(src).float()
        region = None
        if reflector_mask is not None:
            region = F.interpolate(reflector_mask[:1, None].float(), size=(h, w), mode="bilinear",
                                   align_corners=False)[0, 0].cpu().numpy()
        R_sep = None
        if reflection_layer is not None:
            R_sep = F.interpolate(sc.to_bchw(reflection_layer[:1]).float(), size=(h, w), mode="bilinear",
                                  align_corners=False)[0].permute(1, 2, 0).cpu().numpy()
        mirror = source_eye == "right"
        if mirror:  # canonical form: source = left eye; mirror everything horizontally
            x = torch.flip(x, [-1])
            Z = Z[:, ::-1].copy()
            K[0, 2] = w - K[0, 2]
            region = region[:, ::-1].copy() if region is not None else None
            R_sep = R_sep[:, ::-1].copy() if R_sep is not None else None

        img_lr = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
        img_lr = img_lr.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        layers = None
        if reflections and region is not None:
            layers = ls.reflection_layers(img_lr, Z, K, region, R_sep)
            if layers["plane"] is None:
                _log("no floor plane found; reflections rendered as surface")
        if parallax_mode == "metric_ipd":
            bscale = parallax / 1000.0
        else:
            bscale = ls.choose_baseline_scale(Z, K, W, w, parallax / 100.0)
        inp = None
        if inpaint == "lama":
            if "lama" not in _CACHE:
                _CACHE["lama"] = ls.lama_inpaint_fn()
            inp = _CACHE["lama"]

        def go(dev):
            return ls.render_right_eye(x.to(dev), Z, K, layers, bscale, inp, hole_dilate_px)

        try:
            right, dbg = go(mm.get_torch_device())
        except torch.cuda.OutOfMemoryError:
            _log("GPU out of memory, rendering on CPU")
            mm.soft_empty_cache()
            right, dbg = go(torch.device("cpu"))
        holes = dbg["holes"]
        rho = dbg.get("rho", torch.zeros_like(holes))
        if mirror:
            right, holes, rho = (torch.flip(t, [-1]) for t in (right, holes, rho))
        gen = torch.round(right.float().cpu() * 255) / 255
        gen = sc.to_bhwc(gen)
        left_eye, right_eye = (gen, src) if mirror else (src, gen)
        sbs = torch.cat([left_eye, right_eye], dim=2)
        ana = sc.to_bhwc(sc.anaglyph(sc.to_bchw(left_eye), sc.to_bchw(right_eye)))
        # reflection preview: source tinted by the reflected fraction
        pv = sc.to_bchw(src).cpu().clone()
        r = rho.float().cpu()
        pv = pv * (1 - 0.6 * r) + torch.tensor([0.1, 0.6, 1.0]).view(1, 3, 1, 1) * 0.6 * r
        d = dbg["d_s"].float().cpu()
        _log(f"parallax: max {100 * float(d.max()) / W:.2f}% of width, "
             f"disoccluded {100 * float(holes.mean()):.2f}% of pixels, reflective {100 * float((r > 0.05).float().mean()):.1f}%")
        return (sbs, gen, holes[:, 0].float().cpu(), sc.to_bhwc(pv.clamp(0, 1)), ana)


NODE_CLASS_MAPPINGS = {
    "StereoForgeDA3Depth": StereoForgeDA3Depth,
    "StereoForgeReflectorMask": StereoForgeReflectorMask,
    "StereoForgeLayeredRender": StereoForgeLayeredRender,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "StereoForgeDA3Depth": "StereoForge: Depth Anything 3",
    "StereoForgeReflectorMask": "StereoForge: Reflector Mask (GroundingDINO + SAM2)",
    "StereoForgeLayeredRender": "StereoForge: Layered Stereo Render (parallel SBS)",
}
