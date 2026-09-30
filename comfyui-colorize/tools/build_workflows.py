"""Builds the drag-and-drop ComfyUI workflow files in ../workflows.

Widget order is taken from a running ComfyUI's /object_info (saved to a JSON
file), so every value lands in the right slot:

    python build_workflows.py object_info.json

Only needed if you want to change the workflows; the generated .json files
are committed.
"""

import json
import os
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "workflows")

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}

HF21 = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main"
MODELS = {
    "qwen_image_2.1_bf16.safetensors": (f"{HF21}/diffusion_models/qwen_image_2.1_bf16.safetensors", "diffusion_models"),
    "qwen3vl_8b_int8_convrot.safetensors": (f"{HF21}/text_encoders/qwen3vl_8b_int8_convrot.safetensors", "text_encoders"),
    "qwen_image_2.1_vae_bf16.safetensors": (f"{HF21}/vae/qwen_image_2.1_vae_bf16.safetensors", "vae"),
    "qwen_image_edit_2511_fp8mixed.safetensors": ("https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors", "diffusion_models"),
    "qwen_2.5_vl_7b_fp8_scaled.safetensors": ("https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "text_encoders"),
    "qwen_image_vae.safetensors": ("https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors", "vae"),
    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors": ("https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "loras"),
    "scunet_color_real_psnr.pth": ("https://huggingface.co/deepinv/scunet/resolve/main/scunet_color_real_psnr.pth", "upscale_models"),
}

COLORS = {"input": "#335", "restore": "#533", "prep": "#353", "model": "#3f3f5f", "finish": "#543", "hint": "#454"}
GROUP_COLORS = {"input": "#3f789e", "restore": "#a1309b", "prep": "#8A8", "hint": "#b58b2a", "model": "#88A", "finish": "#b06634"}


class Graph:
    def __init__(self, info):
        self.info = info
        self.nodes = []
        self.links = []
        self.groups = []
        self._id = 0
        self._link = 0

    def node(self, type_, pos, size=None, title=None, widgets=None, mode=0, extra_inputs=(), color=None, widget_list=None):
        self._id += 1
        spec = self.info[type_] if type_ in self.info else None
        inputs, wvals = [], []
        if spec is not None:
            order = spec.get("input_order", {})
            all_in = {**spec["input"].get("required", {}), **spec["input"].get("optional", {})}
            optional = set(spec["input"].get("optional", {}))
            for name in order.get("required", []) + order.get("optional", []):
                t, opts = all_in[name][0], (all_in[name][1] if len(all_in[name]) > 1 else {})
                if t == "COMFY_AUTOGROW_V3":
                    continue
                is_widget = isinstance(t, list) or t in WIDGET_TYPES
                if opts.get("forceInput") or opts.get("socketless") and not is_widget:
                    is_widget = False
                if opts.get("socketless"):
                    continue
                if is_widget:
                    if widgets and name in widgets:
                        v = widgets[name]
                    elif "default" in opts:
                        v = opts["default"]
                    elif isinstance(t, list):
                        v = t[0] if t else ""
                    elif t == "COMBO":
                        v = opts.get("options", [""])[0]
                    else:
                        v = {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}[t]
                    wvals.append(v)
                    if opts.get("control_after_generate") or name in ("seed", "noise_seed"):
                        wvals.append((widgets or {}).get("control_after_generate", "randomize"))
                    if opts.get("image_upload"):
                        wvals.append("image")
                else:
                    inp = {"localized_name": name, "name": name, "type": t, "link": None}
                    if name in optional:
                        inp["shape"] = 7
                    inputs.append(inp)
            outputs = [{"localized_name": n, "name": n, "type": t, "links": []}
                       for t, n in zip(spec["output"], spec.get("output_name", spec["output"]))]
        else:
            outputs = []
        for name, t in extra_inputs:
            inputs.append({"localized_name": name.split(".")[-1], "name": name, "type": t, "shape": 7, "link": None})
        if widget_list is not None:
            wvals = widget_list
        n = {"id": self._id, "type": type_, "pos": list(pos), "size": list(size or [320, 120]), "flags": {},
             "order": self._id, "mode": mode, "inputs": inputs, "outputs": outputs,
             "properties": {"Node name for S&R": type_}, "widgets_values": wvals}
        if title:
            n["title"] = title
        if color:
            n["color"] = color
        models = [m for m in (widgets or {}).values() if isinstance(m, str) and m in MODELS]
        if models:
            n["properties"]["models"] = [{"name": m, "url": MODELS[m][0], "directory": MODELS[m][1]} for m in models]
        self.nodes.append(n)
        return n

    def note(self, pos, size, text, title="Note"):
        self._id += 1
        n = {"id": self._id, "type": "MarkdownNote", "pos": list(pos), "size": list(size), "flags": {},
             "order": self._id, "mode": 0, "inputs": [], "outputs": [], "title": title,
             "properties": {}, "widgets_values": [text], "color": "#432", "bgcolor": "#653"}
        self.nodes.append(n)
        return n

    def link(self, src, src_slot, dst, dst_input):
        if isinstance(src_slot, str):
            src_slot = next(i for i, o in enumerate(src["outputs"]) if o["name"] == src_slot)
        out = src["outputs"][src_slot]
        idx = next((i for i, inp in enumerate(dst["inputs"]) if inp["name"] == dst_input), None)
        if idx is None:
            # a widget fed by a link becomes an input socket that refers to its widget
            spec = self.info[dst["type"]]
            all_in = {**spec["input"].get("required", {}), **spec["input"].get("optional", {})}
            t = all_in[dst_input][0]
            dst["inputs"].append({"localized_name": dst_input, "name": dst_input,
                                  "type": "COMBO" if isinstance(t, list) else t,
                                  "widget": {"name": dst_input}, "link": None})
            idx = len(dst["inputs"]) - 1
        self._link += 1
        dst["inputs"][idx]["link"] = self._link
        out["links"].append(self._link)
        self.links.append([self._link, src["id"], src_slot, dst["id"], idx, out["type"]])

    def group(self, title, bounding, kind):
        self.groups.append({"id": len(self.groups) + 1, "title": title, "bounding": list(bounding),
                            "color": GROUP_COLORS[kind], "font_size": 24, "flags": {}})

    def dump(self, path, ds=None):
        for n in self.nodes:
            for o in n["outputs"]:
                if not o["links"]:
                    o["links"] = None
        wf = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, os.path.basename(path))), "revision": 0,
              "last_node_id": self._id, "last_link_id": self._link, "nodes": self.nodes, "links": self.links,
              "groups": self.groups, "definitions": {"subgraphs": []}, "config": {},
              "extra": {"ds": ds or {"scale": 0.55, "offset": [80, 120]}}, "version": 0.4}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wf, f, indent=1, ensure_ascii=False)
        print("wrote", path, len(self.nodes), "nodes", len(self.links), "links")


# --------------------------------------------------------------------------- shared parts

def load_and_restore(g, x0=0, y0=0):
    photo = g.node("LoadImage", (x0 + 20, y0 + 70), (340, 420),
                   title="Black-and-white photo (right-click > Open in MaskEditor to paint a region hint)",
                   widgets={"image": "example.png"})
    g.group("1 · Photo", (x0, y0, 380, 520), "input")

    rx = x0 + 420
    B = 4  # bypassed until you switch the group on
    up = g.node("UpscaleModelLoader", (rx + 20, y0 + 70), (330, 60), title="Grain remover model (SCUNet)",
                widgets={"model_name": "scunet_color_real_psnr.pth"}, mode=B)
    den = g.node("ImageUpscaleWithModel", (rx + 20, y0 + 180), (330, 50), title="Remove grain / noise", mode=B)
    blend = g.node("ImageBlend", (rx + 20, y0 + 280), (330, 110), title="Grain removal strength (blend_factor)",
                   widgets={"blend_factor": 0.8, "blend_mode": "normal"}, mode=B)
    dm = g.node("LumaLockDefectMask", (rx + 380, y0 + 70), (330, 200), title="Detect dust & scratches", mode=B)
    prev = g.node("PreviewImage", (rx + 380, y0 + 300), (330, 330), title="Detected damage shown in red — check this", mode=B)
    fill = g.node("LumaLockFillDefects", (rx + 740, y0 + 70), (330, 90), title="Fill dust & scratches", mode=B)
    tone = g.node("LumaLockTone", (rx + 740, y0 + 200), (330, 200), title="Restore faded contrast", mode=B)
    g.link(photo, 0, den, "image")
    g.link(up, 0, den, "upscale_model")
    g.link(photo, 0, blend, "image1")
    g.link(den, 0, blend, "image2")
    g.link(photo, 0, dm, "image")  # detect on the untouched scan; denoising makes texture look like specks
    g.link(dm, "preview", prev, "images")
    g.link(blend, 0, fill, "image")
    g.link(dm, "mask", fill, "mask")
    g.link(fill, 0, tone, "image")
    g.group("2 · Restoration of scanned prints (OPTIONAL — off by default: click the group title, then Ctrl+B to switch on)",
            (rx, y0, 1090, 660), "restore")
    g.note((rx + 20, y0 + 420), (330, 220),
           "**Restoration (optional)**\n\nEverything in this purple group is *bypassed* (purple) until you switch it on. "
           "Click the group title and press **Ctrl+B**, or select single nodes and press Ctrl+B to use only some steps.\n\n"
           "- **Grain**: SCUNet denoiser, blended back by *blend_factor* (0.8 = 80% denoised).\n"
           "- **Dust & scratches**: check the red preview. Detection is deliberately conservative. Raise *sensitivity* or *max_width_px* if damage is missed; lower them if catchlights, hair highlights, patterned fabric or lettering turn red. For tears or big stains, paint them yourself (MaskEditor on the photo) and connect that mask to *extra_mask*. "
           "Choose *big-lama.pt* in the fill node for tears.\n"
           "- **Faded contrast**: one global tone curve; cannot move edges or detail.",
           title="How to use restoration")

    px = rx + 1130
    grey = g.node("LumaLockToGray", (px + 20, y0 + 70), (300, 60), title="Neutral grey (removes sepia / scan cast)")
    g.link(tone, 0, grey, "image")
    g.group("3 · Prepare", (px, y0, 340, 170), "prep")
    return photo, grey, px + 380


def finish(g, photo, grey, colour, x0, y0, merge_widgets=None, ref=None):
    merge = g.node("LumaLockMerge", (x0 + 20, y0 + 70), (340, 230),
                   title="Keep original brightness, take only colour",
                   widgets=merge_widgets or {})
    g.link(grey, 0, merge, "original")
    g.link(colour, 0, merge, "colour_source")
    mood = g.node("LumaLockReferenceMood", (x0 + 20, y0 + 340), (340, 110),
                  title="Colour mood from reference photo (does nothing without one)", widgets={"strength": 0.4})
    g.link(merge, 0, mood, "image")
    if ref is not None:
        g.link(ref, 0, mood, "reference")
    fin = g.node("LumaLockColorFinish", (x0 + 20, y0 + 490), (340, 160),
                 title="White balance & saturation")
    g.link(mood, 0, fin, "image")
    hint = g.node("LumaLockRegionHint", (x0 + 400, y0 + 70), (340, 220),
                  title="Region colour hint (uses the mask painted on the photo; nothing painted = no change)")
    g.link(fin, 0, hint, "image")
    g.link(photo, "MASK", hint, "mask")
    save = g.node("SaveImageAdvanced", (x0 + 400, y0 + 330), (340, 330),
                  title="Save full-resolution result (16-bit PNG)",
                  widget_list=["LumaLock_colourised", "png", "16-bit", "sRGB"])
    g.link(hint, 0, save, "images")
    cmp_ = g.node("ImageCompare", (x0 + 780, y0 + 70), (620, 600), title="Before / after (drag the slider)")
    g.link(grey, 0, cmp_, "image_a")
    g.link(hint, 0, cmp_, "image_b")
    g.group("6 · Lock luminance, finish, save", (x0, y0, 1420, 700), "finish")
    return merge


def reference_and_prompt(g, x0, y0, family):
    ref = g.node("LoadImage", (x0 + 20, y0 + 70), (300, 330),
                 title="OPTIONAL colour-reference photo (bypassed; select + Ctrl+B to use)",
                 widgets={"image": "example.png"}, mode=4)
    prompt = g.node("LumaLockColorizePrompt", (x0 + 350, y0 + 70), (380, 330), title="Colourise instruction + colour hints",
                    widgets={"model_family": family})
    g.link(ref, 0, prompt, "reference")
    g.group("4 · Colour hints & reference (optional)", (x0, y0, 750, 430), "hint")
    return ref, prompt


# --------------------------------------------------------------------------- workflows

def build_qwen21(info):
    g = Graph(info)
    photo, grey, x = load_and_restore(g)
    ref, prompt = reference_and_prompt(g, 0, 720, "Qwen-Image 2.1")

    mx, my = 790, 720
    unet = g.node("UNETLoader", (mx + 20, my + 70), (360, 90), widgets={"unet_name": "qwen_image_2.1_bf16.safetensors"})
    clip = g.node("CLIPLoader", (mx + 20, my + 200), (360, 110),
                  widgets={"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image"})
    vae = g.node("VAELoader", (mx + 20, my + 350), (360, 60), widgets={"vae_name": "qwen_image_2.1_vae_bf16.safetensors"})
    enc = g.node("TextEncodeQwenImage21", (mx + 410, my + 70), (400, 300), title="Encode photo + instruction",
                 widgets={"negative_prompt": "", "resolution": 1536},
                 extra_inputs=[("images.image_1", "IMAGE"), ("images.image_2", "IMAGE"), ("images.image_3", "IMAGE")])
    ks = g.node("KSampler", (mx + 840, my + 70), (300, 270),
                widgets={"seed": 20250930, "control_after_generate": "fixed", "steps": 30, "cfg": 1.0,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0})
    dec = g.node("VAEDecode", (mx + 840, my + 380), (300, 50))
    raw = g.node("PreviewImage", (mx + 1170, my + 70), (300, 330), title="Raw model output (colour source only)")
    g.link(clip, 0, enc, "clip")
    g.link(vae, 0, enc, "vae")
    g.link(prompt, 0, enc, "prompt")
    g.link(grey, 0, enc, "images.image_1")
    g.link(ref, 0, enc, "images.image_2")
    g.link(unet, 0, ks, "model")
    g.link(enc, "positive", ks, "positive")
    g.link(enc, "negative", ks, "negative")
    g.link(enc, "latent", ks, "latent_image")
    g.link(ks, 0, dec, "samples")
    g.link(vae, 0, dec, "vae")
    g.link(dec, 0, raw, "images")
    g.group("5 · Colourise with Qwen-Image 2.1 (generates colour at ~2.4 MP; brightness is replaced by the original's in step 6)",
            (mx, my, 1500, 450), "model")
    finish(g, photo, grey, dec, 0, 1190, ref=ref)
    g.note((1460, 1190 + 70), (600, 610), GUIDE_QWEN21, title="Read me")
    g.dump(os.path.join(OUT, "LumaLock_Colourise_QwenImage21.json"))


def build_qwen2511(info):
    g = Graph(info)
    photo, grey, x = load_and_restore(g)
    ref, prompt = reference_and_prompt(g, 0, 720, "Qwen-Image-Edit 2509/2511")

    mx, my = 790, 720
    unet = g.node("UNETLoader", (mx + 20, my + 70), (360, 90), widgets={"unet_name": "qwen_image_edit_2511_fp8mixed.safetensors"})
    lora = g.node("LoraLoaderModelOnly", (mx + 20, my + 190), (360, 90),
                  title="Lightning 4-step LoRA (bypassed; to use: Ctrl+B, then steps 4, cfg 1)",
                  widgets={"lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "strength_model": 1.0}, mode=4)
    msa = g.node("ModelSamplingAuraFlow", (mx + 20, my + 310), (360, 60), widgets={"shift": 3.1})
    cfgn = g.node("CFGNorm", (mx + 20, my + 400), (360, 60), widgets={"strength": 1.0})
    clip = g.node("CLIPLoader", (mx + 410, my + 70), (360, 110),
                  widgets={"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"})
    vae = g.node("VAELoader", (mx + 410, my + 210), (360, 60), widgets={"vae_name": "qwen_image_vae.safetensors"})
    scale = g.node("LumaLockScaleForEdit", (mx + 410, my + 300), (360, 90), title="Scale to 1 MP (keeps aspect ratio)",
                   widgets={"megapixels": 1.0, "multiple_of": 8})
    pos = g.node("TextEncodeQwenImageEditPlus", (mx + 800, my + 70), (330, 150), title="Encode (positive)")
    neg = g.node("TextEncodeQwenImageEditPlus", (mx + 800, my + 250), (330, 150), title="Encode (negative, empty)",
                 widgets={"prompt": ""})
    mp = g.node("FluxKontextMultiReferenceLatentMethod", (mx + 1160, my + 70), (300, 60),
                widgets={"reference_latents_method": "index_timestep_zero"})
    mn = g.node("FluxKontextMultiReferenceLatentMethod", (mx + 1160, my + 160), (300, 60),
                widgets={"reference_latents_method": "index_timestep_zero"})
    venc = g.node("VAEEncode", (mx + 1160, my + 250), (300, 50))
    ks = g.node("KSampler", (mx + 1490, my + 70), (300, 270),
                widgets={"seed": 20250930, "control_after_generate": "fixed", "steps": 30, "cfg": 4.0,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0})
    dec = g.node("VAEDecode", (mx + 1490, my + 370), (300, 50))
    raw = g.node("PreviewImage", (mx + 1820, my + 70), (300, 330), title="Raw model output (colour source only)")
    g.link(unet, 0, lora, "model")
    g.link(lora, 0, msa, "model")
    g.link(msa, 0, cfgn, "model")
    g.link(grey, 0, scale, "image")
    for enc in (pos, neg):
        g.link(clip, 0, enc, "clip")
        g.link(vae, 0, enc, "vae")
        g.link(scale, 0, enc, "image1")
        g.link(ref, 0, enc, "image2")
    g.link(prompt, 0, pos, "prompt")
    g.link(pos, 0, mp, "conditioning")
    g.link(neg, 0, mn, "conditioning")
    g.link(scale, 0, venc, "pixels")
    g.link(vae, 0, venc, "vae")
    g.link(cfgn, 0, ks, "model")
    g.link(mp, 0, ks, "positive")
    g.link(mn, 0, ks, "negative")
    g.link(venc, 0, ks, "latent_image")
    g.link(ks, 0, dec, "samples")
    g.link(vae, 0, dec, "vae")
    g.link(dec, 0, raw, "images")
    g.group("5 · Colourise with Qwen-Image-Edit-2511 (fallback engine; brightness is replaced by the original's in step 6)",
            (mx, my, 2140, 460), "model")
    finish(g, photo, grey, dec, 0, 1200, ref=ref)
    g.note((1460, 1200 + 70), (600, 610), GUIDE_QWEN2511, title="Read me")
    g.dump(os.path.join(OUT, "LumaLock_Colourise_QwenImageEdit2511.json"))


def build_ddcolor(info):
    g = Graph(info)
    photo, grey, x = load_and_restore(g)
    mx, my = 0, 720
    dd = g.node("DDColor_Colorize", (mx + 20, my + 70), (340, 110), title="DDColor (automatic, no prompt)",
                widgets={"model_input_size": 512, "checkpoint": "ddcolor_modelscope.pth"})
    down = g.node("LumaLockScaleForEdit", (mx + 20, my + 220), (340, 90),
                  title="Back to model resolution (removes DDColor's blocky colour upscale)",
                  widgets={"megapixels": 0.25, "multiple_of": 1})
    g.link(grey, 0, dd, "image")
    g.link(dd, 0, down, "image")
    g.group("5 · Colourise with DDColor (fast draft engine)", (mx, my, 380, 350), "model")
    finish(g, photo, grey, down, 420, 720, merge_widgets={"alignment": "off"})
    g.note((1880, 720 + 70), (520, 520), GUIDE_DDCOLOR, title="Read me")
    g.dump(os.path.join(OUT, "LumaLock_Colourise_DDColor_fast.json"))


GUIDE_COMMON = """
**How it works.** The model only proposes colour. Step 6 throws away the model's brightness and detail, keeps its colour (CIE a\\*b\\*), lines it up with your photo to fix any small shift or zoom, snaps colour edges to your photo's edges, and puts it under your photo's **exact** original lightness at full resolution. Faces, text and texture therefore come from your original pixels.

**Steps**
1. Load your photo in *Black-and-white photo*.
2. (Optional) switch on group 2 for grain, dust/scratches and faded contrast.
3. (Optional) type colour hints, e.g. `the woman's dress is deep red, the car is pale blue`.
4. (Optional) paint a mask on the photo (right-click > *Open in MaskEditor*) and pick a colour in *Region colour hint*.
5. (Optional) load a colour-reference photo and press Ctrl+B on it to switch it on.
6. Click **Run**. Try another *seed* in the KSampler for a different colour interpretation.

**Tuning**
- Too strong/weak colour: *saturation* / *vibrance* in *White balance & saturation*; *max_chroma* stops neon patches.
- Yellow/sepia cast left: raise *white_balance* (1.0 = full neutralisation). Lower it for sunsets and warm interiors.
- Colour bleeding over edges: raise *edge_snap* to 1.0 in *Keep original brightness…*.
"""

GUIDE_QWEN21 = "## LumaLock colourisation · Qwen-Image 2.1\n" + GUIDE_COMMON + """
**Speed/quality**: *resolution* in *Encode photo + instruction* is the size the colour is generated at (1536 ≈ 2.4 MP). 1024 is faster; 2048 is the model's maximum. Your output is always full resolution. 30 steps is a good default; the official pipeline uses 40–50.
"""

GUIDE_QWEN2511 = "## LumaLock colourisation · Qwen-Image-Edit-2511\n" + GUIDE_COMMON + """
Use this version if your ComfyUI does not have the *Text Encode Qwen Image 2.1* node yet.

**Speed**: 30 steps at cfg 4 by default. For about 8x faster runs, select the Lightning LoRA node, press Ctrl+B, then set steps 4 and cfg 1 in the KSampler.
"""

GUIDE_DDCOLOR = "## LumaLock colourisation · DDColor (fast)\n" + GUIDE_COMMON.replace(
    "3. (Optional) type colour hints, e.g. `the woman's dress is deep red, the car is pale blue`.\n", "").replace(
    "5. (Optional) load a colour-reference photo and press Ctrl+B on it to switch it on.\n", "").replace(
    " Try another *seed* in the KSampler for a different colour interpretation.", "") + """
DDColor is a fast, automatic colouriser (seconds per photo, no prompt). Its colours are less accurate than Qwen's and it tends to be muted, but it is a handy quick draft. Try *ddcolor_artistic.pth* for more vivid colour. Needs the **ComfyUI-DDColor** node pack (by kijai) from the Manager.
"""


if __name__ == "__main__":
    info = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "object_info.json")))
    os.makedirs(OUT, exist_ok=True)
    build_qwen21(info)
    build_qwen2511(info)
    build_ddcolor(info)
