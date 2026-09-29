"""Generate the shipped ComfyUI workflows (UI format, importable by drag & drop)
plus matching API-format prompts used by the automated tests.

    python tools/build_workflows.py            # writes ../workflows/*.json
    python tools/build_workflows.py --test     # also writes CPU test variants
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "workflows")

FLUX_PROMPT = ("The same photograph, with the background seamlessly continued behind the foreground: "
               "matching surfaces, texture, lighting, grain, focus and perspective. No new objects.")

NOTE = """StereoForge - single image -> side-by-side stereograph (2W x H)

1. Load your image (left). The source becomes the LEFT eye pixel-for-pixel
   (set source_eye = right on the Render node to make it the right eye).
2. Downscale is OFF by default (bit-exact pass-through).
3. Depth: MoGe-3 (native ComfyUI >= 0.37) -> 1/z disparity.
4. Render Opposite Eye: full-res z-buffered rendering, depth-aware Lanczos
   resampling, soft-edge (hair/fur) matting; only disocclusions are left.
5. Tiled Inpaint: FLUX.1 Fill regenerates just those regions at 1:1 pixel
   scale (1024 px tiles).
6. Compose: source eye untouched + generated eye -> side-by-side PNG.

Comfort: depth_budget_pct 2-3 (screens) / 3-5 (VR). stereo_window keeps
frame-edge objects behind the screen. Check the anaglyph preview with
red/cyan glasses or view the SBS cross/parallel."""


MODELS = {
    "flux1-fill-dev.safetensors": ("https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/split_files/diffusion_models/flux1-fill-dev.safetensors", "diffusion_models"),
    "clip_l.safetensors": ("https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors", "text_encoders"),
    "t5xxl_fp8_e4m3fn_scaled.safetensors": ("https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
    "ae.safetensors": ("https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors", "vae"),
    "moge_3_vitg_fp16.safetensors": ("https://huggingface.co/Comfy-Org/MoGe/resolve/main/geometry_estimation/moge_3_vitg_fp16.safetensors", "geometry_estimation"),
    "moge_2_vitl_normal_fp16.safetensors": ("https://huggingface.co/Comfy-Org/MoGe/resolve/main/geometry_estimation/moge_2_vitl_normal_fp16.safetensors", "geometry_estimation"),
}


class G:
    def __init__(self):
        self.nodes, self.links = [], []
        self._id = 0
        self._link = 0
        self.api = {}

    def node(self, type_, pos, widgets=(), inputs=(), outputs=(), size=(320, 120), title=None, api_widgets=None,
             mode=0):
        self._id += 1
        n = {"id": self._id, "type": type_, "pos": list(pos), "size": list(size), "flags": {}, "order": self._id - 1,
             "mode": mode,
             "inputs": [{"name": nm, "type": tp, "link": None} for nm, tp in inputs],
             "outputs": [{"name": nm, "type": tp, "links": [], "slot_index": i} for i, (nm, tp) in enumerate(outputs)],
             "properties": {"Node name for S&R": type_}, "widgets_values": list(widgets)}
        if title:
            n["title"] = title
        models = [{"name": w, "url": MODELS[w][0], "directory": MODELS[w][1]}
                  for w in widgets if isinstance(w, str) and w in MODELS]
        if models:
            n["properties"]["models"] = models
        self.nodes.append(n)
        if api_widgets is not None and mode == 0:
            self.api[str(self._id)] = {"class_type": type_, "inputs": dict(api_widgets)}
        return self._id

    def link(self, src, slot, dst, input_name):
        self._link += 1
        s = next(n for n in self.nodes if n["id"] == src)
        d = next(n for n in self.nodes if n["id"] == dst)
        out = s["outputs"][slot]
        inp = next(i for i in d["inputs"] if i["name"] == input_name)
        out["links"].append(self._link)
        inp["link"] = self._link
        self.links.append([self._link, src, slot, dst, d["inputs"].index(inp), out["type"]])
        if str(dst) in self.api:
            self.api[str(dst)]["inputs"][input_name] = [str(src), slot]

    def ui(self, groups=()):
        return {"last_node_id": self._id, "last_link_id": self._link, "nodes": self.nodes, "links": self.links,
                "groups": list(groups), "config": {}, "extra": {"ds": {"scale": 0.75, "offset": [0, 0]}},
                "version": 0.4}


def build(depth: str = "moge", test: bool = False, image_name: str = "example.png"):
    """depth: 'moge' (native MoGe-3), 'builtin' (StereoForge HF depth)."""
    g = G()
    g.node("Note", (-420, 40), [NOTE], size=(400, 560), title="READ ME")

    load = g.node("LoadImage", (0, 40), [image_name, "image"], outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
                  size=(320, 380), api_widgets={"image": image_name})
    down = g.node("SF_OptionalDownscale", (0, 470), [False, 3840], inputs=[("image", "IMAGE")],
                  outputs=[("IMAGE", "IMAGE")], api_widgets={"enabled": False, "max_long_side": 3840})
    g.link(load, 0, down, "image")

    if depth == "moge":
        moge_file = "moge_2_vitl_normal_fp16.safetensors" if test else "moge_3_vitg_fp16.safetensors"
        ml = g.node("LoadMoGeModel", (380, 40), [moge_file], outputs=[("MOGE_MODEL", "MOGE_MODEL")],
                    api_widgets={"model_name": moge_file})
        res_level = 2 if test else 9
        mi = g.node("MoGeInference", (380, 170), [res_level, 0.0, 1, True, True, 3],
                    inputs=[("moge_model", "MOGE_MODEL"), ("image", "IMAGE")],
                    outputs=[("moge_geometry", "MOGE_GEOMETRY")], size=(320, 220),
                    api_widgets={"resolution_level": res_level, "fov_x_degrees": 0.0, "batch_size": 1,
                                 "force_projection": True, "apply_mask": True, "refine_steps": 3})
        g.link(ml, 0, mi, "moge_model")
        g.link(down, 0, mi, "image")
        dg = g.node("SF_DisparityFromGeometry", (380, 440), [],
                    inputs=[("moge_geometry", "MOGE_GEOMETRY"), ("da3_geometry", "DA3_GEOMETRY")],
                    outputs=[("disparity", "MASK"), ("preview", "IMAGE")], api_widgets={})
        g.link(mi, 0, dg, "moge_geometry")
    else:
        model = "Depth-Anything-V2-Small" if test else "Depth-Anything-V2-Large"
        pres = 518 if test else 1036
        dg = g.node("SF_DepthEstimate", (380, 40), [model, pres, True, "fp16"], inputs=[("image", "IMAGE")],
                    outputs=[("disparity", "MASK"), ("preview", "IMAGE")], size=(320, 160),
                    api_widgets={"model": model, "process_res": pres, "detail_pass": True, "precision": "fp16"})
        g.link(down, 0, dg, "image")

    dprev = g.node("PreviewImage", (380, 600), [], inputs=[("images", "IMAGE")], size=(320, 260),
                   title="Disparity (white = near)", api_widgets={})
    g.link(dg, 1, dprev, "images")

    rw = ["left", 3.0, 0.25, True, "inpaint", -1, 0, 0, 0.35, -1, -1]
    ren = g.node("SF_StereoRender", (760, 40), rw, inputs=[("image", "IMAGE"), ("disparity", "MASK")],
                 outputs=[("stereo", "SF_STEREO"), ("rendered_view", "IMAGE"), ("inpaint_mask", "MASK"),
                          ("disparity_preview", "IMAGE"), ("hole_mask", "MASK")], size=(340, 330),
                 api_widgets={"source_eye": "left", "depth_budget_pct": 3.0, "convergence": 0.25,
                              "stereo_window": True, "border_fill": "inpaint", "edge_refine_px": -1,
                              "fg_dilate_px": 0, "edge_band_px": 0, "edge_slope": 0.35, "soft_edge_px": -1, "fg_guard_px": -1})
    g.link(down, 0, ren, "image")
    g.link(dg, 0, ren, "disparity")
    pprev = g.node("PreviewImage", (760, 420), [], inputs=[("images", "IMAGE")], size=(340, 260),
                   title="Parallax (red = in front of screen, blue = behind)", api_widgets={})
    g.link(ren, 3, pprev, "images")

    # ---- generative fill -------------------------------------------------
    if test:
        ck = g.node("CheckpointLoaderSimple", (1160, 40), ["sd-v1-5-inpainting.ckpt"],
                    outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
                    api_widgets={"ckpt_name": "sd-v1-5-inpainting.ckpt"})
        model_src, clip_src, vae_src = (ck, 0), (ck, 1), (ck, 2)
    else:
        un = g.node("UNETLoader", (1160, 40), ["flux1-fill-dev.safetensors", "default"],
                    outputs=[("MODEL", "MODEL")], api_widgets={"unet_name": "flux1-fill-dev.safetensors",
                                                               "weight_dtype": "default"})
        cl = g.node("DualCLIPLoader", (1160, 170),
                    ["clip_l.safetensors", "t5xxl_fp8_e4m3fn_scaled.safetensors", "flux", "default"],
                    outputs=[("CLIP", "CLIP")], size=(320, 130),
                    api_widgets={"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp8_e4m3fn_scaled.safetensors",
                                 "type": "flux", "device": "default"})
        va = g.node("VAELoader", (1160, 330), ["ae.safetensors"], outputs=[("VAE", "VAE")],
                    api_widgets={"vae_name": "ae.safetensors"})
        model_src, clip_src, vae_src = (un, 0), (cl, 0), (va, 0)

    te = g.node("CLIPTextEncode", (1160, 440), [FLUX_PROMPT], inputs=[("clip", "CLIP")],
                outputs=[("CONDITIONING", "CONDITIONING")], size=(320, 140), api_widgets={"text": FLUX_PROMPT})
    g.link(*clip_src, te, "clip")
    if test:
        pos_src = (te, 0)
    else:
        fg = g.node("FluxGuidance", (1160, 620), [30.0], inputs=[("conditioning", "CONDITIONING")],
                    outputs=[("CONDITIONING", "CONDITIONING")], api_widgets={"guidance": 30.0})
        g.link(te, 0, fg, "conditioning")
        pos_src = (fg, 0)
    zo = g.node("ConditioningZeroOut", (1160, 720), [], inputs=[("conditioning", "CONDITIONING")],
                outputs=[("CONDITIONING", "CONDITIONING")], api_widgets={})
    g.link(te, 0, zo, "conditioning")

    steps, cfg, tile, sched = (3, 4.0, 512, "normal") if test else (28, 1.0, 1024, "simple")
    ti = g.node("SF_TiledInpaint", (1540, 40),
                [0, "fixed", steps, cfg, "euler", sched, 1.0, tile, 256 if not test else 128, 4],
                inputs=[("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                        ("vae", "VAE"), ("image", "IMAGE"), ("mask", "MASK")],
                outputs=[("image", "IMAGE")], size=(340, 360),
                api_widgets={"seed": 0, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": sched,
                             "denoise": 1.0, "tile_size": tile, "tile_overlap": 256 if not test else 128,
                             "mask_grow_px": 4})
    g.link(*model_src, ti, "model")
    g.link(*pos_src, ti, "positive")
    g.link(zo, 0, ti, "negative")
    g.link(*vae_src, ti, "vae")
    g.link(ren, 1, ti, "image")
    g.link(ren, 2, ti, "mask")

    co = g.node("SF_StereoCompose", (1920, 40), ["parallel (left | right)", True],
                inputs=[("stereo", "SF_STEREO"), ("source_image", "IMAGE"), ("inpainted", "IMAGE")],
                outputs=[("side_by_side", "IMAGE"), ("generated_eye", "IMAGE"), ("anaglyph_preview", "IMAGE")],
                size=(340, 130), api_widgets={"layout": "parallel (left | right)", "reject_foreground_bleed": True})
    g.link(ren, 0, co, "stereo")
    g.link(down, 0, co, "source_image")
    g.link(ti, 0, co, "inpainted")

    sv = g.node("SaveImage", (1920, 220), ["StereoForge/sbs"], inputs=[("images", "IMAGE")], size=(520, 360),
                api_widgets={"filename_prefix": "StereoForge/sbs"})
    g.link(co, 0, sv, "images")
    an = g.node("PreviewImage", (1920, 620), [], inputs=[("images", "IMAGE")], size=(520, 360),
                title="Anaglyph check (red/cyan)", api_widgets={})
    g.link(co, 2, an, "images")

    groups = [
        {"title": "1. Input", "bounding": [-20, -30, 360, 700], "color": "#3f789e"},
        {"title": "2. Geometry", "bounding": [360, -30, 360, 900], "color": "#8A8"},
        {"title": "3. Render opposite eye (full res)", "bounding": [740, -30, 380, 730], "color": "#a1309b"},
        {"title": "4. Generative fill of disocclusions (1:1 tiles)", "bounding": [1140, -30, 760, 860],
         "color": "#b58b2a"},
        {"title": "5. Side-by-side output", "bounding": [1900, -30, 560, 1030], "color": "#3f789e"},
    ]
    return g.ui(groups), g.api


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for depth, name in [("moge", "StereoForge_MoGe3_FluxFill"), ("builtin", "StereoForge_BuiltinDepth_FluxFill")]:
        ui, api = build(depth)
        with open(os.path.join(OUT, name + ".json"), "w") as f:
            json.dump(ui, f, indent=1)
        with open(os.path.join(OUT, name + ".api.json"), "w") as f:
            json.dump(api, f, indent=1)
    if "--test" in sys.argv:
        tdir = sys.argv[sys.argv.index("--test") + 1]
        img = sys.argv[sys.argv.index("--test") + 2]
        os.makedirs(tdir, exist_ok=True)
        for depth in ("moge", "builtin"):
            ui, api = build(depth, test=True, image_name=img)
            json.dump(ui, open(os.path.join(tdir, f"test_{depth}.json"), "w"), indent=1)
            json.dump(api, open(os.path.join(tdir, f"test_{depth}.api.json"), "w"), indent=1)
    print("ok")
