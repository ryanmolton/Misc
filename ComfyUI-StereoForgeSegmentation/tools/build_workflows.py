"""Generate the shipped workflow (UI format) + API-format twin for tests.

    python tools/build_workflows.py [--test OUT_DIR IMAGE]
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "workflows")

MODELS = {
    "moge_3_vitg_fp16.safetensors": ("https://huggingface.co/Comfy-Org/MoGe/resolve/main/geometry_estimation/moge_3_vitg_fp16.safetensors", "geometry_estimation"),
    "moge_2_vitl_normal_fp16.safetensors": ("https://huggingface.co/Comfy-Org/MoGe/resolve/main/geometry_estimation/moge_2_vitl_normal_fp16.safetensors", "geometry_estimation"),
    "sam3.1_multiplex_fp16.safetensors": ("https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors", "checkpoints"),
    "flux1-fill-dev.safetensors": ("https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/split_files/diffusion_models/flux1-fill-dev.safetensors", "diffusion_models"),
    "clip_l.safetensors": ("https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors", "text_encoders"),
    "t5xxl_fp8_e4m3fn_scaled.safetensors": ("https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
    "ae.safetensors": ("https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors", "vae"),
}

NOTE = """StereoForge Segmentation - clean-edged stereo pairs

HOW IT WORKS
Edges come from object masks + a matting network, not from the depth map.
Everything the new eye sees that the photo does not contain is invented ONCE
as a flat "clean plate" you can inspect and fix. If the mattes and plates look
right as flat images, the stereo pair is clean.

1. Load your photo (Load Image, top left). It stays the LEFT eye, bit-exact.
2. Objects: SAM 3 finds what you type in "Objects to find". Add :N for up
   to N of each (default "person:4"); several kinds, e.g.
   "person:4, dog, gift box:3". Check 'Objects' preview. Wrong/missing?
   - drop bad ones by number in Collect Object Masks > drop
   - paint a missing object: un-mute 'Paint extra object', load the SAME
     photo, right-click > Open in MaskEditor, paint it, connect to mask_1.
3. Run. Check 'Mattes' (edges) and 'Plate review' (outlined = invented).
4. A plate looks wrong? Either
   a) un-bypass 'Refine Plates (diffusion)' to redo it with FLUX Fill
      (paint 'only_here' to redo just one spot), or
   b) open the saved plate PNG (output/StereoForgeSeg/plates_*) in any
      editor, fix it, load it in 'Edited plate', un-bypass 'Apply Plate
      Edit' and set its layer number.
5. 'Stereo review': cyan = invented content visible in the new eye,
   magenta = auto-filled (depth edges with no mask - add a mask there if it
   matters). The side-by-side is saved to output/StereoForgeSeg/."""


class G:
    def __init__(self):
        self.nodes, self.links, self.api = [], [], {}
        self._id = self._link = 0

    def node(self, type_, pos, widgets=(), inputs=(), outputs=(), size=(320, 120), title=None, api=None, mode=0):
        self._id += 1
        n = {"id": self._id, "type": type_, "pos": list(pos), "size": list(size), "flags": {}, "order": self._id - 1,
             "mode": mode, "inputs": [{"name": a, "type": b, "link": None} for a, b in inputs],
             "outputs": [{"name": a, "type": b, "links": [], "slot_index": i} for i, (a, b) in enumerate(outputs)],
             "properties": {"Node name for S&R": type_}, "widgets_values": list(widgets)}
        if title:
            n["title"] = title
        models = [{"name": w, "url": MODELS[w][0], "directory": MODELS[w][1]} for w in widgets
                  if isinstance(w, str) and w in MODELS]
        if models:
            n["properties"]["models"] = models
        self.nodes.append(n)
        if api is not None and mode == 0:
            self.api[str(self._id)] = {"class_type": type_, "inputs": dict(api)}
        return self._id

    def link(self, src, slot, dst, name):
        self._link += 1
        s = next(n for n in self.nodes if n["id"] == src)
        d = next(n for n in self.nodes if n["id"] == dst)
        inp = next(i for i in d["inputs"] if i["name"] == name)
        s["outputs"][slot]["links"].append(self._link)
        inp["link"] = self._link
        self.links.append([self._link, src, slot, dst, d["inputs"].index(inp), s["outputs"][slot]["type"]])
        if str(dst) in self.api and str(src) in self.api:
            self.api[str(dst)]["inputs"][name] = [str(src), slot]


def build(test=False, image="example.png"):
    g = G()
    g.node("Note", (-460, 0), [NOTE], size=(430, 760), title="READ ME")
    load = g.node("LoadImage", (0, 0), [image, "image"], outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
                  size=(320, 360), title="Photo", api={"image": image})
    down = g.node("SFS_OptionalDownscale", (0, 400), [False, 3840], inputs=[("image", "IMAGE")],
                  outputs=[("IMAGE", "IMAGE")], api={"enabled": False, "max_long_side": 3840})
    g.link(load, 0, down, "image")
    paint = g.node("LoadImage", (0, 560), [image, "image"], outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
                   size=(320, 360), title="Paint extra object (optional)", mode=2)

    moge_file = "moge_2_vitl_normal_fp16.safetensors" if test else "moge_3_vitg_fp16.safetensors"
    ml = g.node("LoadMoGeModel", (380, 0), [moge_file], outputs=[("MOGE_MODEL", "MOGE_MODEL")],
                api={"model_name": moge_file})
    mi = g.node("MoGeInference", (380, 120), [9, 0.0, 1, True, True, 3],
                inputs=[("moge_model", "MOGE_MODEL"), ("image", "IMAGE")], outputs=[("moge_geometry", "MOGE_GEOMETRY")],
                size=(320, 220), api={"resolution_level": 9, "fov_x_degrees": 0.0, "batch_size": 1,
                                      "force_projection": True, "apply_mask": True, "refine_steps": 3})
    g.link(ml, 0, mi, "moge_model")
    g.link(down, 0, mi, "image")
    dg = g.node("SFS_DisparityFromGeometry", (380, 380), [], inputs=[("moge_geometry", "MOGE_GEOMETRY"),
                ("da3_geometry", "DA3_GEOMETRY")], outputs=[("disparity", "MASK"), ("preview", "IMAGE")], api={})
    g.link(mi, 0, dg, "moge_geometry")

    sam = g.node("CheckpointLoaderSimple", (380, 520), ["sam3.1_multiplex_fp16.safetensors"],
                 outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")], title="SAM 3.1",
                 api={"ckpt_name": "sam3.1_multiplex_fp16.safetensors"})
    te = g.node("CLIPTextEncode", (380, 660), ["person:4"], inputs=[("clip", "CLIP")],
                outputs=[("CONDITIONING", "CONDITIONING")], size=(320, 100), title="Objects to find",
                api={"text": "person:4"})
    g.link(sam, 1, te, "clip")
    det = g.node("SAM3_Detect", (380, 800), [0.5, 2, True],
                 inputs=[("model", "MODEL"), ("image", "IMAGE"), ("conditioning", "CONDITIONING")],
                 outputs=[("masks", "MASK"), ("bboxes", "BOUNDING_BOX")], size=(320, 150),
                 api={"threshold": 0.5, "refine_iterations": 2, "individual_masks": True})
    g.link(sam, 0, det, "model")
    g.link(down, 0, det, "image")
    g.link(te, 0, det, "conditioning")

    col = g.node("SFS_CollectObjects", (760, 0), ["", 0.05],
                 inputs=[("image", "IMAGE"), ("detected", "MASK"), ("mask_1", "MASK"), ("mask_2", "MASK"),
                         ("mask_3", "MASK"), ("mask_4", "MASK")],
                 outputs=[("objects", "MASK"), ("preview", "IMAGE")], size=(320, 200),
                 api={"drop": "", "min_area_pct": 0.05})
    g.link(down, 0, col, "image")
    g.link(det, 0, col, "detected")
    g.link(paint, 1, col, "mask_1")
    op = g.node("PreviewImage", (760, 240), [], inputs=[("images", "IMAGE")], size=(320, 260), title="Objects",
                api={})
    g.link(col, 1, op, "images")

    bl = g.node("SFS_BuildLayers", (1120, 0), ["left", 3.0, 0.25, True, "ViTMatte-Base", 0, 0.0],
                inputs=[("image", "IMAGE"), ("disparity", "MASK"), ("objects", "MASK")],
                outputs=[("layers", "SFS_LAYERS"), ("matte_preview", "IMAGE"), ("plate_review", "IMAGE"),
                         ("plates", "IMAGE"), ("plate_masks", "MASK")], size=(340, 280),
                api={"source_eye": "left", "depth_budget_pct": 3.0, "convergence": 0.25, "stereo_window": True,
                     "matting": "ViTMatte-Base", "edge_band_px": 0, "object_smoothing_px": 0.0})
    g.link(down, 0, bl, "image")
    g.link(dg, 0, bl, "disparity")
    g.link(col, 0, bl, "objects")
    mp = g.node("PreviewImage", (1120, 320), [], inputs=[("images", "IMAGE")], size=(340, 260), title="Mattes",
                api={})
    g.link(bl, 1, mp, "images")
    pr = g.node("PreviewImage", (1120, 620), [], inputs=[("images", "IMAGE")], size=(340, 260),
                title="Plate review (outlined = invented)", api={})
    g.link(bl, 2, pr, "images")
    ps = g.node("SaveImage", (1120, 920), ["StereoForgeSeg/plates"], inputs=[("images", "IMAGE")],
                size=(340, 260), title="Plates (edit these if needed)", api={"filename_prefix": "StereoForgeSeg/plates"})
    g.link(bl, 3, ps, "images")

    # optional fixes (bypassed)
    ed_img = g.node("LoadImage", (1500, 620), [image, "image"], outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
                    size=(320, 320), title="Edited plate (optional)", mode=2)
    un = g.node("UNETLoader", (1860, 620), ["flux1-fill-dev.safetensors", "default"], outputs=[("MODEL", "MODEL")],
                mode=2)
    cl = g.node("DualCLIPLoader", (1860, 730), ["clip_l.safetensors", "t5xxl_fp8_e4m3fn_scaled.safetensors", "flux",
                                               "default"], outputs=[("CLIP", "CLIP")], size=(320, 130), mode=2)
    va = g.node("VAELoader", (1860, 890), ["ae.safetensors"], outputs=[("VAE", "VAE")], mode=2)
    ft = g.node("CLIPTextEncode", (1860, 1000), ["The background behind the people, continued naturally: same surfaces, "
                                                 "shadows, texture, grain and lighting. No people, no new objects."],
                inputs=[("clip", "CLIP")], outputs=[("CONDITIONING", "CONDITIONING")], size=(320, 120), mode=2)
    fg = g.node("FluxGuidance", (1860, 1150), [30.0], inputs=[("conditioning", "CONDITIONING")],
                outputs=[("CONDITIONING", "CONDITIONING")], mode=2)
    zo = g.node("ConditioningZeroOut", (1860, 1250), [], inputs=[("conditioning", "CONDITIONING")],
                outputs=[("CONDITIONING", "CONDITIONING")], mode=2)
    g.link(cl, 0, ft, "clip")
    g.link(ft, 0, fg, "conditioning")
    g.link(ft, 0, zo, "conditioning")
    rf = g.node("SFS_RefinePlates", (1500, 0), [-1, 0, "fixed", 20, 1.0, "euler", "simple", 96],
                inputs=[("layers", "SFS_LAYERS"), ("model", "MODEL"), ("positive", "CONDITIONING"),
                        ("negative", "CONDITIONING"), ("vae", "VAE"), ("only_here", "MASK")],
                outputs=[("SFS_LAYERS", "SFS_LAYERS")], size=(320, 300), title="Refine Plates (diffusion) - optional",
                mode=4)
    g.link(bl, 0, rf, "layers")
    g.link(un, 0, rf, "model")
    g.link(fg, 0, rf, "positive")
    g.link(zo, 0, rf, "negative")
    g.link(va, 0, rf, "vae")
    ep = g.node("SFS_EditPlate", (1500, 340), [0], inputs=[("layers", "SFS_LAYERS"), ("edited_plate", "IMAGE"),
                                                           ("only_here", "MASK")],
                outputs=[("SFS_LAYERS", "SFS_LAYERS")], size=(320, 120), title="Apply Plate Edit - optional", mode=4)
    g.link(rf, 0, ep, "layers")
    g.link(ed_img, 0, ep, "edited_plate")

    rd = g.node("SFS_Render", (2240, 0), ["parallel (left | right)"],
                inputs=[("layers", "SFS_LAYERS"), ("source_image", "IMAGE")],
                outputs=[("side_by_side", "IMAGE"), ("generated_eye", "IMAGE"), ("review", "IMAGE"),
                         ("anaglyph", "IMAGE")], size=(320, 140), api={"layout": "parallel (left | right)"})
    g.link(ep, 0, rd, "layers")
    g.link(down, 0, rd, "source_image")
    # API twin: bypassed fixes are skipped, so Render reads the layers directly
    g.api[str(rd)]["inputs"]["layers"] = [str(bl), 0]
    sv = g.node("SaveImage", (2240, 180), ["StereoForgeSeg/sbs"], inputs=[("images", "IMAGE")], size=(520, 340),
                api={"filename_prefix": "StereoForgeSeg/sbs"})
    g.link(rd, 0, sv, "images")
    rv = g.node("PreviewImage", (2240, 560), [], inputs=[("images", "IMAGE")], size=(520, 340),
                title="Stereo review (cyan = invented, magenta = auto-filled)", api={})
    g.link(rd, 2, rv, "images")

    ui = {"last_node_id": g._id, "last_link_id": g._link, "nodes": g.nodes, "links": g.links, "groups": [
        {"title": "1. Photo", "bounding": [-20, -60, 360, 1000], "color": "#3f789e"},
        {"title": "2. Depth + objects", "bounding": [360, -60, 740, 1040], "color": "#8A8"},
        {"title": "3. Layers, mattes, clean plates (REVIEW)", "bounding": [1100, -60, 380, 1260], "color": "#a1309b"},
        {"title": "4. Optional plate fixes", "bounding": [1480, -60, 740, 1400], "color": "#b58b2a"},
        {"title": "5. Stereo pair", "bounding": [2220, -60, 560, 980], "color": "#3f789e"}],
        "config": {}, "extra": {"ds": {"scale": 0.6, "offset": [480, 80]}}, "version": 0.4}
    return ui, g.api


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ui, api = build()
    json.dump(ui, open(os.path.join(OUT, "StereoForgeSeg_MoGe3_SAM3.json"), "w"), indent=1)
    json.dump(api, open(os.path.join(OUT, "StereoForgeSeg_MoGe3_SAM3.api.json"), "w"), indent=1)
    if "--test" in sys.argv:
        tdir, img = sys.argv[sys.argv.index("--test") + 1], sys.argv[sys.argv.index("--test") + 2]
        os.makedirs(tdir, exist_ok=True)
        ui, api = build(test=True, image=img)
        json.dump(ui, open(os.path.join(tdir, "seg_test.json"), "w"), indent=1)
        json.dump(api, open(os.path.join(tdir, "seg_test.api.json"), "w"), indent=1)
    print("ok")
