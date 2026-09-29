# StereoForge for ComfyUI

This node pack turns a single image into a side-by-side stereograph. It keeps the
source image as one eye, pixel for pixel, and synthesises the other eye at the
same full resolution. An input of **W × H** gives an output of **2W × H**.

It has no extra Python dependencies (it uses the `torch`, `scipy` and
`transformers` that ship with ComfyUI) and works at any resolution or aspect ratio.

---

## 1. Which approach, and why

I looked at the current options for mono-to-stereo conversion:

| Family | Examples | Why it isn't the main engine here |
|---|---|---|
| End-to-end generative stereo | [StereoSpace](https://arxiv.org/abs/2512.10959) (SD2, 512–768 px), [GenStereo](https://arxiv.org/abs/2503.12720) (SD, 512–768 px), [StereoPilot](https://github.com/klingteam/stereopilot) (Wan 2.1 1.3B, 832×480 video), [Eye2Eye](https://arxiv.org/abs/2505.00135) (not released), [StereoDiffusion](https://arxiv.org/abs/2403.04965) | They learn view-dependent effects, but only at 0.25–0.4 MP. Upscaled next to a 12–40 MP source eye, the new eye is visibly softer. It also drifts in identity, text and fine structure, which is exactly what you want to avoid. |
| Depth warp + generative disocclusion fill | [StereoCrafter](https://arxiv.org/abs/2409.07447), [HairGuard (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Guardians_of_the_Hair_Rescuing_Soft_Boundaries_in_Depth_Stereo_CVPR_2026_paper.html), [αDepth (2026)](https://arxiv.org/abs/2606.00386) | Pixels visible in both eyes are *moved*, not re-imagined, so sharpness, identity and text survive at any resolution. The 2026 soft-boundary papers report beating Eye2Eye and StereoCrafter by fixing this family's weak spots: hair and fur edges, and the background leaking in. |

A naive "monocular depth → warp" is poor for well-known reasons: stretched edges,
ghost outlines, flying pixels, halos, soft resampling, and hallucinated
foreground in the holes. None of those is inherent to the method, though; each
one is an engineering failure. So StereoForge is the second family, built
carefully at full resolution:

```
image ─┬─► geometry (MoGe-3, native) ─► 1/z disparity ─┐
       │                                               ▼
       └──────────────────────────────►  Render Opposite Eye  (full res)
                                         • guided depth-edge alignment
                                         • scale-free occlusion-edge snapping
                                         • row-wise mesh rasterisation + z-buffer
                                         • depth-aware Lanczos-3 resampling
                                         • soft-boundary matting layer (hair/fur)
                                         • background-side pre-fill, fg guard band
                                                 │ holes only
                                                 ▼
                                   Tiled Inpaint (FLUX.1 Fill, 1:1 pixel tiles)
                                                 ▼
                         Compose: foreground-bleed rejection, matte, SBS
```

### What each stage fixes

1. **Geometry.** The default is **MoGe-3** (native in ComfyUI ≥ 0.37, MIT
   licence). It predicts true affine-free geometry, so `1/z` is the physically
   correct disparity up to one global scale. The depth budget and convergence
   set exactly the two degrees of freedom a stereographer controls. Its
   sparse-voxel refinement gives sharp occlusion boundaries. Alternatives:
   Depth Anything 3 (native), or the built-in Depth Anything V2 / Depth Pro node
   with a 2× tiled detail pass that recovers thin structures such as whiskers
   and cables.
2. **Edge alignment.** A guided filter snaps the upsampled depth edges to the
   full-resolution colour edges.
3. **Occlusion-edge snapping.** Any horizontal disparity ramp steeper than
   0.35 px/px is an occlusion, not a surface: such a surface would be visibly
   stretched anyway. The ramp is snapped to a clean step. This test doesn't
   depend on scale, so it works on 3 px or 30 px ramps. The result is one clean
   hole instead of a rubber sheet of flying pixels (the classic cause of
   stretched and duplicated edges).
4. **Rasterisation.** Each row is rasterised as a 1-D triangle strip with a
   disparity z-buffer. Coverage has no cracks, occlusion is exact, and 1-px
   structures are point-splatted so they survive.
5. **Resampling.** Depth-aware Lanczos-3 resampling uses only taps from the same
   depth layer, so foreground and background colours never bleed into each
   other. Integer shifts are bit-exact and fractional shifts stay full-sharpness
   (no bilinear softening), so **the two eyes have matching acuity**.
6. **Soft boundaries** (hair, fur, whiskers, defocus). A two-layer
   decomposition, in the spirit of αDepth/HairGuard but done analytically,
   estimates the local foreground colour, background colour and alpha for
   pixels just outside each occlusion edge. The de-contaminated strands then
   move with the foreground as a matte, instead of staying behind as a ghost
   outline.
7. **Generative fill.** Only disocclusions wide enough to need new content
   (plus the image-side strips that come into frame) are regenerated. Slivers
   narrower than about 0.25 % of the width keep the background-mirrored fill,
   which looks the same at that size and costs nothing. **FLUX.1 Fill [dev]**
   works at **1:1 pixel scale** on crops sized to the gaps plus a context
   margin (never larger than 1024 px), so the new pixels have native
   resolution and grain. Crops own disjoint pixels and later crops see earlier
   results as context, so there are no tile seams.
8. **Keeping the inpainter to background.** A guard band hides the foreground
   next to each hole from the inpainter, so it continues the *background*
   rather than growing the object. A model-independent post-check then rejects
   any generated pixel that still reads as the adjacent foreground.
9. **Compose.** The source eye is copied bit-exactly. The generated eye is the
   exact render, then the fill (only where needed), then the hair matte.

---

## 2. Installation

### 2.1 The node pack
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ryanmolton/Misc.git StereoForge-src
cp -r StereoForge-src/ComfyUI-StereoForge ./ComfyUI-StereoForge
rm -rf StereoForge-src
# restart ComfyUI
```
You don't need `pip install`. Everything used ships with ComfyUI.

**ComfyUI version.** You need **v0.37.0 or later** for the recommended
workflow (native MoGe-3 nodes). On older builds, use the *BuiltinDepth* workflow,
which only needs this pack.

### 2.2 Models

| File | Folder (`ComfyUI/models/…`) | Size | Link | Licence |
|---|---|---|---|---|
| `moge_3_vitg_fp16.safetensors` | `geometry_estimation/` | 2.5 GB | [download](https://huggingface.co/Comfy-Org/MoGe/resolve/main/geometry_estimation/moge_3_vitg_fp16.safetensors) | MIT |
| `flux1-fill-dev.safetensors` | `diffusion_models/` | 23.8 GB | [download](https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/split_files/diffusion_models/flux1-fill-dev.safetensors) | FLUX.1 [dev] non-commercial |
| `clip_l.safetensors` | `text_encoders/` | 0.25 GB | [download](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors) | – |
| `t5xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders/` | 5 GB | [download](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors) | Apache-2.0 |
| `ae.safetensors` | `vae/` | 0.3 GB | [download](https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors) | – |

The workflow embeds these links, so ComfyUI's *missing models* dialog offers
them directly. The built-in depth node downloads
`depth-anything/Depth-Anything-V2-Large-hf` (CC-BY-NC-4.0) or
`apple/DepthPro-hf` into `models/stereoforge/` on first use. Optional
alternatives: `moge_3_vitl_fp16` (0.74 GB, less VRAM), or Depth Anything 3
from [Comfy-Org/Depth-Anything-3](https://huggingface.co/Comfy-Org/Depth-Anything-3)
(Apache-2.0).

**VRAM.** FLUX Fill in bf16 wants about 24 GB. On 12–16 GB cards, set the
`UNETLoader` `weight_dtype` to `fp8_e4m3fn`. The rendering stage runs on the
GPU; I estimate about 1 GB of working memory per 10 MP (not measured on a GPU).

### 2.3 Workflows
Drag one of these onto the ComfyUI canvas:

* `workflows/StereoForge_MoGe3_FluxFill.json`: **recommended**
* `workflows/StereoForge_BuiltinDepth_FluxFill.json`: runs without the native MoGe nodes

The `*.api.json` files are the same graphs in API format, for scripted use.

---

## 3. Nodes and settings

| Node | Purpose |
|---|---|
| **Optional Downscale** | `enabled = false` (default) passes the tensor through untouched. When enabled, Lanczos-downscales to `max_long_side`; it never upscales. |
| **Disparity from MoGe / DA3** | Native geometry → `1/z` → robustly normalised disparity (MASK, white = near). Sky or invalid pixels map to infinity. |
| **Depth Estimate (built-in)** | Depth Anything V2 (global pass plus a 2× tiled detail pass fused in the frequency domain) or Depth Pro. |
| **Disparity from Depth Image** | Use any other depth node (Marigold, DepthCrafter…). Choose the encoding. |
| **Flatten Regions to Plane** | Optional. Give it a mask of pictures, screens, posters or signs (for example from SAM) and each masked region becomes the plane of its surroundings. |
| **Render Opposite Eye** | The full-resolution geometric renderer described above. |
| **Tiled Inpaint (native res)** | Works with any inpainting model. FLUX Fill: cfg 1, FluxGuidance 30, 20 steps, euler/simple. SD1.5/SDXL inpainting: tile 512/1024 with normal cfg. |
| **Compose Side-by-Side** | Parallel (L\|R) or cross-eyed (R\|L), plus the generated eye alone and a red/cyan anaglyph for quick checks. |

**Comfort and composition** (Render node):

* `depth_budget_pct`: the total parallax range as a percentage of image width.
  **2–3 %** suits monitors and TVs; **3–5 %** suits VR headsets and small
  prints. More isn't better: past about 3 % on large screens, viewers get
  eye strain and the holes to fill grow.
* `convergence`: where the screen plane sits within the depth range (0 =
  farthest, 1 = nearest).
* `stereo_window` (default on): automatically pushes the screen plane back so
  that nothing touching the left or right frame edge floats in front of the
  screen. That avoids window violations, the most uncomfortable stereo error.
* `border_fill`: `inpaint` regenerates the thin strip that comes into view at
  the side edges; `black` gives a classic floating window instead.
* `source_eye`: `left` (default) or `right`.
* `min_fill_px` (advanced, default auto ≈ 0.25 % of width): gaps thinner
  than this skip the inpainting model. Raise it for speed, or set it to 0 to
  inpaint every gap.
* Other advanced settings (all have automatic, resolution-scaled defaults):
  `soft_edge_px` (hair and fur matte width), `fg_guard_px`, `edge_band_px`,
  `edge_slope`, `edge_refine_px`.

---

## 4. View-dependent effects: what is handled, and how honestly

| Element | Behaviour |
|---|---|
| **Shadows** | They are view-*independent* (they lie on surfaces), so rendering them from the surface depth is physically correct. |
| **Planar mirrors, glass, screens** | A mirror's reflection appears at its *virtual* depth behind the glass. MoGe and Depth Anything usually predict that virtual depth, so the reflection gets correct stereo automatically. If they predict the mirror's surface instead, the mirror reads as a flat picture. Nothing ghosts or doubles in either case. Use *Flatten Regions to Plane* to force the flat-surface interpretation (screens, posters, photos, signage). |
| **Specular highlights, metal** | In real stereo, highlights on curved surfaces sit slightly *behind* the surface and shift by a few pixels between the eyes. Here they stay on the surface. Most viewers read this as a matte or painted gloss only on large, glossy, near objects. It never produces rivalry or ghosting. |
| **Transparency, layered glass, smoke** | A single depth per pixel means the transparent layer and what lies behind it share one disparity: flattened, but consistent. |
| **Hair, fur, whiskers, defocus** | The soft-boundary matte moves the strands with their owner. Strands the depth model assigns to the background from far out beyond the silhouette (for example long isolated whiskers) can still break into segments. |
| **Text, faces, fabric, fine detail** | These are moved, never re-generated, so identity, expressions and text stay intact. Pixels are only generated where the second eye sees something the first eye could not. |

**Candidly:** fully correct view-dependent appearance (moving highlights,
reflections of off-screen objects, refraction) needs a model that has learned
real stereo appearance. Today those models (StereoSpace, StereoPilot, Eye2Eye)
run at under 1 MP and would cost the resolution, identity and pixel fidelity
you asked for. At high resolution this pipeline makes the deliberate trade:
geometrically exact and artefact-free, with view-dependent shading kept as in
the source eye. If a high-resolution learned stereo model appears, the right
place to add it is a low-frequency, disocclusion-masked "appearance residual" on
top of this render, not a replacement for it.

## 5. Other limitations

* Depth errors become depth errors. Monocular depth is very good but not
  perfect: a mis-estimated object is placed at the wrong depth. It won't be
  distorted or duplicated. Check the *disparity* preview.
* Large disocclusions (a big near object in front of a detailed background,
  or high `depth_budget_pct`) depend on FLUX Fill's plausibility. The content
  will be plausible, not the hidden truth.
* Time: the FLUX fill dominates. Its cost is the total crop area times the
  step count. When gaps outline a large subject, crop area stays around the
  image's own area: roughly one FLUX pass per megapixel. On an RTX 3090 (about
  1.4 s per step at 1 MP), expect roughly 30 s per megapixel at 20 steps. To
  go faster: lower `steps` (try 16; check results on your own images), raise `min_fill_px`, set
  `border_fill` to black, or run without the fill entirely (disconnect
  `inpainted` from the Compose node). That last option is near-instant but
  uses the mirrored fill in wide gaps too. Everything before the fill takes
  seconds on a GPU.

## 6. Validation done in development

* `tests/test_core.py` (CPU, no GPU needed) checks the following: a flat scene
  is reproduced identically; integer shifts are bit-exact; each disocclusion is
  a single contiguous gap of the right width on the right side; smooth slopes
  do not tear; zero-parallax background is untouched; the stereo window rule
  holds; soft fringes move with the foreground.
* The full MoGe workflow was executed end to end in a current ComfyUI build
  (v0.37-era, CPU, with MoGe-2 and SD1.5-inpainting substituted for the large
  models). The left half of the output PNG is **bit-identical** to the input
  PNG and the output is exactly 2W × H.
* All shipped `.json` workflows were loaded in the real ComfyUI frontend
  (headless Chromium) and converted to exactly the expected API prompts.
* **Not verified:** FLUX.1 Fill and MoGe-3 themselves were not run (no GPU in
  the development environment). The FLUX path uses the same concat-conditioning
  code path as ComfyUI's `InpaintModelConditioning`, which was exercised with
  the SD1.5 inpainting model.

## 7. Layout of this folder
```
__init__.py, nodes.py        ComfyUI node definitions
stereoforge/core.py          rendering / matting / compositing (pure PyTorch)
stereoforge/depth_hf.py      built-in Depth Anything V2 / Depth Pro
workflows/                   importable workflows (+ API format)
tools/build_workflows.py     regenerates the workflows
tests/test_core.py           geometry tests
```
