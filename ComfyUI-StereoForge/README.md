# ComfyUI-StereoForge

Turns one photo into a full-resolution side-by-side stereo pair. The input image is copied into one eye bit-for-bit. The other eye is synthesised by a stereo diffusion model, then rebuilt at full resolution from the original pixels.

```
input  W x H  ──────────────────────────────────────────────►  source eye (bit-exact)
   │                                                                   │
   ├─► StereoSpace (depth-free stereo diffusion, ~0.6 MP) ─► low-res opposite eye
   │        ▲ auto_comfort: probe → measure real parallax → re-render with the right baseline
   │                                                                   │
   ├─► RAFT correspondence (source ↔ generated) ─► binocular disparity + reliability
   │                                                                   │
   └─► Lift: edge-snapped full-res disparity → z-buffered resampling of the ORIGINAL pixels
            + low-frequency transfer of view-dependent changes (highlights, glass tint)
            + generated content only where needed (disocclusions, frame edge, reflections)
            + multi-band (Laplacian) blending, sharpness matching
                     │
                     ▼
         Tiled Masked Refine (FLUX.1-dev, native resolution, masked regions only)
                     │
                     ▼
         Compose: grain match → [L | R] (or cross-eyed) → 2W x H PNG
```

## Why this design

| Approach | Problem |
|---|---|
| Monocular depth + warp + inpaint (most existing ComfyUI stereo nodes, StereoCrafter-style) | A single depth value per pixel can't represent reflections, mirrors, glass or specular highlights. You get "painted-on" reflections, stretched edges, and inpainting that ignores what the other eye should see. |
| Pure generative novel-view synthesis (StereoSpace, GenStereo, camera-control LoRAs for image-edit models) | The view-dependent geometry is right, but output is limited to about 768 px. It also redraws everything, so text, faces and fine texture drift, and the two eyes differ badly in sharpness. |
| Image-edit models asked to "move the camera 6 cm" | The baseline isn't metric or controllable, and they add structural changes. |

StereoForge uses the generative model as the geometry and appearance oracle and uses the original pixels for detail:

1. **[StereoSpace](https://github.com/prs-eth/stereospace)** (ETH Zürich, MIT license, arXiv:2512.10959) synthesises the opposite eye from the photo and a metric camera baseline. It uses no depth map and no warping. It was trained on rectified stereo, including layered and transparent scenes, and in the authors' benchmarks (iSQoE / MEt3R) it beats warp-and-inpaint methods on reflections, glass and speculars. The vendored copy here was changed to render any aspect ratio (correct Plücker intrinsics, H/8 × W/8 latents) instead of the reference script's centre crop.
2. **Binocular, not monocular, disparity.** RAFT optical flow (torchvision, BSD) is run between the source and the generated eye, both forwards and backwards. The disparity therefore describes what the generator actually rendered. For example, a planar mirror's content gets its virtual depth, not the depth of the mirror surface. Forward/backward, in-frame and epipolar checks give a reliability mask.
3. **Full-resolution lift.** Disparity is upsampled with a guided filter driven by the full-res source. Near discontinuities, each pixel snaps to either the foreground or background value, so no pixel gets an in-between disparity and edges don't stretch. Disparity is then forward-splatted with a z-buffer (nearest surface wins), and the original pixels are resampled bicubically. Faces, text and texture come straight from the photo, shifted, not redrawn.
4. **Where the warp is wrong, the generator decides, but only if it can be verified.** The warped view is compared with the generated one at generator scale:
   - Smooth appearance differences (a highlight sliding, glass tint, a soft reflection) are added as a low-frequency residual on top of the full-res detail.
   - Structural disagreement, disocclusions and the strip that only the new eye can see are taken from the generated view. They are joined with multi-band blending, so there are no seams.
   - Disagreement is trusted only where the generated pixels have a valid, forward/backward-consistent, vertically aligned correspondence with the source. That covers real highlights and reflections, which are still the same scene seen from elsewhere. Content with no such match is a generator hallucination (for example, StereoSpace replacing a street-level block with fog). There the original pixels are kept, and the region is exposed as a "rejected" debug mask.
5. **Detail refine.** Only those synthesised regions are re-detailed at native resolution: tiled FLUX.1-dev img2img with a soft mask and Differential Diffusion at denoise 0.5. Pixels outside the mask are never changed.
6. **Eye matching.** The warp's resampling softness is measured and compensated. Sensor grain is estimated on the source and added to synthesised regions.

## Installation

Requirements: a recent ComfyUI (2025 or later), Python 3.10+, and an NVIDIA GPU with 16 GB of VRAM or more for the FLUX refine (the StereoSpace stage alone needs about 6 GB). The pipeline also runs on CPU, but slowly.

```bash
cd ComfyUI/custom_nodes
git clone <this repo> stereoforge-src
cp -r stereoforge-src/ComfyUI-StereoForge ./ComfyUI-StereoForge   # or symlink it
pip install -r ComfyUI-StereoForge/requirements.txt              # use ComfyUI's python / venv
```

(With ComfyUI portable on Windows: `python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-StereoForge\requirements.txt`.)

### Models

| Model | Where | How |
|---|---|---|
| StereoSpace v1.0 (~11 GB: 2 UNets, VAE, CLIP-H) | `ComfyUI/models/stereospace/stereospace-v1-0/` | **Downloaded automatically** on first run from `prs-eth/stereospace-v1-0`. To pre-fetch it: `huggingface-cli download prs-eth/stereospace-v1-0 --local-dir ComfyUI/models/stereospace/stereospace-v1-0 --include "*.json" "*.pth" "vae/diffusion_pytorch_model.safetensors" "CLIP-ViT-H-14-laion2B-s32B-b79K/model.safetensors"`. If the Hub asks for authentication, run `huggingface-cli login` first. |
| RAFT-large (20 MB) | torch hub cache | Downloaded automatically by torchvision. If it can't be fetched, OpenCV SGBM is used as a fallback. |
| FLUX.1-dev fp8 checkpoint (17 GB) | `ComfyUI/models/checkpoints/flux1-dev-fp8.safetensors` | https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors (FLUX.1-dev non-commercial license; accept it on the model page). |

Then drag `workflows/StereoForge_SBS.json` into ComfyUI, choose an image in **Load Image**, and queue the workflow.

## Using it

- **Output:** `ComfyUI/output/StereoForge/sbs_XXXXX.png` is 2W × H, lossless, with the left eye on the left. The generated eye is also saved on its own. There are previews for the anaglyph, the target-view disparity, the low-res StereoSpace view and the generation info.
- **`generate_eye`:** set to `right` (default) to keep the input as the left eye, or `left` to keep it as the right eye. The output stays L|R ordered either way, unless you choose `cross-eyed`.
- **Downscaling:** off by default. With it off, the input pixels appear unchanged in the output (verified bit-exact through LoadImage → SaveImage). Turn it on, with `max_long_side`, only for very large inputs.
- **Depth strength:**
  - `auto_comfort` (default) runs a 20-step probe at `baseline_m`, measures the real 98th-percentile parallax, and re-renders with the baseline that gives `target_max_disparity_pct`: 2.5 % of width, which is comfortable on screens and in VR.
  - `fixed` uses `baseline_m` directly (0.065 m is human eye separation, i.e. orthostereo).
  - `fov_deg` is the assumed field of view across the long side. A wider FOV gives less disparity for the same baseline.
- **Convergence:** the pair is a parallel rig, so infinity is at zero parallax and everything else appears in front of the screen plane. This is the correct geometry for VR headsets and parallel free-viewing. For 3D TVs, move the convergence in your player (horizontal image translation) so the source eye is not resampled.
- **`view_dependent_sensitivity`:** at 1.0 (the default) original pixels are kept wherever they can be warped. Only disocclusions come from the generator, plus a clipped low-frequency tint for moving highlights. Values between 1 and 2 increasingly let the generated view replace warped pixels where they disagree. Use this for mirror- or glass-heavy scenes, and check the result: on real test photos the generator's disagreements were often hallucinations (copied lens flares, erased kerbs) rather than true view-dependent effects.
- **Bypassing the refine** (select the node, Ctrl+B): the pipeline still works and runs faster. Filled regions stay slightly soft because they are upscaled from about 0.6 MP.
- **Other refiners:** the refine node accepts any MODEL / VAE / conditioning. For SDXL, remove FluxGuidance and use cfg 4–6 with a real negative prompt.

## Nodes

| Node | Purpose |
|---|---|
| StereoForge: Load StereoSpace | Loads the model (and downloads it on first use), with auto precision: bf16 or fp16 on GPU, fp32 on CPU. |
| StereoForge: Prepare Input | Optional downscale. Passes the input through untouched by default. |
| StereoForge: Generate Opposite Eye | StereoSpace at any aspect ratio, with auto-comfort baseline and a probe. Offloads to CPU afterwards so FLUX has VRAM. |
| StereoForge: Lift to Full Resolution | The core described above. Outputs the target eye, the refine mask, the disparity preview and the re-synthesis mask. Takes an optional `generated_upscaled` input, e.g. the low-res view passed through a 4× ESRGAN, which is used for filled areas. |
| StereoForge: Tiled Masked Refine | Masked, tiled img2img at native resolution. Only tiles that intersect the mask are sampled. |
| StereoForge: Compose Side-by-Side | Grain match, 8-bit-exact quantisation, SBS / cross-eyed layout and a Dubois anaglyph. |

## What was verified

These were checked in a CPU-only environment, so no GPU timings were measured:

- The nodes load in current ComfyUI. The full graph runs through the ComfyUI API end to end, with both `generate_eye=right` and `left`.
- The output is exactly 2W × H, and the source half is byte-identical to the input PNG.
- StereoSpace renders non-square frames correctly (tested at 512×384, 576×384 and 384×576).
- The lift was run on inputs from 0.7 MP up to 26 MP (6240×4160). The 26 MP run took about 95 s on 4 CPU threads; on a GPU it takes seconds.
- The FLUX refine path could not be run here. The refine node was exercised with an SD1.5 checkpoint, and its FLUX wiring uses only stock ComfyUI nodes (CheckpointLoaderSimple, FluxGuidance, DifferentialDiffusion).

## Limitations

- **One layer per pixel in the lift.** Where two layers overlap at different depths (a reflection on a window with the scene behind it, or semi-transparent layers), the full-res detail follows the dominant layer. The other layer's shift comes only from the low-frequency residual, or from the generated view where the structure differs strongly. Crisp, high-contrast reflections are therefore correct only at the generator's resolution (~0.6 MP) and get their sharpness back from the refiner, not from the photo. Raising `view_dependent_sensitivity` trades detail for correctness in those regions.
- **Generator resolution.** StereoSpace is an SD2-class model trained at 768². Its view drives geometry and filled regions. Very wide panoramas (aspect ratio above about 2.5:1) move outside its training distribution. For those, crop into sections or reduce `processing_megapixels`.
- **Hallucination is limited to masked regions.** Disoccluded areas and the new edge strip are invented content. They are plausible and blended, but not "true". Because the refine uses a generic prompt and is masked, it never alters faces or text that were successfully warped. A face that is itself heavily disoccluded (e.g. a profile behind an object) is regenerated by the model.
- **The metric baseline is inferred.** StereoSpace estimates scene scale from content, which is why auto-comfort measures the actual parallax instead of trusting `baseline_m`.
- **Licenses.** StereoSpace code and weights are MIT. RAFT (torchvision) is BSD. FLUX.1-dev is non-commercial; swap in another refiner for commercial use.

## Files

- `stereospace_vendor/` – StereoSpace model code (MIT, © ETH Zürich; see its LICENSE) plus `pipeline.py`, the arbitrary-aspect inference wrapper.
- `stereo_core.py` – correspondence, disparity completion, edge-snapped upsampling, z-buffer splatting, appearance transfer, blending, grain and sharpness matching.
- `nodes.py` – the ComfyUI nodes.
- `workflows/StereoForge_SBS.json` – the importable workflow.
