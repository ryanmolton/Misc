# ComfyUI-StereoForge

Turns a single photo into a **parallel side-by-side stereo pair** at full resolution. The photo becomes the left eye, copied bit-exactly. The right eye is rendered from the **original pixels** using Depth Anything 3 geometry. The one thing that makes it different from a plain depth warp: floor reflections (puddles, wet streets, glossy floors) are rendered at their physically correct *mirrored* depth instead of being painted flat onto the ground.

## Why a plain depth warp flattens gloss

A monocular depth model assigns one depth per pixel. On a puddle, Depth Anything 3 (like every other depth model we checked) returns the depth of the **ground surface**, so the reflected sky, buildings and trees are warped as if they were printed on the pavement. In a real stereo pair, a reflection sits *behind* the mirror, at the distance of the reflected object. The effect is strongest exactly where people notice it: puddles, rain-soaked streets, polished floors.

## How v2 works

```
photo ─► Depth Anything 3 (metric depth + intrinsics, ~1 MP)
  │
  ├─► reflector mask: Grounding DINO boxes ("puddle. sky reflected in water. mirror.")
  │                    → SAM 2.1 masks → GrabCut edge snap
  │
  ├─► floor plane: RANSAC on the DA3 point cloud
  │     ├─ virtual depth: mirror the visible scene across the plane and z-buffer it
  │     │   into the camera → exact depth of every reflected object that is in frame;
  │     │   off-frame content (sky, overhead branches) → infinity
  │     └─ reflected fraction per pixel: Fresnel reflectance of water at each pixel's
  │         viewing angle × brightness budget vs. the wet floor, plus mirror-consistency
  │         (the photo matches the predicted mirror image) for dark reflected objects
  │
  └─► full-resolution layered render (all original pixels, bicubic resampling)
        surface layer  = photo − reflection   → moves with the floor depth
        reflection     = reflected fraction   → moves with the virtual depth,
                                                clipped by the puddle outline, which
                                                moves with the floor
        disparity: edge-snapped upsampling (no in-between depths → no stretched edges),
        z-buffered splatting (near surfaces occlude correctly), LaMa for disocclusions
```

Design choices that came out of testing on real puddle photos:

- **Fresnel decides what is reflection.** Looking steeply down at a puddle, water reflects only ~2%, so the bricks under it stay on the floor. At grazing angles it's close to a mirror. A cap derived from the Fresnel term stops submerged texture from being dragged to the reflection's depth.
- **No fractional texture split.** Low frequencies are split by the reflected fraction. Fine detail goes to the reflection layer only where reflection clearly dominates. A 50/50 split of texture would ghost when the layers move by different amounts.
- **Clean rims.** The reflection layer is extended past the puddle outline by mirroring its own texture horizontally, since all stereo motion is horizontal. That way, sliding under the rim reveals matching texture, not smear. Anti-aliased rim pixels are excluded from that texture source.
- **Fail-safe.** If no floor plane or no reflector is found, the render is exactly the plain DA3 warp.

## Installation

Requirements: ComfyUI (2025 or later) and Python 3.10+. A GPU is recommended; everything also runs on CPU (about 1–2 min per image).

```bash
cd ComfyUI/custom_nodes
git clone <this repo> stereoforge-src
cp -r stereoforge-src/ComfyUI-StereoForge ./ComfyUI-StereoForge
python ComfyUI-StereoForge/install.py          # use ComfyUI's python / venv
```

`install.py` installs `requirements.txt`, then installs Depth Anything 3 and `simple-lama-inpainting` with `--no-deps`. Their own dependency pins (numpy<2, xformers, open3d…) would otherwise downgrade ComfyUI's environment.

Models download automatically on first use:

| Model | Size | License |
|---|---|---|
| depth-anything/DA3NESTED-GIANT-LARGE-1.1 | ~5.6 GB | CC BY-NC 4.0 (DA3METRIC-LARGE is Apache 2.0) |
| IDEA-Research/grounding-dino-base | 0.9 GB | Apache 2.0 |
| facebook/sam2.1-hiera-large | 0.9 GB | Apache 2.0 |
| big-lama (via simple-lama-inpainting) | 0.2 GB | Apache 2.0 |

Then load `workflows/StereoForge_v2_LayeredSBS.json`, pick an image and queue it.

## Nodes

| Node | Purpose |
|---|---|
| **Depth Anything 3** | Metric depth + camera intrinsics. `process_res` 1008 by default; the render itself is always full resolution. |
| **Reflector Mask** | Open-vocabulary reflector segmentation. The optional `extra_mask` input is OR-ed in, so you can add SAM 3 masks or a hand-painted mask. |
| **Layered Stereo Render** | Produces the parallel SBS (`2W × H`), the generated eye, the disoccluded-pixel mask (for an external inpainter if you prefer, e.g. FLUX Fill), a reflection preview and an anaglyph preview. |

Render settings:
- `parallax_mode`
  - `percent_of_width` (default 3.0): the nearest 2% of the scene gets 3% of the image width as parallax.
  - `metric_ipd`: the value is a camera separation in mm. 65 gives a physically scaled pair; this needs the metric DA3 model.
- `source_eye`: set to `right` if your photo should be the right eye. The output stays L|R.
- `reflections`: set to off, or bypass the Reflector Mask node, for scenes without reflective floors.
- `reflection_layer` (optional input): feeds the output of a single-image reflection-separation model into the thin-film path. We tested RDNet here; it changed only about 0.3% of pixels on our test images, so it's not wired by default.

## What was tested

Tested on three rain photos (Toronto puddle, bollard puddle, cobblestone street), CPU only:

- The input half of the SBS is byte-identical to the input PNG, and the output is exactly 2W × H.
- The full graph runs through the ComfyUI API, and the workflow imports cleanly in the frontend.
- The puddle reflections (clouds, the bollard, buildings, trees) now carry disparity consistent with their mirrored distance instead of the ground's. Floor texture and submerged bricks keep the floor's disparity. Object edges (pedestrian, umbrella, tree trunks) show no halos or stretching.

## Limitations

- **Floor reflectors only.** Reflections on horizontal surfaces (puddles, wet or polished floors) are handled. Vertical mirrors, shop windows and car paint are not yet: they are rendered at DA3's depth like any plain warp. The same machinery extends to them (fit the mirror plane from its frame instead of the floor), but that isn't implemented.
- **Off-frame reflected content is put at infinity.** That's correct for sky and a mild approximation for, say, overhead branches.
- **Segmentation can miss or over-include.** The reflector mask is the least reliable stage. Check the "Reflections found" preview, and use `extra_mask` to correct it if needed.
- **The base geometry is DA3's.** Its errors (thin wires, glass, sky boundary) carry through, as in any depth-warp pipeline.

---

## Legacy: StereoSpace nodes (v1)

The first version generated the opposite eye with StereoSpace (a depth-free stereo diffusion model) and lifted it to full resolution. On real photos the generator hallucinated too often: it rendered wrong-eye views at small baselines, duplicated lens flares and erased kerbs. The v1 nodes remain under the `StereoForge/legacy (StereoSpace)` category, with their workflow in `workflows/StereoForge_SBS.json`, but v2 is the recommended path.
