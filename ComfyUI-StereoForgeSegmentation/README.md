# StereoForge Segmentation for ComfyUI

This pack turns one photo into a side-by-side stereo pair with clean edges. It is
separate from the original **StereoForge** pack (every node starts with `SFS_`),
so both can be installed side by side.

## The idea

Artifacts at object edges come from two sources:

1. **Wrong edges.** Depth maps put object outlines a few pixels off, so edge
   pixels travel with the wrong surface. That is what you see as ghosting.
2. **Invented background.** The second eye sees a sliver of background the
   camera never recorded. That content has to be invented, and a bad
   invention shows up as a smudge.

This pack handles both the way professional 2D-to-3D conversion does:

* **Edges come from object masks, not depth.** SAM 3 (or you) outlines each
  object, and a matting network (ViTMatte) refines the outline down to hair
  level. Depth only decides how far each object moves.
* **Every invented pixel is made once, as a flat "clean plate" you can see.**
  The review images outline exactly what was invented. If a spot looks wrong,
  regenerate it with FLUX Fill or fix it yourself in any image editor.
* **Rendering is deterministic.** Each object is a smooth layer composited
  over the plates, and the photo itself stays bit-exact as one eye. *If the
  mattes and plates look right as flat images, the stereo pair is clean.*

## Install (no command line)

1. You need **ComfyUI v0.37 or newer** (for the native SAM 3 and MoGe nodes).
2. Download this repository as a ZIP from the branch it lives on and unzip it.
3. Copy the **`ComfyUI-StereoForgeSegmentation`** folder into
   `ComfyUI\custom_nodes\`. With the ComfyUI Desktop app on Windows, that is
   usually `Documents\ComfyUI\custom_nodes`.
4. Restart ComfyUI.
5. Drag `workflows\StereoForgeSeg_MoGe3_SAM3.json` onto the canvas. Ignore the
   `.api.json` file.
6. Models (ComfyUI's missing-models dialog offers these links):

| File | Folder (`ComfyUI\models\…`) | Size | Licence |
|---|---|---|---|
| [sam3.1_multiplex_fp16.safetensors](https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors) | `checkpoints` | 1.75 GB | SAM License (Meta) |
| [moge_3_vitg_fp16.safetensors](https://huggingface.co/Comfy-Org/MoGe/resolve/main/geometry_estimation/moge_3_vitg_fp16.safetensors) | `geometry_estimation` | 2.5 GB | MIT |
| ViTMatte-Base (downloaded automatically on first run) | `vitmatte` | 0.4 GB | Apache-2.0 |
| LaMa `big-lama.pt` (downloaded automatically on first run) | `inpaint` | 0.2 GB | Apache-2.0 |
| *Optional, only for Refine Plates:* FLUX.1 Fill dev, clip_l, t5xxl fp8, ae (same files as the original StereoForge pack) | | | FLUX.1 [dev] non-commercial |

No `pip install` is needed.

## Using it

1. **Load your photo** in the *Photo* node. It becomes the left eye, bit-exact.
2. **Objects.** Type what matters in *Objects to find*. `person:4` finds up to
   four people. Add more kinds with commas, for example
   `person:4, dog, gift box:3`. Check the *Objects* preview:
   * **A wrong mask:** type its number in *Collect Object Masks → drop*.
   * **A missing object:** un-mute *Paint extra object*, load the same photo
     into it, open *MaskEditor* (right-click), paint the object and run again.
     It is connected to `mask_1`; for more painted objects, add more Load
     Image nodes and connect them to `mask_2`, `mask_3` and `mask_4`.
   * **Rule of thumb:** anything with an outline you care about should have a
     mask. Unmasked depth edges still work, but they are auto-filled (see
     step 5).
3. **Run.** Check the three review images:
   * **Mattes:** the object outlines.
   * **Plate review:** one image per layer, with invented content outlined in
     magenta. Judge it like a normal photo: does the background behind each
     person look natural?
   * **Stereo review:** the new eye. Cyan outlines mark invented content that
     is visible. Magenta marks auto-filled areas: the frame-edge strip, and
     depth edges that have no mask.
4. **Fix only what needs fixing.** There are two options:
   * **Regenerate with FLUX Fill.** Set the Refine node (*Refine Plates
     (diffusion)*) to *Always* (right-click → Mode, or select it and press
     Ctrl+B) and un-mute the FLUX loaders. To redo only one spot, paint it as
     a mask and connect it to `only_here`.
   * **Fix it yourself.** Open the saved plate
     (`output\StereoForgeSeg\plates_000NN_.png`, where the number is the layer
     index plus one) in any editor, then clone, heal or paint the outlined
     area. Load the result into *Edited plate*, set the Apply node (*Apply
     Plate Edit*) to *Always* and give it the layer number. Only the invented
     pixels are taken from your edit.
5. The side-by-side pair is saved to `output\StereoForgeSeg\sbs_*.png`.

### Settings (Build Layers)

| Setting | Meaning |
|---|---|
| `source_eye` | Which eye the photo is. Switching it puts all invented background on the *other* side of every object. If the areas behind your subjects' fronts are busy (like a flash shadow in front of a face), try `right`. |
| `depth_budget_pct` | Total parallax as % of the image width. 2–3 suits screens, 3–5 suits VR. Less parallax means narrower invented areas. |
| `convergence`, `stereo_window` | Where the screen plane sits. `stereo_window` keeps objects that touch the frame edge behind the screen. |
| `matting` | ViTMatte-Base (best), Small (faster), or hard edges. |
| `edge_band_px` | How far the matte may move each mask edge. Raise it for loose masks or fluffy hair. 0 = auto. |
| `object_smoothing_px` | Smooths each object's own depth, so parts of one object that overlap (an arm across the chest) stretch slightly instead of tearing. 0 = auto. |

## Limits (honest)

* **Invented content is invented.** It will be plausible, not what was really
  there. That is why it is exposed for review.
* **The mattes are only as good as the masks.** A mask that misses a limb
  moves that limb with the background. The Objects preview is the place to
  catch it.
* **Parts of the same object at different depths** are kept as one smooth
  surface. If you need strong relief within an object (an outstretched arm
  pointing at the camera), give that part its own mask.
* **Reflections, highlights and transparency** move with the surface they
  are on.

## Validation done in development

* The full workflow was run end to end in ComfyUI v0.37-era builds on CPU
  (with MoGe‑2 standing in for MoGe‑3) on a real family photo. SAM 3 found
  both people, and the source eye was bit-identical in the output.
* The workflow JSON was loaded in the real ComfyUI frontend (headless Chromium)
  and converted and queued successfully, including the muted and bypassed
  optional nodes.
* *Apply Plate Edit* and *Refine Plates* were exercised, the latter with an
  SD1.5 inpainting model standing in for FLUX Fill.
* `tests/test_layers.py` (CPU): background depth near objects is not
  contaminated by misaligned depth edges, only revealed pixels are invented,
  edits touch only invented pixels, and objects move with their matte.
* **Not verified here:** FLUX Fill and MoGe‑3 themselves (no GPU).
