# Bas Relief Generator for Blender

Turns whatever the active camera sees into a printable bas-relief panel, similar to
ZBrush's Bas Relief feature. You set up the scene and the camera, press one button, and
get a watertight mesh at real-world size, ready to export to STL.

![example](docs/example_render.png)

## Why it isn't just a depth map

A depth map makes every silhouette a cliff. Each object sticks straight out toward you
and the surface detail gets squashed. This add-on works on surface **slopes**, taken
from the camera-space normal map, so it behaves like a real relief carving:

1. One ray is cast per output pixel from the active camera. The add-on records the depth and the
   *smooth interpolated* normal, which respects shade smooth, auto smooth and custom normals.
2. Occlusion edges (silhouettes, one object in front of another) are detected and their
   depth jumps are removed.
3. Slopes are compressed logarithmically. Steep areas flatten a lot and gentle detail
   barely changes, so all the detail stays readable in a few millimetres.
4. Optional detail boost (unsharp masking in the gradient domain).
5. The slopes are integrated back into a height field with an exact Poisson solve.
6. A controllable, softened amount of depth ordering is blended back in, so objects still
   separate from the background and from each other.

Top: this add-on. Bottom: a plain depth map at the same total depth.

![comparison](docs/relief_vs_depthmap.png)

## Install

Blender 4.2 or newer: *Edit → Preferences → Get Extensions → ▾ → Install from Disk…* and pick
`blender_bas_relief.zip`. For Blender 4.1, use *Add-ons → Install…* and pick the same zip.

The panel is in the **3D Viewport sidebar (N) → Relief** tab.

## Workflow

1. Build your scene and add a camera. **Orthographic** gives the truest reliefs. Perspective
   works, but it bakes the perspective into the panel.
2. Set **Output Properties → Resolution**. The aspect ratio becomes the panel shape, and the
   pixel count becomes the mesh density (one vertex per pixel, scaled by the % slider).
   For example, 1500 × 1000 px at 150 mm wide gives 0.1 mm detail.
3. Set the panel width, relief depth and base thickness, then press **Generate Relief**.
4. Adjust the shape settings and press **Rebuild (Reuse Capture)**. This skips the ray casting,
   so it's fast. Press Generate again after you move anything in the scene.
5. Export with *File → Export → STL*. With the default "1 Unit = 1 mm" setting, the numbers in
   the STL are millimetres, which is what slicers expect.

The camera's clip end acts as the background: anything past it, or empty space, becomes the
flat background plate. You can also include a backdrop object.

## Settings

| Setting | What it does |
|---|---|
| Source | Visible objects, selected objects only, or one collection. The generated relief is always ignored. |
| Panel Width | Physical width in mm. The height follows the camera aspect ratio. |
| Relief Depth | Height of the highest point above the background, in mm. |
| Base Thickness | Backing plate thickness, in mm. |
| Units | `1 Unit = 1 mm` (best for STL) or true size in scene units. |
| **Compression** | How much steep slopes flatten compared with gentle ones. 0 = no compression. 3–8 is typical. Higher gives a flatter, more even, more "coin-like" relief. |
| **Detail Boost** | Exaggerates fine surface detail. 1 = unchanged. |
| Detail Size | Pixel size of what counts as "fine" detail for the boost. |
| **Depth Layering** | 0 = pure normal-based relief (edges flush with the background). 1 = plain depth map. 0.15–0.4 is a good range. |
| Background Lift | How far even the farthest object stands off the background. |
| **Edge Softness** | Rounds the silhouette steps (in pixels). This helps printing and looks more hand-carved. |
| Edge Threshold | Slope above which a depth jump counts as an occlusion edge. This also caps slopes. 3 ≈ 72°. |
| Smoothing | Final blur of the height field (in pixels). This removes stair-stepping. |
| Use Smooth Normals | Takes slopes from interpolated normals. Turn it off to use the raw depth buffer. |
| Invert | Intaglio: carves the image into the panel instead of raising it (useful for moulds). |
| Mesh Step | Uses every Nth pixel as a vertex, to make the mesh lighter. |
| Watertight Solid | Adds sides and a flat back so the mesh is manifold and printable. |
| Heightmap PNG | Optionally also writes a 16-bit grayscale heightmap (for CNC or displacement). |

## Tips

* Go for **more relief depth and higher compression** rather than a deep, uncompressed relief.
  Reliefs read through light and shadow, not real depth.
* Thin parts that sit at different depths, like fingers in front of a body, separate better
  with a bit more Depth Layering and a small Edge Softness.
* Ray casting runs at about 1 M rays per second, plus processing. A few megapixels take
  only a few seconds.
