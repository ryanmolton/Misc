bl_info = {
    "name": "Bas Relief Generator",
    "author": "Claude Code",
    "version": (1, 1, 0),
    "blender": (4, 1, 0),
    "location": "3D Viewport > Sidebar (N) > Relief",
    "description": "Turn the active camera's view of the scene into a printable bas-relief panel",
    "category": "Object",
}

import time

import bpy
import numpy as np
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from . import relief_core

OUTPUT_TAG = "bas_relief_output"

# Last camera capture, so settings can be re-tuned without re-casting rays.
_CAPTURE = {}


# ==========================================================================
# Scene capture: cast one ray per output pixel from the active camera
# ==========================================================================

def _wanted(inst, settings):
    src = inst.parent if inst.is_instance and inst.parent else inst.object
    orig = src.original
    if orig.get(OUTPUT_TAG) or inst.object.original.get(OUTPUT_TAG):
        return False
    if inst.object.type not in {"MESH", "CURVE", "SURFACE", "META", "FONT", "CURVES"}:
        return False
    if settings.source == "SELECTED":
        return orig.select_get()
    if settings.source == "COLLECTION":
        col = settings.collection
        return col is not None and orig.name in col.all_objects
    return not orig.hide_render


def _gather_geometry(context, settings, cam_inv):
    depsgraph = context.evaluated_depsgraph_get()
    verts_all, tris_all, nors_all, oid_all = [], [], [], []
    names = []          # per source object: (instancer name or "", object name)
    offset = 0
    for inst in depsgraph.object_instances:
        if not _wanted(inst, settings):
            continue
        ob = inst.object
        try:
            me = ob.to_mesh()
        except RuntimeError:
            continue
        if me is None:
            continue
        try:
            me.calc_loop_triangles()
            nv, nt = len(me.vertices), len(me.loop_triangles)
            if nt == 0:
                continue
            co = np.empty(nv * 3, np.float64)
            me.vertices.foreach_get("co", co)
            tri_v = np.empty(nt * 3, np.int64)
            me.loop_triangles.foreach_get("vertices", tri_v)
            tri_l = np.empty(nt * 3, np.int64)
            me.loop_triangles.foreach_get("loops", tri_l)
            cn = np.empty(len(me.loops) * 3, np.float64)
            if hasattr(me, "corner_normals"):
                me.corner_normals.foreach_get("vector", cn)
            else:  # Blender < 4.1
                me.calc_normals_split()
                me.loops.foreach_get("normal", cn)
        finally:
            ob.to_mesh_clear()

        M = cam_inv @ inst.matrix_world
        A = np.array(M, dtype=np.float64)
        co = co.reshape(-1, 3) @ A[:3, :3].T + A[:3, 3]
        N = np.array(M.to_3x3().inverted_safe().transposed(), dtype=np.float64)
        cn = cn.reshape(-1, 3) @ N.T
        cn /= np.maximum(np.linalg.norm(cn, axis=1, keepdims=True), 1e-12)

        verts_all.append(co)
        tris_all.append(tri_v.reshape(-1, 3) + offset)
        nors_all.append(cn[tri_l].reshape(-1, 3, 3))
        par = inst.parent.original.name if (inst.is_instance and inst.parent) else ""
        oid_all.append(np.full(nt, len(names), np.int64))
        names.append((par, ob.original.name))
        offset += nv

    if not verts_all:
        return None
    return (np.concatenate(verts_all), np.concatenate(tris_all), np.concatenate(nors_all),
            np.concatenate(oid_all), names)


def capture(context, settings):
    scene = context.scene
    cam = scene.camera
    if cam is None or cam.type != "CAMERA":
        raise RuntimeError("The scene needs an active camera")
    r = scene.render
    W = max(2, int(r.resolution_x * r.resolution_percentage / 100))
    H = max(2, int(r.resolution_y * r.resolution_percentage / 100))

    cam_mat = cam.matrix_world.normalized()
    cam_inv = cam_mat.inverted()
    geo = _gather_geometry(context, settings, cam_inv)
    if geo is None:
        raise RuntimeError("No visible geometry found for the chosen source")
    verts, tris, cnors, tri_oid, obj_names = geo

    bvh = BVHTree.FromPolygons(verts.tolist(), tris.tolist(), all_triangles=True)

    cd = cam.data
    frame = [Vector(v) for v in cd.view_frame(scene=scene)]
    xmin, xmax = min(v.x for v in frame), max(v.x for v in frame)
    ymin, ymax = min(v.y for v in frame), max(v.y for v in frame)
    fz = frame[0].z
    ortho = cd.type == "ORTHO"
    if cd.type == "PANO":
        raise RuntimeError("Panoramic cameras are not supported")
    clip_s, clip_e = cd.clip_start, cd.clip_end

    xs = xmin + (np.arange(W) + 0.5) * (xmax - xmin) / W
    ys = ymax - (np.arange(H) + 0.5) * (ymax - ymin) / H      # row 0 = top

    hit_loc = np.zeros((H, W, 3))
    hit_idx = np.full((H, W), -1, np.int64)
    ray = bvh.ray_cast

    wm = context.window_manager
    wm.progress_begin(0, H)
    t0 = time.time()
    try:
        if ortho:
            d = Vector((0.0, 0.0, -1.0))
            span = clip_e - clip_s
            for i in range(H):
                y = float(ys[i])
                for j in range(W):
                    loc, _n, idx, _d = ray(Vector((float(xs[j]), y, -clip_s)), d, span)
                    if idx is not None:
                        hit_idx[i, j] = idx
                        hit_loc[i, j] = loc
                wm.progress_update(i)
        else:
            for i in range(H):
                y = float(ys[i])
                for j in range(W):
                    d = Vector((float(xs[j]), y, fz))
                    d.normalize()
                    loc, _n, idx, _d = ray(d * clip_s, d, clip_e)
                    if idx is not None:
                        hit_idx[i, j] = idx
                        hit_loc[i, j] = loc
                wm.progress_update(i)
    finally:
        wm.progress_end()

    mask = hit_idx >= 0
    depth = np.where(mask, -hit_loc[..., 2], 0.0)
    if not ortho:
        depth = np.where(mask & (depth > clip_e), 0.0, depth)

    # interpolated (smooth / custom split) normals via barycentric coordinates
    normals = np.zeros((H, W, 3))
    normals[..., 2] = 1.0
    if mask.any():
        idx = hit_idx[mask]
        p = hit_loc[mask]
        tv = verts[tris[idx]]                           # (n,3,3)
        a, b, c = tv[:, 0], tv[:, 1], tv[:, 2]
        v0, v1, v2 = b - a, c - a, p - a
        d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
        d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
        den = d00 * d11 - d01 * d01
        den = np.where(np.abs(den) < 1e-20, 1e-20, den)
        wv = (d11 * d20 - d01 * d21) / den
        ww = (d00 * d21 - d01 * d20) / den
        wu = 1.0 - wv - ww
        cn = cnors[idx]
        n = wu[:, None] * cn[:, 0] + wv[:, None] * cn[:, 1] + ww[:, None] * cn[:, 2]
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        # face the camera
        view = np.tile([0.0, 0.0, -1.0], (len(p), 1)) if ortho else p / np.linalg.norm(p, axis=1, keepdims=True)
        flip = (n * view).sum(1) > 0
        n[flip] *= -1.0
        normals[mask] = n

    if ortho:
        pix = np.full((H, W), (xmax - xmin) / W)
    else:
        pix = depth * ((xmax - xmin) / W) / abs(fz)

    obj_id = np.where(mask, tri_oid[np.maximum(hit_idx, 0)], -1)

    return dict(depth=depth, normals=normals, mask=mask, pix=pix, W=W, H=H,
                obj_id=obj_id, obj_names=obj_names,
                aspect=(ymax - ymin) / (xmax - xmin), seconds=time.time() - t0,
                camera=cam.name, scene=scene.name)


# ==========================================================================
# Mesh output
# ==========================================================================

def _build_mesh(name, height, fy, fx, width, panel_h, depth_amt, base, solid):
    """height: (H,W) in [0,1], row 0 = top; fy/fx: 0..1 positions of rows/cols.
    Returns a bpy mesh centred on the origin, relief facing +Z."""
    H, W = height.shape
    xs = (fx - 0.5) * width
    ys = (0.5 - fy) * panel_h
    X, Y = np.meshgrid(xs, ys)
    Z = base + height * depth_amt
    top = np.stack([X, Y, Z], -1).reshape(-1, 3)

    idx = np.arange(H * W).reshape(H, W)
    q = np.stack([idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:], idx[:-1, :-1]], -1).reshape(-1, 4)  # CCW from +Z
    verts = [top]
    faces_flat = [q.reshape(-1)]
    sizes = [np.full(len(q), 4)]

    if solid:
        # perimeter of the top grid, walked clockwise when seen from +Z (top-left -> top-right -> ...)
        ring = np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])
        n = len(ring)
        bottom = top[ring].copy()
        bottom[:, 2] = 0.0
        bidx = np.arange(n) + H * W
        verts.append(bottom)
        nxt = np.roll(np.arange(n), -1)
        side = np.stack([ring, ring[nxt], bidx[nxt], bidx], -1)
        faces_flat.append(side.reshape(-1))
        sizes.append(np.full(n, 4))
        faces_flat.append(bidx)          # bottom n-gon, clockwise from +Z = facing down
        sizes.append(np.array([n]))

    verts = np.concatenate(verts)
    loops = np.concatenate(faces_flat).astype(np.int32)
    sizes = np.concatenate(sizes).astype(np.int32)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32)

    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    me.loops.add(len(loops))
    me.loops.foreach_set("vertex_index", loops)
    me.polygons.add(len(sizes))
    me.polygons.foreach_set("loop_start", starts)
    try:
        me.polygons.foreach_set("loop_total", sizes)
    except (AttributeError, TypeError, RuntimeError):
        pass  # derived automatically in Blender 4.0+
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    return me


def build_relief(context, settings, cap):
    params = dict(
        compression=settings.compression,
        far_compression=settings.far_compression,
        far_flatten=settings.far_flatten,
        depth_curve=settings.depth_curve,
        detail_boost=settings.detail_boost,
        detail_radius=settings.detail_radius,
        layering=settings.layering,
        background_lift=settings.background_lift,
        edge_softness=settings.edge_softness,
        edge_threshold=settings.edge_threshold,
        smoothing=0.0,
        use_normals=settings.use_normals,
    )
    layers = [l for l in settings.layers if l.collection is not None] if settings.use_layers else []
    if not layers:
        hgt = relief_core.compute_relief(cap["depth"], cap["normals"], cap["mask"], cap["pix"], params)
    else:
        # which layer does every captured object belong to (first match wins)?
        members = [set(l.collection.all_objects.keys()) for l in layers]
        lut = np.full(len(cap["obj_names"]) + 1, len(layers), np.int64)
        for i, (par, name) in enumerate(cap["obj_names"]):
            for k, mem in enumerate(members):
                if name in mem or (par and par in mem):
                    lut[i] = k
                    break
        labels = np.where(cap["mask"], lut[np.maximum(cap["obj_id"], 0)], -1)
        regions = []
        for l in layers:
            prm = dict(params)
            prm.update(compression=l.compression, far_compression=l.compression * l.far_factor,
                       detail_boost=l.detail_boost, layering=l.layering)
            regions.append(dict(band=(min(l.band_min, l.band_max), max(l.band_min, l.band_max)),
                                params=prm))
        regions.append(dict(band=(min(settings.other_band_min, settings.other_band_max),
                                  max(settings.other_band_min, settings.other_band_max)),
                            params=params))
        hgt = relief_core.compose_layers(cap["depth"], cap["normals"], cap["mask"], cap["pix"],
                                         labels, regions, settings.edge_softness)
    if settings.smoothing > 0:
        hgt = relief_core.gaussian_blur(hgt, settings.smoothing)
    hgt = np.clip(hgt, 0.0, None)
    if hgt.max() > 1.0:
        hgt /= hgt.max()
    if settings.invert:
        hgt = 1.0 - hgt

    if settings.heightmap_path:
        path = bpy.path.abspath(settings.heightmap_path)
        if not path.lower().endswith(".png"):
            path += ".png"
        relief_core.write_png16(path, hgt)

    step = max(1, settings.mesh_step)
    rows = np.unique(np.append(np.arange(0, hgt.shape[0], step), hgt.shape[0] - 1))
    cols = np.unique(np.append(np.arange(0, hgt.shape[1], step), hgt.shape[1] - 1))
    hm = hgt[np.ix_(rows, cols)]

    if settings.units == "MM":
        k = 1.0
    else:  # true physical size in scene units
        k = 0.001 / max(context.scene.unit_settings.scale_length, 1e-12)
    width = settings.width_mm * k
    panel_h = width * cap["aspect"]
    me = _build_mesh("BasRelief", hm, rows / (hgt.shape[0] - 1), cols / (hgt.shape[1] - 1),
                     width, panel_h, settings.depth_mm * k, settings.base_mm * k, settings.solid)

    ob = None
    if settings.replace:
        for o in context.scene.objects:
            if o.get(OUTPUT_TAG):
                ob = o
                break
    if ob is not None and ob.type == "MESH":
        old = ob.data
        ob.data = me
        if old.users == 0:
            bpy.data.meshes.remove(old)
    else:
        ob = bpy.data.objects.new("BasRelief", me)
        ob[OUTPUT_TAG] = True
        context.scene.collection.objects.link(ob)
    return ob, hm.shape


# ==========================================================================
# Settings / UI
# ==========================================================================

class BASRELIEF_Layer(bpy.types.PropertyGroup):
    collection: PointerProperty(name="Collection", type=bpy.types.Collection,
                                description="Objects in this collection form this depth layer")
    band_min: FloatProperty(name="Band Bottom", default=0.35, min=0.0, max=1.0, subtype="FACTOR",
                            description="Lowest point of this layer, as a fraction of Relief Depth")
    band_max: FloatProperty(name="Band Top", default=1.0, min=0.0, max=1.0, subtype="FACTOR",
                            description="Highest point of this layer, as a fraction of Relief Depth")
    compression: FloatProperty(name="Compression", default=2.0, min=0.0, soft_max=30.0,
                               description="Slope compression inside this layer (low = rounder, fuller forms)")
    far_factor: FloatProperty(name="Far Compression x", default=1.0, min=0.0, soft_max=10.0,
                              description="Compression at the back of this layer, as a multiple of its Compression")
    detail_boost: FloatProperty(name="Detail Boost", default=1.2, min=0.0, soft_max=4.0)
    layering: FloatProperty(name="Depth Layering", default=0.45, min=0.0, max=1.0, subtype="FACTOR",
                            description="How much real depth is kept inside this layer "
                                        "(e.g. an arm in front of a body)")


class BASRELIEF_Settings(bpy.types.PropertyGroup):
    source: EnumProperty(
        name="Source",
        items=[("VISIBLE", "Visible Objects", "Every object that is visible and renderable"),
               ("SELECTED", "Selected Objects", "Only selected objects"),
               ("COLLECTION", "Collection", "Only objects in a collection")],
        default="VISIBLE")
    collection: PointerProperty(name="Collection", type=bpy.types.Collection)

    width_mm: FloatProperty(name="Panel Width", default=150.0, min=1.0, soft_max=2000.0,
                            description="Physical width of the panel in mm. Height follows the camera aspect ratio")
    depth_mm: FloatProperty(name="Relief Depth", default=6.0, min=0.0, soft_max=50.0,
                            description="Height of the tallest point of the relief above the background, in mm")
    base_mm: FloatProperty(name="Base Thickness", default=2.0, min=0.0, soft_max=20.0,
                           description="Thickness of the backing plate under the relief, in mm")
    units: EnumProperty(
        name="Units",
        items=[("MM", "1 Unit = 1 mm", "Write coordinates in millimetres (STL exports at the right size for slicers)"),
               ("SCENE", "Scene Units", "True physical size using the scene's unit scale")],
        default="MM")

    compression: FloatProperty(name="Compression", default=4.0, min=0.0, soft_max=30.0,
                               description="How strongly steep slopes are flattened compared with gentle ones. "
                                           "Higher = flatter, more even relief with detail kept readable")
    far_compression: FloatProperty(name="Far Compression", default=4.0, min=0.0, soft_max=30.0,
                                   description="Compression used for the farthest parts of the scene. Set it higher "
                                               "than Compression to keep foreground forms round and flatten the "
                                               "background")
    far_flatten: FloatProperty(name="Background Flattening", default=0.0, min=0.0, max=1.0, subtype="FACTOR",
                               description="Shrinks all relief towards the far end of the scene, so distant things "
                                           "become almost drawn into the surface (rilievo schiacciato)")
    depth_curve: FloatProperty(name="Foreground Emphasis", default=1.0, min=0.2, soft_max=5.0,
                               description="Shapes Depth Layering: above 1 gives the nearest objects most of the "
                                           "depth and squeezes the distance; below 1 does the opposite")
    detail_boost: FloatProperty(name="Detail Boost", default=1.5, min=0.0, soft_max=4.0,
                                description="Exaggerates fine surface detail (1 = unchanged)")
    detail_radius: FloatProperty(name="Detail Size", default=6.0, min=0.5, soft_max=50.0, subtype="PIXEL",
                                 description="Size (in pixels) of what counts as fine detail for Detail Boost")
    layering: FloatProperty(name="Depth Layering", default=0.25, min=0.0, max=1.0, subtype="FACTOR",
                            description="How much real depth ordering is kept. 0 = pure normal-based relief "
                                        "(edges flush), 1 = plain depth map")
    background_lift: FloatProperty(name="Background Lift", default=0.25, min=0.0, max=1.0, subtype="FACTOR",
                                   description="How far even the farthest object stands off the background "
                                               "(as part of the depth layering)")
    edge_softness: FloatProperty(name="Edge Softness", default=2.0, min=0.0, soft_max=30.0, subtype="PIXEL",
                                 description="Rounds off silhouette steps (pixels)")
    edge_threshold: FloatProperty(name="Edge Threshold", default=3.0, min=0.1, soft_max=20.0,
                                  description="Slope above which a depth jump is treated as a separate "
                                              "edge/occlusion (3 is about 72 degrees). Also caps slopes")
    smoothing: FloatProperty(name="Smoothing", default=0.0, min=0.0, soft_max=10.0, subtype="PIXEL",
                             description="Final smoothing of the height field (pixels)")
    use_normals: BoolProperty(name="Use Smooth Normals", default=True,
                              description="Take slopes from interpolated surface normals (smooth shading, "
                                          "custom normals). Off = from the raw depth buffer")
    invert: BoolProperty(name="Invert (Intaglio / Mould)", default=False,
                         description="Carve the image into the panel instead of raising it")

    use_layers: BoolProperty(name="Depth Layers", default=False,
                             description="Give collections their own depth band and settings")
    layers: bpy.props.CollectionProperty(type=BASRELIEF_Layer)
    layers_index: IntProperty(default=0)
    other_band_min: FloatProperty(name="Bottom", default=0.0, min=0.0, max=1.0, subtype="FACTOR",
                                  description="Depth band of objects that are in no layer")
    other_band_max: FloatProperty(name="Top", default=0.25, min=0.0, max=1.0, subtype="FACTOR",
                                  description="Depth band of objects that are in no layer")

    mesh_step: IntProperty(name="Mesh Step", default=1, min=1, max=16,
                           description="Use every Nth pixel as a vertex (1 = one vertex per pixel)")
    solid: BoolProperty(name="Watertight Solid", default=True,
                        description="Add sides and a flat back so the mesh is printable")
    replace: BoolProperty(name="Replace Previous", default=True,
                          description="Overwrite the previously generated relief object")
    heightmap_path: StringProperty(name="Heightmap PNG", subtype="FILE_PATH", default="",
                                   description="Optional: also save a 16-bit grayscale heightmap (for CNC etc.)")


class BASRELIEF_OT_generate(bpy.types.Operator):
    bl_idname = "basrelief.generate"
    bl_label = "Generate Relief"
    bl_description = "Capture the scene through the active camera and build a bas-relief mesh"
    bl_options = {"REGISTER", "UNDO"}

    recapture: BoolProperty(default=True, options={"HIDDEN"})

    def execute(self, context):
        s = context.scene.bas_relief
        global _CAPTURE
        try:
            if self.recapture or not _CAPTURE or _CAPTURE.get("scene") != context.scene.name:
                _CAPTURE = capture(context, s)
            t0 = time.time()
            ob, shape = build_relief(context, s, _CAPTURE)
        except RuntimeError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        msg = "Relief %dx%d px" % (_CAPTURE["W"], _CAPTURE["H"])
        if self.recapture:
            msg += " (capture %.1fs)" % _CAPTURE["seconds"]
        msg += ", processing %.1fs" % (time.time() - t0)
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class BASRELIEF_OT_layer_add(bpy.types.Operator):
    bl_idname = "basrelief.layer_add"
    bl_label = "Add Depth Layer"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.bas_relief
        l = s.layers.add()
        col = context.collection
        if col is not None and col != context.scene.collection:
            l.collection = col
        s.layers_index = len(s.layers) - 1
        s.use_layers = True
        return {"FINISHED"}


class BASRELIEF_OT_layer_remove(bpy.types.Operator):
    bl_idname = "basrelief.layer_remove"
    bl_label = "Remove Depth Layer"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.bas_relief
        if 0 <= s.layers_index < len(s.layers):
            s.layers.remove(s.layers_index)
            s.layers_index = max(0, s.layers_index - 1)
        return {"FINISHED"}


class BASRELIEF_OT_layer_move(bpy.types.Operator):
    bl_idname = "basrelief.layer_move"
    bl_label = "Move Depth Layer"
    bl_options = {"REGISTER", "UNDO"}
    direction: EnumProperty(items=[("UP", "Up", ""), ("DOWN", "Down", "")])

    def execute(self, context):
        s = context.scene.bas_relief
        i = s.layers_index
        j = i - 1 if self.direction == "UP" else i + 1
        if 0 <= i < len(s.layers) and 0 <= j < len(s.layers):
            s.layers.move(i, j)
            s.layers_index = j
        return {"FINISHED"}


PRESETS = {
    "LOW": dict(depth_mm=3.0, compression=8.0, far_compression=8.0, far_flatten=0.0, depth_curve=1.0,
                detail_boost=1.8, layering=0.12, background_lift=0.3, edge_softness=1.5),
    "MEDIUM": dict(depth_mm=6.0, compression=4.0, far_compression=4.0, far_flatten=0.0, depth_curve=1.0,
                   detail_boost=1.5, layering=0.25, background_lift=0.25, edge_softness=2.0),
    "HIGH": dict(depth_mm=15.0, compression=1.5, far_compression=12.0, far_flatten=0.7, depth_curve=2.0,
                 detail_boost=1.3, layering=0.5, background_lift=0.05, edge_softness=3.0),
}


class BASRELIEF_OT_preset(bpy.types.Operator):
    bl_idname = "basrelief.preset"
    bl_label = "Relief Preset"
    bl_description = "Load a starting point for the shape settings"
    bl_options = {"REGISTER", "UNDO"}
    preset: EnumProperty(items=[
        ("LOW", "Low Relief (Coin)", "Very shallow, even, detail-driven relief"),
        ("MEDIUM", "Bas Relief", "The default balanced bas-relief"),
        ("HIGH", "High Relief (Ghiberti)", "Deep, rounded foreground; background flattened almost to a drawing"),
    ])

    def execute(self, context):
        s = context.scene.bas_relief
        for k, v in PRESETS[self.preset].items():
            setattr(s, k, v)
        return {"FINISHED"}


class BASRELIEF_UL_layers(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "collection", text="", emboss=False, icon="OUTLINER_COLLECTION")
        row.label(text="%.0f-%.0f%%" % (100 * item.band_min, 100 * item.band_max))


class BASRELIEF_PT_panel(bpy.types.Panel):
    bl_label = "Bas Relief"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Relief"

    def draw(self, context):
        s = context.scene.bas_relief
        L = self.layout
        cam = context.scene.camera
        r = context.scene.render
        w = int(r.resolution_x * r.resolution_percentage / 100)
        h = int(r.resolution_y * r.resolution_percentage / 100)
        if cam is None:
            L.label(text="No active camera", icon="ERROR")
        else:
            L.label(text="%s  %d x %d px" % (cam.name, w, h), icon="CAMERA_DATA")
            if cam.type == "CAMERA" and cam.data.type == "PERSP":
                L.label(text="Tip: Orthographic gives truer reliefs", icon="INFO")

        col = L.column(align=True)
        col.prop(s, "source", text="")
        if s.source == "COLLECTION":
            col.prop(s, "collection", text="")

        b = L.box()
        b.label(text="Size")
        c = b.column(align=True)
        c.prop(s, "width_mm")
        c.label(text="Height: %.1f mm" % (s.width_mm * h / max(w, 1) *
                                         (r.pixel_aspect_y / r.pixel_aspect_x)))
        c.prop(s, "depth_mm")
        c.prop(s, "base_mm")
        b.prop(s, "units", text="")

        b = L.box()
        row = b.row()
        row.label(text="Shape")
        row.operator_menu_enum(BASRELIEF_OT_preset.bl_idname, "preset", text="Presets", icon="PRESET")
        c = b.column(align=True)
        c.prop(s, "compression")
        c.prop(s, "far_compression")
        c.prop(s, "far_flatten")
        c = b.column(align=True)
        c.prop(s, "detail_boost")
        c.prop(s, "detail_radius")
        c = b.column(align=True)
        c.prop(s, "layering")
        c.prop(s, "depth_curve")
        c.prop(s, "background_lift")
        c.prop(s, "edge_softness")
        c.prop(s, "edge_threshold")
        c.prop(s, "smoothing")
        b.prop(s, "use_normals")
        b.prop(s, "invert")

        b = L.box()
        b.prop(s, "use_layers")
        if s.use_layers:
            row = b.row()
            row.template_list("BASRELIEF_UL_layers", "", s, "layers", s, "layers_index", rows=3)
            col = row.column(align=True)
            col.operator(BASRELIEF_OT_layer_add.bl_idname, text="", icon="ADD")
            col.operator(BASRELIEF_OT_layer_remove.bl_idname, text="", icon="REMOVE")
            col.separator()
            col.operator(BASRELIEF_OT_layer_move.bl_idname, text="", icon="TRIA_UP").direction = "UP"
            col.operator(BASRELIEF_OT_layer_move.bl_idname, text="", icon="TRIA_DOWN").direction = "DOWN"
            if 0 <= s.layers_index < len(s.layers):
                l = s.layers[s.layers_index]
                c = b.column(align=True)
                c.prop(l, "collection", text="")
                r2 = c.row(align=True)
                r2.prop(l, "band_min", text="Bottom")
                r2.prop(l, "band_max", text="Top")
                c = b.column(align=True)
                c.prop(l, "compression")
                c.prop(l, "far_factor")
                c.prop(l, "detail_boost")
                c.prop(l, "layering")
            b.label(text="Objects in no layer:")
            r2 = b.row(align=True)
            r2.prop(s, "other_band_min")
            r2.prop(s, "other_band_max")
            b.label(text="Uses the global Shape settings", icon="INFO")

        b = L.box()
        b.label(text="Output")
        b.prop(s, "mesh_step")
        b.prop(s, "solid")
        b.prop(s, "replace")
        b.prop(s, "heightmap_path")

        col = L.column(align=True)
        col.scale_y = 1.5
        op = col.operator(BASRELIEF_OT_generate.bl_idname, text="Generate Relief", icon="MOD_DISPLACE")
        op.recapture = True
        sub = L.column()
        sub.enabled = bool(_CAPTURE)
        op = sub.operator(BASRELIEF_OT_generate.bl_idname, text="Rebuild (Reuse Capture)", icon="FILE_REFRESH")
        op.recapture = False


classes = (BASRELIEF_Layer, BASRELIEF_Settings, BASRELIEF_OT_generate, BASRELIEF_OT_layer_add,
           BASRELIEF_OT_layer_remove, BASRELIEF_OT_layer_move, BASRELIEF_OT_preset,
           BASRELIEF_UL_layers, BASRELIEF_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.bas_relief = PointerProperty(type=BASRELIEF_Settings)


def unregister():
    del bpy.types.Scene.bas_relief
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
