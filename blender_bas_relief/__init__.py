bl_info = {
    "name": "Bas Relief Generator",
    "author": "Claude Code",
    "version": (1, 0, 0),
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
    verts_all, tris_all, nors_all = [], [], []
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
        offset += nv

    if not verts_all:
        return None
    return (np.concatenate(verts_all), np.concatenate(tris_all), np.concatenate(nors_all))


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
    verts, tris, cnors = geo

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

    return dict(depth=depth, normals=normals, mask=mask, pix=pix, W=W, H=H,
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
        detail_boost=settings.detail_boost,
        detail_radius=settings.detail_radius,
        layering=settings.layering,
        background_lift=settings.background_lift,
        edge_softness=settings.edge_softness,
        edge_threshold=settings.edge_threshold,
        smoothing=settings.smoothing,
        use_normals=settings.use_normals,
    )
    hgt = relief_core.compute_relief(cap["depth"], cap["normals"], cap["mask"], cap["pix"], params)
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
        b.label(text="Shape")
        c = b.column(align=True)
        c.prop(s, "compression")
        c.prop(s, "detail_boost")
        c.prop(s, "detail_radius")
        c = b.column(align=True)
        c.prop(s, "layering")
        c.prop(s, "background_lift")
        c.prop(s, "edge_softness")
        c.prop(s, "edge_threshold")
        c.prop(s, "smoothing")
        b.prop(s, "use_normals")
        b.prop(s, "invert")

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


classes = (BASRELIEF_Settings, BASRELIEF_OT_generate, BASRELIEF_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.bas_relief = PointerProperty(type=BASRELIEF_Settings)


def unregister():
    del bpy.types.Scene.bas_relief
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
