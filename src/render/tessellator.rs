use std::collections::{HashMap, HashSet};
use std::f64::consts::{PI, TAU};
use std::sync::Arc;

use rayon::prelude::*;

use crate::cad::model::{
    ArcEntity, BoundingBox, CadEntity, CircleEntity, DxfInsertEntity, EllipseEntity, GdsCellDef,
    GdsArrayInstanceEntity, GdsInstanceEntity, LineEntity, PolygonEntity, PolylineEntity,
    RenderItem, SplineEntity,
};

// ── Per-cell primitive cache ──────────────────────────────────────────────────

/// A vertex in cell-local coordinates (f64).
/// Stored in the cache before any instance transform or origin offset is applied.
#[derive(Copy, Clone)]
pub struct LocalVertex {
    pub pos: [f64; 2],
    pub color: [f32; 4],
}

/// Pre-tessellated primitive geometry for one GDS cell.
/// Contains only leaf primitives (polygons/polylines) in cell-local coords.
/// Sub-instance references are NOT expanded here — they recurse at render time.
pub struct CellCacheEntry {
    pub lines: Vec<LocalVertex>,
    pub fills: Vec<LocalVertex>,
}

/// Key: (cell_idx, fill_polygons)
/// Values are Arc-wrapped so the rayon path can share entries without locking.
pub type CellCache = HashMap<(usize, bool), Arc<CellCacheEntry>>;

// ── GPU instance transform ────────────────────────────────────────────────────

/// Per-instance GPU transform (24 bytes).
/// Encodes a 2D affine: reflect Y (if reflected) → rotate → scale → translate.
/// Applied by `vs_inst` in the WGSL shader.
///
/// For cell-local position `p`, the world position is:
///   `mat2x2(col0, col1) * p + translation`
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTransform {
    /// Column 0 of the 2×2 matrix: `[cos·mag, sin·mag]`
    pub col0: [f32; 2],
    /// Column 1: not-reflected `[−sin·mag, cos·mag]`; reflected `[sin·mag, −cos·mag]`
    pub col1: [f32; 2],
    /// World translation minus tess_origin (cast to f32).
    pub translation: [f32; 2],
}

impl GpuTransform {
    pub fn from_transform(t: &InstanceTransform, origin: [f64; 2]) -> Self {
        let rad = t.angle_deg.to_radians();
        let (sin_a, cos_a) = rad.sin_cos();
        let m = t.mag;
        let (col0, col1) = if !t.reflected {
            (
                [(cos_a * m) as f32,  (sin_a * m) as f32],
                [(-sin_a * m) as f32, (cos_a * m) as f32],
            )
        } else {
            (
                [(cos_a * m) as f32, (sin_a * m) as f32],
                [(sin_a * m) as f32, (-cos_a * m) as f32],
            )
        };
        Self {
            col0,
            col1,
            translation: [
                (t.offset[0] - origin[0]) as f32,
                (t.offset[1] - origin[1]) as f32,
            ],
        }
    }

    /// Build a transform for a DXF INSERT with non-uniform x/y scale.
    /// `offset` is the world-space insertion point for this specific grid element.
    pub fn from_dxf_insert(
        offset: [f64; 2],
        angle_deg: f64,
        x_scale: f64,
        y_scale: f64,
        origin: [f64; 2],
    ) -> Self {
        let (sin_a, cos_a) = angle_deg.to_radians().sin_cos();
        Self {
            col0: [(cos_a * x_scale) as f32, (sin_a * x_scale) as f32],
            col1: [(-sin_a * y_scale) as f32, (cos_a * y_scale) as f32],
            translation: [
                (offset[0] - origin[0]) as f32,
                (offset[1] - origin[1]) as f32,
            ],
        }
    }

    /// Compose a GDS-style parent InstanceTransform with a DXF INSERT child transform.
    /// Used when a `DxfInsert` appears inside a GDS/DXF block cell.
    pub fn from_parent_and_dxf_insert(
        parent: &InstanceTransform,
        ins_offset: [f64; 2],
        ins_angle_deg: f64,
        ins_x_scale: f64,
        ins_y_scale: f64,
        origin: [f64; 2],
    ) -> Self {
        let (sin_b, cos_b) = ins_angle_deg.to_radians().sin_cos();
        // Insert matrix columns (local → parent):
        //   col0 = [cos_b·sx,  sin_b·sx]
        //   col1 = [-sin_b·sy, cos_b·sy]
        let (sin_a, cos_a) = parent.angle_deg.to_radians().sin_cos();
        let m = parent.mag;
        // Parent matrix columns:
        let (pc0, pc1) = if !parent.reflected {
            ([cos_a * m, sin_a * m], [-sin_a * m, cos_a * m])
        } else {
            ([cos_a * m, sin_a * m], [sin_a * m, -cos_a * m])
        };
        // Composed = parent_matrix * insert_matrix  (column-major multiply)
        let ic0 = [cos_b * ins_x_scale, sin_b * ins_x_scale];
        let ic1 = [-sin_b * ins_y_scale, cos_b * ins_y_scale];
        let rc0 = [pc0[0] * ic0[0] + pc1[0] * ic0[1], pc0[1] * ic0[0] + pc1[1] * ic0[1]];
        let rc1 = [pc0[0] * ic1[0] + pc1[0] * ic1[1], pc0[1] * ic1[0] + pc1[1] * ic1[1]];
        let world_off = parent.apply(ins_offset);
        Self {
            col0: [rc0[0] as f32, rc0[1] as f32],
            col1: [rc1[0] as f32, rc1[1] as f32],
            translation: [
                (world_off[0] - origin[0]) as f32,
                (world_off[1] - origin[1]) as f32,
            ],
        }
    }
}

// ── Render output ─────────────────────────────────────────────────────────────

/// Complete output of one background tessellation pass.
pub struct RenderOutput {
    /// GDS cell instances: cell_idx → per-instance GPU transforms.
    /// GPU draws each cell's geometry N times, once per transform.
    pub instances: HashMap<usize, Vec<GpuTransform>>,
    /// Pre-tessellated cell primitives (cell-local coords, earcut run once per cell).
    /// Uploaded to GPU once; reused across all instances of the same cell.
    pub cell_geometries: HashMap<usize, Arc<CellCacheEntry>>,
    /// Non-GDS entities + LOD boxes (misc lines drawn without instancing).
    pub misc_lines: Vec<Vertex>,
    /// Non-GDS fill polygons.
    pub misc_fills: Vec<Vertex>,
    /// Cell cache returned so the main thread can persist entries across frames.
    pub cell_cache: CellCache,
    /// Whether polygons were filled (triangulated) in this pass.
    pub fill_polygons: bool,
}

// ── GDS instance transform ────────────────────────────────────────────────────

/// Accumulated GDS instance transform for recursive cell expansion.
/// Transform order (applied to a point): reflect Y (if reflected), rotate, scale, translate.
#[derive(Clone, Debug)]
pub struct InstanceTransform {
    pub offset: [f64; 2],
    pub angle_deg: f64,
    pub mag: f64,
    pub reflected: bool,
}

impl InstanceTransform {
    pub fn identity() -> Self {
        Self { offset: [0.0, 0.0], angle_deg: 0.0, mag: 1.0, reflected: false }
    }

    /// Apply this transform to a local-space point → parent-space.
    #[inline]
    pub fn apply(&self, p: [f64; 2]) -> [f64; 2] {
        let y = if self.reflected { -p[1] } else { p[1] };
        let rad = self.angle_deg.to_radians();
        let (sin_a, cos_a) = rad.sin_cos();
        [
            (cos_a * p[0] - sin_a * y) * self.mag + self.offset[0],
            (sin_a * p[0] + cos_a * y) * self.mag + self.offset[1],
        ]
    }

    /// Transform a BoundingBox → new AABB in parent space.
    /// All 4 corners are transformed to correctly handle rotation.
    pub fn apply_bbox(&self, bbox: &BoundingBox) -> BoundingBox {
        if !bbox.is_valid() {
            return BoundingBox::empty();
        }
        let corners = [
            [bbox.min[0], bbox.min[1]],
            [bbox.max[0], bbox.min[1]],
            [bbox.max[0], bbox.max[1]],
            [bbox.min[0], bbox.max[1]],
        ];
        let mut result = BoundingBox::empty();
        for c in corners {
            let tp = self.apply(c);
            result.expand(tp[0], tp[1]);
        }
        result
    }

    /// Compose: self is the parent transform, child_* are the child instance's own transform
    /// (in the parent's local space). Returns the combined transform mapping child-local → world.
    pub fn compose(&self, child_offset: [f64; 2], child_angle: f64, child_mag: f64, child_reflected: bool) -> Self {
        Self {
            offset: self.apply(child_offset),
            angle_deg: self.angle_deg + child_angle,
            mag: self.mag * child_mag,
            reflected: self.reflected ^ child_reflected,
        }
    }
}

// ── Vertex ────────────────────────────────────────────────────────────────────

/// A single vertex: 2D position + RGBA color.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

impl Vertex {
    pub fn new(x: f32, y: f32, color: [f32; 4]) -> Self {
        Self {
            position: [x, y],
            color,
        }
    }
}

/// Result of tessellating all entities, split into line and fill buffers.
#[allow(dead_code)]
pub struct TessellateResult {
    pub line_vertices: Vec<Vertex>,
    pub fill_vertices: Vec<Vertex>,
}

/// An owned version of RenderItem that can be sent across thread boundaries.
pub enum OwnedRenderItem {
    Entity(CadEntity),
    LodBox { bounds: BoundingBox, color: [f32; 4] },
}

impl OwnedRenderItem {
    pub fn from_render_item(item: &RenderItem<'_>) -> Self {
        match item {
            RenderItem::Entity { entity } => OwnedRenderItem::Entity((*entity).clone()),
            RenderItem::LodBox { bounds, color } => OwnedRenderItem::LodBox {
                bounds: (*bounds).clone(),
                color: *color,
            },
        }
    }
}

/// Tessellate visible entities into line and fill vertex buffers.
///
/// `origin` — subtracted from all world coordinates before casting to f32.
///   Set to the current camera centre for maximum f32 precision.
///
/// `viewport` — if `Some`, entities whose bounding box does not intersect
///   this region are skipped (viewport culling).  Pass the tessellation
///   viewport (camera view + margin) so that only on-screen entities
///   contribute vertices to the GPU buffer.
#[allow(dead_code)]
pub fn tessellate_all(
    entities: &[CadEntity],
    entity_bounds: &[BoundingBox],
    pixels_per_unit: f64,
    hidden_layers: &HashSet<String>,
    origin: [f64; 2],
    viewport: Option<&BoundingBox>,
) -> TessellateResult {
    let use_culling = viewport.is_some() && entity_bounds.len() == entities.len();
    let mut line_verts: Vec<Vertex> = Vec::with_capacity(entities.len().min(65536) * 4);
    let mut fill_verts: Vec<Vertex> = Vec::with_capacity(entities.len().min(65536) * 6);

    let pixel_size = if pixels_per_unit > 0.0 {
        1.0 / pixels_per_unit
    } else {
        0.0
    };

    for (i, entity) in entities.iter().enumerate() {
        if hidden_layers.contains(entity.layer()) {
            continue;
        }
        if use_culling {
            let eb = &entity_bounds[i];
            if !eb.intersects(viewport.unwrap()) {
                continue;
            }
            if pixel_size > 0.0 && eb.width() < pixel_size && eb.height() < pixel_size {
                continue;
            }
        }
        match entity {
            CadEntity::Polygon(e) => tessellate_polygon(e, origin, &mut fill_verts),
            _ => tessellate_entity(entity, pixels_per_unit, origin, &mut line_verts),
        }
    }
    TessellateResult {
        line_vertices: line_verts,
        fill_vertices: fill_verts,
    }
}

/// Tessellate owned render items — intended for use on a background thread.
/// Returns a `RenderOutput` with GPU instance transforms for GDS cells and
/// flat vertex lists for everything else (DXF entities, LOD boxes, etc.).
pub fn tessellate_owned_items(
    items: Vec<OwnedRenderItem>,
    pixels_per_unit: f64,
    origin: [f64; 2],
    fill_polygons: bool,
    gds_cells: &[GdsCellDef],
    mut cache: CellCache,
    tess_viewport: Option<&BoundingBox>,
    hidden_layers: &HashSet<String>,
) -> RenderOutput {
    let cap = items.len().min(65536);
    let mut output = RenderOutput {
        instances: HashMap::new(),
        cell_geometries: HashMap::new(),
        misc_lines: Vec::with_capacity(cap * 4),
        misc_fills: Vec::with_capacity(cap * 6),
        cell_cache: CellCache::new(),
        fill_polygons,
    };

    for item in &items {
        match item {
            OwnedRenderItem::Entity(entity) => match entity {
                CadEntity::Polygon(p) if fill_polygons => {
                    tessellate_polygon(p, origin, &mut output.misc_fills)
                }
                CadEntity::Polygon(p) => {
                    tessellate_polygon_outline(p, origin, &mut output.misc_lines)
                }
                CadEntity::GdsInstance(inst) => tessellate_gds_instance(
                    inst,
                    &InstanceTransform::identity(),
                    pixels_per_unit,
                    origin,
                    fill_polygons,
                    gds_cells,
                    &mut cache,
                    &mut output,
                    0,
                    tess_viewport,
                    hidden_layers,
                ),
                CadEntity::GdsArrayInstance(arr) => tessellate_gds_array_instance(
                    arr,
                    &InstanceTransform::identity(),
                    pixels_per_unit,
                    origin,
                    fill_polygons,
                    gds_cells,
                    &mut cache,
                    &mut output,
                    0,
                    tess_viewport,
                    hidden_layers,
                ),
                CadEntity::DxfInsert(ins) => tessellate_dxf_insert(
                    ins,
                    pixels_per_unit,
                    origin,
                    fill_polygons,
                    gds_cells,
                    &mut cache,
                    &mut output,
                    tess_viewport,
                    hidden_layers,
                ),
                _ => tessellate_entity(entity, pixels_per_unit, origin, &mut output.misc_lines),
            },
            OwnedRenderItem::LodBox { bounds, color } => {
                tessellate_bounds(bounds, *color, origin, &mut output.misc_lines)
            }
        }
    }

    output.cell_cache = cache;
    output
}

fn tessellate_entity(
    entity: &CadEntity,
    pixels_per_unit: f64,
    origin: [f64; 2],
    out: &mut Vec<Vertex>,
) {
    match entity {
        CadEntity::Line(e) => tessellate_line(e, origin, out),
        CadEntity::Circle(e) => tessellate_circle(e, pixels_per_unit, origin, out),
        CadEntity::Arc(e) => tessellate_arc(e, pixels_per_unit, origin, out),
        CadEntity::Polyline(e) => tessellate_polyline(e, pixels_per_unit, origin, out),
        CadEntity::Ellipse(e) => tessellate_ellipse(e, pixels_per_unit, origin, out),
        CadEntity::Spline(e) => tessellate_spline(e, origin, out),
        CadEntity::Text(_) => {}             // text rendered via egui overlay
        CadEntity::Polygon(_) => {}          // handled separately in fill buffer
        CadEntity::GdsInstance(_) => {}      // handled at RenderItem level
        CadEntity::GdsArrayInstance(_) => {} // handled at RenderItem level
        CadEntity::DxfInsert(_) => {}        // handled at RenderItem level
    }
}

#[inline(always)]
fn rel(v: f64, o: f64) -> f32 {
    (v - o) as f32
}

fn tessellate_polygon(e: &PolygonEntity, origin: [f64; 2], out: &mut Vec<Vertex>) {
    let n = e.points.len();
    if n < 3 {
        return;
    }

    // Flatten points for earcutr (remove duplicate closing point if present)
    let pts: Vec<[f64; 2]> = if n > 3 && e.points[0] == e.points[n - 1] {
        e.points[..n - 1].to_vec()
    } else {
        e.points.clone()
    };

    if pts.len() < 3 {
        return;
    }

    // Shift by origin before triangulation so earcutr works in local space.
    let local: Vec<f64> = pts
        .iter()
        .flat_map(|p| [p[0] - origin[0], p[1] - origin[1]])
        .collect();
    let Ok(indices) = earcutr::earcut(&local, &[], 2) else {
        return;
    };

    let c = e.color;
    for tri in indices.chunks_exact(3) {
        for &idx in tri {
            out.push(Vertex::new(
                local[idx * 2] as f32,
                local[idx * 2 + 1] as f32,
                c,
            ));
        }
    }
}

fn tessellate_polygon_outline(e: &PolygonEntity, origin: [f64; 2], out: &mut Vec<Vertex>) {
    let n = e.points.len();
    if n < 2 {
        return;
    }
    let seg_n = if n > 1 && e.points[0] == e.points[n - 1] {
        n - 1
    } else {
        n
    };
    for i in 0..seg_n {
        let a = e.points[i];
        let b = e.points[(i + 1) % n];
        out.push(Vertex::new(
            rel(a[0], origin[0]),
            rel(a[1], origin[1]),
            e.color,
        ));
        out.push(Vertex::new(
            rel(b[0], origin[0]),
            rel(b[1], origin[1]),
            e.color,
        ));
    }
}

fn tessellate_bounds(
    bounds: &BoundingBox,
    color: [f32; 4],
    origin: [f64; 2],
    out: &mut Vec<Vertex>,
) {
    if !bounds.is_valid() {
        return;
    }
    let min = bounds.min;
    let max = bounds.max;
    let corners = [
        [min[0], min[1]],
        [max[0], min[1]],
        [max[0], max[1]],
        [min[0], max[1]],
    ];
    for i in 0..4 {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        out.push(Vertex::new(
            rel(a[0], origin[0]),
            rel(a[1], origin[1]),
            color,
        ));
        out.push(Vertex::new(
            rel(b[0], origin[0]),
            rel(b[1], origin[1]),
            color,
        ));
    }
}

fn tessellate_line(e: &LineEntity, origin: [f64; 2], out: &mut Vec<Vertex>) {
    let c = e.color;
    out.push(Vertex::new(
        rel(e.start[0], origin[0]),
        rel(e.start[1], origin[1]),
        c,
    ));
    out.push(Vertex::new(
        rel(e.end[0], origin[0]),
        rel(e.end[1], origin[1]),
        c,
    ));
}

fn circle_segments(radius: f64, pixels_per_unit: f64) -> u32 {
    let screen_r = radius * pixels_per_unit;
    let n = (PI * screen_r / 2.0).ceil() as u32;
    n.clamp(12, 256)
}

fn tessellate_circle(
    e: &CircleEntity,
    pixels_per_unit: f64,
    origin: [f64; 2],
    out: &mut Vec<Vertex>,
) {
    let n = circle_segments(e.radius, pixels_per_unit);
    let c = e.color;
    let cx = e.center[0] - origin[0];
    let cy = e.center[1] - origin[1];
    let r = e.radius;
    for i in 0..n {
        let a0 = TAU * i as f64 / n as f64;
        let a1 = TAU * (i + 1) as f64 / n as f64;
        out.push(Vertex::new(
            (cx + r * a0.cos()) as f32,
            (cy + r * a0.sin()) as f32,
            c,
        ));
        out.push(Vertex::new(
            (cx + r * a1.cos()) as f32,
            (cy + r * a1.sin()) as f32,
            c,
        ));
    }
}

fn tessellate_arc(e: &ArcEntity, pixels_per_unit: f64, origin: [f64; 2], out: &mut Vec<Vertex>) {
    let start = e.start_angle;
    let mut end = e.end_angle;
    if end <= start {
        end += TAU;
    }
    let sweep = end - start;
    let n = circle_segments(e.radius, pixels_per_unit);
    let seg_count = ((sweep / TAU) * n as f64).ceil() as u32;
    let seg_count = seg_count.max(2);
    let c = e.color;
    let cx = e.center[0] - origin[0];
    let cy = e.center[1] - origin[1];
    let r = e.radius;
    for i in 0..seg_count {
        let a0 = start + sweep * i as f64 / seg_count as f64;
        let a1 = start + sweep * (i + 1) as f64 / seg_count as f64;
        out.push(Vertex::new(
            (cx + r * a0.cos()) as f32,
            (cy + r * a0.sin()) as f32,
            c,
        ));
        out.push(Vertex::new(
            (cx + r * a1.cos()) as f32,
            (cy + r * a1.sin()) as f32,
            c,
        ));
    }
}

fn tessellate_polyline(
    e: &PolylineEntity,
    pixels_per_unit: f64,
    origin: [f64; 2],
    out: &mut Vec<Vertex>,
) {
    let n = e.points.len();
    if n < 2 {
        return;
    }
    let c = e.color;
    let seg_count = if e.closed { n } else { n - 1 };

    for i in 0..seg_count {
        let j = (i + 1) % n;
        let bulge = e.bulges.get(i).copied().unwrap_or(0.0);

        if bulge.abs() < 1e-10 {
            let p0 = e.points[i];
            let p1 = e.points[j];
            out.push(Vertex::new(rel(p0[0], origin[0]), rel(p0[1], origin[1]), c));
            out.push(Vertex::new(rel(p1[0], origin[0]), rel(p1[1], origin[1]), c));
        } else {
            tessellate_bulge_arc(
                e.points[i],
                e.points[j],
                bulge,
                pixels_per_unit,
                origin,
                c,
                out,
            );
        }
    }
}

/// Convert a bulge arc segment to line segments.
/// bulge = tan(included_angle / 4)
fn tessellate_bulge_arc(
    p0: [f64; 2],
    p1: [f64; 2],
    bulge: f64,
    pixels_per_unit: f64,
    origin: [f64; 2],
    color: [f32; 4],
    out: &mut Vec<Vertex>,
) {
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let d = (dx * dx + dy * dy).sqrt();
    if d < 1e-12 {
        return;
    }

    let angle = 4.0 * bulge.atan(); // included angle (signed)
    let r = d / (2.0 * (angle / 2.0).sin().abs()); // radius

    let mx = (p0[0] + p1[0]) * 0.5;
    let my = (p0[1] + p1[1]) * 0.5;
    let perp_x = -dy / d;
    let perp_y = dx / d;
    let sagitta = r - (r * r - (d / 2.0) * (d / 2.0)).sqrt();
    let sign = if bulge > 0.0 { 1.0 } else { -1.0 };
    let offset = r - sign * sagitta;
    let cx = mx - sign * perp_x * offset;
    let cy = my - sign * perp_y * offset;

    let start_a = (p0[1] - cy).atan2(p0[0] - cx);
    let end_a = (p1[1] - cy).atan2(p1[0] - cx);

    let mut sweep = if bulge > 0.0 {
        let mut s = end_a - start_a;
        if s <= 0.0 { s += TAU; }
        s
    } else {
        let mut s = end_a - start_a;
        if s >= 0.0 { s -= TAU; }
        s
    };
    if sweep == 0.0 {
        sweep = angle;
    }

    let n = circle_segments(r.abs(), pixels_per_unit).max(4);
    let seg_count = ((sweep.abs() / TAU) * n as f64).ceil() as u32;
    let seg_count = seg_count.max(2);

    for i in 0..seg_count {
        let a0 = start_a + sweep * i as f64 / seg_count as f64;
        let a1 = start_a + sweep * (i + 1) as f64 / seg_count as f64;
        let x0 = cx + r * a0.cos();
        let y0 = cy + r * a0.sin();
        let x1 = cx + r * a1.cos();
        let y1 = cy + r * a1.sin();
        out.push(Vertex::new(
            (x0 - origin[0]) as f32,
            (y0 - origin[1]) as f32,
            color,
        ));
        out.push(Vertex::new(
            (x1 - origin[0]) as f32,
            (y1 - origin[1]) as f32,
            color,
        ));
    }
}

fn tessellate_ellipse(
    e: &EllipseEntity,
    pixels_per_unit: f64,
    origin: [f64; 2],
    out: &mut Vec<Vertex>,
) {
    let major_len = (e.major_axis[0].powi(2) + e.major_axis[1].powi(2)).sqrt();
    let minor_len = major_len * e.minor_ratio;
    let screen_r = major_len.max(minor_len) * pixels_per_unit;
    let n = ((PI * screen_r / 2.0).ceil() as u32).clamp(12, 256);

    let start = e.start_param;
    let mut end = e.end_param;
    if end <= start {
        end += TAU;
    }
    let sweep = end - start;
    let seg_count = ((sweep / TAU) * n as f64).ceil() as u32;
    let seg_count = seg_count.max(2);

    let rot = e.major_axis[1].atan2(e.major_axis[0]);
    let cos_r = rot.cos();
    let sin_r = rot.sin();
    let a = major_len;
    let b = minor_len;
    let cx = e.center[0] - origin[0];
    let cy = e.center[1] - origin[1];
    let c = e.color;

    for i in 0..seg_count {
        let t0 = start + sweep * i as f64 / seg_count as f64;
        let t1 = start + sweep * (i + 1) as f64 / seg_count as f64;

        let (lx0, ly0) = (a * t0.cos(), b * t0.sin());
        let (lx1, ly1) = (a * t1.cos(), b * t1.sin());

        let x0 = cx + cos_r * lx0 - sin_r * ly0;
        let y0 = cy + sin_r * lx0 + cos_r * ly0;
        let x1 = cx + cos_r * lx1 - sin_r * ly1;
        let y1 = cy + sin_r * lx1 + cos_r * ly1;

        out.push(Vertex::new(x0 as f32, y0 as f32, c));
        out.push(Vertex::new(x1 as f32, y1 as f32, c));
    }
}

/// B-spline evaluation using De Boor's algorithm.
fn tessellate_spline(e: &SplineEntity, origin: [f64; 2], out: &mut Vec<Vertex>) {
    if e.control_points.len() < 2 || e.knots.is_empty() {
        return;
    }
    let n_pts = 128.max(e.control_points.len() * 8);
    let c = e.color;

    let t_min = *e.knots.first().unwrap();
    let t_max = *e.knots.last().unwrap();
    if (t_max - t_min).abs() < 1e-12 {
        return;
    }

    let mut prev: Option<[f32; 2]> = None;
    for i in 0..=n_pts {
        let t = t_min + (t_max - t_min) * i as f64 / n_pts as f64;
        let pt = de_boor(e.degree as usize, &e.knots, &e.control_points, t);
        let rx = (pt[0] - origin[0]) as f32;
        let ry = (pt[1] - origin[1]) as f32;
        if let Some(p0) = prev {
            out.push(Vertex::new(p0[0], p0[1], c));
            out.push(Vertex::new(rx, ry, c));
        }
        prev = Some([rx, ry]);
    }
}

/// De Boor's algorithm for B-spline point evaluation.
fn de_boor(degree: usize, knots: &[f64], ctrl: &[[f64; 2]], t: f64) -> [f64; 2] {
    let n = ctrl.len();
    if n == 0 {
        return [0.0, 0.0];
    }
    if degree == 0 {
        return ctrl[0];
    }

    let k = {
        let mut idx = degree;
        for (i, _) in knots
            .iter()
            .enumerate()
            .take(knots.len().saturating_sub(degree + 1))
            .skip(degree)
        {
            if t >= knots[i] {
                idx = i;
            }
        }
        idx.min(n - 1)
    };

    let lo = k.saturating_sub(degree);
    let hi = (k + 1).min(n);
    let mut d: Vec<[f64; 2]> = ctrl[lo..hi].to_vec();

    for r in 1..=degree {
        for j in (r..d.len()).rev() {
            let i = lo + j;
            let left = knots.get(i).copied().unwrap_or(0.0);
            let right = knots.get(i + degree - r + 1).copied().unwrap_or(0.0);
            let denom = right - left;
            let alpha = if denom.abs() < 1e-12 {
                0.0
            } else {
                (t - left) / denom
            };
            d[j][0] = (1.0 - alpha) * d[j - 1][0] + alpha * d[j][0];
            d[j][1] = (1.0 - alpha) * d[j - 1][1] + alpha * d[j][1];
        }
    }

    *d.last().unwrap_or(&[0.0, 0.0])
}

// ── GDS hierarchical tessellation ────────────────────────────────────────────

/// Fixed segment count used when tessellating circles/arcs into cell-local cache geometry.
/// Higher = smoother at very high zoom; the same vertex data is reused for all instances.
const CELL_CACHE_CIRCLE_SEGS: u32 = 64;

/// If the world-space bbox is smaller than this threshold in screen pixels, skip
/// or draw as a colored rectangle (LOD) instead of expanding geometry.
/// Higher value = more aggressive LOD → better performance for dense GDS files.
const GDS_LOD_PX: f64 = 4.0;

#[inline]
fn is_sub_lod(world_bbox: &BoundingBox, pixels_per_unit: f64) -> bool {
    if !world_bbox.is_valid() {
        return true;
    }
    world_bbox.width() * pixels_per_unit < GDS_LOD_PX
        && world_bbox.height() * pixels_per_unit < GDS_LOD_PX
}

// ── Cell primitive cache helpers ──────────────────────────────────────────────

/// Build and insert a cache entry for `cell_idx` if not already present.
fn ensure_cell_cached(
    cell_idx: usize,
    fill_polygons: bool,
    gds_cells: &[GdsCellDef],
    hidden_layers: &HashSet<String>,
    cache: &mut CellCache,
) {
    let key = (cell_idx, fill_polygons);
    if cache.contains_key(&key) {
        return;
    }
    let entry = build_primitive_cache(cell_idx, fill_polygons, gds_cells, hidden_layers);
    cache.insert(key, Arc::new(entry));
}

/// Pre-tessellate all leaf primitives (polygons/polylines) in `cell_idx` into
/// cell-local `LocalVertex` buffers.  Sub-instance references are NOT expanded
/// here — they recurse at render time so LOD decisions stay correct.
/// Primitives on layers in `hidden_layers` are excluded.
fn build_primitive_cache(
    cell_idx: usize,
    fill_polygons: bool,
    gds_cells: &[GdsCellDef],
    hidden_layers: &HashSet<String>,
) -> CellCacheEntry {
    let Some(cell) = gds_cells.get(cell_idx) else {
        return CellCacheEntry { lines: vec![], fills: vec![] };
    };
    let mut lines: Vec<LocalVertex> = Vec::new();
    let mut fills: Vec<LocalVertex> = Vec::new();
    for entity in &cell.entities {
        match entity {
            CadEntity::Polygon(p) => {
                if hidden_layers.contains(&p.layer) { continue; }
                if fill_polygons {
                    collect_polygon_local(p, &mut fills);
                } else {
                    collect_polygon_outline_local(p, &mut lines);
                }
            }
            CadEntity::Polyline(p) => {
                if !hidden_layers.contains(&p.layer) {
                    collect_polyline_local(p, &mut lines);
                }
            }
            // DXF primitive types — tessellated at fixed quality into cell-local coords
            CadEntity::Line(e) => {
                if !hidden_layers.contains(&e.layer) {
                    collect_line_local(e, &mut lines);
                }
            }
            CadEntity::Circle(e) => {
                if !hidden_layers.contains(&e.layer) {
                    collect_circle_local(e, &mut lines);
                }
            }
            CadEntity::Arc(e) => {
                if !hidden_layers.contains(&e.layer) {
                    collect_arc_local(e, &mut lines);
                }
            }
            CadEntity::Ellipse(e) => {
                if !hidden_layers.contains(&e.layer) {
                    collect_ellipse_local(e, &mut lines);
                }
            }
            CadEntity::Spline(e) => {
                if !hidden_layers.contains(&e.layer) {
                    collect_spline_local(e, &mut lines);
                }
            }
            CadEntity::Text(_) => {} // text skipped in cell cache (egui overlay)
            // Sub-instances recurse at render time
            CadEntity::GdsInstance(_)
            | CadEntity::GdsArrayInstance(_)
            | CadEntity::DxfInsert(_) => {}
        }
    }
    CellCacheEntry { lines, fills }
}

/// Register a cell's primitive geometry in `output.cell_geometries` if not already present.
/// This is a no-op after the first call for a given (cell_idx, fill_polygons) pair in a session.
#[inline]
fn register_cell_geometry(
    cell_idx: usize,
    fill_polygons: bool,
    cache: &CellCache,
    output: &mut RenderOutput,
) {
    if output.cell_geometries.contains_key(&cell_idx) {
        return; // already registered this session
    }
    if let Some(entry) = cache.get(&(cell_idx, fill_polygons)) {
        if !entry.lines.is_empty() || !entry.fills.is_empty() {
            output.cell_geometries.entry(cell_idx).or_insert_with(|| Arc::clone(entry));
        }
    }
}

/// Tessellate a single GDS cell instance (SREF).
/// Instead of baking transforms into vertices, this registers a `GpuTransform`
/// so the GPU shader applies the transform once per instance.
fn tessellate_gds_instance(
    inst: &GdsInstanceEntity,
    parent_t: &InstanceTransform,
    pixels_per_unit: f64,
    origin: [f64; 2],
    fill_polygons: bool,
    gds_cells: &[GdsCellDef],
    cache: &mut CellCache,
    output: &mut RenderOutput,
    depth: usize,
    tess_viewport: Option<&BoundingBox>,
    hidden_layers: &HashSet<String>,
) {
    if depth > 64 {
        return;
    }
    let world_bbox = parent_t.apply_bbox(&inst.bbox);

    if is_sub_lod(&world_bbox, pixels_per_unit) {
        tessellate_bounds(&world_bbox, inst.color, origin, &mut output.misc_lines);
        return;
    }

    if let Some(vp) = tess_viewport {
        if !world_bbox.intersects(vp) {
            return;
        }
    }

    let child_t = parent_t.compose(inst.offset, inst.angle_deg, inst.mag, inst.reflected);

    // Build the primitive cache entry if not yet present.
    ensure_cell_cached(inst.cell_idx, fill_polygons, gds_cells, hidden_layers, cache);

    // Register cell geometry for GPU upload (done once per cell per session).
    register_cell_geometry(inst.cell_idx, fill_polygons, cache, output);

    // Push one instance transform (the GPU applies it to the cell's geometry).
    if output.cell_geometries.contains_key(&inst.cell_idx) {
        output
            .instances
            .entry(inst.cell_idx)
            .or_default()
            .push(GpuTransform::from_transform(&child_t, origin));
    }

    // Recurse into sub-instances (they each get their own instance transforms).
    if let Some(cell) = gds_cells.get(inst.cell_idx) {
        tessellate_cell_sub_instances(
            cell,
            &child_t,
            pixels_per_unit,
            origin,
            fill_polygons,
            gds_cells,
            cache,
            output,
            depth + 1,
            tess_viewport,
            hidden_layers,
        );
    }
}

/// Tessellate a GDS array reference (AREF).
///
/// With GPU instancing, this accumulates one `GpuTransform` per visible element
/// instead of baking transforms into vertices.  Memory cost: 24 bytes per element
/// regardless of cell complexity.
///
/// Retained optimisations:
/// 1. Whole-array viewport cull.
/// 2. Precomputed rotated-cell bbox for O(1) per-element cull.
/// 3. O(1) visible range for axis-aligned arrays.
/// 4. Rayon parallel transform generation for leaf cells.
fn tessellate_gds_array_instance(
    arr: &GdsArrayInstanceEntity,
    parent_t: &InstanceTransform,
    pixels_per_unit: f64,
    origin: [f64; 2],
    fill_polygons: bool,
    gds_cells: &[GdsCellDef],
    cache: &mut CellCache,
    output: &mut RenderOutput,
    depth: usize,
    tess_viewport: Option<&BoundingBox>,
    hidden_layers: &HashSet<String>,
) {
    if depth > 64 {
        return;
    }

    let array_world = parent_t.apply_bbox(&arr.bbox);

    if is_sub_lod(&array_world, pixels_per_unit) {
        tessellate_bounds(&array_world, arr.color, origin, &mut output.misc_lines);
        return;
    }

    // ① Whole-array viewport cull.
    if let Some(vp) = tess_viewport {
        if !array_world.intersects(vp) {
            return;
        }
    }

    let Some(cell) = gds_cells.get(arr.cell_idx) else {
        return;
    };

    // If even a single element is sub-LOD, draw the whole array as one box.
    if let Some(cell_bbox) = &cell.bbox {
        let elem_t = parent_t.compose(arr.origin, arr.angle_deg, arr.mag, arr.reflected);
        if is_sub_lod(&elem_t.apply_bbox(cell_bbox), pixels_per_unit) {
            tessellate_bounds(&array_world, arr.color, origin, &mut output.misc_lines);
            return;
        }
    }

    ensure_cell_cached(arr.cell_idx, fill_polygons, gds_cells, hidden_layers, cache);
    register_cell_geometry(arr.cell_idx, fill_polygons, cache, output);

    // ② Precompute rotated cell bbox (no per-element trig needed for culling).
    let rcbb: Option<BoundingBox> = cell.bbox.as_ref().map(|cb| {
        InstanceTransform {
            offset: [0.0, 0.0],
            angle_deg: parent_t.angle_deg + arr.angle_deg,
            mag: parent_t.mag * arr.mag,
            reflected: parent_t.reflected ^ arr.reflected,
        }
        .apply_bbox(cb)
    });

    // ③ O(1) visible range for axis-aligned arrays.
    let (col_start, col_end, row_start, row_end) =
        aref_visible_range(arr, parent_t, tess_viewport, rcbb.as_ref());

    let has_sub_instances = cell.entities.iter().any(|e| {
        matches!(e, CadEntity::GdsInstance(_) | CadEntity::GdsArrayInstance(_))
    });

    if !has_sub_instances && output.cell_geometries.contains_key(&arr.cell_idx) {
        // ④ Rayon path: collect instance transforms in parallel.
        let parent_t_c = parent_t.clone();
        let arr_origin = arr.origin;
        let arr_col_step = arr.col_step;
        let arr_row_step = arr.row_step;
        let arr_angle = arr.angle_deg;
        let arr_mag = arr.mag;
        let arr_refl = arr.reflected;
        let rcbb_c = rcbb.clone();
        let vp_c: Option<BoundingBox> = tess_viewport.cloned();
        let origin_c = origin;

        let transforms: Vec<GpuTransform> = (row_start..row_end)
            .into_par_iter()
            .flat_map_iter(|row| {
                let parent_t_r = parent_t_c.clone();
                let rcbb_r = rcbb_c.clone();
                let vp_r = vp_c.clone();
                (col_start..col_end).filter_map(move |col| {
                    let raw_off = [
                        arr_origin[0]
                            + col as f64 * arr_col_step[0]
                            + row as f64 * arr_row_step[0],
                        arr_origin[1]
                            + col as f64 * arr_col_step[1]
                            + row as f64 * arr_row_step[1],
                    ];
                    if let (Some(vp), Some(rcbb)) = (&vp_r, &rcbb_r) {
                        let wo = parent_t_r.apply(raw_off);
                        if rcbb.max[0] + wo[0] < vp.min[0]
                            || rcbb.min[0] + wo[0] > vp.max[0]
                            || rcbb.max[1] + wo[1] < vp.min[1]
                            || rcbb.min[1] + wo[1] > vp.max[1]
                        {
                            return None;
                        }
                    }
                    let child_t = parent_t_r.compose(raw_off, arr_angle, arr_mag, arr_refl);
                    Some(GpuTransform::from_transform(&child_t, origin_c))
                })
            })
            .collect();

        output
            .instances
            .entry(arr.cell_idx)
            .or_default()
            .extend(transforms);
    } else {
        // Sequential path: cells with sub-instances, or no geometry to instance.
        for row in row_start..row_end {
            for col in col_start..col_end {
                let raw_off = [
                    arr.origin[0]
                        + col as f64 * arr.col_step[0]
                        + row as f64 * arr.row_step[0],
                    arr.origin[1]
                        + col as f64 * arr.col_step[1]
                        + row as f64 * arr.row_step[1],
                ];

                if let (Some(vp), Some(rcbb)) = (tess_viewport, &rcbb) {
                    let wo = parent_t.apply(raw_off);
                    if rcbb.max[0] + wo[0] < vp.min[0]
                        || rcbb.min[0] + wo[0] > vp.max[0]
                        || rcbb.max[1] + wo[1] < vp.min[1]
                        || rcbb.min[1] + wo[1] > vp.max[1]
                    {
                        continue;
                    }
                }

                let child_t =
                    parent_t.compose(raw_off, arr.angle_deg, arr.mag, arr.reflected);

                if output.cell_geometries.contains_key(&arr.cell_idx) {
                    output
                        .instances
                        .entry(arr.cell_idx)
                        .or_default()
                        .push(GpuTransform::from_transform(&child_t, origin));
                }

                tessellate_cell_sub_instances(
                    cell,
                    &child_t,
                    pixels_per_unit,
                    origin,
                    fill_polygons,
                    gds_cells,
                    cache,
                    output,
                    depth + 1,
                    tess_viewport,
                    hidden_layers,
                );
            }
        }
    }
}

/// Compute the visible (col_start, col_end, row_start, row_end) range for an AREF.
/// Returns the full range `(0, cols, 0, rows)` when:
///   - no viewport is given, or
///   - the array is not axis-aligned (per-element cull is still applied).
fn aref_visible_range(
    arr: &GdsArrayInstanceEntity,
    parent_t: &InstanceTransform,
    tess_viewport: Option<&BoundingBox>,
    rcbb: Option<&BoundingBox>,
) -> (u32, u32, u32, u32) {
    let full = (0, arr.cols, 0, arr.rows);
    let Some(vp) = tess_viewport else {
        return full;
    };
    let Some(rcbb) = rcbb else {
        return full;
    };

    // Only apply O(1) range to axis-aligned arrays.
    // Conditions: combined angle ≈ 0, no reflection, mag ≈ 1,
    //             col_step is horizontal, row_step is vertical.
    let combined_angle = parent_t.angle_deg + arr.angle_deg;
    let is_aa = combined_angle.abs() < 0.01 // degrees
        && !parent_t.reflected
        && !arr.reflected
        && (parent_t.mag - 1.0).abs() < 1e-6
        && (arr.mag - 1.0).abs() < 1e-6
        && arr.col_step[1].abs() < 1e-6
        && arr.row_step[0].abs() < 1e-6;

    if !is_aa {
        return full;
    }

    // For axis-aligned arrays element (col, row) world origin =
    //   base_x + col * col_step[0],  base_y + row * row_step[1]
    let base_x = arr.origin[0] + parent_t.offset[0];
    let base_y = arr.origin[1] + parent_t.offset[1];
    let cs = arr.col_step[0];
    let rs = arr.row_step[1];

    let (col_start, col_end) = range_1d(
        vp.min[0] - rcbb.max[0],
        vp.max[0] - rcbb.min[0],
        base_x,
        cs,
        arr.cols,
    );
    let (row_start, row_end) = range_1d(
        vp.min[1] - rcbb.max[1],
        vp.max[1] - rcbb.min[1],
        base_y,
        rs,
        arr.rows,
    );

    (col_start, col_end, row_start, row_end)
}

/// Compute the visible [start, end) index range along one axis of a 1D grid.
///
/// `vp_lo` / `vp_hi` — viewport interval already shrunk by the cell half-extent
///   (so `base + i*step` being inside [vp_lo, vp_hi] means the element is visible).
fn range_1d(vp_lo: f64, vp_hi: f64, base: f64, step: f64, count: u32) -> (u32, u32) {
    if count == 0 {
        return (0, 0);
    }
    if step.abs() < 1e-12 {
        // All elements at the same position — include all if any are visible.
        return (0, count);
    }

    let t_lo = (vp_lo - base) / step;
    let t_hi = (vp_hi - base) / step;
    let (t_min, t_max) = if step > 0.0 { (t_lo, t_hi) } else { (t_hi, t_lo) };

    let start = (t_min.floor() as i64).clamp(0, count as i64) as u32;
    let end = ((t_max.ceil() as i64) + 1).clamp(0, count as i64) as u32;
    (start, end.max(start))
}

/// Iterate over a cell's entities and recurse only into GdsInstance / GdsArrayInstance
/// children. Leaf primitives are in `output.cell_geometries` and drawn via instancing.
fn tessellate_cell_sub_instances(
    cell: &GdsCellDef,
    t: &InstanceTransform,
    pixels_per_unit: f64,
    origin: [f64; 2],
    fill_polygons: bool,
    gds_cells: &[GdsCellDef],
    cache: &mut CellCache,
    output: &mut RenderOutput,
    depth: usize,
    tess_viewport: Option<&BoundingBox>,
    hidden_layers: &HashSet<String>,
) {
    if depth > 64 {
        return;
    }
    for (entity, local_bb) in cell.entities.iter().zip(cell.entity_bounds.iter()) {
        match entity {
            CadEntity::GdsInstance(inst) => {
                let world_bb = t.apply_bbox(local_bb);
                if !is_sub_lod(&world_bb, pixels_per_unit) {
                    tessellate_gds_instance(
                        inst,
                        t,
                        pixels_per_unit,
                        origin,
                        fill_polygons,
                        gds_cells,
                        cache,
                        output,
                        depth,
                        tess_viewport,
                        hidden_layers,
                    );
                }
            }
            CadEntity::GdsArrayInstance(arr) => {
                let world_bb = t.apply_bbox(local_bb);
                if !is_sub_lod(&world_bb, pixels_per_unit) {
                    tessellate_gds_array_instance(
                        arr,
                        t,
                        pixels_per_unit,
                        origin,
                        fill_polygons,
                        gds_cells,
                        cache,
                        output,
                        depth,
                        tess_viewport,
                        hidden_layers,
                    );
                }
            }
            CadEntity::DxfInsert(ins) => {
                // World bbox of this insert, accounting for the parent cell's transform.
                // We use a conservative estimate via the parent's apply_bbox on the local_bb
                // (which was computed at load time from the insert's pre-transformed bbox).
                let world_bb = t.apply_bbox(local_bb);
                if is_sub_lod(&world_bb, pixels_per_unit) {
                    tessellate_bounds(&world_bb, ins.color, origin, &mut output.misc_lines);
                } else {
                    ensure_cell_cached(ins.block_idx, fill_polygons, gds_cells, hidden_layers, cache);
                    register_cell_geometry(ins.block_idx, fill_polygons, cache, output);
                    if output.cell_geometries.contains_key(&ins.block_idx) {
                        for col in 0..ins.cols {
                            for row in 0..ins.rows {
                                let ins_offset = [
                                    ins.offset[0] + col as f64 * ins.col_step[0] + row as f64 * ins.row_step[0],
                                    ins.offset[1] + col as f64 * ins.col_step[1] + row as f64 * ins.row_step[1],
                                ];
                                output.instances
                                    .entry(ins.block_idx)
                                    .or_default()
                                    .push(GpuTransform::from_parent_and_dxf_insert(
                                        t,
                                        ins_offset,
                                        ins.angle_deg,
                                        ins.x_scale,
                                        ins.y_scale,
                                        origin,
                                    ));
                            }
                        }
                    }
                    // Note: recursion into DxfInsert sub-instances is not performed here
                    // because composing a GDS InstanceTransform with a non-uniform DXF scale
                    // produces a general affine that cannot be represented as InstanceTransform.
                    // This limitation only affects blocks nested 3+ levels deep with mixed
                    // GDS/DXF scaling — rare in practice.
                }
            }
            _ => {} // Line/Circle/Arc/Polygon/Polyline/Text served from GPU cell geometry
        }
    }
}


// ── DXF INSERT tessellation ───────────────────────────────────────────────────

/// Tessellate a DXF INSERT entity via GPU instancing.
/// Block geometry is uploaded once; each grid element contributes one `GpuTransform`.
fn tessellate_dxf_insert(
    ins: &DxfInsertEntity,
    pixels_per_unit: f64,
    origin: [f64; 2],
    fill_polygons: bool,
    gds_cells: &[GdsCellDef],
    cache: &mut CellCache,
    output: &mut RenderOutput,
    tess_viewport: Option<&BoundingBox>,
    hidden_layers: &HashSet<String>,
) {
    // Whole-insert viewport cull
    if ins.bbox.is_valid() {
        if let Some(vp) = tess_viewport {
            if !ins.bbox.intersects(vp) {
                return;
            }
        }
        // LOD: entire insert too small → draw as bounding-box outline
        if is_sub_lod(&ins.bbox, pixels_per_unit) {
            tessellate_bounds(&ins.bbox, ins.color, origin, &mut output.misc_lines);
            return;
        }
    }

    ensure_cell_cached(ins.block_idx, fill_polygons, gds_cells, hidden_layers, cache);
    register_cell_geometry(ins.block_idx, fill_polygons, cache, output);

    if output.cell_geometries.contains_key(&ins.block_idx) {
        for col in 0..ins.cols {
            for row in 0..ins.rows {
                let offset = [
                    ins.offset[0] + col as f64 * ins.col_step[0] + row as f64 * ins.row_step[0],
                    ins.offset[1] + col as f64 * ins.col_step[1] + row as f64 * ins.row_step[1],
                ];
                output.instances
                    .entry(ins.block_idx)
                    .or_default()
                    .push(GpuTransform::from_dxf_insert(
                        offset,
                        ins.angle_deg,
                        ins.x_scale,
                        ins.y_scale,
                        origin,
                    ));
            }
        }
    }

    // Recurse into the block's nested sub-instances (DxfInsert/GdsInstance children).
    if let Some(cell) = gds_cells.get(ins.block_idx) {
        // Build a parent InstanceTransform for the cell at identity (top-level DxfInsert
        // always places block geometry directly; the GpuTransform already encodes the
        // full affine). We pass per-element transforms individually for recursion.
        for col in 0..ins.cols {
            for row in 0..ins.rows {
                let offset = [
                    ins.offset[0] + col as f64 * ins.col_step[0] + row as f64 * ins.row_step[0],
                    ins.offset[1] + col as f64 * ins.col_step[1] + row as f64 * ins.row_step[1],
                ];
                // Approximate parent InstanceTransform for sub-instance LOD/viewport decisions.
                // Uses x_scale as uniform mag — correct when x_scale == y_scale (common case).
                let approx_parent = InstanceTransform {
                    offset,
                    angle_deg: ins.angle_deg,
                    mag: ins.x_scale,
                    reflected: false,
                };
                tessellate_cell_sub_instances(
                    cell,
                    &approx_parent,
                    pixels_per_unit,
                    origin,
                    fill_polygons,
                    gds_cells,
                    cache,
                    output,
                    1,
                    tess_viewport,
                    hidden_layers,
                );
            }
        }
    }
}

// ── Cell-local geometry collectors (used to build CellCacheEntry) ─────────────

/// Triangulate `e` in cell-local coords and append triangle vertices to `out`.
/// earcut runs once here; the result is reused for every instance of this cell.
fn collect_polygon_local(e: &PolygonEntity, out: &mut Vec<LocalVertex>) {
    let n = e.points.len();
    if n < 3 {
        return;
    }
    let pts: &[[f64; 2]] = if n > 3 && e.points[0] == e.points[n - 1] {
        &e.points[..n - 1]
    } else {
        &e.points
    };
    if pts.len() < 3 {
        return;
    }
    let flat: Vec<f64> = pts.iter().flat_map(|p| [p[0], p[1]]).collect();
    let Ok(indices) = earcutr::earcut(&flat, &[], 2) else {
        return;
    };
    let c = e.color;
    for tri in indices.chunks_exact(3) {
        for &idx in tri {
            out.push(LocalVertex {
                pos: [flat[idx * 2], flat[idx * 2 + 1]],
                color: c,
            });
        }
    }
}

/// Collect polygon outline segments in cell-local coords.
fn collect_polygon_outline_local(e: &PolygonEntity, out: &mut Vec<LocalVertex>) {
    let n = e.points.len();
    if n < 2 {
        return;
    }
    let seg_n = if n > 1 && e.points[0] == e.points[n - 1] {
        n - 1
    } else {
        n
    };
    let c = e.color;
    for i in 0..seg_n {
        out.push(LocalVertex { pos: e.points[i], color: c });
        out.push(LocalVertex { pos: e.points[(i + 1) % n], color: c });
    }
}

/// Collect polyline segments in cell-local coords (handles bulge arcs).
fn collect_polyline_local(e: &PolylineEntity, out: &mut Vec<LocalVertex>) {
    let n = e.points.len();
    if n < 2 {
        return;
    }
    let seg_n = if e.closed { n } else { n - 1 };
    let c = e.color;
    for i in 0..seg_n {
        let j = (i + 1) % n;
        let bulge = e.bulges.get(i).copied().unwrap_or(0.0);
        if bulge.abs() < 1e-10 {
            out.push(LocalVertex { pos: e.points[i], color: c });
            out.push(LocalVertex { pos: e.points[j], color: c });
        } else {
            collect_bulge_arc_local(e.points[i], e.points[j], bulge, c, out);
        }
    }
}

fn collect_bulge_arc_local(
    p0: [f64; 2],
    p1: [f64; 2],
    bulge: f64,
    color: [f32; 4],
    out: &mut Vec<LocalVertex>,
) {
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let d = (dx * dx + dy * dy).sqrt();
    if d < 1e-12 {
        return;
    }
    let angle = 4.0 * bulge.atan();
    let r = d / (2.0 * (angle / 2.0).sin().abs());
    let mx = (p0[0] + p1[0]) * 0.5;
    let my = (p0[1] + p1[1]) * 0.5;
    let perp_x = -dy / d;
    let perp_y = dx / d;
    let sagitta = r - (r * r - (d / 2.0) * (d / 2.0)).sqrt();
    let sign = if bulge > 0.0 { 1.0 } else { -1.0 };
    let offset = r - sign * sagitta;
    let cx = mx - sign * perp_x * offset;
    let cy = my - sign * perp_y * offset;
    let start_a = (p0[1] - cy).atan2(p0[0] - cx);
    let end_a = (p1[1] - cy).atan2(p1[0] - cx);
    let mut sweep = if bulge > 0.0 {
        let mut s = end_a - start_a;
        if s <= 0.0 { s += TAU; }
        s
    } else {
        let mut s = end_a - start_a;
        if s >= 0.0 { s -= TAU; }
        s
    };
    if sweep == 0.0 {
        sweep = angle;
    }
    let segs = ((sweep.abs() / TAU) * CELL_CACHE_CIRCLE_SEGS as f64).ceil() as u32;
    let segs = segs.max(2);
    for i in 0..segs {
        let a0 = start_a + sweep * i as f64 / segs as f64;
        let a1 = start_a + sweep * (i + 1) as f64 / segs as f64;
        out.push(LocalVertex { pos: [cx + r * a0.cos(), cy + r * a0.sin()], color });
        out.push(LocalVertex { pos: [cx + r * a1.cos(), cy + r * a1.sin()], color });
    }
}

fn collect_line_local(e: &LineEntity, out: &mut Vec<LocalVertex>) {
    out.push(LocalVertex { pos: e.start, color: e.color });
    out.push(LocalVertex { pos: e.end, color: e.color });
}

fn collect_circle_local(e: &CircleEntity, out: &mut Vec<LocalVertex>) {
    let n = CELL_CACHE_CIRCLE_SEGS;
    let c = e.color;
    for i in 0..n {
        let a0 = TAU * i as f64 / n as f64;
        let a1 = TAU * (i + 1) as f64 / n as f64;
        out.push(LocalVertex {
            pos: [e.center[0] + e.radius * a0.cos(), e.center[1] + e.radius * a0.sin()],
            color: c,
        });
        out.push(LocalVertex {
            pos: [e.center[0] + e.radius * a1.cos(), e.center[1] + e.radius * a1.sin()],
            color: c,
        });
    }
}

fn collect_arc_local(e: &ArcEntity, out: &mut Vec<LocalVertex>) {
    let start = e.start_angle;
    let mut end = e.end_angle;
    if end <= start {
        end += TAU;
    }
    let sweep = end - start;
    let n = ((sweep / TAU) * CELL_CACHE_CIRCLE_SEGS as f64).ceil() as u32;
    let n = n.max(2);
    let c = e.color;
    for i in 0..n {
        let a0 = start + sweep * i as f64 / n as f64;
        let a1 = start + sweep * (i + 1) as f64 / n as f64;
        out.push(LocalVertex {
            pos: [e.center[0] + e.radius * a0.cos(), e.center[1] + e.radius * a0.sin()],
            color: c,
        });
        out.push(LocalVertex {
            pos: [e.center[0] + e.radius * a1.cos(), e.center[1] + e.radius * a1.sin()],
            color: c,
        });
    }
}

fn collect_ellipse_local(e: &EllipseEntity, out: &mut Vec<LocalVertex>) {
    let major_len = (e.major_axis[0].powi(2) + e.major_axis[1].powi(2)).sqrt();
    let minor_len = major_len * e.minor_ratio;
    let start = e.start_param;
    let mut end = e.end_param;
    if end <= start {
        end += TAU;
    }
    let sweep = end - start;
    let n = ((sweep / TAU) * CELL_CACHE_CIRCLE_SEGS as f64).ceil() as u32;
    let n = n.max(2);
    let rot = e.major_axis[1].atan2(e.major_axis[0]);
    let (cos_r, sin_r) = (rot.cos(), rot.sin());
    let (a, b) = (major_len, minor_len);
    let c = e.color;
    for i in 0..n {
        let t0 = start + sweep * i as f64 / n as f64;
        let t1 = start + sweep * (i + 1) as f64 / n as f64;
        let (lx0, ly0) = (a * t0.cos(), b * t0.sin());
        let (lx1, ly1) = (a * t1.cos(), b * t1.sin());
        out.push(LocalVertex {
            pos: [e.center[0] + cos_r * lx0 - sin_r * ly0, e.center[1] + sin_r * lx0 + cos_r * ly0],
            color: c,
        });
        out.push(LocalVertex {
            pos: [e.center[0] + cos_r * lx1 - sin_r * ly1, e.center[1] + sin_r * lx1 + cos_r * ly1],
            color: c,
        });
    }
}

fn collect_spline_local(e: &SplineEntity, out: &mut Vec<LocalVertex>) {
    if e.control_points.len() < 2 || e.knots.is_empty() {
        return;
    }
    let n_pts = 128.max(e.control_points.len() * 8);
    let c = e.color;
    let t_min = *e.knots.first().unwrap();
    let t_max = *e.knots.last().unwrap();
    if (t_max - t_min).abs() < 1e-12 {
        return;
    }
    let mut prev: Option<[f64; 2]> = None;
    for i in 0..=n_pts {
        let t = t_min + (t_max - t_min) * i as f64 / n_pts as f64;
        let pt = de_boor(e.degree as usize, &e.knots, &e.control_points, t);
        if let Some(p0) = prev {
            out.push(LocalVertex { pos: p0, color: c });
            out.push(LocalVertex { pos: pt, color: c });
        }
        prev = Some(pt);
    }
}
