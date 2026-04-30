use std::collections::{HashMap, HashSet};

/// Axis-aligned bounding box in 2D world space.
#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub min: [f64; 2],
    pub max: [f64; 2],
}

impl BoundingBox {
    pub fn empty() -> Self {
        Self {
            min: [f64::MAX, f64::MAX],
            max: [f64::MIN, f64::MIN],
        }
    }

    pub fn expand(&mut self, x: f64, y: f64) {
        if x < self.min[0] {
            self.min[0] = x;
        }
        if y < self.min[1] {
            self.min[1] = y;
        }
        if x > self.max[0] {
            self.max[0] = x;
        }
        if y > self.max[1] {
            self.max[1] = y;
        }
    }

    pub fn is_valid(&self) -> bool {
        self.min[0] <= self.max[0] && self.min[1] <= self.max[1]
    }

    pub fn intersects(&self, other: &BoundingBox) -> bool {
        self.max[0] >= other.min[0]
            && self.min[0] <= other.max[0]
            && self.max[1] >= other.min[1]
            && self.min[1] <= other.max[1]
    }

    pub fn width(&self) -> f64 {
        self.max[0] - self.min[0]
    }
    pub fn height(&self) -> f64 {
        self.max[1] - self.min[1]
    }
    pub fn center(&self) -> [f64; 2] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
        ]
    }

    /// True if `self` fully encloses `other`.
    pub fn contains_bounds(&self, other: &BoundingBox) -> bool {
        self.min[0] <= other.min[0]
            && self.min[1] <= other.min[1]
            && self.max[0] >= other.max[0]
            && self.max[1] >= other.max[1]
    }

    /// Return a copy expanded by `factor` times the current size on each side.
    /// `factor = 1.0` doubles the box (100% margin on each side).
    #[allow(dead_code)]
    pub fn expanded(&self, factor: f64) -> Self {
        let hw = self.width() * factor * 0.5;
        let hh = self.height() * factor * 0.5;
        Self {
            min: [self.min[0] - hw, self.min[1] - hh],
            max: [self.max[0] + hw, self.max[1] + hh],
        }
    }
}

/// A batch of entities emitted by a background loader.
#[derive(Debug, Default)]
pub struct DrawingChunk {
    pub layers: Vec<Layer>,
    pub entities: Vec<CadEntity>,
    pub entity_bounds: Vec<BoundingBox>,
    pub bounds: Option<BoundingBox>,
}

/// Loader messages consumed by the UI thread.
#[derive(Debug)]
pub enum LoadEvent {
    Started { format: &'static str },
    /// GDS only: all cell definitions, sent once before the first Chunk event.
    GdsCells(Vec<GdsCellDef>),
    Chunk(DrawingChunk),
    Progress { loaded_entities: usize },
    Finished,
    Failed(String),
}

/// Render query result. Exact entities are used when the visible set is small
/// enough; LOD boxes summarize dense tiles when zoomed out.
pub enum RenderItem<'a> {
    Entity {
        entity: &'a CadEntity,
    },
    LodBox {
        bounds: &'a BoundingBox,
        color: [f32; 4],
    },
}

#[derive(Debug, Clone)]
pub struct LodTile {
    pub bounds: BoundingBox,
    pub layer: String,
    pub color: [f32; 4],
    pub entity_count: usize,
}

#[derive(Debug, Clone, Default)]
pub struct SpatialIndex {
    bounds: Option<BoundingBox>,
    cols: usize,
    rows: usize,
    cells: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, Default)]
pub struct LodCache {
    bounds: Option<BoundingBox>,
    cols: usize,
    rows: usize,
    tiles: Vec<LodTile>,
}

const SPATIAL_TARGET_CELL_ENTITIES: usize = 256;
const MAX_SPATIAL_DIM: usize = 256;
const LOD_GRID_DIM: usize = 96;
const EXACT_ENTITY_LIMIT: usize = 120_000;
const LOD_TRIGGER_ENTITY_COUNT: usize = 20_000;
const LOD_MIN_TILE_PIXELS: f64 = 2.0;

/// A CAD layer.
#[derive(Debug, Clone)]
pub struct Layer {
    pub name: String,
    pub color: [f32; 4],
    #[allow(dead_code)]
    pub visible: bool,
}

/// LINE entity
#[derive(Debug, Clone)]
pub struct LineEntity {
    pub start: [f64; 2],
    pub end: [f64; 2],
    pub layer: String,
    pub color: [f32; 4],
}

/// CIRCLE entity
#[derive(Debug, Clone)]
pub struct CircleEntity {
    pub center: [f64; 2],
    pub radius: f64,
    pub layer: String,
    pub color: [f32; 4],
}

/// ARC entity
#[derive(Debug, Clone)]
pub struct ArcEntity {
    pub center: [f64; 2],
    pub radius: f64,
    /// Start angle in radians (counter-clockwise from +X)
    pub start_angle: f64,
    /// End angle in radians (counter-clockwise from +X)
    pub end_angle: f64,
    pub layer: String,
    pub color: [f32; 4],
}

/// POLYLINE / LWPOLYLINE entity
#[derive(Debug, Clone)]
pub struct PolylineEntity {
    pub points: Vec<[f64; 2]>,
    /// Bulge values per segment (len == points.len() - 1, or points.len() if closed)
    pub bulges: Vec<f64>,
    pub closed: bool,
    pub layer: String,
    pub color: [f32; 4],
}

/// ELLIPSE entity
#[derive(Debug, Clone)]
pub struct EllipseEntity {
    pub center: [f64; 2],
    /// Semi-major axis vector (defines orientation and size)
    pub major_axis: [f64; 2],
    /// Ratio of minor to major axis (0..1)
    pub minor_ratio: f64,
    /// Start parameter (radians)
    pub start_param: f64,
    /// End parameter (radians)
    pub end_param: f64,
    pub layer: String,
    pub color: [f32; 4],
}

/// SPLINE entity (B-spline)
#[derive(Debug, Clone)]
pub struct SplineEntity {
    pub degree: i32,
    pub control_points: Vec<[f64; 2]>,
    pub knots: Vec<f64>,
    #[allow(dead_code)]
    pub closed: bool,
    pub layer: String,
    pub color: [f32; 4],
}

/// POLYGON entity (GDSII BOUNDARY, filled polygon)
#[derive(Debug, Clone)]
pub struct PolygonEntity {
    pub points: Vec<[f64; 2]>,
    pub layer: String,
    pub color: [f32; 4],
}

// ── GDS hierarchy types ──────────────────────────────────────────────────────

/// A GDS cell definition (one GDS struct), stored in cell-local coordinates.
/// All entity geometry uses the cell's own coordinate system — no instance transforms applied.
/// Populated only for GDS files; empty for DXF/DWG.
#[derive(Debug, Clone, Default)]
pub struct GdsCellDef {
    #[allow(dead_code)]
    pub name: String,
    /// Entities in cell-local coords. May include GdsInstance / GdsArrayInstance children.
    pub entities: Vec<CadEntity>,
    /// Per-entity bounding boxes in cell-local coordinates, parallel to `entities`.
    pub entity_bounds: Vec<BoundingBox>,
    /// Union of all entity_bounds (cell-local). None if cell is empty.
    pub bbox: Option<BoundingBox>,
}

/// A single-cell reference (GDS SREF). Stores the placement transform.
/// `bbox` is pre-computed in the **parent cell's local coordinates**.
#[derive(Debug, Clone)]
pub struct GdsInstanceEntity {
    /// Index into `Drawing::gds_cells`.
    pub cell_idx: usize,
    pub offset: [f64; 2],
    pub angle_deg: f64,
    pub mag: f64,
    pub reflected: bool,
    pub layer: String,
    pub color: [f32; 4],
    /// Pre-computed bbox of the referenced cell after applying this instance's transform,
    /// expressed in the parent cell's local coordinate system.
    pub bbox: BoundingBox,
}

/// An array reference (GDS AREF). Stored compactly as a grid descriptor rather than
/// expanding to `cols × rows` individual instance records.
/// `bbox` covers the entire array in the **parent cell's local coordinates**.
#[derive(Debug, Clone)]
pub struct GdsArrayInstanceEntity {
    pub cell_idx: usize,
    /// Position of the (col=0, row=0) element in parent-cell-local coordinates.
    pub origin: [f64; 2],
    /// Offset from one column to the next (parent-cell-local).
    pub col_step: [f64; 2],
    /// Offset from one row to the next (parent-cell-local).
    pub row_step: [f64; 2],
    pub cols: u32,
    pub rows: u32,
    /// Transform applied to every element (each element rotated/scaled/reflected identically).
    pub angle_deg: f64,
    pub mag: f64,
    pub reflected: bool,
    pub layer: String,
    pub color: [f32; 4],
    /// Pre-computed bbox of the entire array in parent-cell-local coordinates.
    pub bbox: BoundingBox,
}

/// TEXT / MTEXT entity
#[derive(Debug, Clone)]
pub struct TextEntity {
    pub position: [f64; 2],
    pub text: String,
    pub height: f64,
    #[allow(dead_code)]
    pub rotation: f64,
    pub layer: String,
    pub color: [f32; 4],
}

/// A DXF INSERT reference (single or array). Stored compactly — block geometry
/// is tessellated once and GPU-instanced, just like GDS SREF/AREF.
/// `block_idx` indexes into `Drawing::gds_cells` (DXF blocks are stored alongside GDS cells).
#[derive(Debug, Clone)]
pub struct DxfInsertEntity {
    /// Index into `Drawing::gds_cells`.
    pub block_idx: usize,
    /// Insertion point in world space (col=0, row=0 element).
    pub offset: [f64; 2],
    /// Rotation angle in degrees.
    pub angle_deg: f64,
    /// X axis scale factor (may differ from y_scale).
    pub x_scale: f64,
    /// Y axis scale factor (may differ from x_scale).
    pub y_scale: f64,
    pub layer: String,
    pub color: [f32; 4],
    /// Pre-computed world-space AABB of the entire insert (all grid cells).
    pub bbox: BoundingBox,
    /// Number of columns in array insert (≥1; normal INSERT has 1).
    pub cols: u32,
    /// Number of rows in array insert (≥1; normal INSERT has 1).
    pub rows: u32,
    /// World-space translation per column step.
    pub col_step: [f64; 2],
    /// World-space translation per row step.
    pub row_step: [f64; 2],
}

/// All supported CAD entity types.
#[derive(Debug, Clone)]
pub enum CadEntity {
    Line(LineEntity),
    Circle(CircleEntity),
    Arc(ArcEntity),
    Polyline(PolylineEntity),
    Ellipse(EllipseEntity),
    Spline(SplineEntity),
    Text(TextEntity),
    Polygon(PolygonEntity),
    /// GDS SREF: single placed cell instance.
    GdsInstance(Box<GdsInstanceEntity>),
    /// GDS AREF: compact grid array — not expanded at load time.
    GdsArrayInstance(Box<GdsArrayInstanceEntity>),
    /// DXF INSERT: single or array block reference — GPU instanced like GDS.
    DxfInsert(Box<DxfInsertEntity>),
}

impl CadEntity {
    pub fn layer(&self) -> &str {
        match self {
            CadEntity::Line(e) => &e.layer,
            CadEntity::Circle(e) => &e.layer,
            CadEntity::Arc(e) => &e.layer,
            CadEntity::Polyline(e) => &e.layer,
            CadEntity::Ellipse(e) => &e.layer,
            CadEntity::Spline(e) => &e.layer,
            CadEntity::Text(e) => &e.layer,
            CadEntity::Polygon(e) => &e.layer,
            CadEntity::GdsInstance(e) => &e.layer,
            CadEntity::GdsArrayInstance(e) => &e.layer,
            CadEntity::DxfInsert(e) => &e.layer,
        }
    }

    #[allow(dead_code)]
    pub fn color(&self) -> [f32; 4] {
        match self {
            CadEntity::Line(e) => e.color,
            CadEntity::Circle(e) => e.color,
            CadEntity::Arc(e) => e.color,
            CadEntity::Polyline(e) => e.color,
            CadEntity::Ellipse(e) => e.color,
            CadEntity::Spline(e) => e.color,
            CadEntity::Text(e) => e.color,
            CadEntity::Polygon(e) => e.color,
            CadEntity::GdsInstance(e) => e.color,
            CadEntity::GdsArrayInstance(e) => e.color,
            CadEntity::DxfInsert(e) => e.color,
        }
    }

    /// Expand a bounding box to include this entity.
    pub fn expand_bounds(&self, bb: &mut BoundingBox) {
        match self {
            CadEntity::Line(e) => {
                bb.expand(e.start[0], e.start[1]);
                bb.expand(e.end[0], e.end[1]);
            }
            CadEntity::Circle(e) => {
                bb.expand(e.center[0] - e.radius, e.center[1] - e.radius);
                bb.expand(e.center[0] + e.radius, e.center[1] + e.radius);
            }
            CadEntity::Arc(e) => {
                bb.expand(e.center[0] - e.radius, e.center[1] - e.radius);
                bb.expand(e.center[0] + e.radius, e.center[1] + e.radius);
            }
            CadEntity::Polyline(e) => {
                for p in &e.points {
                    bb.expand(p[0], p[1]);
                }
            }
            CadEntity::Ellipse(e) => {
                let r = (e.major_axis[0].powi(2) + e.major_axis[1].powi(2)).sqrt();
                bb.expand(e.center[0] - r, e.center[1] - r);
                bb.expand(e.center[0] + r, e.center[1] + r);
            }
            CadEntity::Spline(e) => {
                for p in &e.control_points {
                    bb.expand(p[0], p[1]);
                }
            }
            CadEntity::Text(e) => {
                bb.expand(e.position[0], e.position[1]);
            }
            CadEntity::Polygon(e) => {
                for p in &e.points {
                    bb.expand(p[0], p[1]);
                }
            }
            CadEntity::GdsInstance(e) => {
                if e.bbox.is_valid() {
                    bb.expand(e.bbox.min[0], e.bbox.min[1]);
                    bb.expand(e.bbox.max[0], e.bbox.max[1]);
                }
            }
            CadEntity::GdsArrayInstance(e) => {
                if e.bbox.is_valid() {
                    bb.expand(e.bbox.min[0], e.bbox.min[1]);
                    bb.expand(e.bbox.max[0], e.bbox.max[1]);
                }
            }
            CadEntity::DxfInsert(e) => {
                if e.bbox.is_valid() {
                    bb.expand(e.bbox.min[0], e.bbox.min[1]);
                    bb.expand(e.bbox.max[0], e.bbox.max[1]);
                }
            }
        }
    }
}

impl GdsCellDef {
    /// Append an entity and compute its local-coordinate bounding box.
    /// For GdsInstance/GdsArrayInstance the bbox starts empty and is filled later
    /// by the bottom-up bbox pass in the loader.
    pub fn push_entity(&mut self, entity: CadEntity) {
        let mut bb = BoundingBox::empty();
        entity.expand_bounds(&mut bb);
        self.entity_bounds.push(bb);
        self.entities.push(entity);
    }
}

/// A fully loaded CAD drawing.
#[derive(Debug, Default)]
pub struct Drawing {
    pub layers: Vec<Layer>,
    pub entities: Vec<CadEntity>,
    pub bounds: Option<BoundingBox>,
    /// Per-entity bounding boxes (parallel to `entities`), built by `compute_bounds()`.
    pub entity_bounds: Vec<BoundingBox>,
    pub spatial_index: SpatialIndex,
    pub lod_cache: LodCache,
    /// GDS cell library (populated for GDS files; empty for DXF/DWG).
    /// Indexed by `cell_idx` in GdsInstanceEntity / GdsArrayInstanceEntity.
    /// Wrapped in Arc so background tessellation threads can share it cheaply.
    pub gds_cells: std::sync::Arc<Vec<GdsCellDef>>,
}

impl Drawing {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn compute_bounds(&mut self) {
        let mut bb = BoundingBox::empty();
        self.entity_bounds = self
            .entities
            .iter()
            .map(|e| {
                let mut ebb = BoundingBox::empty();
                e.expand_bounds(&mut ebb);
                if ebb.is_valid() {
                    bb.expand(ebb.min[0], ebb.min[1]);
                    bb.expand(ebb.max[0], ebb.max[1]);
                }
                ebb
            })
            .collect();
        self.bounds = if bb.is_valid() { Some(bb) } else { None };
        self.rebuild_render_indexes();
    }

    pub fn append_chunk_deferred(&mut self, chunk: DrawingChunk) {
        let mut layers_changed = false;
        for layer in chunk.layers {
            if self
                .layers
                .iter()
                .all(|existing| existing.name != layer.name)
            {
                self.layers.push(layer);
                layers_changed = true;
            }
        }
        if layers_changed {
            self.layers.sort_by(|a, b| a.name.cmp(&b.name));
        }

        if let Some(chunk_bounds) = chunk.bounds {
            self.expand_bounds(&chunk_bounds);
        }
        self.entities.extend(chunk.entities);
        self.entity_bounds.extend(chunk.entity_bounds);
    }

    pub fn finalize_indexes(&mut self) {
        if self.bounds.is_none() || self.entity_bounds.len() != self.entities.len() {
            self.compute_bounds();
        } else {
            self.rebuild_render_indexes();
        }
    }

    pub fn query_render_items<'a>(
        &'a self,
        viewport: &BoundingBox,
        pixels_per_unit: f64,
        hidden_layers: &HashSet<String>,
    ) -> Vec<RenderItem<'a>> {
        let Some(bounds) = &self.bounds else {
            return Vec::new();
        };
        if !bounds.intersects(viewport) {
            return Vec::new();
        }

        let candidate_indices = self.spatial_index.query(viewport, self.entities.len());
        let pixel_size = if pixels_per_unit > 0.0 {
            1.0 / pixels_per_unit
        } else {
            0.0
        };
        let mut exact = Vec::with_capacity(candidate_indices.len().min(EXACT_ENTITY_LIMIT));

        for idx in candidate_indices {
            let Some(entity) = self.entities.get(idx) else {
                continue;
            };
            if hidden_layers.contains(entity.layer()) {
                continue;
            }
            let Some(eb) = self.entity_bounds.get(idx) else {
                continue;
            };
            if !eb.intersects(viewport) {
                continue;
            }
            if pixel_size > 0.0 && eb.width() < pixel_size && eb.height() < pixel_size {
                continue;
            }
            exact.push(RenderItem::Entity { entity });
            if exact.len() > EXACT_ENTITY_LIMIT {
                break;
            }
        }

        let tile_screen = self
            .lod_cache
            .tile_world_size()
            .map(|s| s * pixels_per_unit)
            .unwrap_or(f64::MAX);
        let use_lod = exact.len() > EXACT_ENTITY_LIMIT
            || (self.entities.len() > LOD_TRIGGER_ENTITY_COUNT
                && tile_screen < LOD_MIN_TILE_PIXELS);

        if !use_lod {
            return exact;
        }

        let lod = self.lod_cache.query(viewport, hidden_layers);
        if lod.is_empty() {
            return exact;
        }
        lod
    }

    fn rebuild_render_indexes(&mut self) {
        self.spatial_index = SpatialIndex::build(self.bounds.as_ref(), &self.entity_bounds);
        self.lod_cache = LodCache::build(self.bounds.as_ref(), &self.entities, &self.entity_bounds);
    }

    fn expand_bounds(&mut self, other: &BoundingBox) {
        if !other.is_valid() {
            return;
        }
        match &mut self.bounds {
            Some(bounds) => {
                bounds.expand(other.min[0], other.min[1]);
                bounds.expand(other.max[0], other.max[1]);
            }
            None => self.bounds = Some(other.clone()),
        }
    }

    #[allow(dead_code)]
    pub fn layer_by_name(&self, name: &str) -> Option<&Layer> {
        self.layers.iter().find(|l| l.name == name)
    }
}

impl DrawingChunk {
    pub fn from_parts(
        layers: Vec<Layer>,
        entities: Vec<CadEntity>,
        entity_bounds: Vec<BoundingBox>,
        bounds: Option<BoundingBox>,
    ) -> Self {
        Self {
            layers,
            entities,
            entity_bounds,
            bounds,
        }
    }
}

impl SpatialIndex {
    pub fn build(bounds: Option<&BoundingBox>, entity_bounds: &[BoundingBox]) -> Self {
        let Some(bounds) = bounds.filter(|b| b.is_valid() && b.width() > 0.0 && b.height() > 0.0)
        else {
            return Self::default();
        };
        let entity_count = entity_bounds.len().max(1);
        let dim = ((entity_count / SPATIAL_TARGET_CELL_ENTITIES).max(1) as f64)
            .sqrt()
            .ceil() as usize;
        let cols = dim.clamp(1, MAX_SPATIAL_DIM);
        let rows = dim.clamp(1, MAX_SPATIAL_DIM);
        let mut index = Self {
            bounds: Some(bounds.clone()),
            cols,
            rows,
            cells: vec![Vec::new(); cols * rows],
        };

        for (idx, eb) in entity_bounds.iter().enumerate() {
            if !eb.is_valid() {
                continue;
            }
            let (min_c, max_c, min_r, max_r) = index.cell_range(eb);
            for row in min_r..=max_r {
                for col in min_c..=max_c {
                    index.cells[row * cols + col].push(idx);
                }
            }
        }
        index
    }

    pub fn query(&self, viewport: &BoundingBox, entity_count: usize) -> Vec<usize> {
        let Some(_) = &self.bounds else {
            return (0..entity_count).collect();
        };
        if self.cells.is_empty() {
            return (0..entity_count).collect();
        }
        let (min_c, max_c, min_r, max_r) = self.cell_range(viewport);
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for row in min_r..=max_r {
            for col in min_c..=max_c {
                for &idx in &self.cells[row * self.cols + col] {
                    if seen.insert(idx) {
                        out.push(idx);
                    }
                }
            }
        }
        out
    }

    fn cell_range(&self, bb: &BoundingBox) -> (usize, usize, usize, usize) {
        let bounds = self.bounds.as_ref().expect("spatial bounds");
        let x0 = normalized_axis(bb.min[0], bounds.min[0], bounds.width());
        let x1 = normalized_axis(bb.max[0], bounds.min[0], bounds.width());
        let y0 = normalized_axis(bb.min[1], bounds.min[1], bounds.height());
        let y1 = normalized_axis(bb.max[1], bounds.min[1], bounds.height());
        let min_c = ((x0.min(x1) * self.cols as f64).floor() as isize)
            .clamp(0, self.cols as isize - 1) as usize;
        let max_c = ((x0.max(x1) * self.cols as f64).floor() as isize)
            .clamp(0, self.cols as isize - 1) as usize;
        let min_r = ((y0.min(y1) * self.rows as f64).floor() as isize)
            .clamp(0, self.rows as isize - 1) as usize;
        let max_r = ((y0.max(y1) * self.rows as f64).floor() as isize)
            .clamp(0, self.rows as isize - 1) as usize;
        (min_c, max_c, min_r, max_r)
    }
}

impl LodCache {
    pub fn build(
        bounds: Option<&BoundingBox>,
        entities: &[CadEntity],
        entity_bounds: &[BoundingBox],
    ) -> Self {
        let Some(bounds) = bounds.filter(|b| b.is_valid() && b.width() > 0.0 && b.height() > 0.0)
        else {
            return Self::default();
        };
        let cols = LOD_GRID_DIM;
        let rows = LOD_GRID_DIM;
        let mut grouped: HashMap<(usize, String), LodTile> = HashMap::new();

        for (idx, entity) in entities.iter().enumerate() {
            let Some(eb) = entity_bounds.get(idx) else {
                continue;
            };
            if !eb.is_valid() {
                continue;
            }
            let center = eb.center();
            let col = ((normalized_axis(center[0], bounds.min[0], bounds.width()) * cols as f64)
                .floor() as isize)
                .clamp(0, cols as isize - 1) as usize;
            let row = ((normalized_axis(center[1], bounds.min[1], bounds.height()) * rows as f64)
                .floor() as isize)
                .clamp(0, rows as isize - 1) as usize;
            let cell = row * cols + col;
            let layer = entity.layer().to_string();
            let entry = grouped
                .entry((cell, layer.clone()))
                .or_insert_with(|| LodTile {
                    bounds: BoundingBox::empty(),
                    layer,
                    color: entity.color(),
                    entity_count: 0,
                });
            entry.bounds.expand(eb.min[0], eb.min[1]);
            entry.bounds.expand(eb.max[0], eb.max[1]);
            entry.entity_count += 1;
        }

        Self {
            bounds: Some(bounds.clone()),
            cols,
            rows,
            tiles: grouped
                .into_values()
                .filter(|tile| tile.bounds.is_valid())
                .collect(),
        }
    }

    pub fn query<'a>(
        &'a self,
        viewport: &BoundingBox,
        hidden_layers: &HashSet<String>,
    ) -> Vec<RenderItem<'a>> {
        self.tiles
            .iter()
            .filter(|tile| tile.bounds.intersects(viewport) && !hidden_layers.contains(&tile.layer))
            .map(|tile| RenderItem::LodBox {
                bounds: &tile.bounds,
                color: tile.color,
            })
            .collect()
    }

    fn tile_world_size(&self) -> Option<f64> {
        let bounds = self.bounds.as_ref()?;
        Some((bounds.width() / self.cols as f64).max(bounds.height() / self.rows as f64))
    }
}

fn normalized_axis(value: f64, min: f64, len: f64) -> f64 {
    if len <= 0.0 {
        0.0
    } else {
        ((value - min) / len).clamp(0.0, 0.999_999)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(layer: &str, x: f64) -> CadEntity {
        CadEntity::Line(LineEntity {
            start: [x, 0.0],
            end: [x + 0.5, 0.5],
            layer: layer.to_string(),
            color: [1.0, 1.0, 1.0, 1.0],
        })
    }

    #[test]
    fn query_uses_exact_entities_for_small_visible_sets() {
        let mut drawing = Drawing::new();
        drawing.entities = vec![line("A", 0.0), line("A", 2.0), line("B", 4.0)];
        drawing.compute_bounds();

        let items = drawing.query_render_items(
            &BoundingBox {
                min: [-1.0, -1.0],
                max: [10.0, 10.0],
            },
            10.0,
            &HashSet::new(),
        );

        assert_eq!(items.len(), 3);
        assert!(items
            .iter()
            .all(|item| matches!(item, RenderItem::Entity { .. })));
    }

    #[test]
    fn query_falls_back_to_lod_boxes_when_zoomed_out() {
        let mut drawing = Drawing::new();
        drawing.entities = (0..20_001).map(|i| line("A", i as f64)).collect();
        drawing.compute_bounds();

        let items = drawing.query_render_items(
            &BoundingBox {
                min: [-1.0, -1.0],
                max: [20_010.0, 10.0],
            },
            0.001,
            &HashSet::new(),
        );

        assert!(!items.is_empty());
        assert!(items
            .iter()
            .all(|item| matches!(item, RenderItem::LodBox { .. })));
    }

    #[test]
    fn query_applies_hidden_layers_to_lod_results() {
        let mut drawing = Drawing::new();
        drawing.entities = (0..20_001).map(|i| line("A", i as f64)).collect();
        drawing.compute_bounds();

        let mut hidden = HashSet::new();
        hidden.insert("A".to_string());
        let items = drawing.query_render_items(
            &BoundingBox {
                min: [-1.0, -1.0],
                max: [20_010.0, 10.0],
            },
            0.001,
            &hidden,
        );

        assert!(items.is_empty());
    }
}
