use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::mpsc::Sender;

use gds21::{GdsElement, GdsLibrary};

use crate::cad::model::{
    BoundingBox, CadEntity, Drawing, DrawingChunk, GdsArrayInstanceEntity, GdsCellDef,
    GdsInstanceEntity, Layer, LoadEvent, PolygonEntity, PolylineEntity, TextEntity,
};
use crate::util::color::aci_to_rgba;

const LOAD_CHUNK_SIZE: usize = 10_000;

// ── Public API ────────────────────────────────────────────────────────────────

/// Load a GDSII file non-streaming (for tests / non-UI use).
#[allow(dead_code)]
pub fn load_gds(path: &Path) -> Result<Drawing, String> {
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();
    load_gds_streaming(path, &tx);
    drop(tx);

    let mut drawing = Drawing::new();
    for event in rx {
        match event {
            LoadEvent::GdsCells(cells) => drawing.gds_cells = std::sync::Arc::new(cells),
            LoadEvent::Chunk(chunk) => drawing.append_chunk_deferred(chunk),
            LoadEvent::Finished => {
                drawing.finalize_indexes();
                return Ok(drawing);
            }
            LoadEvent::Failed(e) => return Err(e),
            _ => {}
        }
    }
    Err("GDS loader: channel closed without Finished event".to_string())
}

pub fn load_gds_streaming(path: &Path, tx: &Sender<LoadEvent>) {
    let _ = tx.send(LoadEvent::Started { format: "GDS" });
    match load_hierarchical(path, tx) {
        Ok(()) => {
            let _ = tx.send(LoadEvent::Finished);
        }
        Err(e) => {
            let _ = tx.send(LoadEvent::Failed(e));
        }
    }
}

// ── Top-level loader ──────────────────────────────────────────────────────────

fn load_hierarchical(path: &Path, tx: &Sender<LoadEvent>) -> Result<(), String> {
    let lib = GdsLibrary::open(path).map_err(|e| format!("GDS parse error: {e}"))?;

    // User-unit scale: multiply integer GDS coords by this to get user units.
    let scale = if lib.units.0 > 0.0 {
        lib.units.1 / lib.units.0
    } else {
        1.0
    };

    // struct name → struct index (stable across the library)
    let struct_map: HashMap<String, usize> = lib
        .structs
        .iter()
        .enumerate()
        .map(|(i, s)| (s.name.clone(), i))
        .collect();

    // ── Phase 1: build per-cell geometry (cell-local coords, no instance transforms) ──
    let mut layer_cache: HashMap<(i16, i16), (String, [f32; 4])> = HashMap::new();
    let mut layer_set: BTreeMap<String, [f32; 4]> = BTreeMap::new();
    let mut cell_defs =
        build_cell_defs(&lib.structs, &struct_map, scale, &mut layer_cache, &mut layer_set);

    // ── Phase 2: topological sort (leaves first) and bottom-up bbox pass ──────
    let topo_order = topo_sort(lib.structs.len(), &lib.structs, &struct_map);
    compute_bboxes_bottom_up(&mut cell_defs, &topo_order);

    // Back-fill representative colors for instance entities now that cell bboxes are known.
    assign_instance_colors(&mut cell_defs);

    // ── Phase 3: send cell library ────────────────────────────────────────────
    tx.send(LoadEvent::GdsCells(cell_defs.clone()))
        .map_err(|_| "GDS load cancelled".to_string())?;

    // ── Phase 4: identify top-level cells and stream their entities as chunks ──
    let mut referenced: HashSet<String> = HashSet::new();
    for s in &lib.structs {
        for elem in &s.elems {
            match elem {
                GdsElement::GdsStructRef(r) => {
                    referenced.insert(r.name.clone());
                }
                GdsElement::GdsArrayRef(r) => {
                    referenced.insert(r.name.clone());
                }
                _ => {}
            }
        }
    }
    let top_indices: Vec<usize> = lib
        .structs
        .iter()
        .enumerate()
        .filter(|(_, s)| !referenced.contains(&s.name))
        .filter_map(|(_, s)| struct_map.get(&s.name).copied())
        .collect();
    // Fall back to all structs if nothing is unreferenced (circular or single-cell file).
    let top_indices: Vec<usize> = if top_indices.is_empty() {
        (0..cell_defs.len()).collect()
    } else {
        top_indices
    };

    let layers: Vec<Layer> = layer_set
        .iter()
        .map(|(name, &color)| Layer {
            name: name.clone(),
            color,
            visible: true,
        })
        .collect();

    // Stream top-level entities as chunks.
    // The top-level cells' local space IS world space (no parent transform → identity).
    let mut chunk_ents: Vec<CadEntity> = Vec::with_capacity(LOAD_CHUNK_SIZE);
    let mut chunk_bbs: Vec<BoundingBox> = Vec::with_capacity(LOAD_CHUNK_SIZE);
    let mut chunk_overall = BoundingBox::empty();
    let mut loaded = 0usize;

    for cell_idx in top_indices {
        let cell = &cell_defs[cell_idx];
        for (entity, eb) in cell.entities.iter().zip(cell.entity_bounds.iter()) {
            if eb.is_valid() {
                chunk_overall.expand(eb.min[0], eb.min[1]);
                chunk_overall.expand(eb.max[0], eb.max[1]);
            }
            chunk_bbs.push(eb.clone());
            chunk_ents.push(entity.clone());
            loaded += 1;

            if chunk_ents.len() >= LOAD_CHUNK_SIZE {
                flush_chunk(
                    &mut chunk_ents,
                    &mut chunk_bbs,
                    &mut chunk_overall,
                    loaded,
                    &layers,
                    tx,
                )?;
            }
        }
    }
    flush_chunk(
        &mut chunk_ents,
        &mut chunk_bbs,
        &mut chunk_overall,
        loaded,
        &layers,
        tx,
    )
}

fn flush_chunk(
    ents: &mut Vec<CadEntity>,
    bbs: &mut Vec<BoundingBox>,
    overall: &mut BoundingBox,
    loaded: usize,
    layers: &[Layer],
    tx: &Sender<LoadEvent>,
) -> Result<(), String> {
    if ents.is_empty() {
        return Ok(());
    }
    let entities = std::mem::take(ents);
    let entity_bounds = std::mem::take(bbs);
    let bounds = if overall.is_valid() {
        Some(std::mem::replace(overall, BoundingBox::empty()))
    } else {
        *overall = BoundingBox::empty();
        None
    };
    tx.send(LoadEvent::Chunk(DrawingChunk::from_parts(
        layers.to_vec(),
        entities,
        entity_bounds,
        bounds,
    )))
    .and_then(|_| tx.send(LoadEvent::Progress { loaded_entities: loaded }))
    .map_err(|_| "GDS load cancelled".to_string())
}

// ── Phase 1: build_cell_defs ──────────────────────────────────────────────────

fn build_cell_defs(
    structs: &[gds21::GdsStruct],
    struct_map: &HashMap<String, usize>,
    scale: f64,
    layer_cache: &mut HashMap<(i16, i16), (String, [f32; 4])>,
    layer_set: &mut BTreeMap<String, [f32; 4]>,
) -> Vec<GdsCellDef> {
    let palette: &[[f32; 4]] = &[
        [0.2, 0.6, 1.0, 0.8],
        [1.0, 0.4, 0.2, 0.8],
        [0.2, 1.0, 0.4, 0.8],
        [1.0, 0.9, 0.1, 0.8],
        [0.8, 0.2, 1.0, 0.8],
        [0.1, 0.9, 0.9, 0.8],
        [1.0, 0.5, 0.8, 0.8],
        [0.5, 0.8, 0.2, 0.8],
        [1.0, 0.7, 0.2, 0.8],
        [0.4, 0.4, 1.0, 0.8],
        [0.6, 1.0, 0.6, 0.8],
        [1.0, 0.3, 0.5, 0.8],
        [0.3, 0.8, 0.8, 0.8],
        [0.9, 0.6, 0.1, 0.8],
        [0.7, 0.3, 0.9, 0.8],
        [0.2, 0.7, 0.5, 0.8],
    ];
    let layer_color = |layer_num: i16| -> [f32; 4] {
        if (1..=255).contains(&layer_num) {
            let c = aci_to_rgba(layer_num);
            [c[0], c[1], c[2], 0.75]
        } else {
            palette[(layer_num.unsigned_abs() as usize) % palette.len()]
        }
    };

    let mut cell_defs: Vec<GdsCellDef> = structs
        .iter()
        .map(|s| GdsCellDef {
            name: s.name.clone(),
            ..Default::default()
        })
        .collect();

    for (struct_idx, gds_struct) in structs.iter().enumerate() {
        for elem in &gds_struct.elems {
            match elem {
                GdsElement::GdsBoundary(b) => {
                    let (layer_name, color) =
                        get_layer_info(b.layer, b.datatype, layer_cache, layer_set, &layer_color);
                    let pts: Vec<[f64; 2]> = b
                        .xy
                        .iter()
                        .map(|p| [p.x as f64 * scale, p.y as f64 * scale])
                        .collect();
                    if pts.len() >= 3 {
                        cell_defs[struct_idx].push_entity(CadEntity::Polygon(PolygonEntity {
                            points: pts,
                            layer: layer_name,
                            color,
                        }));
                    }
                }
                GdsElement::GdsPath(p) => {
                    let (layer_name, color) =
                        get_layer_info(p.layer, p.datatype, layer_cache, layer_set, &layer_color);
                    let pts: Vec<[f64; 2]> = p
                        .xy
                        .iter()
                        .map(|pt| [pt.x as f64 * scale, pt.y as f64 * scale])
                        .collect();
                    if pts.len() >= 2 {
                        cell_defs[struct_idx].push_entity(CadEntity::Polyline(PolylineEntity {
                            points: pts,
                            bulges: vec![],
                            closed: false,
                            layer: layer_name,
                            color,
                        }));
                    }
                }
                GdsElement::GdsTextElem(t) => {
                    let (layer_name, color) = get_layer_info(
                        t.layer,
                        t.texttype,
                        layer_cache,
                        layer_set,
                        &layer_color,
                    );
                    let pos = [t.xy.x as f64 * scale, t.xy.y as f64 * scale];
                    let text_angle = t.strans.as_ref().and_then(|s| s.angle).unwrap_or(0.0);
                    let text_mag = t.strans.as_ref().and_then(|s| s.mag).unwrap_or(1.0);
                    cell_defs[struct_idx].push_entity(CadEntity::Text(TextEntity {
                        position: pos,
                        text: t.string.clone(),
                        height: text_mag * scale,
                        rotation: text_angle,
                        layer: layer_name,
                        color,
                    }));
                }
                GdsElement::GdsStructRef(r) => {
                    if let Some(&child_idx) = struct_map.get(&r.name) {
                        let offset = [r.xy.x as f64 * scale, r.xy.y as f64 * scale];
                        let angle_deg =
                            r.strans.as_ref().and_then(|s| s.angle).unwrap_or(0.0);
                        let mag = r.strans.as_ref().and_then(|s| s.mag).unwrap_or(1.0);
                        let reflected =
                            r.strans.as_ref().map(|s| s.reflected).unwrap_or(false);
                        // layer/color will be back-filled after bbox pass
                        cell_defs[struct_idx].push_entity(CadEntity::GdsInstance(Box::new(
                            GdsInstanceEntity {
                                cell_idx: child_idx,
                                offset,
                                angle_deg,
                                mag,
                                reflected,
                                layer: r.name.clone(),
                                color: [0.7, 0.7, 0.7, 0.8],
                                bbox: BoundingBox::empty(), // filled in phase 2
                            },
                        )));
                    }
                }
                GdsElement::GdsArrayRef(r) => {
                    if let Some(&child_idx) = struct_map.get(&r.name) {
                        if r.xy.len() >= 3 {
                            let (cols, rows) = (r.cols as u32, r.rows as u32);
                            let orig = [r.xy[0].x as f64 * scale, r.xy[0].y as f64 * scale];
                            let c_end = [r.xy[1].x as f64 * scale, r.xy[1].y as f64 * scale];
                            let r_end = [r.xy[2].x as f64 * scale, r.xy[2].y as f64 * scale];
                            let col_step = if cols > 1 {
                                [
                                    (c_end[0] - orig[0]) / cols as f64,
                                    (c_end[1] - orig[1]) / cols as f64,
                                ]
                            } else {
                                [0.0, 0.0]
                            };
                            let row_step = if rows > 1 {
                                [
                                    (r_end[0] - orig[0]) / rows as f64,
                                    (r_end[1] - orig[1]) / rows as f64,
                                ]
                            } else {
                                [0.0, 0.0]
                            };
                            let angle_deg =
                                r.strans.as_ref().and_then(|s| s.angle).unwrap_or(0.0);
                            let mag = r.strans.as_ref().and_then(|s| s.mag).unwrap_or(1.0);
                            let reflected =
                                r.strans.as_ref().map(|s| s.reflected).unwrap_or(false);
                            cell_defs[struct_idx].push_entity(CadEntity::GdsArrayInstance(
                                Box::new(GdsArrayInstanceEntity {
                                    cell_idx: child_idx,
                                    origin: orig,
                                    col_step,
                                    row_step,
                                    cols,
                                    rows,
                                    angle_deg,
                                    mag,
                                    reflected,
                                    layer: r.name.clone(),
                                    color: [0.7, 0.7, 0.7, 0.8],
                                    bbox: BoundingBox::empty(), // filled in phase 2
                                }),
                            ));
                        }
                    }
                }
                GdsElement::GdsBox(b) => {
                    let (layer_name, color) =
                        get_layer_info(b.layer, b.boxtype, layer_cache, layer_set, &layer_color);
                    let pts: Vec<[f64; 2]> = b
                        .xy
                        .iter()
                        .map(|p| [p.x as f64 * scale, p.y as f64 * scale])
                        .collect();
                    if pts.len() >= 3 {
                        cell_defs[struct_idx].push_entity(CadEntity::Polygon(PolygonEntity {
                            points: pts,
                            layer: layer_name,
                            color,
                        }));
                    }
                }
                GdsElement::GdsNode(_) => {}
            }
        }
    }
    cell_defs
}

fn get_layer_info(
    layer: i16,
    datatype: i16,
    cache: &mut HashMap<(i16, i16), (String, [f32; 4])>,
    layer_set: &mut BTreeMap<String, [f32; 4]>,
    layer_color: &impl Fn(i16) -> [f32; 4],
) -> (String, [f32; 4]) {
    cache
        .entry((layer, datatype))
        .or_insert_with(|| {
            let name = format!("L{}D{}", layer, datatype);
            let color = layer_color(layer);
            layer_set.entry(name.clone()).or_insert(color);
            (name, color)
        })
        .clone()
}

// ── Phase 2: topological sort + bottom-up bbox computation ────────────────────

/// Kahn's BFS topological sort — returns cell indices with leaves first.
fn topo_sort(
    n: usize,
    structs: &[gds21::GdsStruct],
    struct_map: &HashMap<String, usize>,
) -> Vec<usize> {
    // Build reverse edges: referenced_by[child] = parents that reference child.
    let mut referenced_by: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut dep_count: Vec<usize> = vec![0; n];

    for s in structs {
        if let Some(&parent_idx) = struct_map.get(&s.name) {
            for elem in &s.elems {
                let child_name = match elem {
                    GdsElement::GdsStructRef(r) => Some(&r.name),
                    GdsElement::GdsArrayRef(r) => Some(&r.name),
                    _ => None,
                };
                if let Some(name) = child_name {
                    if let Some(&child_idx) = struct_map.get(name.as_str()) {
                        if child_idx != parent_idx {
                            referenced_by[child_idx].push(parent_idx);
                            dep_count[parent_idx] += 1;
                        }
                    }
                }
            }
        }
    }

    let mut queue: VecDeque<usize> = (0..n).filter(|&i| dep_count[i] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(idx) = queue.pop_front() {
        order.push(idx);
        for &parent in &referenced_by[idx] {
            dep_count[parent] = dep_count[parent].saturating_sub(1);
            if dep_count[parent] == 0 {
                queue.push_back(parent);
            }
        }
    }
    // Append any remaining cells in cycles (malformed GDS).
    for i in 0..n {
        if dep_count[i] > 0 {
            order.push(i);
        }
    }
    order
}

/// Fill in `GdsInstance.bbox`, `GdsArrayInstance.bbox`, and `GdsCellDef.bbox`
/// using a bottom-up traversal (leaves before parents).
fn compute_bboxes_bottom_up(cell_defs: &mut Vec<GdsCellDef>, order: &[usize]) {
    for &cell_idx in order {
        let n = cell_defs[cell_idx].entities.len();

        // First: back-fill instance bboxes using the already-computed child cell bboxes.
        for ent_idx in 0..n {
            let new_bb: Option<BoundingBox> = match &cell_defs[cell_idx].entities[ent_idx] {
                CadEntity::GdsInstance(inst) => {
                    cell_defs[inst.cell_idx]
                        .bbox
                        .as_ref()
                        .map(|cb| transform_bbox(cb, inst.offset, inst.angle_deg, inst.mag, inst.reflected))
                }
                CadEntity::GdsArrayInstance(arr) => {
                    cell_defs[arr.cell_idx].bbox.as_ref().map(|cb| {
                        // The array's bbox is the union of the 4 corner elements' bboxes.
                        // For a rectangular grid this is exact; for tilted grids it's a
                        // tight-enough outer bound computed in O(1) rather than O(rows×cols).
                        let mut result = BoundingBox::empty();
                        for &c in &[0u32, arr.cols.saturating_sub(1)] {
                            for &r in &[0u32, arr.rows.saturating_sub(1)] {
                                let origin = [
                                    arr.origin[0]
                                        + c as f64 * arr.col_step[0]
                                        + r as f64 * arr.row_step[0],
                                    arr.origin[1]
                                        + c as f64 * arr.col_step[1]
                                        + r as f64 * arr.row_step[1],
                                ];
                                let ib =
                                    transform_bbox(cb, origin, arr.angle_deg, arr.mag, arr.reflected);
                                if ib.is_valid() {
                                    result.expand(ib.min[0], ib.min[1]);
                                    result.expand(ib.max[0], ib.max[1]);
                                }
                            }
                        }
                        result
                    })
                }
                _ => None,
            };

            if let Some(bb) = new_bb {
                // Write back into the entity
                match &mut cell_defs[cell_idx].entities[ent_idx] {
                    CadEntity::GdsInstance(inst) => inst.bbox = bb.clone(),
                    CadEntity::GdsArrayInstance(arr) => arr.bbox = bb.clone(),
                    _ => {}
                }
                cell_defs[cell_idx].entity_bounds[ent_idx] = bb;
            }
        }

        // Second: recompute cell bbox as the union of all entity_bounds.
        let mut cell_bb = BoundingBox::empty();
        for eb in &cell_defs[cell_idx].entity_bounds {
            if eb.is_valid() {
                cell_bb.expand(eb.min[0], eb.min[1]);
                cell_bb.expand(eb.max[0], eb.max[1]);
            }
        }
        cell_defs[cell_idx].bbox = if cell_bb.is_valid() { Some(cell_bb) } else { None };
    }
}

/// Transform a local-coord bbox through a GDS instance transform.
/// Returns the new axis-aligned bbox in parent-local coordinates.
/// All 4 corners of the input bbox are transformed to account for rotation.
fn transform_bbox(
    bbox: &BoundingBox,
    offset: [f64; 2],
    angle_deg: f64,
    mag: f64,
    reflected: bool,
) -> BoundingBox {
    if !bbox.is_valid() {
        return BoundingBox::empty();
    }
    let corners = [
        [bbox.min[0], bbox.min[1]],
        [bbox.max[0], bbox.min[1]],
        [bbox.max[0], bbox.max[1]],
        [bbox.min[0], bbox.max[1]],
    ];
    let rad = angle_deg.to_radians();
    let (sin_a, cos_a) = rad.sin_cos();
    let mut result = BoundingBox::empty();
    for c in corners {
        let x = c[0];
        let y = if reflected { -c[1] } else { c[1] };
        result.expand(
            (cos_a * x - sin_a * y) * mag + offset[0],
            (sin_a * x + cos_a * y) * mag + offset[1],
        );
    }
    result
}

/// After bboxes are computed, propagate a representative color from child cells
/// into GdsInstance/GdsArrayInstance entities (used for LOD box rendering).
fn assign_instance_colors(cell_defs: &mut Vec<GdsCellDef>) {
    for cell_idx in 0..cell_defs.len() {
        for ent_idx in 0..cell_defs[cell_idx].entities.len() {
            let child_color: Option<[f32; 4]> = match &cell_defs[cell_idx].entities[ent_idx] {
                CadEntity::GdsInstance(inst) => primary_color_of_cell(cell_defs, inst.cell_idx, 0),
                CadEntity::GdsArrayInstance(arr) => {
                    primary_color_of_cell(cell_defs, arr.cell_idx, 0)
                }
                _ => None,
            };
            if let Some(color) = child_color {
                match &mut cell_defs[cell_idx].entities[ent_idx] {
                    CadEntity::GdsInstance(inst) => inst.color = color,
                    CadEntity::GdsArrayInstance(arr) => arr.color = color,
                    _ => {}
                }
            }
        }
    }
}

fn primary_color_of_cell(
    cell_defs: &[GdsCellDef],
    cell_idx: usize,
    depth: usize,
) -> Option<[f32; 4]> {
    if depth > 4 {
        return None;
    }
    let cell = cell_defs.get(cell_idx)?;
    for entity in &cell.entities {
        match entity {
            CadEntity::Polygon(p) => return Some(p.color),
            CadEntity::Polyline(p) => return Some(p.color),
            CadEntity::GdsInstance(inst) => {
                if let Some(c) = primary_color_of_cell(cell_defs, inst.cell_idx, depth + 1) {
                    return Some(c);
                }
            }
            CadEntity::GdsArrayInstance(arr) => {
                if let Some(c) = primary_color_of_cell(cell_defs, arr.cell_idx, depth + 1) {
                    return Some(c);
                }
            }
            _ => {}
        }
    }
    None
}
