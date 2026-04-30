use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc::Sender;

use dxf::entities::EntityType;
use dxf::Drawing as DxfDrawing;

use crate::cad::model::{
    ArcEntity, BoundingBox, CadEntity, CircleEntity, Drawing, DrawingChunk, DxfInsertEntity,
    EllipseEntity, GdsCellDef, Layer, LineEntity, LoadEvent, PolylineEntity, SplineEntity,
    TextEntity,
};
use crate::util::color::aci_to_rgba;

const LOAD_CHUNK_SIZE: usize = 10_000;

/// Load a DXF file and convert it to the internal Drawing model.
#[allow(dead_code)]
pub fn load_dxf(path: &Path) -> Result<Drawing, String> {
    load_dxf_inner(path, None)
}

pub fn load_dxf_streaming(path: &Path, tx: &Sender<LoadEvent>) {
    let _ = tx.send(LoadEvent::Started { format: "DXF" });
    match load_dxf_inner(path, Some(tx)) {
        Ok(_) => {
            let _ = tx.send(LoadEvent::Finished);
        }
        Err(e) => {
            let _ = tx.send(LoadEvent::Failed(e));
        }
    }
}

fn load_dxf_inner(path: &Path, tx: Option<&Sender<LoadEvent>>) -> Result<Drawing, String> {
    // First attempt: normal load
    let dxf = match DxfDrawing::load_file(path.to_str().unwrap_or("")) {
        Ok(d) => d,
        Err(e) => {
            let msg = e.to_string();
            // If the error is thumbnail/BMP related, strip the section and retry
            if msg.contains("Bmp") || msg.contains("bitmap") || msg.contains("decoder") {
                log::warn!("DXF thumbnail error, retrying without THUMBNAILIMAGE section: {msg}");
                load_dxf_without_thumbnail(path)?
            } else {
                return Err(format!("DXF parse error: {msg}"));
            }
        }
    };

    // --- Build layer map ---
    let mut layer_map: HashMap<String, [f32; 4]> = HashMap::new();
    for layer in dxf.layers() {
        let color = resolve_color(layer.color.index().map(|i| i as i16).unwrap_or(7));
        layer_map.insert(layer.name.clone(), color);
    }

    // --- Build GPU-instanced block cells from DXF block definitions ---
    // Phase 1: assign an index to every block name (including *Model_Space etc.)
    let mut block_name_to_idx: HashMap<String, usize> = HashMap::new();
    for (idx, block) in dxf.blocks().enumerate() {
        block_name_to_idx.insert(block.name.clone(), idx);
    }

    // Phase 2: build GdsCellDef for each block with primitive entities only
    let mut block_cells: Vec<GdsCellDef> = dxf
        .blocks()
        .map(|block| {
            let mut cell = GdsCellDef {
                name: block.name.clone(),
                ..Default::default()
            };
            for entity in &block.entities {
                if !matches!(entity.specific, EntityType::Insert(_)) {
                    if let Some(cad) = convert_entity(entity, &layer_map) {
                        cell.push_entity(cad);
                    }
                }
            }
            // Compute initial bbox from leaf primitives only
            let mut bb = BoundingBox::empty();
            for eb in &cell.entity_bounds {
                if eb.is_valid() {
                    bb.expand(eb.min[0], eb.min[1]);
                    bb.expand(eb.max[0], eb.max[1]);
                }
            }
            cell.bbox = if bb.is_valid() { Some(bb) } else { None };
            cell
        })
        .collect();

    // Phase 3: add INSERT references to block cells using already-computed primitive bboxes.
    // We iterate once; blocks defined before their references get correct bboxes.
    // Forward-referenced blocks get empty bbox (rare in practice) — geometry still renders.
    for (bi, block) in dxf.blocks().enumerate() {
        let mut new_inserts: Vec<CadEntity> = Vec::new();
        for entity in &block.entities {
            if let EntityType::Insert(ref ins) = entity.specific {
                if let Some(&nested_idx) = block_name_to_idx.get(&ins.name) {
                    let child_bbox = block_cells[nested_idx].bbox.clone();
                    let dxf_insert = make_dxf_insert_entity(
                        ins,
                        nested_idx,
                        child_bbox.as_ref(),
                        &entity.common,
                        &layer_map,
                    );
                    new_inserts.push(CadEntity::DxfInsert(Box::new(dxf_insert)));
                }
            }
        }
        for e in new_inserts {
            block_cells[bi].push_entity(e);
        }
        // Recompute bbox including insert contributions
        let mut bb = BoundingBox::empty();
        for eb in &block_cells[bi].entity_bounds {
            if eb.is_valid() {
                bb.expand(eb.min[0], eb.min[1]);
                bb.expand(eb.max[0], eb.max[1]);
            }
        }
        block_cells[bi].bbox = if bb.is_valid() { Some(bb) } else { None };
    }

    // --- Send block cells to the main thread (same event as GDS) ---
    if let Some(tx) = tx {
        let _ = tx.send(LoadEvent::GdsCells(block_cells.clone()));
    }

    // --- Convert model entities ---
    let mut drawing = Drawing::new();

    for (name, color) in &layer_map {
        drawing.layers.push(Layer {
            name: name.clone(),
            color: *color,
            visible: true,
        });
    }
    drawing.layers.sort_by(|a, b| a.name.cmp(&b.name));

    let mut chunk_entities = Vec::with_capacity(LOAD_CHUNK_SIZE);
    let mut chunk_entity_bounds = Vec::with_capacity(LOAD_CHUNK_SIZE);
    let mut chunk_bounds = BoundingBox::empty();
    let mut loaded_entities = 0usize;

    for entity in dxf.entities() {
        let cad = if let EntityType::Insert(ref ins) = entity.specific {
            // INSERT → DxfInsertEntity (GPU instanced, not expanded)
            if let Some(&block_idx) = block_name_to_idx.get(&ins.name) {
                let child_bbox = block_cells[block_idx].bbox.clone();
                let dxf_insert = make_dxf_insert_entity(
                    ins,
                    block_idx,
                    child_bbox.as_ref(),
                    &entity.common,
                    &layer_map,
                );
                Some(CadEntity::DxfInsert(Box::new(dxf_insert)))
            } else {
                None
            }
        } else {
            convert_entity(entity, &layer_map)
        };

        if let Some(cad) = cad {
            push_loaded_entity(
                cad,
                &mut drawing,
                &mut chunk_entities,
                &mut chunk_entity_bounds,
                &mut chunk_bounds,
                tx,
                &mut loaded_entities,
            )?;
        }
    }

    flush_chunk(
        tx,
        &drawing.layers,
        &mut chunk_entities,
        &mut chunk_entity_bounds,
        &mut chunk_bounds,
        loaded_entities,
    )?;

    if tx.is_none() {
        drawing.gds_cells = std::sync::Arc::new(block_cells);
        drawing.compute_bounds();
    }

    log::info!(
        "Loaded {} entities across {} layers from {:?}",
        loaded_entities,
        drawing.layers.len(),
        path.file_name().unwrap_or_default()
    );

    Ok(drawing)
}

/// Build a `DxfInsertEntity` from a dxf INSERT record.
/// `child_bbox` is the referenced block's world-space bbox (may be None for empty blocks).
fn make_dxf_insert_entity(
    ins: &dxf::entities::Insert,
    block_idx: usize,
    child_bbox: Option<&BoundingBox>,
    common: &dxf::entities::EntityCommon,
    layer_map: &HashMap<String, [f32; 4]>,
) -> DxfInsertEntity {
    let layer = common.layer.clone();
    let color = entity_color(&common.color, &layer, layer_map);
    let rot_rad = ins.rotation.to_radians();
    let (sin_a, cos_a) = rot_rad.sin_cos();
    let sx = ins.x_scale_factor;
    let sy = ins.y_scale_factor;
    let tx = ins.location.x;
    let ty = ins.location.y;
    let cols = (ins.column_count as u32).max(1);
    let rows = (ins.row_count as u32).max(1);
    // Column and row steps are world-axis-aligned (DXF MINSERT spacing is in world XY).
    let col_step = [ins.column_spacing, 0.0];
    let row_step = [0.0, ins.row_spacing];

    let bbox = if let Some(cb) = child_bbox {
        compute_insert_bbox(cb, tx, ty, sin_a, cos_a, sx, sy, cols, rows, col_step, row_step)
    } else {
        BoundingBox::empty()
    };

    DxfInsertEntity {
        block_idx,
        offset: [tx, ty],
        angle_deg: ins.rotation,
        x_scale: sx,
        y_scale: sy,
        layer,
        color,
        bbox,
        cols,
        rows,
        col_step,
        row_step,
    }
}

/// Compute the world-space AABB for a grid of INSERT placements.
fn compute_insert_bbox(
    child_bbox: &BoundingBox,
    tx: f64,
    ty: f64,
    sin_a: f64,
    cos_a: f64,
    sx: f64,
    sy: f64,
    cols: u32,
    rows: u32,
    col_step: [f64; 2],
    row_step: [f64; 2],
) -> BoundingBox {
    let corners = [
        [child_bbox.min[0], child_bbox.min[1]],
        [child_bbox.max[0], child_bbox.min[1]],
        [child_bbox.max[0], child_bbox.max[1]],
        [child_bbox.min[0], child_bbox.max[1]],
    ];
    let mut bb = BoundingBox::empty();
    for col in 0..cols {
        for row in 0..rows {
            let ox = tx + col as f64 * col_step[0] + row as f64 * row_step[0];
            let oy = ty + col as f64 * col_step[1] + row as f64 * row_step[1];
            for &[cx, cy] in &corners {
                let lx = cx * sx;
                let ly = cy * sy;
                bb.expand(
                    cos_a * lx - sin_a * ly + ox,
                    sin_a * lx + cos_a * ly + oy,
                );
            }
        }
    }
    bb
}

fn push_loaded_entity(
    entity: CadEntity,
    drawing: &mut Drawing,
    chunk_entities: &mut Vec<CadEntity>,
    chunk_entity_bounds: &mut Vec<BoundingBox>,
    chunk_bounds: &mut BoundingBox,
    tx: Option<&Sender<LoadEvent>>,
    loaded_entities: &mut usize,
) -> Result<(), String> {
    if tx.is_some() {
        let mut bb = BoundingBox::empty();
        entity.expand_bounds(&mut bb);
        if bb.is_valid() {
            chunk_bounds.expand(bb.min[0], bb.min[1]);
            chunk_bounds.expand(bb.max[0], bb.max[1]);
        }
        chunk_entity_bounds.push(bb);
        chunk_entities.push(entity);
    } else {
        drawing.entities.push(entity);
    }
    *loaded_entities += 1;
    if chunk_entities.len() >= LOAD_CHUNK_SIZE {
        flush_chunk(
            tx,
            &drawing.layers,
            chunk_entities,
            chunk_entity_bounds,
            chunk_bounds,
            *loaded_entities,
        )?;
    }
    Ok(())
}

fn flush_chunk(
    tx: Option<&Sender<LoadEvent>>,
    layers: &[Layer],
    chunk_entities: &mut Vec<CadEntity>,
    chunk_entity_bounds: &mut Vec<BoundingBox>,
    chunk_bounds: &mut BoundingBox,
    loaded_entities: usize,
) -> Result<(), String> {
    let Some(tx) = tx else { return Ok(()) };
    if chunk_entities.is_empty() {
        return Ok(());
    }
    let entities = std::mem::take(chunk_entities);
    let entity_bounds = std::mem::take(chunk_entity_bounds);
    let bounds = if chunk_bounds.is_valid() {
        Some(std::mem::replace(chunk_bounds, BoundingBox::empty()))
    } else {
        *chunk_bounds = BoundingBox::empty();
        None
    };
    tx.send(LoadEvent::Chunk(DrawingChunk::from_parts(
        layers.to_vec(),
        entities,
        entity_bounds,
        bounds,
    )))
    .map_err(|_| "DXF load cancelled".to_string())?;
    tx.send(LoadEvent::Progress { loaded_entities })
        .map_err(|_| "DXF load cancelled".to_string())?;
    Ok(())
}

/// Determine the final RGBA for an entity, respecting BYLAYER fallback.
fn entity_color(
    entity_color: &dxf::Color,
    layer: &str,
    layer_map: &HashMap<String, [f32; 4]>,
) -> [f32; 4] {
    if entity_color.is_by_layer() || entity_color.is_by_block() {
        layer_map
            .get(layer)
            .copied()
            .unwrap_or([1.0, 1.0, 1.0, 1.0])
    } else if let Some(idx) = entity_color.index() {
        aci_to_rgba(idx as i16)
    } else {
        [1.0, 1.0, 1.0, 1.0]
    }
}

fn resolve_color(index: i16) -> [f32; 4] {
    if (1..=255).contains(&index) {
        aci_to_rgba(index)
    } else {
        [1.0, 1.0, 1.0, 1.0]
    }
}

/// Convert a single dxf::Entity to a CadEntity, or None if unsupported.
fn convert_entity(
    entity: &dxf::entities::Entity,
    layer_map: &HashMap<String, [f32; 4]>,
) -> Option<CadEntity> {
    let layer = entity.common.layer.clone();
    let color = entity_color(&entity.common.color, &layer, layer_map);

    match &entity.specific {
        EntityType::Line(line) => Some(CadEntity::Line(LineEntity {
            start: [line.p1.x, line.p1.y],
            end: [line.p2.x, line.p2.y],
            layer,
            color,
        })),

        EntityType::Circle(circle) => Some(CadEntity::Circle(CircleEntity {
            center: [circle.center.x, circle.center.y],
            radius: circle.radius,
            layer,
            color,
        })),

        EntityType::Arc(arc) => Some(CadEntity::Arc(ArcEntity {
            center: [arc.center.x, arc.center.y],
            radius: arc.radius,
            start_angle: arc.start_angle.to_radians(),
            end_angle: arc.end_angle.to_radians(),
            layer,
            color,
        })),

        EntityType::LwPolyline(poly) => {
            let points: Vec<[f64; 2]> = poly.vertices.iter().map(|v| [v.x, v.y]).collect();
            let bulges: Vec<f64> = poly.vertices.iter().map(|v| v.bulge).collect();
            Some(CadEntity::Polyline(PolylineEntity {
                points,
                bulges,
                closed: poly.is_closed(),
                layer,
                color,
            }))
        }

        EntityType::Polyline(poly) => {
            let vertices: Vec<_> = poly.vertices().collect();
            let points: Vec<[f64; 2]> = vertices
                .iter()
                .map(|v| [v.location.x, v.location.y])
                .collect();
            let bulges = vec![0.0; points.len()];
            Some(CadEntity::Polyline(PolylineEntity {
                points,
                bulges,
                closed: poly.is_closed(),
                layer,
                color,
            }))
        }

        EntityType::Ellipse(ellipse) => Some(CadEntity::Ellipse(EllipseEntity {
            center: [ellipse.center.x, ellipse.center.y],
            major_axis: [ellipse.major_axis.x, ellipse.major_axis.y],
            minor_ratio: ellipse.minor_axis_ratio,
            start_param: ellipse.start_parameter,
            end_param: ellipse.end_parameter,
            layer,
            color,
        })),

        EntityType::Spline(spline) => {
            let control_points: Vec<[f64; 2]> =
                spline.control_points.iter().map(|p| [p.x, p.y]).collect();
            Some(CadEntity::Spline(SplineEntity {
                degree: spline.degree_of_curve,
                control_points,
                knots: spline.knot_values.clone(),
                closed: spline.is_closed(),
                layer,
                color,
            }))
        }

        EntityType::Text(text) => Some(CadEntity::Text(TextEntity {
            position: [text.location.x, text.location.y],
            text: text.value.clone(),
            height: text.text_height,
            rotation: text.rotation,
            layer,
            color,
        })),

        EntityType::MText(mtext) => Some(CadEntity::Text(TextEntity {
            position: [mtext.insertion_point.x, mtext.insertion_point.y],
            text: mtext.text.clone(),
            height: mtext.initial_text_height,
            rotation: mtext.rotation_angle,
            layer,
            color,
        })),

        // Unsupported entities are silently skipped
        _ => None,
    }
}

/// Fallback: strip the THUMBNAILIMAGE section from the DXF text and re-parse.
fn load_dxf_without_thumbnail(path: &Path) -> Result<DxfDrawing, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("Cannot read file: {e}"))?;

    let stripped = strip_dxf_section(&content, "THUMBNAILIMAGE");

    DxfDrawing::load(&mut std::io::Cursor::new(stripped.as_bytes()))
        .map_err(|e| format!("DXF parse error (stripped): {e}"))
}

/// Remove a named SECTION…ENDSEC block from DXF text content.
fn strip_dxf_section(content: &str, section_name: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let n = lines.len();
    let mut start = None;
    let mut end = None;

    // Find: lines[i]="0", lines[i+1]="SECTION", lines[i+2]="2", lines[i+3]=section_name
    for i in 0..n.saturating_sub(3) {
        if lines[i].trim() == "0"
            && lines[i + 1].trim() == "SECTION"
            && lines[i + 2].trim() == "2"
            && lines[i + 3].trim() == section_name
        {
            start = Some(i);
            break;
        }
    }

    if let Some(s) = start {
        // Find "0\nENDSEC" after the section start
        for i in (s + 4)..n.saturating_sub(1) {
            if lines[i].trim() == "0" && lines[i + 1].trim() == "ENDSEC" {
                end = Some(i + 2); // skip past ENDSEC line
                break;
            }
        }
    }

    match (start, end) {
        (Some(s), Some(e)) => {
            let mut out = lines[..s].to_vec();
            out.extend_from_slice(&lines[e..]);
            out.join("\n")
        }
        _ => content.to_string(),
    }
}
