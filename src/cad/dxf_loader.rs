use std::collections::HashMap;
use std::path::Path;

use dxf::entities::EntityType;
use dxf::Drawing as DxfDrawing;

use crate::cad::model::{
    ArcEntity, BoundingBox, CadEntity, CircleEntity, Drawing, EllipseEntity, Layer, LineEntity,
    PolylineEntity, SplineEntity, TextEntity,
};
use crate::util::color::aci_to_rgba;

/// Load a DXF file and convert it to the internal Drawing model.
pub fn load_dxf(path: &Path) -> Result<Drawing, String> {
    let dxf = DxfDrawing::load_file(path.to_str().unwrap_or(""))
        .map_err(|e| format!("DXF parse error: {e}"))?;

    // --- Build layer map ---
    let mut layer_map: HashMap<String, [f32; 4]> = HashMap::new();
    for layer in dxf.layers() {
        let color = resolve_color(layer.color.index().map(|i| i as i16).unwrap_or(7));
        layer_map.insert(layer.name.clone(), color);
    }

    // --- Collect block definitions (for INSERT expansion) ---
    let mut block_map: HashMap<String, Vec<CadEntity>> = HashMap::new();
    for block in dxf.blocks() {
        let block_entities: Vec<CadEntity> = block
            .entities
            .iter()
            .flat_map(|e| convert_entity(e, &layer_map))
            .collect();
        block_map.insert(block.name.clone(), block_entities);
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

    for entity in dxf.entities() {
        if let EntityType::Insert(ref ins) = entity.specific {
            // Expand INSERT: look up block and apply transform
            if let Some(block_entities) = block_map.get(&ins.name) {
                let sx = ins.x_scale_factor;
                let sy = ins.y_scale_factor;
                let rot = ins.rotation.to_radians();
                let tx = ins.location.x;
                let ty = ins.location.y;
                let (sin_r, cos_r) = (rot.sin(), rot.cos());

                for e in block_entities {
                    drawing.entities.push(transform_entity(e, tx, ty, sx, sy, cos_r, sin_r));
                }
            }
        } else if let Some(cad) = convert_entity(entity, &layer_map) {
            drawing.entities.push(cad);
        }
    }

    drawing.compute_bounds();

    log::info!(
        "Loaded {} entities across {} layers from {:?}",
        drawing.entities.len(),
        drawing.layers.len(),
        path.file_name().unwrap_or_default()
    );

    Ok(drawing)
}

/// Determine the final RGBA for an entity, respecting BYLAYER fallback.
fn entity_color(entity_color: &dxf::Color, layer: &str, layer_map: &HashMap<String, [f32; 4]>) -> [f32; 4] {
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
    if index >= 1 && index <= 255 {
        aci_to_rgba(index)
    } else {
        [1.0, 1.0, 1.0, 1.0]
    }
}

/// Convert a single dxf::Entity to a CadEntity, or None if unsupported.
fn convert_entity(entity: &dxf::entities::Entity, layer_map: &HashMap<String, [f32; 4]>) -> Option<CadEntity> {
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
            let points: Vec<[f64; 2]> = vertices.iter().map(|v| [v.location.x, v.location.y]).collect();
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
            let control_points: Vec<[f64; 2]> = spline.control_points.iter()
                .map(|p| [p.x, p.y])
                .collect();
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

/// Apply a 2D transform (translate + scale + rotate) to a CadEntity.
/// Used when flattening INSERT block references.
fn transform_entity(
    entity: &CadEntity,
    tx: f64, ty: f64,
    sx: f64, sy: f64,
    cos_r: f64, sin_r: f64,
) -> CadEntity {
    let tp = |x: f64, y: f64| -> [f64; 2] {
        let lx = x * sx;
        let ly = y * sy;
        [
            cos_r * lx - sin_r * ly + tx,
            sin_r * lx + cos_r * ly + ty,
        ]
    };

    match entity {
        CadEntity::Line(e) => CadEntity::Line(LineEntity {
            start: tp(e.start[0], e.start[1]),
            end: tp(e.end[0], e.end[1]),
            layer: e.layer.clone(),
            color: e.color,
        }),
        CadEntity::Circle(e) => {
            let [cx, cy] = tp(e.center[0], e.center[1]);
            CadEntity::Circle(CircleEntity {
                center: [cx, cy],
                radius: e.radius * sx.abs().max(sy.abs()),
                layer: e.layer.clone(),
                color: e.color,
            })
        }
        CadEntity::Arc(e) => {
            let [cx, cy] = tp(e.center[0], e.center[1]);
            CadEntity::Arc(ArcEntity {
                center: [cx, cy],
                radius: e.radius * sx.abs().max(sy.abs()),
                start_angle: e.start_angle + cos_r.atan2(sin_r),
                end_angle: e.end_angle + cos_r.atan2(sin_r),
                layer: e.layer.clone(),
                color: e.color,
            })
        }
        CadEntity::Polyline(e) => CadEntity::Polyline(PolylineEntity {
            points: e.points.iter().map(|p| tp(p[0], p[1])).collect(),
            bulges: e.bulges.clone(),
            closed: e.closed,
            layer: e.layer.clone(),
            color: e.color,
        }),
        CadEntity::Ellipse(e) => {
            let [cx, cy] = tp(e.center[0], e.center[1]);
            CadEntity::Ellipse(EllipseEntity {
                center: [cx, cy],
                major_axis: [
                    cos_r * e.major_axis[0] * sx - sin_r * e.major_axis[1] * sy,
                    sin_r * e.major_axis[0] * sx + cos_r * e.major_axis[1] * sy,
                ],
                minor_ratio: e.minor_ratio,
                start_param: e.start_param,
                end_param: e.end_param,
                layer: e.layer.clone(),
                color: e.color,
            })
        }
        CadEntity::Spline(e) => CadEntity::Spline(SplineEntity {
            degree: e.degree,
            control_points: e.control_points.iter().map(|p| tp(p[0], p[1])).collect(),
            knots: e.knots.clone(),
            closed: e.closed,
            layer: e.layer.clone(),
            color: e.color,
        }),
        CadEntity::Text(e) => {
            let pos = tp(e.position[0], e.position[1]);
            CadEntity::Text(TextEntity {
                position: pos,
                text: e.text.clone(),
                height: e.height * sy.abs(),
                rotation: e.rotation + cos_r.atan2(sin_r).to_degrees(),
                layer: e.layer.clone(),
                color: e.color,
            })
        }
    }
}

/// Compute bounding box of visible entities (used for camera fit-to-view).
pub fn compute_bounds(entities: &[CadEntity]) -> Option<BoundingBox> {
    let mut bb = BoundingBox::empty();
    for e in entities {
        e.expand_bounds(&mut bb);
    }
    if bb.is_valid() { Some(bb) } else { None }
}
