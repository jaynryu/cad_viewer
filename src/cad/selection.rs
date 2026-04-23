use crate::cad::model::{BoundingBox, CadEntity};

/// Find the index of the entity closest to `world_pos` within `tolerance` world units.
pub fn find_entity_at(
    entities: &[CadEntity],
    entity_bounds: &[BoundingBox],
    world_pos: [f64; 2],
    tolerance: f64,
) -> Option<usize> {
    let has_bounds = entity_bounds.len() == entities.len();

    // Expand a small box around the click point for quick pre-filter
    let query = BoundingBox {
        min: [world_pos[0] - tolerance, world_pos[1] - tolerance],
        max: [world_pos[0] + tolerance, world_pos[1] + tolerance],
    };

    let mut best_idx = None;
    let mut best_dist = f64::MAX;

    for (i, entity) in entities.iter().enumerate() {
        // Quick bbox pre-filter
        if has_bounds && !entity_bounds[i].intersects(&query) {
            continue;
        }

        let d = entity_dist(entity, world_pos);
        if d < tolerance && d < best_dist {
            best_dist = d;
            best_idx = Some(i);
        }
    }
    best_idx
}

fn entity_dist(entity: &CadEntity, p: [f64; 2]) -> f64 {
    match entity {
        CadEntity::Line(e) => segment_dist(p, e.start, e.end),
        CadEntity::Circle(e) => (dist(p, e.center) - e.radius).abs(),
        CadEntity::Arc(e) => {
            let d = dist(p, e.center);
            let angle = (p[1] - e.center[1]).atan2(p[0] - e.center[0]);
            if angle_in_arc(angle, e.start_angle, e.end_angle) {
                (d - e.radius).abs()
            } else {
                // Distance to nearest arc endpoint
                let a0 = [
                    e.center[0] + e.radius * e.start_angle.cos(),
                    e.center[1] + e.radius * e.start_angle.sin(),
                ];
                let a1 = [
                    e.center[0] + e.radius * e.end_angle.cos(),
                    e.center[1] + e.radius * e.end_angle.sin(),
                ];
                dist(p, a0).min(dist(p, a1))
            }
        }
        CadEntity::Polyline(e) => {
            if e.points.len() < 2 {
                return e.points.first().map(|&pt| dist(p, pt)).unwrap_or(f64::MAX);
            }
            let n = e.points.len();
            let seg_count = if e.closed { n } else { n - 1 };
            (0..seg_count)
                .map(|i| segment_dist(p, e.points[i], e.points[(i + 1) % n]))
                .fold(f64::MAX, f64::min)
        }
        CadEntity::Ellipse(e) => {
            // Approximate: distance to ellipse center minus semi-major
            let major = (e.major_axis[0].powi(2) + e.major_axis[1].powi(2)).sqrt();
            (dist(p, e.center) - major).abs()
        }
        CadEntity::Spline(e) => {
            // Approximate: distance to nearest control point
            e.control_points
                .iter()
                .map(|&cp| dist(p, cp))
                .fold(f64::MAX, f64::min)
        }
        CadEntity::Text(e) => dist(p, e.position),
    }
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

fn segment_dist(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-20 {
        return dist(p, a);
    }
    let t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let closest = [a[0] + t * dx, a[1] + t * dy];
    dist(p, closest)
}

fn angle_in_arc(angle: f64, start: f64, end: f64) -> bool {
    use std::f64::consts::TAU;
    let mut a = angle;
    let mut s = start;
    let mut e = end;
    // Normalize to [0, TAU)
    a = ((a % TAU) + TAU) % TAU;
    s = ((s % TAU) + TAU) % TAU;
    e = ((e % TAU) + TAU) % TAU;
    if e <= s { e += TAU; }
    if a < s { a += TAU; }
    a <= e
}
