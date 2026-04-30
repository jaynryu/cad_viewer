use std::f64::consts::TAU;

use crate::cad::model::{BoundingBox, CadEntity};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SnapKind {
    Endpoint,
    Midpoint,
    Center,
}

#[derive(Debug, Clone)]
pub struct SnapPoint {
    pub world: [f64; 2],
    pub kind: SnapKind,
}

/// Find the best snap point near `cursor` within `tolerance` world units.
/// Priority: Endpoint > Center > Midpoint.
pub fn find_snap(
    entities: &[CadEntity],
    entity_bounds: &[BoundingBox],
    cursor: [f64; 2],
    tolerance: f64,
) -> Option<SnapPoint> {
    let has_bounds = entity_bounds.len() == entities.len();
    let query = BoundingBox {
        min: [cursor[0] - tolerance, cursor[1] - tolerance],
        max: [cursor[0] + tolerance, cursor[1] + tolerance],
    };

    let mut best: Option<(f64, SnapPoint)> = None;

    let mut try_point = |world: [f64; 2], kind: SnapKind| {
        let d = dist(cursor, world);
        if d <= tolerance {
            // Priority score: endpoint=0, center=1, midpoint=2
            let priority = match kind {
                SnapKind::Endpoint => 0.0f64,
                SnapKind::Center => 1.0,
                SnapKind::Midpoint => 2.0,
            };
            // Compare: lower priority wins, then closer distance
            let score = priority * tolerance * 2.0 + d;
            if best.as_ref().is_none_or(|(s, _)| score < *s) {
                best = Some((score, SnapPoint { world, kind }));
            }
        }
    };

    for (i, entity) in entities.iter().enumerate() {
        if has_bounds && !entity_bounds[i].intersects(&query) {
            continue;
        }
        collect_snap_candidates(entity, &mut try_point);
    }

    best.map(|(_, sp)| sp)
}

fn collect_snap_candidates(entity: &CadEntity, cb: &mut impl FnMut([f64; 2], SnapKind)) {
    match entity {
        CadEntity::Line(l) => {
            cb(l.start, SnapKind::Endpoint);
            cb(l.end, SnapKind::Endpoint);
            cb(midpoint(l.start, l.end), SnapKind::Midpoint);
        }
        CadEntity::Circle(c) => {
            cb(c.center, SnapKind::Center);
            // 4 quadrant points
            for i in 0..4 {
                let a = TAU * i as f64 / 4.0;
                cb(
                    [
                        c.center[0] + c.radius * a.cos(),
                        c.center[1] + c.radius * a.sin(),
                    ],
                    SnapKind::Endpoint,
                );
            }
        }
        CadEntity::Arc(a) => {
            cb(a.center, SnapKind::Center);
            // Start and end points of arc
            cb(
                [
                    a.center[0] + a.radius * a.start_angle.cos(),
                    a.center[1] + a.radius * a.start_angle.sin(),
                ],
                SnapKind::Endpoint,
            );
            cb(
                [
                    a.center[0] + a.radius * a.end_angle.cos(),
                    a.center[1] + a.radius * a.end_angle.sin(),
                ],
                SnapKind::Endpoint,
            );
        }
        CadEntity::Polyline(p) => {
            let n = p.points.len();
            if n == 0 {
                return;
            }
            for &pt in &p.points {
                cb(pt, SnapKind::Endpoint);
            }
            // Midpoints of each segment
            let seg_count = if p.closed { n } else { n - 1 };
            for i in 0..seg_count {
                let a = p.points[i];
                let b = p.points[(i + 1) % n];
                cb(midpoint(a, b), SnapKind::Midpoint);
            }
        }
        CadEntity::Ellipse(e) => {
            cb(e.center, SnapKind::Center);
        }
        CadEntity::Spline(s) => {
            if let Some(&first) = s.control_points.first() {
                cb(first, SnapKind::Endpoint);
            }
            if let Some(&last) = s.control_points.last() {
                cb(last, SnapKind::Endpoint);
            }
        }
        CadEntity::Text(t) => {
            cb(t.position, SnapKind::Endpoint);
        }
        CadEntity::Polygon(p) => {
            let n = p.points.len();
            for &pt in &p.points {
                cb(pt, SnapKind::Endpoint);
            }
            let seg_count = if n > 1 && p.points[0] == p.points[n - 1] {
                n - 1
            } else {
                n
            };
            for i in 0..seg_count.saturating_sub(1) {
                cb(midpoint(p.points[i], p.points[i + 1]), SnapKind::Midpoint);
            }
        }
        CadEntity::GdsInstance(_) | CadEntity::GdsArrayInstance(_) | CadEntity::DxfInsert(_) => {
            // No snap candidates for cell instance references.
        }
    }
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

fn midpoint(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}
