use std::collections::HashSet;
use std::f64::consts::{PI, TAU};

use crate::cad::model::{
    ArcEntity, BoundingBox, CadEntity, CircleEntity, EllipseEntity, LineEntity, PolylineEntity, SplineEntity,
};

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

/// Tessellate visible entities into a flat vertex list.
/// Entities on layers in `hidden_layers` are skipped.
/// If `entity_bounds` has the same length as `entities` and `viewport` is Some,
/// entities outside the viewport are culled.
pub fn tessellate_all(
    entities: &[CadEntity],
    entity_bounds: &[BoundingBox],
    pixels_per_unit: f64,
    hidden_layers: &HashSet<String>,
    viewport: Option<&BoundingBox>,
) -> Vec<Vertex> {
    let use_culling = viewport.is_some() && entity_bounds.len() == entities.len();
    let mut verts: Vec<Vertex> = Vec::with_capacity(entities.len() * 4);
    for (i, entity) in entities.iter().enumerate() {
        if hidden_layers.contains(entity.layer()) {
            continue;
        }
        if use_culling {
            if let Some(vp) = viewport {
                if !entity_bounds[i].intersects(vp) {
                    continue;
                }
            }
        }
        tessellate_entity(entity, pixels_per_unit, &mut verts);
    }
    verts
}

fn tessellate_entity(entity: &CadEntity, pixels_per_unit: f64, out: &mut Vec<Vertex>) {
    match entity {
        CadEntity::Line(e) => tessellate_line(e, out),
        CadEntity::Circle(e) => tessellate_circle(e, pixels_per_unit, out),
        CadEntity::Arc(e) => tessellate_arc(e, pixels_per_unit, out),
        CadEntity::Polyline(e) => tessellate_polyline(e, pixels_per_unit, out),
        CadEntity::Ellipse(e) => tessellate_ellipse(e, pixels_per_unit, out),
        CadEntity::Spline(e) => tessellate_spline(e, out),
        CadEntity::Text(_) => {} // text rendered via egui overlay
    }
}

fn tessellate_line(e: &LineEntity, out: &mut Vec<Vertex>) {
    let c = e.color;
    out.push(Vertex::new(e.start[0] as f32, e.start[1] as f32, c));
    out.push(Vertex::new(e.end[0] as f32, e.end[1] as f32, c));
}

fn circle_segments(radius: f64, pixels_per_unit: f64) -> u32 {
    let screen_r = radius * pixels_per_unit;
    // Approx: segment length ≤ 2px
    let n = (PI * screen_r / 2.0).ceil() as u32;
    n.clamp(12, 256)
}

fn tessellate_circle(e: &CircleEntity, pixels_per_unit: f64, out: &mut Vec<Vertex>) {
    let n = circle_segments(e.radius, pixels_per_unit);
    let c = e.color;
    let cx = e.center[0] as f32;
    let cy = e.center[1] as f32;
    let r = e.radius as f32;
    for i in 0..n {
        let a0 = TAU * i as f64 / n as f64;
        let a1 = TAU * (i + 1) as f64 / n as f64;
        out.push(Vertex::new(cx + r * a0.cos() as f32, cy + r * a0.sin() as f32, c));
        out.push(Vertex::new(cx + r * a1.cos() as f32, cy + r * a1.sin() as f32, c));
    }
}

fn tessellate_arc(e: &ArcEntity, pixels_per_unit: f64, out: &mut Vec<Vertex>) {
    let mut start = e.start_angle;
    let mut end = e.end_angle;
    // Ensure arc goes CCW from start to end
    if end <= start {
        end += TAU;
    }
    let sweep = end - start;
    let n = circle_segments(e.radius, pixels_per_unit);
    let seg_count = ((sweep / TAU) * n as f64).ceil() as u32;
    let seg_count = seg_count.max(2);
    let c = e.color;
    let cx = e.center[0] as f32;
    let cy = e.center[1] as f32;
    let r = e.radius as f32;
    for i in 0..seg_count {
        let a0 = start + sweep * i as f64 / seg_count as f64;
        let a1 = start + sweep * (i + 1) as f64 / seg_count as f64;
        out.push(Vertex::new(cx + r * a0.cos() as f32, cy + r * a0.sin() as f32, c));
        out.push(Vertex::new(cx + r * a1.cos() as f32, cy + r * a1.sin() as f32, c));
    }
}

fn tessellate_polyline(e: &PolylineEntity, pixels_per_unit: f64, out: &mut Vec<Vertex>) {
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
            // Straight segment
            let p0 = e.points[i];
            let p1 = e.points[j];
            out.push(Vertex::new(p0[0] as f32, p0[1] as f32, c));
            out.push(Vertex::new(p1[0] as f32, p1[1] as f32, c));
        } else {
            // Bulge arc segment
            tessellate_bulge_arc(e.points[i], e.points[j], bulge, pixels_per_unit, c, out);
        }
    }
}

/// Convert a bulge arc segment to line segments.
/// bulge = tan(included_angle / 4)
fn tessellate_bulge_arc(
    p0: [f64; 2], p1: [f64; 2], bulge: f64,
    pixels_per_unit: f64,
    color: [f32; 4], out: &mut Vec<Vertex>,
) {
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let d = (dx * dx + dy * dy).sqrt();
    if d < 1e-12 { return; }

    let angle = 4.0 * bulge.atan();       // included angle (signed)
    let r = d / (2.0 * (angle / 2.0).sin().abs()); // radius

    // Midpoint of chord, then perpendicular offset to center
    let mx = (p0[0] + p1[0]) * 0.5;
    let my = (p0[1] + p1[1]) * 0.5;
    let perp_x = -dy / d;
    let perp_y = dx / d;
    let sagitta = r - (r * r - (d / 2.0) * (d / 2.0)).sqrt();
    let sign = if bulge > 0.0 { 1.0 } else { -1.0 };
    let offset = r - sign * sagitta; // offset from midpoint to center
    let cx = mx - sign * perp_x * offset;
    let cy = my - sign * perp_y * offset;

    let start_a = (p0[1] - cy).atan2(p0[0] - cx);
    let end_a = (p1[1] - cy).atan2(p1[0] - cx);

    // Ensure CCW direction consistent with bulge sign
    let mut sweep = if bulge > 0.0 {
        let mut s = end_a - start_a;
        if s <= 0.0 { s += TAU; }
        s
    } else {
        let mut s = end_a - start_a;
        if s >= 0.0 { s -= TAU; }
        s
    };
    if sweep == 0.0 { sweep = angle; }

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
        out.push(Vertex::new(x0 as f32, y0 as f32, color));
        out.push(Vertex::new(x1 as f32, y1 as f32, color));
    }
}

fn tessellate_ellipse(e: &EllipseEntity, pixels_per_unit: f64, out: &mut Vec<Vertex>) {
    let major_len = (e.major_axis[0].powi(2) + e.major_axis[1].powi(2)).sqrt();
    let minor_len = major_len * e.minor_ratio;
    let screen_r = major_len.max(minor_len) * pixels_per_unit;
    let n = ((PI * screen_r / 2.0).ceil() as u32).clamp(12, 256);

    let mut start = e.start_param;
    let mut end = e.end_param;
    if end <= start { end += TAU; }
    let sweep = end - start;
    let seg_count = ((sweep / TAU) * n as f64).ceil() as u32;
    let seg_count = seg_count.max(2);

    // Rotation angle of major axis
    let rot = e.major_axis[1].atan2(e.major_axis[0]);
    let cos_r = rot.cos() as f32;
    let sin_r = rot.sin() as f32;
    let a = major_len as f32;
    let b = minor_len as f32;
    let cx = e.center[0] as f32;
    let cy = e.center[1] as f32;
    let c = e.color;

    for i in 0..seg_count {
        let t0 = start + sweep * i as f64 / seg_count as f64;
        let t1 = start + sweep * (i + 1) as f64 / seg_count as f64;

        let (lx0, ly0) = (a * t0.cos() as f32, b * t0.sin() as f32);
        let (lx1, ly1) = (a * t1.cos() as f32, b * t1.sin() as f32);

        let x0 = cx + cos_r * lx0 - sin_r * ly0;
        let y0 = cy + sin_r * lx0 + cos_r * ly0;
        let x1 = cx + cos_r * lx1 - sin_r * ly1;
        let y1 = cy + sin_r * lx1 + cos_r * ly1;

        out.push(Vertex::new(x0, y0, c));
        out.push(Vertex::new(x1, y1, c));
    }
}

/// B-spline evaluation using De Boor's algorithm.
fn tessellate_spline(e: &SplineEntity, out: &mut Vec<Vertex>) {
    if e.control_points.len() < 2 || e.knots.is_empty() {
        return;
    }
    let n_pts = 128.max(e.control_points.len() * 8);
    let c = e.color;

    let t_min = *e.knots.first().unwrap();
    let t_max = *e.knots.last().unwrap();
    if (t_max - t_min).abs() < 1e-12 { return; }

    let mut prev: Option<[f32; 2]> = None;
    for i in 0..=n_pts {
        let t = t_min + (t_max - t_min) * i as f64 / n_pts as f64;
        let pt = de_boor(e.degree as usize, &e.knots, &e.control_points, t);
        if let Some(p0) = prev {
            out.push(Vertex::new(p0[0], p0[1], c));
            out.push(Vertex::new(pt[0] as f32, pt[1] as f32, c));
        }
        prev = Some([pt[0] as f32, pt[1] as f32]);
    }
}

/// De Boor's algorithm for B-spline point evaluation.
fn de_boor(degree: usize, knots: &[f64], ctrl: &[[f64; 2]], t: f64) -> [f64; 2] {
    let n = ctrl.len();
    if n == 0 { return [0.0, 0.0]; }
    if degree == 0 { return ctrl[0]; }

    // Find knot span index k such that knots[k] <= t < knots[k+1]
    let k = {
        let mut idx = degree;
        for i in degree..knots.len().saturating_sub(degree + 1) {
            if t >= knots[i] {
                idx = i;
            }
        }
        idx.min(n - 1)
    };

    // Copy relevant control points
    let lo = k.saturating_sub(degree);
    let hi = (k + 1).min(n);
    let mut d: Vec<[f64; 2]> = ctrl[lo..hi].to_vec();

    for r in 1..=degree {
        for j in (r..d.len()).rev() {
            let i = lo + j;
            let left = knots.get(i).copied().unwrap_or(0.0);
            let right = knots.get(i + degree - r + 1).copied().unwrap_or(0.0);
            let denom = right - left;
            let alpha = if denom.abs() < 1e-12 { 0.0 } else { (t - left) / denom };
            d[j][0] = (1.0 - alpha) * d[j - 1][0] + alpha * d[j][0];
            d[j][1] = (1.0 - alpha) * d[j - 1][1] + alpha * d[j][1];
        }
    }

    *d.last().unwrap_or(&[0.0, 0.0])
}
