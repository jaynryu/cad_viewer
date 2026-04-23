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
        if x < self.min[0] { self.min[0] = x; }
        if y < self.min[1] { self.min[1] = y; }
        if x > self.max[0] { self.max[0] = x; }
        if y > self.max[1] { self.max[1] = y; }
    }

    pub fn is_valid(&self) -> bool {
        self.min[0] <= self.max[0] && self.min[1] <= self.max[1]
    }

    pub fn intersects(&self, other: &BoundingBox) -> bool {
        self.max[0] >= other.min[0] && self.min[0] <= other.max[0] &&
        self.max[1] >= other.min[1] && self.min[1] <= other.max[1]
    }

    pub fn width(&self) -> f64 { self.max[0] - self.min[0] }
    pub fn height(&self) -> f64 { self.max[1] - self.min[1] }
    pub fn center(&self) -> [f64; 2] {
        [(self.min[0] + self.max[0]) * 0.5, (self.min[1] + self.max[1]) * 0.5]
    }
}

/// A CAD layer.
#[derive(Debug, Clone)]
pub struct Layer {
    pub name: String,
    pub color: [f32; 4],
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
    pub closed: bool,
    pub layer: String,
    pub color: [f32; 4],
}

/// TEXT / MTEXT entity
#[derive(Debug, Clone)]
pub struct TextEntity {
    pub position: [f64; 2],
    pub text: String,
    pub height: f64,
    pub rotation: f64,
    pub layer: String,
    pub color: [f32; 4],
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
        }
    }

    pub fn color(&self) -> [f32; 4] {
        match self {
            CadEntity::Line(e) => e.color,
            CadEntity::Circle(e) => e.color,
            CadEntity::Arc(e) => e.color,
            CadEntity::Polyline(e) => e.color,
            CadEntity::Ellipse(e) => e.color,
            CadEntity::Spline(e) => e.color,
            CadEntity::Text(e) => e.color,
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
        }
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
}

impl Drawing {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn compute_bounds(&mut self) {
        let mut bb = BoundingBox::empty();
        self.entity_bounds = self.entities.iter().map(|e| {
            let mut ebb = BoundingBox::empty();
            e.expand_bounds(&mut ebb);
            bb.expand(ebb.min[0], ebb.min[1]);
            bb.expand(ebb.max[0], ebb.max[1]);
            ebb
        }).collect();
        self.bounds = if bb.is_valid() { Some(bb) } else { None };
    }

    pub fn layer_by_name(&self, name: &str) -> Option<&Layer> {
        self.layers.iter().find(|l| l.name == name)
    }
}
