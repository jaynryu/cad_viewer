use crate::cad::model::BoundingBox;

/// 2D camera for the CAD viewport.
/// `center` is the world-space point at the viewport center.
/// `zoom` is pixels per world unit.
pub struct Camera2D {
    pub center: [f64; 2],
    pub zoom: f64,
    pub viewport_size: [f32; 2],
}

impl Default for Camera2D {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0],
            zoom: 1.0,
            viewport_size: [1.0, 1.0],
        }
    }
}

impl Camera2D {
    /// Fit the camera so that `bounds` is fully visible with a small margin.
    pub fn fit_to_bounds(&mut self, bounds: &BoundingBox) {
        let w = bounds.width().max(1e-6);
        let h = bounds.height().max(1e-6);
        let [cx, cy] = bounds.center();
        self.center = [cx, cy];
        let zoom_x = self.viewport_size[0] as f64 / w;
        let zoom_y = self.viewport_size[1] as f64 / h;
        self.zoom = zoom_x.min(zoom_y) * 0.9; // 10% margin
    }

    /// Convert a screen position (pixels, top-left origin) to world coords.
    pub fn screen_to_world(&self, screen: [f32; 2]) -> [f64; 2] {
        let vw = self.viewport_size[0] as f64;
        let vh = self.viewport_size[1] as f64;
        let wx = self.center[0] + (screen[0] as f64 - vw * 0.5) / self.zoom;
        // Y is flipped: screen Y increases downward, world Y increases upward
        let wy = self.center[1] - (screen[1] as f64 - vh * 0.5) / self.zoom;
        [wx, wy]
    }

    /// Build an orthographic projection matrix (column-major, for wgpu NDC).
    ///
    /// `tess_origin` is the world-space point that was subtracted from all
    /// vertex positions at tessellation time. Pass the same value here so that
    /// the translation stays a small number (camera_center − tess_origin) and
    /// f32 precision is preserved even for very large coordinate values.
    pub fn view_projection_matrix(&self, tess_origin: [f64; 2]) -> [[f32; 4]; 4] {
        let vw = self.viewport_size[0] as f64;
        let vh = self.viewport_size[1] as f64;

        let sx = (2.0 * self.zoom / vw) as f32;
        let sy = (2.0 * self.zoom / vh) as f32;

        // Translate by (center − tess_origin): both are large but their
        // difference is small, so the cast to f32 stays precise.
        let rel_x = self.center[0] - tess_origin[0];
        let rel_y = self.center[1] - tess_origin[1];
        let tx = (-rel_x * 2.0 * self.zoom / vw) as f32;
        let ty = (-rel_y * 2.0 * self.zoom / vh) as f32;

        [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [tx, ty, 0.0, 1.0],
        ]
    }

    /// Return the world-space bounding box of the current viewport.
    /// `margin` expands the box: 0.0 = exact viewport, 1.0 = 2× size (100% margin each side).
    pub fn viewport_world_bounds(&self, margin: f64) -> BoundingBox {
        let hw = self.viewport_size[0] as f64 * 0.5 / self.zoom * (1.0 + margin);
        let hh = self.viewport_size[1] as f64 * 0.5 / self.zoom * (1.0 + margin);
        BoundingBox {
            min: [self.center[0] - hw, self.center[1] - hh],
            max: [self.center[0] + hw, self.center[1] + hh],
        }
    }

    /// Pan the camera by `delta` screen pixels.
    pub fn pan(&mut self, delta: [f32; 2]) {
        self.center[0] -= delta[0] as f64 / self.zoom;
        self.center[1] += delta[1] as f64 / self.zoom; // flip Y
    }

    /// Zoom centered on `screen_pos` (screen pixels) by a multiplicative `factor`.
    pub fn zoom_at(&mut self, screen_pos: [f32; 2], factor: f64) {
        let world_before = self.screen_to_world(screen_pos);
        self.zoom *= factor;
        self.zoom = self.zoom.clamp(1e-10, 1e10);
        let world_after = self.screen_to_world(screen_pos);
        self.center[0] += world_before[0] - world_after[0];
        self.center[1] += world_before[1] - world_after[1];
    }
}
