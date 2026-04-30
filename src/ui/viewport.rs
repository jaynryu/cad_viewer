use std::collections::HashSet;
use std::sync::{mpsc, Arc};

use egui::{Response, Sense, Ui};

use crate::cad::model::{BoundingBox, Drawing};
use crate::cad::selection::find_entity_at;
use crate::cad::snap::{find_snap, SnapKind, SnapPoint};
use crate::render::camera::Camera2D;
use crate::render::pipeline::{CadPaintCallback, CadRenderResources, CameraUniform};
use crate::render::tessellator::{
    tessellate_owned_items, CellCache, OwnedRenderItem, RenderOutput,
};

/// State for the two-point distance measurement tool.
#[derive(Clone, Debug, Default)]
pub enum MeasureState {
    #[default]
    Idle,
    FirstPicked([f64; 2]),
    Done([f64; 2], [f64; 2]),
}

/// Result delivered from the background tessellation thread.
struct BackgroundTessResult {
    /// Tessellation output for GPU upload (Arc avoids copying).
    output: Arc<RenderOutput>,
    /// Cell cache returned so the main thread can persist entries across frames.
    cell_cache: CellCache,
    tess_origin: [f64; 2],
    tess_viewport: BoundingBox,
}

pub struct Viewport {
    pub camera: Camera2D,

    /// True when geometry must be re-tessellated (new file, layer toggle, etc.).
    needs_tessellation: bool,
    /// Last zoom level used for tessellation (for LOD rebuild decision).
    last_tess_zoom: f64,
    /// Latest tessellation output; sent to CadPaintCallback when ready.
    pending_output: Option<Arc<RenderOutput>>,
    /// Receiver for the background tessellation thread result. None when idle.
    tess_rx: Option<mpsc::Receiver<BackgroundTessResult>>,

    /// Deferred fit: apply on first frame when viewport size is known.
    pending_fit: Option<BoundingBox>,
    pub hidden_layers: HashSet<String>,
    pub selected: Option<usize>,
    pub measure: MeasureState,
    pub measure_mode: bool,
    snap_point: Option<SnapPoint>,
    /// When false, polygons (GDS BOUNDARY) are drawn as outlines only.
    pub fill_polygons: bool,
    /// Pre-tessellated primitive geometry per GDS cell (earcut runs once per cell).
    /// Cleared when a new file is loaded; entries accumulate across zoom/pan.
    gds_cell_cache: CellCache,
    /// World-space origin subtracted from vertex positions at tessellation time.
    tess_origin: [f64; 2],
    /// World-space bounding box used for viewport culling during the last tessellation.
    /// When the camera moves outside this box, re-tessellation is triggered.
    tess_viewport: BoundingBox,
    /// Signal to the GPU: discard all cell geometry buffers (new file loaded).
    clear_cell_geom: bool,
}

impl Viewport {
    pub fn new(wgpu_state: &eframe::egui_wgpu::RenderState) -> Self {
        let resources = CadRenderResources::new(&wgpu_state.device, wgpu_state.target_format);
        wgpu_state
            .renderer
            .write()
            .callback_resources
            .insert(resources);

        Self {
            camera: Camera2D::default(),
            needs_tessellation: true,
            last_tess_zoom: 0.0,
            pending_output: None,
            tess_rx: None,
            pending_fit: None,
            hidden_layers: HashSet::new(),
            selected: None,
            measure: MeasureState::Idle,
            measure_mode: false,
            snap_point: None,
            fill_polygons: true,
            gds_cell_cache: CellCache::new(),
            tess_origin: [0.0, 0.0],
            tess_viewport: BoundingBox::empty(),
            clear_cell_geom: false,
        }
    }

    /// Signal that geometry has changed and full re-tessellation is needed.
    pub fn mark_dirty(&mut self) {
        self.needs_tessellation = true;
    }

    /// Called when the layer visibility set changes.
    /// Clears the GDS cell cache (CPU + GPU) so cells are rebuilt with the new layer filter.
    pub fn mark_layers_changed(&mut self) {
        self.gds_cell_cache.clear();
        self.clear_cell_geom = true;
        self.needs_tessellation = true;
    }

    /// Discard the GDS cell primitive cache (call when a new file is loaded).
    /// Also cancels any in-flight background tessellation for the previous file.
    pub fn reset_cell_cache(&mut self) {
        self.gds_cell_cache.clear();
        self.pending_output = None;
        // Drop the receiver — the background thread will see a closed channel and exit.
        self.tess_rx = None;
        // Signal the GPU to discard stale cell geometry from the previous file.
        self.clear_cell_geom = true;
    }

    pub fn request_fit(&mut self, bounds: BoundingBox) {
        self.pending_fit = Some(bounds);
        self.needs_tessellation = true;
    }

    pub fn show(&mut self, ui: &mut Ui, drawing: Option<&Drawing>) {
        let available = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(available, Sense::click_and_drag());

        self.camera.viewport_size = [available.width(), available.height()];

        // Apply deferred fit once viewport size is properly known (> 10px)
        if available.width() > 10.0 && available.height() > 10.0 {
            if let Some(bounds) = self.pending_fit.take() {
                self.camera.fit_to_bounds(&bounds);
                self.needs_tessellation = true;
                ui.ctx().request_repaint();
            }
        }

        // Snap computation (measure mode only)
        if self.measure_mode {
            let snap_tol = 15.0 / self.camera.zoom;
            let hover_world = ui.input(|i| i.pointer.hover_pos()).and_then(|pos| {
                if available.contains(pos) {
                    let local = [pos.x - available.min.x, pos.y - available.min.y];
                    Some(self.camera.screen_to_world(local))
                } else {
                    None
                }
            });
            self.snap_point = hover_world.and_then(|w| {
                drawing.and_then(|d| find_snap(&d.entities, &d.entity_bounds, w, snap_tol))
            });
            if self.snap_point.is_some() {
                ui.ctx().request_repaint();
            }
        } else {
            self.snap_point = None;
        }

        self.handle_input(ui, &response, drawing);

        // ── Viewport-change culling ───────────────────────────────────────────
        {
            let current_vp = self.camera.viewport_world_bounds(0.0);
            if !self.tess_viewport.contains_bounds(&current_vp) {
                self.needs_tessellation = true;
            }
        }

        // ── Precision-driven retessellation ──────────────────────────────────
        {
            let max_offset = (self.camera.center[0] - self.tess_origin[0])
                .abs()
                .max((self.camera.center[1] - self.tess_origin[1]).abs());
            let f32_precision = max_offset * 1.192e-7;
            let pixel_world = 1.0 / self.camera.zoom.max(1e-30);
            if f32_precision > pixel_world * 0.1 {
                self.needs_tessellation = true;
            }
        }

        // ── LOD-driven retessellation ─────────────────────────────────────────
        if self.camera.zoom > self.last_tess_zoom * 2.0
            || self.camera.zoom < self.last_tess_zoom * 0.5
        {
            self.needs_tessellation = true;
        }

        // ── Poll background tessellation result ───────────────────────────────
        let mut upload_geometry = false;
        if let Some(rx) = &self.tess_rx {
            match rx.try_recv() {
                Ok(result) => {
                    self.gds_cell_cache = result.cell_cache;
                    self.pending_output = Some(result.output);
                    self.tess_origin = result.tess_origin;
                    self.tess_viewport = result.tess_viewport;
                    self.tess_rx = None;
                    upload_geometry = true;
                    // Re-check if camera has moved significantly while tessellation ran.
                    let current_vp = self.camera.viewport_world_bounds(0.0);
                    if !self.tess_viewport.contains_bounds(&current_vp) {
                        self.needs_tessellation = true;
                    }
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.tess_rx = None;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    // Still running — keep showing existing GPU content.
                    ui.ctx().request_repaint();
                }
            }
        }

        // ── Spawn background tessellation if needed ───────────────────────────
        if self.needs_tessellation && self.tess_rx.is_none() {
            self.needs_tessellation = false;
            self.last_tess_zoom = self.camera.zoom;

            match drawing {
                Some(d) => {
                    let origin = self.camera.center;
                    let tess_viewport = self.camera.viewport_world_bounds(1.0);
                    let zoom = self.camera.zoom;
                    let fill_polygons = self.fill_polygons;

                    let items: Vec<OwnedRenderItem> = d
                        .query_render_items(&tess_viewport, zoom, &self.hidden_layers)
                        .iter()
                        .map(OwnedRenderItem::from_render_item)
                        .collect();

                    let cache = std::mem::take(&mut self.gds_cell_cache);
                    let gds_cells = Arc::clone(&d.gds_cells);
                    let hidden_layers = self.hidden_layers.clone();

                    let (tx, rx) = mpsc::channel();
                    self.tess_rx = Some(rx);
                    let ctx = ui.ctx().clone();

                    std::thread::spawn(move || {
                        let hidden = hidden_layers.clone();
                        let mut output = tessellate_owned_items(
                            items,
                            zoom,
                            origin,
                            fill_polygons,
                            &gds_cells,
                            cache,
                            Some(&tess_viewport),
                            &hidden,
                        );
                        // Extract the cell cache before Arc-wrapping the output.
                        let cell_cache = std::mem::take(&mut output.cell_cache);
                        let result = BackgroundTessResult {
                            output: Arc::new(output),
                            cell_cache,
                            tess_origin: origin,
                            tess_viewport,
                        };
                        let _ = tx.send(result);
                        ctx.request_repaint();
                    });
                }
                None => {
                    self.pending_output = None;
                    upload_geometry = true;
                }
            }
        }

        let camera_uniform = CameraUniform {
            view_proj: self.camera.view_projection_matrix(self.tess_origin),
        };

        let clear_cells = self.clear_cell_geom;
        self.clear_cell_geom = false;

        let callback = eframe::egui_wgpu::Callback::new_paint_callback(
            available,
            CadPaintCallback {
                render_output: if upload_geometry {
                    self.pending_output.clone()
                } else {
                    None
                },
                camera_uniform,
                needs_rebuild: upload_geometry,
                clear_cells,
            },
        );
        ui.painter().add(callback);

        if drawing.is_none() {
            ui.painter().text(
                available.center(),
                egui::Align2::CENTER_CENTER,
                "Drop a DXF/DWG file here\nor use File > Open",
                egui::FontId::proportional(24.0),
                egui::Color32::from_gray(120),
            );
        }

        self.draw_measure_overlay(ui, &response, drawing);

        // Measure mode crosshair cursor
        if self.measure_mode {
            if let Some(hover) = ui.input(|i| i.pointer.hover_pos()) {
                if available.contains(hover) {
                    let painter = ui.painter();
                    let color = egui::Color32::from_rgba_unmultiplied(255, 200, 0, 180);
                    let len = 12.0;
                    painter.line_segment(
                        [hover - egui::vec2(len, 0.0), hover + egui::vec2(len, 0.0)],
                        egui::Stroke::new(1.0, color),
                    );
                    painter.line_segment(
                        [hover - egui::vec2(0.0, len), hover + egui::vec2(0.0, len)],
                        egui::Stroke::new(1.0, color),
                    );
                    ui.ctx().request_repaint();
                }
            }
        }
    }

    fn world_to_screen(&self, world: [f64; 2], rect: egui::Rect) -> egui::Pos2 {
        let vw = self.camera.viewport_size[0] as f64;
        let vh = self.camera.viewport_size[1] as f64;
        let sx = (world[0] - self.camera.center[0]) * self.camera.zoom + vw * 0.5;
        let sy = -(world[1] - self.camera.center[1]) * self.camera.zoom + vh * 0.5;
        egui::pos2(rect.min.x + sx as f32, rect.min.y + sy as f32)
    }

    fn draw_measure_overlay(&self, ui: &mut Ui, response: &Response, _drawing: Option<&Drawing>) {
        let rect = response.rect;
        let painter = ui.painter();
        let yellow = egui::Color32::from_rgb(255, 220, 50);
        let dot_color = egui::Color32::from_rgb(255, 160, 0);
        let text_bg = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 200);

        let draw_point = |p: [f64; 2]| {
            let sp = self.world_to_screen(p, rect);
            painter.circle_filled(sp, 5.0, dot_color);
            painter.circle_stroke(sp, 5.0, egui::Stroke::new(1.5, yellow));
        };

        let draw_dashed_line = |a: [f64; 2], b: [f64; 2]| {
            let sa = self.world_to_screen(a, rect);
            let sb = self.world_to_screen(b, rect);
            let steps = 20usize;
            for i in 0..steps {
                if i % 2 == 0 {
                    let t0 = i as f32 / steps as f32;
                    let t1 = (i + 1) as f32 / steps as f32;
                    let p0 = sa + (sb - sa) * t0;
                    let p1 = sa + (sb - sa) * t1;
                    painter.line_segment([p0, p1], egui::Stroke::new(1.5, yellow));
                }
            }
        };

        let draw_distance_label = |a: [f64; 2], b: [f64; 2]| {
            let sa = self.world_to_screen(a, rect);
            let sb = self.world_to_screen(b, rect);
            let dx = b[0] - a[0];
            let dy = b[1] - a[1];
            let dist = (dx * dx + dy * dy).sqrt();
            let label = if dist >= 1000.0 {
                format!("{:.2} m", dist / 1000.0)
            } else if dist >= 1.0 {
                format!("{:.3}", dist)
            } else {
                format!("{:.6}", dist)
            };
            let mid = egui::pos2((sa.x + sb.x) * 0.5, (sa.y + sb.y) * 0.5);
            let font = egui::FontId::proportional(14.0);
            let galley = painter.layout_no_wrap(label.clone(), font.clone(), yellow);
            let text_rect = egui::Rect::from_center_size(
                mid + egui::vec2(0.0, -16.0),
                galley.size() + egui::vec2(6.0, 4.0),
            );
            painter.rect_filled(text_rect, 3.0, text_bg);
            painter.text(
                mid + egui::vec2(0.0, -16.0),
                egui::Align2::CENTER_CENTER,
                &label,
                font,
                yellow,
            );
        };

        if let Some(sp) = &self.snap_point {
            self.draw_snap_indicator(painter, sp, rect);
        }

        match &self.measure {
            MeasureState::Idle => {}
            MeasureState::FirstPicked(a) => {
                draw_point(*a);
                let world_b = if let Some(sp) = &self.snap_point {
                    Some(sp.world)
                } else {
                    ui.input(|i| i.pointer.hover_pos()).and_then(|hover| {
                        if rect.contains(hover) {
                            let local = [hover.x - rect.min.x, hover.y - rect.min.y];
                            Some(self.camera.screen_to_world(local))
                        } else {
                            None
                        }
                    })
                };
                if let Some(b) = world_b {
                    draw_dashed_line(*a, b);
                    draw_distance_label(*a, b);
                }
            }
            MeasureState::Done(a, b) => {
                draw_point(*a);
                draw_point(*b);
                draw_dashed_line(*a, *b);
                draw_distance_label(*a, *b);
            }
        }
    }

    fn draw_snap_indicator(&self, painter: &egui::Painter, sp: &SnapPoint, rect: egui::Rect) {
        let pos = self.world_to_screen(sp.world, rect);
        let snap_color = egui::Color32::from_rgb(0, 220, 255);
        let stroke = egui::Stroke::new(2.0, snap_color);
        let size = 8.0f32;
        match sp.kind {
            SnapKind::Endpoint => {
                painter.rect_stroke(
                    egui::Rect::from_center_size(pos, egui::vec2(size * 2.0, size * 2.0)),
                    0.0,
                    stroke,
                );
            }
            SnapKind::Center => {
                painter.circle_stroke(pos, size, stroke);
                let s = size * 0.6;
                painter.line_segment([pos - egui::vec2(s, 0.0), pos + egui::vec2(s, 0.0)], stroke);
                painter.line_segment([pos - egui::vec2(0.0, s), pos + egui::vec2(0.0, s)], stroke);
            }
            SnapKind::Midpoint => {
                let h = size * 1.2;
                let w = size;
                let pts = [
                    pos + egui::vec2(0.0, -h),
                    pos + egui::vec2(-w, h * 0.5),
                    pos + egui::vec2(w, h * 0.5),
                ];
                painter.line_segment([pts[0], pts[1]], stroke);
                painter.line_segment([pts[1], pts[2]], stroke);
                painter.line_segment([pts[2], pts[0]], stroke);
            }
        }
    }

    fn handle_input(&mut self, ui: &mut Ui, response: &Response, drawing: Option<&Drawing>) {
        // M key: toggle measure mode
        if ui.input(|i| i.key_pressed(egui::Key::M)) {
            self.measure_mode = !self.measure_mode;
            if !self.measure_mode {
                self.measure = MeasureState::Idle;
            }
        }

        // Pan: middle-mouse drag, or left drag (not in measure mode)
        let is_dragging = response.dragged_by(egui::PointerButton::Middle)
            || (!self.measure_mode && response.dragged_by(egui::PointerButton::Primary));
        if is_dragging {
            let delta = response.drag_delta();
            self.camera.pan([delta.x, delta.y]);
        }

        // Click: measure or entity selection
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let origin = response.rect.min;
                let local = [pos.x - origin.x, pos.y - origin.y];
                let world = self.camera.screen_to_world(local);

                if self.measure_mode {
                    let pick = self.snap_point.as_ref().map(|sp| sp.world).unwrap_or(world);
                    match self.measure {
                        MeasureState::Idle | MeasureState::Done(_, _) => {
                            self.measure = MeasureState::FirstPicked(pick);
                        }
                        MeasureState::FirstPicked(a) => {
                            self.measure = MeasureState::Done(a, pick);
                        }
                    }
                } else if let Some(d) = drawing {
                    let tolerance = 5.0 / self.camera.zoom;
                    self.selected = find_entity_at(&d.entities, &d.entity_bounds, world, tolerance);
                }
            }
        }

        // Zoom: scroll wheel (cursor-centered)
        let (scroll, hover_pos) = ui.input(|i| (i.smooth_scroll_delta.y, i.pointer.hover_pos()));
        if scroll.abs() > 0.1 {
            let origin = response.rect.min;
            let pos = hover_pos.unwrap_or(response.rect.center());
            let local_pos = [pos.x - origin.x, pos.y - origin.y];
            let factor = (1.0_f64 + scroll as f64 * 0.003).clamp(0.5, 2.0);
            self.camera.zoom_at(local_pos, factor);
        }

        // F key: fit to bounds
        if ui.input(|i| i.key_pressed(egui::Key::F)) {
            if let Some(d) = drawing {
                if let Some(bounds) = &d.bounds {
                    self.camera.fit_to_bounds(bounds);
                    self.needs_tessellation = true;
                }
            }
        }

        // Esc: deselect / clear measure
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.selected = None;
            self.measure = MeasureState::Idle;
            self.measure_mode = false;
        }
    }
}
