use std::path::{Path, PathBuf};
use std::sync::mpsc;

use crate::cad::dxf_loader::load_dxf;
use crate::cad::model::{CadEntity, Drawing};
use crate::ui::viewport::Viewport;

const MAX_RECENT: usize = 10;

pub struct CadViewerApp {
    viewport: Option<Viewport>,
    drawing: Option<Drawing>,
    error_msg: Option<String>,
    file_rx: Option<mpsc::Receiver<PathBuf>>,
    drawing_rx: Option<mpsc::Receiver<Result<Drawing, String>>>,
    loading: bool,
    show_layers: bool,
    recent_files: Vec<PathBuf>,
}

impl CadViewerApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let viewport = cc
            .wgpu_render_state
            .as_ref()
            .map(|wgpu_state| Viewport::new(wgpu_state));

        let recent_files = load_recent(&recent_files_path());

        Self {
            viewport,
            drawing: None,
            error_msg: None,
            file_rx: None,
            drawing_rx: None,
            loading: false,
            show_layers: true,
            recent_files,
        }
    }

    fn open_file_dialog(&mut self, ctx: &egui::Context) {
        let (tx, rx) = mpsc::channel();
        self.file_rx = Some(rx);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let path = rfd::FileDialog::new()
                .add_filter("CAD Files", &["dxf", "dwg"])
                .pick_file();
            if let Some(p) = path {
                let _ = tx.send(p);
                ctx.request_repaint();
            }
        });
    }

    fn load_file(&mut self, path: PathBuf, ctx: &egui::Context) {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "dxf" => {
                self.loading = true;
                self.error_msg = None;
                // Add to recent files
                self.push_recent(path.clone());
                let (tx, rx) = mpsc::channel();
                self.drawing_rx = Some(rx);
                let ctx = ctx.clone();
                std::thread::spawn(move || {
                    log::info!("Loading: {:?}", path);
                    let result = load_dxf(&path);
                    let _ = tx.send(result);
                    ctx.request_repaint();
                });
            }
            "dwg" => self.error_msg = Some("DWG support coming soon".to_string()),
            _ => self.error_msg = Some(format!("Unsupported file type: .{ext}")),
        }
    }

    fn push_recent(&mut self, path: PathBuf) {
        self.recent_files.retain(|p| p != &path);
        self.recent_files.insert(0, path);
        self.recent_files.truncate(MAX_RECENT);
        save_recent(&recent_files_path(), &self.recent_files);
    }

    fn apply_drawing(&mut self, drawing: Drawing) {
        log::info!("Applied drawing: {} entities", drawing.entities.len());
        self.drawing = Some(drawing);
        if let Some(vp) = &mut self.viewport {
            vp.hidden_layers.clear();
            vp.selected = None;
            if let Some(d) = &self.drawing {
                if let Some(bounds) = &d.bounds {
                    vp.request_fit(bounds.clone());
                }
            }
            vp.mark_dirty();
        }
    }

    fn show_layer_panel(&mut self, ctx: &egui::Context) {
        let Some(drawing) = &self.drawing else { return };
        let Some(viewport) = &mut self.viewport else { return };

        egui::SidePanel::left("layer_panel")
            .min_width(160.0)
            .max_width(300.0)
            .show(ctx, |ui| {
                ui.heading("Layers");
                ui.separator();

                ui.horizontal(|ui| {
                    if ui.small_button("All On").clicked() {
                        viewport.hidden_layers.clear();
                        viewport.mark_dirty();
                    }
                    if ui.small_button("All Off").clicked() {
                        for layer in &drawing.layers {
                            viewport.hidden_layers.insert(layer.name.clone());
                        }
                        viewport.mark_dirty();
                    }
                });
                ui.separator();

                egui::ScrollArea::vertical().show(ui, |ui| {
                    for layer in &drawing.layers {
                        let is_visible = !viewport.hidden_layers.contains(&layer.name);
                        let [r, g, b, _] = layer.color;
                        let color = egui::Color32::from_rgb(
                            (r * 255.0) as u8,
                            (g * 255.0) as u8,
                            (b * 255.0) as u8,
                        );

                        ui.horizontal(|ui| {
                            let (rect, _) = ui.allocate_exact_size(
                                egui::vec2(14.0, 14.0),
                                egui::Sense::hover(),
                            );
                            ui.painter().rect_filled(rect, 2.0, color);

                            let mut visible = is_visible;
                            if ui.checkbox(&mut visible, &layer.name).changed() {
                                if visible {
                                    viewport.hidden_layers.remove(&layer.name);
                                } else {
                                    viewport.hidden_layers.insert(layer.name.clone());
                                }
                                viewport.mark_dirty();
                            }
                        });
                    }
                });
            });
    }

    fn show_entity_panel(&self, ctx: &egui::Context) {
        let Some(vp) = &self.viewport else { return };
        let Some(idx) = vp.selected else { return };
        let Some(drawing) = &self.drawing else { return };
        let Some(entity) = drawing.entities.get(idx) else { return };

        egui::TopBottomPanel::bottom("entity_info")
            .min_height(80.0)
            .max_height(140.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading(entity_type_name(entity));
                    ui.separator();
                    ui.label(format!("Layer: {}", entity.layer()));
                });
                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    entity_details(entity, ui);
                });
            });
    }
}

impl eframe::App for CadViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll file dialog result
        if let Some(rx) = &self.file_rx {
            if let Ok(path) = rx.try_recv() {
                self.file_rx = None;
                self.load_file(path, ctx);
            }
        }

        // Poll background drawing load
        if let Some(rx) = &self.drawing_rx {
            if let Ok(result) = rx.try_recv() {
                self.drawing_rx = None;
                self.loading = false;
                match result {
                    Ok(drawing) => self.apply_drawing(drawing),
                    Err(e) => {
                        log::error!("Load error: {e}");
                        self.error_msg = Some(e);
                    }
                }
            } else if self.loading {
                ctx.request_repaint();
            }
        }

        // Handle drag-and-drop
        let dropped_files = ctx.input(|i| i.raw.dropped_files.clone());
        for file in dropped_files {
            if let Some(path) = file.path {
                self.load_file(path, ctx);
            }
        }

        // Cmd+O shortcut
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::O)) {
            self.open_file_dialog(ctx);
        }

        // Menu bar
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open…   ⌘O").clicked() {
                        ui.close_menu();
                        self.open_file_dialog(ctx);
                    }

                    if !self.recent_files.is_empty() {
                        ui.separator();
                        ui.label("Recent Files");
                        // Collect to avoid borrow issues
                        let recents: Vec<PathBuf> = self.recent_files.clone();
                        for path in &recents {
                            let name = path
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_else(|| path.to_string_lossy().to_string());
                            if ui.button(&name).clicked() {
                                ui.close_menu();
                                self.load_file(path.clone(), ctx);
                            }
                        }
                    }
                });

                ui.menu_button("View", |ui| {
                    if ui.checkbox(&mut self.show_layers, "Layers Panel").clicked() {
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Fit to Drawing   F").clicked() {
                        ui.close_menu();
                        if let (Some(vp), Some(d)) = (&mut self.viewport, &self.drawing) {
                            if let Some(bounds) = &d.bounds {
                                vp.request_fit(bounds.clone());
                            }
                        }
                    }
                });

                ui.menu_button("Tools", |ui| {
                    let measuring = self.viewport.as_ref().map(|v| v.measure_mode).unwrap_or(false);
                    let label = if measuring { "Measure (ON)   M" } else { "Measure   M" };
                    if ui.button(label).clicked() {
                        ui.close_menu();
                        if let Some(vp) = &mut self.viewport {
                            vp.measure_mode = !vp.measure_mode;
                            if !vp.measure_mode {
                                vp.measure = crate::ui::viewport::MeasureState::Idle;
                            }
                        }
                    }
                });

                ui.separator();

                let count = self.drawing.as_ref().map(|d| d.entities.len()).unwrap_or(0);
                if count > 0 {
                    ui.label(format!("{count} entities"));
                }

                if let Some(err) = &self.error_msg {
                    ui.separator();
                    ui.colored_label(egui::Color32::from_rgb(255, 80, 80), err);
                }

                if self.loading {
                    ui.separator();
                    ui.weak("Loading...");
                } else if self.drawing.is_none() && self.error_msg.is_none() {
                    ui.separator();
                    ui.weak("Open a DXF file to get started");
                }
            });
        });

        // Entity info panel (bottom, when something is selected)
        if self.viewport.as_ref().and_then(|v| v.selected).is_some() {
            self.show_entity_panel(ctx);
        }

        // Layer panel (left side)
        if self.show_layers && self.drawing.is_some() {
            self.show_layer_panel(ctx);
        }

        // Central viewport
        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(egui::Color32::from_gray(25)))
            .show(ctx, |ui| {
                if let Some(viewport) = &mut self.viewport {
                    viewport.show(ui, self.drawing.as_ref());
                }
                if self.loading {
                    ui.painter().text(
                        ui.clip_rect().center(),
                        egui::Align2::CENTER_CENTER,
                        "Loading...",
                        egui::FontId::proportional(28.0),
                        egui::Color32::from_gray(200),
                    );
                }
            });
    }
}

// ── Recent files helpers ────────────────────────────────────────────────────

fn recent_files_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cad_viewer_recent.json")
}

fn load_recent(path: &Path) -> Vec<PathBuf> {
    let content = std::fs::read_to_string(path).unwrap_or_default();
    serde_json::from_str::<Vec<String>>(&content)
        .unwrap_or_default()
        .into_iter()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .collect()
}

fn save_recent(path: &Path, recent: &[PathBuf]) {
    let strings: Vec<String> = recent
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    if let Ok(json) = serde_json::to_string(&strings) {
        let _ = std::fs::write(path, json);
    }
}

// ── Entity info helpers ─────────────────────────────────────────────────────

fn entity_type_name(e: &CadEntity) -> &'static str {
    match e {
        CadEntity::Line(_) => "LINE",
        CadEntity::Circle(_) => "CIRCLE",
        CadEntity::Arc(_) => "ARC",
        CadEntity::Polyline(_) => "POLYLINE",
        CadEntity::Ellipse(_) => "ELLIPSE",
        CadEntity::Spline(_) => "SPLINE",
        CadEntity::Text(_) => "TEXT",
    }
}

fn entity_details(e: &CadEntity, ui: &mut egui::Ui) {
    match e {
        CadEntity::Line(l) => {
            let dx = l.end[0] - l.start[0];
            let dy = l.end[1] - l.start[1];
            let length = (dx * dx + dy * dy).sqrt();
            ui.label(format!("Start: ({:.3}, {:.3})", l.start[0], l.start[1]));
            ui.label(format!("End: ({:.3}, {:.3})", l.end[0], l.end[1]));
            ui.label(format!("Length: {:.3}", length));
        }
        CadEntity::Circle(c) => {
            ui.label(format!("Center: ({:.3}, {:.3})", c.center[0], c.center[1]));
            ui.label(format!("Radius: {:.3}", c.radius));
            ui.label(format!("Circumference: {:.3}", c.radius * 2.0 * std::f64::consts::PI));
        }
        CadEntity::Arc(a) => {
            let start_deg = a.start_angle.to_degrees();
            let end_deg = a.end_angle.to_degrees();
            ui.label(format!("Center: ({:.3}, {:.3})", a.center[0], a.center[1]));
            ui.label(format!("Radius: {:.3}", a.radius));
            ui.label(format!("Angles: {:.1}° → {:.1}°", start_deg, end_deg));
        }
        CadEntity::Polyline(p) => {
            ui.label(format!("Vertices: {}", p.points.len()));
            ui.label(if p.closed { "Closed" } else { "Open" });
        }
        CadEntity::Ellipse(e) => {
            let major = (e.major_axis[0].powi(2) + e.major_axis[1].powi(2)).sqrt();
            ui.label(format!("Center: ({:.3}, {:.3})", e.center[0], e.center[1]));
            ui.label(format!("Semi-major: {:.3}", major));
            ui.label(format!("Semi-minor: {:.3}", major * e.minor_ratio));
        }
        CadEntity::Spline(s) => {
            ui.label(format!("Degree: {}", s.degree));
            ui.label(format!("Control points: {}", s.control_points.len()));
        }
        CadEntity::Text(t) => {
            ui.label(format!("Position: ({:.3}, {:.3})", t.position[0], t.position[1]));
            ui.label(format!("Height: {:.3}", t.height));
            ui.label(format!("Text: \"{}\"", t.text));
        }
    }
}
