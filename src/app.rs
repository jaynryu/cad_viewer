use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use crate::cad::dxf_loader::{load_dxf, load_dxf_streaming};
use crate::cad::gds_loader::load_gds_streaming;
use crate::cad::model::{CadEntity, Drawing, LoadEvent};
use crate::ui::viewport::Viewport;

const MAX_RECENT: usize = 10;

pub struct CadViewerApp {
    viewport: Option<Viewport>,
    drawing: Option<Drawing>,
    error_msg: Option<String>,
    file_rx: Option<mpsc::Receiver<PathBuf>>,
    drawing_rx: Option<mpsc::Receiver<LoadEvent>>,
    loading: bool,
    loading_format: Option<&'static str>,
    loaded_entities: usize,
    show_layers: bool,
    recent_files: Vec<PathBuf>,
}

impl CadViewerApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        setup_fonts(&cc.egui_ctx);
        apply_macos_style(&cc.egui_ctx);

        let viewport = cc.wgpu_render_state.as_ref().map(Viewport::new);

        let recent_files = load_recent(&recent_files_path());

        Self {
            viewport,
            drawing: None,
            error_msg: None,
            file_rx: None,
            drawing_rx: None,
            loading: false,
            loading_format: None,
            loaded_entities: 0,
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
                .add_filter("CAD Files", &["dxf", "dwg", "gds", "gds2"])
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
                self.start_loading();
                self.push_recent(path.clone());
                let (tx, rx) = mpsc::channel();
                self.drawing_rx = Some(rx);
                let ctx = ctx.clone();
                std::thread::spawn(move || {
                    log::info!("Loading DXF: {:?}", path);
                    load_dxf_streaming(&path, &tx);
                    ctx.request_repaint();
                });
            }
            "dwg" => {
                self.start_loading();
                self.push_recent(path.clone());
                let (tx, rx) = mpsc::channel();
                self.drawing_rx = Some(rx);
                let ctx = ctx.clone();
                std::thread::spawn(move || {
                    log::info!("Loading DWG: {:?}", path);
                    load_dwg_via_converter_streaming(&path, &tx);
                    ctx.request_repaint();
                });
            }
            "gds" | "gds2" => {
                self.start_loading();
                self.push_recent(path.clone());
                let (tx, rx) = mpsc::channel();
                self.drawing_rx = Some(rx);
                let ctx = ctx.clone();
                std::thread::spawn(move || {
                    log::info!("Loading GDS: {:?}", path);
                    load_gds_streaming(&path, &tx);
                    ctx.request_repaint();
                });
            }
            _ => self.error_msg = Some(format!("Unsupported file type: .{ext}")),
        }
    }

    fn start_loading(&mut self) {
        self.loading = true;
        self.loading_format = None;
        self.loaded_entities = 0;
        self.error_msg = None;
        self.drawing = None;
        if let Some(vp) = &mut self.viewport {
            vp.selected = None;
            vp.hidden_layers.clear();
            vp.mark_dirty();
        }
    }

    fn push_recent(&mut self, path: PathBuf) {
        self.recent_files.retain(|p| p != &path);
        self.recent_files.insert(0, path);
        self.recent_files.truncate(MAX_RECENT);
        save_recent(&recent_files_path(), &self.recent_files);
    }

    fn handle_load_event(&mut self, event: LoadEvent) -> bool {
        match event {
            LoadEvent::Started { format } => {
                self.loading = true;
                self.loading_format = Some(format);
                self.loaded_entities = 0;
                // Discard primitive cache from any previous file.
                if let Some(vp) = &mut self.viewport {
                    vp.reset_cell_cache();
                }
                eprintln!("Loading {format}...");
                false
            }
            LoadEvent::GdsCells(cells) => {
                let drawing = self.drawing.get_or_insert_with(Drawing::new);
                drawing.gds_cells = std::sync::Arc::new(cells);
                false
            }
            LoadEvent::Chunk(chunk) => {
                let needs_initial_fit = self
                    .drawing
                    .as_ref()
                    .and_then(|d| d.bounds.as_ref())
                    .is_none();
                let drawing = self.drawing.get_or_insert_with(Drawing::new);
                drawing.append_chunk_deferred(chunk);
                self.loaded_entities = drawing.entities.len();
                if let Some(vp) = &mut self.viewport {
                    if needs_initial_fit {
                        if let Some(bounds) = &drawing.bounds {
                            vp.request_fit(bounds.clone());
                        }
                    }
                }
                false
            }
            LoadEvent::Progress { loaded_entities } => {
                self.loaded_entities = loaded_entities;
                self.print_load_progress();
                false
            }
            LoadEvent::Finished => {
                self.loading = false;
                self.loading_format = None;
                let mut final_count = self.loaded_entities;
                if let Some(drawing) = &mut self.drawing {
                    drawing.finalize_indexes();
                    final_count = drawing.entities.len();
                    log::info!("Applied drawing: {} entities", drawing.entities.len());
                    if let Some(vp) = &mut self.viewport {
                        if let Some(bounds) = &drawing.bounds {
                            vp.request_fit(bounds.clone());
                        }
                        vp.mark_dirty();
                    }
                }
                eprintln!("\rLoaded {} entities.{}", final_count, " ".repeat(24));
                true
            }
            LoadEvent::Failed(e) => {
                self.loading = false;
                self.loading_format = None;
                eprintln!("\nLoad failed: {e}");
                log::error!("Load error: {e}");
                self.error_msg = Some(e);
                true
            }
        }
    }

    fn print_load_progress(&self) {
        let format = self.loading_format.unwrap_or("CAD");
        eprint!("\rLoading {format}... {} entities", self.loaded_entities);
        let _ = io::stderr().flush();
    }

    fn show_layer_panel(&mut self, ctx: &egui::Context) {
        let Some(drawing) = &self.drawing else { return };
        let Some(viewport) = &mut self.viewport else {
            return;
        };

        egui::SidePanel::left("layer_panel")
            .min_width(160.0)
            .max_width(300.0)
            .show(ctx, |ui| {
                ui.heading("Layers");
                ui.separator();

                ui.horizontal(|ui| {
                    if ui.small_button("All On").clicked() {
                        viewport.hidden_layers.clear();
                        viewport.mark_layers_changed();
                    }
                    if ui.small_button("All Off").clicked() {
                        for layer in &drawing.layers {
                            viewport.hidden_layers.insert(layer.name.clone());
                        }
                        viewport.mark_layers_changed();
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
                            let (rect, _) = ui
                                .allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
                            ui.painter().rect_filled(rect, 2.0, color);

                            let mut visible = is_visible;
                            if ui.checkbox(&mut visible, &layer.name).changed() {
                                if visible {
                                    viewport.hidden_layers.remove(&layer.name);
                                } else {
                                    viewport.hidden_layers.insert(layer.name.clone());
                                }
                                viewport.mark_layers_changed();
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
        let Some(entity) = drawing.entities.get(idx) else {
            return;
        };

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
        if let Some(rx) = self.drawing_rx.take() {
            let mut keep_rx = true;
            loop {
                match rx.try_recv() {
                    Ok(event) => {
                        if self.handle_load_event(event) {
                            keep_rx = false;
                            break;
                        }
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        self.loading = false;
                        self.loading_format = None;
                        self.error_msg = Some("Load failed (worker thread crashed)".to_string());
                        keep_rx = false;
                        break;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                }
            }
            if keep_rx {
                self.drawing_rx = Some(rx);
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
                    // Polygon fill toggle (GDS)
                    if let Some(vp) = &mut self.viewport {
                        let prev = vp.fill_polygons;
                        if ui
                            .checkbox(&mut vp.fill_polygons, "Fill Polygons")
                            .clicked()
                        {
                            if vp.fill_polygons != prev {
                                vp.mark_dirty();
                            }
                            ui.close_menu();
                        }
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
                    let measuring = self
                        .viewport
                        .as_ref()
                        .map(|v| v.measure_mode)
                        .unwrap_or(false);
                    let label = if measuring {
                        "Measure (ON)   M"
                    } else {
                        "Measure   M"
                    };
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
                    let format = self.loading_format.unwrap_or("CAD");
                    ui.weak(format!(
                        "Loading {format}... {} entities",
                        self.loaded_entities
                    ));
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
                    let format = self.loading_format.unwrap_or("CAD");
                    ui.painter().text(
                        ui.clip_rect().center(),
                        egui::Align2::CENTER_CENTER,
                        format!("Loading {format}...\n{} entities", self.loaded_entities),
                        egui::FontId::proportional(28.0),
                        egui::Color32::from_gray(200),
                    );
                }
            });
    }
}

// ── DWG → DXF conversion ────────────────────────────────────────────────────

/// Convert a DWG file to a temporary DXF using dwg2dxf, then load it.
#[allow(dead_code)]
fn load_dwg_via_converter(dwg_path: &std::path::Path) -> Result<Drawing, String> {
    // Locate dwg2dxf (common install paths)
    let converter = [
        "dwg2dxf",
        "/usr/local/bin/dwg2dxf",
        "/opt/homebrew/bin/dwg2dxf",
    ]
    .iter()
    .find(|&&p| {
        std::process::Command::new(p)
            .arg("--version")
            .output()
            .is_ok()
    })
    .copied()
    .ok_or_else(|| {
        "dwg2dxf not found. Install libredwg: https://www.gnu.org/software/libredwg/".to_string()
    })?;

    // Write to a temp file
    let tmp_dxf = std::env::temp_dir().join(format!(
        "cad_viewer_{}.dxf",
        dwg_path.file_stem().unwrap_or_default().to_string_lossy()
    ));

    let output = std::process::Command::new(converter)
        .args([
            "-y", // overwrite
            "--as",
            "r2000", // safe DXF version
            "-o",
            tmp_dxf.to_str().unwrap_or(""),
            dwg_path.to_str().unwrap_or(""),
        ])
        .output()
        .map_err(|e| format!("Failed to run dwg2dxf: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("dwg2dxf error: {stderr}"));
    }

    if !tmp_dxf.exists() {
        return Err("dwg2dxf produced no output file".to_string());
    }

    let result = load_dxf(&tmp_dxf);
    let _ = std::fs::remove_file(&tmp_dxf); // clean up temp file
    result
}

fn load_dwg_via_converter_streaming(dwg_path: &std::path::Path, tx: &mpsc::Sender<LoadEvent>) {
    let _ = tx.send(LoadEvent::Started { format: "DWG" });
    match convert_dwg_to_temp_dxf(dwg_path) {
        Ok(tmp_dxf) => {
            load_dxf_streaming(&tmp_dxf, tx);
            let _ = std::fs::remove_file(&tmp_dxf);
        }
        Err(e) => {
            let _ = tx.send(LoadEvent::Failed(e));
        }
    }
}

fn convert_dwg_to_temp_dxf(dwg_path: &std::path::Path) -> Result<PathBuf, String> {
    let converter = [
        "dwg2dxf",
        "/usr/local/bin/dwg2dxf",
        "/opt/homebrew/bin/dwg2dxf",
    ]
    .iter()
    .find(|&&p| {
        std::process::Command::new(p)
            .arg("--version")
            .output()
            .is_ok()
    })
    .copied()
    .ok_or_else(|| {
        "dwg2dxf not found. Install libredwg: https://www.gnu.org/software/libredwg/".to_string()
    })?;

    let tmp_dxf = std::env::temp_dir().join(format!(
        "cad_viewer_{}.dxf",
        dwg_path.file_stem().unwrap_or_default().to_string_lossy()
    ));

    let output = std::process::Command::new(converter)
        .args([
            "-y",
            "--as",
            "r2000",
            "-o",
            tmp_dxf.to_str().unwrap_or(""),
            dwg_path.to_str().unwrap_or(""),
        ])
        .output()
        .map_err(|e| format!("Failed to run dwg2dxf: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("dwg2dxf error: {stderr}"));
    }

    if !tmp_dxf.exists() {
        return Err("dwg2dxf produced no output file".to_string());
    }

    Ok(tmp_dxf)
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
        CadEntity::Polygon(_) => "POLYGON",
        CadEntity::GdsInstance(_) => "GDS SREF",
        CadEntity::GdsArrayInstance(_) => "GDS AREF",
        CadEntity::DxfInsert(_) => "INSERT",
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
            ui.label(format!(
                "Circumference: {:.3}",
                c.radius * 2.0 * std::f64::consts::PI
            ));
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
            ui.label(format!(
                "Position: ({:.3}, {:.3})",
                t.position[0], t.position[1]
            ));
            ui.label(format!("Height: {:.3}", t.height));
            ui.label(format!("Text: \"{}\"", t.text));
        }
        CadEntity::Polygon(p) => {
            ui.label(format!("Vertices: {}", p.points.len()));
        }
        CadEntity::GdsInstance(inst) => {
            ui.label(format!("Cell: {}", inst.layer));
            ui.label(format!("Offset: ({:.3}, {:.3})", inst.offset[0], inst.offset[1]));
            if inst.angle_deg != 0.0 {
                ui.label(format!("Angle: {:.2}°", inst.angle_deg));
            }
            if (inst.mag - 1.0).abs() > 1e-9 {
                ui.label(format!("Mag: {:.4}×", inst.mag));
            }
            if inst.reflected {
                ui.label("Reflected");
            }
        }
        CadEntity::GdsArrayInstance(arr) => {
            ui.label(format!("Cell: {} ({}×{})", arr.layer, arr.cols, arr.rows));
            ui.label(format!("Origin: ({:.3}, {:.3})", arr.origin[0], arr.origin[1]));
            if arr.angle_deg != 0.0 {
                ui.label(format!("Angle: {:.2}°", arr.angle_deg));
            }
        }
        CadEntity::DxfInsert(ins) => {
            ui.label(format!("Offset: ({:.3}, {:.3})", ins.offset[0], ins.offset[1]));
            if ins.angle_deg != 0.0 {
                ui.label(format!("Angle: {:.2}°", ins.angle_deg));
            }
            if (ins.x_scale - 1.0).abs() > 1e-9 || (ins.y_scale - 1.0).abs() > 1e-9 {
                ui.label(format!("Scale: {:.4}×, {:.4}×", ins.x_scale, ins.y_scale));
            }
            if ins.cols > 1 || ins.rows > 1 {
                ui.label(format!("Array: {}×{}", ins.cols, ins.rows));
            }
        }
    }
}

// ── macOS style ─────────────────────────────────────────────────────────────

fn apply_macos_style(ctx: &egui::Context) {
    use egui::{
        style::{WidgetVisuals, Widgets},
        Color32, FontId, Margin, Rounding, Shadow, Stroke, Vec2, Visuals,
    };

    // ── macOS dark mode system colors ──
    let bg = Color32::from_rgb(28, 28, 30); // systemBackground
    let bg2 = Color32::from_rgb(44, 44, 46); // secondarySystemBackground
    let bg3 = Color32::from_rgb(58, 58, 60); // tertiarySystemBackground
    let bg4 = Color32::from_rgb(72, 72, 74); // hover
    let label = Color32::from_rgb(235, 235, 245); // label
    let label2 = Color32::from_rgba_premultiplied(235, 235, 245, 153); // secondaryLabel (60%)
    let separator = Color32::from_rgba_premultiplied(84, 84, 88, 165);
    let accent = Color32::from_rgb(10, 132, 255); // systemBlue dark

    let mut v = Visuals::dark();

    // ── backgrounds ──
    v.panel_fill = bg;
    v.window_fill = bg2;
    v.extreme_bg_color = Color32::from_rgb(18, 18, 20);
    v.code_bg_color = bg3;
    v.faint_bg_color = Color32::from_rgb(36, 36, 38);

    // ── window chrome ──
    v.window_rounding = Rounding::same(12.0);
    v.window_stroke = Stroke::new(0.5, separator);
    v.window_shadow = Shadow {
        offset: Vec2::new(0.0, 4.0),
        blur: 24.0,
        spread: 0.0,
        color: Color32::from_black_alpha(120),
    };

    // ── popups / menus ──
    v.menu_rounding = Rounding::same(8.0);
    v.popup_shadow = Shadow {
        offset: Vec2::new(0.0, 2.0),
        blur: 12.0,
        spread: 0.0,
        color: Color32::from_black_alpha(100),
    };

    // ── selection ──
    v.selection.bg_fill = Color32::from_rgba_premultiplied(10, 132, 255, 55);
    v.selection.stroke = Stroke::new(1.0, accent);
    v.hyperlink_color = accent;

    // ── indent ──
    v.indent_has_left_vline = false;

    // ── widgets ──
    let r6 = Rounding::same(6.0);
    let r8 = Rounding::same(8.0);

    v.widgets = Widgets {
        noninteractive: WidgetVisuals {
            bg_fill: bg2,
            weak_bg_fill: bg,
            bg_stroke: Stroke::new(0.5, separator),
            fg_stroke: Stroke::new(1.0, label2),
            rounding: r6,
            expansion: 0.0,
        },
        inactive: WidgetVisuals {
            bg_fill: bg3,
            weak_bg_fill: bg2,
            bg_stroke: Stroke::NONE,
            fg_stroke: Stroke::new(1.0, label),
            rounding: r6,
            expansion: 0.0,
        },
        hovered: WidgetVisuals {
            bg_fill: bg4,
            weak_bg_fill: bg3,
            bg_stroke: Stroke::new(0.5, Color32::from_rgb(100, 100, 102)),
            fg_stroke: Stroke::new(1.5, Color32::WHITE),
            rounding: r6,
            expansion: 1.0,
        },
        active: WidgetVisuals {
            bg_fill: Color32::from_rgb(0, 112, 220),
            weak_bg_fill: Color32::from_rgb(10, 132, 255),
            bg_stroke: Stroke::NONE,
            fg_stroke: Stroke::new(1.5, Color32::WHITE),
            rounding: r6,
            expansion: 1.0,
        },
        open: WidgetVisuals {
            bg_fill: bg3,
            weak_bg_fill: bg2,
            bg_stroke: Stroke::new(0.5, separator),
            fg_stroke: Stroke::new(1.5, label),
            rounding: r8,
            expansion: 0.0,
        },
    };

    ctx.set_visuals(v);

    // ── spacing & typography ──
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(8.0, 5.0);
    style.spacing.button_padding = Vec2::new(12.0, 5.0);
    style.spacing.menu_margin = Margin::same(4.0);
    style.spacing.window_margin = Margin::same(12.0);
    style.spacing.indent = 16.0;
    // macOS-style thin floating scrollbars
    style.spacing.scroll = egui::style::ScrollStyle {
        floating: true,
        bar_width: 6.0,
        floating_width: 4.0,
        ..egui::style::ScrollStyle::floating()
    };

    // Default text size — SF Pro 13pt looks native at macOS resolution
    style.text_styles.insert(
        egui::TextStyle::Body,
        FontId::new(13.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Button,
        FontId::new(13.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Small,
        FontId::new(11.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Heading,
        FontId::new(15.0, egui::FontFamily::Proportional),
    );

    ctx.set_style(style);
}

// ── System font setup ───────────────────────────────────────────────────────

fn setup_fonts(ctx: &egui::Context) {
    // macOS system font candidates (San Francisco → Helvetica Neue → fallback)
    let candidates = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/SFNSText.ttf",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/System/Library/Fonts/SF Pro.ttf",
        "/System/Library/Fonts/SF Pro Display.ttf",
        "/System/Library/Fonts/SF Pro Text.ttf",
        "/Library/Fonts/SF-Pro-Display-Regular.otf",
        "/Library/Fonts/SF-Pro-Text-Regular.otf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
    ];

    let font_data = candidates
        .iter()
        .find_map(|path| std::fs::read(path).ok().map(|data| (*path, data)));

    if let Some((path, data)) = font_data {
        let mut fonts = egui::FontDefinitions::default();
        fonts
            .font_data
            .insert("system_font".to_owned(), egui::FontData::from_owned(data));
        // Place the system font first in the proportional family
        fonts
            .families
            .entry(egui::FontFamily::Proportional)
            .or_default()
            .insert(0, "system_font".to_owned());
        ctx.set_fonts(fonts);
        log::info!("Loaded system font: {path}");
    }
}
