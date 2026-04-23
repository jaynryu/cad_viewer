mod app;
mod cad;
mod render;
mod ui;
mod util;

fn main() -> eframe::Result<()> {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([640.0, 480.0])
            .with_title("CAD Viewer"),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "CAD Viewer",
        options,
        Box::new(|cc| Ok(Box::new(app::CadViewerApp::new(cc)))),
    )
}
