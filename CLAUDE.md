# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
cargo run                                    # Launch the viewer
cargo build                                  # Debug build (fast type-check)
cargo build --release                        # Optimized build (opt-level 3 + LTO)
cargo test                                   # Run all tests
cargo test <test_name>                       # Run a single test by name (substring match)
cargo fmt                                    # Format code
cargo clippy --all-targets --all-features    # Lint
```

## Architecture

This is a macOS-only CAD file viewer (DXF/DWG/GDSII) built with egui + wgpu. The three concerns — parsing, rendering, UI — are strictly separated into modules.

### Data flow

1. **File loading** (`app.rs` → `cad/dxf_loader.rs` or `cad/gds_loader.rs`): A background thread parses the file and emits `LoadEvent` variants (`Started`, `Chunk`, `Progress`, `Finished`, `Failed`) over an `mpsc` channel. `app.rs::handle_load_event()` accumulates them into a `Drawing` incrementally so the UI stays responsive.

2. **Tessellation** (`ui/viewport.rs` → `render/tessellator.rs`): Triggered when the camera moves outside the cached region, when zoom crosses a 2×/0.5× threshold (LOD change), or when f32 precision drift from large world-space offsets exceeds ~10% pixel error. `Drawing::query_render_items()` applies spatial indexing and LOD, then `tessellate_render_items()` converts entities to line/fill vertex lists. Results are cached in `Viewport::cached_line` / `cached_fill`.

3. **GPU render** (`render/pipeline.rs`): A `CadPaintCallback` uploads cached vertices to wgpu buffers and fires two render passes — `TriangleList` fills first (back), then `LineList` strokes on top. Camera projection is sent as a uniform.

### Key architectural decisions

- **Camera-relative tessellation**: Vertices are stored relative to `tess_origin` (world offset subtracted before converting to `f32`) to avoid precision loss for drawings far from the origin.
- **Spatial index + LOD**: `model.rs` maintains a spatial grid; entities below a per-zoom-level pixel threshold are dropped before tessellation.
- **DWG support via external tool**: DWG files are converted to DXF on-the-fly using `dwg2dxf` (from libredwg). The binary is searched at `dwg2dxf`, `/usr/local/bin/dwg2dxf`, and `/opt/homebrew/bin/dwg2dxf`. The temp DXF is deleted after loading.
- **Tessellation entry point**: `tessellate_render_items()` in `tessellator.rs` is the live render path — it takes the pre-filtered `RenderItem` slice from `Drawing::query_render_items()`. The `tessellate_all()` function is dead code (`#[allow(dead_code)]`) and is not called in the render loop.
- **Tessellation viewport margin**: `tess_viewport` covers camera view plus a 100% margin on each side so small pans don't retrigger tessellation on every frame.
- **Recent files**: Persisted to `~/.cad_viewer_recent.json` (up to 10 entries). Files that no longer exist are filtered out on load.
- **File loading**: Supports drag-and-drop onto the viewport in addition to File > Open and Cmd+O.

### Module responsibilities

| Module | Responsibility |
|--------|---------------|
| `app.rs` | App state, file dialogs, menu bar, layer/entity panels, load orchestration |
| `cad/model.rs` | `CadEntity` types, `Drawing` container, spatial index, bounding boxes, LOD cache |
| `cad/dxf_loader.rs` | Streaming DXF parser; DWG → DXF conversion via `dwg2dxf` |
| `cad/gds_loader.rs` | Streaming GDSII parser; cell flattening |
| `cad/selection.rs` | Entity picking by screen-distance per entity type |
| `cad/snap.rs` | Measurement snap (endpoints, arc centers, intersections) |
| `render/pipeline.rs` | wgpu device/pipeline setup, WGSL shaders, vertex buffer management, camera uniform |
| `render/camera.rs` | 2D orthographic camera: zoom/pan/fit-to-bounds, screen ↔ world conversions |
| `render/tessellator.rs` | Entity → GPU vertices: arc/circle sampling, polygon triangulation via `earcutr` |
| `ui/viewport.rs` | egui `Viewport` widget: interaction handling, tessellation triggers, measure-tool overlay |
| `util/color.rs` | AutoCAD Color Index (ACI 0–255) → RGBA lookup table |

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `F` | Fit drawing to viewport |
| `M` | Toggle measure tool |
| `Esc` | Deselect entity / cancel measure |
| `Cmd+O` | Open file dialog |

In measure mode: first click sets point A, second click sets point B, then distance is shown. Snap indicators (cyan) show endpoints (square), centers (circle+crosshair), and midpoints (triangle).

## Code style

Rust 2021 edition. `rustfmt` enforced. Naming: `snake_case` functions/variables, `CamelCase` types/enums, `SCREAMING_SNAKE_CASE` constants. Keep parsers in `cad/`, GPU code in `render/`, egui glue in `ui/`.

## macOS bundle config

App bundle metadata (name, identifier, icon path) lives under `[package.metadata.bundle]` in `Cargo.toml`.
