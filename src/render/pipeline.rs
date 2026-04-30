#[allow(unused_imports)]
use eframe::egui_wgpu;
use eframe::wgpu;
use eframe::wgpu::util::DeviceExt;
use std::collections::HashMap;
use std::sync::Arc;

use crate::render::tessellator::{GpuTransform, RenderOutput, Vertex};

/// Uniform buffer layout (must match shader)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

// ── Per-cell GPU geometry ─────────────────────────────────────────────────────

/// GPU buffers for one GDS cell's geometry.
/// Geometry buffers (line/fill) are immutable after upload.
/// The instance buffer is updated each tessellation frame.
pub struct CellGpuGeom {
    pub line_buffer: Option<wgpu::Buffer>,
    pub line_count: u32,
    pub fill_buffer: Option<wgpu::Buffer>,
    pub fill_count: u32,
    /// Per-instance transform buffer (GpuTransform × instance_capacity).
    pub instance_buffer: Option<wgpu::Buffer>,
    /// Number of instances to draw this frame.
    pub instance_count: u32,
    instance_capacity: u32,
}

impl CellGpuGeom {
    fn new(device: &wgpu::Device, line_verts: &[Vertex], fill_verts: &[Vertex]) -> Self {
        let line_buffer = if !line_verts.is_empty() {
            Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cell_line"),
                contents: bytemuck::cast_slice(line_verts),
                usage: wgpu::BufferUsages::VERTEX,
            }))
        } else {
            None
        };
        let fill_buffer = if !fill_verts.is_empty() {
            Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cell_fill"),
                contents: bytemuck::cast_slice(fill_verts),
                usage: wgpu::BufferUsages::VERTEX,
            }))
        } else {
            None
        };
        Self {
            line_count: line_verts.len() as u32,
            fill_count: fill_verts.len() as u32,
            line_buffer,
            fill_buffer,
            instance_buffer: None,
            instance_count: 0,
            instance_capacity: 0,
        }
    }

    fn upload_instances(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        transforms: &[GpuTransform],
    ) {
        let count = transforms.len() as u32;
        self.instance_count = count;
        if count == 0 {
            return;
        }
        if count > self.instance_capacity {
            let new_cap = count.next_power_of_two();
            self.instance_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cell_instance"),
                size: new_cap as u64 * std::mem::size_of::<GpuTransform>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.instance_capacity = new_cap;
        }
        if let Some(buf) = &self.instance_buffer {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(transforms));
        }
    }
}

// ── CadRenderResources ────────────────────────────────────────────────────────

/// All persistent wgpu resources for CAD rendering.
/// Stored in egui_wgpu::CallbackResources so it lives as long as the renderer.
pub struct CadRenderResources {
    // ── Non-instanced (misc / DXF) pipelines ─────────────────────────────────
    /// LineList pipeline (DXF lines, arcs, LOD boxes, etc.)
    pub pipeline: wgpu::RenderPipeline,
    /// TriangleList pipeline (DXF filled polygons)
    pub fill_pipeline: wgpu::RenderPipeline,

    // ── Instanced (GDS cell) pipelines ────────────────────────────────────────
    /// LineList instanced pipeline (GDS cell line geometry × instance transforms)
    pub inst_pipeline: wgpu::RenderPipeline,
    /// TriangleList instanced pipeline (GDS cell fill geometry × instance transforms)
    pub inst_fill_pipeline: wgpu::RenderPipeline,

    // ── Shared ───────────────────────────────────────────────────────────────
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,

    // ── Misc vertex buffers (non-instanced draw) ──────────────────────────────
    pub vertex_buffer: wgpu::Buffer,
    pub vertex_count: u32,
    vertex_capacity: u32,
    pub fill_vertex_buffer: wgpu::Buffer,
    pub fill_vertex_count: u32,
    fill_vertex_capacity: u32,

    // ── GDS cell geometry + instance buffers ─────────────────────────────────
    pub cell_geom: HashMap<usize, CellGpuGeom>,

    /// fill_polygons value from the last uploaded RenderOutput.
    /// Used to detect fill-mode changes that require re-uploading cell geometry.
    last_fill_polygons: bool,
}

impl CadRenderResources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cad_shader"),
            source: wgpu::ShaderSource::Wgsl(CAD_SHADER.into()),
        });

        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cad_uniform_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cad_uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cad_uniform_bg"),
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cad_pipeline_layout"),
            bind_group_layouts: &[&uniform_bgl],
            push_constant_ranges: &[],
        });

        // ── Vertex / instance buffer layouts ─────────────────────────────────

        let vertex_attrs = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 8,
                shader_location: 1,
            },
        ];
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vertex_attrs,
        };

        // GpuTransform: col0 @location(2), col1 @location(3), translation @location(4)
        let instance_attrs = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 8,
                shader_location: 3,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 16,
                shader_location: 4,
            },
        ];
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuTransform>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &instance_attrs,
        };

        // ── Non-instanced pipelines ───────────────────────────────────────────

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cad_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: std::slice::from_ref(&vertex_layout),
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let fill_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cad_fill_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_layout.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ── Instanced pipelines ───────────────────────────────────────────────

        let inst_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cad_inst_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_inst",
                buffers: &[vertex_layout.clone(), instance_layout.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let inst_fill_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cad_inst_fill_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_inst",
                buffers: &[vertex_layout.clone(), instance_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let initial_capacity = 1024u32;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cad_vertex_buffer"),
            size: initial_capacity as u64 * std::mem::size_of::<Vertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let fill_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cad_fill_vertex_buffer"),
            size: initial_capacity as u64 * std::mem::size_of::<Vertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            fill_pipeline,
            inst_pipeline,
            inst_fill_pipeline,
            uniform_buffer,
            uniform_bind_group,
            vertex_buffer,
            vertex_count: 0,
            vertex_capacity: initial_capacity,
            fill_vertex_buffer,
            fill_vertex_count: 0,
            fill_vertex_capacity: initial_capacity,
            cell_geom: HashMap::new(),
            last_fill_polygons: true,
        }
    }

    fn max_buffer_vertices(device: &wgpu::Device) -> u64 {
        let cap = device.limits().max_buffer_size.min(512 * 1024 * 1024);
        cap / std::mem::size_of::<Vertex>() as u64
    }

    pub fn upload_vertices(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[Vertex],
    ) {
        let max = Self::max_buffer_vertices(device);
        let count = (vertices.len() as u64).min(max);
        self.vertex_count = count as u32;
        if count == 0 {
            return;
        }
        let needed_cap = count.next_power_of_two().min(max);
        if needed_cap > self.vertex_capacity as u64 {
            self.vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cad_vertex_buffer"),
                size: needed_cap * std::mem::size_of::<Vertex>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.vertex_capacity = needed_cap as u32;
        }
        queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(&vertices[..count as usize]),
        );
    }

    pub fn upload_fill_vertices(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[Vertex],
    ) {
        let max = Self::max_buffer_vertices(device);
        let count = (vertices.len() as u64).min(max);
        self.fill_vertex_count = count as u32;
        if count == 0 {
            return;
        }
        let needed_cap = count.next_power_of_two().min(max);
        if needed_cap > self.fill_vertex_capacity as u64 {
            self.fill_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cad_fill_vertex_buffer"),
                size: needed_cap * std::mem::size_of::<Vertex>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.fill_vertex_capacity = needed_cap as u32;
        }
        queue.write_buffer(
            &self.fill_vertex_buffer,
            0,
            bytemuck::cast_slice(&vertices[..count as usize]),
        );
    }

    pub fn upload_camera(&self, queue: &wgpu::Queue, uniform: &CameraUniform) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[*uniform]));
    }
}

// ── CadPaintCallback ──────────────────────────────────────────────────────────

/// The egui_wgpu paint callback.
pub struct CadPaintCallback {
    /// New tessellation output to upload; None if only the camera changed.
    pub render_output: Option<Arc<RenderOutput>>,
    pub camera_uniform: CameraUniform,
    /// When true, upload render_output to GPU buffers.
    pub needs_rebuild: bool,
    /// When true, discard all cell GPU geometry (new file loaded).
    pub clear_cells: bool,
}

impl eframe::egui_wgpu::CallbackTrait for CadPaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &eframe::egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut eframe::egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(res) = resources.get_mut::<CadRenderResources>() {
            if self.clear_cells {
                res.cell_geom.clear();
            }

            if self.needs_rebuild {
                if let Some(output) = &self.render_output {
                    // Upload misc (DXF / LOD-box) geometry.
                    res.upload_vertices(device, queue, &output.misc_lines);
                    res.upload_fill_vertices(device, queue, &output.misc_fills);

                    // If fill mode changed, all cell geometry must be re-uploaded.
                    if output.fill_polygons != res.last_fill_polygons {
                        res.cell_geom.clear();
                        res.last_fill_polygons = output.fill_polygons;
                    }

                    // Upload cell geometry once per cell (immutable after first upload).
                    for (&cell_idx, entry) in &output.cell_geometries {
                        if !res.cell_geom.contains_key(&cell_idx) {
                            let line_verts: Vec<Vertex> = entry
                                .lines
                                .iter()
                                .map(|lv| {
                                    Vertex::new(lv.pos[0] as f32, lv.pos[1] as f32, lv.color)
                                })
                                .collect();
                            let fill_verts: Vec<Vertex> = entry
                                .fills
                                .iter()
                                .map(|lv| {
                                    Vertex::new(lv.pos[0] as f32, lv.pos[1] as f32, lv.color)
                                })
                                .collect();
                            res.cell_geom
                                .insert(cell_idx, CellGpuGeom::new(device, &line_verts, &fill_verts));
                        }
                    }

                    // Upload per-frame instance transforms.
                    for (&cell_idx, transforms) in &output.instances {
                        if let Some(geom) = res.cell_geom.get_mut(&cell_idx) {
                            geom.upload_instances(device, queue, transforms);
                        }
                    }

                    // Zero out instance counts for cells absent from this frame
                    // (they're off-screen or LOD-culled).
                    for (&cell_idx, geom) in &mut res.cell_geom {
                        if !output.instances.contains_key(&cell_idx) {
                            geom.instance_count = 0;
                        }
                    }
                } else {
                    // No drawing data — clear everything.
                    res.upload_vertices(device, queue, &[]);
                    res.upload_fill_vertices(device, queue, &[]);
                    for geom in res.cell_geom.values_mut() {
                        geom.instance_count = 0;
                    }
                }
            }

            res.upload_camera(queue, &self.camera_uniform);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &eframe::egui_wgpu::CallbackResources,
    ) {
        let Some(res) = resources.get::<CadRenderResources>() else {
            return;
        };
        render_pass.set_bind_group(0, &res.uniform_bind_group, &[]);

        // ── 1. Instanced fills (back) ─────────────────────────────────────────
        render_pass.set_pipeline(&res.inst_fill_pipeline);
        for geom in res.cell_geom.values() {
            if geom.fill_count > 0 && geom.instance_count > 0 {
                if let (Some(fb), Some(ib)) = (&geom.fill_buffer, &geom.instance_buffer) {
                    let inst_bytes =
                        geom.instance_count as u64 * std::mem::size_of::<GpuTransform>() as u64;
                    render_pass.set_vertex_buffer(0, fb.slice(..));
                    render_pass.set_vertex_buffer(1, ib.slice(..inst_bytes));
                    render_pass.draw(0..geom.fill_count, 0..geom.instance_count);
                }
            }
        }

        // ── 2. Misc fills ─────────────────────────────────────────────────────
        if res.fill_vertex_count > 0 {
            render_pass.set_pipeline(&res.fill_pipeline);
            render_pass.set_vertex_buffer(0, res.fill_vertex_buffer.slice(..));
            render_pass.draw(0..res.fill_vertex_count, 0..1);
        }

        // ── 3. Instanced lines ────────────────────────────────────────────────
        render_pass.set_pipeline(&res.inst_pipeline);
        for geom in res.cell_geom.values() {
            if geom.line_count > 0 && geom.instance_count > 0 {
                if let (Some(lb), Some(ib)) = (&geom.line_buffer, &geom.instance_buffer) {
                    let inst_bytes =
                        geom.instance_count as u64 * std::mem::size_of::<GpuTransform>() as u64;
                    render_pass.set_vertex_buffer(0, lb.slice(..));
                    render_pass.set_vertex_buffer(1, ib.slice(..inst_bytes));
                    render_pass.draw(0..geom.line_count, 0..geom.instance_count);
                }
            }
        }

        // ── 4. Misc lines (DXF, LOD boxes) ───────────────────────────────────
        if res.vertex_count > 0 {
            render_pass.set_pipeline(&res.pipeline);
            render_pass.set_vertex_buffer(0, res.vertex_buffer.slice(..));
            render_pass.draw(0..res.vertex_count, 0..1);
        }
    }
}

const CAD_SHADER: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct InstanceInput {
    @location(2) col0: vec2<f32>,
    @location(3) col1: vec2<f32>,
    @location(4) translation: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

/// Non-instanced pass: position is already in camera-relative world coords.
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 0.0, 1.0);
    out.color = in.color;
    return out;
}

/// Instanced pass: apply 2×2 affine instance transform then camera projection.
/// Transform: world_pos = mat2x2(col0, col1) * cell_local_pos + translation
@vertex
fn vs_inst(in: VertexInput, inst: InstanceInput) -> VertexOutput {
    let m = mat2x2<f32>(inst.col0, inst.col1);
    let world = m * in.position + inst.translation;
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;
