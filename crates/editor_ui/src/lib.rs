use egui::Color32;
use engine::{camera::FlyCamera, diagnostics::FrameStatistics};
use renderer::{LightingSettings, ViewportOutput};
use shaders::{SHADER_ASSETS, SlangHotReloadStatus};

pub struct EditorUiState {
    pub viewport_output: ViewportOutput,
    pub show_demo_window: bool,
}

impl Default for EditorUiState {
    fn default() -> Self {
        Self {
            viewport_output: ViewportOutput::FinalComposite,
            show_demo_window: false,
        }
    }
}

pub fn draw(
    context: &egui::Context,
    state: &mut EditorUiState,
    camera: &mut FlyCamera,
    statistics: &FrameStatistics,
    lighting: &mut LightingSettings,
    shader_error: Option<&str>,
    slang_status: &SlangHotReloadStatus,
) -> bool {
    let mut recompile_shaders = false;
    egui::Window::new("Renderer")
        .default_width(280.0)
        .max_width(360.0)
        .max_height(context.available_rect().height() - 44.0)
        .show(context, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    egui::CollapsingHeader::new("Viewport output")
                        .default_open(true)
                        .show(ui, |ui| {
                            egui::ComboBox::from_id_salt("viewport-output")
                                .selected_text(state.viewport_output.label())
                                .show_ui(ui, |ui| {
                                    for output in ViewportOutput::ALL {
                                        ui.selectable_value(
                                            &mut state.viewport_output,
                                            output,
                                            output.label(),
                                        );
                                    }
                                });
                        });
                    egui::CollapsingHeader::new("Lighting").show(ui, |ui| {
                        ui.add(
                            egui::Slider::new(&mut lighting.directional_intensity, 0.0..=8.0)
                                .text("Sun intensity"),
                        );
                        ui.add(
                            egui::Slider::new(&mut lighting.point_intensity, 0.0..=80.0)
                                .text("Point intensity"),
                        );
                        ui.add(
                            egui::Slider::new(&mut lighting.point_range, 0.5..=20.0)
                                .text("Point range"),
                        );
                        ui.add(
                            egui::Slider::new(&mut lighting.spot_intensity, 0.0..=100.0)
                                .text("Spot intensity"),
                        );
                        ui.add(
                            egui::Slider::new(&mut lighting.spot_range, 0.5..=20.0)
                                .text("Spot range"),
                        );
                        ui.add(
                            egui::Slider::new(&mut lighting.ambient_intensity, 0.0..=0.5)
                                .text("Ambient"),
                        );
                        ui.add(
                            egui::Slider::new(&mut lighting.exposure, 0.1..=3.0).text("Exposure"),
                        );
                        ui.add(
                            egui::Slider::new(&mut lighting.environment_intensity, 0.0..=2.0)
                                .text("Environment intensity"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut lighting.environment_rotation,
                                -std::f32::consts::PI..=std::f32::consts::PI,
                            )
                            .text("Environment rotation"),
                        );
                    });
                    egui::CollapsingHeader::new("Camera").show(ui, |ui| {
                        ui.add(egui::Slider::new(&mut camera.speed, 0.1..=100.0).text("Speed"));
                        ui.add(
                            egui::Slider::new(&mut camera.sensitivity, 0.0001..=0.02)
                                .logarithmic(true)
                                .text("Mouse sensitivity"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut camera.fov_y_radians,
                                20_f32.to_radians()..=120_f32.to_radians(),
                            )
                            .text("FOV (radians)"),
                        );
                        ui.label(format!(
                            "Position: {:.2}, {:.2}, {:.2}",
                            camera.position.x, camera.position.y, camera.position.z
                        ));
                    });
                    egui::CollapsingHeader::new("Statistics")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.label(format!("{:.1} FPS", statistics.fps()));
                            ui.label(format!("{:.2} ms/frame", statistics.frame_time_ms()));
                            frame_time_graph(ui, statistics);
                        });
                    egui::CollapsingHeader::new("Slang shaders").show(ui, |ui| {
                        if slang_status.compiling {
                            ui.spinner();
                            ui.label("Compiling in background…");
                        } else if slang_status.last_error.is_some() {
                            ui.colored_label(
                                Color32::LIGHT_RED,
                                "Compilation failed — using previous pipelines",
                            );
                        } else {
                            ui.colored_label(Color32::LIGHT_GREEN, "Ready · Slang 2026.12");
                        }
                        if ui.button("Recompile all").clicked() {
                            recompile_shaders = true;
                        }
                        if let Some(error) = &slang_status.last_error {
                            ui.code(error);
                        }
                        if let Some(last_success) = slang_status.last_success {
                            ui.label(format!(
                                "Last successful compile: {:.1}s ago",
                                last_success.elapsed().as_secs_f32()
                            ));
                        }
                        egui::CollapsingHeader::new("Shader assets and diagnostics")
                            .default_open(true)
                            .show(ui, |ui| {
                                for asset in SHADER_ASSETS {
                                    ui.group(|ui| {
                                        ui.strong(asset.name);
                                        ui.monospace(asset.source);
                                        ui.label(format!("Entries: {}", asset.entries));
                                        ui.label(format!("Definitions: {}", asset.definitions));
                                        ui.label(format!("Variants: {}", asset.variants));
                                    });
                                }
                                if let Some(output) = &slang_status.last_output {
                                    ui.label("Last compiler output:");
                                    egui::ScrollArea::vertical()
                                        .max_height(120.0)
                                        .show(ui, |ui| ui.code(output));
                                }
                            });
                        ui.label("RMB: look · WASD: move · Space/Ctrl: vertical · Shift: boost");
                        if let Some(error) = shader_error {
                            ui.separator();
                            ui.colored_label(Color32::LIGHT_RED, "Last shader reload failed");
                            ui.code(error);
                        }
                    });
                });
        });
    egui::TopBottomPanel::bottom("status-bar").show(context, |ui| {
        ui.horizontal(|ui| {
            ui.label("3dempire · Phase 4");
            ui.separator();
            ui.label("Slang hot reload active");
            ui.checkbox(&mut state.show_demo_window, "egui demo");
        });
    });
    if state.show_demo_window {
        egui::Window::new("Egui demo").show(context, |ui| {
            ui.label("Phase 1 engine shell");
        });
    }
    recompile_shaders
}

fn frame_time_graph(ui: &mut egui::Ui, statistics: &FrameStatistics) {
    let desired = egui::vec2(ui.available_width(), 92.0);
    let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 3.0, egui::Color32::from_gray(22));
    let values: Vec<_> = statistics.frame_times_ms().collect();
    if values.len() < 2 {
        return;
    }
    let max_ms = values
        .iter()
        .copied()
        .fold(16.67_f32, f32::max)
        .ceil()
        .max(20.0);
    for target in [16.67, 33.33] {
        let y = rect.bottom() - rect.height() * (target / max_ms).min(1.0);
        painter.line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            egui::Stroke::new(1.0_f32, egui::Color32::DARK_GRAY),
        );
    }
    let points: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, value)| {
            let x = rect.left() + rect.width() * i as f32 / (values.len() - 1) as f32;
            let y = rect.bottom() - rect.height() * (*value / max_ms).min(1.0);
            egui::pos2(x, y)
        })
        .collect();
    painter.add(egui::Shape::line(
        points,
        egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(92, 190, 255)),
    ));
    painter.text(
        rect.left_top() + egui::vec2(4.0, 3.0),
        egui::Align2::LEFT_TOP,
        format!("Frame time · 0–{max_ms:.0} ms"),
        egui::FontId::monospace(10.0),
        egui::Color32::LIGHT_GRAY,
    );
}
