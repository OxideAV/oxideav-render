//! Render every `oxideav_render::testscenes` scene with the scanline
//! backend in PBR mode (or the backend named by the second argument:
//! `scanline`, `raycast`, `pathtrace`; the third argument sets the
//! path tracer's samples per pixel, default 64) and write PNGs to the
//! directory given as the first argument (default: current directory).
//!
//! `cargo run -p oxideav-render --example dump_testscenes --release -- out/ pathtrace 256`

use oxideav_mesh3d::Sampler;
use oxideav_render::testscenes::*;
use oxideav_render::{
    make_renderer, BackgroundColor, PathTraceOptions, RenderBackend, RenderOptions, ShadingMode,
    ToneMap,
};

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());
    std::fs::create_dir_all(&dir).expect("create output dir");
    let backend = match std::env::args().nth(2).as_deref() {
        Some("pathtrace") => RenderBackend::PathTrace,
        Some("raycast") => RenderBackend::Raycast,
        _ => RenderBackend::Scanline,
    };
    let spp = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let scenes = [
        ("cornell", cornell_box(), None),
        ("spheres", sphere_grid(5, 4), None),
        (
            "checker_floor",
            checker_floor(Sampler::default_sampler()),
            None,
        ),
        ("alpha", alpha_planes(), None),
        ("shadow", shadow_box(), None),
        ("beam_t0", skinned_morph_beam(), Some(0.0)),
        ("beam_t1", skinned_morph_beam(), Some(1.0)),
        ("normal_map", normal_mapped_quad(0.8), None),
    ];
    for (name, scene, time) in scenes {
        let opts = RenderOptions {
            width: 320,
            height: 240,
            shading: ShadingMode::Pbr,
            scene_camera: Some(0),
            background: BackgroundColor([20, 20, 30, 255]),
            tone_map: ToneMap::AcesFitted,
            shadows: true,
            aa: 2,
            time,
            path_trace: PathTraceOptions {
                samples_per_pixel: spp,
                ..PathTraceOptions::default()
            },
            ..RenderOptions::default()
        };
        let t = std::time::Instant::now();
        let img = make_renderer(backend)
            .expect("renderer")
            .render(&scene, &opts)
            .expect("render");
        let png =
            oxideav_png::encode_rgba8(img.width, img.height, &img.pixels, &Default::default())
                .expect("png");
        let path = format!("{dir}/{name}.png");
        std::fs::write(&path, png).expect("write");
        println!("{path} ({:?})", t.elapsed());
    }
}
