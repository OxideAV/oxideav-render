//! Rendered properties of the KHR material extensions only the path
//! tracer shades (`KHR_materials_anisotropy`, `_iridescence`,
//! `_diffuse_transmission`, `_dispersion`) on the `testscenes`
//! reference scenes, plus the showcase scene's determinism and
//! convergence.

use oxideav_render::testscenes::*;
use oxideav_render::{
    make_renderer, BackgroundColor, HdrImage, LightSpec, PathTraceOptions, RenderBackend,
    RenderOptions, ShadingMode,
};

fn opts(w: u32, h: u32, spp: u32, bounces: u32) -> RenderOptions {
    RenderOptions {
        width: w,
        height: h,
        shading: ShadingMode::Pbr,
        scene_camera: Some(0),
        use_scene_lights: true,
        ambient: 0.0,
        light: LightSpec {
            intensity: 0.0,
            ..LightSpec::default_light()
        },
        background: BackgroundColor([0, 0, 0, 255]),
        path_trace: PathTraceOptions {
            samples_per_pixel: spp,
            max_bounces: bounces,
            ..PathTraceOptions::default()
        },
        ..RenderOptions::default()
    }
}

fn hdr(scene: &oxideav_mesh3d::Scene3D, o: &RenderOptions) -> HdrImage {
    make_renderer(RenderBackend::PathTrace)
        .unwrap()
        .render_hdr(scene, o)
        .unwrap()
}

fn lum(p: &[f32]) -> f32 {
    0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
}

/// Luminance-weighted standard deviation of the highlight (pixels
/// above `thr`) along image x and y.
fn highlight_extent(img: &HdrImage, thr: f32) -> (f64, f64) {
    let (mut sw, mut sx, mut sy, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for y in 0..img.height {
        for x in 0..img.width {
            let p = img.pixel(x, y).unwrap();
            let l = lum(&p);
            if l > thr {
                let w = l as f64;
                sw += w;
                sx += w * x as f64;
                sy += w * y as f64;
                sxx += w * (x * x) as f64;
                syy += w * (y * y) as f64;
            }
        }
    }
    assert!(sw > 0.0, "no highlight above {thr}");
    let (mx, my) = (sx / sw, sy / sw);
    (
        (sxx / sw - mx * mx).max(0.0).sqrt(),
        (syy / sw - my * my).max(0.0).sqrt(),
    )
}

#[test]
fn brushed_metal_highlight_stretches_along_the_anisotropy_direction() {
    let o = opts(96, 96, 16, 1);
    let ratio = |strength: f32, rot: f32| {
        let img = hdr(&brushed_metal_plate(strength, rot), &o);
        let (sx, sy) = highlight_extent(&img, 0.5);
        sx / sy
    };
    let iso = ratio(0.0, 0.0);
    let along_x = ratio(0.9, 0.0);
    let along_z = ratio(0.9, std::f32::consts::FRAC_PI_2);
    assert!(
        along_x > 1.4 * iso,
        "rotation 0 stretches along image x: {along_x} vs isotropic {iso}"
    );
    assert!(
        along_z < iso / 1.4,
        "rotation π/2 stretches along image y: {along_z} vs isotropic {iso}"
    );
    assert!(along_x / along_z > 2.5, "{along_x} / {along_z}");
}

#[test]
fn iridescent_sphere_tints_a_white_highlight() {
    let o = opts(96, 96, 16, 1);
    let saturation = |scene| {
        let img = hdr(&scene, &o);
        let (mut s, mut n) = (0.0f64, 0usize);
        for p in img.pixels.chunks_exact(4) {
            let max = p[0].max(p[1]).max(p[2]);
            // Well above the dark sphere's diffuse level (≈ 0.1).
            if max > 0.3 {
                let min = p[0].min(p[1]).min(p[2]);
                s += ((max - min) / max) as f64;
                n += 1;
            }
        }
        assert!(n > 10, "highlight too small: {n} pixels");
        s / n as f64
    };
    let grey = saturation(iridescent_sphere(0.0, 550.0));
    let tinted = saturation(iridescent_sphere(1.0, 550.0));
    assert!(
        grey < 0.03,
        "white light on grey dielectric stays grey: {grey}"
    );
    assert!(tinted > 0.08, "thin film tints the highlight: {tinted}");
}

#[test]
fn backlit_leaf_glows_only_with_diffuse_transmission() {
    let o = opts(64, 64, 16, 1);
    let mean = |scene| {
        let img = hdr(&scene, &o);
        let (mut s, mut n) = ([0.0f64; 3], 0usize);
        for p in img.pixels.chunks_exact(4) {
            if p[3] > 0.5 {
                for k in 0..3 {
                    s[k] += p[k] as f64;
                }
                n += 1;
            }
        }
        [s[0] / n as f64, s[1] / n as f64, s[2] / n as f64]
    };
    let dark = mean(backlit_leaf(0.0));
    let lit = mean(backlit_leaf(1.0));
    let half = mean(backlit_leaf(0.5));
    assert!(dark.iter().all(|&c| c < 1e-4), "no transmission: {dark:?}");
    assert!(lit[1] > 0.05, "backlight comes through: {lit:?}");
    // The transmitted colour is the diffuse-transmission colour, not
    // the base colour: green-dominant with red above blue.
    assert!(lit[1] > lit[0] && lit[0] > lit[2], "{lit:?}");
    for k in 0..3 {
        assert!(
            (half[k] / lit[k] - 0.5).abs() < 0.1,
            "linear in dt: {half:?} vs {lit:?}"
        );
    }
}

#[test]
fn dispersion_fringes_colour_the_refracted_checker_without_changing_its_mean() {
    let o = opts(96, 96, 32, 6);
    let stats = |scene| {
        let img = hdr(&scene, &o);
        let (cx, cy, r2) = (48.0f32, 48.0f32, 20.0f32 * 20.0);
        let (mut l, mut spread, mut n) = (0.0f64, 0.0f64, 0usize);
        for y in 0..img.height {
            for x in 0..img.width {
                let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                if dx * dx + dy * dy > r2 {
                    continue;
                }
                let p = img.pixel(x, y).unwrap();
                l += lum(&p) as f64;
                spread += (p[0].max(p[1]).max(p[2]) - p[0].min(p[1]).min(p[2])) as f64;
                n += 1;
            }
        }
        (l / n as f64, spread / n as f64)
    };
    let (l0, s0) = stats(dispersive_sphere(0.0));
    let (l1, s1) = stats(dispersive_sphere(1.0));
    assert!(s0 < 0.01, "achromatic glass shows a grey checker: {s0}");
    assert!(
        s1 > 5.0 * s0.max(0.002),
        "dispersion separates the channels: {s1}"
    );
    assert!(
        (l1 - l0).abs() < 0.08 * l0.max(0.1),
        "one-sample channel selection is unbiased: {l1} vs {l0}"
    );
}

#[test]
fn showcase_is_deterministic_and_converges() {
    let scene = material_extensions();
    let o = opts(96, 64, 8, 4);
    let a = make_renderer(RenderBackend::PathTrace)
        .unwrap()
        .render(&scene, &o)
        .unwrap();
    let b = make_renderer(RenderBackend::PathTrace)
        .unwrap()
        .render(&scene, &o)
        .unwrap();
    assert_eq!(a.pixels, b.pixels);
    assert!(
        a.pixels.chunks_exact(4).any(|p| p[0] > 10),
        "something rendered"
    );
    let reference = make_renderer(RenderBackend::PathTrace)
        .unwrap()
        .render(
            &scene,
            &RenderOptions {
                path_trace: PathTraceOptions {
                    samples_per_pixel: 128,
                    seed: 99,
                    ..o.path_trace
                },
                ..o.clone()
            },
        )
        .unwrap();
    let p8 = psnr(&a, &reference);
    let c = make_renderer(RenderBackend::PathTrace)
        .unwrap()
        .render(
            &scene,
            &RenderOptions {
                path_trace: PathTraceOptions {
                    samples_per_pixel: 32,
                    ..o.path_trace
                },
                ..o.clone()
            },
        )
        .unwrap();
    let p32 = psnr(&c, &reference);
    assert!(
        p32 > p8 + 2.0,
        "more samples converge: {p8:.1} dB → {p32:.1} dB"
    );
}
