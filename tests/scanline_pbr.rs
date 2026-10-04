//! Property tests of the scanline backend's PBR / texture / alpha /
//! shadow / animation paths over the shared `testscenes`, plus small
//! raw goldens (`tests/goldens/*.rgba`, regenerate with
//! `OXIDEAV_RENDER_BLESS=1 cargo test -p oxideav-render --test scanline_pbr`).

use oxideav_mesh3d::{
    AlphaMode, Material, Mesh, MinFilter, Node, Primitive, Sampler, Scene3D, Topology, Transform,
};
use oxideav_render::testscenes::*;
use oxideav_render::{
    make_renderer, BackgroundColor, Camera, DepthRange, PreparedScene, RenderBackend,
    RenderOptions, RgbaImage, ShadingMode, ToneMap,
};

const BG: [u8; 4] = [0, 0, 0, 255];

fn opts(w: u32, h: u32) -> RenderOptions {
    RenderOptions {
        width: w,
        height: h,
        shading: ShadingMode::Pbr,
        scene_camera: Some(0),
        background: BackgroundColor(BG),
        ..RenderOptions::default()
    }
}

fn render(scene: &Scene3D, o: &RenderOptions) -> RgbaImage {
    make_renderer(RenderBackend::Scanline)
        .unwrap()
        .render(scene, o)
        .unwrap()
}

/// Pixel coordinates of world point `p` under the camera `o` selects.
fn project(scene: &Scene3D, o: &RenderOptions, p: [f32; 3]) -> (u32, u32) {
    let prepared = PreparedScene::from_render_options(scene, o);
    let cam = Camera::resolve(&prepared, o, o.width, o.height);
    let m = cam.view_projection(DepthRange::NegOneToOne);
    let c: Vec<f32> = (0..4)
        .map(|r| m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3])
        .collect();
    let x = (c[0] / c[3] * 0.5 + 0.5) * o.width as f32;
    let y = (1.0 - (c[1] / c[3] * 0.5 + 0.5)) * o.height as f32;
    (x as u32, y as u32)
}

fn px(img: &RgbaImage, xy: (u32, u32)) -> [u8; 4] {
    img.pixel(xy.0, xy.1).unwrap()
}

fn close(a: [u8; 4], b: [u8; 4], tol: u8) -> bool {
    (0..4).all(|k| a[k].abs_diff(b[k]) <= tol)
}

#[test]
fn cornell_walls_take_their_colours() {
    let scene = cornell_box();
    let o = opts(160, 120);
    let img = render(&scene, &o);
    let left = px(&img, project(&scene, &o, [-0.99, 0.0, -0.5]));
    let right = px(&img, project(&scene, &o, [0.99, 0.0, -0.5]));
    let back = px(&img, project(&scene, &o, [0.4, 0.5, -1.0]));
    assert!(
        left[0] > 2 * left[1] && left[0] > 2 * left[2],
        "left {left:?}"
    );
    assert!(
        right[1] > right[0] && right[1] > right[2],
        "right {right:?}"
    );
    assert!(
        back[0].abs_diff(back[1]) < 12 && back[0] > 60,
        "back {back:?}"
    );
    // Inverse-square: ceiling right under the light outshines the
    // floor's back corner.
    let ceil = luma(px(&img, project(&scene, &o, [0.0, 0.99, -0.5])));
    let corner = luma(px(&img, project(&scene, &o, [-0.9, -0.99, -0.9])));
    assert!(ceil > corner + 20.0, "{ceil} vs {corner}");
}

#[test]
fn smooth_spheres_have_tighter_brighter_highlights() {
    // 2×2 grid: bottom row smooth (roughness 0.05), top row rough.
    let scene = sphere_grid(2, 2);
    let o = opts(128, 128);
    let img = render(&scene, &o);
    let max_luma = |x0, y0, x1, y1| {
        let mut m: f64 = 0.0;
        for y in y0..y1 {
            for x in x0..x1 {
                m = m.max(luma(img.pixel(x, y).unwrap()));
            }
        }
        m
    };
    let smooth = max_luma(0, 64, 64, 128);
    let rough = max_luma(0, 0, 64, 64);
    assert!(smooth > rough + 15.0, "smooth {smooth} rough {rough}");
    // Metals (right column) have no diffuse lobe: darker on average
    // than dielectrics under the same light.
    let dielectric_rough = region_mean(&img, 0, 0, 64, 64);
    let metal_rough = region_mean(&img, 64, 0, 128, 64);
    assert!(
        dielectric_rough[1] > metal_rough[1],
        "{dielectric_rough:?} {metal_rough:?}"
    );
}

#[test]
fn checker_floor_minifies_to_grey_with_mipmaps() {
    let o = RenderOptions {
        tone_map: ToneMap::Clamp,
        ..opts(160, 120)
    };
    let trilinear = checker_floor(Sampler::default_sampler());
    let nearest = checker_floor(Sampler::default_sampler().with_min_filter(MinFilter::Nearest));
    let img_t = render(&trilinear, &o);
    let img_n = render(&nearest, &o);
    // Near band: full-contrast checker either way.
    let (_, near_y) = project(&trilinear, &o, [0.0, 0.0, 1.0]);
    assert!(region_stddev(&img_t, 20, near_y - 3, 140, near_y + 3) > 80.0);
    // Far band (z ≈ -20): trilinear converges to mid-grey; nearest aliases.
    let (_, far_y) = project(&trilinear, &o, [0.0, 0.0, -20.0]);
    let (x0, x1) = (70, 90);
    let (y0, y1) = (far_y + 1, far_y + 4);
    let sd_t = region_stddev(&img_t, x0, y0, x1, y1);
    let sd_n = region_stddev(&img_n, x0, y0, x1, y1);
    let mean_t = region_mean(&img_t, x0, y0, x1, y1);
    assert!(sd_t < 25.0, "trilinear far stddev {sd_t}");
    assert!(sd_n > sd_t + 20.0, "nearest {sd_n} vs trilinear {sd_t}");
    // Linear 0.5 → sRGB ≈ 188.
    assert!((mean_t[0] - 188.0).abs() < 25.0, "{mean_t:?}");
}

#[test]
fn textured_quad_samples_texels_in_place() {
    let nearest = Sampler::default_sampler()
        .with_mag_filter(oxideav_mesh3d::MagFilter::Nearest)
        .with_min_filter(MinFilter::Nearest);
    let scene = textured_quad(nearest);
    let o = opts(96, 96);
    let img = render(&scene, &o);
    // UV (0,0) is the top-left texel: red. Cells alternate.
    let red = [255, 0, 0, 255];
    let blue = [0, 0, 255, 255];
    assert_eq!(px(&img, project(&scene, &o, [-0.75, 0.75, 0.0])), red);
    assert_eq!(px(&img, project(&scene, &o, [-0.25, 0.75, 0.0])), blue);
    assert_eq!(px(&img, project(&scene, &o, [-0.75, 0.25, 0.0])), blue);
    assert_eq!(px(&img, project(&scene, &o, [0.75, -0.75, 0.0])), red);
}

#[test]
fn shared_diagonal_has_no_cracks() {
    // Regression: with the auto-framed camera the quad's shared
    // diagonal runs exactly through pixel centres; float rounding of
    // the two triangles' edge functions used to leave background
    // pixels on it.
    for (w, h) in [(160, 120), (97, 61), (128, 128), (333, 250)] {
        let scene = textured_quad(Sampler::default());
        let o = RenderOptions {
            scene_camera: None,
            background: BackgroundColor([16, 16, 20, 255]),
            ..opts(w, h)
        };
        let img = render(&scene, &o);
        let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0, 0);
        for y in 0..h {
            for x in 0..w {
                if img.pixel(x, y) != Some([16, 16, 20, 255]) {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
            }
        }
        for y in y0..=y1 {
            for x in x0..=x1 {
                assert_ne!(
                    img.pixel(x, y),
                    Some([16, 16, 20, 255]),
                    "crack at {x},{y} ({w}x{h})"
                );
            }
        }
    }
}

#[test]
fn alpha_mask_and_blend_composite_correctly() {
    let scene = alpha_planes();
    let o = opts(320, 240);
    let img = render(&scene, &o);
    let at = |x, y| px(&img, project(&scene, &o, [x, y, 0.0]));
    // Linear 0.5 → sRGB 188.
    assert!(
        close(at(-0.5, 0.5), [188, 188, 0, 255], 2),
        "red over green {:?}",
        at(-0.5, 0.5)
    );
    assert!(
        close(at(-0.5, -0.5), [188, 0, 188, 255], 2),
        "red over blue {:?}",
        at(-0.5, -0.5)
    );
    assert!(
        close(at(0.5, 0.5), [0, 0, 255, 255], 0),
        "masked-out cell shows blue"
    );
    assert!(
        close(at(0.5, -0.5), [0, 255, 0, 255], 0),
        "masked-in cell green"
    );
    // Legacy modes ignore alpha modes: the red plane is opaque.
    let flat = render(
        &scene,
        &RenderOptions {
            shading: ShadingMode::Flat,
            ..o.clone()
        },
    );
    let p = px(&flat, project(&scene, &o, [-0.5, 0.5, 0.0]));
    assert_eq!(&p[..3], &[255, 0, 0]);
}

#[test]
fn shadow_maps_darken_occluded_floor_only() {
    let scene = shadow_box();
    let off = opts(160, 120);
    let on = RenderOptions {
        shadows: true,
        ..off.clone()
    };
    let a = render(&scene, &off);
    let b = render(&scene, &on);
    // Floor just in front of the cube (visible, under its shadow).
    let shadowed = project(&scene, &off, [0.0, 0.0, 0.45]);
    let lit = project(&scene, &off, [2.0, 0.0, 2.0]);
    assert!(luma(px(&b, shadowed)) + 40.0 < luma(px(&a, shadowed)));
    assert!(
        (luma(px(&b, lit)) - luma(px(&a, lit))).abs() < 2.0,
        "no acne on lit floor"
    );
    // The cube's top face is lit (no self-shadowing).
    let top = project(&scene, &off, [0.0, 2.0, 0.0]);
    assert!((luma(px(&b, top)) - luma(px(&a, top))).abs() < 2.0);
}

#[test]
fn skinned_morph_animation_moves_the_beam() {
    let scene = skinned_morph_beam();
    let t0 = RenderOptions {
        time: Some(0.0),
        ..opts(160, 120)
    };
    let t1 = RenderOptions {
        time: Some(1.0),
        ..t0.clone()
    };
    let a = render(&scene, &t0);
    let b = render(&scene, &t1);
    let tip_rest = project(&scene, &t0, [1.9, 0.0, 0.0]);
    let tip_bent = project(&scene, &t0, [0.72, 0.95, 0.0]);
    assert_ne!(px(&a, tip_rest), BG, "rest tip painted");
    assert_eq!(px(&a, tip_bent), BG);
    assert_eq!(px(&b, tip_rest), BG, "bent beam left the rest tip");
    assert_ne!(px(&b, tip_bent), BG, "bent + morphed tip painted");
    // Rest pose without time is the t = 0 pose here.
    let rest = render(
        &scene,
        &RenderOptions {
            time: None,
            ..t0.clone()
        },
    );
    assert_eq!(rest.pixels, a.pixels);
}

#[test]
fn normal_map_tilts_shading() {
    let o = opts(64, 64);
    let flat = render(&normal_mapped_quad(0.0), &o);
    let tilted = render(&normal_mapped_quad(1.0), &o);
    let c = (32, 32);
    assert!(luma(px(&tilted, c)) > luma(px(&flat, c)) + 20.0);
}

fn single_quad(mat: Material, colors: Option<[f32; 4]>) -> Scene3D {
    let mut scene = Scene3D::new();
    let m = scene.add_material(mat);
    let mut p = quad(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        [1.0, 1.0],
    );
    p.material = Some(m);
    if let Some(c) = colors {
        p.colors = vec![vec![c; 4]];
    }
    let mesh = scene.add_mesh(Mesh::new(None).with_primitive(p));
    let n = scene.add_node(Node::new().with_mesh(mesh));
    scene.add_root(n);
    add_camera(&mut scene, [0.0, 0.0, 2.5], [0.0; 3], 0.9);
    scene
}

#[test]
fn unlit_vertex_colours_and_emissive() {
    let mut m = Material::new();
    m.ext.unlit = true;
    m.base_color = [1.0, 1.0, 1.0, 1.0];
    let img = render(&single_quad(m, Some([0.0, 1.0, 0.0, 1.0])), &opts(32, 32));
    assert_eq!(img.pixel(16, 16), Some([0, 255, 0, 255]));

    let mut e = Material::new();
    e.base_color = [0.0, 0.0, 0.0, 1.0];
    e.emissive_factor = [1.0, 0.25, 0.0];
    e.ext.emissive_strength = Some(4.0);
    let scene = single_quad(e, None);
    let o = RenderOptions {
        ambient: 0.0,
        use_scene_lights: false,
        light: oxideav_render::LightSpec {
            intensity: 0.0,
            ..Default::default()
        },
        ..opts(32, 32)
    };
    let img = render(&scene, &o);
    assert_eq!(
        img.pixel(16, 16),
        Some([255, 255, 0, 255]),
        "clamped 4.0 / 1.0"
    );
    // HDR output keeps the unclamped radiance.
    let hdr = make_renderer(RenderBackend::Scanline)
        .unwrap()
        .render_hdr(&scene, &o)
        .unwrap();
    let p = hdr.pixel(16, 16).unwrap();
    assert!(
        (p[0] - 4.0).abs() < 1e-3 && (p[1] - 1.0).abs() < 1e-3,
        "{p:?}"
    );
    // Tone mapping compresses instead of clipping; exposure scales.
    let r = render(
        &scene,
        &RenderOptions {
            tone_map: ToneMap::Reinhard,
            ..o.clone()
        },
    );
    let rp = r.pixel(16, 16).unwrap();
    // Luminance-based Reinhard: L = 1.56 → scale 0.39; green 1.0 → 0.39.
    assert!(rp[1] < 180, "{rp:?}");
    let dark = render(
        &scene,
        &RenderOptions {
            exposure: 0.1,
            ..o.clone()
        },
    );
    assert!(dark.pixel(16, 16).unwrap()[0] < 200);
    // Background bypasses the tone map.
    assert_eq!(r.pixel(0, 0), Some(BG));
}

#[test]
fn back_faces_culled_unless_double_sided() {
    let mut m = Material::new();
    m.ext.unlit = true;
    let mut scene = single_quad(m.clone(), None);
    // Flip the quad to face away (rotate 180° about Y).
    scene.nodes[0].transform = Transform::Trs {
        translation: [0.0; 3],
        rotation: [0.0, 1.0, 0.0, 0.0],
        scale: [1.0; 3],
    };
    let img = render(&scene, &opts(16, 16));
    assert_eq!(img.pixel(8, 8), Some(BG));
    scene.materials[0].double_sided = true;
    let img = render(&scene, &opts(16, 16));
    assert_ne!(img.pixel(8, 8), Some(BG));
    // Legacy modes never cull.
    scene.materials[0].double_sided = false;
    let img = render(
        &scene,
        &RenderOptions {
            shading: ShadingMode::Flat,
            ..opts(16, 16)
        },
    );
    assert_ne!(img.pixel(8, 8), Some(BG));
}

#[test]
fn geometry_crossing_the_near_plane_is_clipped_not_dropped() {
    // A floor running from in front of the camera to well behind it.
    let mut m = Material::new();
    m.ext.unlit = true;
    m.double_sided = true;
    let mut scene = Scene3D::new();
    let mat = scene.add_material(m);
    let mut p = Primitive::new(Topology::Triangles);
    p.positions = vec![[-5.0, -1.0, 10.0], [5.0, -1.0, 10.0], [0.0, -1.0, -10.0]];
    p.material = Some(mat);
    let mesh = scene.add_mesh(Mesh::new(None).with_primitive(p));
    let n = scene.add_node(Node::new().with_mesh(mesh));
    scene.add_root(n);
    add_camera(&mut scene, [0.0, 0.0, 0.0], [0.0, -0.3, -1.0], 1.2);
    let img = render(&scene, &opts(64, 64));
    // Bottom rows (floor right under / in front of the camera) painted.
    assert_ne!(img.pixel(32, 62), Some(BG));
}

#[test]
fn mask_cutoff_respected_with_constant_alpha() {
    let mut m = Material::new();
    m.ext.unlit = true;
    m.base_color = [1.0, 1.0, 1.0, 0.4];
    m.alpha_mode = AlphaMode::Mask { cutoff: 0.5 };
    let img = render(&single_quad(m.clone(), None), &opts(16, 16));
    assert_eq!(img.pixel(8, 8), Some(BG));
    m.alpha_mode = AlphaMode::Mask { cutoff: 0.3 };
    let img = render(&single_quad(m, None), &opts(16, 16));
    assert_eq!(
        img.pixel(8, 8),
        Some([255, 255, 255, 255]),
        "MASK output is opaque"
    );
}

#[test]
fn parallel_render_is_deterministic() {
    let scene = cornell_box();
    let o = RenderOptions {
        shadows: true,
        aa: 2,
        ..opts(300, 200)
    };
    let a = render(&scene, &o);
    let b = render(&scene, &o);
    assert_eq!(a.pixels, b.pixels);
}

#[cfg(feature = "registry")]
#[test]
fn registry_resolver_decodes_png_textures() {
    use std::sync::Arc;
    let rgba = checker_rgba8(4, 4, [255, 0, 0, 255], [0, 0, 255, 255]);
    let png = oxideav_png::encode_rgba8(4, 4, &rgba, &Default::default()).unwrap();
    let mut scene = textured_quad(Sampler::default_sampler());
    let mut tex = oxideav_mesh3d::Texture::from_encoded("image/png", png);
    tex.sampler = Sampler::default_sampler()
        .with_mag_filter(oxideav_mesh3d::MagFilter::Nearest)
        .with_min_filter(MinFilter::Nearest);
    scene.textures[0] = tex;
    let o = opts(96, 96);
    // Without a decoder the texture is absent: base factor (white).
    let plain = render(&scene, &o);
    assert_eq!(
        px(&plain, project(&scene, &o, [-0.75, 0.75, 0.0])),
        [255, 255, 255, 255]
    );
    let mut ctx = oxideav_core::RuntimeContext::new();
    oxideav_png::register(&mut ctx);
    let resolver: Arc<dyn oxideav_render::TextureResolver> =
        Arc::new(oxideav_render::RegistryTextureResolver::new(Arc::new(ctx)));
    let mut r = oxideav_render::ScanlineRenderer::with_texture_resolver(resolver.clone());
    use oxideav_render::Renderer;
    let img = r.render(&scene, &o).unwrap();
    // Same through the backend-agnostic trait hook.
    let mut dynr = make_renderer(RenderBackend::Scanline).unwrap();
    dynr.set_texture_resolver(resolver);
    assert_eq!(dynr.render(&scene, &o).unwrap().pixels, img.pixels);
    assert_eq!(
        px(&img, project(&scene, &o, [-0.75, 0.75, 0.0])),
        [255, 0, 0, 255]
    );
    assert_eq!(
        px(&img, project(&scene, &o, [-0.25, 0.75, 0.0])),
        [0, 0, 255, 255]
    );
}

// ---------------------------------------------------------------------
// Raw goldens.
// ---------------------------------------------------------------------

fn golden(name: &str, img: &RgbaImage, max_mae: f64) {
    let path = format!("{}/tests/goldens/{name}.rgba", env!("CARGO_MANIFEST_DIR"));
    if std::env::var_os("OXIDEAV_RENDER_BLESS").is_some() {
        std::fs::create_dir_all(format!("{}/tests/goldens", env!("CARGO_MANIFEST_DIR"))).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&img.width.to_le_bytes());
        bytes.extend_from_slice(&img.height.to_le_bytes());
        bytes.extend_from_slice(&img.pixels);
        std::fs::write(&path, bytes).unwrap();
        return;
    }
    let bytes = std::fs::read(&path).unwrap_or_else(|_| panic!("missing golden {path}"));
    let w = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let h = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let gold = RgbaImage {
        width: w,
        height: h,
        stride: w as usize * 4,
        pixels: bytes[8..].to_vec(),
    };
    let mae = mean_abs_error(img, &gold);
    assert!(
        mae <= max_mae,
        "{name}: MAE {mae} > {max_mae} (PSNR {})",
        psnr(img, &gold)
    );
}

#[test]
fn golden_cornell_aces() {
    let o = RenderOptions {
        tone_map: ToneMap::AcesFitted,
        shadows: true,
        aa: 2,
        ..opts(64, 48)
    };
    golden("cornell_64x48", &render(&cornell_box(), &o), 1.0);
}

#[test]
fn golden_sphere_grid() {
    golden(
        "spheres_64x64",
        &render(&sphere_grid(3, 3), &opts(64, 64)),
        1.0,
    );
}

#[test]
fn golden_alpha_planes() {
    golden("alpha_48x36", &render(&alpha_planes(), &opts(48, 36)), 0.5);
}
