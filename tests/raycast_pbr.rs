//! Raycast `Pbr` tests: parity with the scanline backend where ray
//! tracing and rasterising agree, and properties of what only rays
//! do (hard shadows for every light, reflection, refraction, MASK /
//! BLEND through rays).

use oxideav_mesh3d::{
    material::{Transmission, Volume},
    AlphaMode, Light, Material, Mesh, MinFilter, Node, Primitive, Sampler, Scene3D, TextureRef,
    Transform,
};
use oxideav_render::testscenes::*;
use oxideav_render::{
    make_renderer, BackgroundColor, Camera, DepthRange, PreparedScene, RenderBackend,
    RenderOptions, RgbaImage, ShadingMode,
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

fn render(backend: RenderBackend, scene: &Scene3D, o: &RenderOptions) -> RgbaImage {
    make_renderer(backend).unwrap().render(scene, o).unwrap()
}

fn parity_scenes() -> Vec<(&'static str, Scene3D, RenderOptions)> {
    let flat = RenderOptions {
        max_ray_depth: 0,
        ..opts(160, 120)
    };
    vec![
        ("cornell_box", cornell_box(), flat.clone()),
        ("sphere_grid", sphere_grid(3, 3), flat.clone()),
        (
            "checker_floor",
            checker_floor(Sampler::default_sampler()),
            flat.clone(),
        ),
        (
            "textured_quad",
            textured_quad(Sampler::default_sampler()),
            flat.clone(),
        ),
        ("alpha_planes", alpha_planes(), flat.clone()),
        ("shadow_box", shadow_box(), flat.clone()),
        ("normal_mapped_quad", normal_mapped_quad(1.0), flat.clone()),
        (
            "skinned_morph_beam",
            skinned_morph_beam(),
            RenderOptions {
                time: Some(0.5),
                ..flat.clone()
            },
        ),
    ]
}

/// Fraction of pixels differing by more than `tol` in any channel.
fn outlier_fraction(a: &RgbaImage, b: &RgbaImage, tol: u8) -> f64 {
    let n = a
        .pixels_rgba()
        .zip(b.pixels_rgba())
        .filter(|(p, q)| (0..4).any(|k| p[k].abs_diff(q[k]) > tol))
        .count();
    n as f64 / (a.width * a.height) as f64
}

#[test]
fn pbr_matches_scanline_without_secondary_rays() {
    // Same prepared scene, camera, BRDF, texture derivatives and
    // resolve: with secondary rays and shadows off the two backends
    // differ only where rasterisation coverage and ray sampling
    // disagree on a silhouette pixel.
    for (name, scene, o) in parity_scenes() {
        for aa in [1, 2] {
            let o = RenderOptions { aa, ..o.clone() };
            let s = render(RenderBackend::Scanline, &scene, &o);
            let r = render(RenderBackend::Raycast, &scene, &o);
            let mae = mean_abs_error(&s, &r);
            let p = psnr(&s, &r);
            let out = outlier_fraction(&s, &r, 2);
            eprintln!(
                "parity {name} aa{aa}: MAE {mae:.4} PSNR {p:.2} dB outliers(>2) {:.3}%",
                out * 100.0
            );
            assert!(mae < 0.25, "{name} aa{aa}: MAE {mae}");
            assert!(out < 0.005, "{name} aa{aa}: outliers {out}");
        }
    }
}

#[test]
fn hdr_output_matches_scanline_and_keeps_radiance() {
    let mut e = Material::new();
    e.base_color = [0.0, 0.0, 0.0, 1.0];
    e.emissive_factor = [1.0, 0.25, 0.0];
    e.ext.emissive_strength = Some(4.0);
    let scene = single_quad(e);
    let o = RenderOptions {
        ambient: 0.0,
        use_scene_lights: false,
        light: oxideav_render::LightSpec {
            intensity: 0.0,
            ..Default::default()
        },
        ..opts(32, 32)
    };
    let hdr = make_renderer(RenderBackend::Raycast)
        .unwrap()
        .render_hdr(&scene, &o)
        .unwrap();
    let p = hdr.pixel(16, 16).unwrap();
    assert!(
        (p[0] - 4.0).abs() < 1e-3 && (p[1] - 1.0).abs() < 1e-3,
        "{p:?}"
    );
    let s = make_renderer(RenderBackend::Scanline)
        .unwrap()
        .render_hdr(&scene, &o)
        .unwrap();
    assert_eq!(hdr.pixels, s.pixels);
    // Background is linear-decoded verbatim; tone mapping bypasses it.
    let img = render(RenderBackend::Raycast, &scene, &o);
    assert_eq!(img.pixel(0, 0), Some(BG));
    assert_eq!(img.pixel(16, 16), Some([255, 255, 0, 255]));
}

// ---------------------------------------------------------------------
// Scene helpers.
// ---------------------------------------------------------------------

fn add(scene: &mut Scene3D, mut p: Primitive, mat: Material) {
    let m = scene.add_material(mat);
    p.material = Some(m);
    let mesh = scene.add_mesh(Mesh::new(None).with_primitive(p));
    let n = scene.add_node(Node::new().with_mesh(mesh));
    scene.add_root(n);
}

fn single_quad(mat: Material) -> Scene3D {
    let mut scene = Scene3D::new();
    add(
        &mut scene,
        quad(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            [1.0, 1.0],
        ),
        mat,
    );
    add_camera(&mut scene, [0.0, 0.0, 2.5], [0.0; 3], 0.9);
    scene
}

fn unlit(rgb: [f32; 3]) -> Material {
    let mut m = Material::new();
    m.base_color = [rgb[0], rgb[1], rgb[2], 1.0];
    m.ext.unlit = true;
    m
}

fn floor_quad(half: f32, y: f32) -> Primitive {
    quad(
        [
            [-half, y, half],
            [half, y, half],
            [half, y, -half],
            [-half, y, -half],
        ],
        [1.0, 1.0],
    )
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

fn ray(scene: &Scene3D, o: &RenderOptions) -> RgbaImage {
    render(RenderBackend::Raycast, scene, o)
}

// ---------------------------------------------------------------------
// Shadows.
// ---------------------------------------------------------------------

#[test]
fn ray_traced_shadows_agree_loosely_with_shadow_maps() {
    let scene = shadow_box();
    let off = opts(160, 120);
    let on = RenderOptions {
        shadows: true,
        ..off.clone()
    };
    let r_off = ray(&scene, &off);
    let r_on = ray(&scene, &on);
    let s_on = render(RenderBackend::Scanline, &scene, &on);
    let shadowed = project(&scene, &off, [0.0, 0.0, 0.45]);
    let lit = project(&scene, &off, [2.0, 0.0, 2.0]);
    assert!(luma(px(&r_on, shadowed)) + 40.0 < luma(px(&r_off, shadowed)));
    assert_eq!(px(&r_on, lit), px(&r_off, lit), "no acne on lit floor");
    let top = project(&scene, &off, [0.0, 2.0, 0.0]);
    assert_eq!(px(&r_on, top), px(&r_off, top), "no self-shadowing");
    // Hard ray-traced shadows vs PCF-filtered shadow maps: same
    // shadow, different penumbra / bias.
    let mae = mean_abs_error(&r_on, &s_on);
    let out = outlier_fraction(&r_on, &s_on, 8);
    eprintln!(
        "shadows vs shadow maps: MAE {mae:.3} outliers(>8) {:.2}%",
        out * 100.0
    );
    assert!(mae < 1.5, "MAE {mae}");
    assert!(out < 0.03, "outliers {out}");
}

#[test]
fn point_lights_cast_ray_traced_shadows() {
    // The Cornell box's point light: shadow maps skip point lights,
    // shadow rays do not. The floor's back-left corner sits behind
    // the tall block as seen from the light.
    let scene = cornell_box();
    let off = opts(160, 120);
    let on = RenderOptions {
        shadows: true,
        ..off.clone()
    };
    let a = ray(&scene, &off);
    let b = ray(&scene, &on);
    let hidden = project(&scene, &off, [-0.9, -1.0, -0.9]);
    let lit = project(&scene, &off, [0.9, -1.0, -0.6]);
    assert!(
        luma(px(&b, hidden)) + 15.0 < luma(px(&a, hidden)),
        "{:?} vs {:?}",
        px(&b, hidden),
        px(&a, hidden)
    );
    assert_eq!(px(&b, lit), px(&a, lit));
}

/// White floor under a horizontal occluder plane (`y = 1`, single
/// sided facing up, so the low camera sees the floor through its
/// culled underside) lit straight down.
fn occluder_scene(occluder: Material) -> Scene3D {
    let mut scene = Scene3D::new();
    add(
        &mut scene,
        floor_quad(3.0, 0.0),
        diffuse_material([0.8; 3], 1.0),
    );
    add(&mut scene, floor_quad(1.0, 1.0), occluder);
    add_light(
        &mut scene,
        Light::Directional {
            color: [1.0; 3],
            intensity: 3.0,
        },
        [0.0; 3],
        [0.0, -1.0, 0.0],
    );
    add_camera(&mut scene, [0.0, 0.5, 4.0], [0.0, 0.0, 0.0], 0.9);
    scene
}

#[test]
fn mask_cutouts_and_blend_surfaces_filter_shadow_rays() {
    let o = RenderOptions {
        shadows: true,
        ..opts(160, 120)
    };
    let free = ray(
        &occluder_scene(Material::new()),
        &RenderOptions {
            shadows: false,
            ..o.clone()
        },
    );
    // MASK: 2×2 checker of opaque / cut-out cells.
    let mut mask_scene = occluder_scene(Material::new());
    let tex = mask_scene.add_texture(raw_texture(
        2,
        2,
        &[
            255, 255, 255, 255, 255, 255, 255, 0, //
            255, 255, 255, 0, 255, 255, 255, 255,
        ],
        Sampler::default_sampler()
            .with_mag_filter(oxideav_mesh3d::MagFilter::Nearest)
            .with_min_filter(MinFilter::Nearest),
    ));
    let m = &mut mask_scene.materials[1];
    m.base_color_texture = Some(TextureRef::new(tex));
    m.alpha_mode = AlphaMode::Mask { cutoff: 0.5 };
    let masked = ray(&mask_scene, &o);
    let cells = [
        [-0.5, 0.0, -0.5],
        [0.5, 0.0, -0.5],
        [-0.5, 0.0, 0.5],
        [0.5, 0.0, 0.5],
    ];
    let mut lit = 0;
    let mut dark = 0;
    for c in cells {
        let xy = project(&mask_scene, &o, c);
        let (l, f) = (luma(px(&masked, xy)), luma(px(&free, xy)));
        if (l - f).abs() < 1.0 {
            lit += 1;
        } else if l + 40.0 < f {
            dark += 1;
        }
    }
    assert_eq!((lit, dark), (2, 2), "two cut-out cells pass light");

    // BLEND α = 0.5: half the direct light.
    let mut b = Material::new();
    b.base_color = [1.0, 1.0, 1.0, 0.5];
    b.alpha_mode = AlphaMode::Blend;
    let blend_scene = occluder_scene(b);
    let blended = ray(&blend_scene, &o);
    let xy = project(&blend_scene, &o, [0.0, 0.0, 0.0]);
    let opaque = ray(&occluder_scene(Material::new()), &o);
    let (lb, lf, lo) = (
        luma(px(&blended, xy)),
        luma(px(&free, xy)),
        luma(px(&opaque, xy)),
    );
    assert!(lo + 20.0 < lb && lb + 20.0 < lf, "{lo} < {lb} < {lf}");
}

// ---------------------------------------------------------------------
// MASK / BLEND seen by camera rays.
// ---------------------------------------------------------------------

#[test]
fn alpha_mask_and_blend_composite_like_scanline() {
    let scene = alpha_planes();
    let o = opts(320, 240);
    let img = ray(&scene, &o);
    let at = |x, y| px(&img, project(&scene, &o, [x, y, 0.0]));
    let close = |a: [u8; 4], b: [u8; 4], tol: u8| (0..4).all(|k| a[k].abs_diff(b[k]) <= tol);
    assert!(
        close(at(-0.5, 0.5), [188, 188, 0, 255], 2),
        "{:?}",
        at(-0.5, 0.5)
    );
    assert!(
        close(at(-0.5, -0.5), [188, 0, 188, 255], 2),
        "{:?}",
        at(-0.5, -0.5)
    );
    assert_eq!(at(0.5, 0.5), [0, 0, 255, 255], "masked-out cell shows blue");
    assert_eq!(at(0.5, -0.5), [0, 255, 0, 255], "masked-in cell green");
    // A lone BLEND layer over the background composites in linear
    // space against the background (scanline contract).
    let mut m = unlit([1.0, 0.0, 0.0]);
    m.base_color[3] = 0.5;
    m.alpha_mode = AlphaMode::Blend;
    let lone = single_quad(m);
    let o = RenderOptions {
        background: BackgroundColor([0, 0, 255, 255]),
        ..opts(32, 32)
    };
    let s = render(RenderBackend::Scanline, &lone, &o);
    let r = ray(&lone, &o);
    assert_eq!(r.pixel(16, 16), s.pixel(16, 16));
    assert_eq!(r.pixel(16, 16), Some([188, 0, 188, 255]));
}

#[test]
fn back_faces_culled_unless_double_sided() {
    let mut scene = single_quad(unlit([1.0; 3]));
    scene.nodes[0].transform = Transform::Trs {
        translation: [0.0; 3],
        rotation: [0.0, 1.0, 0.0, 0.0],
        scale: [1.0; 3],
    };
    assert_eq!(ray(&scene, &opts(16, 16)).pixel(8, 8), Some(BG));
    scene.materials[0].double_sided = true;
    assert_ne!(ray(&scene, &opts(16, 16)).pixel(8, 8), Some(BG));
}

// ---------------------------------------------------------------------
// Reflection.
// ---------------------------------------------------------------------

/// A floor of material `floor` in front of an unlit red wall.
fn mirror_scene(floor: Material) -> Scene3D {
    let mut scene = Scene3D::new();
    add(&mut scene, floor_quad(2.0, 0.0), floor);
    add(
        &mut scene,
        quad(
            [
                [-2.0, 0.0, -2.0],
                [2.0, 0.0, -2.0],
                [2.0, 2.0, -2.0],
                [-2.0, 2.0, -2.0],
            ],
            [1.0, 1.0],
        ),
        unlit([1.0, 0.0, 0.0]),
    );
    add_light(
        &mut scene,
        Light::Directional {
            color: [1.0; 3],
            intensity: 2.0,
        },
        [0.0; 3],
        [0.3, -1.0, -0.2],
    );
    add_camera(&mut scene, [0.0, 1.0, 3.0], [0.0, 0.2, -1.0], 0.9);
    scene
}

fn metal(roughness: f32) -> Material {
    let mut m = Material::new();
    m.base_color = [1.0; 4];
    m.metallic = 1.0;
    m.roughness = roughness;
    m
}

#[test]
fn smooth_metal_mirrors_the_wall() {
    let o = opts(128, 96);
    let scene = mirror_scene(metal(0.0));
    let img = ray(&scene, &o);
    // A floor point whose mirror ray hits the wall: a white metal
    // with F0 = 1 reflects the unlit red wall (radiance 1, 0, 0).
    let p = px(&img, project(&scene, &o, [0.0, 0.0, -1.0]));
    assert!(p[0] > 240 && p[1] < 60 && p[2] < 60, "{p:?}");
    // The same floor without secondary rays is a dark metal floor.
    let flat = ray(
        &scene,
        &RenderOptions {
            max_ray_depth: 0,
            ..o.clone()
        },
    );
    let q = px(&flat, project(&scene, &o, [0.0, 0.0, -1.0]));
    assert!(q[0] < 200, "{q:?}");
}

#[test]
fn reflection_fades_with_roughness_and_stops_at_the_cutoff() {
    let o = opts(96, 72);
    let probe = [0.0, 0.0, -1.0];
    let redness = |r: f32| {
        let scene = mirror_scene(metal(r));
        let p = px(&ray(&scene, &o), project(&scene, &o, probe));
        p[0] as i32 - p[1] as i32
    };
    let (r0, r2, r4) = (redness(0.0), redness(0.2), redness(0.4));
    assert!(r0 > r2 && r2 > r4, "{r0} > {r2} > {r4}");
    // At / above the cut-off: exactly the depth-0 (scanline) result.
    let scene = mirror_scene(metal(0.6));
    let a = ray(&scene, &o);
    let b = ray(
        &scene,
        &RenderOptions {
            max_ray_depth: 0,
            ..o.clone()
        },
    );
    assert_eq!(a.pixels, b.pixels);
    let s = render(RenderBackend::Scanline, &scene, &o);
    assert!(mean_abs_error(&a, &s) < 0.25);
    // A zero cut-off disables reflection rays entirely.
    let scene = mirror_scene(metal(0.0));
    let none = ray(
        &scene,
        &RenderOptions {
            reflection_roughness_cutoff: 0.0,
            ..o.clone()
        },
    );
    let flat = ray(
        &scene,
        &RenderOptions {
            max_ray_depth: 0,
            ..o.clone()
        },
    );
    assert_eq!(none.pixels, flat.pixels);
}

#[test]
fn dielectric_mirror_reflects_by_fresnel_only() {
    let o = opts(96, 72);
    let mut glossy = diffuse_material([0.05; 3], 0.0);
    glossy.metallic = 0.0;
    let scene = mirror_scene(glossy);
    let d = ray(&scene, &o);
    let m = ray(&mirror_scene(metal(0.0)), &o);
    let xy = project(&scene, &o, [0.0, 0.0, -1.0]);
    let (pd, pm) = (px(&d, xy), px(&m, xy));
    // Smooth black plastic still shows a (weaker, Fresnel-weighted)
    // red reflection.
    assert!(pd[0] > pd[1] + 20, "{pd:?}");
    assert!(pd[0] < pm[0], "{pd:?} vs {pm:?}");
}

// ---------------------------------------------------------------------
// Refraction.
// ---------------------------------------------------------------------

/// Unlit wall at `z = -3`: red for `x < 0`, green for `x > 0`; a unit
/// sphere of material `ball` at the origin; camera on `+Z`.
fn lens_scene(ball: Material) -> Scene3D {
    let mut scene = Scene3D::new();
    for (x0, x1, c) in [(-4.0, 0.0, [1.0, 0.0, 0.0]), (0.0, 4.0, [0.0, 1.0, 0.0])] {
        add(
            &mut scene,
            quad(
                [
                    [x0, -4.0, -3.0],
                    [x1, -4.0, -3.0],
                    [x1, 4.0, -3.0],
                    [x0, 4.0, -3.0],
                ],
                [1.0, 1.0],
            ),
            unlit(c),
        );
    }
    add(&mut scene, uv_sphere(1.0, 64, 32), ball);
    add_camera(&mut scene, [0.0, 0.0, 4.0], [0.0; 3], 0.7);
    scene
}

fn glass(thickness: f32) -> Material {
    let mut m = Material::new();
    m.base_color = [1.0; 4];
    m.metallic = 0.0;
    m.roughness = 0.0;
    m.ext.transmission = Some(Transmission {
        factor: 1.0,
        factor_texture: None,
    });
    m.ext.ior = Some(1.5);
    if thickness > 0.0 {
        m.ext.volume = Some(Volume {
            thickness,
            thickness_texture: None,
            attenuation_distance: None,
            attenuation_color: [1.0; 3],
        });
    }
    m
}

#[test]
fn glass_sphere_inverts_the_background() {
    let o = RenderOptions {
        ambient: 0.0,
        ..opts(128, 128)
    };
    let solid = lens_scene(glass(2.0));
    let img = ray(&solid, &o);
    // Left part of the ball: a ball lens (f ≈ 1.5) flips the wall,
    // so the green right half shows up on the left.
    let left = px(&img, project(&solid, &o, [-0.4, 0.0, 0.9]));
    let right = px(&img, project(&solid, &o, [0.4, 0.0, 0.9]));
    assert!(left[1] > 2 * left[0].max(10), "left {left:?}");
    assert!(right[0] > 2 * right[1].max(10), "right {right:?}");
    // A thin-walled ball (no volume) passes rays straight through:
    // no inversion.
    let thin = lens_scene(glass(0.0));
    let img = ray(&thin, &o);
    let left = px(&img, project(&thin, &o, [-0.4, 0.0, 0.9]));
    assert!(left[0] > 2 * left[1].max(10), "thin left {left:?}");
    // Without bounces the glass is shaded as an opaque (white,
    // diffuse-lit) surface: neither wall colour shows.
    let flat = ray(
        &solid,
        &RenderOptions {
            max_ray_depth: 0,
            ..o.clone()
        },
    );
    let c = px(&flat, project(&solid, &o, [0.0, 0.0, 1.0]));
    assert!(c[0] == c[1] && c[1] == c[2], "{c:?}");
}

#[test]
fn volume_attenuation_tints_and_ior_one_is_invisible() {
    let o = RenderOptions {
        ambient: 0.0,
        ..opts(96, 96)
    };
    let clear = lens_scene(glass(2.0));
    let mut tinted_mat = glass(2.0);
    tinted_mat.ext.volume = Some(Volume {
        thickness: 2.0,
        thickness_texture: None,
        attenuation_distance: Some(0.5),
        attenuation_color: [0.2, 0.2, 1.0],
    });
    let tinted = lens_scene(tinted_mat);
    let xy = project(&clear, &o, [-0.4, 0.0, 0.9]);
    let (a, b) = (px(&ray(&clear, &o), xy), px(&ray(&tinted, &o), xy));
    assert!(b[1] + 60 < a[1], "absorption dims green: {a:?} → {b:?}");
    // Index-matched (ior 1), untinted thin glass with F0 = 0 is
    // invisible apart from its zero-strength specular highlight.
    let mut m = glass(0.0);
    m.ext.ior = Some(1.0);
    let ghost = lens_scene(m);
    let none = {
        let mut s = lens_scene(Material::new());
        s.meshes[2].primitives.clear();
        s
    };
    let o = RenderOptions {
        use_scene_lights: false,
        light: oxideav_render::LightSpec {
            intensity: 0.0,
            ..Default::default()
        },
        ..o
    };
    let (g, n) = (ray(&ghost, &o), ray(&none, &o));
    assert!(mean_abs_error(&g, &n) < 0.5, "{}", mean_abs_error(&g, &n));
}

#[test]
fn deep_recursion_terminates_between_parallel_mirrors() {
    let mut scene = Scene3D::new();
    for z in [-1.0f32, 1.0] {
        let (a, b) = if z < 0.0 { (-2.0, 2.0) } else { (2.0, -2.0) };
        add(
            &mut scene,
            quad(
                [[a, -2.0, z], [b, -2.0, z], [b, 2.0, z], [a, 2.0, z]],
                [1.0, 1.0],
            ),
            metal(0.0),
        );
    }
    add_camera(&mut scene, [0.0, 0.0, 0.5], [0.0, 0.0, -1.0], 0.9);
    for depth in [0, 4, 16, 1000] {
        let img = ray(
            &scene,
            &RenderOptions {
                max_ray_depth: depth,
                shadows: true,
                ..opts(32, 32)
            },
        );
        assert_eq!(img.width, 32);
    }
}

#[test]
fn raycast_pbr_is_deterministic_with_aa_and_shadows() {
    let scene = cornell_box();
    let o = RenderOptions {
        shadows: true,
        aa: 2,
        ..opts(150, 101)
    };
    let a = ray(&scene, &o);
    let b = ray(&scene, &o);
    assert_eq!(a.pixels, b.pixels);
}

#[cfg(feature = "registry")]
#[test]
fn texture_resolver_reaches_the_raycaster() {
    use oxideav_render::Renderer;
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
    let plain = ray(&scene, &o);
    let tl = project(&scene, &o, [-0.75, 0.75, 0.0]);
    assert_eq!(px(&plain, tl), [255, 255, 255, 255]);
    let mut ctx = oxideav_core::RuntimeContext::new();
    oxideav_png::register(&mut ctx);
    let resolver: Arc<dyn oxideav_render::TextureResolver> =
        Arc::new(oxideav_render::RegistryTextureResolver::new(Arc::new(ctx)));
    let mut r = oxideav_render::RaycastRenderer::with_texture_resolver(resolver.clone());
    let img = r.render(&scene, &o).unwrap();
    let mut dynr = make_renderer(RenderBackend::Raycast).unwrap();
    dynr.set_texture_resolver(resolver);
    assert_eq!(dynr.render(&scene, &o).unwrap().pixels, img.pixels);
    assert_eq!(px(&img, tl), [255, 0, 0, 255]);
    assert_eq!(
        px(&img, project(&scene, &o, [-0.25, 0.75, 0.0])),
        [0, 0, 255, 255]
    );
}
