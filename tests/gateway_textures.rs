//! Textures and environment maps resolved through the `oxideav-image`
//! gateway (`ImageTextureResolver`): PNG, JPEG and OpenEXR sources on
//! every backend, float sources staying linear, cross-backend parity
//! with the raw-texture reference, and the deprecated wrapper.

#![cfg(feature = "registry")]

use std::sync::Arc;

use oxideav_core::{PixelFormat, RuntimeContext};
use oxideav_image::{encode, Image, SaveOptions};
use oxideav_mesh3d::{MagFilter, MinFilter, Sampler, Scene3D, Texture};
use oxideav_render::testscenes::*;
use oxideav_render::texture::{DecodedPixels, TextureCache};
use oxideav_render::{
    make_renderer, BackgroundColor, Camera, DepthRange, EnvironmentMap, ImageTextureResolver,
    LightSpec, PathTraceOptions, PathTracer, PreparedScene, RenderBackend, RenderOptions,
    RgbaImage, ShadingMode, TextureResolver,
};

const BG: [u8; 4] = [0, 0, 0, 255];

fn ctx() -> Arc<RuntimeContext> {
    let mut ctx = RuntimeContext::new();
    oxideav_png::register(&mut ctx);
    oxideav_mjpeg::register(&mut ctx);
    oxideav_openexr::register(&mut ctx);
    Arc::new(ctx)
}

fn resolver() -> Arc<ImageTextureResolver> {
    Arc::new(ImageTextureResolver::new(ctx()))
}

fn opts(w: u32, h: u32) -> RenderOptions {
    RenderOptions {
        width: w,
        height: h,
        shading: ShadingMode::Pbr,
        scene_camera: Some(0),
        background: BackgroundColor(BG),
        max_ray_depth: 0,
        ..RenderOptions::default()
    }
}

fn nearest() -> Sampler {
    Sampler::default_sampler()
        .with_mag_filter(MagFilter::Nearest)
        .with_min_filter(MinFilter::Nearest)
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

/// Fraction of pixels differing by more than `tol` in any channel.
fn outlier_fraction(a: &RgbaImage, b: &RgbaImage, tol: u8) -> f64 {
    let n = a
        .pixels_rgba()
        .zip(b.pixels_rgba())
        .filter(|(p, q)| (0..4).any(|k| p[k].abs_diff(q[k]) > tol))
        .count();
    n as f64 / (a.width * a.height) as f64
}

/// The 4×4 red / blue checker of `testscenes::textured_quad`, encoded
/// by the gateway into `format` (`"png"`, `"jpeg"`, `"exr"`).
fn checker_encoded(ctx: &RuntimeContext, format: &str) -> (Vec<u8>, &'static str) {
    let rgba = checker_rgba8(4, 4, [255, 0, 0, 255], [0, 0, 255, 255]);
    let (img, mime) = match format {
        "png" => (Image::from_rgba8(4, 4, rgba).unwrap(), "image/png"),
        "jpeg" => {
            // Lossy 4:2:0 would smear a 1-texel checker; the same 4×4
            // cells on a 64×64 image put every 16×16 chroma block
            // inside one flat cell.
            let rgb: Vec<u8> = checker_rgba8(64, 4, [255, 0, 0, 255], [0, 0, 255, 255])
                .chunks_exact(4)
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect();
            (Image::from_rgb8(64, 64, rgb).unwrap(), "image/jpeg")
        }
        "exr" => {
            // Scene-linear float: pure red / blue at 1.0, opaque.
            let mut f = Vec::with_capacity(4 * 4 * 16);
            for p in rgba.chunks_exact(4) {
                for c in p {
                    f.extend_from_slice(&(*c as f32 / 255.0).to_le_bytes());
                }
            }
            (
                Image::from_raw(4, 4, PixelFormat::RgbaF32Le, f).unwrap(),
                "image/x-exr",
            )
        }
        other => panic!("unknown fixture format {other}"),
    };
    let bytes = encode(ctx, &img, format, &SaveOptions::default())
        .unwrap_or_else(|e| panic!("encode {format}: {e}"));
    (bytes, mime)
}

/// A lit, grey, rough floor (`[-2, 2]²` at `y = 0`, normal `+Y`) seen
/// from above — no lights, so only a sky can illuminate it.
fn diffuse_floor() -> Scene3D {
    use oxideav_mesh3d::{Mesh, Node, Transform};
    let mut scene = Scene3D::new();
    let mat = scene.add_material(diffuse_material([0.8, 0.8, 0.8], 0.8));
    let mut p = quad(
        [
            [-2.0, 0.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 0.0, -2.0],
            [-2.0, 0.0, -2.0],
        ],
        [1.0, 1.0],
    );
    p.material = Some(mat);
    let m = scene.add_mesh(Mesh::new(None).with_primitive(p));
    let n = scene.add_node(
        Node::new()
            .with_mesh(m)
            .with_transform(Transform::identity()),
    );
    scene.add_root(n);
    add_camera(&mut scene, [0.0, 3.0, 3.0], [0.0; 3], 0.9);
    scene
}

fn quad_with(bytes: Vec<u8>, mime: &str) -> Scene3D {
    let mut scene = textured_quad(nearest());
    let mut tex = Texture::from_encoded(mime, bytes);
    tex.sampler = nearest();
    scene.textures[0] = tex;
    scene
}

fn render_with(backend: RenderBackend, scene: &Scene3D, o: &RenderOptions) -> RgbaImage {
    let mut r = make_renderer(backend).unwrap();
    r.set_texture_resolver(resolver());
    r.render(scene, o).unwrap()
}

#[test]
fn png_jpeg_and_exr_textures_resolve_on_every_backend() {
    let ctx = ctx();
    let o = opts(96, 96);
    let reference = textured_quad(nearest());
    let tl = project(&reference, &o, [-0.75, 0.75, 0.0]);
    let tr = project(&reference, &o, [-0.25, 0.75, 0.0]);
    for format in ["png", "jpeg", "exr"] {
        let (bytes, mime) = checker_encoded(&ctx, format);
        let scene = quad_with(bytes, mime);
        // Without a resolver the texture is absent: white quad.
        let plain = make_renderer(RenderBackend::Raycast)
            .unwrap()
            .render(&scene, &o)
            .unwrap();
        assert_eq!(
            px(&plain, tl),
            [255, 255, 255, 255],
            "{format}: no resolver"
        );
        for backend in [
            RenderBackend::Scanline,
            RenderBackend::Raycast,
            RenderBackend::PathTrace,
        ] {
            let o = RenderOptions {
                path_trace: PathTraceOptions {
                    samples_per_pixel: 4,
                    ..PathTraceOptions::default()
                },
                ..o.clone()
            };
            let img = render_with(backend, &scene, &o);
            // JPEG chroma subsampling shifts the saturated cells a
            // little; PNG / EXR are exact.
            let tol = if format == "jpeg" { 24 } else { 0 };
            assert!(
                close(px(&img, tl), [255, 0, 0, 255], tol),
                "{format} / {backend:?}: top-left cell {:?}",
                px(&img, tl)
            );
            assert!(
                close(px(&img, tr), [0, 0, 255, 255], tol),
                "{format} / {backend:?}: second cell {:?}",
                px(&img, tr)
            );
        }
    }
}

#[test]
fn gateway_textures_keep_cross_backend_parity() {
    // The same scene rendered from the raw-RGBA8 reference texture
    // and from its PNG / EXR encodings must agree pixel for pixel on
    // scanline, and the three backends must stay within the README's
    // parity tolerance (coverage differences only).
    let ctx = ctx();
    let o = opts(128, 96);
    let reference = textured_quad(nearest());
    let ref_img = make_renderer(RenderBackend::Scanline)
        .unwrap()
        .render(&reference, &o)
        .unwrap();
    for format in ["png", "exr"] {
        let (bytes, mime) = checker_encoded(&ctx, format);
        let scene = quad_with(bytes, mime);
        let scan = render_with(RenderBackend::Scanline, &scene, &o);
        assert_eq!(scan.pixels, ref_img.pixels, "{format}: scanline vs raw");
        let ray = render_with(RenderBackend::Raycast, &scene, &o);
        let outliers = outlier_fraction(&scan, &ray, 1);
        assert!(outliers < 0.01, "{format}: raycast outliers {outliers}");
        let pt = render_with(
            RenderBackend::PathTrace,
            &scene,
            &RenderOptions {
                path_trace: PathTraceOptions {
                    samples_per_pixel: 16,
                    ..PathTraceOptions::default()
                },
                ..o.clone()
            },
        );
        // Unlit textured quad: the path tracer's only noise is the
        // box pixel filter straddling the quad border and the six
        // texel boundaries (≈ 6 % of the frame at this size), which
        // the centre-sampled rasteriser resolves without blending.
        let outliers = outlier_fraction(&scan, &pt, 1);
        assert!(outliers < 0.08, "{format}: pathtrace outliers {outliers}");
    }
}

#[test]
fn float_sources_decode_linear_and_integer_sources_rgba8() {
    let ctx = ctx();
    let r = ImageTextureResolver::new(ctx.clone());
    // EXR with radiance above 1 survives as f32 (no clamp, no sRGB).
    let mut f = Vec::new();
    for v in [[4.0f32, 0.5, 0.0, 1.0], [0.0, 0.0, 16.0, 0.25]] {
        for c in v {
            f.extend_from_slice(&c.to_le_bytes());
        }
    }
    let img = Image::from_raw(2, 1, PixelFormat::RgbaF32Le, f).unwrap();
    let exr = encode(&ctx, &img, "exr", &SaveOptions::default()).unwrap();
    let dec = r.decode_image(&exr, Some("image/x-exr")).unwrap();
    assert_eq!((dec.width, dec.height), (2, 1));
    match &dec.pixels {
        DecodedPixels::RgbaF32(p) => {
            assert_eq!(p[..4], [4.0, 0.5, 0.0, 1.0]);
            assert_eq!(p[4..], [0.0, 0.0, 16.0, 0.25]);
        }
        other => panic!("EXR must decode to RgbaF32, got {other:?}"),
    }
    let hdr = dec.to_hdr();
    assert_eq!(hdr.pixel(1, 0), Some([0.0, 0.0, 16.0, 0.25]));
    // Without a MIME type the probe still finds the format by magic.
    assert!(r.decode(&exr, None).is_some());

    // PNG stays RGBA8 (the sampler applies the slot's colour space).
    let png = encode(
        &ctx,
        &Image::from_rgb8(1, 1, vec![128, 64, 32]).unwrap(),
        "png",
        &SaveOptions::default(),
    )
    .unwrap();
    let dec = r.decode_image(&png, Some("image/png")).unwrap();
    assert_eq!(dec.pixels, DecodedPixels::Rgba8(vec![128, 64, 32, 255]));
    // 16-bit PNG rounds to 8 bits.
    let deep = encode(
        &ctx,
        &Image::from_raw(
            1,
            1,
            PixelFormat::Rgba64Le,
            vec![0xff, 0xff, 0x00, 0x80, 0x00, 0x00, 0xff, 0xff],
        )
        .unwrap(),
        "png",
        &SaveOptions::default(),
    )
    .unwrap();
    let dec = r.decode_image(&deep, Some("image/png")).unwrap();
    assert_eq!(dec.pixels, DecodedPixels::Rgba8(vec![255, 128, 0, 255]));
    // JPEG (YCbCr 4:2:0) reaches RGBA8 through the gateway's colour
    // signal; alpha opaque.
    let jpg = encode(
        &ctx,
        &Image::from_rgb8(8, 8, vec![200; 8 * 8 * 3]).unwrap(),
        "jpeg",
        &SaveOptions::default(),
    )
    .unwrap();
    let dec = r.decode_image(&jpg, Some("image/jpeg")).unwrap();
    match &dec.pixels {
        DecodedPixels::Rgba8(p) => {
            assert_eq!(p.len(), 8 * 8 * 4);
            assert!(p
                .chunks_exact(4)
                .all(|q| q[3] == 255 && q[0].abs_diff(200) <= 3));
        }
        other => panic!("JPEG must decode to Rgba8, got {other:?}"),
    }
}

#[test]
fn gateway_errors_are_typed_and_the_cache_memoises_them() {
    let ctx = ctx();
    let r = ImageTextureResolver::new(ctx.clone()).with_max_pixels(4);
    // Garbage: no container probes it by magic; with a PNG hint the
    // container may be tried and fail — either way a typed error.
    assert!(matches!(
        r.decode_image(b"not an image at all", None),
        Err(oxideav_image::ImageError::UnknownFormat(_))
    ));
    assert!(r
        .decode_image(b"not an image at all", Some("image/png"))
        .is_err());
    assert!(r
        .decode(b"not an image at all", Some("image/png"))
        .is_none());
    // Over the pixel budget: refused before decoding.
    let big = encode(
        &ctx,
        &Image::from_rgb8(4, 4, vec![0; 48]).unwrap(),
        "png",
        &SaveOptions::default(),
    )
    .unwrap();
    assert!(matches!(
        r.decode_image(&big, Some("image/png")),
        Err(oxideav_image::ImageError::LimitExceeded(_))
    ));
    assert!(r.decode(&big, Some("image/png")).is_none());
    // The cache records the failure once and never re-decodes.
    let mut cache = TextureCache::new(Arc::new(r));
    let tex = Texture::from_encoded("image/png", big);
    assert!(cache.prepare(&tex).is_none());
    assert!(cache.prepare(&tex).is_none());
    assert_eq!(cache.len(), 1);
}

#[test]
fn external_uris_resolve_against_the_base_dir() {
    let ctx = ctx();
    let dir = std::env::temp_dir().join(format!(
        "oxideav-render-gateway-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let (png, _) = checker_encoded(&ctx, "png");
    std::fs::write(dir.join("checker.png"), &png).unwrap();
    let r = ImageTextureResolver::new(ctx).with_base_dir(&dir);
    assert_eq!(r.base_dir(), Some(dir.as_path()));
    let mut scene = textured_quad(nearest());
    let mut tex = Texture::from_uri("checker.png");
    tex.sampler = nearest();
    scene.textures[0] = tex;
    let o = opts(64, 64);
    let mut renderer = make_renderer(RenderBackend::Scanline).unwrap();
    renderer.set_texture_resolver(Arc::new(r.clone()));
    let img = renderer.render(&scene, &o).unwrap();
    assert_eq!(
        px(&img, project(&scene, &o, [-0.75, 0.75, 0.0])),
        [255, 0, 0, 255]
    );
    // `open_image` takes the same base directory.
    let dec = r.open_image("checker.png").unwrap();
    assert_eq!((dec.width, dec.height), (4, 4));
    assert!(r.fetch_external("http://example.invalid/x.png").is_none());
    assert!(r.fetch_external("data:image/png;base64,AAAA").is_none());
    assert!(r.fetch_external("missing.png").is_none());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn exr_environment_map_lights_the_path_tracer() {
    // A 4×2 equirectangular EXR: the top row (+Y) emits 8.0 white, the
    // bottom row is black. A diffuse floor facing +Y under it must be
    // lit (and lit more than under a constant sky of the row mean).
    let ctx = ctx();
    let mut f = Vec::new();
    for y in 0..2 {
        for _x in 0..4 {
            let v = if y == 0 { 8.0f32 } else { 0.0 };
            for c in [v, v, v, 1.0] {
                f.extend_from_slice(&c.to_le_bytes());
            }
        }
    }
    let exr = encode(
        &ctx,
        &Image::from_raw(4, 2, PixelFormat::RgbaF32Le, f).unwrap(),
        "exr",
        &SaveOptions::default(),
    )
    .unwrap();
    let r = ImageTextureResolver::new(ctx);
    let sky = r.decode_image(&exr, Some("image/x-exr")).unwrap().to_hdr();
    assert_eq!(sky.pixel(0, 0), Some([8.0, 8.0, 8.0, 1.0]));
    let env = Arc::new(EnvironmentMap::new(Arc::new(sky), 1.0));
    assert!(!env.is_uniform());

    let scene = diffuse_floor();
    let o = RenderOptions {
        use_scene_lights: true,
        // No scene light and a zero fallback light: only a sky can
        // light the floor.
        light: LightSpec {
            intensity: 0.0,
            ..LightSpec::default_light()
        },
        ambient: 0.0,
        path_trace: PathTraceOptions {
            samples_per_pixel: 32,
            max_bounces: 2,
            ..PathTraceOptions::default()
        },
        ..opts(48, 48)
    };
    let mut dark = PathTracer::with_texture_resolver(Arc::new(r.clone()));
    dark.sync(&scene, &o);
    dark.refine(32);
    let mut lit = PathTracer::with_texture_resolver(Arc::new(r));
    lit.set_environment(Some(env));
    lit.sync(&scene, &o);
    lit.refine(32);
    let dark_hdr = dark.hdr();
    let lit_hdr = lit.hdr();
    let mean = |h: &oxideav_render::HdrImage| {
        let mut s = 0.0f64;
        let mut n = 0usize;
        for p in h.pixels.chunks_exact(4) {
            if p[3] > 0.5 {
                s += (p[0] + p[1] + p[2]) as f64 / 3.0;
                n += 1;
            }
        }
        s / n.max(1) as f64
    };
    let (d, l) = (mean(&dark_hdr), mean(&lit_hdr));
    assert!(d < 1.0e-3, "no sky, no light: {d}");
    // Floor albedo ≤ 1 under a hemisphere of radiance 8 → at most 8.
    assert!(l > 0.5 && l < 8.0, "lit by the EXR sky: {l}");
}

#[test]
#[allow(deprecated)]
fn deprecated_registry_resolver_still_decodes() {
    let ctx = ctx();
    let (png, mime) = checker_encoded(&ctx, "png");
    let old = oxideav_render::RegistryTextureResolver::new(ctx);
    let a = old.decode(&png, Some(mime)).unwrap();
    let b = old.clone().into_inner().decode(&png, Some(mime)).unwrap();
    assert_eq!(a, b);
    assert_eq!((a.width, a.height), (4, 4));
}
