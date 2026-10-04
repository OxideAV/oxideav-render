//! Scanline software rasteriser — the backend behind
//! [`crate::RenderBackend::Scanline`].
//!
//! Pure-Rust, zero external rendering dependencies. The scene is
//! flattened by the shared [`crate::prepare`] layer (animation,
//! morphs, skinning, world-space attributes, materials, textures,
//! lights), framed by the shared [`crate::camera::Camera`], and drawn
//! in passes by the [`crate::raster`] core:
//!
//! 1. **Shadow maps** (optional, [`ShadingMode::Pbr`] +
//!    [`RenderOptions::shadows`]) — one depth map per directional /
//!    spot light (Williams, "Casting Curved Shadows on Curved
//!    Surfaces", SIGGRAPH 1978), sampled with 3×3 bilinear
//!    percentage-closer filtering (Reeves, Salesin, Cook, "Rendering
//!    Antialiased Shadows with Depth Maps", SIGGRAPH 1987) and a
//!    normal-offset + constant bias.
//! 2. **Visibility pass** — opaque and alpha-masked triangles (MASK
//!    cutoff evaluated per fragment), then lines / points, into a
//!    depth + primitive-id + barycentric buffer (clipped,
//!    perspective-correct, top-left fill rule, back-face culling for
//!    single-sided PBR materials).
//! 3. **Resolve** — every visible pixel is shaded exactly once in
//!    scene-linear `f32`.
//! 4. **Blend pass** ([`ShadingMode::Pbr`]) — `BLEND` triangles sorted
//!    back-to-front by view depth, depth-tested against the opaque
//!    result without depth writes, shaded forward and composited with
//!    the straight-alpha *over* operator (Porter & Duff, "Compositing
//!    Digital Images", SIGGRAPH 1984).
//! 5. **Display** — exposure, [`crate::ToneMap`], SSAA box filter
//!    (premultiplied), sRGB encode. Background samples bypass tone
//!    mapping so the requested background bytes come back verbatim.
//!
//! Shading modes: `Flat` / `Gouraud` / `Phong` / `Wireframe` keep the
//! historical model (material base-colour factor, the options'
//! directional light, constant 0.2 ambient — shared with the raycast
//! backend via [`crate::shade`]); `NormalDebug` / `DepthDebug` are
//! colour-keyed visualisers; `Pbr` is the glTF 2.0 metallic-roughness
//! model of [`crate::brdf`] with textures, normal / occlusion /
//! emissive maps, vertex colours, unlit, punctual lights, alpha modes
//! and shadows.
//!
//! Clean-room policy: every algorithm has a published source, cited at
//! its implementation. No reference renderer source code was consulted.

use oxideav_mesh3d::{AlphaMode, Scene3D};

use crate::brdf::{self, BrdfParams};
use crate::camera::{look_at, orthographic, perspective, Camera};
use crate::hdr::{linear_to_srgb_byte, srgb_u8_lut, HdrImage};
use crate::image::RgbaImage;
use crate::math::{mat4_mul, mat4_mul_vec4, vec3_cross, vec3_dot, vec3_normalise, vec3_sub};
use crate::options::{Projection, RenderOptions, ShadingMode};
use crate::prepare::{
    DrawItem, DrawTopology, LightKind, PrepareOptions, PreparedLight, PreparedMaterial,
    PreparedScene, TextureBinding,
};
use crate::raster::{
    bin_tris, par_bands, raster_visibility, setup_line, setup_point, setup_triangle, Cull,
    ScreenLine, ScreenTri, VisPixel, LINE_FLAG, NONE,
};
use crate::shade::{build_light, shade_pixel, DirLight};
use crate::texture::{ColorSpace, TextureCache};

// ---------------------------------------------------------------------
// Public entry points.
// ---------------------------------------------------------------------

/// Render `scene` into a packed RGBA8 image per `opts`, resolving only
/// built-in raw textures.
#[cfg(test)]
pub(crate) fn render_scene(scene: &Scene3D, opts: &RenderOptions) -> RgbaImage {
    render_with_cache(scene, opts, &mut TextureCache::default())
}

/// [`render_scene`] decoding textures through `cache`.
pub(crate) fn render_with_cache(
    scene: &Scene3D,
    opts: &RenderOptions,
    cache: &mut TextureCache,
) -> RgbaImage {
    let frame = render_frame(scene, opts, cache);
    frame.to_rgba8(opts)
}

/// Scene-linear output (pre exposure / tone map).
pub(crate) fn render_hdr_with_cache(
    scene: &Scene3D,
    opts: &RenderOptions,
    cache: &mut TextureCache,
) -> HdrImage {
    let frame = render_frame(scene, opts, cache);
    frame.to_hdr()
}

// ---------------------------------------------------------------------
// Frame result + display transform.
// ---------------------------------------------------------------------

/// One shaded sample: scene-linear straight-alpha colour, and whether
/// any geometry covers it.
///
/// Shared with the ray-based backends, which fill a [`Frame`] the same
/// way so every CPU backend has one resolve contract.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Sample {
    pub(crate) c: [f32; 4],
    pub(crate) covered: bool,
}

/// A render-resolution sample buffer plus the display transform
/// (exposure, tone map, SSAA box filter, sRGB encode) shared by every
/// CPU backend.
pub(crate) struct Frame {
    pub(crate) out_w: u32,
    pub(crate) out_h: u32,
    pub(crate) aa: u32,
    /// `out_w·aa × out_h·aa` samples, row-major.
    pub(crate) samples: Vec<Sample>,
    pub(crate) background: [u8; 4],
    /// Debug visualisers: display transform is a plain clamp.
    pub(crate) display_referred: bool,
}

impl Frame {
    pub(crate) fn bg_linear(&self) -> [f32; 4] {
        let lut = srgb_u8_lut();
        let b = self.background;
        [
            lut[b[0] as usize],
            lut[b[1] as usize],
            lut[b[2] as usize],
            b[3] as f32 / 255.0,
        ]
    }

    /// Box-filter the `aa × aa` sample block of every output pixel,
    /// mapping each sample through `map` first; averages premultiplied
    /// by alpha (straight average when the block is fully
    /// transparent).
    fn resolve(&self, map: impl Fn(&Sample) -> [f32; 4] + Sync) -> Vec<[f32; 4]> {
        let aa = self.aa as usize;
        let ow = self.out_w as usize;
        let rw = ow * aa;
        let n = (aa * aa) as f32;
        let mut out = vec![[0.0f32; 4]; ow * self.out_h as usize];
        par_bands(&mut out, ow, |_, y0, rows| {
            for (i, o) in rows.iter_mut().enumerate() {
                let (ox, oy) = (i % ow, y0 + i / ow);
                let mut pre = [0.0f32; 3];
                let mut straight = [0.0f32; 3];
                let mut a_sum = 0.0f32;
                for j in 0..aa {
                    for i in 0..aa {
                        let c = map(&self.samples[(oy * aa + j) * rw + ox * aa + i]);
                        for k in 0..3 {
                            pre[k] += c[k] * c[3];
                            straight[k] += c[k];
                        }
                        a_sum += c[3];
                    }
                }
                *o = if a_sum > 0.0 {
                    [pre[0] / a_sum, pre[1] / a_sum, pre[2] / a_sum, a_sum / n]
                } else {
                    [straight[0] / n, straight[1] / n, straight[2] / n, 0.0]
                };
            }
        });
        out
    }

    pub(crate) fn to_rgba8(&self, opts: &RenderOptions) -> RgbaImage {
        let bg = self.bg_linear();
        let (tm, exposure) = if self.display_referred {
            (crate::hdr::ToneMap::Clamp, 1.0)
        } else {
            (opts.tone_map, opts.exposure)
        };
        let px = self.resolve(|s| {
            if !s.covered {
                return bg;
            }
            let m = tm.apply([s.c[0] * exposure, s.c[1] * exposure, s.c[2] * exposure]);
            [m[0], m[1], m[2], s.c[3].clamp(0.0, 1.0)]
        });
        let mut bytes = vec![[0u8; 4]; px.len()];
        par_bands(&mut bytes, self.out_w as usize, |band, _, rows| {
            let start = band * crate::raster::BAND_ROWS * self.out_w as usize;
            for (o, c) in rows.iter_mut().zip(&px[start..]) {
                *o = [
                    linear_to_srgb_byte(c[0]),
                    linear_to_srgb_byte(c[1]),
                    linear_to_srgb_byte(c[2]),
                    (c[3].clamp(0.0, 1.0) * 255.0).round() as u8,
                ];
            }
        });
        let pixels: Vec<u8> = bytes.into_iter().flatten().collect();
        RgbaImage {
            width: self.out_w,
            height: self.out_h,
            stride: self.out_w as usize * 4,
            pixels,
        }
    }

    pub(crate) fn to_hdr(&self) -> HdrImage {
        let bg = self.bg_linear();
        let px = self.resolve(|s| if s.covered { s.c } else { bg });
        HdrImage {
            width: self.out_w,
            height: self.out_h,
            pixels: px.into_iter().flatten().collect(),
        }
    }
}

// ---------------------------------------------------------------------
// Frame rendering.
// ---------------------------------------------------------------------

/// Global triangle reference: item index + triangle index in the item.
#[derive(Debug, Clone, Copy)]
struct TriRef {
    item: u32,
    tri: u32,
}

/// Line / point reference: item + the two corner vertex indices.
#[derive(Debug, Clone, Copy)]
struct LineRef {
    item: u32,
    a: u32,
    b: u32,
}

fn render_frame(scene: &Scene3D, opts: &RenderOptions, cache: &mut TextureCache) -> Frame {
    let width = opts.width.max(1);
    let height = opts.height.max(1);
    let aa = opts.aa.clamp(1, 8);
    let rw = width.saturating_mul(aa).max(1);
    let rh = height.saturating_mul(aa).max(1);
    let mode = opts.shading;
    let pbr = mode == ShadingMode::Pbr;

    let prepared = PreparedScene::build(scene, &PrepareOptions::from_render_options(opts), cache);
    let camera = Camera::resolve(&prepared, opts, rw, rh);
    let vp = mat4_mul(camera.proj, camera.view);

    // ---- Collect primitives.
    let mut tri_refs: Vec<TriRef> = Vec::new();
    let mut screen_tris: Vec<ScreenTri> = Vec::new();
    let mut blend: Vec<(f32, TriRef)> = Vec::new();
    let mut line_refs: Vec<LineRef> = Vec::new();
    let mut screen_lines: Vec<ScreenLine> = Vec::new();
    let clip = |p: [f32; 3]| mat4_mul_vec4(&vp, [p[0], p[1], p[2], 1.0]);

    for (ii, item) in prepared.items.iter().enumerate() {
        let mat = &prepared.materials[item.material];
        match item.topology {
            DrawTopology::Triangles if mode == ShadingMode::Wireframe => {
                for t in 0..item.positions.len() / 3 {
                    let b = 3 * t as u32;
                    for (a, c) in [(b, b + 1), (b + 1, b + 2), (b + 2, b)] {
                        push_line(
                            &mut line_refs,
                            &mut screen_lines,
                            LineRef {
                                item: ii as u32,
                                a,
                                b: c,
                            },
                            clip(item.positions[a as usize]),
                            clip(item.positions[c as usize]),
                            rw,
                            rh,
                        );
                    }
                }
            }
            DrawTopology::Triangles => {
                let cull = if pbr && !mat.double_sided {
                    Cull::Back
                } else {
                    Cull::None
                };
                let masked = pbr && matches!(mat.alpha_mode, AlphaMode::Mask { .. });
                for (t, v) in item.positions.chunks_exact(3).enumerate() {
                    let r = TriRef {
                        item: ii as u32,
                        tri: t as u32,
                    };
                    if pbr && mat.alpha_mode == AlphaMode::Blend {
                        let c = [
                            (v[0][0] + v[1][0] + v[2][0]) / 3.0,
                            (v[0][1] + v[1][1] + v[2][1]) / 3.0,
                            (v[0][2] + v[1][2] + v[2][2]) / 3.0,
                        ];
                        blend.push((camera.view_depth(c), r));
                        continue;
                    }
                    let id = tri_refs.len() as u32;
                    tri_refs.push(r);
                    setup_triangle(
                        [clip(v[0]), clip(v[1]), clip(v[2])],
                        id,
                        cull,
                        masked,
                        rw,
                        rh,
                        &mut screen_tris,
                    );
                }
            }
            DrawTopology::Lines => {
                for s in 0..item.positions.len() / 2 {
                    let (a, b) = (2 * s as u32, 2 * s as u32 + 1);
                    push_line(
                        &mut line_refs,
                        &mut screen_lines,
                        LineRef {
                            item: ii as u32,
                            a,
                            b,
                        },
                        clip(item.positions[a as usize]),
                        clip(item.positions[b as usize]),
                        rw,
                        rh,
                    );
                }
            }
            DrawTopology::Points => {
                for (i, p) in item.positions.iter().enumerate() {
                    if let Some(sl) = setup_point(clip(*p), rw, rh) {
                        line_refs.push(LineRef {
                            item: ii as u32,
                            a: i as u32,
                            b: i as u32,
                        });
                        screen_lines.push(sl);
                    }
                }
            }
        }
    }

    let ctx = ShadeCtx {
        prepared: &prepared,
        camera: &camera,
        mode,
        legacy_light: build_light(opts.light),
        ambient: opts.ambient,
        tri_refs: &tri_refs,
        shadows: if pbr && opts.shadows {
            build_shadow_maps(&prepared, opts.shadow_map_size.clamp(16, 8192))
        } else {
            Vec::new()
        },
    };

    // ---- Visibility pass.
    let mask_test = |t: &ScreenTri, b: [f32; 3], x: i32, y: i32| -> bool {
        let r = tri_refs[t.prim as usize];
        let item = &prepared.items[r.item as usize];
        let mat = &prepared.materials[item.material];
        let AlphaMode::Mask { cutoff } = mat.alpha_mode else {
            return true;
        };
        let g = Grads::of(t, b, x, y);
        ctx.alpha(item, mat, 3 * r.tri as usize, b, &g) >= cutoff
    };
    let vis = raster_visibility(&screen_tris, &screen_lines, rw, rh, &mask_test);

    // ---- Resolve.
    let w = rw as usize;
    let mut samples = vec![
        Sample {
            c: [0.0; 4],
            covered: false,
        };
        w * rh as usize
    ];
    par_bands(&mut samples, w, |_, y0, rows| {
        for (i, s) in rows.iter_mut().enumerate() {
            let x = (i % w) as i32;
            let y = (y0 + i / w) as i32;
            let v = vis[y as usize * w + x as usize];
            if v.id == NONE {
                continue;
            }
            s.c = if v.id & LINE_FLAG != 0 {
                ctx.shade_line(line_refs[(v.id & !LINE_FLAG) as usize], v.b[0])
            } else {
                let t = &screen_tris[v.id as usize];
                ctx.shade_tri(t, &v, x, y)
            };
            s.covered = true;
        }
    });

    // ---- Blend pass.
    if !blend.is_empty() {
        // Back-to-front; stable so equal depths keep submission order.
        blend.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut btris = Vec::new();
        let mut brefs = Vec::new();
        for (_, r) in &blend {
            let item = &prepared.items[r.item as usize];
            let mat = &prepared.materials[item.material];
            let base = 3 * r.tri as usize;
            let id = brefs.len() as u32;
            brefs.push(*r);
            setup_triangle(
                [
                    clip(item.positions[base]),
                    clip(item.positions[base + 1]),
                    clip(item.positions[base + 2]),
                ],
                id,
                if mat.double_sided {
                    Cull::None
                } else {
                    Cull::Back
                },
                false,
                rw,
                rh,
                &mut btris,
            );
        }
        let bins = bin_tris(&btris, rh);
        let bg = {
            let lut = srgb_u8_lut();
            let b = opts.background.0;
            [
                lut[b[0] as usize],
                lut[b[1] as usize],
                lut[b[2] as usize],
                b[3] as f32 / 255.0,
            ]
        };
        let bctx = ShadeCtx {
            tri_refs: &brefs,
            ..ctx.clone_ctx()
        };
        par_bands(&mut samples, w, |band, y0, rows| {
            let y1 = y0 + rows.len() / w;
            for &ti in &bins[band] {
                let t = &btris[ti as usize];
                t.raster_rows(y0 as i32, y1 as i32, |x, y, z, b| {
                    let gi = y as usize * w + x as usize;
                    if z >= vis[gi].depth {
                        return;
                    }
                    let v = VisPixel {
                        depth: z,
                        id: ti,
                        b,
                    };
                    let src = bctx.shade_tri(t, &v, x, y);
                    let s = &mut rows[gi - y0 * w];
                    let dst = if s.covered { s.c } else { bg };
                    s.c = over(src, dst);
                    s.covered = true;
                });
            }
        });
    }

    Frame {
        out_w: width,
        out_h: height,
        aa,
        samples,
        background: opts.background.0,
        display_referred: matches!(mode, ShadingMode::NormalDebug | ShadingMode::DepthDebug),
    }
}

fn push_line(
    refs: &mut Vec<LineRef>,
    lines: &mut Vec<ScreenLine>,
    r: LineRef,
    a: [f32; 4],
    b: [f32; 4],
    w: u32,
    h: u32,
) {
    if let Some(l) = setup_line(a, b, w, h) {
        refs.push(r);
        lines.push(l);
    }
}

/// Straight-alpha Porter–Duff *over*.
fn over(src: [f32; 4], dst: [f32; 4]) -> [f32; 4] {
    let a = src[3].clamp(0.0, 1.0);
    let ad = dst[3].clamp(0.0, 1.0);
    let ao = a + ad * (1.0 - a);
    if ao <= 0.0 {
        return [0.0; 4];
    }
    let mut c = [0.0; 4];
    for k in 0..3 {
        c[k] = (src[k] * a + dst[k] * ad * (1.0 - a)) / ao;
    }
    c[3] = ao;
    c
}

// ---------------------------------------------------------------------
// Shading.
// ---------------------------------------------------------------------

/// Barycentric screen derivatives of a fragment.
struct Grads {
    dx: [f32; 3],
    dy: [f32; 3],
}

impl Grads {
    fn of(t: &ScreenTri, b: [f32; 3], x: i32, y: i32) -> Self {
        let bx = t.bary_at(x as f32 + 1.5, y as f32 + 0.5);
        let by = t.bary_at(x as f32 + 0.5, y as f32 + 1.5);
        Self {
            dx: [bx[0] - b[0], bx[1] - b[1], bx[2] - b[2]],
            dy: [by[0] - b[0], by[1] - b[1], by[2] - b[2]],
        }
    }
}

fn interp3(v: &[[f32; 3]], base: usize, b: [f32; 3]) -> [f32; 3] {
    let (a, c, d) = (v[base], v[base + 1], v[base + 2]);
    [
        a[0] * b[0] + c[0] * b[1] + d[0] * b[2],
        a[1] * b[0] + c[1] * b[1] + d[1] * b[2],
        a[2] * b[0] + c[2] * b[1] + d[2] * b[2],
    ]
}

fn interp2(v: &[[f32; 2]], base: usize, b: [f32; 3]) -> [f32; 2] {
    let (a, c, d) = (v[base], v[base + 1], v[base + 2]);
    [
        a[0] * b[0] + c[0] * b[1] + d[0] * b[2],
        a[1] * b[0] + c[1] * b[1] + d[1] * b[2],
    ]
}

fn interp4(v: &[[f32; 4]], base: usize, b: [f32; 3]) -> [f32; 4] {
    let (a, c, d) = (v[base], v[base + 1], v[base + 2]);
    let mut o = [0.0; 4];
    for k in 0..4 {
        o[k] = a[k] * b[0] + c[k] * b[1] + d[k] * b[2];
    }
    o
}

struct ShadeCtx<'a> {
    prepared: &'a PreparedScene,
    camera: &'a Camera,
    mode: ShadingMode,
    legacy_light: DirLight,
    ambient: f32,
    tri_refs: &'a [TriRef],
    shadows: Vec<Option<ShadowMap>>,
}

impl<'a> ShadeCtx<'a> {
    /// Copy of the context (shadow maps are shared `Arc` data).
    fn clone_ctx(&self) -> ShadeCtx<'_> {
        ShadeCtx {
            prepared: self.prepared,
            camera: self.camera,
            mode: self.mode,
            legacy_light: self.legacy_light,
            ambient: self.ambient,
            tri_refs: self.tri_refs,
            shadows: self.shadows.clone(),
        }
    }

    fn sample_tex(
        &self,
        item: &DrawItem,
        binding: &TextureBinding,
        base: usize,
        b: [f32; 3],
        g: &Grads,
        space: ColorSpace,
    ) -> Option<[f32; 4]> {
        let tex = self.prepared.texture(binding)?;
        let uvs = item.uv_set(binding.uv_set)?;
        let uv = interp2(uvs, base, b);
        let duv = |d: [f32; 3]| -> [f32; 2] {
            let raw = [uv[0] + 0.0, uv[1] + 0.0];
            let moved = [
                raw[0] + uvs[base][0] * d[0] + uvs[base + 1][0] * d[1] + uvs[base + 2][0] * d[2],
                raw[1] + uvs[base][1] * d[0] + uvs[base + 1][1] * d[1] + uvs[base + 2][1] * d[2],
            ];
            let a = binding.transform_uv(raw);
            let m = binding.transform_uv(moved);
            [m[0] - a[0], m[1] - a[1]]
        };
        let dx = duv(g.dx);
        let dy = duv(g.dy);
        Some(tex.sample_grad(binding.transform_uv(uv), dx, dy, space))
    }

    /// `baseColor` (factor × texture × vertex colour), linear RGBA.
    fn base_color(
        &self,
        item: &DrawItem,
        mat: &PreparedMaterial,
        base: usize,
        b: [f32; 3],
        g: &Grads,
    ) -> [f32; 4] {
        let mut c = mat.base_color;
        if let Some(bind) = &mat.base_color_texture {
            if let Some(t) = self.sample_tex(item, bind, base, b, g, ColorSpace::Srgb) {
                for k in 0..4 {
                    c[k] *= t[k];
                }
            }
        }
        if !item.colors.is_empty() {
            let v = interp4(&item.colors, base, b);
            for k in 0..4 {
                c[k] *= v[k];
            }
        }
        c
    }

    fn alpha(
        &self,
        item: &DrawItem,
        mat: &PreparedMaterial,
        base: usize,
        b: [f32; 3],
        g: &Grads,
    ) -> f32 {
        self.base_color(item, mat, base, b, g)[3]
    }

    fn shade_line(&self, r: LineRef, t: f32) -> [f32; 4] {
        let item = &self.prepared.items[r.item as usize];
        let mat = &self.prepared.materials[item.material];
        let mut c = mat.base_color;
        if self.mode == ShadingMode::Pbr && !item.colors.is_empty() {
            let (a, b) = (item.colors[r.a as usize], item.colors[r.b as usize]);
            for k in 0..4 {
                c[k] *= a[k] + (b[k] - a[k]) * t;
            }
        }
        c
    }

    fn shade_tri(&self, st: &ScreenTri, v: &VisPixel, x: i32, y: i32) -> [f32; 4] {
        let r = self.tri_refs[st.prim as usize];
        let item = &self.prepared.items[r.item as usize];
        let mat = &self.prepared.materials[item.material];
        let base = 3 * r.tri as usize;
        let b = v.b;
        match self.mode {
            ShadingMode::Flat | ShadingMode::Wireframe => mat.base_color,
            ShadingMode::Gouraud => {
                let n = &item.normals;
                let ca = shade_pixel(mat.base_color, n[base], &self.legacy_light);
                let cb = shade_pixel(mat.base_color, n[base + 1], &self.legacy_light);
                let cc = shade_pixel(mat.base_color, n[base + 2], &self.legacy_light);
                let mut o = [0.0; 4];
                for k in 0..4 {
                    o[k] = ca[k] * b[0] + cb[k] * b[1] + cc[k] * b[2];
                }
                o
            }
            ShadingMode::Phong => {
                let n = vec3_normalise(interp3(&item.normals, base, b));
                shade_pixel(mat.base_color, n, &self.legacy_light)
            }
            ShadingMode::NormalDebug => {
                let n = vec3_normalise(interp3(&item.normals, base, b));
                let lut = srgb_u8_lut();
                [
                    lut[normal_to_byte(n[0]) as usize],
                    lut[normal_to_byte(n[1]) as usize],
                    lut[normal_to_byte(n[2]) as usize],
                    1.0,
                ]
            }
            ShadingMode::DepthDebug => {
                let g = srgb_u8_lut()[depth_to_byte(v.depth) as usize];
                [g, g, g, 1.0]
            }
            _ => {
                let g = Grads::of(st, b, x, y);
                self.shade_pbr(item, mat, base, b, &g, st.front)
            }
        }
    }

    fn view_dir(&self, p: [f32; 3]) -> [f32; 3] {
        match self.camera.projection {
            Projection::Orthographic => [
                -self.camera.forward[0],
                -self.camera.forward[1],
                -self.camera.forward[2],
            ],
            _ => vec3_normalise(vec3_sub(self.camera.eye, p)),
        }
    }

    fn shade_pbr(
        &self,
        item: &DrawItem,
        mat: &PreparedMaterial,
        base: usize,
        b: [f32; 3],
        g: &Grads,
        front: bool,
    ) -> [f32; 4] {
        let col = self.base_color(item, mat, base, b, g);
        let alpha = match mat.alpha_mode {
            AlphaMode::Blend => col[3].clamp(0.0, 1.0),
            _ => 1.0,
        };
        if mat.unlit {
            return [col[0], col[1], col[2], alpha];
        }
        let p = interp3(&item.positions, base, b);
        let pos = &item.positions;
        let mut ng = vec3_normalise(vec3_cross(
            vec3_sub(pos[base + 1], pos[base]),
            vec3_sub(pos[base + 2], pos[base]),
        ));
        let mut n = if item.has_vertex_normals {
            let n = vec3_normalise(interp3(&item.normals, base, b));
            if vec3_dot(n, n) > 0.5 {
                n
            } else {
                ng
            }
        } else {
            ng
        };
        if !front {
            n = [-n[0], -n[1], -n[2]];
            ng = [-ng[0], -ng[1], -ng[2]];
        }
        let v = self.view_dir(p);

        // Metallic / roughness.
        let mut metallic = mat.metallic;
        let mut roughness = mat.roughness;
        if let Some(bind) = &mat.metallic_roughness_texture {
            if let Some(t) = self.sample_tex(item, bind, base, b, g, ColorSpace::Linear) {
                roughness *= t[1];
                metallic *= t[2];
            }
        }

        // Normal map (glTF §3.9.3: tangent-space, +Y up in UV space).
        if let Some(bind) = &mat.normal_texture {
            if !item.tangents.is_empty() {
                if let Some(t) = self.sample_tex(item, bind, base, b, g, ColorSpace::Linear) {
                    let tg = interp4(&item.tangents, base, b);
                    let w = if item.tangents[base][3] < 0.0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let t3 = [tg[0], tg[1], tg[2]];
                    let d = vec3_dot(n, t3);
                    let tt = vec3_normalise([t3[0] - n[0] * d, t3[1] - n[1] * d, t3[2] - n[2] * d]);
                    if vec3_dot(tt, tt) > 0.5 {
                        let bt = vec3_cross(n, tt);
                        let bt = [bt[0] * w, bt[1] * w, bt[2] * w];
                        let s = mat.normal_scale;
                        let ts = [
                            (t[0] * 2.0 - 1.0) * s,
                            (t[1] * 2.0 - 1.0) * s,
                            t[2] * 2.0 - 1.0,
                        ];
                        let nn = vec3_normalise([
                            tt[0] * ts[0] + bt[0] * ts[1] + n[0] * ts[2],
                            tt[1] * ts[0] + bt[1] * ts[1] + n[1] * ts[2],
                            tt[2] * ts[0] + bt[2] * ts[1] + n[2] * ts[2],
                        ]);
                        if vec3_dot(nn, nn) > 0.5 {
                            n = nn;
                        }
                    }
                }
            }
        }

        let mut ao = 1.0;
        if let Some(bind) = &mat.occlusion_texture {
            if let Some(t) = self.sample_tex(item, bind, base, b, g, ColorSpace::Linear) {
                ao = 1.0 + mat.occlusion_strength * (t[0] - 1.0);
            }
        }
        let mut emissive = mat.emissive;
        if let Some(bind) = &mat.emissive_texture {
            if let Some(t) = self.sample_tex(item, bind, base, b, g, ColorSpace::Srgb) {
                for k in 0..3 {
                    emissive[k] *= t[k];
                }
            }
        }

        let params = BrdfParams::metallic_roughness(
            [col[0], col[1], col[2]],
            metallic,
            roughness,
            mat.dielectric_f0(),
        );
        let mut out = [0.0f32; 3];
        for k in 0..3 {
            out[k] = emissive[k] + self.ambient * ao * (params.c_diff[k] + params.f0[k]);
        }
        for (li, light) in self.prepared.lights.iter().enumerate() {
            let Some(s) = light.sample(p) else { continue };
            let f = brdf::eval(&params, n, v, s.l);
            if f == [0.0; 3] {
                continue;
            }
            let vis = match self.shadows.get(li) {
                Some(Some(sm)) => sm.visibility(p, ng, s.l),
                _ => 1.0,
            };
            for k in 0..3 {
                out[k] += f[k] * s.radiance[k] * vis;
            }
        }
        [out[0], out[1], out[2], alpha]
    }
}

/// Map a single normal component in `[-1, 1]` into a `u8` colour
/// channel via `(n + 1) / 2 * 255`. NaNs fall back to `128` (the
/// encoded zero).
pub(crate) fn normal_to_byte(n: f32) -> u8 {
    if !n.is_finite() {
        return 128;
    }
    let v = ((n.clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0).round();
    v.clamp(0.0, 255.0) as u8
}

/// Map an NDC z value (`[-1, 1]`, near = -1) to a grayscale byte where
/// near = 255 and far = 0. NaN / out-of-range values are clamped.
pub(crate) fn depth_to_byte(z: f32) -> u8 {
    if !z.is_finite() {
        return 0;
    }
    let zc = z.clamp(-1.0, 1.0);
    let v = ((1.0 - (zc * 0.5 + 0.5)) * 255.0).round();
    v.clamp(0.0, 255.0) as u8
}

// ---------------------------------------------------------------------
// Shadow maps.
// ---------------------------------------------------------------------

/// Depth map of one light: linear depth along the light direction per
/// texel (`INFINITY` = nothing).
#[derive(Debug, Clone)]
struct ShadowMap {
    view_proj: [[f32; 4]; 4],
    size: u32,
    depth: std::sync::Arc<Vec<f32>>,
    eye: [f32; 3],
    dir: [f32; 3],
    /// World size of one texel: constant (ortho) or per unit distance
    /// (perspective).
    texel: f32,
    perspective: bool,
}

impl ShadowMap {
    /// Fraction of the light reaching `p` (geometric normal `ng`,
    /// direction to the light `l`): 3×3 bilinear PCF.
    fn visibility(&self, p: [f32; 3], ng: [f32; 3], l: [f32; 3]) -> f32 {
        let dist = vec3_dot(vec3_sub(p, self.eye), self.dir);
        let texel = if self.perspective {
            self.texel * dist.max(1.0e-4)
        } else {
            self.texel
        };
        // Normal offset toward the lit side, scaled by grazing angle.
        let ng = if vec3_dot(ng, l) < 0.0 {
            [-ng[0], -ng[1], -ng[2]]
        } else {
            ng
        };
        let cos = vec3_dot(ng, l).clamp(0.0, 1.0);
        let off = texel * (1.0 + 2.0 * (1.0 - cos));
        let q = [p[0] + ng[0] * off, p[1] + ng[1] * off, p[2] + ng[2] * off];
        let c = mat4_mul_vec4(&self.view_proj, [q[0], q[1], q[2], 1.0]);
        if c[3] <= 0.0 {
            return 1.0;
        }
        let s = self.size as f32;
        let u = (c[0] / c[3] * 0.5 + 0.5) * s - 0.5;
        let v = (1.0 - (c[1] / c[3] * 0.5 + 0.5)) * s - 0.5;
        if !(u > -1.0 && v > -1.0 && u < s && v < s) {
            return 1.0;
        }
        let d_recv = vec3_dot(vec3_sub(q, self.eye), self.dir) - texel;
        let lit = |x: i64, y: i64| -> f32 {
            if x < 0 || y < 0 || x >= self.size as i64 || y >= self.size as i64 {
                return 1.0;
            }
            if self.depth[y as usize * self.size as usize + x as usize] >= d_recv {
                1.0
            } else {
                0.0
            }
        };
        let mut sum = 0.0;
        for oy in -1..=1 {
            for ox in -1..=1 {
                let (fu, fv) = (u + ox as f32, v + oy as f32);
                let (x0, y0) = (fu.floor(), fv.floor());
                let (tx, ty) = (fu - x0, fv - y0);
                let (x0, y0) = (x0 as i64, y0 as i64);
                let a = lit(x0, y0) * (1.0 - tx) + lit(x0 + 1, y0) * tx;
                let b = lit(x0, y0 + 1) * (1.0 - tx) + lit(x0 + 1, y0 + 1) * tx;
                sum += a * (1.0 - ty) + b * ty;
            }
        }
        sum / 9.0
    }
}

fn build_shadow_maps(prepared: &PreparedScene, size: u32) -> Vec<Option<ShadowMap>> {
    prepared
        .lights
        .iter()
        .map(|l| build_shadow_map(prepared, l, size))
        .collect()
}

fn build_shadow_map(
    prepared: &PreparedScene,
    light: &PreparedLight,
    size: u32,
) -> Option<ShadowMap> {
    if light.kind == LightKind::Point {
        return None;
    }
    let (mn, mx) = prepared.bounds()?;
    let c = [
        (mn[0] + mx[0]) * 0.5,
        (mn[1] + mx[1]) * 0.5,
        (mn[2] + mx[2]) * 0.5,
    ];
    let r = (vec3_dot(vec3_sub(mx, mn), vec3_sub(mx, mn)).sqrt() * 0.5).max(1.0e-3);
    let dir = vec3_normalise(light.direction);
    let up = if dir[1].abs() > 0.99 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let (eye, view, proj, texel, perspective_map) = match light.kind {
        LightKind::Directional => {
            let eye = [
                c[0] - dir[0] * 2.0 * r,
                c[1] - dir[1] * 2.0 * r,
                c[2] - dir[2] * 2.0 * r,
            ];
            let view = look_at(eye, c, up);
            let proj = orthographic(-r, r, -r, r, r * 0.5, r * 3.5);
            (eye, view, proj, 2.0 * r / size as f32, false)
        }
        _ => {
            let eye = light.position;
            let target = [eye[0] + dir[0], eye[1] + dir[1], eye[2] + dir[2]];
            let view = look_at(eye, target, up);
            let fov = (2.0 * light.outer_cone_angle + 0.05).min(170f32.to_radians());
            let mut far: f32 = 0.0;
            for i in 0..8 {
                let k = [
                    if i & 1 == 0 { mn[0] } else { mx[0] },
                    if i & 2 == 0 { mn[1] } else { mx[1] },
                    if i & 4 == 0 { mn[2] } else { mx[2] },
                ];
                far = far.max(vec3_dot(vec3_sub(k, eye), vec3_sub(k, eye)).sqrt());
            }
            let near = (r * 1.0e-3).max(1.0e-4);
            let far = (far * 1.01).max(near * 4.0);
            let proj = perspective(fov, 1.0, near, far);
            (eye, view, proj, 2.0 * (fov * 0.5).tan() / size as f32, true)
        }
    };
    let view_proj = mat4_mul(proj, view);

    // Casters: opaque + masked triangles (BLEND surfaces cast no
    // shadow), both faces.
    let mut refs: Vec<TriRef> = Vec::new();
    let mut tris = Vec::new();
    for (ii, item) in prepared.items.iter().enumerate() {
        if item.topology != DrawTopology::Triangles {
            continue;
        }
        let mat = &prepared.materials[item.material];
        if mat.alpha_mode == AlphaMode::Blend {
            continue;
        }
        let masked = matches!(mat.alpha_mode, AlphaMode::Mask { .. });
        for (t, v) in item.positions.chunks_exact(3).enumerate() {
            let id = refs.len() as u32;
            refs.push(TriRef {
                item: ii as u32,
                tri: t as u32,
            });
            let cl = |p: [f32; 3]| mat4_mul_vec4(&view_proj, [p[0], p[1], p[2], 1.0]);
            setup_triangle(
                [cl(v[0]), cl(v[1]), cl(v[2])],
                id,
                Cull::None,
                masked,
                size,
                size,
                &mut tris,
            );
        }
    }
    let ctx = ShadeCtx {
        prepared,
        camera: &Camera::frame_bounds(1, 1, mn, mx, &RenderOptions::default()),
        mode: ShadingMode::Pbr,
        legacy_light: build_light(crate::options::LightSpec::default_light()),
        ambient: 0.0,
        tri_refs: &refs,
        shadows: Vec::new(),
    };
    let mask_test = |t: &ScreenTri, b: [f32; 3], x: i32, y: i32| -> bool {
        let r = refs[t.prim as usize];
        let item = &prepared.items[r.item as usize];
        let mat = &prepared.materials[item.material];
        let AlphaMode::Mask { cutoff } = mat.alpha_mode else {
            return true;
        };
        ctx.alpha(item, mat, 3 * r.tri as usize, b, &Grads::of(t, b, x, y)) >= cutoff
    };
    let vis = raster_visibility(&tris, &[], size, size, &mask_test);
    let depth: Vec<f32> = vis
        .iter()
        .map(|v| {
            if v.id == NONE {
                return f32::INFINITY;
            }
            let r = refs[tris[v.id as usize].prim as usize];
            let item = &prepared.items[r.item as usize];
            let p = interp3(&item.positions, 3 * r.tri as usize, v.b);
            vec3_dot(vec3_sub(p, eye), dir)
        })
        .collect();
    Some(ShadowMap {
        view_proj,
        size,
        depth: std::sync::Arc::new(depth),
        eye,
        dir,
        texel,
        perspective: perspective_map,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::BackgroundColor;
    use oxideav_mesh3d::{Mesh, MeshId, Node as MNode, Primitive as MPrimitive, Scene3D, Topology};

    fn unit_triangle_scene() -> Scene3D {
        let mut prim = MPrimitive::new(Topology::Triangles);
        prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
        let mesh = Mesh::new("triangle".to_string()).with_primitive(prim);
        let mut scene = Scene3D::new();
        scene.meshes.push(mesh);
        let node = MNode {
            mesh: Some(MeshId(0)),
            ..MNode::default()
        };
        scene.nodes.push(node);
        scene.roots.push(oxideav_mesh3d::NodeId(0));
        scene
    }

    fn render_with_mode(mode: ShadingMode) -> RgbaImage {
        let scene = unit_triangle_scene();
        let opts = RenderOptions {
            width: 64,
            height: 64,
            shading: mode,
            background: BackgroundColor([255, 255, 255, 255]),
            ..RenderOptions::default()
        };
        render_scene(&scene, &opts)
    }

    #[test]
    fn renders_triangle_changes_some_pixels() {
        let img = render_with_mode(ShadingMode::Flat);
        assert_eq!(img.width, 64);
        assert_eq!(img.height, 64);
        assert_eq!(img.pixels.len(), 64 * 64 * 4);
        let drawn = img
            .pixels
            .chunks_exact(4)
            .any(|p| p != [255, 255, 255, 255]);
        assert!(drawn, "expected at least one rasterised pixel");
    }

    #[test]
    fn wireframe_paints_triangle_edges() {
        let scene = unit_triangle_scene();
        let opts = RenderOptions {
            width: 32,
            height: 32,
            shading: ShadingMode::Wireframe,
            background: BackgroundColor([0, 0, 0, 255]),
            ..RenderOptions::default()
        };
        let img = render_scene(&scene, &opts);
        let drawn = img.pixels.chunks_exact(4).any(|p| p != [0, 0, 0, 255]);
        assert!(drawn, "wireframe should paint at least one edge pixel");
    }

    #[test]
    fn gouraud_renders_pixels() {
        let img = render_with_mode(ShadingMode::Gouraud);
        let drawn = img
            .pixels
            .chunks_exact(4)
            .any(|p| p != [255, 255, 255, 255]);
        assert!(drawn, "gouraud should rasterise the triangle");
    }

    #[test]
    fn phong_renders_pixels() {
        let img = render_with_mode(ShadingMode::Phong);
        let drawn = img
            .pixels
            .chunks_exact(4)
            .any(|p| p != [255, 255, 255, 255]);
        assert!(drawn, "phong should rasterise the triangle");
    }

    #[test]
    fn empty_scene_yields_pure_background() {
        let scene = Scene3D::new();
        let opts = RenderOptions {
            width: 8,
            height: 8,
            background: BackgroundColor([42, 7, 99, 255]),
            ..RenderOptions::default()
        };
        let img = render_scene(&scene, &opts);
        for px in img.pixels.chunks_exact(4) {
            assert_eq!(px, &[42, 7, 99, 255]);
        }
    }

    #[test]
    fn normal_to_byte_endpoints() {
        assert_eq!(normal_to_byte(-1.0), 0);
        assert_eq!(normal_to_byte(0.0), 128);
        assert_eq!(normal_to_byte(1.0), 255);
    }

    #[test]
    fn normal_to_byte_clamps_out_of_range() {
        assert_eq!(normal_to_byte(-2.0), 0);
        assert_eq!(normal_to_byte(2.0), 255);
        assert_eq!(normal_to_byte(f32::NAN), 128);
    }

    #[test]
    fn depth_to_byte_endpoints() {
        assert_eq!(depth_to_byte(-1.0), 255);
        assert_eq!(depth_to_byte(1.0), 0);
        let mid = depth_to_byte(0.0);
        assert!(
            (120..=140).contains(&mid),
            "mid-z grayscale should be ~128, got {mid}"
        );
    }

    #[test]
    fn normal_debug_renders_pixels() {
        let img = render_with_mode(ShadingMode::NormalDebug);
        let drawn = img
            .pixels
            .chunks_exact(4)
            .any(|p| p != [255, 255, 255, 255]);
        assert!(drawn, "normal-debug should rasterise the triangle");
    }

    #[test]
    fn depth_debug_renders_grayscale() {
        let img = render_with_mode(ShadingMode::DepthDebug);
        let drawn = img
            .pixels
            .chunks_exact(4)
            .any(|p| p != [255, 255, 255, 255]);
        assert!(drawn, "depth-debug should rasterise the triangle");
        for px in img.pixels.chunks_exact(4) {
            if px == [255, 255, 255, 255] {
                continue;
            }
            assert_eq!(px[0], px[1], "depth-debug pixel must be grayscale");
            assert_eq!(px[1], px[2], "depth-debug pixel must be grayscale");
            assert_eq!(px[3], 255, "depth-debug pixel alpha must be opaque");
        }
    }

    #[test]
    fn aa_default_keeps_output_dims() {
        let scene = unit_triangle_scene();
        let opts = RenderOptions {
            width: 32,
            height: 32,
            background: BackgroundColor([255, 255, 255, 255]),
            ..RenderOptions::default()
        };
        let img = render_scene(&scene, &opts);
        assert_eq!(img.width, 32);
        assert_eq!(img.height, 32);
        assert_eq!(img.pixels.len(), 32 * 32 * 4);
    }

    #[test]
    fn aa_factor_two_keeps_output_dims() {
        let scene = unit_triangle_scene();
        let opts = RenderOptions {
            width: 32,
            height: 32,
            background: BackgroundColor([255, 255, 255, 255]),
            aa: 2,
            ..RenderOptions::default()
        };
        let img = render_scene(&scene, &opts);
        assert_eq!(img.width, 32);
        assert_eq!(img.height, 32);
        assert_eq!(img.pixels.len(), 32 * 32 * 4);
    }

    #[test]
    fn aa_softens_triangle_edges() {
        let scene = unit_triangle_scene();
        let bg = BackgroundColor([255, 255, 255, 255]);
        let opts_1 = RenderOptions {
            width: 64,
            height: 64,
            background: bg,
            ..RenderOptions::default()
        };
        let opts_4 = RenderOptions {
            aa: 4,
            ..opts_1.clone()
        };
        let img1 = render_scene(&scene, &opts_1);
        let img4 = render_scene(&scene, &opts_4);
        let intermediate = |img: &RgbaImage| -> usize {
            img.pixels
                .chunks_exact(4)
                .filter(|p| {
                    let is_bg = p == &[255, 255, 255, 255];
                    let r = p[0];
                    let same = p[0] == p[1] && p[1] == p[2];
                    !is_bg && !(same && (r == 0 || r == 255))
                })
                .count()
        };
        let int1 = intermediate(&img1);
        let int4 = intermediate(&img4);
        assert!(
            int4 > int1,
            "expected SSAA to introduce more intermediate edge pixels; got 1×={int1}, 4×={int4}"
        );
    }

    #[test]
    fn aa_factor_one_matches_no_aa() {
        let scene = unit_triangle_scene();
        let bg = BackgroundColor([200, 100, 50, 255]);
        let opts_off = RenderOptions {
            width: 16,
            height: 16,
            background: bg,
            ..RenderOptions::default()
        };
        let opts_one = RenderOptions {
            aa: 1,
            ..opts_off.clone()
        };
        let img_off = render_scene(&scene, &opts_off);
        let img_one = render_scene(&scene, &opts_one);
        assert_eq!(img_off.pixels, img_one.pixels);
    }
}
