//! Scanline software rasteriser — the Phase B backend behind
//! [`crate::RenderBackend::Scanline`].
//!
//! Pure-Rust, zero external rendering dependencies. Walks the
//! [`Scene3D`](oxideav_mesh3d::Scene3D) node forest, projects every
//! vertex through a model-view-projection chain, then rasterises one
//! triangle / line / point at a time with a half-space edge-function
//! pipeline backed by a per-pixel z-buffer.
//!
//! Migration provenance: the algorithms here were authored in
//! `oxideav-cli-convert/src/mesh3d_render.rs` (round 44 + 45 in that
//! crate) and moved verbatim into this crate in Phase B so the renderer
//! lives behind a stable trait. The only structural change is the
//! caller-facing options struct — [`Mesh3DOptions`] from cli-convert
//! becomes the framework-wide [`RenderOptions`] defined in
//! [`crate::options`].
//!
//! Clean-room policy: every algorithm here has a textbook source.
//! Half-space edge-function rasterisation traces to Pineda's 1988
//! SIGGRAPH paper "A Parallel Algorithm for Polygon Rasterization".
//! Bresenham's 1965 IBM Systems Journal paper covers the line walker.
//! sRGB encoding follows IEC 61966-2-1. Look-at / perspective /
//! orthographic matrices use the standard right-handed OpenGL
//! conventions. No reference renderer source code was consulted at
//! any stage.
//!
//! ## Module layout
//!
//! The internal sections below mirror the module split planned for a
//! future refactor (see [`crate`] docs): camera framing, lighting,
//! matrix helpers, scene bbox walk, framebuffer, and the per-shading-
//! mode rasterisers. Kept in one file for now so the diff against the
//! cli-convert original stays auditable.

use oxideav_mesh3d::{Indices, Material, Node, NodeId, Primitive, Scene3D, Topology};

use crate::image::RgbaImage;
use crate::options::{CameraSpec, LightSpec, Projection, RenderOptions, ShadingMode};

/// Default vertical field of view in degrees (perspective projection).
/// Mirrors [`RenderOptions::default`]; redeclared as a constant here so
/// the auto-frame math stays readable when called with a custom FOV.
#[allow(dead_code)]
const DEFAULT_FOV_DEG: f32 = 60.0;

/// Constant ambient term added to every shaded pixel so back-faces
/// stay visible at the price of contrast. Matches the typical
/// software-renderer baseline (textbook Phong-with-ambient).
const AMBIENT: f32 = 0.2;

// ---------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------

/// Render `scene` into a packed RGBA8 buffer per `opts`.
///
/// When `opts.aa >= 2`, the scene is rasterised at
/// `aa × width` × `aa × height` and box-filtered back down to the
/// requested output — classic SSAA, no temporal jitter, no rotated
/// grid.
pub fn render_scene(scene: &Scene3D, opts: &RenderOptions) -> RgbaImage {
    let width = opts.width.max(1);
    let height = opts.height.max(1);
    let aa = opts.aa.clamp(1, 8);
    let render_w = width.saturating_mul(aa).max(1);
    let render_h = height.saturating_mul(aa).max(1);

    let background = opts.background.0;
    let mut fb = Framebuffer::new(render_w, render_h, background);

    let bbox = scene_bbox(scene);
    let camera = Camera::build(render_w, render_h, bbox, opts);
    let light = build_light(opts.light);
    let mode = opts.shading;

    // Walk the node forest in pre-order, composing world matrices.
    for &root in &scene.roots {
        walk_node(scene, root, identity4(), &camera, &light, &mut fb, mode);
    }

    let img = fb.into_image();
    if aa <= 1 {
        img
    } else {
        downsample_box(&img, width, height, aa)
    }
}

// ---------------------------------------------------------------------
// Anti-aliasing — box-filter SSAA downsample.
// ---------------------------------------------------------------------

/// Box-filter `src` (which is `aa × dst_w` by `aa × dst_h`) down to
/// `dst_w × dst_h`. Each output pixel averages `aa²` source pixels in
/// straight linear-byte space — no gamma round-trip.
fn downsample_box(src: &RgbaImage, dst_w: u32, dst_h: u32, aa: u32) -> RgbaImage {
    let aa = aa.max(1);
    let aa_us = aa as usize;
    let dst_w_us = dst_w as usize;
    let dst_h_us = dst_h as usize;
    let src_stride = src.stride;
    let mut pixels = Vec::with_capacity(dst_w_us * dst_h_us * 4);
    let div = (aa_us * aa_us) as u32;
    for dy in 0..dst_h_us {
        let sy0 = dy * aa_us;
        for dx in 0..dst_w_us {
            let sx0 = dx * aa_us;
            let mut acc = [0u32; 4];
            for j in 0..aa_us {
                let row_base = (sy0 + j) * src_stride + sx0 * 4;
                for i in 0..aa_us {
                    let p = row_base + i * 4;
                    acc[0] += src.pixels[p] as u32;
                    acc[1] += src.pixels[p + 1] as u32;
                    acc[2] += src.pixels[p + 2] as u32;
                    acc[3] += src.pixels[p + 3] as u32;
                }
            }
            pixels.push((acc[0] / div) as u8);
            pixels.push((acc[1] / div) as u8);
            pixels.push((acc[2] / div) as u8);
            pixels.push((acc[3] / div) as u8);
        }
    }
    RgbaImage {
        width: dst_w,
        height: dst_h,
        stride: dst_w_us * 4,
        pixels,
    }
}

// ---------------------------------------------------------------------
// Scene walk.
// ---------------------------------------------------------------------

fn walk_node(
    scene: &Scene3D,
    id: NodeId,
    parent_world: [[f32; 4]; 4],
    camera: &Camera,
    light: &DirLight,
    fb: &mut Framebuffer,
    mode: ShadingMode,
) {
    let Some(node) = scene.nodes.get(id.0 as usize) else {
        return;
    };
    let world = mat4_mul(parent_world, node.transform.to_matrix());
    if let Some(mesh_id) = node.mesh {
        if let Some(mesh) = scene.meshes.get(mesh_id.0 as usize) {
            for prim in &mesh.primitives {
                let colour = primitive_colour_linear(scene, prim);
                draw_primitive(prim, &world, camera, light, colour, fb, mode);
            }
        }
    }
    for &child in &node.children {
        walk_node(scene, child, world, camera, light, fb, mode);
    }
}

/// Linear-space (0..=1 RGBA, premultiplied alpha NOT applied) colour for
/// a primitive. We keep the colour in linear space for shading and only
/// convert to sRGB right before writing to the framebuffer.
fn primitive_colour_linear(scene: &Scene3D, prim: &Primitive) -> [f32; 4] {
    let mat = prim
        .material
        .and_then(|mid| scene.materials.get(mid.0 as usize));
    mat.map(|m: &Material| m.base_color)
        .unwrap_or([0.7, 0.7, 0.75, 1.0])
}

fn linear_rgba_to_srgb_u8(c: [f32; 4]) -> [u8; 4] {
    [
        linear_to_srgb_u8(c[0]),
        linear_to_srgb_u8(c[1]),
        linear_to_srgb_u8(c[2]),
        (c[3].clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

fn linear_to_srgb_u8(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    // Single-segment approximation — matches IEC 61966-2-1 well
    // enough for an MVP renderer.
    let v = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

// ---------------------------------------------------------------------
// Per-primitive triangle / line / point expansion + rasterise.
// ---------------------------------------------------------------------

/// Triangle-list expansion + projection + rasterisation for one
/// [`Primitive`]. Lines / line strips / line loops / points / triangle
/// strips / fans are all walked into ordered `(i0, i1, i2)` triplets
/// here so the inner draw loop is topology-agnostic.
fn draw_primitive(
    prim: &Primitive,
    world: &[[f32; 4]; 4],
    camera: &Camera,
    light: &DirLight,
    colour_linear: [f32; 4],
    fb: &mut Framebuffer,
    mode: ShadingMode,
) {
    if prim.positions.is_empty() {
        return;
    }

    // Project every vertex once.
    let view_proj = mat4_mul(camera.proj, camera.view);
    let mvp = mat4_mul(view_proj, *world);
    let projected: Vec<Option<[f32; 3]>> = prim
        .positions
        .iter()
        .map(|p| project_vertex(*p, &mvp, fb.width as f32, fb.height as f32))
        .collect();

    // World-space normals are needed for Gouraud / Phong / NormalDebug.
    // The world matrix is assumed rigid+uniform-scale (the only kind
    // composed from `Transform::translation/rotation/scale`); its 3x3
    // upper-left therefore suffices for normal transformation (no
    // inverse-transpose needed). DepthDebug doesn't need normals, and
    // Flat / Wireframe never look at them.
    let world_normals: Option<Vec<[f32; 3]>> = match mode {
        ShadingMode::Gouraud | ShadingMode::Phong | ShadingMode::NormalDebug => Some(
            prim.normals
                .as_ref()
                .map(|ns| {
                    ns.iter()
                        .map(|n| vec3_normalise(mat3_mul_vec3(world, *n)))
                        .collect()
                })
                .unwrap_or_default(),
        ),
        _ => None,
    };

    // Walk the topology and draw each triangle / line.
    let indices: Vec<u32> = match &prim.indices {
        Some(Indices::U16(v)) => v.iter().map(|&x| x as u32).collect(),
        Some(Indices::U32(v)) => v.clone(),
        None => (0..prim.positions.len() as u32).collect(),
    };
    let n = indices.len();

    match prim.topology {
        Topology::Triangles => {
            let mut i = 0;
            while i + 2 < n {
                draw_tri(
                    prim,
                    world,
                    world_normals.as_deref(),
                    &projected,
                    indices[i] as usize,
                    indices[i + 1] as usize,
                    indices[i + 2] as usize,
                    colour_linear,
                    light,
                    fb,
                    mode,
                );
                i += 3;
            }
        }
        Topology::TriangleStrip => {
            let mut i = 0;
            while i + 2 < n {
                let (a, b, c) = if i % 2 == 0 {
                    (indices[i], indices[i + 1], indices[i + 2])
                } else {
                    (indices[i + 1], indices[i], indices[i + 2])
                };
                draw_tri(
                    prim,
                    world,
                    world_normals.as_deref(),
                    &projected,
                    a as usize,
                    b as usize,
                    c as usize,
                    colour_linear,
                    light,
                    fb,
                    mode,
                );
                i += 1;
            }
        }
        Topology::TriangleFan => {
            if n >= 3 {
                for i in 1..(n - 1) {
                    draw_tri(
                        prim,
                        world,
                        world_normals.as_deref(),
                        &projected,
                        indices[0] as usize,
                        indices[i] as usize,
                        indices[i + 1] as usize,
                        colour_linear,
                        light,
                        fb,
                        mode,
                    );
                }
            }
        }
        Topology::Lines => {
            let colour_srgb = linear_rgba_to_srgb_u8(colour_linear);
            let mut i = 0;
            while i + 1 < n {
                draw_line_pair(
                    &projected,
                    indices[i] as usize,
                    indices[i + 1] as usize,
                    colour_srgb,
                    fb,
                );
                i += 2;
            }
        }
        Topology::LineStrip => {
            let colour_srgb = linear_rgba_to_srgb_u8(colour_linear);
            for i in 0..(n.saturating_sub(1)) {
                draw_line_pair(
                    &projected,
                    indices[i] as usize,
                    indices[i + 1] as usize,
                    colour_srgb,
                    fb,
                );
            }
        }
        Topology::LineLoop => {
            let colour_srgb = linear_rgba_to_srgb_u8(colour_linear);
            for i in 0..(n.saturating_sub(1)) {
                draw_line_pair(
                    &projected,
                    indices[i] as usize,
                    indices[i + 1] as usize,
                    colour_srgb,
                    fb,
                );
            }
            if n >= 2 {
                draw_line_pair(
                    &projected,
                    indices[n - 1] as usize,
                    indices[0] as usize,
                    colour_srgb,
                    fb,
                );
            }
        }
        Topology::Points => {
            let colour_srgb = linear_rgba_to_srgb_u8(colour_linear);
            for &i in &indices {
                if let Some(p) = projected.get(i as usize).and_then(|v| v.as_ref()) {
                    let x = p[0].round() as i32;
                    let y = p[1].round() as i32;
                    fb.set_pixel(x, y, p[2], colour_srgb);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_tri(
    prim: &Primitive,
    world: &[[f32; 4]; 4],
    world_normals: Option<&[[f32; 3]]>,
    projected: &[Option<[f32; 3]>],
    a: usize,
    b: usize,
    c: usize,
    colour_linear: [f32; 4],
    light: &DirLight,
    fb: &mut Framebuffer,
    mode: ShadingMode,
) {
    let (Some(va), Some(vb), Some(vc)) = (
        projected.get(a).and_then(|v| *v),
        projected.get(b).and_then(|v| *v),
        projected.get(c).and_then(|v| *v),
    ) else {
        return;
    };
    match mode {
        ShadingMode::Wireframe => {
            let colour_srgb = linear_rgba_to_srgb_u8(colour_linear);
            draw_line(fb, va, vb, colour_srgb);
            draw_line(fb, vb, vc, colour_srgb);
            draw_line(fb, vc, va, colour_srgb);
        }
        ShadingMode::Flat => {
            let colour_srgb = linear_rgba_to_srgb_u8(colour_linear);
            rasterise_triangle_flat(fb, va, vb, vc, colour_srgb);
        }
        ShadingMode::Gouraud | ShadingMode::Phong => {
            // Pick the three vertex normals for this triangle. Either
            // pulled from the (transformed) per-vertex normal buffer, or
            // synthesised from the face normal of the world-space
            // positions.
            let (na, nb, nc) = vertex_normals_for_face(prim, world, world_normals, a, b, c);
            if matches!(mode, ShadingMode::Gouraud) {
                // Light per-vertex, interpolate colour.
                let ca = shade_pixel(colour_linear, na, light);
                let cb = shade_pixel(colour_linear, nb, light);
                let cc = shade_pixel(colour_linear, nc, light);
                rasterise_triangle_gouraud(fb, va, vb, vc, ca, cb, cc);
            } else {
                // Phong — interpolate normal, light per-pixel.
                rasterise_triangle_phong(fb, va, vb, vc, na, nb, nc, colour_linear, light);
            }
        }
        ShadingMode::NormalDebug => {
            let (na, nb, nc) = vertex_normals_for_face(prim, world, world_normals, a, b, c);
            rasterise_triangle_normal_debug(fb, va, vb, vc, na, nb, nc);
        }
        ShadingMode::DepthDebug => {
            rasterise_triangle_depth_debug(fb, va, vb, vc);
        }
    }
}

/// Pick the three world-space vertex normals for a face. When the
/// primitive has its own per-vertex normals (and they're long enough),
/// those are used. Otherwise we fall back to the face normal computed
/// from the world-space positions of the three vertices — Gouraud
/// degenerates to flat in that case, but per-pixel Phong still
/// interpolates a clean direction across the triangle.
fn vertex_normals_for_face(
    prim: &Primitive,
    world: &[[f32; 4]; 4],
    world_normals: Option<&[[f32; 3]]>,
    a: usize,
    b: usize,
    c: usize,
) -> ([f32; 3], [f32; 3], [f32; 3]) {
    if let Some(ns) = world_normals {
        if a < ns.len() && b < ns.len() && c < ns.len() {
            return (ns[a], ns[b], ns[c]);
        }
    }
    // Fall back to face normal in world space.
    let pa = prim.positions.get(a).copied().unwrap_or([0.0, 0.0, 0.0]);
    let pb = prim.positions.get(b).copied().unwrap_or([0.0, 0.0, 0.0]);
    let pc = prim.positions.get(c).copied().unwrap_or([0.0, 0.0, 0.0]);
    let wa = mat4_mul_point(world, pa);
    let wb = mat4_mul_point(world, pb);
    let wc = mat4_mul_point(world, pc);
    let n = vec3_normalise(vec3_cross(vec3_sub(wb, wa), vec3_sub(wc, wa)));
    (n, n, n)
}

fn draw_line_pair(
    projected: &[Option<[f32; 3]>],
    a: usize,
    b: usize,
    colour: [u8; 4],
    fb: &mut Framebuffer,
) {
    let (Some(va), Some(vb)) = (
        projected.get(a).and_then(|v| *v),
        projected.get(b).and_then(|v| *v),
    ) else {
        return;
    };
    draw_line(fb, va, vb, colour);
}

/// Project an object-space vertex through the model-view-projection
/// matrix, perspective-divide, and map to viewport pixel coords.
/// Returns `None` for vertices behind the near plane (`w <= 0`); the
/// caller skips any triangle / line that touches a clipped vertex.
fn project_vertex(pos: [f32; 3], mvp: &[[f32; 4]; 4], width: f32, height: f32) -> Option<[f32; 3]> {
    let v = mat4_mul_vec4(mvp, [pos[0], pos[1], pos[2], 1.0]);
    if v[3] <= 0.0 {
        return None;
    }
    let ndc_x = v[0] / v[3];
    let ndc_y = v[1] / v[3];
    let ndc_z = v[2] / v[3];
    let sx = (ndc_x * 0.5 + 0.5) * width;
    let sy = (1.0 - (ndc_y * 0.5 + 0.5)) * height;
    Some([sx, sy, ndc_z])
}

// ---------------------------------------------------------------------
// Per-shading-mode triangle rasterisers (half-space edge functions).
// ---------------------------------------------------------------------

/// Flat shading — every covered pixel takes the same sRGB colour.
fn rasterise_triangle_flat(
    fb: &mut Framebuffer,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    colour: [u8; 4],
) {
    let area = edge(a, b, c);
    if area.abs() < 1.0e-6 {
        return;
    }
    let (min_x, min_y, max_x, max_y) = tri_bbox(fb, a, b, c);
    let area_inv = 1.0 / area;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = [x as f32 + 0.5, y as f32 + 0.5, 0.0];
            let w0 = edge(b, c, p) * area_inv;
            let w1 = edge(c, a, p) * area_inv;
            let w2 = edge(a, b, p) * area_inv;
            let inside =
                (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0);
            if !inside {
                continue;
            }
            let z = w0 * a[2] + w1 * b[2] + w2 * c[2];
            fb.set_pixel(x, y, z, colour);
        }
    }
}

/// Gouraud shading — bilinearly interpolate the per-vertex (already
/// lit) colour across the triangle.
fn rasterise_triangle_gouraud(
    fb: &mut Framebuffer,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    ca: [f32; 4],
    cb: [f32; 4],
    cc: [f32; 4],
) {
    let area = edge(a, b, c);
    if area.abs() < 1.0e-6 {
        return;
    }
    let (min_x, min_y, max_x, max_y) = tri_bbox(fb, a, b, c);
    let area_inv = 1.0 / area;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = [x as f32 + 0.5, y as f32 + 0.5, 0.0];
            let w0 = edge(b, c, p) * area_inv;
            let w1 = edge(c, a, p) * area_inv;
            let w2 = edge(a, b, p) * area_inv;
            let inside =
                (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0);
            if !inside {
                continue;
            }
            let z = w0 * a[2] + w1 * b[2] + w2 * c[2];
            let r = w0 * ca[0] + w1 * cb[0] + w2 * cc[0];
            let g = w0 * ca[1] + w1 * cb[1] + w2 * cc[1];
            let bl = w0 * ca[2] + w1 * cb[2] + w2 * cc[2];
            let al = w0 * ca[3] + w1 * cb[3] + w2 * cc[3];
            let pix = linear_rgba_to_srgb_u8([r, g, bl, al]);
            fb.set_pixel(x, y, z, pix);
        }
    }
}

/// Phong shading — interpolate the per-vertex normal across the
/// triangle, evaluate the lighting equation at every pixel.
#[allow(clippy::too_many_arguments)]
fn rasterise_triangle_phong(
    fb: &mut Framebuffer,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    na: [f32; 3],
    nb: [f32; 3],
    nc: [f32; 3],
    colour_linear: [f32; 4],
    light: &DirLight,
) {
    let area = edge(a, b, c);
    if area.abs() < 1.0e-6 {
        return;
    }
    let (min_x, min_y, max_x, max_y) = tri_bbox(fb, a, b, c);
    let area_inv = 1.0 / area;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = [x as f32 + 0.5, y as f32 + 0.5, 0.0];
            let w0 = edge(b, c, p) * area_inv;
            let w1 = edge(c, a, p) * area_inv;
            let w2 = edge(a, b, p) * area_inv;
            let inside =
                (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0);
            if !inside {
                continue;
            }
            let z = w0 * a[2] + w1 * b[2] + w2 * c[2];
            let nx = w0 * na[0] + w1 * nb[0] + w2 * nc[0];
            let ny = w0 * na[1] + w1 * nb[1] + w2 * nc[1];
            let nz = w0 * na[2] + w1 * nb[2] + w2 * nc[2];
            let normal = vec3_normalise([nx, ny, nz]);
            let lit = shade_pixel(colour_linear, normal, light);
            fb.set_pixel(x, y, z, linear_rgba_to_srgb_u8(lit));
        }
    }
}

/// NormalDebug — interpolate the per-vertex normal across the triangle
/// and write `(n + 1) / 2 * 255` per channel. Matches the classic
/// "normal map" colour-key. Lighting / material settings ignored.
fn rasterise_triangle_normal_debug(
    fb: &mut Framebuffer,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    na: [f32; 3],
    nb: [f32; 3],
    nc: [f32; 3],
) {
    let area = edge(a, b, c);
    if area.abs() < 1.0e-6 {
        return;
    }
    let (min_x, min_y, max_x, max_y) = tri_bbox(fb, a, b, c);
    let area_inv = 1.0 / area;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = [x as f32 + 0.5, y as f32 + 0.5, 0.0];
            let w0 = edge(b, c, p) * area_inv;
            let w1 = edge(c, a, p) * area_inv;
            let w2 = edge(a, b, p) * area_inv;
            let inside =
                (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0);
            if !inside {
                continue;
            }
            let z = w0 * a[2] + w1 * b[2] + w2 * c[2];
            let nx = w0 * na[0] + w1 * nb[0] + w2 * nc[0];
            let ny = w0 * na[1] + w1 * nb[1] + w2 * nc[1];
            let nz = w0 * na[2] + w1 * nb[2] + w2 * nc[2];
            let n = vec3_normalise([nx, ny, nz]);
            let pix = [
                normal_to_byte(n[0]),
                normal_to_byte(n[1]),
                normal_to_byte(n[2]),
                255,
            ];
            fb.set_pixel(x, y, z, pix);
        }
    }
}

/// Map a single normal component in `[-1, 1]` into a `u8` colour
/// channel via `(n + 1) / 2 * 255`. NaNs and zero-length normals fall
/// back to `128` (the encoded zero).
fn normal_to_byte(n: f32) -> u8 {
    if !n.is_finite() {
        return 128;
    }
    let v = ((n.clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0).round();
    v.clamp(0.0, 255.0) as u8
}

/// DepthDebug — paint each pixel a grayscale value derived from the
/// interpolated NDC z. Near (-1) → white (255), far (+1) → black (0).
fn rasterise_triangle_depth_debug(fb: &mut Framebuffer, a: [f32; 3], b: [f32; 3], c: [f32; 3]) {
    let area = edge(a, b, c);
    if area.abs() < 1.0e-6 {
        return;
    }
    let (min_x, min_y, max_x, max_y) = tri_bbox(fb, a, b, c);
    let area_inv = 1.0 / area;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = [x as f32 + 0.5, y as f32 + 0.5, 0.0];
            let w0 = edge(b, c, p) * area_inv;
            let w1 = edge(c, a, p) * area_inv;
            let w2 = edge(a, b, p) * area_inv;
            let inside =
                (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0);
            if !inside {
                continue;
            }
            let z = w0 * a[2] + w1 * b[2] + w2 * c[2];
            let g = depth_to_byte(z);
            fb.set_pixel(x, y, z, [g, g, g, 255]);
        }
    }
}

/// Map an NDC z value (`[-1, 1]`, near = -1) to a grayscale byte where
/// near = 255 and far = 0. NaN / out-of-range values are clamped.
fn depth_to_byte(z: f32) -> u8 {
    if !z.is_finite() {
        return 0;
    }
    let zc = z.clamp(-1.0, 1.0);
    let v = ((1.0 - (zc * 0.5 + 0.5)) * 255.0).round();
    v.clamp(0.0, 255.0) as u8
}

fn tri_bbox(fb: &Framebuffer, a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> (i32, i32, i32, i32) {
    let min_x = a[0].min(b[0]).min(c[0]).max(0.0).floor() as i32;
    let min_y = a[1].min(b[1]).min(c[1]).max(0.0).floor() as i32;
    let max_x = a[0].max(b[0]).max(c[0]).min(fb.width as f32 - 1.0).ceil() as i32;
    let max_y = a[1].max(b[1]).max(c[1]).min(fb.height as f32 - 1.0).ceil() as i32;
    (min_x, min_y, max_x, max_y)
}

/// Lambertian diffuse + ambient. `normal` is in world space, `light`
/// carries a unit direction TOWARD the light. Result is in linear
/// space, alpha pulled straight from the base colour.
fn shade_pixel(base: [f32; 4], normal: [f32; 3], light: &DirLight) -> [f32; 4] {
    let cos_theta = vec3_dot(normal, light.direction).max(0.0);
    let diffuse = AMBIENT + (1.0 - AMBIENT) * cos_theta * light.intensity;
    let factor = diffuse.clamp(0.0, 1.0);
    [
        base[0] * factor,
        base[1] * factor,
        base[2] * factor,
        base[3],
    ]
}

/// Bresenham line on a screen-space pair of projected vertices. Used
/// by [`ShadingMode::Wireframe`] and the `Lines*` topologies.
fn draw_line(fb: &mut Framebuffer, a: [f32; 3], b: [f32; 3], colour: [u8; 4]) {
    let mut x0 = a[0].round() as i32;
    let mut y0 = a[1].round() as i32;
    let x1 = b[0].round() as i32;
    let y1 = b[1].round() as i32;
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let len = ((dx as f32).hypot(dy as f32)).max(1.0);
    let dz = b[2] - a[2];
    loop {
        let t =
            (((x0 - a[0].round() as i32) as f32).hypot((y0 - a[1].round() as i32) as f32)) / len;
        let z = a[2] + dz * t;
        fb.set_pixel(x0, y0, z, colour);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

#[inline]
fn edge(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

// ---------------------------------------------------------------------
// Light helper.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct DirLight {
    /// Unit vector pointing TOWARD the light source from the surface.
    direction: [f32; 3],
    intensity: f32,
}

fn build_light(spec: LightSpec) -> DirLight {
    let az = spec.azimuth_deg.to_radians();
    let el = spec.elevation_deg.to_radians();
    let cos_el = el.cos();
    let dir = [cos_el * az.sin(), el.sin(), cos_el * az.cos()];
    DirLight {
        direction: vec3_normalise(dir),
        intensity: spec.intensity,
    }
}

// ---------------------------------------------------------------------
// 4x4 matrix helpers (column-vector convention, row-major storage).
// ---------------------------------------------------------------------

fn identity4() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0_f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] =
                a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    out
}

fn mat4_mul_vec4(m: &[[f32; 4]; 4], v: [f32; 4]) -> [f32; 4] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3] * v[3],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3] * v[3],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3] * v[3],
        m[3][0] * v[0] + m[3][1] * v[1] + m[3][2] * v[2] + m[3][3] * v[3],
    ]
}

fn mat4_mul_point(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    let v = mat4_mul_vec4(m, [p[0], p[1], p[2], 1.0]);
    if v[3].abs() > f32::EPSILON {
        [v[0] / v[3], v[1] / v[3], v[2] / v[3]]
    } else {
        [v[0], v[1], v[2]]
    }
}

/// Multiply the 3x3 upper-left of `m` by `v`. Used to transform
/// directions (normals) without picking up the translation column.
fn mat3_mul_vec3(m: &[[f32; 4]; 4], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

// ---------------------------------------------------------------------
// Camera framing.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Camera {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
}

impl Camera {
    /// Build a camera honouring [`RenderOptions::camera`] /
    /// [`RenderOptions::projection`] / [`RenderOptions::fov_deg`].
    fn build(width: u32, height: u32, bbox: BBox, opts: &RenderOptions) -> Self {
        let (cx, cy, cz) = (
            (bbox.min[0] + bbox.max[0]) * 0.5,
            (bbox.min[1] + bbox.max[1]) * 0.5,
            (bbox.min[2] + bbox.max[2]) * 0.5,
        );
        let extent = ((bbox.max[0] - bbox.min[0]).max(bbox.max[1] - bbox.min[1]))
            .max(bbox.max[2] - bbox.min[2])
            .max(1.0e-3);
        let aspect = (width.max(1) as f32) / (height.max(1) as f32);
        let radius = extent * 0.5 * 1.2;

        let projection = opts.projection;
        let fov_y = opts.fov_deg.to_radians();

        // Camera placement.
        let (eye, dist_units) = match opts.camera {
            Some(cam) => Self::user_orbit(cam, projection, fov_y, radius, extent, [cx, cy, cz]),
            None => {
                // Default: look down +Z toward -Z.
                let dist = match projection {
                    Projection::Perspective => radius / (fov_y * 0.5).tan(),
                    Projection::Orthographic => extent * 1.5,
                };
                ([cx, cy, cz + dist], dist)
            }
        };

        let target = [cx, cy, cz];
        let up = [0.0, 1.0, 0.0];
        let view = look_at(eye, target, up);

        let proj = match projection {
            Projection::Perspective => {
                let near = (dist_units - extent).max(extent * 0.01);
                let far = dist_units + extent * 2.0;
                perspective(fov_y, aspect, near, far)
            }
            Projection::Orthographic => {
                // Frame the full extent on the smaller axis with a
                // 1.2x margin (matching the perspective fit).
                let half = radius;
                let half_w = if aspect >= 1.0 { half * aspect } else { half };
                let half_h = if aspect >= 1.0 { half } else { half / aspect };
                let near = (dist_units - extent * 2.0).min(-extent);
                let far = dist_units + extent * 2.0;
                orthographic(-half_w, half_w, -half_h, half_h, near, far)
            }
        };
        Self { view, proj }
    }

    fn user_orbit(
        cam: CameraSpec,
        projection: Projection,
        fov_y: f32,
        radius: f32,
        extent: f32,
        center: [f32; 3],
    ) -> ([f32; 3], f32) {
        let el = cam.elevation_deg.to_radians();
        let az = cam.azimuth_deg.to_radians();
        // `cam.distance` is a multiplier of the bounding-sphere radius
        // auto-frame distance.
        let auto_dist = match projection {
            Projection::Perspective => radius / (fov_y * 0.5).tan(),
            Projection::Orthographic => extent * 1.5,
        };
        let dist = auto_dist * cam.distance;
        let cos_el = el.cos();
        let dir = [cos_el * az.sin(), el.sin(), cos_el * az.cos()];
        let eye = [
            center[0] + dir[0] * dist,
            center[1] + dir[1] * dist,
            center[2] + dir[2] * dist,
        ];
        (eye, dist)
    }
}

fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = vec3_normalise(vec3_sub(target, eye));
    let s = vec3_normalise(vec3_cross(f, up));
    let u = vec3_cross(s, f);
    [
        [s[0], s[1], s[2], -vec3_dot(s, eye)],
        [u[0], u[1], u[2], -vec3_dot(u, eye)],
        [-f[0], -f[1], -f[2], vec3_dot(f, eye)],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_y * 0.5).tan();
    let nf = 1.0 / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) * nf, 2.0 * far * near * nf],
        [0.0, 0.0, -1.0, 0.0],
    ]
}

/// Standard right-handed orthographic projection matrix (OpenGL
/// convention; outputs `w = 1` so the perspective-divide in
/// [`project_vertex`] is a no-op).
fn orthographic(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> [[f32; 4]; 4] {
    let rl = right - left;
    let tb = top - bottom;
    let fne = far - near;
    [
        [2.0 / rl, 0.0, 0.0, -(right + left) / rl],
        [0.0, 2.0 / tb, 0.0, -(top + bottom) / tb],
        [0.0, 0.0, -2.0 / fne, -(far + near) / fne],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn vec3_normalise(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < f32::EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

// ---------------------------------------------------------------------
// Scene bbox walk.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct BBox {
    min: [f32; 3],
    max: [f32; 3],
}

impl BBox {
    fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
        }
    }
    fn extend(&mut self, p: [f32; 3]) {
        for (i, &v) in p.iter().enumerate() {
            if v < self.min[i] {
                self.min[i] = v;
            }
            if v > self.max[i] {
                self.max[i] = v;
            }
        }
    }
    fn is_empty(&self) -> bool {
        self.min[0] > self.max[0]
    }
}

fn scene_bbox(scene: &Scene3D) -> BBox {
    let mut bbox = BBox::empty();
    for &root in &scene.roots {
        accumulate_node_bbox(scene, root, identity4(), &mut bbox);
    }
    if bbox.is_empty() {
        // Centre on origin with a unit half-extent so the camera math
        // stays finite when the scene has no geometry.
        BBox {
            min: [-0.5, -0.5, -0.5],
            max: [0.5, 0.5, 0.5],
        }
    } else {
        bbox
    }
}

fn accumulate_node_bbox(scene: &Scene3D, id: NodeId, parent_world: [[f32; 4]; 4], bbox: &mut BBox) {
    let Some(node) = scene.nodes.get(id.0 as usize) else {
        return;
    };
    let world = mat4_mul(parent_world, node.transform.to_matrix());
    if let Some(mesh_id) = node.mesh {
        if let Some(mesh) = scene.meshes.get(mesh_id.0 as usize) {
            for prim in &mesh.primitives {
                for p in &prim.positions {
                    let v = mat4_mul_vec4(&world, [p[0], p[1], p[2], 1.0]);
                    if v[3].abs() > f32::EPSILON {
                        bbox.extend([v[0] / v[3], v[1] / v[3], v[2] / v[3]]);
                    }
                }
            }
        }
    }
    let children: Vec<NodeId> = scene
        .nodes
        .get(id.0 as usize)
        .map(|n: &Node| n.children.clone())
        .unwrap_or_default();
    for child in children {
        accumulate_node_bbox(scene, child, world, bbox);
    }
}

// ---------------------------------------------------------------------
// Framebuffer.
// ---------------------------------------------------------------------

struct Framebuffer {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
    zbuf: Vec<f32>,
}

impl Framebuffer {
    fn new(width: u32, height: u32, background: [u8; 4]) -> Self {
        let n = (width as usize) * (height as usize);
        let mut rgba = Vec::with_capacity(n * 4);
        for _ in 0..n {
            rgba.extend_from_slice(&background);
        }
        Self {
            width,
            height,
            rgba,
            zbuf: vec![f32::INFINITY; n],
        }
    }

    fn set_pixel(&mut self, x: i32, y: i32, z: f32, colour: [u8; 4]) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let idx = (y as usize) * (self.width as usize) + (x as usize);
        if z < self.zbuf[idx] {
            self.zbuf[idx] = z;
            let p = idx * 4;
            self.rgba[p] = colour[0];
            self.rgba[p + 1] = colour[1];
            self.rgba[p + 2] = colour[2];
            self.rgba[p + 3] = colour[3];
        }
    }

    fn into_image(self) -> RgbaImage {
        RgbaImage {
            width: self.width,
            height: self.height,
            stride: (self.width as usize) * 4,
            pixels: self.rgba,
        }
    }
}

// ---------------------------------------------------------------------
// Tests — round-tripped from cli-convert's mesh3d_render tests with
// adapted option types.
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::{BackgroundColor, CameraSpec};
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
    fn linear_to_srgb_handles_endpoints() {
        assert_eq!(linear_to_srgb_u8(0.0), 0);
        assert_eq!(linear_to_srgb_u8(1.0), 255);
    }

    #[test]
    fn camera_with_user_override_is_finite() {
        let bbox = BBox {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };
        let opts = RenderOptions {
            camera: Some(CameraSpec {
                elevation_deg: 30.0,
                azimuth_deg: 45.0,
                distance: 1.5,
            }),
            ..RenderOptions::default()
        };
        let cam = Camera::build(64, 64, bbox, &opts);
        for row in cam.view {
            for v in row {
                assert!(v.is_finite(), "camera view matrix component not finite");
            }
        }
    }

    #[test]
    fn ortho_projection_matrix_has_zero_w_translation() {
        let bbox = BBox {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };
        let opts = RenderOptions {
            projection: Projection::Orthographic,
            ..RenderOptions::default()
        };
        let cam = Camera::build(64, 64, bbox, &opts);
        // Ortho proj's bottom row is [0, 0, 0, 1] (no perspective divide).
        assert_eq!(cam.proj[3], [0.0, 0.0, 0.0, 1.0]);
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

    #[test]
    fn downsample_box_averages_uniform_field() {
        let src = RgbaImage {
            width: 4,
            height: 4,
            stride: 16,
            pixels: vec![
                200, 100, 50, 255, 200, 100, 50, 255, 200, 100, 50, 255, 200, 100, 50, 255, 200,
                100, 50, 255, 200, 100, 50, 255, 200, 100, 50, 255, 200, 100, 50, 255, 200, 100,
                50, 255, 200, 100, 50, 255, 200, 100, 50, 255, 200, 100, 50, 255, 200, 100, 50,
                255, 200, 100, 50, 255, 200, 100, 50, 255, 200, 100, 50, 255,
            ],
        };
        let dst = downsample_box(&src, 2, 2, 2);
        assert_eq!(dst.width, 2);
        assert_eq!(dst.height, 2);
        for px in dst.pixels.chunks_exact(4) {
            assert_eq!(px, &[200, 100, 50, 255]);
        }
    }

    #[test]
    fn downsample_box_averages_split_field() {
        let src = RgbaImage {
            width: 2,
            height: 2,
            stride: 8,
            pixels: vec![
                0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255,
            ],
        };
        let dst = downsample_box(&src, 1, 1, 2);
        assert_eq!(dst.width, 1);
        assert_eq!(dst.height, 1);
        assert_eq!(dst.pixels[0], 127);
        assert_eq!(dst.pixels[1], 127);
        assert_eq!(dst.pixels[2], 127);
        assert_eq!(dst.pixels[3], 255);
    }
}
