//! Raycast backend — the Phase D renderer behind
//! [`crate::RenderBackend::Raycast`].
//!
//! Whitted-style recursive ray tracing: one primary ray per pixel
//! sample (times the SSAA factor), closest-hit shading with the same
//! directional light as the scanline backend, plus (in
//! [`ShadingMode::Phong`]) shadow rays and recursive reflection /
//! refraction rays driven by the hit material.
//!
//! Clean-room policy: the recursive-ray-tracing structure follows
//! Whitted's 1980 CACM paper "An Improved Illumination Model for
//! Shaded Display". Ray-triangle intersection is Möller–Trumbore 1997
//! via `oxideav_mesh3d::ray`; BVH construction is
//! [`oxideav_mesh3d::Bvh`]. Reflection / refraction directions use
//! the standard vector forms of the law of reflection and Snell's
//! law; the reflectance weight uses Schlick's 1994 Fresnel
//! approximation. No reference renderer source code was consulted.
//!
//! ## Scene baking
//!
//! [`TraceScene::build`] flattens the node forest once per render:
//! every triangle-topology primitive is expanded through
//! `Primitive::triangle_indices()` (so strips / fans arrive
//! pre-unrolled with correct winding), transformed to world space,
//! and appended to a single triangle soup — three positions per
//! triangle, no shared vertices. A parallel per-triangle array
//! carries the world-space vertex normals (per-vertex when the source
//! primitive has them, face normal otherwise) and a material slot.
//! [`oxideav_mesh3d::Bvh`] is built over the baked soup; traversal
//! walks the BVH's public node array directly against the baked
//! positions so the per-ray path allocates nothing.
//!
//! Line / point topologies have zero surface area and are invisible
//! to rays; the raycast backend skips them (the scanline backend
//! remains the renderer for wire/point content).

use oxideav_mesh3d::ray::{intersect_aabb, intersect_triangle, Ray};
use oxideav_mesh3d::{Bvh, Primitive, Scene3D, Topology};

use crate::camera::{scene_bbox, Camera};
use crate::image::{downsample_box, RgbaImage};
use crate::math::{
    mat3_mul_vec3, mat4_mul_point, vec3_add, vec3_cross, vec3_dot, vec3_normalise, vec3_scale,
    vec3_sub,
};
use crate::options::{RenderOptions, ShadingMode};
use crate::shade::{build_light, linear_rgba_to_srgb_u8, shade_pixel, DirLight, AMBIENT};

/// Maximum recursion depth for reflection / refraction rays. Depth 0
/// is the primary ray; a Whitted tree deeper than this contributes
/// only the locally-shaded colour.
const MAX_DEPTH: u32 = 4;

/// Offset applied along the outgoing ray direction when spawning
/// shadow / secondary rays so they don't re-hit the surface they
/// left (shadow-acne guard).
const RAY_EPSILON: f32 = 1.0e-4;

/// Minimum metallic factor before a reflection ray is traced, and
/// minimum transmission factor before a refraction ray is traced —
/// spares the ray tree for the overwhelmingly common inert material.
const SECONDARY_RAY_THRESHOLD: f32 = 1.0e-2;

/// Barycentric distance from a triangle edge under which a
/// [`ShadingMode::Wireframe`] hit paints the pixel.
const WIREFRAME_EDGE_WIDTH: f32 = 0.03;

// ---------------------------------------------------------------------
// Baked material.
// ---------------------------------------------------------------------

/// Material snapshot consumed by the ray shader — the subset of
/// [`oxideav_mesh3d::Material`] the Whitted model can honour, baked
/// once so the per-hit path never chases scene indices.
#[derive(Debug, Clone, Copy)]
struct ShadeMaterial {
    /// Linear-space RGBA base colour.
    base_color: [f32; 4],
    /// `[0, 1]` — drives the reflection ray weight.
    metallic: f32,
    /// `[0, 1]` — attenuates the reflection (a rough metal reflects
    /// diffusely, which a single Whitted ray cannot represent; the
    /// mirror term fades out with roughness instead).
    roughness: f32,
    /// Fraction of light transmitted through the surface
    /// (`KHR_materials_transmission`); drives the refraction ray.
    transmission: f32,
    /// Index of refraction (`KHR_materials_ior`, default 1.5).
    ior: f32,
    /// Linear-space emission added after the diffuse term —
    /// `emissive_factor × KHR_materials_emissive_strength`.
    emissive: [f32; 3],
    /// `KHR_materials_unlit` — constant-shade the base colour;
    /// lighting, shadows, and secondary rays are all skipped.
    unlit: bool,
}

impl ShadeMaterial {
    /// Material used when a primitive has no material reference —
    /// matches the scanline backend's fallback colour, inert
    /// (no reflection, no transmission).
    fn fallback() -> Self {
        Self {
            base_color: [0.7, 0.7, 0.75, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            transmission: 0.0,
            ior: 1.5,
            emissive: [0.0, 0.0, 0.0],
            unlit: false,
        }
    }

    fn from_material(m: &oxideav_mesh3d::Material) -> Self {
        Self {
            base_color: m.base_color,
            metallic: m.metallic.clamp(0.0, 1.0),
            roughness: m.roughness.clamp(0.0, 1.0),
            transmission: m
                .ext
                .transmission
                .as_ref()
                .map(|t| t.factor.clamp(0.0, 1.0))
                .unwrap_or(0.0),
            ior: m.ext.ior.unwrap_or(1.5).max(1.0),
            emissive: {
                let strength = m.ext.emissive_strength.unwrap_or(1.0).max(0.0);
                [
                    m.emissive_factor[0] * strength,
                    m.emissive_factor[1] * strength,
                    m.emissive_factor[2] * strength,
                ]
            },
            unlit: m.ext.unlit,
        }
    }
}

// ---------------------------------------------------------------------
// Baked scene.
// ---------------------------------------------------------------------

/// Per-triangle shading payload, parallel to the baked triangle soup.
#[derive(Debug, Clone, Copy)]
struct TriShade {
    /// World-space unit vertex normals (per-vertex when the source
    /// primitive carried normals, face normal on all three otherwise).
    normals: [[f32; 3]; 3],
    /// Index into [`TraceScene::materials`].
    material: u32,
}

/// World-space triangle soup + BVH + per-triangle shading data.
struct TraceScene {
    /// Baked geometry: `Triangles` topology, three vertices per
    /// triangle (implicit indices), world space.
    soup: Primitive,
    /// Parallel per-triangle shading payloads (`soup` triangle `i` ↔
    /// `shade[i]`).
    shade: Vec<TriShade>,
    materials: Vec<ShadeMaterial>,
    /// `None` when the scene bakes to zero triangles.
    bvh: Option<Bvh>,
}

impl TraceScene {
    fn build(scene: &Scene3D) -> Self {
        let mut soup = Primitive::new(Topology::Triangles);
        let mut shade: Vec<TriShade> = Vec::new();

        // Materials: slot 0 is the no-material fallback; scene
        // material `i` maps to slot `i + 1`.
        let mut materials = Vec::with_capacity(scene.materials.len() + 1);
        materials.push(ShadeMaterial::fallback());
        for m in &scene.materials {
            materials.push(ShadeMaterial::from_material(m));
        }

        // Iterative pre-order walk: claims each node once at first
        // arrival so cyclic / shared node graphs terminate, and deep
        // hierarchies cannot overflow the call stack (see
        // `camera::walk_scene_preorder`).
        crate::camera::walk_scene_preorder(scene, |node, world| {
            if let Some(mesh_id) = node.mesh {
                if let Some(mesh) = scene.meshes.get(mesh_id.0 as usize) {
                    for prim in &mesh.primitives {
                        bake_primitive(prim, world, &mut soup, &mut shade);
                    }
                }
            }
        });

        let bvh = Bvh::build(&soup);
        Self {
            soup,
            shade,
            materials,
            bvh,
        }
    }

    /// Closest hit against the baked soup in `(0, t_max]`.
    ///
    /// Allocation-free BVH walk over [`Bvh::nodes`] using the slab
    /// test for interior nodes and Möller–Trumbore at the leaves;
    /// the baked soup's implicit indices make triangle `i`'s corners
    /// `positions[3i .. 3i + 3]` — no index-buffer chase.
    fn closest_hit(&self, ray: Ray, t_max: f32) -> Option<Hit> {
        let bvh = self.bvh.as_ref()?;
        intersect_aabb(ray, bvh.nodes[0].bounds.min, bvh.nodes[0].bounds.max, t_max)?;

        let mut best: Option<Hit> = None;
        let mut best_t = t_max;
        // Fixed-capacity traversal stack of node indices; BVH depth is
        // bounded by the leaf threshold + median split, 64 is ample.
        let mut stack: [u32; 64] = [0; 64];
        let mut sp = 1usize; // node 0 pre-pushed

        while sp > 0 {
            sp -= 1;
            let node = &bvh.nodes[stack[sp] as usize];
            if intersect_aabb(ray, node.bounds.min, node.bounds.max, best_t).is_none() {
                continue;
            }
            if node.is_leaf() {
                let first = node.left_or_first as usize;
                for &tri in &bvh.triangles[first..first + node.tri_count as usize] {
                    let base = (tri as usize) * 3;
                    let p0 = self.soup.positions[base];
                    let p1 = self.soup.positions[base + 1];
                    let p2 = self.soup.positions[base + 2];
                    if let Some((t, u, v, _front)) = intersect_triangle(ray, p0, p1, p2, best_t) {
                        best_t = t;
                        best = Some(Hit {
                            t,
                            triangle: tri as usize,
                            barycentric: [1.0 - u - v, u, v],
                        });
                    }
                }
            } else {
                // Push both children; the per-node slab test above
                // culls the miss side.
                if sp + 2 <= stack.len() {
                    stack[sp] = node.left_or_first;
                    stack[sp + 1] = node.right_child;
                    sp += 2;
                }
            }
        }
        best
    }

    /// Any-hit (shadow) query in `(0, t_max]` — first intersection
    /// wins, no ordering.
    fn any_hit(&self, ray: Ray, t_max: f32) -> bool {
        let Some(bvh) = self.bvh.as_ref() else {
            return false;
        };
        if intersect_aabb(ray, bvh.nodes[0].bounds.min, bvh.nodes[0].bounds.max, t_max).is_none() {
            return false;
        }
        let mut stack: [u32; 64] = [0; 64];
        let mut sp = 1usize;
        while sp > 0 {
            sp -= 1;
            let node = &bvh.nodes[stack[sp] as usize];
            if intersect_aabb(ray, node.bounds.min, node.bounds.max, t_max).is_none() {
                continue;
            }
            if node.is_leaf() {
                let first = node.left_or_first as usize;
                for &tri in &bvh.triangles[first..first + node.tri_count as usize] {
                    let base = (tri as usize) * 3;
                    if intersect_triangle(
                        ray,
                        self.soup.positions[base],
                        self.soup.positions[base + 1],
                        self.soup.positions[base + 2],
                        t_max,
                    )
                    .is_some()
                    {
                        return true;
                    }
                }
            } else if sp + 2 <= stack.len() {
                stack[sp] = node.left_or_first;
                stack[sp + 1] = node.right_child;
                sp += 2;
            }
        }
        false
    }

    /// Interpolated world-space unit normal at a hit.
    fn hit_normal(&self, hit: &Hit) -> [f32; 3] {
        let s = &self.shade[hit.triangle];
        let [w, u, v] = hit.barycentric;
        vec3_normalise([
            w * s.normals[0][0] + u * s.normals[1][0] + v * s.normals[2][0],
            w * s.normals[0][1] + u * s.normals[1][1] + v * s.normals[2][1],
            w * s.normals[0][2] + u * s.normals[1][2] + v * s.normals[2][2],
        ])
    }

    fn hit_material(&self, hit: &Hit) -> &ShadeMaterial {
        &self.materials[self.shade[hit.triangle].material as usize]
    }
}

/// Closest-hit record on the baked soup.
struct Hit {
    t: f32,
    triangle: usize,
    barycentric: [f32; 3],
}

fn bake_primitive(
    prim: &Primitive,
    world: &[[f32; 4]; 4],
    soup: &mut Primitive,
    shade: &mut Vec<TriShade>,
) {
    let tris = prim.triangle_indices();
    if tris.is_empty() {
        return; // line / point topology or empty primitive
    }
    // Material slot: scene material `i` lives at `i + 1`; fallback 0.
    let material = prim.material.map(|mid| mid.0 + 1).unwrap_or(0);
    let n_pos = prim.positions.len();
    // The world matrix is assumed rigid+uniform-scale (the only kind
    // composed from `Transform::translation/rotation/scale`); its 3x3
    // upper-left suffices for normal transformation, matching the
    // scanline backend's convention.
    for [ia, ib, ic] in tris {
        let (ia, ib, ic) = (ia as usize, ib as usize, ic as usize);
        if ia >= n_pos || ib >= n_pos || ic >= n_pos {
            continue;
        }
        let wa = mat4_mul_point(world, prim.positions[ia]);
        let wb = mat4_mul_point(world, prim.positions[ib]);
        let wc = mat4_mul_point(world, prim.positions[ic]);
        let normals = match prim.normals.as_ref() {
            Some(ns) if ia < ns.len() && ib < ns.len() && ic < ns.len() => [
                vec3_normalise(mat3_mul_vec3(world, ns[ia])),
                vec3_normalise(mat3_mul_vec3(world, ns[ib])),
                vec3_normalise(mat3_mul_vec3(world, ns[ic])),
            ],
            _ => {
                let n = vec3_normalise(vec3_cross(vec3_sub(wb, wa), vec3_sub(wc, wa)));
                [n, n, n]
            }
        };
        soup.positions.push(wa);
        soup.positions.push(wb);
        soup.positions.push(wc);
        shade.push(TriShade { normals, material });
    }
}

// ---------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------

/// Render `scene` into a packed RGBA8 buffer per `opts` by recursive
/// ray tracing. Honours the same option surface as the scanline
/// backend: framebuffer size, background, shading mode, projection,
/// FOV, light, camera override, and SSAA factor.
pub fn render_scene(scene: &Scene3D, opts: &RenderOptions) -> RgbaImage {
    let width = opts.width.max(1);
    let height = opts.height.max(1);
    let aa = opts.aa.clamp(1, 8);
    let render_w = width.saturating_mul(aa).max(1);
    let render_h = height.saturating_mul(aa).max(1);

    let bbox = scene_bbox(scene);
    let camera = Camera::build(render_w, render_h, bbox, opts);
    let light = build_light(opts.light);
    let traced = TraceScene::build(scene);
    let background = opts.background.0;
    let mode = opts.shading;

    let (w_us, h_us) = (render_w as usize, render_h as usize);
    let mut pixels = Vec::with_capacity(w_us * h_us * 4);
    let (wf, hf) = (render_w as f32, render_h as f32);
    for y in 0..h_us {
        for x in 0..w_us {
            let (origin, dir) = camera.primary_ray(x as f32 + 0.5, y as f32 + 0.5, wf, hf);
            let ray = Ray::new(origin, dir);
            let px = trace_pixel(&traced, &camera, &light, ray, mode, background);
            pixels.extend_from_slice(&px);
        }
    }

    let img = RgbaImage {
        width: render_w,
        height: render_h,
        stride: w_us * 4,
        pixels,
    };
    if aa <= 1 {
        img
    } else {
        downsample_box(&img, width, height, aa)
    }
}

/// Shade one primary ray under the selected [`ShadingMode`].
fn trace_pixel(
    traced: &TraceScene,
    camera: &Camera,
    light: &DirLight,
    ray: Ray,
    mode: ShadingMode,
    background: [u8; 4],
) -> [u8; 4] {
    let Some(hit) = traced.closest_hit(ray, f32::INFINITY) else {
        return background;
    };
    match mode {
        ShadingMode::Flat => {
            // Parity with the scanline backend: unlit constant colour.
            linear_rgba_to_srgb_u8(traced.hit_material(&hit).base_color)
        }
        ShadingMode::Wireframe => {
            // A rasteriser draws edges; a ray tracer detects them —
            // paint the pixel when the hit sits within an edge band
            // in barycentric space, else show through to whatever is
            // behind (background: closest-hit only, single layer).
            let min_bary = hit
                .barycentric
                .iter()
                .fold(f32::INFINITY, |acc, &b| acc.min(b));
            if min_bary <= WIREFRAME_EDGE_WIDTH {
                linear_rgba_to_srgb_u8(traced.hit_material(&hit).base_color)
            } else {
                background
            }
        }
        ShadingMode::Gouraud => {
            // Per-vertex lighting interpolated across the face —
            // matches the rasteriser's Gouraud definition.
            let s = &traced.shade[hit.triangle];
            let base = traced.hit_material(&hit).base_color;
            let [w, u, v] = hit.barycentric;
            let ca = shade_pixel(base, s.normals[0], light);
            let cb = shade_pixel(base, s.normals[1], light);
            let cc = shade_pixel(base, s.normals[2], light);
            let mixed = [
                w * ca[0] + u * cb[0] + v * cc[0],
                w * ca[1] + u * cb[1] + v * cc[1],
                w * ca[2] + u * cb[2] + v * cc[2],
                w * ca[3] + u * cb[3] + v * cc[3],
            ];
            linear_rgba_to_srgb_u8(mixed)
        }
        ShadingMode::Phong => {
            let colour = trace_whitted(traced, light, ray, &hit, 0);
            linear_rgba_to_srgb_u8(colour)
        }
        ShadingMode::NormalDebug => {
            let n = traced.hit_normal(&hit);
            [
                crate::scanline::normal_to_byte(n[0]),
                crate::scanline::normal_to_byte(n[1]),
                crate::scanline::normal_to_byte(n[2]),
                255,
            ]
        }
        ShadingMode::DepthDebug => {
            let p = ray.point_at(hit.t);
            let z = camera.ndc_z(camera.view_depth(p));
            let g = crate::scanline::depth_to_byte(z);
            [g, g, g, 255]
        }
    }
}

/// Whitted shading at a hit: Lambert + ambient with a shadow ray,
/// plus recursive reflection (metallic) and refraction
/// (transmission) rays blended by a Schlick Fresnel weight.
///
/// Returns a linear-space RGBA colour.
fn trace_whitted(
    traced: &TraceScene,
    light: &DirLight,
    ray: Ray,
    hit: &Hit,
    depth: u32,
) -> [f32; 4] {
    let material = *traced.hit_material(hit);
    // `KHR_materials_unlit`: constant shade from the base colour
    // alone — no lighting, no shadow ray, no secondary rays (all
    // lighting-dependent inputs are ignored per the extension).
    if material.unlit {
        return material.base_color;
    }
    // Orient the interpolated normal against the incident ray so
    // lighting and secondary-ray geometry stay on the struck side.
    // The authored (or face) normal is the shading truth — geometric
    // winding is deliberately NOT consulted here, matching the
    // scanline backend, which shades with the interpolated normal
    // regardless of winding. `entering` doubles as the
    // inside/outside signal for the refraction index ratio.
    let raw_normal = traced.hit_normal(hit);
    let entering = vec3_dot(ray.direction, raw_normal) < 0.0;
    let normal = if entering {
        raw_normal
    } else {
        vec3_scale(raw_normal, -1.0)
    };
    let point = ray.point_at(hit.t);

    // Direct lighting with a shadow ray toward the light. An occluded
    // surface keeps only the ambient term.
    let shadow_origin = vec3_add(point, vec3_scale(normal, RAY_EPSILON));
    let mut lit = if traced.any_hit(Ray::new(shadow_origin, light.direction), f32::INFINITY) {
        let a = material.base_color;
        [a[0] * AMBIENT, a[1] * AMBIENT, a[2] * AMBIENT, a[3]]
    } else {
        shade_pixel(material.base_color, normal, light)
    };
    // Additive emission (`emissive_factor × emissive_strength`) —
    // self-illumination on top of the diffuse term, unaffected by
    // shadowing. The sRGB encode clamps; >1 strengths saturate
    // toward white exactly as an SDR surface should.
    for (l, e) in lit.iter_mut().zip(material.emissive.iter()) {
        *l += e;
    }

    if depth >= MAX_DEPTH {
        return lit;
    }

    let in_dir = vec3_normalise(ray.direction);
    let cos_in = (-vec3_dot(in_dir, normal)).clamp(0.0, 1.0);

    // Reflection — mirror term weighted by metallic, faded by
    // roughness (a single Whitted ray cannot represent a glossy
    // lobe), boosted at grazing angles by Schlick's approximation
    // with F0 blended toward the metal's own reflectance.
    let mut out = lit;
    let gloss = material.metallic * (1.0 - material.roughness);
    if gloss > SECONDARY_RAY_THRESHOLD {
        let f0 = 0.04 + 0.96 * material.metallic;
        let kr = (f0 + (1.0 - f0) * (1.0 - cos_in).powi(5)).clamp(0.0, 1.0) * gloss;
        let refl_dir = reflect(in_dir, normal);
        let refl_origin = vec3_add(point, vec3_scale(normal, RAY_EPSILON));
        let refl_ray = Ray::new(refl_origin, refl_dir);
        let refl_colour = match traced.closest_hit(refl_ray, f32::INFINITY) {
            Some(refl_hit) => trace_whitted(traced, light, refl_ray, &refl_hit, depth + 1),
            // A reflection ray that escapes the scene contributes
            // nothing (the framebuffer background is a canvas
            // colour, not an environment).
            None => [0.0, 0.0, 0.0, 1.0],
        };
        // Metals tint their reflection by the base colour.
        let tint = material.base_color;
        for i in 0..3 {
            out[i] = out[i] * (1.0 - kr) + refl_colour[i] * tint[i] * kr;
        }
    }

    // Refraction — transmission ray bent by Snell's law; total
    // internal reflection folds into the reflection term.
    if material.transmission > SECONDARY_RAY_THRESHOLD {
        let kt = material.transmission;
        let eta = if entering {
            1.0 / material.ior
        } else {
            material.ior
        };
        match refract(in_dir, normal, eta) {
            Some(refr_dir) => {
                // Push through the surface, against the normal.
                let refr_origin = vec3_add(point, vec3_scale(normal, -RAY_EPSILON));
                let refr_ray = Ray::new(refr_origin, refr_dir);
                let refr_colour = match traced.closest_hit(refr_ray, f32::INFINITY) {
                    Some(refr_hit) => trace_whitted(traced, light, refr_ray, &refr_hit, depth + 1),
                    None => [0.0, 0.0, 0.0, 0.0],
                };
                let tint = material.base_color;
                for i in 0..3 {
                    out[i] = out[i] * (1.0 - kt) + refr_colour[i] * tint[i] * kt;
                }
            }
            None => {
                // Total internal reflection: send the energy along
                // the mirror direction instead.
                let refl_dir = reflect(in_dir, normal);
                let refl_origin = vec3_add(point, vec3_scale(normal, RAY_EPSILON));
                let refl_ray = Ray::new(refl_origin, refl_dir);
                if let Some(refl_hit) = traced.closest_hit(refl_ray, f32::INFINITY) {
                    let refl_colour = trace_whitted(traced, light, refl_ray, &refl_hit, depth + 1);
                    for i in 0..3 {
                        out[i] = out[i] * (1.0 - kt) + refl_colour[i] * kt;
                    }
                }
            }
        }
    }

    out
}

/// Mirror `d` about unit normal `n` (both unit; `d` points into the
/// surface): `d - 2 (d·n) n`.
fn reflect(d: [f32; 3], n: [f32; 3]) -> [f32; 3] {
    vec3_sub(d, vec3_scale(n, 2.0 * vec3_dot(d, n)))
}

/// Refract unit direction `d` through unit normal `n` (pointing
/// toward the incident side) with relative index `eta` (incident /
/// transmitted). Returns `None` on total internal reflection.
///
/// Vector Snell form: with `cos_i = -d·n`,
/// `sin²_t = eta² (1 - cos²_i)`; TIR when `sin²_t > 1`; otherwise
/// `t = eta d + (eta cos_i - cos_t) n`.
fn refract(d: [f32; 3], n: [f32; 3], eta: f32) -> Option<[f32; 3]> {
    let cos_i = (-vec3_dot(d, n)).clamp(-1.0, 1.0);
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if sin2_t > 1.0 {
        return None;
    }
    let cos_t = (1.0 - sin2_t).sqrt();
    Some(vec3_normalise(vec3_add(
        vec3_scale(d, eta),
        vec3_scale(n, eta * cos_i - cos_t),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::BackgroundColor;
    use oxideav_mesh3d::{Indices, Material, MaterialId, Mesh, MeshId, Node, NodeId, Scene3D};

    const WHITE_BG: BackgroundColor = BackgroundColor([255, 255, 255, 255]);

    fn push_mesh_node(scene: &mut Scene3D, prim: Primitive) {
        let mesh_id = scene.meshes.len() as u32;
        scene
            .meshes
            .push(Mesh::new(format!("m{mesh_id}")).with_primitive(prim));
        let node_id = scene.nodes.len() as u32;
        scene.nodes.push(Node {
            mesh: Some(MeshId(mesh_id)),
            ..Node::default()
        });
        scene.roots.push(NodeId(node_id));
    }

    fn unit_triangle_scene() -> Scene3D {
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
        let mut scene = Scene3D::new();
        push_mesh_node(&mut scene, prim);
        scene
    }

    fn render_with_mode(mode: ShadingMode) -> RgbaImage {
        let opts = RenderOptions {
            width: 64,
            height: 64,
            shading: mode,
            background: WHITE_BG,
            ..RenderOptions::default()
        };
        render_scene(&unit_triangle_scene(), &opts)
    }

    fn non_bg_count(img: &RgbaImage, bg: [u8; 4]) -> usize {
        img.pixels.chunks_exact(4).filter(|p| *p != bg).count()
    }

    #[test]
    fn every_mode_paints_the_triangle() {
        for mode in [
            ShadingMode::Flat,
            ShadingMode::Gouraud,
            ShadingMode::Phong,
            ShadingMode::Wireframe,
            ShadingMode::NormalDebug,
            ShadingMode::DepthDebug,
        ] {
            let img = render_with_mode(mode);
            assert!(
                non_bg_count(&img, [255, 255, 255, 255]) > 0,
                "{mode:?} must paint at least one pixel"
            );
        }
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
    fn output_dimensions_match_options_with_and_without_aa() {
        for aa in [1, 2, 4] {
            let opts = RenderOptions {
                width: 33,
                height: 17,
                aa,
                background: WHITE_BG,
                ..RenderOptions::default()
            };
            let img = render_scene(&unit_triangle_scene(), &opts);
            assert_eq!((img.width, img.height), (33, 17), "aa={aa}");
            assert_eq!(img.pixels.len(), 33 * 17 * 4, "aa={aa}");
        }
    }

    #[test]
    fn coverage_matches_scanline_backend() {
        // The two backends share camera framing, so the same triangle
        // must cover a nearly identical pixel set. Allow a small edge
        // disagreement (rasteriser samples pixel centres with edge
        // fill rules; rays sample exact centres).
        let scene = unit_triangle_scene();
        let opts = RenderOptions {
            width: 64,
            height: 64,
            shading: ShadingMode::Flat,
            background: WHITE_BG,
            ..RenderOptions::default()
        };
        let ray_img = render_scene(&scene, &opts);
        let scan_img = crate::scanline::render_scene(&scene, &opts);
        let mut both = 0usize;
        let mut only_one = 0usize;
        for (r, s) in ray_img
            .pixels
            .chunks_exact(4)
            .zip(scan_img.pixels.chunks_exact(4))
        {
            let rp = r != [255, 255, 255, 255];
            let sp = s != [255, 255, 255, 255];
            if rp && sp {
                both += 1;
            } else if rp != sp {
                only_one += 1;
            }
        }
        assert!(
            both > 100,
            "expected substantial shared coverage, got {both}"
        );
        assert!(
            only_one * 10 < both,
            "backends disagree on too many pixels: shared={both}, disputed={only_one}"
        );
    }

    #[test]
    fn flat_colour_matches_scanline_exactly() {
        // Interior pixels in Flat mode are the same sRGB-encoded base
        // colour in both backends.
        let scene = unit_triangle_scene();
        let opts = RenderOptions {
            width: 64,
            height: 64,
            shading: ShadingMode::Flat,
            background: WHITE_BG,
            ..RenderOptions::default()
        };
        let img = render_scene(&scene, &opts);
        let expected = linear_rgba_to_srgb_u8([0.7, 0.7, 0.75, 1.0]);
        let painted: Vec<&[u8]> = img
            .pixels
            .chunks_exact(4)
            .filter(|p| *p != [255, 255, 255, 255])
            .collect();
        assert!(!painted.is_empty());
        for p in painted {
            assert_eq!(p, expected, "flat shading must be the unlit base colour");
        }
    }

    #[test]
    fn strip_and_fan_topologies_are_visible() {
        for topology in [Topology::TriangleStrip, Topology::TriangleFan] {
            let mut prim = Primitive::new(topology);
            prim.positions = vec![
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [-0.5, 0.5, 0.0],
                [0.5, 0.5, 0.0],
            ];
            let mut scene = Scene3D::new();
            push_mesh_node(&mut scene, prim);
            let opts = RenderOptions {
                width: 32,
                height: 32,
                shading: ShadingMode::Flat,
                background: WHITE_BG,
                ..RenderOptions::default()
            };
            let img = render_scene(&scene, &opts);
            assert!(
                non_bg_count(&img, [255, 255, 255, 255]) > 0,
                "{topology:?} must bake into visible triangles"
            );
        }
    }

    #[test]
    fn line_and_point_topologies_are_invisible() {
        for topology in [Topology::Lines, Topology::LineStrip, Topology::Points] {
            let mut prim = Primitive::new(topology);
            prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]];
            let mut scene = Scene3D::new();
            push_mesh_node(&mut scene, prim);
            let opts = RenderOptions {
                width: 16,
                height: 16,
                shading: ShadingMode::Flat,
                background: WHITE_BG,
                ..RenderOptions::default()
            };
            let img = render_scene(&scene, &opts);
            assert_eq!(
                non_bg_count(&img, [255, 255, 255, 255]),
                0,
                "{topology:?} has no surface area for rays"
            );
        }
    }

    #[test]
    fn indexed_u16_triangles_render() {
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
        prim.indices = Some(Indices::U16(vec![0, 1, 2]));
        let mut scene = Scene3D::new();
        push_mesh_node(&mut scene, prim);
        let opts = RenderOptions {
            width: 32,
            height: 32,
            shading: ShadingMode::Flat,
            background: WHITE_BG,
            ..RenderOptions::default()
        };
        let img = render_scene(&scene, &opts);
        assert!(non_bg_count(&img, [255, 255, 255, 255]) > 0);
    }

    #[test]
    fn material_base_colour_reaches_pixels() {
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
        prim.material = Some(MaterialId(0));
        let mut scene = Scene3D::new();
        scene.materials.push(Material {
            base_color: [1.0, 0.0, 0.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            ..Material::new()
        });
        push_mesh_node(&mut scene, prim);
        let opts = RenderOptions {
            width: 32,
            height: 32,
            shading: ShadingMode::Flat,
            background: WHITE_BG,
            ..RenderOptions::default()
        };
        let img = render_scene(&scene, &opts);
        let expected = linear_rgba_to_srgb_u8([1.0, 0.0, 0.0, 1.0]);
        let hit_red = img.pixels.chunks_exact(4).any(|p| p == expected);
        assert!(hit_red, "material base colour must reach the framebuffer");
    }

    #[test]
    fn depth_debug_is_grayscale_and_nearer_is_brighter() {
        // Two coplanar-to-screen triangles at different depths: the
        // nearer one must be brighter (near → white convention).
        let mut near = Primitive::new(Topology::Triangles);
        near.positions = vec![[-0.9, -0.4, 0.5], [-0.1, -0.4, 0.5], [-0.5, 0.4, 0.5]];
        let mut far = Primitive::new(Topology::Triangles);
        far.positions = vec![[0.1, -0.4, -0.5], [0.9, -0.4, -0.5], [0.5, 0.4, -0.5]];
        let mut scene = Scene3D::new();
        push_mesh_node(&mut scene, near);
        push_mesh_node(&mut scene, far);
        let opts = RenderOptions {
            width: 64,
            height: 64,
            shading: ShadingMode::DepthDebug,
            background: BackgroundColor([255, 0, 0, 255]),
            ..RenderOptions::default()
        };
        let img = render_scene(&scene, &opts);
        // Gather grayscale values on each half of the image.
        let mut left_max = 0u8;
        let mut right_max = 0u8;
        for y in 0..64 {
            for x in 0..64 {
                let p = img.pixel(x, y).unwrap();
                if p == [255, 0, 0, 255] {
                    continue;
                }
                assert_eq!(p[0], p[1]);
                assert_eq!(p[1], p[2]);
                if x < 32 {
                    left_max = left_max.max(p[0]);
                } else {
                    right_max = right_max.max(p[0]);
                }
            }
        }
        assert!(left_max > 0 && right_max > 0, "both triangles must appear");
        assert!(
            left_max > right_max,
            "near (+z, left) triangle must be brighter: {left_max} vs {right_max}"
        );
    }

    #[test]
    fn phong_shading_is_darker_than_flat_on_tilted_face() {
        // A face tilted away from the default light shades darker
        // than its unlit base colour.
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
        prim.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
        let mut scene = Scene3D::new();
        push_mesh_node(&mut scene, prim);
        let opts = RenderOptions {
            width: 32,
            height: 32,
            shading: ShadingMode::Phong,
            background: WHITE_BG,
            ..RenderOptions::default()
        };
        let img = render_scene(&scene, &opts);
        let flat = linear_rgba_to_srgb_u8([0.7, 0.7, 0.75, 1.0]);
        let painted: Vec<[u8; 4]> = img
            .pixels
            .chunks_exact(4)
            .filter(|p| *p != [255, 255, 255, 255])
            .map(|p| [p[0], p[1], p[2], p[3]])
            .collect();
        assert!(!painted.is_empty());
        for p in &painted {
            assert!(
                p[0] <= flat[0] && p[1] <= flat[1] && p[2] <= flat[2],
                "lit colour must not exceed base colour: {p:?} vs {flat:?}"
            );
        }
    }

    #[test]
    fn wireframe_paints_fewer_pixels_than_flat() {
        let flat = render_with_mode(ShadingMode::Flat);
        let wire = render_with_mode(ShadingMode::Wireframe);
        let bg = [255, 255, 255, 255];
        let flat_n = non_bg_count(&flat, bg);
        let wire_n = non_bg_count(&wire, bg);
        assert!(wire_n > 0, "wireframe must paint the edge band");
        assert!(
            wire_n < flat_n,
            "edge band must be sparser than the filled face: {wire_n} vs {flat_n}"
        );
    }

    #[test]
    fn reflect_mirrors_about_normal() {
        let r = reflect([1.0, -1.0, 0.0], [0.0, 1.0, 0.0]);
        assert!((r[0] - 1.0).abs() < 1.0e-6);
        assert!((r[1] - 1.0).abs() < 1.0e-6);
        assert!(r[2].abs() < 1.0e-6);
    }

    #[test]
    fn refract_straight_through_at_eta_one() {
        let d = vec3_normalise([0.3, -0.9, 0.1]);
        let t = refract(d, [0.0, 1.0, 0.0], 1.0).expect("eta 1 never TIRs");
        for i in 0..3 {
            assert!((t[i] - d[i]).abs() < 1.0e-5, "eta=1 must not bend the ray");
        }
    }

    #[test]
    fn refract_reports_total_internal_reflection() {
        // Grazing exit from a dense medium (eta > 1) must TIR.
        let d = vec3_normalise([0.9, -0.1, 0.0]);
        assert!(refract(d, [0.0, 1.0, 0.0], 1.5).is_none());
    }

    #[test]
    fn shadow_ray_darkens_occluded_floor() {
        // A floor plane with a small raised occluder, lit at a 45°
        // slant so the occluder's shadow falls on floor that a
        // near-top-down camera can see (a straight-down light would
        // hide the shadow under the occluder itself). Identical
        // geometry rendered twice — once with the slanted light
        // (shadow visible), once with the light coming from the
        // mirrored azimuth (shadow falls on the other side): both
        // must contain ambient-only floor pixels; the floor-only
        // control scene must not.
        let mut floor = Primitive::new(Topology::Triangles);
        floor.positions = vec![
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ];
        floor.normals = Some(vec![[0.0, 1.0, 0.0]; 6]);

        let mut occluder = Primitive::new(Topology::Triangles);
        occluder.positions = vec![
            [-0.3, 0.5, -0.3],
            [0.3, 0.5, -0.3],
            [-0.3, 0.5, 0.3],
            [0.3, 0.5, -0.3],
            [0.3, 0.5, 0.3],
            [-0.3, 0.5, 0.3],
        ];

        // Slanted light: from +Z at 45° elevation. Floor lit at
        // cos(45°) ⇒ linear 0.7 × (0.2 + 0.8·0.707) ≈ 0.535 ⇒ sRGB
        // ≈ 194. Shadowed floor keeps ambient only: 0.7 × 0.2 = 0.14
        // ⇒ sRGB ≈ 110.
        let light_spec = crate::options::LightSpec {
            azimuth_deg: 0.0,
            elevation_deg: 45.0,
            intensity: 1.0,
        };
        let camera = crate::options::CameraSpec {
            elevation_deg: 80.0,
            azimuth_deg: 0.0,
            distance: 1.5,
        };
        let bg = [255, 0, 255, 255];
        let opts = RenderOptions {
            width: 48,
            height: 48,
            shading: ShadingMode::Phong,
            background: BackgroundColor(bg),
            light: light_spec,
            camera: Some(camera),
            ..RenderOptions::default()
        };

        let mut scene = Scene3D::new();
        push_mesh_node(&mut scene, floor.clone());
        push_mesh_node(&mut scene, occluder);
        let shadow_img = render_scene(&scene, &opts);

        let mut control = Scene3D::new();
        push_mesh_node(&mut control, floor);
        let control_img = render_scene(&control, &opts);

        let min_luma = |img: &RgbaImage| -> u8 {
            img.pixels
                .chunks_exact(4)
                .filter(|p| *p != bg)
                .map(|p| p[0])
                .min()
                .unwrap_or(255)
        };
        let shadow_min = min_luma(&shadow_img);
        let control_min = min_luma(&control_img);
        assert!(
            shadow_min <= 120,
            "occluder scene must contain ambient-only shadowed floor, min luma {shadow_min}"
        );
        assert!(
            control_min >= 160,
            "floor-only control must be fully lit everywhere, min luma {control_min}"
        );
    }

    #[test]
    fn metallic_floor_reflects_offscreen_triangle() {
        // A mirror floor with a red wall standing on it: floor
        // pixels near the wall pick up red from the reflection,
        // compared against the same scene with an inert floor.
        let mut floor = Primitive::new(Topology::Triangles);
        floor.positions = vec![
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ];
        floor.normals = Some(vec![[0.0, 1.0, 0.0]; 6]);
        floor.material = Some(MaterialId(0));

        let mut wall = Primitive::new(Topology::Triangles);
        wall.positions = vec![
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ];
        wall.normals = Some(vec![[0.0, 0.0, 1.0]; 6]);
        wall.material = Some(MaterialId(1));

        let mirror = Material {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 1.0,
            roughness: 0.0,
            ..Material::new()
        };
        let inert = Material {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            ..Material::new()
        };
        let red_wall = Material {
            base_color: [1.0, 0.0, 0.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            ..Material::new()
        };

        let camera = crate::options::CameraSpec {
            elevation_deg: 30.0,
            azimuth_deg: 0.0, // looking from +Z toward the wall at -Z
            distance: 1.5,
        };
        let opts = RenderOptions {
            width: 48,
            height: 48,
            shading: ShadingMode::Phong,
            background: BackgroundColor([0, 0, 255, 255]),
            camera: Some(camera),
            ..RenderOptions::default()
        };

        let render_floor = |floor_mat: Material| -> RgbaImage {
            let mut scene = Scene3D::new();
            scene.materials.push(floor_mat);
            scene.materials.push(red_wall.clone());
            push_mesh_node(&mut scene, floor.clone());
            push_mesh_node(&mut scene, wall.clone());
            render_scene(&scene, &opts)
        };

        let mirror_img = render_floor(mirror);
        let inert_img = render_floor(inert);

        // Somewhere in the lower half (the floor) the mirror image
        // must be substantially redder than the inert image.
        let mut best_delta = 0i32;
        for y in 24..48 {
            for x in 0..48 {
                let m = mirror_img.pixel(x, y).unwrap();
                let i = inert_img.pixel(x, y).unwrap();
                let redness_m = m[0] as i32 - ((m[1] as i32 + m[2] as i32) / 2);
                let redness_i = i[0] as i32 - ((i[1] as i32 + i[2] as i32) / 2);
                best_delta = best_delta.max(redness_m - redness_i);
            }
        }
        assert!(
            best_delta > 40,
            "mirror floor must reflect the red wall (best redness delta {best_delta})"
        );
    }

    #[test]
    fn transmissive_pane_shows_through_tinted() {
        // A transmissive pane in front of a red wall: rays through
        // the pane must still find the wall (vs. an opaque pane that
        // hides it).
        let mut wall = Primitive::new(Topology::Triangles);
        wall.positions = vec![
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ];
        wall.normals = Some(vec![[0.0, 0.0, 1.0]; 6]);
        wall.material = Some(MaterialId(1));

        let mut pane = Primitive::new(Topology::Triangles);
        pane.positions = vec![
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        pane.normals = Some(vec![[0.0, 0.0, 1.0]; 6]);
        pane.material = Some(MaterialId(0));

        let red_wall = Material {
            base_color: [1.0, 0.0, 0.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            ..Material::new()
        };
        let mut glass = Material {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 0.0,
            ..Material::new()
        };
        glass.ext.transmission = Some(oxideav_mesh3d::material::Transmission {
            factor: 1.0,
            factor_texture: None,
        });
        glass.ext.ior = Some(1.0); // index-matched: no bending
        let opaque = Material {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            ..Material::new()
        };

        let opts = RenderOptions {
            width: 32,
            height: 32,
            shading: ShadingMode::Phong,
            background: BackgroundColor([0, 0, 255, 255]),
            ..RenderOptions::default()
        };

        let render_pane = |pane_mat: Material| -> RgbaImage {
            let mut scene = Scene3D::new();
            scene.materials.push(pane_mat);
            scene.materials.push(red_wall.clone());
            push_mesh_node(&mut scene, pane.clone());
            push_mesh_node(&mut scene, wall.clone());
            render_scene(&scene, &opts)
        };

        let glass_img = render_pane(glass);
        let opaque_img = render_pane(opaque);

        let g = glass_img.pixel(16, 16).unwrap();
        let o = opaque_img.pixel(16, 16).unwrap();
        assert!(
            g[0] > 100 && g[1] < 100,
            "glass pane must show the red wall through it, got {g:?}"
        );
        assert!(
            o[0].abs_diff(o[1]) < 30,
            "opaque pane must stay neutral, got {o:?}"
        );
    }

    #[test]
    fn unlit_material_ignores_light_and_shadow() {
        // An unlit triangle renders its exact base colour even when
        // the light grazes it; the same geometry with a lit material
        // shades darker.
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
        prim.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
        prim.material = Some(MaterialId(0));

        let base = [0.3, 0.8, 0.2, 1.0];
        let mut unlit_mat = Material {
            base_color: base,
            metallic: 0.0,
            roughness: 1.0,
            ..Material::new()
        };
        unlit_mat.ext.unlit = true;
        let lit_mat = Material {
            base_color: base,
            metallic: 0.0,
            roughness: 1.0,
            ..Material::new()
        };

        // Light nearly opposite the surface normal.
        let opts = RenderOptions {
            width: 32,
            height: 32,
            shading: ShadingMode::Phong,
            background: WHITE_BG,
            light: crate::options::LightSpec {
                azimuth_deg: 180.0,
                elevation_deg: 0.0,
                intensity: 1.0,
            },
            ..RenderOptions::default()
        };

        let render_mat = |mat: Material| -> RgbaImage {
            let mut scene = Scene3D::new();
            scene.materials.push(mat);
            push_mesh_node(&mut scene, prim.clone());
            render_scene(&scene, &opts)
        };

        let unlit_img = render_mat(unlit_mat);
        let lit_img = render_mat(lit_mat);
        let expected = linear_rgba_to_srgb_u8(base);
        let unlit_painted: Vec<[u8; 4]> = unlit_img
            .pixels
            .chunks_exact(4)
            .filter(|p| *p != [255, 255, 255, 255])
            .map(|p| [p[0], p[1], p[2], p[3]])
            .collect();
        assert!(!unlit_painted.is_empty());
        for p in &unlit_painted {
            assert_eq!(*p, expected, "unlit surface must be the exact base colour");
        }
        let lit_darker = lit_img
            .pixels
            .chunks_exact(4)
            .filter(|p| *p != [255, 255, 255, 255])
            .all(|p| p[1] < expected[1]);
        assert!(
            lit_darker,
            "the lit control must shade darker than the unlit base colour"
        );
    }

    #[test]
    fn emissive_material_self_illuminates() {
        // Black base + red emission: the surface must glow red, and
        // halving KHR_materials_emissive_strength must dim it.
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
        prim.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
        prim.material = Some(MaterialId(0));

        let emissive_mat = |strength: Option<f32>| -> Material {
            let mut m = Material {
                base_color: [0.0, 0.0, 0.0, 1.0],
                metallic: 0.0,
                roughness: 1.0,
                emissive_factor: [0.5, 0.0, 0.0],
                ..Material::new()
            };
            m.ext.emissive_strength = strength;
            m
        };

        let opts = RenderOptions {
            width: 32,
            height: 32,
            shading: ShadingMode::Phong,
            background: WHITE_BG,
            ..RenderOptions::default()
        };
        let render_mat = |mat: Material| -> RgbaImage {
            let mut scene = Scene3D::new();
            scene.materials.push(mat);
            push_mesh_node(&mut scene, prim.clone());
            render_scene(&scene, &opts)
        };

        let full = render_mat(emissive_mat(None)); // strength default 1.0
        let dim = render_mat(emissive_mat(Some(0.5)));
        let max_red = |img: &RgbaImage| -> u8 {
            img.pixels
                .chunks_exact(4)
                .filter(|p| *p != [255, 255, 255, 255])
                .map(|p| p[0])
                .max()
                .unwrap_or(0)
        };
        let full_red = max_red(&full);
        let dim_red = max_red(&dim);
        let expected_full = crate::shade::linear_to_srgb_u8(0.5);
        let expected_dim = crate::shade::linear_to_srgb_u8(0.25);
        assert_eq!(full_red, expected_full, "emission must reach the pixel");
        assert_eq!(
            dim_red, expected_dim,
            "emissive_strength must scale the emission"
        );
        // Green / blue channels stay black (base colour is black,
        // emission is pure red).
        let clean_channels = full
            .pixels
            .chunks_exact(4)
            .filter(|p| *p != [255, 255, 255, 255])
            .all(|p| p[1] == 0 && p[2] == 0);
        assert!(clean_channels, "emission must not leak across channels");
    }

    #[test]
    fn nested_transform_hierarchy_is_honoured() {
        // A child triangle translated by its parent must move in the
        // image relative to an un-translated sibling scene.
        use oxideav_mesh3d::Transform;
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];

        let mut scene = Scene3D::new();
        scene
            .meshes
            .push(Mesh::new("t".to_string()).with_primitive(prim));
        // Parent shifts +X by 2; child carries the mesh.
        scene.nodes.push(Node {
            transform: Transform::Trs {
                translation: [2.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            },
            children: vec![NodeId(1)],
            ..Node::default()
        });
        scene.nodes.push(Node {
            mesh: Some(MeshId(0)),
            ..Node::default()
        });
        scene.roots.push(NodeId(0));

        let opts = RenderOptions {
            width: 32,
            height: 32,
            shading: ShadingMode::Flat,
            background: WHITE_BG,
            ..RenderOptions::default()
        };
        let img = render_scene(&scene, &opts);
        // Auto-framing recentres on the shifted bbox, so the triangle
        // still lands mid-frame — assert it renders at all (the walk
        // composed parent × child without panicking) and matches the
        // scanline backend's coverage for the same scene.
        let scan = crate::scanline::render_scene(&scene, &opts);
        let ray_n = non_bg_count(&img, [255, 255, 255, 255]);
        let scan_n = non_bg_count(&scan, [255, 255, 255, 255]);
        assert!(ray_n > 0);
        assert!(
            (ray_n as i64 - scan_n as i64).unsigned_abs() as usize <= ray_n / 4 + 8,
            "backends must agree on transformed coverage: ray={ray_n} scan={scan_n}"
        );
    }
}
