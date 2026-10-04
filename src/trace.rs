//! Shared ray-tracing layer over a [`PreparedScene`]: one world-space
//! triangle BVH, hit → shading-attribute interpolation, texture level
//! of detail from a ray footprint, and glTF material evaluation at a
//! hit. Consumed by the ray-based backends (Whitted raycast, path
//! tracer) so they agree on what a ray sees and how a surface looks.
//!
//! * **Acceleration** — every triangle [`DrawItem`] is concatenated
//!   into one unindexed world-space soup; [`oxideav_mesh3d::Bvh`]
//!   (binned SAH) is built over it and queried with the
//!   Woop-Benthin-Wald watertight triangle test (Woop, Benthin, Wald,
//!   "Watertight Ray/Triangle Intersection", JCGT 2(1), 2013) so
//!   rays never leak through shared edges of closed meshes. Global
//!   triangle `g` maps back to `(item, tri)` through [`TriRef`].
//! * **Self-intersection** — [`offset_ray_origin`] implements
//!   Wächter & Binder, "A Fast and Robust Method for Avoiding
//!   Self-Intersection" (Ray Tracing Gems, ch. 6, 2019): the hit
//!   point is pushed along the geometric normal by an amount that
//!   scales with the magnitude of its coordinates (integer ULP steps
//!   far from the origin, a fixed epsilon near it).
//! * **Texture LOD** — ray cones (Akenine-Möller, Nilsson, Andersson,
//!   Barré-Brisebois, Toth, Karras, "Texture Level of Detail
//!   Strategies for Real-Time Ray Tracing", Ray Tracing Gems ch. 20,
//!   2019): `λ = Δ₀ + log₂(w / |n̂·d̂|)` with the per-triangle constant
//!   `Δ₀ = ½ log₂(t_a / p_a)` (texel-space area over world area) and
//!   `w` the cone width at the hit. [`TexLod::Base`] forces level 0.
//! * **Materials** — the glTF 2.0 metallic-roughness inputs
//!   (base colour × texture × `COLOR_0`, metallic-roughness, normal,
//!   occlusion, emissive maps; glTF 2.0 §3.9) plus the KHR material
//!   extensions' scalar / texture inputs (transmission, volume, ior,
//!   specular, clearcoat, sheen), each per its Khronos specification.
//!   Texture references inside [`oxideav_mesh3d::MaterialExt`] are
//!   resolved against [`PreparedScene::textures`] here.

use oxideav_mesh3d::ray::{PreparedRay, Ray, RayQuery, TriangleTest};
use oxideav_mesh3d::{AlphaMode, Bvh, Primitive, TextureRef, Topology};

use crate::math::{vec3_cross, vec3_dot, vec3_normalise, vec3_sub};
use crate::prepare::{DrawItem, DrawTopology, PreparedMaterial, PreparedScene, TextureBinding};
use crate::texture::ColorSpace;

/// Back-reference from a global soup triangle to its draw item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TriRef {
    /// Index into [`PreparedScene::items`].
    pub item: u32,
    /// Triangle index within the item (vertices `3·tri .. 3·tri + 3`).
    pub tri: u32,
}

/// A ray / triangle intersection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceHit {
    /// Ray parameter (world distance for a unit direction).
    pub t: f32,
    /// Global triangle index (into [`TraceScene::tri_refs`]).
    pub global: u32,
    /// Draw item index.
    pub item: u32,
    /// Triangle index within the item.
    pub tri: u32,
    /// `[w, u, v]` barycentrics of the item's corners `3·tri + 0..3`.
    pub barycentric: [f32; 3],
    /// `true` when the ray struck the CCW-front side.
    pub front_face: bool,
}

/// Interpolated geometry at a hit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Surface {
    /// World hit position (barycentric reconstruction — more precise
    /// than `o + t·d`).
    pub position: [f32; 3],
    /// Unit geometric normal of the CCW-front side (not flipped toward
    /// the ray).
    pub geometric_normal: [f32; 3],
    /// Unit interpolated vertex normal (front side; the geometric
    /// normal for items without vertex normals).
    pub shading_normal: [f32; 3],
    /// World-space triangle area.
    pub area: f32,
    /// Index into [`PreparedScene::materials`].
    pub material: usize,
}

/// Texture level-of-detail selection for material evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TexLod {
    /// Sample mip level 0 (magnification filter).
    Base,
    /// Ray-cone footprint: cone `width` (world units) at the hit and
    /// the unit ray direction (for the `|n·d|` foreshortening term).
    Cone {
        /// Cone width at the hit point.
        width: f32,
        /// Unit ray direction.
        dir: [f32; 3],
    },
}

/// The glTF material inputs evaluated at a surface point. Colours are
/// linear. Normals are on the CCW-front side of the triangle (callers
/// flip them for back-face hits).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialSample {
    /// Base colour × texture × vertex colour (RGBA).
    pub base_color: [f32; 4],
    /// Metallic `[0, 1]`.
    pub metallic: f32,
    /// Perceptual roughness `[0, 1]`.
    pub roughness: f32,
    /// Normal-mapped unit shading normal (front side).
    pub normal: [f32; 3],
    /// Ambient occlusion `[0, 1]` (1 = unoccluded).
    pub occlusion: f32,
    /// Emitted radiance (factor × texture × strength).
    pub emissive: [f32; 3],
    /// `KHR_materials_unlit`.
    pub unlit: bool,
    /// Alpha mode of the material.
    pub alpha_mode: AlphaMode,
    /// Double-sided flag.
    pub double_sided: bool,
    /// Index of refraction (`KHR_materials_ior`).
    pub ior: f32,
    /// `KHR_materials_specular` strength (1 = core model).
    pub specular: f32,
    /// `KHR_materials_specular` F0 colour (1 = core model).
    pub specular_color: [f32; 3],
    /// `KHR_materials_transmission` factor × texture R.
    pub transmission: f32,
    /// `KHR_materials_volume` thickness (0 = thin-walled).
    pub thickness: f32,
    /// `KHR_materials_volume` attenuation colour.
    pub attenuation_color: [f32; 3],
    /// `KHR_materials_volume` attenuation distance (∞ = none).
    pub attenuation_distance: f32,
    /// `KHR_materials_clearcoat` factor.
    pub clearcoat: f32,
    /// Clearcoat perceptual roughness.
    pub clearcoat_roughness: f32,
    /// Clearcoat unit normal (front side).
    pub clearcoat_normal: [f32; 3],
    /// `KHR_materials_sheen` colour (black = no sheen).
    pub sheen_color: [f32; 3],
    /// Sheen perceptual roughness.
    pub sheen_roughness: f32,
}

/// A prepared scene plus its world-space triangle BVH.
#[derive(Debug, Clone)]
pub struct TraceScene {
    /// The scene being traced.
    pub prepared: PreparedScene,
    soup: Primitive,
    bvh: Option<Bvh>,
    tri_refs: Vec<TriRef>,
}

impl TraceScene {
    /// Build the triangle soup + BVH over `prepared`'s triangle items
    /// (line / point items have no area and are not traceable).
    pub fn new(prepared: PreparedScene) -> Self {
        let mut soup = Primitive::new(Topology::Triangles);
        let mut tri_refs = Vec::new();
        for (ii, item) in prepared.items.iter().enumerate() {
            if item.topology != DrawTopology::Triangles {
                continue;
            }
            for t in 0..item.positions.len() / 3 {
                soup.positions
                    .extend_from_slice(&item.positions[3 * t..3 * t + 3]);
                tri_refs.push(TriRef {
                    item: ii as u32,
                    tri: t as u32,
                });
            }
        }
        let bvh = Bvh::build(&soup);
        Self {
            prepared,
            soup,
            bvh,
            tri_refs,
        }
    }

    /// Number of traceable triangles.
    pub fn triangle_count(&self) -> usize {
        self.tri_refs.len()
    }

    /// Global triangle → draw-item back-references.
    pub fn tri_refs(&self) -> &[TriRef] {
        &self.tri_refs
    }

    /// World positions of global triangle `g`.
    pub fn triangle_positions(&self, g: u32) -> [[f32; 3]; 3] {
        let b = 3 * g as usize;
        [
            self.soup.positions[b],
            self.soup.positions[b + 1],
            self.soup.positions[b + 2],
        ]
    }

    fn make_hit(&self, g: usize, t: f32, bary: [f32; 3], front: bool) -> TraceHit {
        let r = self.tri_refs[g];
        TraceHit {
            t,
            global: g as u32,
            item: r.item,
            tri: r.tri,
            barycentric: bary,
            front_face: front,
        }
    }

    /// Closest hit in `[t_min, t_max]` along `origin + t·dir`.
    pub fn closest_hit(
        &self,
        origin: [f32; 3],
        dir: [f32; 3],
        t_min: f32,
        t_max: f32,
    ) -> Option<TraceHit> {
        self.closest_hit_filtered(origin, dir, t_min, t_max, |_| true)
    }

    /// Closest hit; `filter` accepts or rejects each candidate (alpha
    /// mask, stochastic transparency, back-face culling).
    pub fn closest_hit_filtered<F: FnMut(&TraceHit) -> bool>(
        &self,
        origin: [f32; 3],
        dir: [f32; 3],
        t_min: f32,
        t_max: f32,
        mut filter: F,
    ) -> Option<TraceHit> {
        let bvh = self.bvh.as_ref()?;
        let ray = PreparedRay::new(Ray::new(origin, dir));
        let q = RayQuery::new(t_max)
            .with_t_min(t_min)
            .with_triangle_test(TriangleTest::Watertight);
        let h = bvh.closest_hit_filtered(&self.soup, &ray, &q, |c| {
            filter(&self.make_hit(c.triangle_index, c.t, c.barycentric, c.front_face))
        })?;
        Some(self.make_hit(h.triangle_index, h.t, h.barycentric, h.front_face))
    }

    /// `true` when anything blocks `origin + t·dir` for `t` in
    /// `[t_min, t_max]`.
    pub fn occluded(&self, origin: [f32; 3], dir: [f32; 3], t_min: f32, t_max: f32) -> bool {
        self.occluded_filtered(origin, dir, t_min, t_max, |_| true)
    }

    /// Occlusion query with a candidate filter (rejected candidates
    /// let the ray through).
    pub fn occluded_filtered<F: FnMut(&TraceHit) -> bool>(
        &self,
        origin: [f32; 3],
        dir: [f32; 3],
        t_min: f32,
        t_max: f32,
        mut filter: F,
    ) -> bool {
        let Some(bvh) = self.bvh.as_ref() else {
            return false;
        };
        let ray = PreparedRay::new(Ray::new(origin, dir));
        let q = RayQuery::new(t_max)
            .with_t_min(t_min)
            .with_triangle_test(TriangleTest::Watertight);
        bvh.occluded_filtered(&self.soup, &ray, &q, |c| {
            filter(&self.make_hit(c.triangle_index, c.t, c.barycentric, c.front_face))
        })
    }

    /// Draw item and material of a hit.
    pub fn item(&self, hit: &TraceHit) -> (&DrawItem, &PreparedMaterial) {
        let item = &self.prepared.items[hit.item as usize];
        (item, &self.prepared.materials[item.material])
    }

    /// Interpolated geometry at `hit`.
    pub fn surface(&self, hit: &TraceHit) -> Surface {
        let item = &self.prepared.items[hit.item as usize];
        let base = 3 * hit.tri as usize;
        let b = hit.barycentric;
        let p = &item.positions;
        let position = interp3(p, base, b);
        let cr = vec3_cross(
            vec3_sub(p[base + 1], p[base]),
            vec3_sub(p[base + 2], p[base]),
        );
        let len = vec3_dot(cr, cr).sqrt();
        let ng = if len > 0.0 {
            [cr[0] / len, cr[1] / len, cr[2] / len]
        } else {
            [0.0, 0.0, 1.0]
        };
        let shading_normal = if item.has_vertex_normals && item.normals.len() == p.len() {
            let n = vec3_normalise(interp3(&item.normals, base, b));
            if vec3_dot(n, n) > 0.5 {
                n
            } else {
                ng
            }
        } else {
            ng
        };
        Surface {
            position,
            geometric_normal: ng,
            shading_normal,
            area: 0.5 * len,
            material: item.material,
        }
    }

    /// Base-colour alpha (factor × texture A × vertex colour A) at a
    /// hit, sampled at mip level 0 — the any-hit coverage input.
    pub fn alpha(&self, hit: &TraceHit) -> f32 {
        let (item, mat) = self.item(hit);
        let base = 3 * hit.tri as usize;
        let b = hit.barycentric;
        let mut a = mat.base_color[3];
        if let Some(bind) = &mat.base_color_texture {
            if let Some(t) = self.sample(item, bind, base, b, None, ColorSpace::Srgb) {
                a *= t[3];
            }
        }
        if item.colors.len() == item.positions.len() {
            a *= interp4(&item.colors, base, b)[3];
        }
        a
    }

    /// `true` when a candidate hit is rejected by the MASK cutoff.
    pub fn masked_out(&self, hit: &TraceHit) -> bool {
        let (_, mat) = self.item(hit);
        match mat.alpha_mode {
            AlphaMode::Mask { cutoff } => self.alpha(hit) < cutoff,
            _ => false,
        }
    }

    /// Ray-cone LOD for `binding` on the hit triangle.
    fn lod(
        &self,
        item: &DrawItem,
        binding: &TextureBinding,
        base: usize,
        lod: &TexLod,
    ) -> Option<f32> {
        let TexLod::Cone { width, dir } = *lod else {
            return None;
        };
        let tex = self.prepared.texture(binding)?;
        let uvs = item.uv_set(binding.uv_set)?;
        let p = &item.positions;
        let uv: [[f32; 2]; 3] = [
            binding.transform_uv(uvs[base]),
            binding.transform_uv(uvs[base + 1]),
            binding.transform_uv(uvs[base + 2]),
        ];
        let img = tex.data.image();
        let ta = (img.width as f32)
            * (img.height as f32)
            * ((uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1])
                - (uv[2][0] - uv[0][0]) * (uv[1][1] - uv[0][1]))
                .abs();
        let cr = vec3_cross(
            vec3_sub(p[base + 1], p[base]),
            vec3_sub(p[base + 2], p[base]),
        );
        let pa = vec3_dot(cr, cr).sqrt();
        if !(ta > 0.0 && pa > 0.0 && width > 0.0) {
            return Some(0.0);
        }
        let n = [cr[0] / pa, cr[1] / pa, cr[2] / pa];
        let cos = vec3_dot(n, dir).abs().max(1.0e-3);
        let l = 0.5 * (ta / pa).log2() + (width / cos).log2();
        Some(if l.is_finite() { l } else { 0.0 })
    }

    fn sample(
        &self,
        item: &DrawItem,
        binding: &TextureBinding,
        base: usize,
        b: [f32; 3],
        lod: Option<f32>,
        space: ColorSpace,
    ) -> Option<[f32; 4]> {
        let tex = self.prepared.texture(binding)?;
        let uvs = item.uv_set(binding.uv_set)?;
        if uvs.len() != item.positions.len() {
            return None;
        }
        let uv = binding.transform_uv(interp2(uvs, base, b));
        Some(tex.sample_lod(uv, lod.unwrap_or(0.0), space))
    }

    fn sample_at(
        &self,
        item: &DrawItem,
        binding: &TextureBinding,
        hit: &TraceHit,
        lod: &TexLod,
        space: ColorSpace,
    ) -> Option<[f32; 4]> {
        let base = 3 * hit.tri as usize;
        let l = self.lod(item, binding, base, lod);
        self.sample(item, binding, base, hit.barycentric, l, space)
    }

    /// Resolve an extension texture reference against the prepared
    /// texture table.
    pub fn bind_ref(&self, r: &Option<TextureRef>) -> Option<TextureBinding> {
        let r = r.as_ref()?;
        let idx = r.texture.0 as usize;
        self.prepared.textures.get(idx)?.as_ref()?;
        Some(TextureBinding {
            texture: idx,
            uv_set: r.effective_uv_set(),
            transform: r.transform.filter(|t| !t.is_identity() && t.is_finite()),
        })
    }

    /// Apply a tangent-space normal map sampled as `t` to `n`.
    fn perturb(
        item: &DrawItem,
        base: usize,
        b: [f32; 3],
        n: [f32; 3],
        t: [f32; 4],
        scale: f32,
    ) -> [f32; 3] {
        if item.tangents.len() != item.positions.len() {
            return n;
        }
        let tg = interp4(&item.tangents, base, b);
        let w = if item.tangents[base][3] < 0.0 {
            -1.0
        } else {
            1.0
        };
        let t3 = [tg[0], tg[1], tg[2]];
        let d = vec3_dot(n, t3);
        let tt = vec3_normalise([t3[0] - n[0] * d, t3[1] - n[1] * d, t3[2] - n[2] * d]);
        if vec3_dot(tt, tt) < 0.5 {
            return n;
        }
        let bt = vec3_cross(n, tt);
        let bt = [bt[0] * w, bt[1] * w, bt[2] * w];
        let ts = [
            (t[0] * 2.0 - 1.0) * scale,
            (t[1] * 2.0 - 1.0) * scale,
            t[2] * 2.0 - 1.0,
        ];
        let nn = vec3_normalise([
            tt[0] * ts[0] + bt[0] * ts[1] + n[0] * ts[2],
            tt[1] * ts[0] + bt[1] * ts[1] + n[1] * ts[2],
            tt[2] * ts[0] + bt[2] * ts[1] + n[2] * ts[2],
        ]);
        if vec3_dot(nn, nn) > 0.5 {
            nn
        } else {
            n
        }
    }

    /// Evaluate every material input at `hit` (`surf` from
    /// [`Self::surface`]).
    pub fn material(&self, hit: &TraceHit, surf: &Surface, lod: TexLod) -> MaterialSample {
        let (item, mat) = self.item(hit);
        let base = 3 * hit.tri as usize;
        let b = hit.barycentric;
        let samp = |bind: &Option<TextureBinding>, space| {
            bind.as_ref()
                .and_then(|bd| self.sample_at(item, bd, hit, &lod, space))
        };

        let mut col = mat.base_color;
        if let Some(t) = samp(&mat.base_color_texture, ColorSpace::Srgb) {
            for k in 0..4 {
                col[k] *= t[k];
            }
        }
        if item.colors.len() == item.positions.len() {
            let v = interp4(&item.colors, base, b);
            for k in 0..4 {
                col[k] *= v[k];
            }
        }
        let mut metallic = mat.metallic;
        let mut roughness = mat.roughness;
        if let Some(t) = samp(&mat.metallic_roughness_texture, ColorSpace::Linear) {
            roughness *= t[1];
            metallic *= t[2];
        }
        let mut normal = surf.shading_normal;
        if let Some(t) = samp(&mat.normal_texture, ColorSpace::Linear) {
            normal = Self::perturb(item, base, b, normal, t, mat.normal_scale);
        }
        let mut occlusion = 1.0;
        if let Some(t) = samp(&mat.occlusion_texture, ColorSpace::Linear) {
            occlusion = 1.0 + mat.occlusion_strength * (t[0] - 1.0);
        }
        let mut emissive = mat.emissive;
        if let Some(t) = samp(&mat.emissive_texture, ColorSpace::Srgb) {
            for k in 0..3 {
                emissive[k] *= t[k];
            }
        }

        let ext = &mat.ext;
        let fin = |v: f32, d: f32| if v.is_finite() { v } else { d };
        let ext_samp = |r: &Option<TextureRef>, space| samp(&self.bind_ref(r), space);

        let (mut specular, mut specular_color) = (1.0, [1.0; 3]);
        if let Some(s) = &ext.specular {
            specular = fin(s.factor, 1.0).clamp(0.0, 1.0);
            if let Some(t) = ext_samp(&s.factor_texture, ColorSpace::Linear) {
                specular *= t[3];
            }
            specular_color = [
                fin(s.color_factor[0], 1.0).max(0.0),
                fin(s.color_factor[1], 1.0).max(0.0),
                fin(s.color_factor[2], 1.0).max(0.0),
            ];
            if let Some(t) = ext_samp(&s.color_texture, ColorSpace::Srgb) {
                for k in 0..3 {
                    specular_color[k] *= t[k];
                }
            }
        }
        let mut transmission = 0.0;
        if let Some(tr) = &ext.transmission {
            transmission = fin(tr.factor, 0.0).clamp(0.0, 1.0);
            if let Some(t) = ext_samp(&tr.factor_texture, ColorSpace::Linear) {
                transmission *= t[0];
            }
        }
        let (mut thickness, mut attenuation_color, mut attenuation_distance) =
            (0.0, [1.0; 3], f32::INFINITY);
        if let Some(v) = &ext.volume {
            thickness = fin(v.thickness, 0.0).max(0.0);
            if let Some(t) = ext_samp(&v.thickness_texture, ColorSpace::Linear) {
                thickness *= t[1];
            }
            attenuation_color = [
                fin(v.attenuation_color[0], 1.0).clamp(0.0, 1.0),
                fin(v.attenuation_color[1], 1.0).clamp(0.0, 1.0),
                fin(v.attenuation_color[2], 1.0).clamp(0.0, 1.0),
            ];
            let d = v.effective_attenuation_distance();
            attenuation_distance = if d.is_finite() && d > 0.0 {
                d
            } else {
                f32::INFINITY
            };
        }
        let (mut clearcoat, mut clearcoat_roughness, mut clearcoat_normal) =
            (0.0, 0.0, surf.shading_normal);
        if let Some(c) = &ext.clearcoat {
            clearcoat = fin(c.factor, 0.0).clamp(0.0, 1.0);
            if let Some(t) = ext_samp(&c.factor_texture, ColorSpace::Linear) {
                clearcoat *= t[0];
            }
            clearcoat_roughness = fin(c.roughness, 0.0).clamp(0.0, 1.0);
            if let Some(t) = ext_samp(&c.roughness_texture, ColorSpace::Linear) {
                clearcoat_roughness *= t[1];
            }
            if let Some(t) = ext_samp(&c.normal_texture, ColorSpace::Linear) {
                clearcoat_normal = Self::perturb(
                    item,
                    base,
                    b,
                    surf.shading_normal,
                    t,
                    fin(c.normal_scale, 1.0),
                );
            }
        }
        let (mut sheen_color, mut sheen_roughness) = ([0.0; 3], 0.0);
        if let Some(s) = &ext.sheen {
            sheen_color = [
                fin(s.color_factor[0], 0.0).clamp(0.0, 1.0),
                fin(s.color_factor[1], 0.0).clamp(0.0, 1.0),
                fin(s.color_factor[2], 0.0).clamp(0.0, 1.0),
            ];
            if let Some(t) = ext_samp(&s.color_texture, ColorSpace::Srgb) {
                for k in 0..3 {
                    sheen_color[k] *= t[k];
                }
            }
            sheen_roughness = fin(s.roughness, 0.0).clamp(0.0, 1.0);
            if let Some(t) = ext_samp(&s.roughness_texture, ColorSpace::Linear) {
                sheen_roughness *= t[3];
            }
        }

        MaterialSample {
            base_color: col,
            metallic: metallic.clamp(0.0, 1.0),
            roughness: roughness.clamp(0.0, 1.0),
            normal,
            occlusion: occlusion.clamp(0.0, 1.0),
            emissive,
            unlit: mat.unlit,
            alpha_mode: mat.alpha_mode,
            double_sided: mat.double_sided,
            ior: mat.ior,
            specular,
            specular_color,
            transmission,
            thickness,
            attenuation_color,
            attenuation_distance,
            clearcoat,
            clearcoat_roughness,
            clearcoat_normal,
            sheen_color,
            sheen_roughness,
        }
    }
}

/// Offset a ray origin `p` off its surface along the geometric normal
/// `n` (pointing to the side the new ray leaves on) — Wächter &
/// Binder 2019 (Ray Tracing Gems ch. 6). Far from the origin the
/// offset is `int_scale · n` ULPs of each coordinate; near the origin,
/// where ULPs are tiny, a fixed `float_scale · n` is added instead.
pub fn offset_ray_origin(p: [f32; 3], n: [f32; 3]) -> [f32; 3] {
    const ORIGIN: f32 = 1.0 / 32.0;
    const FLOAT_SCALE: f32 = 1.0 / 65536.0;
    const INT_SCALE: f32 = 256.0;
    let mut out = [0.0; 3];
    for k in 0..3 {
        let of_i = (INT_SCALE * n[k]) as i32;
        let bits = p[k].to_bits() as i32;
        let moved = if p[k] < 0.0 {
            bits.wrapping_sub(of_i)
        } else {
            bits.wrapping_add(of_i)
        };
        let p_i = f32::from_bits(moved as u32);
        out[k] = if p[k].abs() < ORIGIN {
            p[k] + FLOAT_SCALE * n[k]
        } else if p_i.is_finite() {
            p_i
        } else {
            p[k]
        };
    }
    out
}

/// Barycentric blend of three consecutive corners starting at `base`.
pub fn interp3(v: &[[f32; 3]], base: usize, b: [f32; 3]) -> [f32; 3] {
    let (x, y, z) = (v[base], v[base + 1], v[base + 2]);
    [
        x[0] * b[0] + y[0] * b[1] + z[0] * b[2],
        x[1] * b[0] + y[1] * b[1] + z[1] * b[2],
        x[2] * b[0] + y[2] * b[1] + z[2] * b[2],
    ]
}

/// [`interp3`] for 2-component attributes.
pub fn interp2(v: &[[f32; 2]], base: usize, b: [f32; 3]) -> [f32; 2] {
    let (x, y, z) = (v[base], v[base + 1], v[base + 2]);
    [
        x[0] * b[0] + y[0] * b[1] + z[0] * b[2],
        x[1] * b[0] + y[1] * b[1] + z[1] * b[2],
    ]
}

/// [`interp3`] for 4-component attributes.
pub fn interp4(v: &[[f32; 4]], base: usize, b: [f32; 3]) -> [f32; 4] {
    let (x, y, z) = (v[base], v[base + 1], v[base + 2]);
    let mut o = [0.0; 4];
    for k in 0..4 {
        o[k] = x[k] * b[0] + y[k] * b[1] + z[k] * b[2];
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prepare::PrepareOptions;
    use crate::texture::TextureCache;

    fn box_scene() -> TraceScene {
        let scene = crate::testscenes::cornell_box();
        let p = PreparedScene::build(
            &scene,
            &PrepareOptions::default(),
            &mut TextureCache::default(),
        );
        TraceScene::new(p)
    }

    #[test]
    fn hits_back_wall_and_maps_back_to_item() {
        let ts = box_scene();
        assert!(ts.triangle_count() > 10);
        let h = ts
            .closest_hit([0.0, 0.5, 0.9], [0.0, 0.0, -1.0], 0.0, f32::INFINITY)
            .expect("back wall");
        assert!((h.t - 1.9).abs() < 1e-4, "{}", h.t);
        assert!(h.front_face);
        let s = ts.surface(&h);
        assert!((s.position[2] + 1.0).abs() < 1e-5);
        assert!((s.geometric_normal[2] - 1.0).abs() < 1e-5);
        assert!(ts.occluded([0.0, 0.5, 0.9], [0.0, 0.0, -1.0], 0.0, 2.0));
        assert!(!ts.occluded([0.0, 0.5, 0.9], [0.0, 0.0, -1.0], 0.0, 1.5));
        // Filter rejecting everything sees nothing.
        assert!(ts
            .closest_hit_filtered([0.0, 0.5, 0.9], [0.0, 0.0, -1.0], 0.0, 9.0, |_| false)
            .is_none());
    }

    #[test]
    fn material_reads_factors() {
        let ts = box_scene();
        let h = ts
            .closest_hit([0.0, 0.5, 0.0], [-1.0, 0.0, 0.0], 0.0, f32::INFINITY)
            .unwrap();
        let s = ts.surface(&h);
        let m = ts.material(&h, &s, TexLod::Base);
        assert!(m.base_color[0] > 0.6 && m.base_color[1] < 0.1, "{m:?}");
        assert_eq!(m.transmission, 0.0);
    }

    #[test]
    fn offset_moves_off_the_plane_both_scales() {
        for p in [[0.0f32, 0.0, 0.0], [100.0, -100.0, 5.0]] {
            let q = offset_ray_origin(p, [0.0, 1.0, 0.0]);
            assert!(q[1] > p[1]);
            assert_eq!(q[0], p[0]);
        }
    }
}
