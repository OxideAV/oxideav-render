//! Backend-agnostic scene preparation: [`Scene3D`] (+ animation time)
//! → a flat, world-space render list every backend consumes.
//!
//! [`PreparedScene::build`] runs the glTF 2.0 instantiation pipeline
//! once per frame:
//!
//! 1. **Animation** — when [`PrepareOptions::time`] is set, the
//!    selected animation is sampled into a pose
//!    ([`oxideav_mesh3d::Animation::sample_pose`], Appendix C sampler
//!    semantics) and the posed world matrices are composed
//!    ([`Scene3D::posed_node_transforms`]); otherwise the rest-pose
//!    [`Scene3D::world_node_transforms`] apply.
//! 2. **Morph targets** — per §3.7.4 weight precedence
//!    *animation > node > mesh* ([`oxideav_mesh3d::Primitive::morphed`]).
//! 3. **Skinning** on the CPU — linear-blend skinning against the
//!    (posed) joint palette ([`Scene3D::joint_matrices_with`],
//!    [`oxideav_mesh3d::Primitive::skinned`]); skinned geometry lands
//!    in world space and ignores its node transform (§3.7.3.2).
//!    Unskinned geometry is baked through its node world matrix
//!    ([`oxideav_mesh3d::Primitive::transformed`]: inverse-transpose
//!    for normals, handedness flip for mirrored tangents).
//! 4. **De-indexing** — every primitive becomes an unindexed list:
//!    triangles (strips / fans unrolled), line segments
//!    (strips / loops unrolled), or points. Triangle winding is
//!    normalised so counter-clockwise is always front-facing (glTF
//!    §3.7.2.1: a mirroring world matrix — negative determinant —
//!    reverses the winding, so such triangles are re-ordered).
//!    Triangles with out-of-range indices or non-finite positions are
//!    dropped.
//! 5. **Attributes** — missing normals become flat face normals
//!    (§3.7.2.1); missing tangents are generated when the material
//!    has a normal map (vertex-averaged per-triangle UV gradients via
//!    [`oxideav_mesh3d::Primitive::compute_tangents`], or per-face
//!    when there are no vertex normals).
//!
//! Plus a resolved material table, resolved texture table (decoded
//! through a [`TextureCache`]), world-space punctual lights (or the
//! options' fallback light) and scene camera instances.
//!
//! Everything is plain data — `Vec<[f32; N]>` per attribute, one
//! vertex per corner — so a GPU backend can upload a [`DrawItem`]'s
//! buffers as-is (`bytemuck`-style casts work on `[f32; N]` arrays).

use std::sync::Arc;

use oxideav_mesh3d::{
    AlphaMode, Light, Material, MaterialExt, MaterialId, MaterialVariantId, MeshId, NodeId,
    Primitive, Scene3D, TextureRef, TextureTransform, Topology,
};

use crate::math::{identity4, vec3_cross, vec3_dot, vec3_normalise, vec3_sub};
use crate::options::{LightSpec, RenderOptions};
use crate::texture::{PreparedTexture, TextureCache};

// ---------------------------------------------------------------------
// Options.
// ---------------------------------------------------------------------

/// Inputs to [`PreparedScene::build`].
#[derive(Debug, Clone, PartialEq)]
pub struct PrepareOptions {
    /// Animation time in seconds. `None` = rest pose.
    pub time: Option<f32>,
    /// Animation index sampled at `time` (`None` ⇒ the first).
    pub animation: Option<usize>,
    /// Active `KHR_materials_variants` variant.
    pub material_variant: Option<usize>,
    /// Use the scene's punctual lights when it has any.
    pub use_scene_lights: bool,
    /// Directional light used when the scene contributes none (or
    /// `use_scene_lights` is off). `None` ⇒ no light at all.
    pub fallback_light: Option<LightSpec>,
    /// Generate tangents for normal-mapped primitives lacking them.
    pub generate_tangents: bool,
}

impl Default for PrepareOptions {
    fn default() -> Self {
        Self {
            time: None,
            animation: None,
            material_variant: None,
            use_scene_lights: true,
            fallback_light: Some(LightSpec::default_light()),
            generate_tangents: true,
        }
    }
}

impl PrepareOptions {
    /// The preparation inputs implied by a [`RenderOptions`].
    pub fn from_render_options(opts: &RenderOptions) -> Self {
        Self {
            time: opts.time,
            animation: opts.animation,
            material_variant: opts.material_variant,
            use_scene_lights: opts.use_scene_lights,
            fallback_light: Some(opts.light),
            generate_tangents: true,
        }
    }
}

// ---------------------------------------------------------------------
// Output types.
// ---------------------------------------------------------------------

/// Primitive class of a [`DrawItem`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DrawTopology {
    /// Unindexed triangle list — 3 vertices per triangle, CCW front.
    Triangles,
    /// Unindexed line list — 2 vertices per segment.
    Lines,
    /// Point list — 1 vertex per point.
    Points,
}

/// One drawable: a primitive instance with world-space, de-indexed
/// vertex streams. Every non-empty attribute vector has exactly
/// `positions.len()` entries.
#[derive(Debug, Clone, PartialEq)]
pub struct DrawItem {
    /// Scene node instancing the mesh.
    pub node: NodeId,
    /// Source mesh.
    pub mesh: MeshId,
    /// Primitive index within the mesh.
    pub primitive: usize,
    /// Primitive class.
    pub topology: DrawTopology,
    /// The node world matrix baked into the vertex data (identity for
    /// skinned items, whose vertices the joint palette already put in
    /// world space). Informational — positions are already world
    /// space.
    pub world: [[f32; 4]; 4],
    /// Inverse-transpose of `world`'s upper 3×3 (normal matrix).
    pub normal_matrix: [[f32; 3]; 3],
    /// `true` when the item was skinned on the CPU.
    pub skinned: bool,
    /// World-space positions.
    pub positions: Vec<[f32; 3]>,
    /// World-space unit normals (per vertex; face normals when
    /// `has_vertex_normals` is false). Empty for lines / points without
    /// source normals.
    pub normals: Vec<[f32; 3]>,
    /// `false` when `normals` are synthesised flat face normals.
    pub has_vertex_normals: bool,
    /// World-space tangents (`xyz` unit, `w` = ±1 bitangent sign,
    /// `B = w · (N × T)`). Empty when unavailable.
    pub tangents: Vec<[f32; 4]>,
    /// Texture-coordinate sets (`TEXCOORD_n` at index `n`); a set that
    /// is absent or malformed in the source is empty.
    pub uvs: Vec<Vec<[f32; 2]>>,
    /// `COLOR_0` as linear RGBA, empty when absent.
    pub colors: Vec<[f32; 4]>,
    /// Index into [`PreparedScene::materials`] (always valid).
    pub material: usize,
    /// World AABB minimum (finite; equals `bounds_max` for an empty
    /// item, which never occurs in a built scene).
    pub bounds_min: [f32; 3],
    /// World AABB maximum.
    pub bounds_max: [f32; 3],
}

impl DrawItem {
    /// Number of triangles / segments / points.
    pub fn primitive_count(&self) -> usize {
        match self.topology {
            DrawTopology::Triangles => self.positions.len() / 3,
            DrawTopology::Lines => self.positions.len() / 2,
            DrawTopology::Points => self.positions.len(),
        }
    }

    /// UV set `set`, if present.
    pub fn uv_set(&self, set: u32) -> Option<&[[f32; 2]]> {
        self.uvs
            .get(set as usize)
            .map(Vec::as_slice)
            .filter(|v| !v.is_empty())
    }
}

/// A texture reference resolved against [`PreparedScene::textures`].
/// Only emitted when the texture decoded successfully.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextureBinding {
    /// Index into [`PreparedScene::textures`] (`Some` there).
    pub texture: usize,
    /// Effective `TEXCOORD_n` set (`KHR_texture_transform` override
    /// applied).
    pub uv_set: u32,
    /// `KHR_texture_transform` (`None` = identity).
    pub transform: Option<TextureTransform>,
}

impl TextureBinding {
    /// Apply the binding's UV transform.
    pub fn transform_uv(&self, uv: [f32; 2]) -> [f32; 2] {
        match &self.transform {
            Some(t) => t.apply(uv),
            None => uv,
        }
    }
}

/// glTF metallic-roughness material with factors resolved and
/// textures bound to the prepared texture table.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedMaterial {
    /// Source material (`None` for the default material).
    pub source: Option<MaterialId>,
    /// Linear RGBA base colour factor.
    pub base_color: [f32; 4],
    /// Base colour texture (sRGB).
    pub base_color_texture: Option<TextureBinding>,
    /// Metallic factor `[0, 1]`.
    pub metallic: f32,
    /// Perceptual roughness factor `[0, 1]`.
    pub roughness: f32,
    /// Metallic (B) / roughness (G) texture (linear).
    pub metallic_roughness_texture: Option<TextureBinding>,
    /// Tangent-space normal map (linear).
    pub normal_texture: Option<TextureBinding>,
    /// Normal-map XY scale.
    pub normal_scale: f32,
    /// Occlusion texture (R, linear).
    pub occlusion_texture: Option<TextureBinding>,
    /// Occlusion strength `[0, 1]`.
    pub occlusion_strength: f32,
    /// Linear emissive radiance factor, `KHR_materials_emissive_strength`
    /// already multiplied in.
    pub emissive: [f32; 3],
    /// Emissive texture (sRGB).
    pub emissive_texture: Option<TextureBinding>,
    /// Alpha mode (with MASK cutoff).
    pub alpha_mode: AlphaMode,
    /// Disable back-face culling; back faces shade with flipped
    /// normals.
    pub double_sided: bool,
    /// `KHR_materials_unlit`.
    pub unlit: bool,
    /// Index of refraction (`KHR_materials_ior`, default 1.5).
    pub ior: f32,
    /// The remaining KHR extensions verbatim (clearcoat, sheen,
    /// transmission, volume, …) for backends that model them. Texture
    /// references inside are *unresolved* scene texture ids.
    pub ext: MaterialExt,
}

impl PreparedMaterial {
    /// The material used by primitives without one — matches the
    /// historical renderer fallback colour: grey-blue, dielectric,
    /// fully rough.
    pub fn fallback() -> Self {
        Self {
            source: None,
            base_color: [0.7, 0.7, 0.75, 1.0],
            base_color_texture: None,
            metallic: 0.0,
            roughness: 1.0,
            metallic_roughness_texture: None,
            normal_texture: None,
            normal_scale: 1.0,
            occlusion_texture: None,
            occlusion_strength: 1.0,
            emissive: [0.0; 3],
            emissive_texture: None,
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
            unlit: false,
            ior: 1.5,
            ext: MaterialExt::default(),
        }
    }

    fn from_material(id: MaterialId, m: &Material, textures: &[Option<PreparedTexture>]) -> Self {
        let bind = |r: &Option<TextureRef>| -> Option<TextureBinding> {
            let r = r.as_ref()?;
            let idx = r.texture.0 as usize;
            textures.get(idx)?.as_ref()?;
            Some(TextureBinding {
                texture: idx,
                uv_set: r.effective_uv_set(),
                transform: r.transform.filter(|t| !t.is_identity() && t.is_finite()),
            })
        };
        let fin = |v: f32, d: f32| if v.is_finite() { v } else { d };
        let strength = m
            .ext
            .emissive_strength
            .map(|s| fin(s, 1.0).max(0.0))
            .unwrap_or(1.0);
        Self {
            source: Some(id),
            base_color: [
                fin(m.base_color[0], 1.0),
                fin(m.base_color[1], 1.0),
                fin(m.base_color[2], 1.0),
                fin(m.base_color[3], 1.0).clamp(0.0, 1.0),
            ],
            base_color_texture: bind(&m.base_color_texture),
            metallic: fin(m.metallic, 1.0).clamp(0.0, 1.0),
            roughness: fin(m.roughness, 1.0).clamp(0.0, 1.0),
            metallic_roughness_texture: bind(&m.metallic_roughness_texture),
            normal_texture: bind(&m.normal_texture),
            normal_scale: fin(m.normal_scale, 1.0),
            occlusion_texture: bind(&m.occlusion_texture),
            occlusion_strength: fin(m.occlusion_strength, 1.0).clamp(0.0, 1.0),
            emissive: [
                fin(m.emissive_factor[0], 0.0).max(0.0) * strength,
                fin(m.emissive_factor[1], 0.0).max(0.0) * strength,
                fin(m.emissive_factor[2], 0.0).max(0.0) * strength,
            ],
            emissive_texture: bind(&m.emissive_texture),
            alpha_mode: m.alpha_mode,
            double_sided: m.double_sided,
            unlit: m.ext.unlit,
            ior: m.ext.ior.map(|v| fin(v, 1.5)).unwrap_or(1.5).max(1.0),
            ext: m.ext.clone(),
        }
    }

    /// Dielectric specular reflectance at normal incidence implied by
    /// [`Self::ior`]: `((ior − 1) / (ior + 1))²` (0.04 at 1.5).
    pub fn dielectric_f0(&self) -> f32 {
        let r = (self.ior - 1.0) / (self.ior + 1.0);
        r * r
    }
}

/// Punctual light class (`KHR_lights_punctual`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LightKind {
    /// Infinitely distant; `intensity` is illuminance (lux).
    Directional,
    /// Omnidirectional point; `intensity` is luminous intensity (cd).
    Point,
    /// Cone-limited point; `intensity` in candela.
    Spot,
}

/// A light placed in world space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreparedLight {
    /// Light class.
    pub kind: LightKind,
    /// Linear RGB colour.
    pub color: [f32; 3],
    /// Intensity (lux for directional, candela otherwise).
    pub intensity: f32,
    /// World position (unused for directional).
    pub position: [f32; 3],
    /// Unit world direction the light *travels* (its node's `-Z`);
    /// unused for point lights.
    pub direction: [f32; 3],
    /// Cut-off range (`None` = infinite).
    pub range: Option<f32>,
    /// Spot inner cone half-angle (radians).
    pub inner_cone_angle: f32,
    /// Spot outer cone half-angle (radians).
    pub outer_cone_angle: f32,
    /// Node carrying the light (`None` for the options fallback).
    pub node: Option<NodeId>,
}

/// Light arriving at a shading point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightSample {
    /// Unit vector from the shading point toward the light.
    pub l: [f32; 3],
    /// Distance to the light (`f32::INFINITY` for directional).
    pub distance: f32,
    /// Incident irradiance on a surface facing the light
    /// (`color · intensity · attenuation`), before the `N·L` cosine.
    pub radiance: [f32; 3],
}

impl PreparedLight {
    /// The options fallback light as a directional light. Its
    /// illuminance is `π · spec.intensity`, so a white Lambertian
    /// surface facing it reflects radiance `spec.intensity` — the same
    /// brightness the legacy Lambert shading assigns at unit intensity.
    pub fn from_light_spec(spec: LightSpec) -> Self {
        let az = spec.azimuth_deg.to_radians();
        let el = spec.elevation_deg.to_radians();
        let toward = vec3_normalise([el.cos() * az.sin(), el.sin(), el.cos() * az.cos()]);
        Self {
            kind: LightKind::Directional,
            color: [1.0; 3],
            intensity: std::f32::consts::PI * spec.intensity.max(0.0),
            position: [0.0; 3],
            direction: [-toward[0], -toward[1], -toward[2]],
            range: None,
            inner_cone_angle: 0.0,
            outer_cone_angle: std::f32::consts::FRAC_PI_4,
            node: None,
        }
    }

    fn from_scene(light: &Light, world: &[[f32; 4]; 4], node: NodeId) -> Self {
        let position = [world[0][3], world[1][3], world[2][3]];
        let mut direction = vec3_normalise([-world[0][2], -world[1][2], -world[2][2]]);
        if vec3_dot(direction, direction) < 0.5 {
            direction = [0.0, 0.0, -1.0];
        }
        let fin = |v: f32| if v.is_finite() { v.max(0.0) } else { 0.0 };
        let range = |r: Option<f32>| r.filter(|r| r.is_finite() && *r > 0.0);
        let (kind, color, intensity, range, inner, outer) = match *light {
            Light::Directional { color, intensity } => {
                (LightKind::Directional, color, intensity, None, 0.0, 0.0)
            }
            Light::Point {
                color,
                intensity,
                range: r,
            } => (LightKind::Point, color, intensity, range(r), 0.0, 0.0),
            Light::Spot {
                color,
                intensity,
                range: r,
                inner_cone_angle,
                outer_cone_angle,
            } => {
                let outer = if outer_cone_angle.is_finite() {
                    outer_cone_angle.clamp(1.0e-4, std::f32::consts::FRAC_PI_2)
                } else {
                    std::f32::consts::FRAC_PI_4
                };
                let inner = if inner_cone_angle.is_finite() {
                    inner_cone_angle.clamp(0.0, outer)
                } else {
                    0.0
                };
                (LightKind::Spot, color, intensity, range(r), inner, outer)
            }
        };
        Self {
            kind,
            color: [fin(color[0]), fin(color[1]), fin(color[2])],
            intensity: fin(intensity),
            position,
            direction,
            range,
            inner_cone_angle: inner,
            outer_cone_angle: outer,
            node: Some(node),
        }
    }

    /// Light arriving at world point `p`, per the `KHR_lights_punctual`
    /// attenuation model: inverse-square falloff windowed by
    /// `clamp(1 − (d / range)⁴, 0, 1)` when a range is set, and for
    /// spots the smooth cone term
    /// `saturate((cos θ − cos outer) / (cos inner − cos outer))²`.
    /// `None` when the point receives nothing (outside range / cone).
    pub fn sample(&self, p: [f32; 3]) -> Option<LightSample> {
        let base = [
            self.color[0] * self.intensity,
            self.color[1] * self.intensity,
            self.color[2] * self.intensity,
        ];
        if self.kind == LightKind::Directional {
            return Some(LightSample {
                l: [-self.direction[0], -self.direction[1], -self.direction[2]],
                distance: f32::INFINITY,
                radiance: base,
            });
        }
        let to = vec3_sub(self.position, p);
        let d2 = vec3_dot(to, to);
        if d2.is_nan() || d2 <= 1.0e-12 {
            return None;
        }
        let d = d2.sqrt();
        let l = [to[0] / d, to[1] / d, to[2] / d];
        let mut att = 1.0 / d2;
        if let Some(r) = self.range {
            let x = d / r;
            att *= (1.0 - x * x * x * x).clamp(0.0, 1.0);
        }
        if self.kind == LightKind::Spot {
            let cos_outer = self.outer_cone_angle.cos();
            let cos_inner = self.inner_cone_angle.cos();
            let scale = 1.0 / (cos_inner - cos_outer).max(0.001);
            let offset = -cos_outer * scale;
            let cd = vec3_dot(self.direction, [-l[0], -l[1], -l[2]]);
            let a = (cd * scale + offset).clamp(0.0, 1.0);
            att *= a * a;
        }
        if att <= 0.0 {
            return None;
        }
        Some(LightSample {
            l,
            distance: d,
            radiance: [base[0] * att, base[1] * att, base[2] * att],
        })
    }
}

/// A scene camera placed by a node.
#[derive(Debug, Clone, Copy)]
pub struct SceneCameraInstance {
    /// Node carrying the camera.
    pub node: NodeId,
    /// Index into `Scene3D::cameras`.
    pub camera_index: usize,
    /// The camera definition.
    pub camera: oxideav_mesh3d::Camera,
    /// Node world matrix (posed when animated).
    pub world: [[f32; 4]; 4],
}

/// The flattened, world-space scene a backend renders.
#[derive(Debug, Clone)]
pub struct PreparedScene {
    /// Draw items in scene-graph pre-order (node order, then primitive
    /// order).
    pub items: Vec<DrawItem>,
    /// Material table: one entry per `Scene3D::materials` slot, then
    /// the fallback material at [`Self::default_material`].
    pub materials: Vec<PreparedMaterial>,
    /// Texture table parallel to `Scene3D::textures` (`None` when the
    /// image could not be resolved / decoded).
    pub textures: Vec<Option<PreparedTexture>>,
    /// Lights in world space: the scene's punctual lights (pre-order),
    /// or the fallback light.
    pub lights: Vec<PreparedLight>,
    /// Scene camera instances in pre-order
    /// ([`RenderOptions::scene_camera`] indexes this list).
    pub cameras: Vec<SceneCameraInstance>,
    /// `true` when `lights` came from the scene rather than the
    /// fallback.
    pub scene_lights: bool,
    bounds: Option<([f32; 3], [f32; 3])>,
}

impl PreparedScene {
    /// Prepare `scene` for rendering. Textures are resolved through
    /// `textures` (pass `&mut TextureCache::default()` when no image
    /// decoding is wanted).
    pub fn build(scene: &Scene3D, opts: &PrepareOptions, textures: &mut TextureCache) -> Self {
        let tex_table: Vec<Option<PreparedTexture>> =
            scene.textures.iter().map(|t| textures.prepare(t)).collect();
        let mut materials: Vec<PreparedMaterial> = scene
            .materials
            .iter()
            .enumerate()
            .map(|(i, m)| PreparedMaterial::from_material(MaterialId(i as u32), m, &tex_table))
            .collect();
        let default_material = materials.len();
        materials.push(PreparedMaterial::fallback());

        // 1. Pose.
        let n_nodes = scene.nodes.len();
        let pose = opts.time.filter(|t| t.is_finite()).and_then(|t| {
            let anim = scene.animations.get(opts.animation.unwrap_or(0))?;
            Some(anim.sample_pose(t, n_nodes))
        });
        let worlds = match &pose {
            Some(p) => scene.posed_node_transforms(p),
            None => scene.world_node_transforms(),
        };

        let variant = opts.material_variant.map(|v| MaterialVariantId(v as u32));
        let mut items = Vec::new();
        let mut lights = Vec::new();
        let mut cameras = Vec::new();
        for id in preorder_nodes(scene) {
            let idx = id.0 as usize;
            let (Some(node), Some(Some(world))) = (scene.nodes.get(idx), worlds.get(idx)) else {
                continue;
            };
            let world = *world;
            if let Some(light) = node.light.and_then(|l| scene.lights.get(l.0 as usize)) {
                lights.push(PreparedLight::from_scene(light, &world, id));
            }
            if let Some(ci) = node.camera {
                if let Some(cam) = scene.cameras.get(ci.0 as usize) {
                    cameras.push(SceneCameraInstance {
                        node: id,
                        camera_index: ci.0 as usize,
                        camera: *cam,
                        world,
                    });
                }
            }
            let Some(mesh_id) = node.mesh else { continue };
            let Some(mesh) = scene.meshes.get(mesh_id.0 as usize) else {
                continue;
            };
            let weights: &[f32] = match pose
                .as_ref()
                .and_then(|p| p.morph_weights.get(idx))
                .and_then(|w| w.as_deref())
            {
                Some(w) => w,
                None if !node.weights.is_empty() => &node.weights,
                None => &mesh.weights,
            };
            let palette = if node.skin.is_some() {
                scene.joint_matrices_with(id, &worlds)
            } else {
                None
            };
            for (pi, prim) in mesh.primitives.iter().enumerate() {
                let mat_index = prim
                    .material_for_variant(variant)
                    .map(|m| m.0 as usize)
                    .filter(|&m| m < default_material)
                    .unwrap_or(default_material);
                let needs_tangents =
                    opts.generate_tangents && materials[mat_index].normal_texture.is_some();
                let morphed = prim.morphed(weights);
                let has_influences = morphed.joints.is_some() && morphed.weights.is_some();
                let (baked, item_world, skinned) = match &palette {
                    Some(p) if has_influences => (morphed.skinned(p), identity4(), true),
                    _ => (morphed.transformed(world), world, false),
                };
                if let Some(item) = flatten_primitive(
                    &baked,
                    FlattenCtx {
                        node: id,
                        mesh: mesh_id,
                        primitive: pi,
                        world: item_world,
                        skinned,
                        material: mat_index,
                        needs_tangents,
                        tangent_uv_set: materials[mat_index]
                            .normal_texture
                            .map(|b| b.uv_set)
                            .unwrap_or(0),
                    },
                ) {
                    items.push(item);
                }
            }
        }

        let scene_lights = opts.use_scene_lights && !lights.is_empty();
        if !scene_lights {
            lights = opts
                .fallback_light
                .map(PreparedLight::from_light_spec)
                .into_iter()
                .collect();
        }

        let mut bounds: Option<([f32; 3], [f32; 3])> = None;
        for it in &items {
            bounds = Some(match bounds {
                None => (it.bounds_min, it.bounds_max),
                Some((mn, mx)) => (
                    [
                        mn[0].min(it.bounds_min[0]),
                        mn[1].min(it.bounds_min[1]),
                        mn[2].min(it.bounds_min[2]),
                    ],
                    [
                        mx[0].max(it.bounds_max[0]),
                        mx[1].max(it.bounds_max[1]),
                        mx[2].max(it.bounds_max[2]),
                    ],
                ),
            });
        }

        Self {
            items,
            materials,
            textures: tex_table,
            lights,
            cameras,
            scene_lights,
            bounds,
        }
    }

    /// Convenience: prepare with the inputs implied by `opts` and a
    /// throw-away no-op texture cache (only built-in raw textures
    /// resolve).
    pub fn from_render_options(scene: &Scene3D, opts: &RenderOptions) -> Self {
        Self::build(
            scene,
            &PrepareOptions::from_render_options(opts),
            &mut TextureCache::default(),
        )
    }

    /// Index of the fallback material in [`Self::materials`].
    pub fn default_material(&self) -> usize {
        self.materials.len() - 1
    }

    /// World AABB of all draw items, `None` for an empty scene.
    pub fn bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        self.bounds
    }

    /// [`Self::bounds`], or a unit box at the origin for an empty
    /// scene (keeps camera framing finite).
    pub fn bounds_or_unit(&self) -> ([f32; 3], [f32; 3]) {
        self.bounds.unwrap_or(([-0.5; 3], [0.5; 3]))
    }

    /// Total triangle count over triangle items.
    pub fn triangle_count(&self) -> usize {
        self.items
            .iter()
            .filter(|i| i.topology == DrawTopology::Triangles)
            .map(DrawItem::primitive_count)
            .sum()
    }

    /// Resolve a material's texture binding to its prepared texture.
    pub fn texture(&self, binding: &TextureBinding) -> Option<&PreparedTexture> {
        self.textures.get(binding.texture)?.as_ref()
    }
}

/// Reachable node ids in scene-graph pre-order, each node claimed once
/// at first arrival (cycle- and diamond-safe; out-of-range ids
/// skipped) — the traversal contract of
/// [`Scene3D::world_node_transforms`].
pub fn preorder_nodes(scene: &Scene3D) -> Vec<NodeId> {
    let mut visited = vec![false; scene.nodes.len()];
    let mut out = Vec::new();
    let mut stack: Vec<NodeId> = scene.roots.iter().rev().copied().collect();
    while let Some(id) = stack.pop() {
        match visited.get_mut(id.0 as usize) {
            Some(slot) if !*slot => *slot = true,
            _ => continue,
        }
        out.push(id);
        for &c in scene.nodes[id.0 as usize].children.iter().rev() {
            stack.push(c);
        }
    }
    out
}

// ---------------------------------------------------------------------
// Flattening.
// ---------------------------------------------------------------------

struct FlattenCtx {
    node: NodeId,
    mesh: MeshId,
    primitive: usize,
    world: [[f32; 4]; 4],
    skinned: bool,
    material: usize,
    needs_tangents: bool,
    tangent_uv_set: u32,
}

fn det3(m: &[[f32; 4]; 4]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn normal_matrix(m: &[[f32; 4]; 4]) -> [[f32; 3]; 3] {
    let d = det3(m);
    if !d.is_finite() || d.abs() <= 1.0e-20 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let inv_d = 1.0 / d;
    // Inverse-transpose = cofactor matrix / det.
    let c =
        |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    [
        [
            c(1, 1, 2, 2) * inv_d,
            -c(1, 0, 2, 2) * inv_d,
            c(1, 0, 2, 1) * inv_d,
        ],
        [
            -c(0, 1, 2, 2) * inv_d,
            c(0, 0, 2, 2) * inv_d,
            -c(0, 0, 2, 1) * inv_d,
        ],
        [
            c(0, 1, 1, 2) * inv_d,
            -c(0, 0, 1, 2) * inv_d,
            c(0, 0, 1, 1) * inv_d,
        ],
    ]
}

fn finite3(p: [f32; 3]) -> bool {
    p[0].is_finite() && p[1].is_finite() && p[2].is_finite()
}

fn flatten_primitive(prim: &Primitive, ctx: FlattenCtx) -> Option<DrawItem> {
    let nv = prim.positions.len();
    if nv == 0 {
        return None;
    }
    let normals_src = prim.normals.as_ref().filter(|n| n.len() == nv);
    let mut tangents_src: Option<Vec<[f32; 4]>> =
        prim.tangents.as_ref().filter(|t| t.len() == nv).cloned();
    let topology = match prim.topology {
        Topology::Triangles | Topology::TriangleStrip | Topology::TriangleFan => {
            DrawTopology::Triangles
        }
        Topology::Lines | Topology::LineStrip | Topology::LineLoop => DrawTopology::Lines,
        Topology::Points => DrawTopology::Points,
    };
    if tangents_src.is_none()
        && ctx.needs_tangents
        && topology == DrawTopology::Triangles
        && normals_src.is_some()
    {
        tangents_src = prim.compute_tangents(ctx.tangent_uv_set as usize);
    }

    // Vertex index sequence for the flattened stream.
    let seq: Vec<u32> = match &prim.indices {
        Some(oxideav_mesh3d::Indices::U16(v)) => v.iter().map(|&i| i as u32).collect(),
        Some(oxideav_mesh3d::Indices::U32(v)) => v.clone(),
        None => (0..nv as u32).collect(),
    };
    let flip = !ctx.skinned && det3(&ctx.world) < 0.0;
    let ok = |i: u32| (i as usize) < nv && finite3(prim.positions[i as usize]);
    let mut corners: Vec<u32> = Vec::new();
    match topology {
        DrawTopology::Triangles => {
            for [a, b, c] in prim.triangle_indices() {
                if ok(a) && ok(b) && ok(c) {
                    if flip {
                        corners.extend_from_slice(&[a, c, b]);
                    } else {
                        corners.extend_from_slice(&[a, b, c]);
                    }
                }
            }
        }
        DrawTopology::Lines => {
            let n = seq.len();
            let mut push = |a: u32, b: u32| {
                if ok(a) && ok(b) {
                    corners.extend_from_slice(&[a, b]);
                }
            };
            match prim.topology {
                Topology::Lines => {
                    for pair in seq.chunks_exact(2) {
                        push(pair[0], pair[1]);
                    }
                }
                _ => {
                    for i in 0..n.saturating_sub(1) {
                        push(seq[i], seq[i + 1]);
                    }
                    if prim.topology == Topology::LineLoop && n >= 2 {
                        push(seq[n - 1], seq[0]);
                    }
                }
            }
        }
        DrawTopology::Points => {
            corners.extend(seq.iter().copied().filter(|&i| ok(i)));
        }
    }
    if corners.is_empty() {
        return None;
    }

    let gather3 = |src: &[[f32; 3]]| corners.iter().map(|&i| src[i as usize]).collect::<Vec<_>>();
    let positions = gather3(&prim.positions);

    let (normals, has_vertex_normals) = match normals_src {
        Some(ns) => (
            corners
                .iter()
                .map(|&i| {
                    let n = vec3_normalise(ns[i as usize]);
                    if finite3(n) {
                        n
                    } else {
                        [0.0, 0.0, 1.0]
                    }
                })
                .collect(),
            true,
        ),
        None if topology == DrawTopology::Triangles => {
            let mut out = Vec::with_capacity(positions.len());
            for t in positions.chunks_exact(3) {
                let n = vec3_normalise(vec3_cross(vec3_sub(t[1], t[0]), vec3_sub(t[2], t[0])));
                out.extend_from_slice(&[n, n, n]);
            }
            (out, false)
        }
        None => (Vec::new(), false),
    };

    let uvs: Vec<Vec<[f32; 2]>> = prim
        .uvs
        .iter()
        .map(|set| {
            if set.len() == nv {
                corners.iter().map(|&i| set[i as usize]).collect()
            } else {
                Vec::new()
            }
        })
        .collect();

    let mut tangents: Vec<[f32; 4]> = match &tangents_src {
        Some(ts) => corners
            .iter()
            .map(|&i| {
                let t = ts[i as usize];
                let d = vec3_normalise([t[0], t[1], t[2]]);
                [d[0], d[1], d[2], if t[3] < 0.0 { -1.0 } else { 1.0 }]
            })
            .collect(),
        None => Vec::new(),
    };
    if tangents.is_empty() && ctx.needs_tangents && topology == DrawTopology::Triangles {
        if let Some(uv) = uvs
            .get(ctx.tangent_uv_set as usize)
            .filter(|u| !u.is_empty())
        {
            tangents = face_tangents(&positions, &normals, uv);
        }
    }

    let colors: Vec<[f32; 4]> = match prim.colors.first() {
        Some(set) if set.len() == nv => corners.iter().map(|&i| set[i as usize]).collect(),
        _ => Vec::new(),
    };

    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for p in &positions {
        for k in 0..3 {
            mn[k] = mn[k].min(p[k]);
            mx[k] = mx[k].max(p[k]);
        }
    }

    Some(DrawItem {
        node: ctx.node,
        mesh: ctx.mesh,
        primitive: ctx.primitive,
        topology,
        world: ctx.world,
        normal_matrix: normal_matrix(&ctx.world),
        skinned: ctx.skinned,
        positions,
        normals,
        has_vertex_normals,
        tangents,
        uvs,
        colors,
        material: ctx.material,
        bounds_min: mn,
        bounds_max: mx,
    })
}

/// Per-face tangents from the triangle's UV gradient (Lengyel,
/// "Computing Tangent Space Basis Vectors for an Arbitrary Mesh",
/// 2001), Gram-Schmidt-orthogonalised against each corner normal.
fn face_tangents(pos: &[[f32; 3]], nrm: &[[f32; 3]], uv: &[[f32; 2]]) -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity(pos.len());
    for t in 0..pos.len() / 3 {
        let (p0, p1, p2) = (pos[3 * t], pos[3 * t + 1], pos[3 * t + 2]);
        let (w0, w1, w2) = (uv[3 * t], uv[3 * t + 1], uv[3 * t + 2]);
        let e1 = vec3_sub(p1, p0);
        let e2 = vec3_sub(p2, p0);
        let (du1, dv1) = (w1[0] - w0[0], w1[1] - w0[1]);
        let (du2, dv2) = (w2[0] - w0[0], w2[1] - w0[1]);
        let det = du1 * dv2 - du2 * dv1;
        let (tan, bit) = if det.abs() > 1.0e-12 {
            let r = 1.0 / det;
            (
                [
                    (e1[0] * dv2 - e2[0] * dv1) * r,
                    (e1[1] * dv2 - e2[1] * dv1) * r,
                    (e1[2] * dv2 - e2[2] * dv1) * r,
                ],
                [
                    (e2[0] * du1 - e1[0] * du2) * r,
                    (e2[1] * du1 - e1[1] * du2) * r,
                    (e2[2] * du1 - e1[2] * du2) * r,
                ],
            )
        } else {
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        };
        for k in 0..3 {
            let n = nrm.get(3 * t + k).copied().unwrap_or([0.0, 0.0, 1.0]);
            let d = vec3_dot(n, tan);
            let mut tt = vec3_normalise([tan[0] - n[0] * d, tan[1] - n[1] * d, tan[2] - n[2] * d]);
            if vec3_dot(tt, tt) < 0.5 {
                // Any vector perpendicular to n.
                let a = if n[0].abs() < 0.9 {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 1.0, 0.0]
                };
                tt = vec3_normalise(vec3_cross(a, n));
            }
            let w = if vec3_dot(vec3_cross(n, tt), bit) < 0.0 {
                -1.0
            } else {
                1.0
            };
            out.push([tt[0], tt[1], tt[2], w]);
        }
    }
    out
}

/// Shared handle for callers that want to keep one prepared scene
/// around (e.g. a GPU backend uploading once per frame).
pub type SharedPreparedScene = Arc<PreparedScene>;

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_mesh3d::{
        Animation, AnimationProperty, AnimationSampler, AnimationValues, Indices, Interpolation,
        Mesh, Node, Transform,
    };

    fn tri_scene(transform: Transform) -> Scene3D {
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mut scene = Scene3D::new();
        let m = scene.add_mesh(Mesh::new("t".to_string()).with_primitive(prim));
        let n = scene.add_node(Node::new().with_mesh(m).with_transform(transform));
        scene.add_root(n);
        scene
    }

    fn build(scene: &Scene3D, opts: &PrepareOptions) -> PreparedScene {
        PreparedScene::build(scene, opts, &mut TextureCache::default())
    }

    #[test]
    fn flat_normals_and_world_space_positions() {
        let scene = tri_scene(Transform::Trs {
            translation: [0.0, 0.0, 2.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        });
        let p = build(&scene, &PrepareOptions::default());
        assert_eq!(p.items.len(), 1);
        let it = &p.items[0];
        assert_eq!(it.positions[1], [1.0, 0.0, 2.0]);
        assert!(!it.has_vertex_normals);
        assert_eq!(it.normals[0], [0.0, 0.0, 1.0]);
        assert_eq!(it.material, p.default_material());
        assert_eq!(p.lights.len(), 1, "fallback light");
        assert!(!p.scene_lights);
        assert_eq!(p.bounds().unwrap().1, [1.0, 1.0, 2.0]);
    }

    #[test]
    fn mirrored_world_keeps_ccw_front_facing() {
        let scene = tri_scene(Transform::Trs {
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [-1.0, 1.0, 1.0],
        });
        let p = build(&scene, &PrepareOptions::default());
        let t = &p.items[0].positions;
        let n = vec3_cross(vec3_sub(t[1], t[0]), vec3_sub(t[2], t[0]));
        // Mirroring X of a +Z-facing triangle keeps it facing +Z
        // geometrically; winding must have been fixed so CCW stays front.
        assert!(n[2] > 0.0, "{n:?}");
    }

    #[test]
    fn strips_lines_and_garbage_indices() {
        let mut strip = Primitive::new(Topology::TriangleStrip);
        strip.positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let mut loop_ = Primitive::new(Topology::LineLoop);
        loop_.positions = strip.positions.clone();
        let mut bad = Primitive::new(Topology::Triangles);
        bad.positions = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        bad.indices = Some(Indices::U32(vec![0, 1, 2, 0, 1, 99]));
        let mut scene = Scene3D::new();
        let m = scene.add_mesh(
            Mesh::new("m".to_string())
                .with_primitive(strip)
                .with_primitive(loop_)
                .with_primitive(bad),
        );
        let n = scene.add_node(Node::new().with_mesh(m));
        scene.add_root(n);
        let p = build(&scene, &PrepareOptions::default());
        assert_eq!(p.items.len(), 3);
        assert_eq!(p.items[0].primitive_count(), 2);
        assert_eq!(p.items[1].topology, DrawTopology::Lines);
        assert_eq!(p.items[1].primitive_count(), 4);
        assert_eq!(p.items[2].primitive_count(), 1);
        // Both strip triangles face +Z.
        for t in p.items[0].positions.chunks_exact(3) {
            assert!(vec3_cross(vec3_sub(t[1], t[0]), vec3_sub(t[2], t[0]))[2] > 0.0);
        }
    }

    #[test]
    fn scene_lights_resolve_with_world_transforms() {
        let mut scene = tri_scene(Transform::identity());
        let l = scene.add_light(Light::Spot {
            color: [1.0, 0.5, 0.25],
            intensity: 10.0,
            range: Some(5.0),
            inner_cone_angle: 0.2,
            outer_cone_angle: 0.4,
        });
        let mut node = Node::new().with_transform(Transform::Trs {
            translation: [0.0, 3.0, 0.0],
            // -90° about X: local -Z → world -Y.
            rotation: [
                -std::f32::consts::FRAC_1_SQRT_2,
                0.0,
                0.0,
                std::f32::consts::FRAC_1_SQRT_2,
            ],
            scale: [1.0; 3],
        });
        node.light = Some(l);
        let n = scene.add_node(node);
        scene.add_root(n);
        let p = build(&scene, &PrepareOptions::default());
        assert!(p.scene_lights);
        let light = p.lights[0];
        assert_eq!(light.kind, LightKind::Spot);
        assert!(
            (light.direction[1] + 1.0).abs() < 1e-5,
            "{:?}",
            light.direction
        );
        // Straight below: inside the cone, inverse-square × window.
        let s = light.sample([0.0, 1.0, 0.0]).unwrap();
        let window = 1.0 - (2.0f32 / 5.0).powi(4);
        assert!((s.radiance[0] - 10.0 / 4.0 * window).abs() < 1e-4, "{s:?}");
        // Off-axis outside the outer cone: nothing.
        assert!(light.sample([3.0, 2.0, 0.0]).is_none());
        // Beyond range: nothing.
        assert!(light.sample([0.0, -3.0, 0.0]).is_none());
        // Disabled scene lights → fallback.
        let p2 = build(
            &scene,
            &PrepareOptions {
                use_scene_lights: false,
                ..PrepareOptions::default()
            },
        );
        assert!(!p2.scene_lights);
        assert_eq!(p2.lights[0].kind, LightKind::Directional);
    }

    #[test]
    fn fallback_light_matches_legacy_lambert_scale() {
        let l = PreparedLight::from_light_spec(LightSpec {
            azimuth_deg: 0.0,
            elevation_deg: 90.0,
            intensity: 1.0,
        });
        let s = l.sample([0.0; 3]).unwrap();
        assert!((s.l[1] - 1.0).abs() < 1e-6);
        // E/π = 1 → white Lambert radiance 1.
        assert!((s.radiance[0] / std::f32::consts::PI - 1.0).abs() < 1e-6);
    }

    #[test]
    fn animation_time_moves_geometry() {
        let mut scene = tri_scene(Transform::identity());
        let sampler = AnimationSampler {
            keyframes: vec![0.0, 1.0],
            values: AnimationValues::Vec3(vec![[0.0; 3], [10.0, 0.0, 0.0]]),
            interpolation: Interpolation::Linear,
        };
        scene.add_animation(Animation::new(Some("move".to_string())).with_channel(
            NodeId(0),
            AnimationProperty::Translation,
            sampler,
        ));
        let rest = build(&scene, &PrepareOptions::default());
        assert_eq!(rest.items[0].positions[0], [0.0; 3]);
        let half = build(
            &scene,
            &PrepareOptions {
                time: Some(0.5),
                ..PrepareOptions::default()
            },
        );
        assert!((half.items[0].positions[0][0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn camera_instances_collected() {
        let mut scene = tri_scene(Transform::identity());
        let c = scene.add_camera(oxideav_mesh3d::Camera::perspective(0.8, 0.1));
        let mut node = Node::new();
        node.camera = Some(c);
        let n = scene.add_node(node);
        scene.add_root(n);
        let p = build(&scene, &PrepareOptions::default());
        assert_eq!(p.cameras.len(), 1);
        assert_eq!(p.cameras[0].node, n);
    }

    #[test]
    fn normal_matrix_of_scale_is_inverse() {
        let mut m = identity4();
        m[0][0] = 2.0;
        let n = normal_matrix(&m);
        assert!((n[0][0] - 0.5).abs() < 1e-6);
        assert!((n[1][1] - 1.0).abs() < 1e-6);
    }
}
