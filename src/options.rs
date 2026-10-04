//! [`RenderOptions`] + companion enums — the surface that every
//! [`crate::Renderer`] consumes.

use crate::error::{Error, Result};
pub use crate::hdr::ToneMap;

/// Backend selector used by [`crate::make_renderer`].
///
/// Phase A shipped the selector with no working backend. Phase B
/// filled in `Scanline`. Phase D fills in `Raycast` (Whitted-style
/// primary + shadow + reflection / refraction). Phase E adds
/// `PathTrace` (Kajiya unbiased path tracing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RenderBackend {
    /// Scanline rasteriser — Gouraud / Phong / Wireframe / Flat /
    /// NormalDebug / DepthDebug. Half-space edge-function pipeline
    /// with a per-pixel z-buffer. Fast, no global illumination, no
    /// raytraced shadows.
    Scanline,
    /// Whitted recursive ray tracer — same shading-mode surface as
    /// `Scanline`, rendering the same prepared scene through the same
    /// camera. `Phong` adds raytraced hard shadows and recursive
    /// reflection / refraction driven by material metallic /
    /// roughness / transmission / IOR; `Pbr` evaluates the scanline
    /// glTF model per hit and adds ray-traced shadows for every light
    /// type ([`RenderOptions::shadows`]), Fresnel-weighted mirror
    /// reflection below [`RenderOptions::reflection_roughness_cutoff`],
    /// refraction through `KHR_materials_transmission` / `_volume`,
    /// `MASK` as an any-hit filter and `BLEND` by continued rays, up
    /// to [`RenderOptions::max_ray_depth`] bounces. Line and point
    /// topologies have no surface area and are invisible to rays.
    Raycast,
    /// Unbiased Monte Carlo path tracer (Kajiya 1986): next-event
    /// estimation to punctual and emissive-triangle lights with
    /// multiple importance sampling, the glTF metallic-roughness BSDF
    /// plus transmission / volume / clearcoat / sheen / specular,
    /// Russian roulette, and a uniform sky ([`RenderOptions::ambient`]).
    /// Controlled by [`RenderOptions::path_trace`]; ignores
    /// [`RenderOptions::shading`] and [`RenderOptions::aa`] (the
    /// per-pixel sample count anti-aliases). See
    /// [`crate::pathtrace`].
    PathTrace,
}

/// Which estimator the path tracer uses for light that *can* be
/// sampled explicitly (emissive triangles, an environment map).
/// Punctual lights are always sampled by next-event estimation.
/// [`LightStrategy::Mis`] is the production default; the others exist
/// to verify it (all three converge to the same image).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LightStrategy {
    /// Light sampling and BSDF sampling combined with the power
    /// heuristic (Veach 1997, §9.2).
    #[default]
    Mis,
    /// Next-event estimation only; emitters found by BSDF rays are
    /// ignored.
    LightOnly,
    /// BSDF sampling only; no next-event estimation to area lights /
    /// environment maps.
    BsdfOnly,
}

/// Path-tracer controls ([`RenderBackend::PathTrace`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathTraceOptions {
    /// Samples per pixel a full [`crate::Renderer::render`] takes
    /// (`1..=1_048_576`). Default `64`.
    pub samples_per_pixel: u32,
    /// Maximum number of scattering events per path (`0..=1024`). `0`
    /// shows emitters only, `1` is direct lighting, larger values add
    /// indirect bounces. Default `8`.
    pub max_bounces: u32,
    /// Bounce index from which Russian roulette may terminate paths.
    /// Default `3`.
    pub rr_start: u32,
    /// Firefly clamp: every non-camera-visible radiance contribution
    /// is scaled so its largest channel is at most this value. `0`
    /// (default) disables the clamp; any other value biases the
    /// estimate (darker highlights) in exchange for less noise.
    pub clamp: f32,
    /// Seed mixed into every pixel's sample sequence. Renders are a
    /// deterministic function of (scene, options, seed).
    pub seed: u32,
    /// Area-light / environment estimator. Default MIS.
    pub strategy: LightStrategy,
}

impl Default for PathTraceOptions {
    fn default() -> Self {
        Self {
            samples_per_pixel: 64,
            max_bounces: 8,
            rr_start: 3,
            clamp: 0.0,
            seed: 0,
            strategy: LightStrategy::Mis,
        }
    }
}

/// Shading model selector consumed by every backend.
///
/// Per-pixel shading inputs (material colour, normals) come from
/// [`oxideav_mesh3d::Scene3D`]. The choice of model is decoupled from
/// the choice of backend so that a future raycast backend can also
/// honour [`ShadingMode::Phong`] etc.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ShadingMode {
    /// Constant material colour per triangle (no per-pixel lighting).
    /// Cheapest mode; works on any scene whether or not per-vertex
    /// normals were loaded.
    Flat,
    /// Per-vertex lighting interpolated across the triangle.
    Gouraud,
    /// Per-pixel lighting (normal interpolation + per-pixel lit).
    /// Smoothest result; default for callers who don't pick one.
    #[default]
    Phong,
    /// Bresenham triangle edges only, no fill.
    Wireframe,
    /// Visualise per-pixel normal as `((n + 1) / 2) * 255`.
    NormalDebug,
    /// Visualise NDC depth as grayscale (white = near, black = far).
    DepthDebug,
    /// Physically-based glTF 2.0 metallic-roughness shading (Appendix
    /// B: GGX / Trowbridge-Reitz NDF, height-correlated Smith
    /// visibility, Schlick Fresnel, Lambert diffuse) with textures,
    /// normal / occlusion / emissive maps, vertex colours,
    /// `KHR_materials_unlit` / `_emissive_strength` / `_ior`, the
    /// scene's `KHR_lights_punctual` lights (falling back to
    /// [`RenderOptions::light`]), alpha modes, back-face culling, and
    /// optional shadow maps ([`RenderOptions::shadows`]).
    ///
    /// The `Raycast` backend evaluates the same model per ray hit and
    /// adds ray-traced hard shadows (instead of shadow maps), mirror
    /// reflections, refraction (`KHR_materials_transmission` /
    /// `_volume`) and see-through `BLEND`.
    Pbr,
}

/// Camera projection type.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Projection {
    /// Perspective projection (default).
    #[default]
    Perspective,
    /// Orthographic projection — parallel rays, no foreshortening.
    /// Useful for engineering / isometric / part-diagram renders.
    Orthographic,
}

/// Background colour for the cleared framebuffer (RGBA8). Default is
/// fully transparent black (all zeros).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BackgroundColor(pub [u8; 4]);

impl From<[u8; 4]> for BackgroundColor {
    fn from(rgba: [u8; 4]) -> Self {
        Self(rgba)
    }
}

/// Directional light spec consumed by the Gouraud / Phong rasterisers.
///
/// `azimuth` and `elevation` are in degrees; `intensity` is a unit
/// scalar in `[0.0, ~]` multiplied into the diffuse term. The
/// rasteriser also applies a small constant ambient term so back-faces
/// stay visible.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightSpec {
    /// Rotation around `+Y`, measured from `+Z` toward `+X` (degrees).
    pub azimuth_deg: f32,
    /// Pitch above the `XZ` plane (degrees).
    pub elevation_deg: f32,
    /// Diffuse multiplier. Must be `>= 0.0` and finite.
    pub intensity: f32,
}

impl LightSpec {
    /// Default directional light: from the upper-right-front quadrant
    /// at unit intensity. Matches the renderer baseline so callers
    /// never have to specify a light explicitly.
    pub fn default_light() -> Self {
        Self {
            azimuth_deg: 45.0,
            elevation_deg: 45.0,
            intensity: 1.0,
        }
    }
}

impl Default for LightSpec {
    fn default() -> Self {
        Self::default_light()
    }
}

/// Camera placement override. `elevation` / `azimuth` are in degrees;
/// `distance` is a positive multiplier of the scene bounding-sphere
/// radius (`1.0` ≈ scene touches the framebuffer edge; the auto-frame
/// default is `~1.2`).
///
/// When [`RenderOptions::camera`] is `None`, the scanline backend
/// auto-frames the scene's axis-aligned bounding box at a 60° vertical
/// FOV — exactly the IM `convert` default for vector rasterisation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraSpec {
    /// Pitch of the orbit above the scene's XZ plane (degrees).
    pub elevation_deg: f32,
    /// Yaw of the orbit around the scene's Y axis (degrees).
    pub azimuth_deg: f32,
    /// Distance from the scene center, in units of the auto-frame
    /// bounding-sphere distance. Must be `> 0` and finite.
    pub distance: f32,
}

/// Caller-facing render options. The renderer interprets each field
/// according to the selected [`RenderBackend`].
///
/// `PartialEq` only — `fov_deg` is `f32`, which precludes `Eq`.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderOptions {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Framebuffer clear colour.
    pub background: BackgroundColor,
    /// Shading model (consumed by the scanline backend; future
    /// backends may interpret differently).
    pub shading: ShadingMode,
    /// Camera projection type.
    pub projection: Projection,
    /// Vertical field-of-view in degrees (perspective only). Must be
    /// in `(0, 180)`.
    pub fov_deg: f32,
    /// Directional light. Used by Gouraud / Phong shading;
    /// Flat / Wireframe / debug visualisers ignore it.
    pub light: LightSpec,
    /// Camera placement override. `None` ⇒ auto-frame the scene
    /// bounding box looking down the `+Z` axis toward `-Z`.
    /// For [`Projection::Orthographic`], `distance` scales the view
    /// volume (zoom).
    pub camera: Option<CameraSpec>,
    /// World-space offset added to the auto-frame / orbit look-at
    /// target (the scene bounds centre) — pans the auto / orbit camera
    /// (eye and target move together). Ignored for scene cameras.
    /// Default `[0, 0, 0]`.
    pub camera_target_offset: [f32; 3],
    /// Supersampling factor `[1, 8]`. `1` = off; higher values render
    /// `N×width × N×height` and box-filter down to the requested
    /// output. Hard cap of `8` because at 8× a 1024² render is a
    /// 16 M-pixel framebuffer + an 8 M f32 z-buffer (~80 MB).
    pub aa: u32,
    /// Tone-mapping operator applied to scene-linear radiance before
    /// sRGB encoding. Default [`ToneMap::Clamp`] (historical
    /// behaviour).
    pub tone_map: ToneMap,
    /// Linear exposure multiplier applied before tone mapping. Must be
    /// finite and `>= 0`. Default `1.0`.
    pub exposure: f32,
    /// Scene time in seconds at which animations are sampled. `None`
    /// (default) renders the rest pose (static node transforms, static
    /// morph weights). Skinning is applied either way.
    pub time: Option<f32>,
    /// Index into `Scene3D::animations` sampled at [`Self::time`].
    /// `None` ⇒ the first animation. Ignored when `time` is `None`.
    pub animation: Option<usize>,
    /// Render through a scene camera: an index into the scene's camera
    /// *instances* (nodes carrying a camera, in scene-graph pre-order —
    /// see [`crate::prepare::PreparedScene::cameras`]). `None`
    /// (default), or an out-of-range index, keeps the auto-frame /
    /// [`Self::camera`] orbit behaviour.
    pub scene_camera: Option<usize>,
    /// Use the scene's `KHR_lights_punctual` lights when it has any
    /// (default `true`). When `false`, or when the scene has no
    /// lights, [`Self::light`] is the single directional light.
    pub use_scene_lights: bool,
    /// Uniform ambient (environment) radiance used by
    /// [`ShadingMode::Pbr`], scaled by material occlusion. Must be
    /// finite and `>= 0`. Default `0.2`. The legacy Gouraud / Phong
    /// modes keep their fixed 0.2 ambient term.
    pub ambient: f32,
    /// Shadows in [`ShadingMode::Pbr`]: shadow maps for directional and
    /// spot lights on `Scanline` (Williams 1978, filtered with PCF);
    /// ray-traced hard shadows for every light type on `Raycast`.
    /// Default `false`.
    pub shadows: bool,
    /// Shadow-map resolution (texels per side), `16..=8192`. Default
    /// `1024`.
    pub shadow_map_size: u32,
    /// Active `KHR_materials_variants` variant (index into
    /// `Scene3D::material_variants`). `None` (default) uses each
    /// primitive's base material.
    pub material_variant: Option<usize>,
    /// Ray backends: maximum number of reflection / refraction bounces
    /// per camera ray (`0` = no secondary rays; `BLEND` see-through
    /// continuations are not counted). Clamped to `0..=16`. Default
    /// `4`.
    pub max_ray_depth: u32,
    /// `Raycast` [`ShadingMode::Pbr`]: perceptual roughness at or above
    /// which no mirror-reflection ray is traced (the environment's
    /// uniform ambient stands in for the glossy lobe). Below it the
    /// traced reflection fades in as `(1 − roughness / cutoff)²`. `0`
    /// disables reflection rays. In `[0, 1]`; default `0.5`.
    pub reflection_roughness_cutoff: f32,
    /// Path-tracer controls (sample count, bounces, clamp, seed);
    /// ignored by the other backends.
    pub path_trace: PathTraceOptions,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            background: BackgroundColor::default(),
            shading: ShadingMode::default(),
            projection: Projection::default(),
            fov_deg: 60.0,
            light: LightSpec::default_light(),
            camera: None,
            camera_target_offset: [0.0; 3],
            aa: 1,
            tone_map: ToneMap::Clamp,
            exposure: 1.0,
            time: None,
            animation: None,
            scene_camera: None,
            use_scene_lights: true,
            ambient: 0.2,
            shadows: false,
            shadow_map_size: 1024,
            material_variant: None,
            max_ray_depth: 4,
            reflection_roughness_cutoff: 0.5,
            path_trace: PathTraceOptions::default(),
        }
    }
}

impl RenderOptions {
    /// Validate the field values against the renderer contract and
    /// return a descriptive [`Error::InvalidOptions`] for the first
    /// offending field. `Ok(())` means every backend can consume the
    /// options without an immediate sanity-clamp.
    ///
    /// Constraints enforced (matching the scanline backend's own
    /// expectations, kept identical for the future raycast / path-trace
    /// backends so a single `validate` covers all three):
    ///
    /// * `width` and `height` are `>= 1`.
    /// * `fov_deg` is finite and strictly within `(0, 180)` — only
    ///   meaningful in perspective mode but checked unconditionally so
    ///   a stray NaN doesn't slip through a later mode flip.
    /// * `aa` is within `1..=8` (the scanline backend's documented
    ///   range; clamps silently above that today, but a typed validate
    ///   reflects intent).
    /// * `light.intensity` is finite and `>= 0.0`.
    /// * `light.azimuth_deg` and `light.elevation_deg` are finite.
    /// * If `camera` is `Some`, every field is finite and `distance`
    ///   is `> 0`.
    /// * `camera_target_offset` is finite.
    /// * `exposure` and `ambient` are finite and `>= 0.0`; `time` (if
    ///   set) is finite; `shadow_map_size` is within `16..=8192`;
    ///   `reflection_roughness_cutoff` is within `[0, 1]`.
    /// * `path_trace.samples_per_pixel` is within `1..=1048576`,
    ///   `path_trace.max_bounces <= 1024`, `path_trace.clamp` finite
    ///   and `>= 0`.
    ///
    /// `validate` is **not** called automatically by [`crate::Renderer::render`]
    /// — backends today silently clamp instead — so a caller that wants
    /// strict failure on bad input opts in by calling this method
    /// before `render`. `oxideav-pipeline`'s `Render3D` DAG node is the
    /// expected first consumer.
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 {
            return Err(Error::InvalidOptions(format!(
                "width must be >= 1, got {}",
                self.width
            )));
        }
        if self.height == 0 {
            return Err(Error::InvalidOptions(format!(
                "height must be >= 1, got {}",
                self.height
            )));
        }
        if !self.fov_deg.is_finite() {
            return Err(Error::InvalidOptions(format!(
                "fov_deg must be finite, got {}",
                self.fov_deg
            )));
        }
        if !(0.0 < self.fov_deg && self.fov_deg < 180.0) {
            return Err(Error::InvalidOptions(format!(
                "fov_deg must be in (0, 180), got {}",
                self.fov_deg
            )));
        }
        if !(1..=8).contains(&self.aa) {
            return Err(Error::InvalidOptions(format!(
                "aa must be in 1..=8, got {}",
                self.aa
            )));
        }
        if !self.light.intensity.is_finite() || self.light.intensity < 0.0 {
            return Err(Error::InvalidOptions(format!(
                "light.intensity must be finite and >= 0.0, got {}",
                self.light.intensity
            )));
        }
        if !self.light.azimuth_deg.is_finite() || !self.light.elevation_deg.is_finite() {
            return Err(Error::InvalidOptions(format!(
                "light.azimuth_deg / light.elevation_deg must be finite, got ({}, {})",
                self.light.azimuth_deg, self.light.elevation_deg
            )));
        }
        if !self.camera_target_offset.iter().all(|v| v.is_finite()) {
            return Err(Error::InvalidOptions(format!(
                "camera_target_offset must be finite, got {:?}",
                self.camera_target_offset
            )));
        }
        if !self.exposure.is_finite() || self.exposure < 0.0 {
            return Err(Error::InvalidOptions(format!(
                "exposure must be finite and >= 0.0, got {}",
                self.exposure
            )));
        }
        if !self.ambient.is_finite() || self.ambient < 0.0 {
            return Err(Error::InvalidOptions(format!(
                "ambient must be finite and >= 0.0, got {}",
                self.ambient
            )));
        }
        if let Some(t) = self.time {
            if !t.is_finite() {
                return Err(Error::InvalidOptions(format!(
                    "time must be finite, got {t}"
                )));
            }
        }
        if !(0.0..=1.0).contains(&self.reflection_roughness_cutoff) {
            return Err(Error::InvalidOptions(format!(
                "reflection_roughness_cutoff must be in [0, 1], got {}",
                self.reflection_roughness_cutoff
            )));
        }
        if !(16..=8192).contains(&self.shadow_map_size) {
            return Err(Error::InvalidOptions(format!(
                "shadow_map_size must be in 16..=8192, got {}",
                self.shadow_map_size
            )));
        }
        let pt = &self.path_trace;
        if !(1..=1 << 20).contains(&pt.samples_per_pixel) {
            return Err(Error::InvalidOptions(format!(
                "path_trace.samples_per_pixel must be in 1..=1048576, got {}",
                pt.samples_per_pixel
            )));
        }
        if pt.max_bounces > 1024 {
            return Err(Error::InvalidOptions(format!(
                "path_trace.max_bounces must be <= 1024, got {}",
                pt.max_bounces
            )));
        }
        if !pt.clamp.is_finite() || pt.clamp < 0.0 {
            return Err(Error::InvalidOptions(format!(
                "path_trace.clamp must be finite and >= 0.0, got {}",
                pt.clamp
            )));
        }
        if let Some(cam) = self.camera {
            if !cam.azimuth_deg.is_finite() || !cam.elevation_deg.is_finite() {
                return Err(Error::InvalidOptions(format!(
                    "camera.azimuth_deg / camera.elevation_deg must be finite, got ({}, {})",
                    cam.azimuth_deg, cam.elevation_deg
                )));
            }
            if !cam.distance.is_finite() || cam.distance <= 0.0 {
                return Err(Error::InvalidOptions(format!(
                    "camera.distance must be finite and > 0, got {}",
                    cam.distance
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_stable() {
        let opts = RenderOptions::default();
        assert_eq!(opts.width, 512);
        assert_eq!(opts.height, 512);
        assert_eq!(opts.background, BackgroundColor([0, 0, 0, 0]));
        assert_eq!(opts.shading, ShadingMode::Phong);
        assert_eq!(opts.projection, Projection::Perspective);
        assert!((opts.fov_deg - 60.0).abs() < 1e-6);
        assert_eq!(opts.aa, 1);
        assert!(opts.camera.is_none());
        assert!((opts.light.azimuth_deg - 45.0).abs() < 1e-6);
        assert!((opts.light.elevation_deg - 45.0).abs() < 1e-6);
        assert!((opts.light.intensity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn backend_enum_has_all_three_backends() {
        // Compile-time pin: every live backend stays constructible and
        // distinct.
        assert_ne!(RenderBackend::Scanline, RenderBackend::Raycast);
        assert_ne!(RenderBackend::Raycast, RenderBackend::PathTrace);
        for backend in [
            RenderBackend::Scanline,
            RenderBackend::Raycast,
            RenderBackend::PathTrace,
        ] {
            assert!(crate::make_renderer(backend).is_ok(), "{backend:?}");
        }
    }

    #[test]
    fn validate_rejects_bad_hdr_and_animation_fields() {
        for opts in [
            RenderOptions {
                exposure: -1.0,
                ..RenderOptions::default()
            },
            RenderOptions {
                ambient: f32::NAN,
                ..RenderOptions::default()
            },
            RenderOptions {
                time: Some(f32::INFINITY),
                ..RenderOptions::default()
            },
            RenderOptions {
                shadow_map_size: 4,
                ..RenderOptions::default()
            },
            RenderOptions {
                reflection_roughness_cutoff: f32::NAN,
                ..RenderOptions::default()
            },
            RenderOptions {
                path_trace: PathTraceOptions {
                    samples_per_pixel: 0,
                    ..PathTraceOptions::default()
                },
                ..RenderOptions::default()
            },
            RenderOptions {
                path_trace: PathTraceOptions {
                    clamp: -1.0,
                    ..PathTraceOptions::default()
                },
                ..RenderOptions::default()
            },
        ] {
            assert!(matches!(opts.validate(), Err(Error::InvalidOptions(_))));
        }
    }

    #[test]
    fn validate_accepts_default_options() {
        assert!(RenderOptions::default().validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_width_or_height() {
        let opts = RenderOptions {
            width: 0,
            ..RenderOptions::default()
        };
        let msg = match opts.validate() {
            Err(Error::InvalidOptions(s)) => s,
            other => panic!("expected InvalidOptions, got {other:?}"),
        };
        assert!(msg.contains("width"), "msg should mention width: {msg}");

        let opts = RenderOptions {
            height: 0,
            ..RenderOptions::default()
        };
        let msg = match opts.validate() {
            Err(Error::InvalidOptions(s)) => s,
            other => panic!("expected InvalidOptions, got {other:?}"),
        };
        assert!(msg.contains("height"), "msg should mention height: {msg}");
    }

    #[test]
    fn validate_rejects_out_of_range_fov() {
        for bad in [0.0_f32, -1.0, 180.0, 360.0, f32::NAN, f32::INFINITY] {
            let opts = RenderOptions {
                fov_deg: bad,
                ..RenderOptions::default()
            };
            assert!(
                matches!(opts.validate(), Err(Error::InvalidOptions(_))),
                "fov_deg = {bad} should be rejected"
            );
        }
    }

    #[test]
    fn validate_rejects_out_of_range_aa() {
        for bad in [0_u32, 9, u32::MAX] {
            let opts = RenderOptions {
                aa: bad,
                ..RenderOptions::default()
            };
            assert!(
                matches!(opts.validate(), Err(Error::InvalidOptions(_))),
                "aa = {bad} should be rejected"
            );
        }
        // In-range still passes.
        for ok in [1_u32, 4, 8] {
            let opts = RenderOptions {
                aa: ok,
                ..RenderOptions::default()
            };
            assert!(opts.validate().is_ok(), "aa = {ok} should pass");
        }
    }

    #[test]
    fn validate_rejects_negative_or_non_finite_light() {
        let opts = RenderOptions {
            light: LightSpec {
                intensity: -0.1,
                ..LightSpec::default_light()
            },
            ..RenderOptions::default()
        };
        assert!(matches!(opts.validate(), Err(Error::InvalidOptions(_))));

        let opts = RenderOptions {
            light: LightSpec {
                intensity: f32::NAN,
                ..LightSpec::default_light()
            },
            ..RenderOptions::default()
        };
        assert!(matches!(opts.validate(), Err(Error::InvalidOptions(_))));

        let opts = RenderOptions {
            light: LightSpec {
                azimuth_deg: f32::NAN,
                ..LightSpec::default_light()
            },
            ..RenderOptions::default()
        };
        assert!(matches!(opts.validate(), Err(Error::InvalidOptions(_))));
    }

    #[test]
    fn validate_rejects_bad_camera_distance() {
        let opts = RenderOptions {
            camera: Some(CameraSpec {
                elevation_deg: 30.0,
                azimuth_deg: 45.0,
                distance: 0.0,
            }),
            ..RenderOptions::default()
        };
        assert!(matches!(opts.validate(), Err(Error::InvalidOptions(_))));

        let opts = RenderOptions {
            camera: Some(CameraSpec {
                elevation_deg: 30.0,
                azimuth_deg: 45.0,
                distance: -1.0,
            }),
            ..RenderOptions::default()
        };
        assert!(matches!(opts.validate(), Err(Error::InvalidOptions(_))));

        let opts = RenderOptions {
            camera: Some(CameraSpec {
                elevation_deg: 30.0,
                azimuth_deg: 45.0,
                distance: f32::INFINITY,
            }),
            ..RenderOptions::default()
        };
        assert!(matches!(opts.validate(), Err(Error::InvalidOptions(_))));
    }

    #[test]
    fn validate_accepts_good_camera_override() {
        let opts = RenderOptions {
            camera: Some(CameraSpec {
                elevation_deg: 30.0,
                azimuth_deg: 45.0,
                distance: 1.5,
            }),
            ..RenderOptions::default()
        };
        assert!(opts.validate().is_ok());
    }
}
