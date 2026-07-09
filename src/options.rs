//! [`RenderOptions`] + companion enums — the surface that every
//! [`crate::Renderer`] consumes.

use crate::error::{Error, Result};

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
    /// `Scanline`, plus (in `Phong` mode) raytraced hard shadows and
    /// recursive reflection / refraction driven by material
    /// metallic / roughness / transmission / IOR. Line and point
    /// topologies have no surface area and are invisible to rays.
    Raycast,
}

/// Shading model selector consumed by the scanline backend.
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
    pub camera: Option<CameraSpec>,
    /// Supersampling factor `[1, 8]`. `1` = off; higher values render
    /// `N×width × N×height` and box-filter down to the requested
    /// output. Hard cap of `8` because at 8× a 1024² render is a
    /// 16 M-pixel framebuffer + an 8 M f32 z-buffer (~80 MB).
    pub aa: u32,
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
            aa: 1,
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
    fn backend_enum_phase_d_has_scanline_and_raycast() {
        // Compile-time pin: both live backends stay constructible and
        // distinct. Phase E extends this with `PathTrace`.
        assert_ne!(RenderBackend::Scanline, RenderBackend::Raycast);
        for backend in [RenderBackend::Scanline, RenderBackend::Raycast] {
            assert!(crate::make_renderer(backend).is_ok(), "{backend:?}");
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
