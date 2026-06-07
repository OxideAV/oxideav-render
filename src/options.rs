//! [`RenderOptions`] + companion enums — the surface that every
//! [`crate::Renderer`] consumes.

/// Backend selector used by [`crate::make_renderer`].
///
/// Phase A shipped the selector with no working backend. Phase B
/// fills in `Scanline`. Phase D adds `Raycast` (Whitted-style primary +
/// shadow + reflection / refraction). Phase E adds `PathTrace` (Kajiya
/// unbiased path tracing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RenderBackend {
    /// Scanline rasteriser — Gouraud / Phong / Wireframe / Flat /
    /// NormalDebug / DepthDebug. Half-space edge-function pipeline
    /// with a per-pixel z-buffer. Fast, no global illumination, no
    /// raytraced shadows.
    Scanline,
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
    fn backend_enum_phase_b_only_has_scanline() {
        // Compile-time pin — adding a variant before Phase D should fail
        // a CI test that round-trips every variant. Phase D removes
        // this restriction.
        match RenderBackend::Scanline {
            RenderBackend::Scanline => {}
        }
    }
}
