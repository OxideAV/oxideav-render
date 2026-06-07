//! [`RenderOptions`] + companion enums — the surface that every
//! [`crate::Renderer`] consumes.

/// Backend selector used by [`crate::make_renderer`].
///
/// Phase A ships `Scanline` only. Phase D adds `Raycast` (Whitted-style
/// primary + shadow + reflection / refraction). Phase E adds
/// `PathTrace` (Kajiya unbiased path tracing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RenderBackend {
    /// Scanline rasteriser — Gouraud / Phong / Wireframe / Flat /
    /// NormalDebug / DepthDebug. Fast, no global illumination, no
    /// raytraced shadows. Migrated from `oxideav-cli-convert` in
    /// Phase B.
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
    /// Constant material colour per triangle.
    Flat,
    /// Per-vertex lighting interpolated across the triangle.
    Gouraud,
    /// Per-pixel lighting (normal interpolation + per-pixel lit).
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
    /// Orthographic projection — useful for engineering / isometric
    /// renders.
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
    /// Vertical field-of-view in degrees (perspective only).
    pub fov_deg: f32,
    /// Supersampling factor `[1, 8]`. `1` = off; higher values render
    /// `N×width × N×height` and box-filter down.
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
    }

    #[test]
    fn backend_enum_phase_a_only_has_scanline() {
        // Compile-time pin — adding a variant before Phase D should fail
        // a CI test that round-trips every variant. Phase D removes
        // this restriction.
        match RenderBackend::Scanline {
            RenderBackend::Scanline => {}
        }
    }
}
