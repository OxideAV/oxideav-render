//! # oxideav-render
//!
//! **Status:** Phase A scaffold (2026-06-07).
//!
//! Pure-Rust 3D-scene → raster image/video renderer for the
//! [oxideav](https://github.com/OxideAV/oxideav-workspace) framework.
//! Phase A ships the [`Renderer`] trait, [`RenderBackend`] selector,
//! [`RenderOptions`] surface, [`RgbaImage`] output type, and the
//! [`make_renderer`] factory contract. [`make_renderer`] currently
//! returns [`Error::NotImplemented`] for every backend — the scanline
//! backend lands in Phase B (migrated verbatim from
//! `oxideav-cli-convert`); the raycast and path-tracer backends land
//! in Phases D and E.
//!
//! ## Roadmap
//!
//! | Phase | Surface added                                                                |
//! |-------|-------------------------------------------------------------------------------|
//! | A     | `Renderer` trait + `RenderBackend::Scanline` + `make_renderer` stub.          |
//! | B     | Scanline backend (Gouraud / Phong / Wireframe / Flat / NormalDebug / Depth).  |
//! | C     | `oxideav-pipeline` `DagNode::Render3D` source — emits `Frame::Video`.         |
//! | D     | `RenderBackend::Raycast` — Whitted primary + shadow + reflection / refraction.|
//! | E     | `RenderBackend::PathTrace` — Kajiya path tracing + Disney/Burley BRDF.        |
//!
//! ## Clean-room policy
//!
//! Render math is sourced from published academic papers (Möller–
//! Trumbore 1997 for ray-triangle intersection, Burley 2012 SIGGRAPH
//! course for the Disney BRDF, Kajiya 1986 for the path-tracing
//! equation). Reference renderer source code (PBRT, Cycles, EEVEE,
//! Embree, OptiX, Mitsuba, *.blend* files) is **not** consulted. glTF
//! KHR extensions provide the material vocabulary; output crosswalk
//! to PNG / OpenEXR / video uses the existing oxideav encoder crates.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod error;
pub mod image;
pub mod options;

pub use error::{Error, Result};
pub use image::RgbaImage;
pub use options::{BackgroundColor, Projection, RenderBackend, RenderOptions, ShadingMode};

/// Renderer trait — the surface every backend (Scanline / Raycast /
/// PathTrace) implements.
///
/// A renderer consumes an [`oxideav_mesh3d::Scene3D`] plus
/// [`RenderOptions`] and produces an [`RgbaImage`]. The trait is
/// object-safe (`dyn Renderer`) so that [`make_renderer`] can return
/// a `Box<dyn Renderer>` and callers don't need to know the backend
/// type statically.
///
/// Animation rendering is deferred to Phase C — when
/// `oxideav-pipeline` integrates the renderer as a `FrameSource`,
/// the pipeline drives per-frame scene time advancement and calls
/// `render` once per frame.
pub trait Renderer: Send {
    /// Render `scene` to an [`RgbaImage`] under `opts`.
    fn render(
        &mut self,
        scene: &oxideav_mesh3d::Scene3D,
        opts: &RenderOptions,
    ) -> Result<RgbaImage>;
}

/// Construct a renderer for `backend`.
///
/// Phase A returns [`Error::NotImplemented`] for every backend.
/// Phase B fills in `Scanline`. Phases D and E fill in `Raycast` and
/// `PathTrace`.
pub fn make_renderer(backend: RenderBackend) -> Result<Box<dyn Renderer>> {
    match backend {
        RenderBackend::Scanline => Err(Error::NotImplemented),
    }
}

/// Crate identifier used by `oxideav-meta`'s `register_all`
/// enumeration and by future `RenderRegistry` lookups.
pub const CRATE_NAME: &str = "oxideav-render";

/// `oxideav-core` framework hook.
///
/// Phase A is a stable no-op so the meta crate's `build.rs` can
/// auto-discover this crate and bake it into `register_all`. Phase B
/// keeps the no-op (the scanline backend doesn't need a process-wide
/// registry — it's instantiated direct via [`make_renderer`]). Phase
/// C wires real registration of every backend by name into a
/// `RenderRegistry` so `oxideav-pipeline` can look up backends from
/// the JSON job graph.
#[cfg(feature = "registry")]
pub fn register(_ctx: &mut oxideav_core::RuntimeContext) {}

#[cfg(feature = "registry")]
oxideav_core::register!("render", register);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_name_constant_is_stable() {
        assert_eq!(CRATE_NAME, "oxideav-render");
    }

    #[test]
    fn make_renderer_scanline_returns_not_implemented_in_phase_a() {
        let result = make_renderer(RenderBackend::Scanline);
        assert!(matches!(result, Err(Error::NotImplemented)));
    }

    #[test]
    fn render_options_default_is_512x512_phong_perspective() {
        let opts = RenderOptions::default();
        assert_eq!((opts.width, opts.height), (512, 512));
        assert_eq!(opts.shading, ShadingMode::Phong);
        assert_eq!(opts.projection, Projection::Perspective);
    }
}
