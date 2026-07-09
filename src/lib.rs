//! # oxideav-render
//!
//! **Status:** Phase B — scanline backend live (2026-06-07).
//!
//! Pure-Rust 3D-scene → raster image/video renderer for the
//! [oxideav](https://github.com/OxideAV/oxideav-workspace) framework.
//! Phase A shipped the [`Renderer`] trait, [`RenderBackend`] selector,
//! [`RenderOptions`] surface, [`RgbaImage`] output type, and the
//! [`make_renderer`] factory contract. Phase B fills `Scanline` in
//! with a half-space-edge-function rasteriser supporting Flat /
//! Gouraud / Phong / Wireframe shading plus NormalDebug and DepthDebug
//! visualisations, SSAA (1..=8), perspective and orthographic
//! projection, auto-frame or orbit camera, and a single directional
//! light with ambient. Raycast and path-tracer backends land in Phases
//! D and E.
//!
//! ## Roadmap
//!
//! | Phase | Surface added                                                                |
//! |-------|-------------------------------------------------------------------------------|
//! | A     | `Renderer` trait + `RenderBackend::Scanline` + `make_renderer` stub.          |
//! | B *(now)* | Scanline backend (Gouraud / Phong / Wireframe / Flat / NormalDebug / Depth). |
//! | C     | `oxideav-pipeline` `DagNode::Render3D` source — emits `Frame::Video`.         |
//! | D     | `RenderBackend::Raycast` — Whitted primary + shadow + reflection / refraction.|
//! | E     | `RenderBackend::PathTrace` — Kajiya path tracing + Disney/Burley BRDF.        |
//!
//! ## Quick start
//!
//! ```no_run
//! use oxideav_render::{make_renderer, RenderBackend, RenderOptions, Result};
//! use oxideav_mesh3d::Scene3D;
//!
//! fn render_one(scene: &Scene3D) -> Result<()> {
//!     let mut renderer = make_renderer(RenderBackend::Scanline)?;
//!     let opts = RenderOptions {
//!         width: 1024,
//!         height: 768,
//!         ..Default::default()
//!     };
//!     let _image = renderer.render(scene, &opts)?;
//!     Ok(())
//! }
//! ```
//!
//! ## Clean-room policy
//!
//! Render math is sourced from published academic papers — Pineda 1988
//! (half-space rasterisation), Bresenham 1965 (line walker), Möller–
//! Trumbore 1997 (ray-triangle intersection, Phase D), Burley 2012
//! SIGGRAPH course (Disney BRDF, Phase E), Kajiya 1986 (path-tracing
//! equation, Phase E) — plus IEC 61966-2-1 for sRGB encoding.
//! Reference renderer source code (PBRT, Cycles, EEVEE, Embree, OptiX,
//! Mitsuba, *.blend* files) is **not** consulted. glTF KHR extensions
//! provide the material vocabulary; output crosswalk to PNG / OpenEXR
//! / video uses the existing oxideav encoder crates.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod camera;
pub mod error;
pub mod image;
mod math;
pub mod options;
pub mod registry;
mod scanline;
mod shade;

#[cfg(feature = "registry")]
pub mod source;

pub use error::{Error, Result};
pub use image::RgbaImage;
pub use options::{
    BackgroundColor, CameraSpec, LightSpec, Projection, RenderBackend, RenderOptions, ShadingMode,
};
pub use registry::{register_into, RenderRegistry, RendererFactory};

#[cfg(feature = "registry")]
pub use source::RenderSource;

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
/// Phase B routes `Scanline` to the in-tree scanline backend.
/// Phases D and E fill in `Raycast` and `PathTrace`.
pub fn make_renderer(backend: RenderBackend) -> Result<Box<dyn Renderer>> {
    match backend {
        RenderBackend::Scanline => Ok(Box::new(ScanlineRenderer::new())),
    }
}

/// Scanline rasteriser — half-space edge-function pipeline with a
/// per-pixel z-buffer. Cheap, pure-Rust, no global illumination, no
/// raytraced shadows. See [`scanline`] for the algorithmic detail.
///
/// Constructed via [`make_renderer`] (recommended) or directly when
/// the caller wants to keep the renderer around across many frames.
#[derive(Debug, Default)]
pub struct ScanlineRenderer {
    _priv: (),
}

impl ScanlineRenderer {
    /// Construct a fresh scanline renderer. State-free today — the
    /// rasteriser allocates a per-frame framebuffer + z-buffer inside
    /// `render`, so multiple `render` calls don't share memory.
    pub fn new() -> Self {
        Self { _priv: () }
    }
}

impl Renderer for ScanlineRenderer {
    fn render(
        &mut self,
        scene: &oxideav_mesh3d::Scene3D,
        opts: &RenderOptions,
    ) -> Result<RgbaImage> {
        Ok(scanline::render_scene(scene, opts))
    }
}

/// Crate identifier used by `oxideav-meta`'s `register_all`
/// enumeration and by future `RenderRegistry` lookups.
pub const CRATE_NAME: &str = "oxideav-render";

/// `oxideav-core` framework hook.
///
/// Phase A shipped this as a stable no-op so the meta crate's
/// `build.rs` could auto-discover this crate and bake it into
/// `register_all`. Phase B keeps the no-op (the scanline backend
/// doesn't need a process-wide registry — it's instantiated direct
/// via [`make_renderer`]). Phase C wires real registration of every
/// backend by name into a `RenderRegistry` so `oxideav-pipeline` can
/// look up backends from the JSON job graph.
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
    fn make_renderer_scanline_returns_working_backend_in_phase_b() {
        let renderer = make_renderer(RenderBackend::Scanline);
        assert!(renderer.is_ok(), "Phase B must hand back a real renderer");
    }

    #[test]
    fn render_options_default_is_512x512_phong_perspective() {
        let opts = RenderOptions::default();
        assert_eq!((opts.width, opts.height), (512, 512));
        assert_eq!(opts.shading, ShadingMode::Phong);
        assert_eq!(opts.projection, Projection::Perspective);
    }

    /// End-to-end smoke through the trait: construct a one-triangle
    /// scene, route through `make_renderer(Scanline).render`, verify
    /// at least one pixel changed off the background. This is the
    /// per-crate equivalent of the cli-convert mesh3d_convert test.
    #[test]
    fn scanline_renderer_via_trait_paints_some_pixels() {
        use oxideav_mesh3d::{Mesh, MeshId, Node, Primitive, Scene3D, Topology};

        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
        let mesh = Mesh::new("triangle".to_string()).with_primitive(prim);
        let mut scene = Scene3D::new();
        scene.meshes.push(mesh);
        scene.nodes.push(Node {
            mesh: Some(MeshId(0)),
            ..Node::default()
        });
        scene.roots.push(oxideav_mesh3d::NodeId(0));

        let mut renderer = make_renderer(RenderBackend::Scanline).expect("Scanline must construct");
        let opts = RenderOptions {
            width: 32,
            height: 32,
            background: BackgroundColor([255, 255, 255, 255]),
            shading: ShadingMode::Flat,
            ..RenderOptions::default()
        };
        let img = renderer.render(&scene, &opts).expect("render must succeed");
        assert_eq!(img.width, 32);
        assert_eq!(img.height, 32);
        let any_drawn = img
            .pixels
            .chunks_exact(4)
            .any(|p| p != [255, 255, 255, 255]);
        assert!(any_drawn, "scanline backend must paint at least one pixel");
    }
}
