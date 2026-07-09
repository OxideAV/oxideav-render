//! # oxideav-render
//!
//! **Status:** Phase D — scanline + Whitted raycast backends live.
//!
//! Pure-Rust 3D-scene → raster image/video renderer for the
//! [oxideav](https://github.com/OxideAV/oxideav-workspace) framework.
//! Phase A shipped the [`Renderer`] trait, [`RenderBackend`] selector,
//! [`RenderOptions`] surface, [`RgbaImage`] output type, and the
//! [`make_renderer`] factory contract. Phase B filled `Scanline` in
//! with a half-space-edge-function rasteriser supporting Flat /
//! Gouraud / Phong / Wireframe shading plus NormalDebug and DepthDebug
//! visualisations, SSAA (1..=8), perspective and orthographic
//! projection, auto-frame or orbit camera, and a single directional
//! light with ambient. Phase C bridged the renderer into
//! `oxideav-pipeline` via `RenderSource`. Phase D fills `Raycast`
//! in with a BVH-accelerated Whitted recursive ray tracer covering
//! the same option surface plus raytraced hard shadows and recursive
//! reflection / refraction. The path-tracer backend lands in Phase E.
//!
//! ## Roadmap
//!
//! | Phase | Surface added                                                                |
//! |-------|-------------------------------------------------------------------------------|
//! | A     | `Renderer` trait + `RenderBackend::Scanline` + `make_renderer` stub.          |
//! | B     | Scanline backend (Gouraud / Phong / Wireframe / Flat / NormalDebug / Depth).  |
//! | C     | `oxideav-pipeline` `DagNode::Render3D` source — emits `Frame::Video`.         |
//! | D *(now)* | `RenderBackend::Raycast` — Whitted primary + shadow + reflection / refraction.|
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
mod raycast;
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
/// Phase B routes `Scanline` to the in-tree scanline backend; Phase D
/// routes `Raycast` to the Whitted ray tracer. Phase E fills in
/// `PathTrace`.
pub fn make_renderer(backend: RenderBackend) -> Result<Box<dyn Renderer>> {
    match backend {
        RenderBackend::Scanline => Ok(Box::new(ScanlineRenderer::new())),
        RenderBackend::Raycast => Ok(Box::new(RaycastRenderer::new())),
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

/// Whitted recursive ray tracer — closest-hit shading through a
/// BVH-accelerated world-space triangle soup, with raytraced hard
/// shadows and recursive reflection / refraction in
/// [`ShadingMode::Phong`]. See [`raycast`](crate::options::RenderBackend::Raycast)
/// for the capability envelope.
///
/// Constructed via [`make_renderer`] (recommended) or directly. The
/// scene is baked (flattened + BVH-built) once per `render` call, so
/// re-rendering the same scene re-bakes; per-frame scene mutation is
/// therefore free of stale-acceleration hazards.
#[derive(Debug, Default)]
pub struct RaycastRenderer {
    _priv: (),
}

impl RaycastRenderer {
    /// Construct a fresh raycast renderer (state-free — the trace
    /// scene and framebuffer are per-`render` allocations).
    pub fn new() -> Self {
        Self { _priv: () }
    }
}

impl Renderer for RaycastRenderer {
    fn render(
        &mut self,
        scene: &oxideav_mesh3d::Scene3D,
        opts: &RenderOptions,
    ) -> Result<RgbaImage> {
        Ok(raycast::render_scene(scene, opts))
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

#[cfg(test)]
mod robustness_tests {
    //! Degenerate / hostile scene-graph inputs, exercised through the
    //! public trait surface on BOTH live backends. Every case here
    //! must terminate and return a well-formed image — no panic, no
    //! unbounded recursion.

    use super::*;
    use oxideav_mesh3d::{Indices, Mesh, MeshId, Node, NodeId, Primitive, Scene3D, Topology};

    fn both_backends() -> [RenderBackend; 2] {
        [RenderBackend::Scanline, RenderBackend::Raycast]
    }

    fn tiny_opts() -> RenderOptions {
        RenderOptions {
            width: 16,
            height: 16,
            background: BackgroundColor([9, 9, 9, 255]),
            shading: ShadingMode::Flat,
            ..RenderOptions::default()
        }
    }

    fn triangle_mesh() -> Mesh {
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
        Mesh::new("t".to_string()).with_primitive(prim)
    }

    fn render_ok(scene: &Scene3D) {
        for backend in both_backends() {
            let mut renderer = make_renderer(backend).expect("construct");
            let img = renderer
                .render(scene, &tiny_opts())
                .expect("render must not fail");
            assert_eq!((img.width, img.height), (16, 16), "{backend:?}");
            assert_eq!(img.pixels.len(), 16 * 16 * 4, "{backend:?}");
        }
    }

    #[test]
    fn self_referential_node_terminates() {
        // Node 0 lists itself as a child — an unguarded pre-order
        // walk recurses forever.
        let mut scene = Scene3D::new();
        scene.meshes.push(triangle_mesh());
        scene.nodes.push(Node {
            mesh: Some(MeshId(0)),
            children: vec![NodeId(0)],
            ..Node::default()
        });
        scene.roots.push(NodeId(0));
        render_ok(&scene);
    }

    #[test]
    fn two_node_cycle_terminates() {
        let mut scene = Scene3D::new();
        scene.meshes.push(triangle_mesh());
        scene.nodes.push(Node {
            mesh: Some(MeshId(0)),
            children: vec![NodeId(1)],
            ..Node::default()
        });
        scene.nodes.push(Node {
            children: vec![NodeId(0)],
            ..Node::default()
        });
        scene.roots.push(NodeId(0));
        render_ok(&scene);
    }

    #[test]
    fn diamond_shared_child_renders_once() {
        // Two parents share one mesh-bearing child. Traversal contract:
        // first parent's chain claims the child; the render still
        // paints the triangle exactly as a plain single-parent scene
        // would (identity transforms on every node).
        let mut scene = Scene3D::new();
        scene.meshes.push(triangle_mesh());
        scene.nodes.push(Node {
            children: vec![NodeId(2)],
            ..Node::default()
        });
        scene.nodes.push(Node {
            children: vec![NodeId(2)],
            ..Node::default()
        });
        scene.nodes.push(Node {
            mesh: Some(MeshId(0)),
            ..Node::default()
        });
        scene.roots.push(NodeId(0));
        scene.roots.push(NodeId(1));

        let mut plain = Scene3D::new();
        plain.meshes.push(triangle_mesh());
        plain.nodes.push(Node {
            mesh: Some(MeshId(0)),
            ..Node::default()
        });
        plain.roots.push(NodeId(0));

        for backend in both_backends() {
            let mut renderer = make_renderer(backend).expect("construct");
            let diamond_img = renderer.render(&scene, &tiny_opts()).expect("render");
            let plain_img = renderer.render(&plain, &tiny_opts()).expect("render");
            assert_eq!(
                diamond_img.pixels, plain_img.pixels,
                "{backend:?}: shared child must render once, identically to the plain scene"
            );
        }
    }

    #[test]
    fn out_of_range_node_and_mesh_ids_are_ignored() {
        let mut scene = Scene3D::new();
        scene.meshes.push(triangle_mesh());
        scene.nodes.push(Node {
            mesh: Some(MeshId(999)),
            children: vec![NodeId(999)],
            ..Node::default()
        });
        scene.roots.push(NodeId(0));
        scene.roots.push(NodeId(12345));
        render_ok(&scene);
    }

    #[test]
    fn nan_positions_do_not_poison_the_render() {
        // One healthy triangle + one NaN-poisoned triangle. The
        // camera auto-frame must ignore the NaN vertices and both
        // backends must still terminate cleanly.
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.0, 0.5, 0.0],
            [f32::NAN, 0.0, 0.0],
            [0.0, f32::NAN, 0.0],
            [f32::INFINITY, 0.0, f32::NEG_INFINITY],
        ];
        let mut scene = Scene3D::new();
        scene
            .meshes
            .push(Mesh::new("n".to_string()).with_primitive(prim));
        scene.nodes.push(Node {
            mesh: Some(MeshId(0)),
            ..Node::default()
        });
        scene.roots.push(NodeId(0));

        for backend in both_backends() {
            let mut renderer = make_renderer(backend).expect("construct");
            let img = renderer.render(&scene, &tiny_opts()).expect("render");
            let painted = img.pixels.chunks_exact(4).any(|p| p != [9, 9, 9, 255]);
            assert!(
                painted,
                "{backend:?}: the healthy triangle must survive NaN siblings"
            );
        }
    }

    #[test]
    fn garbage_index_buffer_is_ignored() {
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
        prim.indices = Some(Indices::U32(vec![0, 1, 2, 7, 8, 9, u32::MAX, 1, 2]));
        let mut scene = Scene3D::new();
        scene
            .meshes
            .push(Mesh::new("g".to_string()).with_primitive(prim));
        scene.nodes.push(Node {
            mesh: Some(MeshId(0)),
            ..Node::default()
        });
        scene.roots.push(NodeId(0));
        render_ok(&scene);
    }

    #[test]
    fn one_by_one_pixel_render_works() {
        let mut scene = Scene3D::new();
        scene.meshes.push(triangle_mesh());
        scene.nodes.push(Node {
            mesh: Some(MeshId(0)),
            ..Node::default()
        });
        scene.roots.push(NodeId(0));
        for backend in both_backends() {
            let mut renderer = make_renderer(backend).expect("construct");
            let opts = RenderOptions {
                width: 1,
                height: 1,
                ..RenderOptions::default()
            };
            let img = renderer.render(&scene, &opts).expect("render");
            assert_eq!((img.width, img.height), (1, 1), "{backend:?}");
            assert_eq!(img.pixels.len(), 4, "{backend:?}");
        }
    }

    #[test]
    fn empty_primitive_and_empty_mesh_render_background() {
        let mut scene = Scene3D::new();
        scene
            .meshes
            .push(Mesh::new("e".to_string()).with_primitive(Primitive::new(Topology::Triangles)));
        scene.meshes.push(Mesh::new("empty".to_string()));
        scene.nodes.push(Node {
            mesh: Some(MeshId(0)),
            ..Node::default()
        });
        scene.nodes.push(Node {
            mesh: Some(MeshId(1)),
            ..Node::default()
        });
        scene.roots.push(NodeId(0));
        scene.roots.push(NodeId(1));
        for backend in both_backends() {
            let mut renderer = make_renderer(backend).expect("construct");
            let img = renderer.render(&scene, &tiny_opts()).expect("render");
            for px in img.pixels.chunks_exact(4) {
                assert_eq!(px, &[9, 9, 9, 255], "{backend:?}");
            }
        }
    }

    #[test]
    fn deep_linear_hierarchy_terminates() {
        // 10k-deep parent chain. The iterative pre-order walk keeps
        // its own heap stack, so traversal depth must never touch the
        // call stack (test threads only get 2 MiB).
        const DEPTH: u32 = 10_000;
        let mut scene = Scene3D::new();
        scene.meshes.push(triangle_mesh());
        for i in 0..DEPTH {
            let children = if i + 1 < DEPTH {
                vec![NodeId(i + 1)]
            } else {
                Vec::new()
            };
            scene.nodes.push(Node {
                mesh: if i + 1 == DEPTH {
                    Some(MeshId(0))
                } else {
                    None
                },
                children,
                ..Node::default()
            });
        }
        scene.roots.push(NodeId(0));
        render_ok(&scene);
    }
}
