//! [`RenderRegistry`] — process-wide named lookup for renderer
//! backends.
//!
//! Parallel to [`oxideav_mesh3d::Mesh3DRegistry`]; wires Phase C
//! integration with `oxideav-pipeline`. Downstream consumers — first
//! `oxideav-meta`'s `populate_render_registry`, eventually
//! `oxideav-pipeline`'s executor — consult the registry to instantiate
//! a backend from a string name in the JSON job graph.
//!
//! The renderer surface is simpler than the 3D-codec surface: there is
//! no extension routing, just `name → factory`. Names are the
//! lowercase JSON-graph convention (`"scanline"` today, `"raycast"`
//! and `"pathtrace"` in Phases D + E).
//!
//! Internal storage is a `Vec<(String, RendererFactory)>` so that
//! enumeration via [`RenderRegistry::names`] preserves registration
//! order (matters for stable test output + deterministic JSON-graph
//! validation diagnostics) without pulling in `indexmap`.

use crate::{Error, RenderBackend, Renderer, Result};

/// Factory closure that constructs a fresh boxed [`Renderer`] per
/// call.
///
/// The factory returns [`Result`] so future backends that need to
/// validate environment (e.g. GPU device availability for a Vulkan
/// backend) can fail at construction time rather than at first
/// `render` call.
pub type RendererFactory = Box<dyn Fn() -> Result<Box<dyn Renderer>> + Send + Sync + 'static>;

/// Runtime lookup table from backend name to [`RendererFactory`].
///
/// Cf. [`oxideav_mesh3d::Mesh3DRegistry`] — this is the renderer
/// equivalent. Construct with [`RenderRegistry::new`] (or
/// [`RenderRegistry::default`]); populate with the in-tree built-ins
/// via [`register_into`]; look up a backend by name with
/// [`RenderRegistry::make`].
pub struct RenderRegistry {
    entries: Vec<(String, RendererFactory)>,
}

impl RenderRegistry {
    /// Construct an empty registry.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register `factory` under `name`.
    ///
    /// Registering the same `name` twice keeps both entries; lookup
    /// via [`RenderRegistry::make`] returns the *first* match, so a
    /// caller wanting to override a built-in should register the
    /// override before calling [`register_into`]. This matches the
    /// surrounding `Mesh3DRegistry` policy of "explicit registration
    /// wins."
    pub fn register(&mut self, name: &str, factory: RendererFactory) {
        self.entries.push((name.to_string(), factory));
    }

    /// Look up `name` and invoke the factory.
    ///
    /// Returns [`Error::BackendNotFound`] if no entry is registered
    /// under that name.
    pub fn make(&self, name: &str) -> Result<Box<dyn Renderer>> {
        for (registered_name, factory) in &self.entries {
            if registered_name == name {
                return factory();
            }
        }
        Err(Error::BackendNotFound(name.to_string()))
    }

    /// Enumerate all registered backend names, in registration order.
    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|(n, _)| n.as_str()).collect()
    }
}

impl Default for RenderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for RenderRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderRegistry")
            .field("names", &self.names())
            .finish()
    }
}

/// Populate `reg` with every built-in backend.
///
/// Phase B shipped the scanline rasteriser, Phase D the Whitted ray
/// tracer; Phase E will add `"pathtrace"`. The string keys are the
/// JSON-graph convention used by `oxideav-pipeline`'s
/// `DagNode::Render3D` — lowercase, no `RenderBackend::` prefix.
pub fn register_into(reg: &mut RenderRegistry) {
    reg.register(
        "scanline",
        Box::new(|| crate::make_renderer(RenderBackend::Scanline)),
    );
    reg.register(
        "raycast",
        Box::new(|| crate::make_renderer(RenderBackend::Raycast)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackgroundColor, RenderOptions, ShadingMode};

    #[test]
    fn empty_registry_has_zero_names() {
        let reg = RenderRegistry::new();
        assert!(reg.names().is_empty());
    }

    #[test]
    fn default_registry_is_empty() {
        let reg = RenderRegistry::default();
        assert!(reg.names().is_empty());
    }

    #[test]
    fn register_into_exposes_scanline_and_raycast() {
        let mut reg = RenderRegistry::new();
        register_into(&mut reg);
        let names = reg.names();
        for expected in ["scanline", "raycast"] {
            assert!(
                names.contains(&expected),
                "register_into must register \"{expected}\", got {names:?}"
            );
        }
    }

    #[test]
    fn make_raycast_constructs_working_backend() {
        let mut reg = RenderRegistry::new();
        register_into(&mut reg);
        let mut renderer = reg
            .make("raycast")
            .expect("registry must hand back a raycast renderer");
        let scene = oxideav_mesh3d::Scene3D::new();
        let opts = RenderOptions {
            width: 4,
            height: 4,
            ..RenderOptions::default()
        };
        let img = renderer.render(&scene, &opts).expect("render must succeed");
        assert_eq!((img.width, img.height), (4, 4));
    }

    #[test]
    fn make_scanline_paints_some_pixels() {
        use oxideav_mesh3d::{Mesh, MeshId, Node, Primitive, Scene3D, Topology};

        let mut reg = RenderRegistry::new();
        register_into(&mut reg);

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

        let mut renderer = reg
            .make("scanline")
            .expect("registry must hand back a scanline renderer");
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
        assert!(
            any_drawn,
            "scanline backend via registry must paint at least one pixel"
        );
    }

    #[test]
    fn make_unknown_backend_returns_backend_not_found() {
        let reg = RenderRegistry::new();
        match reg.make("nope") {
            Err(Error::BackendNotFound(name)) => assert_eq!(name, "nope"),
            Err(other) => panic!("expected BackendNotFound(\"nope\"), got {other:?}"),
            Ok(_) => panic!("expected BackendNotFound(\"nope\"), got Ok"),
        }
    }

    #[test]
    fn debug_impl_lists_registered_names() {
        let mut reg = RenderRegistry::new();
        register_into(&mut reg);
        let debug = format!("{reg:?}");
        assert!(
            debug.contains("scanline"),
            "Debug impl must surface registered names, got {debug}"
        );
    }
}
