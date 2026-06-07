//! [`RenderSource`] — wrap a `Scene3D` + [`crate::Renderer`] + [`crate::RenderOptions`]
//! as an [`oxideav_core::FrameSource`] so the pipeline executor can pull
//! rendered frames straight out of a 3D scene without going through a
//! container or decoder.
//!
//! Phase C-3d of the pipeline integration: the consumer
//! (`oxideav-cli-convert`) installs a closure on
//! `oxideav-pipeline::RunContext` that constructs a `RenderSource` and
//! hands it to the pipeline's `Render3D` DAG node. This module ships
//! the type that closure constructs.
//!
//! ## Scope
//!
//! Animation is out of scope for Phase C-3d — `RenderSource::new`
//! produces a single-frame source. The first `next_frame` call returns
//! `Ok(Frame::Video(_))`; every subsequent call returns
//! `Err(Error::Eof)`. A future phase adds an animation-aware variant
//! that advances scene time per `next_frame` call.
//!
//! ## Module gating
//!
//! Gated on the `registry` cargo feature alongside the
//! `oxideav-core` dependency. The standalone build
//! (`--no-default-features`) does not expose this type — it cannot,
//! because `FrameSource` lives in `oxideav-core`.

use oxideav_core::{
    CodecId, CodecParameters, Error as CoreError, Frame, FrameSource, PixelFormat, Rational,
    Result as CoreResult, VideoFrame, VideoPlane,
};

use crate::{RenderOptions, Renderer, RgbaImage};

/// `oxideav_core::FrameSource` that emits the rendered output of a
/// `Scene3D` as a single `Frame::Video`, then `Error::Eof`.
///
/// Constructed from a scene, a `Box<dyn Renderer>`, and `RenderOptions`.
/// The renderer is owned by the source so downstream `next_frame` calls
/// can re-use it. The scanline rasteriser allocates a fresh framebuffer
/// per call so re-use is safe; future backends with persistent state
/// (GPU resource handles, accumulation buffers) are equally compatible
/// with this shape.
///
/// Codec parameters advertise:
/// - `codec_id = "rawvideo"` (matching `oxideav-generator`'s synthetic
///   image sources)
/// - `media_type = MediaType::Video`
/// - `width` / `height` from [`RenderOptions`]
/// - `pixel_format = PixelFormat::Rgba` (matches [`RgbaImage`]'s shape)
/// - `frame_rate = 1/1` — the still scene case; updated when the
///   animation-aware variant lands.
///
/// # Example
///
/// ```ignore
/// // Sketch only — full code in src/source.rs tests.
/// use oxideav_core::{Frame, FrameSource};
/// use oxideav_mesh3d::Scene3D;
/// use oxideav_render::{make_renderer, RenderBackend, RenderOptions, RenderSource};
///
/// let scene = Scene3D::new();
/// let renderer = make_renderer(RenderBackend::Scanline).unwrap();
/// let opts = RenderOptions::default();
/// let mut src = RenderSource::new(scene, renderer, opts);
/// let frame = src.next_frame().unwrap();
/// assert!(matches!(frame, Frame::Video(_)));
/// ```
pub struct RenderSource {
    scene: oxideav_mesh3d::Scene3D,
    renderer: Box<dyn Renderer>,
    opts: RenderOptions,
    params: CodecParameters,
    emitted: bool,
}

impl RenderSource {
    /// Construct a `RenderSource` for a still scene.
    ///
    /// The renderer is consumed; downstream `next_frame` calls re-use
    /// it. `opts.width` / `opts.height` flow into the source's
    /// advertised [`CodecParameters`] so pipeline filters and encoders
    /// downstream can configure themselves before the first frame
    /// arrives.
    pub fn new(
        scene: oxideav_mesh3d::Scene3D,
        renderer: Box<dyn Renderer>,
        opts: RenderOptions,
    ) -> Self {
        let mut params = CodecParameters::video(CodecId::new("rawvideo"));
        params.width = Some(opts.width);
        params.height = Some(opts.height);
        params.pixel_format = Some(PixelFormat::Rgba);
        // Still scene → 1 fps; the animation-aware variant overrides
        // this when scene time advances per call.
        params.frame_rate = Some(Rational::new(1, 1));
        Self {
            scene,
            renderer,
            opts,
            params,
            emitted: false,
        }
    }
}

impl FrameSource for RenderSource {
    fn params(&self) -> &CodecParameters {
        &self.params
    }

    fn next_frame(&mut self) -> CoreResult<Frame> {
        if self.emitted {
            return Err(CoreError::Eof);
        }
        let img = self
            .renderer
            .render(&self.scene, &self.opts)
            .map_err(|e| CoreError::invalid(e.to_string()))?;
        self.emitted = true;
        Ok(Frame::Video(rgba_image_to_video_frame(img, 0)))
    }

    fn duration_micros(&self) -> Option<i64> {
        // One frame at 1 fps → 1 s. Mirrors the still-image
        // `SingleImageFrameSource` in `oxideav-generator`.
        Some(1_000_000)
    }
}

/// Convert an [`RgbaImage`] (the renderer's output) into a packed
/// single-plane [`VideoFrame`]. Mirrors the helper used in
/// `oxideav-generator::source::rgba_image_to_video_frame`.
fn rgba_image_to_video_frame(img: RgbaImage, pts: i64) -> VideoFrame {
    VideoFrame {
        pts: Some(pts),
        planes: vec![VideoPlane {
            stride: img.stride,
            data: img.pixels,
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{make_renderer, BackgroundColor, RenderBackend, ShadingMode};
    use oxideav_core::MediaType;
    use oxideav_mesh3d::{Mesh, MeshId, Node, NodeId, Primitive, Scene3D, Topology};

    /// Build the same one-triangle scene used by the existing
    /// `scanline_renderer_via_trait_paints_some_pixels` smoke test, so
    /// both surfaces exercise an identical scene.
    fn one_triangle_scene() -> Scene3D {
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
        let mesh = Mesh::new("triangle".to_string()).with_primitive(prim);
        let mut scene = Scene3D::new();
        scene.meshes.push(mesh);
        scene.nodes.push(Node {
            mesh: Some(MeshId(0)),
            ..Node::default()
        });
        scene.roots.push(NodeId(0));
        scene
    }

    fn small_opts() -> RenderOptions {
        RenderOptions {
            width: 32,
            height: 32,
            background: BackgroundColor([255, 255, 255, 255]),
            shading: ShadingMode::Flat,
            ..RenderOptions::default()
        }
    }

    #[test]
    fn params_advertise_expected_video_shape() {
        let scene = one_triangle_scene();
        let renderer = make_renderer(RenderBackend::Scanline).expect("scanline");
        let opts = small_opts();
        let src = RenderSource::new(scene, renderer, opts);
        let p = src.params();
        assert_eq!(p.codec_id.as_str(), "rawvideo");
        assert_eq!(p.media_type, MediaType::Video);
        assert_eq!(p.width, Some(32));
        assert_eq!(p.height, Some(32));
        assert_eq!(p.pixel_format, Some(PixelFormat::Rgba));
        assert_eq!(p.frame_rate, Some(Rational::new(1, 1)));
    }

    #[test]
    fn first_next_frame_yields_video_then_eof() {
        let scene = one_triangle_scene();
        let renderer = make_renderer(RenderBackend::Scanline).expect("scanline");
        let opts = small_opts();
        let mut src = RenderSource::new(scene, renderer, opts);

        let frame = src.next_frame().expect("first call must yield a frame");
        match frame {
            Frame::Video(v) => {
                assert_eq!(v.planes.len(), 1, "RGBA8 is single-plane");
                assert_eq!(v.planes[0].stride, 32 * 4);
                assert_eq!(v.planes[0].data.len(), 32 * 32 * 4);
            }
            other => panic!("expected Frame::Video, got {other:?}"),
        }

        // Second call must surface Eof.
        match src.next_frame() {
            Err(CoreError::Eof) => {}
            Err(other) => panic!("expected Eof, got {other:?}"),
            Ok(_) => panic!("expected Eof, got Ok"),
        }

        // Third call still Eof (idempotent post-emit).
        assert!(matches!(src.next_frame(), Err(CoreError::Eof)));
    }

    #[test]
    fn rendered_frame_has_some_non_background_pixels() {
        // Sanity-check that the actual rasterised pixels reach the
        // `VideoFrame` plane bytes — i.e. the pipeline source surface
        // delivers the same pixels the existing trait-level smoke test
        // asserts on.
        let scene = one_triangle_scene();
        let renderer = make_renderer(RenderBackend::Scanline).expect("scanline");
        let opts = small_opts();
        let mut src = RenderSource::new(scene, renderer, opts);

        let frame = src.next_frame().expect("first frame");
        let v = match frame {
            Frame::Video(v) => v,
            other => panic!("expected video, got {other:?}"),
        };
        let any_drawn = v.planes[0]
            .data
            .chunks_exact(4)
            .any(|p| p != [255, 255, 255, 255]);
        assert!(
            any_drawn,
            "scanline backend must paint at least one pixel into the VideoFrame"
        );
    }

    #[test]
    fn duration_micros_reports_one_second() {
        let scene = one_triangle_scene();
        let renderer = make_renderer(RenderBackend::Scanline).expect("scanline");
        let opts = small_opts();
        let src = RenderSource::new(scene, renderer, opts);
        assert_eq!(src.duration_micros(), Some(1_000_000));
    }

    #[test]
    fn dimensions_round_trip_through_params() {
        // A non-square framebuffer to catch any accidental width/height
        // swap inside `RenderSource::new` or `next_frame`.
        let scene = one_triangle_scene();
        let renderer = make_renderer(RenderBackend::Scanline).expect("scanline");
        let opts = RenderOptions {
            width: 48,
            height: 24,
            background: BackgroundColor([0, 0, 0, 0]),
            shading: ShadingMode::Flat,
            ..RenderOptions::default()
        };
        let mut src = RenderSource::new(scene, renderer, opts);
        let p = src.params();
        assert_eq!(p.width, Some(48));
        assert_eq!(p.height, Some(24));

        let frame = src.next_frame().expect("frame");
        match frame {
            Frame::Video(v) => {
                assert_eq!(v.planes[0].stride, 48 * 4);
                assert_eq!(v.planes[0].data.len(), 48 * 24 * 4);
            }
            other => panic!("expected video, got {other:?}"),
        }
    }
}
