//! Error type for `oxideav-render`.

/// Crate-local error.
///
/// `oxideav_mesh3d::Error` is a re-export of `oxideav_core::Error`, so
/// a single `Scene` variant covers both 3D-asset failures and any
/// framework-level error that bubbles up through registration.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Returned by Phase A `make_renderer` calls — the backend selector
    /// is wired but the backend implementation has not landed yet.
    /// Phase B fills in the scanline backend; Phase D adds raycast.
    #[error("renderer backend not implemented in this phase")]
    NotImplemented,

    /// A 3D-scene operation failed before the renderer could touch it
    /// (decode / asset-load / scene-walk problem). Bubbled up from
    /// [`oxideav_mesh3d::Error`].
    #[error("3D scene error: {0}")]
    Scene(#[from] oxideav_mesh3d::Error),
}

/// Crate-local `Result` alias.
pub type Result<T> = std::result::Result<T, Error>;
