//! Error type for `oxideav-render`.

/// Crate-local error.
///
/// `oxideav_mesh3d::Error` is a re-export of `oxideav_core::Error`, so
/// a single `Scene` variant covers both 3D-asset failures and any
/// framework-level error that bubbles up through registration.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Reserved for backend selectors whose implementation has not
    /// landed yet. Phase A returned this from every `make_renderer`
    /// call; the Phase B scanline and Phase D raycast backends both
    /// construct, so nothing returns it today — the Phase E
    /// path-tracer selector will while it is stubbed.
    #[error("renderer backend not implemented in this phase")]
    NotImplemented,

    /// A 3D-scene operation failed before the renderer could touch it
    /// (decode / asset-load / scene-walk problem). Bubbled up from
    /// [`oxideav_mesh3d::Error`].
    #[error("3D scene error: {0}")]
    Scene(#[from] oxideav_mesh3d::Error),

    /// [`crate::RenderRegistry::make`] was called with a name that
    /// nothing had registered. The wrapped string is the name that
    /// missed.
    #[error("renderer backend '{0}' not registered")]
    BackendNotFound(String),

    /// [`crate::RenderOptions::validate`] caught a malformed option
    /// before any backend touched it. The wrapped string describes the
    /// offending field and value so the caller can surface it to a UI
    /// or a job-graph validator without re-deriving the constraint.
    #[error("invalid render options: {0}")]
    InvalidOptions(String),
}

/// Crate-local `Result` alias.
pub type Result<T> = std::result::Result<T, Error>;
