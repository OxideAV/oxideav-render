//! Camera framing + scene bounding-box walk shared by every backend.
//!
//! The scanline backend consumes the [`Camera`]'s `view` / `proj`
//! matrices for vertex projection; the raycast backend consumes the
//! same camera's retained placement (eye, orthonormal basis, frustum
//! scalars) through [`Camera::primary_ray`], so both backends frame an
//! identical view of the same scene from the same [`RenderOptions`].
//!
//! Auto-framing: when [`RenderOptions::camera`] is `None`, the camera
//! is placed on the `+Z` axis looking toward `-Z` at a distance that
//! fits the scene's world-space bounding box with a 1.2× margin. A
//! user [`CameraSpec`] orbits the same bounding-sphere fit by
//! elevation / azimuth, with `distance` as a multiplier of the fit
//! distance. Look-at / perspective / orthographic matrices use the
//! standard right-handed column-vector conventions.

use oxideav_mesh3d::{NodeId, Scene3D};

use crate::math::{
    identity4, mat4_mul, mat4_mul_vec4, vec3_add, vec3_cross, vec3_dot, vec3_normalise, vec3_scale,
    vec3_sub,
};
use crate::options::{CameraSpec, Projection, RenderOptions};

// ---------------------------------------------------------------------
// Scene bounding box.
// ---------------------------------------------------------------------

/// World-space axis-aligned bounding box accumulated over the scene.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BBox {
    pub(crate) min: [f32; 3],
    pub(crate) max: [f32; 3],
}

impl BBox {
    pub(crate) fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
        }
    }

    pub(crate) fn extend(&mut self, p: [f32; 3]) {
        for (i, &v) in p.iter().enumerate() {
            if v < self.min[i] {
                self.min[i] = v;
            }
            if v > self.max[i] {
                self.max[i] = v;
            }
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.min[0] > self.max[0]
    }
}

/// Walk the scene's node forest and accumulate the world-space AABB
/// over every mesh vertex. An empty / geometry-free scene returns a
/// unit half-extent box centred on the origin so the camera maths
/// stays finite.
pub(crate) fn scene_bbox(scene: &Scene3D) -> BBox {
    let mut bbox = BBox::empty();
    for &root in &scene.roots {
        accumulate_node_bbox(scene, root, identity4(), &mut bbox);
    }
    if bbox.is_empty() {
        BBox {
            min: [-0.5, -0.5, -0.5],
            max: [0.5, 0.5, 0.5],
        }
    } else {
        bbox
    }
}

fn accumulate_node_bbox(scene: &Scene3D, id: NodeId, parent_world: [[f32; 4]; 4], bbox: &mut BBox) {
    let Some(node) = scene.nodes.get(id.0 as usize) else {
        return;
    };
    let world = mat4_mul(parent_world, node.transform.to_matrix());
    if let Some(mesh_id) = node.mesh {
        if let Some(mesh) = scene.meshes.get(mesh_id.0 as usize) {
            for prim in &mesh.primitives {
                for p in &prim.positions {
                    let v = mat4_mul_vec4(&world, [p[0], p[1], p[2], 1.0]);
                    if v[3].abs() > f32::EPSILON {
                        bbox.extend([v[0] / v[3], v[1] / v[3], v[2] / v[3]]);
                    }
                }
            }
        }
    }
    let children: Vec<NodeId> = scene
        .nodes
        .get(id.0 as usize)
        .map(|n| n.children.clone())
        .unwrap_or_default();
    for child in children {
        accumulate_node_bbox(scene, child, world, bbox);
    }
}

// ---------------------------------------------------------------------
// Camera.
// ---------------------------------------------------------------------

/// Framed camera — projection matrices for the rasteriser plus the
/// retained placement for per-pixel ray generation.
// TODO(raycast): the `dead_code` allow comes off when the raycast
// backend (the ray-generation consumer) lands in the next milestone.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Camera {
    /// World → view matrix (rows are the camera basis).
    pub(crate) view: [[f32; 4]; 4],
    /// View → clip matrix.
    pub(crate) proj: [[f32; 4]; 4],
    /// Camera position in world space.
    pub(crate) eye: [f32; 3],
    /// Unit forward vector (toward the look-at target).
    pub(crate) forward: [f32; 3],
    /// Unit right vector.
    pub(crate) side: [f32; 3],
    /// Unit up vector (orthogonalised).
    pub(crate) up: [f32; 3],
    /// Selected projection kind.
    pub(crate) projection: Projection,
    /// Near plane distance (world units along `forward`).
    pub(crate) near: f32,
    /// Far plane distance (world units along `forward`).
    pub(crate) far: f32,
    /// Half-extent of the view plane at unit distance, horizontal
    /// (`tan(fov/2) * aspect` for perspective) — or half frustum
    /// width in world units for orthographic.
    pub(crate) half_w: f32,
    /// Vertical counterpart of [`Camera::half_w`].
    pub(crate) half_h: f32,
}

#[allow(dead_code)] // TODO(raycast): consumed by the raycast backend next milestone.
impl Camera {
    /// Build a camera honouring [`RenderOptions::camera`] /
    /// [`RenderOptions::projection`] / [`RenderOptions::fov_deg`].
    pub(crate) fn build(width: u32, height: u32, bbox: BBox, opts: &RenderOptions) -> Self {
        let (cx, cy, cz) = (
            (bbox.min[0] + bbox.max[0]) * 0.5,
            (bbox.min[1] + bbox.max[1]) * 0.5,
            (bbox.min[2] + bbox.max[2]) * 0.5,
        );
        let extent = ((bbox.max[0] - bbox.min[0]).max(bbox.max[1] - bbox.min[1]))
            .max(bbox.max[2] - bbox.min[2])
            .max(1.0e-3);
        let aspect = (width.max(1) as f32) / (height.max(1) as f32);
        let radius = extent * 0.5 * 1.2;

        let projection = opts.projection;
        let fov_y = opts.fov_deg.to_radians();

        // Camera placement.
        let (eye, dist_units) = match opts.camera {
            Some(cam) => Self::user_orbit(cam, projection, fov_y, radius, extent, [cx, cy, cz]),
            None => {
                // Default: look down +Z toward -Z.
                let dist = match projection {
                    Projection::Perspective => radius / (fov_y * 0.5).tan(),
                    Projection::Orthographic => extent * 1.5,
                };
                ([cx, cy, cz + dist], dist)
            }
        };

        let target = [cx, cy, cz];
        let world_up = [0.0, 1.0, 0.0];
        let view = look_at(eye, target, world_up);
        let forward = vec3_normalise(vec3_sub(target, eye));
        let side = vec3_normalise(vec3_cross(forward, world_up));
        let up = vec3_cross(side, forward);

        let tan_half = (fov_y * 0.5).tan();
        let (proj, near, far, half_w, half_h) = match projection {
            Projection::Perspective => {
                let near = (dist_units - extent).max(extent * 0.01);
                let far = dist_units + extent * 2.0;
                (
                    perspective(fov_y, aspect, near, far),
                    near,
                    far,
                    tan_half * aspect,
                    tan_half,
                )
            }
            Projection::Orthographic => {
                // Frame the full extent on the smaller axis with a
                // 1.2x margin (matching the perspective fit).
                let half = radius;
                let half_w = if aspect >= 1.0 { half * aspect } else { half };
                let half_h = if aspect >= 1.0 { half } else { half / aspect };
                let near = (dist_units - extent * 2.0).min(-extent);
                let far = dist_units + extent * 2.0;
                (
                    orthographic(-half_w, half_w, -half_h, half_h, near, far),
                    near,
                    far,
                    half_w,
                    half_h,
                )
            }
        };
        Self {
            view,
            proj,
            eye,
            forward,
            side,
            up,
            projection,
            near,
            far,
            half_w,
            half_h,
        }
    }

    fn user_orbit(
        cam: CameraSpec,
        projection: Projection,
        fov_y: f32,
        radius: f32,
        extent: f32,
        center: [f32; 3],
    ) -> ([f32; 3], f32) {
        let el = cam.elevation_deg.to_radians();
        let az = cam.azimuth_deg.to_radians();
        // `cam.distance` is a multiplier of the bounding-sphere radius
        // auto-frame distance.
        let auto_dist = match projection {
            Projection::Perspective => radius / (fov_y * 0.5).tan(),
            Projection::Orthographic => extent * 1.5,
        };
        let dist = auto_dist * cam.distance;
        let cos_el = el.cos();
        let dir = [cos_el * az.sin(), el.sin(), cos_el * az.cos()];
        let eye = [
            center[0] + dir[0] * dist,
            center[1] + dir[1] * dist,
            center[2] + dir[2] * dist,
        ];
        (eye, dist)
    }

    /// Generate the primary ray through the pixel-space point
    /// `(px, py)` on a `width × height` viewport. Pixel centres are at
    /// `x + 0.5`; the caller passes the sample position it wants.
    ///
    /// Returns `(origin, direction)` with a unit-length direction, so
    /// a hit parameter `t` measures world distance from the origin.
    ///
    /// Perspective: rays fan out from the eye through the view plane —
    /// the inverse of the projection applied by the rasteriser's
    /// vertex path, so both backends agree on which world point covers
    /// which pixel. Orthographic: parallel rays along `forward`,
    /// origins spread across the frustum's near rectangle.
    pub(crate) fn primary_ray(
        &self,
        px: f32,
        py: f32,
        width: f32,
        height: f32,
    ) -> ([f32; 3], [f32; 3]) {
        let ndc_x = (px / width.max(1.0)) * 2.0 - 1.0;
        let ndc_y = 1.0 - (py / height.max(1.0)) * 2.0;
        match self.projection {
            Projection::Perspective => {
                let dir = vec3_normalise(vec3_add(
                    vec3_add(
                        vec3_scale(self.side, ndc_x * self.half_w),
                        vec3_scale(self.up, ndc_y * self.half_h),
                    ),
                    self.forward,
                ));
                (self.eye, dir)
            }
            Projection::Orthographic => {
                let origin = vec3_add(
                    self.eye,
                    vec3_add(
                        vec3_scale(self.side, ndc_x * self.half_w),
                        vec3_scale(self.up, ndc_y * self.half_h),
                    ),
                );
                (origin, self.forward)
            }
        }
    }

    /// Map a hit's forward view depth (world units along
    /// [`Camera::forward`], positive in front of the camera) to the
    /// NDC z the projection matrix would have produced — near plane →
    /// `-1`, far plane → `+1`. Keeps the raycast backend's DepthDebug
    /// output on the same scale as the rasteriser's interpolated z.
    pub(crate) fn ndc_z(&self, view_depth: f32) -> f32 {
        match self.projection {
            Projection::Perspective => {
                let nf = 1.0 / (self.near - self.far);
                let d = view_depth.max(f32::MIN_POSITIVE);
                -(self.far + self.near) * nf + 2.0 * self.far * self.near * nf / d
            }
            Projection::Orthographic => {
                let fne = self.far - self.near;
                if fne.abs() < f32::EPSILON {
                    0.0
                } else {
                    (2.0 * view_depth - (self.far + self.near)) / fne
                }
            }
        }
    }

    /// Forward view depth of a world point: distance along
    /// [`Camera::forward`] from the camera plane through the eye.
    pub(crate) fn view_depth(&self, p: [f32; 3]) -> f32 {
        vec3_dot(vec3_sub(p, self.eye), self.forward)
    }
}

/// Right-handed look-at view matrix (rows: side, up, -forward).
pub(crate) fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = vec3_normalise(vec3_sub(target, eye));
    let s = vec3_normalise(vec3_cross(f, up));
    let u = vec3_cross(s, f);
    [
        [s[0], s[1], s[2], -vec3_dot(s, eye)],
        [u[0], u[1], u[2], -vec3_dot(u, eye)],
        [-f[0], -f[1], -f[2], vec3_dot(f, eye)],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Right-handed perspective projection with `[-1, 1]` NDC depth.
pub(crate) fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_y * 0.5).tan();
    let nf = 1.0 / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) * nf, 2.0 * far * near * nf],
        [0.0, 0.0, -1.0, 0.0],
    ]
}

/// Right-handed orthographic projection (outputs `w = 1` so the
/// perspective divide is a no-op).
pub(crate) fn orthographic(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> [[f32; 4]; 4] {
    let rl = right - left;
    let tb = top - bottom;
    let fne = far - near;
    [
        [2.0 / rl, 0.0, 0.0, -(right + left) / rl],
        [0.0, 2.0 / tb, 0.0, -(top + bottom) / tb],
        [0.0, 0.0, -2.0 / fne, -(far + near) / fne],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::mat4_mul_point;
    use crate::options::CameraSpec;

    fn unit_bbox() -> BBox {
        BBox {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        }
    }

    #[test]
    fn camera_with_user_override_is_finite() {
        let opts = RenderOptions {
            camera: Some(CameraSpec {
                elevation_deg: 30.0,
                azimuth_deg: 45.0,
                distance: 1.5,
            }),
            ..RenderOptions::default()
        };
        let cam = Camera::build(64, 64, unit_bbox(), &opts);
        for row in cam.view {
            for v in row {
                assert!(v.is_finite(), "camera view matrix component not finite");
            }
        }
    }

    #[test]
    fn ortho_projection_matrix_has_zero_w_translation() {
        let opts = RenderOptions {
            projection: Projection::Orthographic,
            ..RenderOptions::default()
        };
        let cam = Camera::build(64, 64, unit_bbox(), &opts);
        // Ortho proj's bottom row is [0, 0, 0, 1] (no perspective divide).
        assert_eq!(cam.proj[3], [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn centre_pixel_primary_ray_points_forward() {
        let cam = Camera::build(64, 64, unit_bbox(), &RenderOptions::default());
        let (origin, dir) = cam.primary_ray(32.0, 32.0, 64.0, 64.0);
        assert_eq!(origin, cam.eye);
        for i in 0..3 {
            assert!(
                (dir[i] - cam.forward[i]).abs() < 1.0e-6,
                "centre ray must align with forward, axis {i}: {dir:?} vs {:?}",
                cam.forward
            );
        }
        // Unit length.
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        assert!((len - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn corner_primary_rays_are_symmetric() {
        let cam = Camera::build(64, 64, unit_bbox(), &RenderOptions::default());
        let (_, tl) = cam.primary_ray(0.0, 0.0, 64.0, 64.0);
        let (_, br) = cam.primary_ray(64.0, 64.0, 64.0, 64.0);
        // The default camera looks down -Z with side = +X-ish, so
        // opposite corners mirror through the forward axis.
        let sum = vec3_add(tl, br);
        let fwd2 = vec3_scale(cam.forward, vec3_dot(sum, cam.forward));
        for i in 0..3 {
            assert!(
                (sum[i] - fwd2[i]).abs() < 1.0e-5,
                "corner rays should mirror across forward: {tl:?} + {br:?}"
            );
        }
    }

    #[test]
    fn ortho_primary_rays_are_parallel() {
        let opts = RenderOptions {
            projection: Projection::Orthographic,
            ..RenderOptions::default()
        };
        let cam = Camera::build(64, 64, unit_bbox(), &opts);
        let (o1, d1) = cam.primary_ray(0.0, 0.0, 64.0, 64.0);
        let (o2, d2) = cam.primary_ray(64.0, 32.0, 64.0, 64.0);
        assert_eq!(d1, d2, "orthographic rays must be parallel");
        assert_ne!(o1, o2, "orthographic ray origins must spread");
    }

    #[test]
    fn ndc_z_endpoints_map_near_far() {
        for projection in [Projection::Perspective, Projection::Orthographic] {
            let opts = RenderOptions {
                projection,
                ..RenderOptions::default()
            };
            let cam = Camera::build(64, 64, unit_bbox(), &opts);
            let zn = cam.ndc_z(cam.near);
            let zf = cam.ndc_z(cam.far);
            assert!(
                (zn - -1.0).abs() < 1.0e-4,
                "{projection:?}: near plane must map to -1, got {zn}"
            );
            assert!(
                (zf - 1.0).abs() < 1.0e-4,
                "{projection:?}: far plane must map to +1, got {zf}"
            );
        }
    }

    #[test]
    fn view_depth_of_target_is_camera_distance() {
        let cam = Camera::build(64, 64, unit_bbox(), &RenderOptions::default());
        // The auto-framed camera looks at the bbox centre (origin).
        let d = cam.view_depth([0.0, 0.0, 0.0]);
        let dist =
            (cam.eye[0] * cam.eye[0] + cam.eye[1] * cam.eye[1] + cam.eye[2] * cam.eye[2]).sqrt();
        assert!(
            (d - dist).abs() < 1.0e-4,
            "view depth {d} vs eye dist {dist}"
        );
    }

    #[test]
    fn camera_basis_is_orthonormal() {
        let opts = RenderOptions {
            camera: Some(CameraSpec {
                elevation_deg: 20.0,
                azimuth_deg: 110.0,
                distance: 2.0,
            }),
            ..RenderOptions::default()
        };
        let cam = Camera::build(64, 48, unit_bbox(), &opts);
        for (a, b) in [
            (cam.forward, cam.side),
            (cam.side, cam.up),
            (cam.up, cam.forward),
        ] {
            assert!(vec3_dot(a, b).abs() < 1.0e-5, "basis not orthogonal");
        }
        for v in [cam.forward, cam.side, cam.up] {
            let len = vec3_dot(v, v).sqrt();
            assert!((len - 1.0).abs() < 1.0e-5, "basis vector not unit");
        }
    }

    #[test]
    fn projection_of_ray_point_lands_on_source_pixel() {
        // Round-trip consistency between the two camera consumers:
        // walk a primary ray to some t, project that world point back
        // through view+proj, and check it lands on the source pixel.
        for projection in [Projection::Perspective, Projection::Orthographic] {
            let opts = RenderOptions {
                projection,
                ..RenderOptions::default()
            };
            let cam = Camera::build(80, 60, unit_bbox(), &opts);
            for (px, py) in [(40.0, 30.0), (10.0, 12.0), (70.0, 55.0)] {
                let (origin, dir) = cam.primary_ray(px, py, 80.0, 60.0);
                let p = vec3_add(origin, vec3_scale(dir, 2.5));
                let viewp = mat4_mul_point(&cam.view, p);
                let clip = mat4_mul_vec4(&cam.proj, [viewp[0], viewp[1], viewp[2], 1.0]);
                let w = if clip[3].abs() > f32::EPSILON {
                    clip[3]
                } else {
                    1.0
                };
                let sx = (clip[0] / w * 0.5 + 0.5) * 80.0;
                let sy = (1.0 - (clip[1] / w * 0.5 + 0.5)) * 60.0;
                assert!(
                    (sx - px).abs() < 0.05 && (sy - py).abs() < 0.05,
                    "{projection:?}: pixel ({px}, {py}) round-tripped to ({sx}, {sy})"
                );
            }
        }
    }

    #[test]
    fn empty_scene_bbox_is_unit_box() {
        let scene = Scene3D::new();
        let bbox = scene_bbox(&scene);
        assert_eq!(bbox.min, [-0.5, -0.5, -0.5]);
        assert_eq!(bbox.max, [0.5, 0.5, 0.5]);
    }
}
