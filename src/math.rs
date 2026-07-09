//! Small vector / matrix helpers shared by every backend.
//!
//! Column-vector convention, row-major storage: `m[row][col]`, points
//! transform as `p' = M · p`. Nothing here is backend-specific — the
//! scanline rasteriser uses the matrix chain for vertex projection,
//! the raycast backend uses the same chain for scene baking and the
//! vector helpers for shading maths.

/// 4×4 identity matrix.
pub(crate) fn identity4() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// `a · b` (matrix product; `b` is applied first to a column vector).
pub(crate) fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0_f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] =
                a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    out
}

/// `m · v` for a homogeneous 4-vector.
pub(crate) fn mat4_mul_vec4(m: &[[f32; 4]; 4], v: [f32; 4]) -> [f32; 4] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3] * v[3],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3] * v[3],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3] * v[3],
        m[3][0] * v[0] + m[3][1] * v[1] + m[3][2] * v[2] + m[3][3] * v[3],
    ]
}

/// Transform a 3D point (`w = 1`) through `m` with perspective divide.
pub(crate) fn mat4_mul_point(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    let v = mat4_mul_vec4(m, [p[0], p[1], p[2], 1.0]);
    if v[3].abs() > f32::EPSILON {
        [v[0] / v[3], v[1] / v[3], v[2] / v[3]]
    } else {
        [v[0], v[1], v[2]]
    }
}

/// Multiply the 3×3 upper-left of `m` by `v`. Used to transform
/// directions (normals) without picking up the translation column.
pub(crate) fn mat3_mul_vec3(m: &[[f32; 4]; 4], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// `a + b`.
pub(crate) fn vec3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// `a - b`.
pub(crate) fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `v * s`.
pub(crate) fn vec3_scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Dot product.
pub(crate) fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product (right-handed).
pub(crate) fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Unit-length copy of `v`; the zero vector maps to itself rather than
/// NaN so degenerate normals stay silent.
pub(crate) fn vec3_normalise(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < f32::EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_round_trips_points() {
        let id = identity4();
        assert_eq!(mat4_mul_point(&id, [1.0, -2.0, 3.0]), [1.0, -2.0, 3.0]);
        assert_eq!(mat4_mul(id, id), id);
    }

    #[test]
    fn cross_of_axes_is_third_axis() {
        assert_eq!(
            vec3_cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            [0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn normalise_zero_vector_is_zero() {
        assert_eq!(vec3_normalise([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn add_scale_compose() {
        let p = vec3_add([1.0, 2.0, 3.0], vec3_scale([2.0, 0.0, -1.0], 0.5));
        assert_eq!(p, [2.0, 2.0, 2.5]);
    }
}
