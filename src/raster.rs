//! Rasterisation core shared by the scanline backend's passes (main
//! view, alpha-blend pass, shadow maps).
//!
//! * **Clipping** — triangles are clipped in homogeneous clip space
//!   against all six frustum planes with the Sutherland–Hodgman
//!   re-entrant polygon clipper (Sutherland & Hodgman, "Reentrant
//!   Polygon Clipping", CACM 17(1), 1974); lines with the parametric
//!   Liang–Barsky test (Liang & Barsky, "A New Concept and Method for
//!   Line Clipping", ACM TOG 3(1), 1984). Each clip vertex carries its
//!   barycentric coordinates with respect to the *original* triangle,
//!   which are linear in clip space, so a sub-triangle produced by
//!   clipping still reports original-triangle barycentrics.
//! * **Setup / coverage** — half-space edge functions (Pineda, "A
//!   Parallel Algorithm for Polygon Rasterization", SIGGRAPH 1988)
//!   evaluated at pixel centres with the top-left fill rule, so pixels
//!   on an edge shared by two triangles are covered exactly once.
//! * **Perspective-correct interpolation** — `attr/w` and `1/w` are
//!   affine in screen space (Heckbert & Moreton, "Interpolation for
//!   Polygon Texture Mapping and Shading", 1991); barycentrics are
//!   recovered as `(Σ λᵢ bᵢ/wᵢ) / (Σ λᵢ / wᵢ)`.
//! * **Parallelism** — the framebuffer is split into horizontal bands
//!   processed on `std::thread::scope` workers. Triangles are binned
//!   per band and processed in submission order inside each band, so
//!   the result is identical to a serial render.
//!
//! The main pass writes a *visibility buffer* (depth + primitive id +
//! barycentrics per pixel); shading happens afterwards, once per
//! visible pixel.

use std::sync::Mutex;

/// Sentinel id for an empty visibility-buffer pixel.
pub(crate) const NONE: u32 = u32::MAX;
/// Id flag marking a line / point entry (index into the line list).
pub(crate) const LINE_FLAG: u32 = 0x8000_0000;

/// Rows per parallel band.
pub(crate) const BAND_ROWS: usize = 16;

/// One visibility-buffer pixel.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VisPixel {
    /// NDC depth (`[-1, 1]`, smaller = nearer).
    pub(crate) depth: f32,
    /// Screen-triangle index, `LINE_FLAG | line index`, or [`NONE`].
    pub(crate) id: u32,
    /// Original-triangle barycentrics (lines: `b[0]` = parameter).
    pub(crate) b: [f32; 3],
}

impl VisPixel {
    pub(crate) const EMPTY: Self = Self {
        depth: f32::INFINITY,
        id: NONE,
        b: [0.0; 3],
    };
}

/// Face-culling mode for setup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cull {
    /// Keep both orientations.
    None,
    /// Drop back faces (clockwise in NDC).
    Back,
}

/// A clipped, projected (sub-)triangle ready for coverage tests.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScreenTri {
    x: [f32; 3],
    y: [f32; 3],
    z: [f32; 3],
    inv_w: [f32; 3],
    bw: [[f32; 3]; 3],
    area_inv: f32,
    top_left: [bool; 3],
    pub(crate) min_x: i32,
    pub(crate) max_x: i32,
    pub(crate) min_y: i32,
    pub(crate) max_y: i32,
    /// Caller primitive id (global triangle index).
    pub(crate) prim: u32,
    /// Counter-clockwise in NDC (front-facing under the CCW rule).
    pub(crate) front: bool,
    /// Run the fragment test (alpha mask) on covered pixels.
    pub(crate) masked: bool,
}

#[inline]
fn edge(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    (bx - ax) * (py - ay) - (by - ay) * (px - ax)
}

/// [`edge`] evaluated with the endpoints in a canonical order, so the
/// two triangles sharing an edge (which traverse it in opposite
/// directions) get bit-exact negated values — floating-point rounding
/// otherwise lets a pixel centre on the edge test outside both and
/// open a crack.
#[inline]
fn edge_canon(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    if (ax, ay) <= (bx, by) {
        edge(ax, ay, bx, by, px, py)
    } else {
        -edge(bx, by, ax, ay, px, py)
    }
}

impl ScreenTri {
    /// Affine screen weights at `(px, py)` (sum to 1; may be negative
    /// outside the triangle).
    #[inline]
    fn lambdas(&self, px: f32, py: f32) -> [f32; 3] {
        let (x, y) = (&self.x, &self.y);
        [
            edge(x[1], y[1], x[2], y[2], px, py) * self.area_inv,
            edge(x[2], y[2], x[0], y[0], px, py) * self.area_inv,
            edge(x[0], y[0], x[1], y[1], px, py) * self.area_inv,
        ]
    }

    /// Perspective-correct original-triangle barycentrics at screen
    /// point `(px, py)` (valid outside the triangle too — used for
    /// finite-difference derivatives).
    #[inline]
    pub(crate) fn bary_at(&self, px: f32, py: f32) -> [f32; 3] {
        let l = self.lambdas(px, py);
        self.bary_from_lambdas(l)
    }

    #[inline]
    fn bary_from_lambdas(&self, l: [f32; 3]) -> [f32; 3] {
        let iw = l[0] * self.inv_w[0] + l[1] * self.inv_w[1] + l[2] * self.inv_w[2];
        let iw = if iw.abs() > f32::MIN_POSITIVE {
            1.0 / iw
        } else {
            0.0
        };
        let mut b = [0.0; 3];
        for (k, bk) in b.iter_mut().enumerate() {
            *bk = (l[0] * self.bw[0][k] + l[1] * self.bw[1][k] + l[2] * self.bw[2][k]) * iw;
        }
        b
    }

    /// Visit covered pixels in rows `y0..y1` (top-left rule), calling
    /// `frag(x, y, ndc_z, bary)`.
    #[inline]
    pub(crate) fn raster_rows(
        &self,
        y0: i32,
        y1: i32,
        mut frag: impl FnMut(i32, i32, f32, [f32; 3]),
    ) {
        let ys = self.min_y.max(y0);
        let ye = self.max_y.min(y1 - 1);
        for py in ys..=ye {
            let fy = py as f32 + 0.5;
            for px in self.min_x..=self.max_x {
                let fx = px as f32 + 0.5;
                let e = [
                    edge_canon(self.x[1], self.y[1], self.x[2], self.y[2], fx, fy),
                    edge_canon(self.x[2], self.y[2], self.x[0], self.y[0], fx, fy),
                    edge_canon(self.x[0], self.y[0], self.x[1], self.y[1], fx, fy),
                ];
                let inside = (0..3).all(|k| e[k] > 0.0 || (e[k] == 0.0 && self.top_left[k]));
                if !inside {
                    continue;
                }
                let l = [
                    e[0] * self.area_inv,
                    e[1] * self.area_inv,
                    e[2] * self.area_inv,
                ];
                let z = l[0] * self.z[0] + l[1] * self.z[1] + l[2] * self.z[2];
                frag(px, py, z, self.bary_from_lambdas(l));
            }
        }
    }
}

#[derive(Clone, Copy)]
struct ClipVert {
    p: [f32; 4],
    b: [f32; 3],
}

fn plane_dist(p: [f32; 4], plane: usize) -> f32 {
    match plane {
        0 => p[3] + p[0],
        1 => p[3] - p[0],
        2 => p[3] + p[1],
        3 => p[3] - p[1],
        4 => p[3] + p[2],
        _ => p[3] - p[2],
    }
}

fn lerp_clip(a: ClipVert, b: ClipVert, t: f32) -> ClipVert {
    ClipVert {
        p: std::array::from_fn(|k| a.p[k] + (b.p[k] - a.p[k]) * t),
        b: std::array::from_fn(|k| a.b[k] + (b.b[k] - a.b[k]) * t),
    }
}

/// Clip, project and set up one triangle given its clip-space
/// vertices; pushes 0..=5 screen triangles onto `out`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setup_triangle(
    clip: [[f32; 4]; 3],
    prim: u32,
    cull: Cull,
    masked: bool,
    width: u32,
    height: u32,
    out: &mut Vec<ScreenTri>,
) {
    if clip.iter().any(|p| p.iter().any(|c| !c.is_finite())) {
        return;
    }
    // Trivial reject / accept.
    let mut all_in = true;
    for plane in 0..6 {
        let d = [
            plane_dist(clip[0], plane),
            plane_dist(clip[1], plane),
            plane_dist(clip[2], plane),
        ];
        if d.iter().all(|&v| v < 0.0) {
            return;
        }
        if d.iter().any(|&v| v < 0.0) {
            all_in = false;
        }
    }
    let bases = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut poly: Vec<ClipVert> = (0..3)
        .map(|i| ClipVert {
            p: clip[i],
            b: bases[i],
        })
        .collect();
    if !all_in {
        for plane in 0..6 {
            if poly.is_empty() {
                return;
            }
            let mut next = Vec::with_capacity(poly.len() + 2);
            for i in 0..poly.len() {
                let a = poly[i];
                let b = poly[(i + 1) % poly.len()];
                let da = plane_dist(a.p, plane);
                let db = plane_dist(b.p, plane);
                if da >= 0.0 {
                    next.push(a);
                }
                if (da >= 0.0) != (db >= 0.0) {
                    let t = da / (da - db);
                    next.push(lerp_clip(a, b, t));
                }
            }
            poly = next;
        }
        if poly.len() < 3 {
            return;
        }
    }
    let (w, h) = (width as f32, height as f32);
    let proj: Vec<([f32; 3], f32, [f32; 3])> = poly
        .iter()
        .map(|v| {
            let iw = if v.p[3].abs() > f32::MIN_POSITIVE {
                1.0 / v.p[3]
            } else {
                0.0
            };
            let sx = (v.p[0] * iw * 0.5 + 0.5) * w;
            let sy = (1.0 - (v.p[1] * iw * 0.5 + 0.5)) * h;
            ([sx, sy, v.p[2] * iw], iw, v.b)
        })
        .collect();
    for i in 1..proj.len() - 1 {
        let tri = [proj[0], proj[i], proj[i + 1]];
        let area = edge(
            tri[0].0[0],
            tri[0].0[1],
            tri[1].0[0],
            tri[1].0[1],
            tri[2].0[0],
            tri[2].0[1],
        );
        if !area.is_finite() || area.abs() < 1.0e-9 {
            continue;
        }
        // y-down screen: CCW in NDC ⇔ negative screen-space area.
        let front = area < 0.0;
        if cull == Cull::Back && !front {
            continue;
        }
        let order = if area > 0.0 { [0, 1, 2] } else { [0, 2, 1] };
        let v = [tri[order[0]], tri[order[1]], tri[order[2]]];
        let x = [v[0].0[0], v[1].0[0], v[2].0[0]];
        let y = [v[0].0[1], v[1].0[1], v[2].0[1]];
        let top_left = |a: usize, b: usize| {
            let dy = y[b] - y[a];
            let dx = x[b] - x[a];
            (dy == 0.0 && dx > 0.0) || dy < 0.0
        };
        let min_x = x.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_x = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_y = y.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_y = y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_x = (min_x - 0.5).ceil().max(0.0) as i32;
        let max_x = ((max_x - 0.5).floor()).min(w - 1.0) as i32;
        let min_y = (min_y - 0.5).ceil().max(0.0) as i32;
        let max_y = ((max_y - 0.5).floor()).min(h - 1.0) as i32;
        if min_x > max_x || min_y > max_y {
            continue;
        }
        let iw = [v[0].1, v[1].1, v[2].1];
        out.push(ScreenTri {
            x,
            y,
            z: [v[0].0[2], v[1].0[2], v[2].0[2]],
            inv_w: iw,
            bw: [
                [v[0].2[0] * iw[0], v[0].2[1] * iw[0], v[0].2[2] * iw[0]],
                [v[1].2[0] * iw[1], v[1].2[1] * iw[1], v[1].2[2] * iw[1]],
                [v[2].2[0] * iw[2], v[2].2[1] * iw[2], v[2].2[2] * iw[2]],
            ],
            area_inv: 1.0 / area.abs(),
            top_left: [top_left(1, 2), top_left(2, 0), top_left(0, 1)],
            min_x,
            max_x,
            min_y,
            max_y,
            prim,
            front,
            masked,
        });
    }
}

/// A clipped, projected line segment (a point when both ends
/// coincide).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScreenLine {
    pub(crate) a: [f32; 3],
    pub(crate) b: [f32; 3],
    /// Original-segment parameters of the clipped endpoints.
    pub(crate) t: [f32; 2],
    pub(crate) point: bool,
}

/// Clip a segment in clip space (Liang–Barsky) and project it.
pub(crate) fn setup_line(a: [f32; 4], b: [f32; 4], width: u32, height: u32) -> Option<ScreenLine> {
    if a.iter().chain(b.iter()).any(|c| !c.is_finite()) {
        return None;
    }
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for plane in 0..6 {
        let da = plane_dist(a, plane);
        let db = plane_dist(b, plane);
        if da < 0.0 && db < 0.0 {
            return None;
        }
        if da < 0.0 {
            t0 = t0.max(da / (da - db));
        } else if db < 0.0 {
            t1 = t1.min(da / (da - db));
        }
    }
    if t0 > t1 {
        return None;
    }
    let at = |t: f32| -> [f32; 3] {
        let p = [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
            a[3] + (b[3] - a[3]) * t,
        ];
        let iw = if p[3].abs() > f32::MIN_POSITIVE {
            1.0 / p[3]
        } else {
            0.0
        };
        [
            (p[0] * iw * 0.5 + 0.5) * width as f32,
            (1.0 - (p[1] * iw * 0.5 + 0.5)) * height as f32,
            p[2] * iw,
        ]
    };
    Some(ScreenLine {
        a: at(t0),
        b: at(t1),
        t: [t0, t1],
        point: false,
    })
}

/// Project a point (`None` when outside the frustum).
pub(crate) fn setup_point(p: [f32; 4], width: u32, height: u32) -> Option<ScreenLine> {
    if p.iter().any(|c| !c.is_finite()) || (0..6).any(|pl| plane_dist(p, pl) < 0.0) {
        return None;
    }
    let iw = 1.0 / p[3];
    let s = [
        (p[0] * iw * 0.5 + 0.5) * width as f32,
        (1.0 - (p[1] * iw * 0.5 + 0.5)) * height as f32,
        p[2] * iw,
    ];
    Some(ScreenLine {
        a: s,
        b: s,
        t: [0.0, 0.0],
        point: true,
    })
}

impl ScreenLine {
    /// Bresenham walk (Bresenham, IBM Systems Journal 4(1), 1965)
    /// restricted to rows `y0..y1`; `frag(x, y, z, t)`.
    pub(crate) fn raster_rows(&self, y0: i32, y1: i32, mut frag: impl FnMut(i32, i32, f32, f32)) {
        let mut x = self.a[0].round() as i32;
        let mut y = self.a[1].round() as i32;
        if self.point {
            if y >= y0 && y < y1 {
                frag(x, y, self.a[2], 0.0);
            }
            return;
        }
        let x1 = self.b[0].round() as i32;
        let y1e = self.b[1].round() as i32;
        let (xs, ys) = (x, y);
        let dx = (x1 - x).abs();
        let dy = -(y1e - y).abs();
        // Rows the segment touches.
        if y.max(y1e) < y0 || y.min(y1e) >= y1 {
            return;
        }
        let sx = if x < x1 { 1 } else { -1 };
        let sy = if y < y1e { 1 } else { -1 };
        let mut err = dx + dy;
        let len = (dx as f32).hypot(dy as f32).max(1.0);
        let steps_max = (dx - dy) as i64 + 2;
        let mut steps = 0i64;
        loop {
            if y >= y0 && y < y1 {
                let s = ((x - xs) as f32).hypot((y - ys) as f32) / len;
                let z = self.a[2] + (self.b[2] - self.a[2]) * s;
                let t = self.t[0] + (self.t[1] - self.t[0]) * s;
                frag(x, y, z, t);
            }
            if (x == x1 && y == y1e) || steps > steps_max {
                break;
            }
            steps += 1;
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }
}

/// Worker count for a `pixels`-sized pass.
pub(crate) fn worker_count(pixels: usize) -> usize {
    if pixels < 64 * 64 {
        return 1;
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(8)
}

/// Run `f(band_index, first_row, rows)` over `BAND_ROWS`-row bands of
/// the row-major `buf` (`width` items per row), in parallel.
pub(crate) fn par_bands<T: Send>(
    buf: &mut [T],
    width: usize,
    f: impl Fn(usize, usize, &mut [T]) + Sync,
) {
    if width == 0 || buf.is_empty() {
        return;
    }
    let chunk = width * BAND_ROWS;
    let n_bands = buf.len().div_ceil(chunk);
    let workers = worker_count(buf.len()).min(n_bands);
    if workers <= 1 {
        for (i, rows) in buf.chunks_mut(chunk).enumerate() {
            f(i, i * BAND_ROWS, rows);
        }
        return;
    }
    let queue: Mutex<Vec<(usize, &mut [T])>> =
        Mutex::new(buf.chunks_mut(chunk).enumerate().rev().collect());
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| loop {
                let job = queue.lock().map(|mut q| q.pop()).unwrap_or(None);
                let Some((i, rows)) = job else { break };
                f(i, i * BAND_ROWS, rows);
            });
        }
    });
}

/// Per-band lists of indices of the screen triangles overlapping each
/// band, in submission order.
pub(crate) fn bin_tris(tris: &[ScreenTri], height: u32) -> Vec<Vec<u32>> {
    let n_bands = (height as usize).div_ceil(BAND_ROWS).max(1);
    let mut bins = vec![Vec::new(); n_bands];
    for (i, t) in tris.iter().enumerate() {
        let b0 = (t.min_y.max(0) as usize) / BAND_ROWS;
        let b1 = ((t.max_y.max(0) as usize) / BAND_ROWS).min(n_bands - 1);
        for bin in bins.iter_mut().take(b1 + 1).skip(b0) {
            bin.push(i as u32);
        }
    }
    bins
}

/// Fragment coverage test for `masked` triangles:
/// `(tri, barycentrics, x, y) -> covered`.
pub(crate) type MaskTest<'a> = dyn Fn(&ScreenTri, [f32; 3], i32, i32) -> bool + Sync + 'a;

/// Visibility-buffer pass: rasterise `tris` then `lines` into a fresh
/// buffer with a strict `<` depth test (first submitted wins ties).
/// `mask_test(tri, bary, x, y)` decides coverage for `masked` tris.
pub(crate) fn raster_visibility(
    tris: &[ScreenTri],
    lines: &[ScreenLine],
    width: u32,
    height: u32,
    mask_test: &MaskTest<'_>,
) -> Vec<VisPixel> {
    let w = width as usize;
    let mut vis = vec![VisPixel::EMPTY; w * height as usize];
    let bins = bin_tris(tris, height);
    par_bands(&mut vis, w, |band, y0, rows| {
        let y1 = y0 + rows.len() / w;
        for &ti in &bins[band] {
            let t = &tris[ti as usize];
            t.raster_rows(y0 as i32, y1 as i32, |x, y, z, b| {
                let px = &mut rows[(y as usize - y0) * w + x as usize];
                if z < px.depth && z >= -1.0 && (!t.masked || mask_test(t, b, x, y)) {
                    *px = VisPixel {
                        depth: z,
                        id: ti,
                        b,
                    };
                }
            });
        }
        for (li, l) in lines.iter().enumerate() {
            l.raster_rows(y0 as i32, y1 as i32, |x, y, z, tp| {
                if x < 0 || x >= w as i32 {
                    return;
                }
                let px = &mut rows[(y as usize - y0) * w + x as usize];
                if z < px.depth {
                    *px = VisPixel {
                        depth: z,
                        id: LINE_FLAG | li as u32,
                        b: [tp, 0.0, 0.0],
                    };
                }
            });
        }
    });
    vis
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ndc_tri(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> [[f32; 4]; 3] {
        [
            [a[0], a[1], 0.0, 1.0],
            [b[0], b[1], 0.0, 1.0],
            [c[0], c[1], 0.0, 1.0],
        ]
    }

    fn coverage(tris: &[[[f32; 4]; 3]], w: u32, h: u32) -> Vec<u32> {
        let mut st = Vec::new();
        for (i, t) in tris.iter().enumerate() {
            setup_triangle(*t, i as u32, Cull::None, false, w, h, &mut st);
        }
        // Count writes per pixel (no depth test).
        let mut count = vec![0u32; (w * h) as usize];
        for t in &st {
            t.raster_rows(0, h as i32, |x, y, _, _| {
                count[(y as u32 * w + x as u32) as usize] += 1
            });
        }
        count
    }

    #[test]
    fn shared_edge_covered_exactly_once() {
        // Full-screen quad as two triangles: every pixel exactly once.
        let a = ndc_tri([-1.0, -1.0], [1.0, -1.0], [1.0, 1.0]);
        let b = ndc_tri([-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]);
        let c = coverage(&[a, b], 17, 13);
        assert!(c.iter().all(|&n| n == 1), "{c:?}");
    }

    #[test]
    fn back_face_culling_by_ndc_winding() {
        let ccw = ndc_tri([-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]);
        let cw = [ccw[0], ccw[2], ccw[1]];
        let mut st = Vec::new();
        setup_triangle(ccw, 0, Cull::Back, false, 8, 8, &mut st);
        assert_eq!(st.len(), 1);
        assert!(st[0].front);
        setup_triangle(cw, 1, Cull::Back, false, 8, 8, &mut st);
        assert_eq!(st.len(), 1, "clockwise triangle culled");
    }

    #[test]
    fn near_plane_clipping_keeps_visible_part() {
        // One vertex behind the eye (w < 0): the old renderer dropped
        // the whole triangle; clipping keeps the in-front part.
        // A floor triangle under a 90° camera reaching behind the eye.
        let proj = crate::camera::perspective(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 10.0);
        let c = |p: [f32; 3]| crate::math::mat4_mul_vec4(&proj, [p[0], p[1], p[2], 1.0]);
        let tri = [
            c([-1.0, -0.5, -2.0]),
            c([1.0, -0.5, -2.0]),
            c([0.0, -0.5, 1.0]),
        ];
        assert!(tri[2][3] < 0.0, "third vertex behind the eye");
        let mut st = Vec::new();
        setup_triangle(tri, 0, Cull::None, false, 32, 32, &mut st);
        assert!(!st.is_empty());
        let mut n = 0;
        for t in &st {
            t.raster_rows(0, 32, |_, _, z, b| {
                n += 1;
                assert!((-1.0..=1.0).contains(&z));
                assert!((b[0] + b[1] + b[2] - 1.0).abs() < 1e-4);
            });
        }
        assert!(n > 0);
    }

    #[test]
    fn perspective_correct_barycentrics() {
        // Vertex 2 far away (w = 4): the screen midpoint of edge 0-2
        // is not the attribute midpoint.
        let tri = [
            [-1.0, -1.0, 0.0, 1.0],
            [1.0, -1.0, 0.0, 1.0],
            [-4.0, 4.0, 0.0, 4.0],
        ];
        let mut st = Vec::new();
        setup_triangle(tri, 0, Cull::None, false, 100, 100, &mut st);
        let t = &st[0];
        // Screen midpoint between v0 (0,100) and v2 (0,0) at x≈0.5.
        let b = t.bary_at(0.5, 50.0);
        // Attribute = 1/w-weighted: b2 = (0.5/4)/(0.5/1 + 0.5/4) = 0.2.
        assert!((b[2] - 0.2).abs() < 0.02, "{b:?}");
    }

    #[test]
    fn line_clipping_and_points() {
        assert!(setup_line([-2.0, 0.0, 0.0, 1.0], [-3.0, 0.0, 0.0, 1.0], 8, 8).is_none());
        let l = setup_line([-2.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], 8, 8).unwrap();
        assert!((l.t[0] - 0.5).abs() < 1e-6);
        assert!(setup_point([0.0, 0.0, 2.0, 1.0], 8, 8).is_none());
        assert!(setup_point([0.0, 0.0, 0.0, 1.0], 8, 8).is_some());
    }

    #[test]
    fn par_bands_visits_every_row_once() {
        let mut buf = vec![0u8; 300 * 301];
        par_bands(&mut buf, 300, |_, _, rows| {
            for v in rows.iter_mut() {
                *v += 1;
            }
        });
        assert!(buf.iter().all(|&v| v == 1));
    }
}
