//! Output image surface produced by a [`crate::Renderer`].

/// Packed RGBA8 raster image. The buffer is the renderer's contract
/// with downstream encoders (PNG / JPEG / OpenEXR / video) — every
/// backend, regardless of internal precision (`f32` framebuffer for
/// scanline, `f64` for path-tracer), commits to this shape on output.
///
/// `stride` is bytes per row; for the default RGBA8 layout it equals
/// `width * 4`. A future backend may emit RGB24 (`width * 3`), which
/// is why the field is carried explicitly rather than computed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Tightly-packed pixel bytes. `pixels.len() == height as usize * stride`.
    pub pixels: Vec<u8>,
    /// Bytes per row. `width * 4` for RGBA8, `width * 3` for RGB24.
    pub stride: usize,
}

impl RgbaImage {
    /// Allocate a `width × height` RGBA8 image filled with the given
    /// `[R, G, B, A]` colour. Useful for tests + as the renderer's
    /// initial framebuffer clear.
    #[must_use]
    pub fn filled(width: u32, height: u32, rgba: [u8; 4]) -> Self {
        let stride = (width as usize) * 4;
        let mut pixels = Vec::with_capacity(stride * (height as usize));
        for _ in 0..(width as usize) * (height as usize) {
            pixels.extend_from_slice(&rgba);
        }
        Self {
            width,
            height,
            pixels,
            stride,
        }
    }

    /// Return the four-byte pixel at `(x, y)`. `None` when out of bounds.
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let base = (y as usize) * self.stride + (x as usize) * 4;
        Some([
            self.pixels[base],
            self.pixels[base + 1],
            self.pixels[base + 2],
            self.pixels[base + 3],
        ])
    }

    /// Overwrite the pixel at `(x, y)` with `rgba`. Returns `true` on a
    /// successful write, `false` when the coordinate is outside the
    /// image (in which case `pixels` is left untouched).
    ///
    /// This mirrors [`RgbaImage::pixel`] as a typed setter so callers
    /// stitching post-processed pixels into the renderer's output don't
    /// have to recompute `stride`-aware byte offsets by hand.
    pub fn set_pixel(&mut self, x: u32, y: u32, rgba: [u8; 4]) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        let base = (y as usize) * self.stride + (x as usize) * 4;
        self.pixels[base] = rgba[0];
        self.pixels[base + 1] = rgba[1];
        self.pixels[base + 2] = rgba[2];
        self.pixels[base + 3] = rgba[3];
        true
    }

    /// Total pixel count (`width * height` as `u64`, widened to avoid
    /// overflow on hypothetical >4 G-pixel framebuffers).
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        (self.width as u64) * (self.height as u64)
    }

    /// `true` when either dimension is zero — no pixels addressable.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Iterate the image as packed `[u8; 4]` pixels in row-major order.
    ///
    /// Walks the buffer stride-aware so it remains correct on a future
    /// `RgbaImage` with `stride > width * 4` (padded rows). Skips any
    /// trailing bytes that don't make up a whole RGBA pixel — the
    /// renderer's own output never has those, but the iterator stays
    /// defensive against caller-constructed buffers.
    pub fn pixels_rgba(&self) -> impl Iterator<Item = [u8; 4]> + '_ {
        let width = self.width as usize;
        self.rows().flat_map(move |row| {
            row.chunks_exact(4)
                .take(width)
                .map(|p| [p[0], p[1], p[2], p[3]])
        })
    }

    /// Iterate the image one row at a time, yielding `&[u8]` slices of
    /// length `stride`. Useful for streaming the framebuffer into a
    /// downstream encoder that wants per-row callbacks (PNG IDAT walker,
    /// JPEG MCU feeder).
    ///
    /// A zero-width / zero-height / zero-stride image yields no rows.
    pub fn rows(&self) -> impl Iterator<Item = &[u8]> + '_ {
        let stride = self.stride;
        let height = self.height as usize;
        // `slice::chunks` panics on a zero chunk size, and zero-height
        // images have nothing to walk anyway — short-circuit both
        // shapes through an empty slice + a `chunks(1)` walker that
        // produces no items because the slice is empty.
        let safe_stride = stride.max(1);
        let usable = stride.saturating_mul(height);
        let buf = if stride == 0 || height == 0 {
            &[][..]
        } else {
            &self.pixels[..usable.min(self.pixels.len())]
        };
        buf.chunks(safe_stride).take(height)
    }
}

// ---------------------------------------------------------------------
// SSAA downsample — shared by every backend.
// ---------------------------------------------------------------------

/// Box-filter `src` (which is `aa × dst_w` by `aa × dst_h`) down to
/// `dst_w × dst_h`. Each output pixel averages `aa²` source pixels in
/// straight linear-byte space — no gamma round-trip. Backends render
/// at the supersampled resolution and hand the result here.
pub(crate) fn downsample_box(src: &RgbaImage, dst_w: u32, dst_h: u32, aa: u32) -> RgbaImage {
    let aa = aa.max(1);
    let aa_us = aa as usize;
    let dst_w_us = dst_w as usize;
    let dst_h_us = dst_h as usize;
    let src_stride = src.stride;
    let mut pixels = Vec::with_capacity(dst_w_us * dst_h_us * 4);
    let div = (aa_us * aa_us) as u32;
    for dy in 0..dst_h_us {
        let sy0 = dy * aa_us;
        for dx in 0..dst_w_us {
            let sx0 = dx * aa_us;
            let mut acc = [0u32; 4];
            for j in 0..aa_us {
                let row_base = (sy0 + j) * src_stride + sx0 * 4;
                for i in 0..aa_us {
                    let p = row_base + i * 4;
                    acc[0] += src.pixels[p] as u32;
                    acc[1] += src.pixels[p + 1] as u32;
                    acc[2] += src.pixels[p + 2] as u32;
                    acc[3] += src.pixels[p + 3] as u32;
                }
            }
            pixels.push((acc[0] / div) as u8);
            pixels.push((acc[1] / div) as u8);
            pixels.push((acc[2] / div) as u8);
            pixels.push((acc[3] / div) as u8);
        }
    }
    RgbaImage {
        width: dst_w,
        height: dst_h,
        stride: dst_w_us * 4,
        pixels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filled_produces_expected_shape() {
        let img = RgbaImage::filled(4, 2, [10, 20, 30, 40]);
        assert_eq!(img.width, 4);
        assert_eq!(img.height, 2);
        assert_eq!(img.stride, 16);
        assert_eq!(img.pixels.len(), 32);
        assert_eq!(img.pixel(0, 0), Some([10, 20, 30, 40]));
        assert_eq!(img.pixel(3, 1), Some([10, 20, 30, 40]));
        assert_eq!(img.pixel(4, 0), None);
        assert_eq!(img.pixel(0, 2), None);
    }

    #[test]
    fn set_pixel_inside_bounds_updates_byte_quad() {
        let mut img = RgbaImage::filled(3, 2, [0, 0, 0, 255]);
        assert!(img.set_pixel(1, 0, [11, 22, 33, 44]));
        assert_eq!(img.pixel(1, 0), Some([11, 22, 33, 44]));
        // Neighbours untouched.
        assert_eq!(img.pixel(0, 0), Some([0, 0, 0, 255]));
        assert_eq!(img.pixel(2, 0), Some([0, 0, 0, 255]));
        assert_eq!(img.pixel(1, 1), Some([0, 0, 0, 255]));
    }

    #[test]
    fn set_pixel_out_of_bounds_returns_false_and_no_op() {
        let mut img = RgbaImage::filled(2, 2, [7, 8, 9, 10]);
        let before = img.pixels.clone();
        assert!(!img.set_pixel(2, 0, [1, 1, 1, 1]));
        assert!(!img.set_pixel(0, 2, [1, 1, 1, 1]));
        assert!(!img.set_pixel(u32::MAX, 0, [1, 1, 1, 1]));
        assert_eq!(img.pixels, before);
    }

    #[test]
    fn pixel_count_and_is_empty() {
        let zero = RgbaImage::filled(0, 5, [0; 4]);
        assert_eq!(zero.pixel_count(), 0);
        assert!(zero.is_empty());

        let one = RgbaImage::filled(1, 1, [0; 4]);
        assert_eq!(one.pixel_count(), 1);
        assert!(!one.is_empty());

        let small = RgbaImage::filled(640, 480, [0; 4]);
        assert_eq!(small.pixel_count(), 640 * 480);
        assert!(!small.is_empty());
    }

    #[test]
    fn pixels_rgba_iterates_row_major() {
        // Row 0 = pure red, row 1 = pure green; verify ordering.
        let img = RgbaImage {
            width: 2,
            height: 2,
            stride: 8,
            pixels: vec![
                255, 0, 0, 255, 255, 0, 0, 255, // row 0
                0, 255, 0, 255, 0, 255, 0, 255, // row 1
            ],
        };
        let collected: Vec<[u8; 4]> = img.pixels_rgba().collect();
        assert_eq!(collected.len(), 4);
        assert_eq!(collected[0], [255, 0, 0, 255]);
        assert_eq!(collected[1], [255, 0, 0, 255]);
        assert_eq!(collected[2], [0, 255, 0, 255]);
        assert_eq!(collected[3], [0, 255, 0, 255]);
    }

    #[test]
    fn pixels_rgba_skips_row_padding() {
        // stride = 12 = 2 pixels * 4 bytes + 4 bytes of trailing
        // per-row padding the iterator must skip.
        let img = RgbaImage {
            width: 2,
            height: 2,
            stride: 12,
            pixels: vec![
                1, 2, 3, 4, 5, 6, 7, 8, 0xFF, 0xFF, 0xFF, 0xFF, // row 0 + pad
                9, 10, 11, 12, 13, 14, 15, 16, 0xFF, 0xFF, 0xFF, 0xFF, // row 1 + pad
            ],
        };
        let collected: Vec<[u8; 4]> = img.pixels_rgba().collect();
        assert_eq!(
            collected,
            vec![
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]
            ]
        );
    }

    #[test]
    fn rows_yields_one_slice_per_row_with_stride_length() {
        let img = RgbaImage::filled(3, 4, [0xAA, 0xBB, 0xCC, 0xDD]);
        let rows: Vec<&[u8]> = img.rows().collect();
        assert_eq!(rows.len(), 4);
        for row in rows {
            assert_eq!(row.len(), img.stride);
            assert_eq!(row.len(), 12);
            // Each pixel is [0xAA, 0xBB, 0xCC, 0xDD].
            for chunk in row.chunks_exact(4) {
                assert_eq!(chunk, &[0xAA, 0xBB, 0xCC, 0xDD]);
            }
        }
    }

    #[test]
    fn rows_on_empty_image_yields_nothing() {
        let empty = RgbaImage::filled(0, 0, [0; 4]);
        assert_eq!(empty.rows().count(), 0);
        assert_eq!(empty.pixels_rgba().count(), 0);
    }

    #[test]
    fn pixels_rgba_count_matches_pixel_count() {
        let img = RgbaImage::filled(7, 5, [0; 4]);
        assert_eq!(img.pixels_rgba().count() as u64, img.pixel_count());
    }

    #[test]
    fn downsample_box_averages_uniform_field() {
        let src = RgbaImage::filled(4, 4, [200, 100, 50, 255]);
        let dst = downsample_box(&src, 2, 2, 2);
        assert_eq!(dst.width, 2);
        assert_eq!(dst.height, 2);
        for px in dst.pixels.chunks_exact(4) {
            assert_eq!(px, &[200, 100, 50, 255]);
        }
    }

    #[test]
    fn downsample_box_averages_split_field() {
        let src = RgbaImage {
            width: 2,
            height: 2,
            stride: 8,
            pixels: vec![
                0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255,
            ],
        };
        let dst = downsample_box(&src, 1, 1, 2);
        assert_eq!(dst.width, 1);
        assert_eq!(dst.height, 1);
        assert_eq!(dst.pixels[0], 127);
        assert_eq!(dst.pixels[1], 127);
        assert_eq!(dst.pixels[2], 127);
        assert_eq!(dst.pixels[3], 255);
    }
}
