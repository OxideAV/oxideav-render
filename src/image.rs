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
}
