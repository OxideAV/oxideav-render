//! Texture resolution (encoded bytes → decoded texels) and CPU
//! texture sampling honouring the glTF 2.0 [`Sampler`] state.
//!
//! ## Resolution
//!
//! A [`oxideav_mesh3d::Texture`] points at its pixels through
//! [`ImageData`]:
//!
//! * `Source(Arc<dyn AssetSource>)` — encoded bytes (PNG / JPEG / …)
//!   plus an optional MIME type;
//! * `External { uri, mime }` — a URI the scene decoder did not
//!   resolve;
//! * `Embedded(VideoFrame)` (mesh3d `registry` feature) — already
//!   decoded pixels.
//!
//! This crate deliberately does not depend on any image codec. Callers
//! plug decoding in through the [`TextureResolver`] trait; under the
//! `registry` feature [`ImageTextureResolver`] decodes through the
//! `oxideav-image` gateway over an `oxideav_core::RuntimeContext`
//! (whatever image containers/codecs the caller registered, e.g. via
//! `oxideav-meta`; float sources such as EXR stay linear `f32`). One format is
//! understood natively, with no resolver: [`RAW_RGBA8_MIME`], a
//! trivial uncompressed container ([`encode_raw_rgba8`]) used by the
//! procedural [`crate::testscenes`] so textured scenes render in every
//! build.
//!
//! [`TextureCache`] memoises decoded images across frames (keyed by
//! asset identity / URI), so an animated render decodes each texture
//! once.
//!
//! ## Sampling
//!
//! [`TextureData`] keeps the decoded image plus two lazily-built mip
//! chains: one sRGB-decoded (colour textures — `baseColorTexture`,
//! `emissiveTexture`, per glTF 2.0 §3.9.2 / §3.9.5) and one linear
//! (data textures — normal, metallic-roughness, occlusion). Mip levels
//! are 2×2 box-filtered in linear light. [`PreparedTexture::sample_grad`]
//! picks a level of detail from screen-space UV derivatives (the
//! isotropic `log2(max(|∂uv/∂x|, |∂uv/∂y|) · size)` rule of
//! Williams 1983, "Pyramidal Parametrics") and filters per the glTF
//! sampler's `magFilter` / `minFilter` (nearest / bilinear, with
//! nearest or linear (trilinear) mip selection). Wrap modes
//! (`REPEAT` / `CLAMP_TO_EDGE` / `MIRRORED_REPEAT`) are applied on
//! integer texel coordinates so bilinear footprints wrap seamlessly.
//! UV `(0, 0)` addresses the first (top-left) texel, per glTF §3.8.

use std::collections::HashMap;
use std::io::Read;
use std::sync::{Arc, OnceLock};

use oxideav_mesh3d::{AssetSource, ImageData, MagFilter, MinFilter, Sampler, Texture, WrapMode};

use crate::hdr::{srgb_u8_lut, HdrImage};

/// MIME type of the crate's trivial uncompressed RGBA8 container,
/// decoded natively without any [`TextureResolver`]. Layout: the 8
/// magic bytes `OXRGBA8\0`, width (`u32` LE), height (`u32` LE), then
/// `width * height * 4` bytes of row-major RGBA8.
pub const RAW_RGBA8_MIME: &str = "image/x-oxideav-rgba8";

const RAW_MAGIC: &[u8; 8] = b"OXRGBA8\0";

/// Pack `rgba` (row-major RGBA8, `width * height * 4` bytes) into the
/// [`RAW_RGBA8_MIME`] container.
pub fn encode_raw_rgba8(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + rgba.len());
    out.extend_from_slice(RAW_MAGIC);
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(rgba);
    out
}

/// Parse a [`RAW_RGBA8_MIME`] payload. `None` when malformed.
pub fn decode_raw_rgba8(bytes: &[u8]) -> Option<DecodedImage> {
    if bytes.len() < 16 || &bytes[..8] != RAW_MAGIC {
        return None;
    }
    let w = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
    let h = u32::from_le_bytes(bytes[12..16].try_into().ok()?);
    let n = (w as usize).checked_mul(h as usize)?.checked_mul(4)?;
    let data = bytes.get(16..16 + n)?;
    DecodedImage::from_rgba8(w, h, data.to_vec())
}

/// Decoded texels handed back by a [`TextureResolver`].
#[derive(Debug, Clone, PartialEq)]
pub enum DecodedPixels {
    /// Row-major RGBA8, `width * height * 4` bytes. Colour channels
    /// are interpreted per the binding (sRGB for colour textures,
    /// linear for data textures); alpha is always linear.
    Rgba8(Vec<u8>),
    /// Row-major RGBA `f32`, `width * height * 4` floats, already
    /// linear (HDR sources — EXR / Radiance). Never sRGB-decoded.
    RgbaF32(Vec<f32>),
}

/// A decoded 2-D image.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedImage {
    /// Width in texels (`>= 1`).
    pub width: u32,
    /// Height in texels (`>= 1`).
    pub height: u32,
    /// Texel payload.
    pub pixels: DecodedPixels,
}

impl DecodedImage {
    /// Wrap an RGBA8 buffer, validating its length. `None` for a zero
    /// dimension or a length mismatch.
    pub fn from_rgba8(width: u32, height: u32, rgba: Vec<u8>) -> Option<Self> {
        let n = (width as usize)
            .checked_mul(height as usize)?
            .checked_mul(4)?;
        (width > 0 && height > 0 && rgba.len() == n).then_some(Self {
            width,
            height,
            pixels: DecodedPixels::Rgba8(rgba),
        })
    }

    /// Wrap a linear RGBA `f32` buffer, validating its length.
    pub fn from_rgba_f32(width: u32, height: u32, rgba: Vec<f32>) -> Option<Self> {
        let n = (width as usize)
            .checked_mul(height as usize)?
            .checked_mul(4)?;
        (width > 0 && height > 0 && rgba.len() == n).then_some(Self {
            width,
            height,
            pixels: DecodedPixels::RgbaF32(rgba),
        })
    }
}

/// Pluggable image decoder used to turn a texture's encoded bytes
/// into texels. Implementations must be thread-safe; they are shared
/// by the renderer across frames.
pub trait TextureResolver: Send + Sync {
    /// Decode `bytes` (an encoded image — PNG, JPEG, …). `mime` is the
    /// asset's declared MIME type when known. Return `None` when the
    /// format is unsupported or the payload is corrupt; the texture is
    /// then treated as absent (the material factor alone applies).
    fn decode(&self, bytes: &[u8], mime: Option<&str>) -> Option<DecodedImage>;

    /// Fetch the bytes of an [`ImageData::External`] URI the scene
    /// decoder left unresolved. The default refuses (returns `None`).
    fn fetch_external(&self, uri: &str) -> Option<Vec<u8>> {
        let _ = uri;
        None
    }
}

/// Resolver that decodes nothing (only [`RAW_RGBA8_MIME`] payloads,
/// which the cache handles itself, resolve). The default for
/// renderers built with [`crate::make_renderer`].
#[derive(Debug, Default, Clone, Copy)]
pub struct NoTextureResolver;

impl TextureResolver for NoTextureResolver {
    fn decode(&self, _bytes: &[u8], _mime: Option<&str>) -> Option<DecodedImage> {
        None
    }
}

/// Guess a MIME type from a URI's extension.
pub fn mime_from_uri(uri: &str) -> Option<&'static str> {
    let ext = uri.rsplit('.').next()?.to_ascii_lowercase();
    Some(match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "ktx2" => "image/ktx2",
        "bmp" => "image/bmp",
        "gif" => "image/gif",
        "tif" | "tiff" => "image/tiff",
        "exr" => "image/x-exr",
        "hdr" => "image/vnd.radiance",
        "tga" => "image/x-tga",
        _ => return None,
    })
}

// ---------------------------------------------------------------------
// Texture data + mip chains.
// ---------------------------------------------------------------------

/// How a texture's colour channels are interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    /// sRGB-encoded colour (base colour, emissive): decoded to linear
    /// before filtering.
    Srgb,
    /// Linear data (normal, metallic-roughness, occlusion).
    Linear,
}

/// One mip level: linear-light RGBA `f32` texels.
#[derive(Debug, Clone, PartialEq)]
pub struct MipLevel {
    /// Level width.
    pub width: u32,
    /// Level height.
    pub height: u32,
    /// Row-major texels.
    pub texels: Vec<[f32; 4]>,
}

impl MipLevel {
    #[inline]
    fn fetch(&self, x: i64, y: i64, ws: WrapMode, wt: WrapMode) -> [f32; 4] {
        let xi = wrap_index(x, self.width as i64, ws);
        let yi = wrap_index(y, self.height as i64, wt);
        self.texels[yi * self.width as usize + xi]
    }
}

fn wrap_index(i: i64, n: i64, mode: WrapMode) -> usize {
    (match mode {
        WrapMode::Repeat => i.rem_euclid(n),
        WrapMode::ClampToEdge => i.clamp(0, n - 1),
        WrapMode::MirroredRepeat => {
            let m = i.rem_euclid(2 * n);
            if m >= n {
                2 * n - 1 - m
            } else {
                m
            }
        }
    }) as usize
}

/// A decoded image plus its lazily-built sRGB and linear mip chains.
/// Shared (via `Arc`) between every texture that references the same
/// asset and cached across frames by [`TextureCache`].
#[derive(Debug)]
pub struct TextureData {
    image: DecodedImage,
    srgb: OnceLock<Vec<MipLevel>>,
    linear: OnceLock<Vec<MipLevel>>,
}

impl TextureData {
    /// Wrap a decoded image.
    pub fn new(image: DecodedImage) -> Self {
        Self {
            image,
            srgb: OnceLock::new(),
            linear: OnceLock::new(),
        }
    }

    /// The decoded source image.
    pub fn image(&self) -> &DecodedImage {
        &self.image
    }

    /// Full mip chain (level 0 = full resolution) for `space`, built
    /// on first use. Thread-safe.
    pub fn mips(&self, space: ColorSpace) -> &[MipLevel] {
        let cell = match space {
            ColorSpace::Srgb => &self.srgb,
            ColorSpace::Linear => &self.linear,
        };
        cell.get_or_init(|| build_mips(&self.image, space))
    }
}

fn build_mips(img: &DecodedImage, space: ColorSpace) -> Vec<MipLevel> {
    let lut = srgb_u8_lut();
    let texels: Vec<[f32; 4]> = match &img.pixels {
        DecodedPixels::Rgba8(b) => b
            .chunks_exact(4)
            .map(|p| {
                let c = |v: u8| match space {
                    ColorSpace::Srgb => lut[v as usize],
                    ColorSpace::Linear => v as f32 / 255.0,
                };
                [c(p[0]), c(p[1]), c(p[2]), p[3] as f32 / 255.0]
            })
            .collect(),
        DecodedPixels::RgbaF32(f) => f
            .chunks_exact(4)
            .map(|p| [p[0], p[1], p[2], p[3]])
            .collect(),
    };
    let mut levels = vec![MipLevel {
        width: img.width,
        height: img.height,
        texels,
    }];
    loop {
        let prev = levels.last().expect("non-empty");
        if prev.width == 1 && prev.height == 1 {
            break;
        }
        let w = (prev.width / 2).max(1);
        let h = (prev.height / 2).max(1);
        let mut texels = Vec::with_capacity(w as usize * h as usize);
        for y in 0..h {
            for x in 0..w {
                let x0 = (2 * x).min(prev.width - 1) as usize;
                let x1 = (2 * x + 1).min(prev.width - 1) as usize;
                let y0 = (2 * y).min(prev.height - 1) as usize;
                let y1 = (2 * y + 1).min(prev.height - 1) as usize;
                let pw = prev.width as usize;
                let mut acc = [0.0f32; 4];
                for idx in [y0 * pw + x0, y0 * pw + x1, y1 * pw + x0, y1 * pw + x1] {
                    let t = prev.texels[idx];
                    for c in 0..4 {
                        acc[c] += t[c];
                    }
                }
                texels.push([acc[0] * 0.25, acc[1] * 0.25, acc[2] * 0.25, acc[3] * 0.25]);
            }
        }
        levels.push(MipLevel {
            width: w,
            height: h,
            texels,
        });
    }
    levels
}

/// A texture ready to sample: shared texel data plus the glTF sampler
/// state of the [`Texture`] that referenced it.
#[derive(Debug, Clone)]
pub struct PreparedTexture {
    /// Decoded texels + mip chains.
    pub data: Arc<TextureData>,
    /// glTF sampler (filters + wrap modes).
    pub sampler: Sampler,
}

impl PreparedTexture {
    /// Sample at `uv` with an explicit level of detail (`lod <= 0` ⇒
    /// magnification filter on level 0). Returns linear RGBA.
    pub fn sample_lod(&self, uv: [f32; 2], lod: f32, space: ColorSpace) -> [f32; 4] {
        let mips = self.data.mips(space);
        let s = &self.sampler;
        if lod.is_nan() || lod <= 0.0 {
            let lin = s.effective_mag_filter() == MagFilter::Linear;
            return filter_level(&mips[0], uv, lin, s.wrap_s, s.wrap_t);
        }
        let min = s.effective_min_filter();
        let lin = min.base_filter() == MagFilter::Linear;
        let max_level = (mips.len() - 1) as f32;
        match min {
            MinFilter::Nearest | MinFilter::Linear => {
                filter_level(&mips[0], uv, lin, s.wrap_s, s.wrap_t)
            }
            MinFilter::NearestMipNearest | MinFilter::LinearMipNearest => {
                let l = lod.round().clamp(0.0, max_level) as usize;
                filter_level(&mips[l], uv, lin, s.wrap_s, s.wrap_t)
            }
            MinFilter::NearestMipLinear | MinFilter::LinearMipLinear => {
                let l = lod.clamp(0.0, max_level);
                let l0 = l.floor() as usize;
                let l1 = (l0 + 1).min(mips.len() - 1);
                let f = l - l0 as f32;
                let a = filter_level(&mips[l0], uv, lin, s.wrap_s, s.wrap_t);
                if f <= 0.0 || l0 == l1 {
                    return a;
                }
                let b = filter_level(&mips[l1], uv, lin, s.wrap_s, s.wrap_t);
                lerp4(a, b, f)
            }
        }
    }

    /// Sample at `uv` choosing the level of detail from the
    /// screen-space derivatives `duv_dx` / `duv_dy` (UV change per
    /// output pixel). Pass zero derivatives to force level 0.
    pub fn sample_grad(
        &self,
        uv: [f32; 2],
        duv_dx: [f32; 2],
        duv_dy: [f32; 2],
        space: ColorSpace,
    ) -> [f32; 4] {
        let img = self.data.image();
        let (w, h) = (img.width as f32, img.height as f32);
        let lx = (duv_dx[0] * w).hypot(duv_dx[1] * h);
        let ly = (duv_dy[0] * w).hypot(duv_dy[1] * h);
        let rho = lx.max(ly);
        let lod = if rho.is_finite() && rho > 0.0 {
            rho.log2()
        } else {
            0.0
        };
        self.sample_lod(uv, lod, space)
    }
}

fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

fn filter_level(
    lvl: &MipLevel,
    uv: [f32; 2],
    linear: bool,
    ws: WrapMode,
    wt: WrapMode,
) -> [f32; 4] {
    let u = if uv[0].is_finite() { uv[0] } else { 0.0 };
    let v = if uv[1].is_finite() { uv[1] } else { 0.0 };
    // Keep the integer conversion well inside i64 for absurd UVs.
    let u = u.clamp(-1.0e6, 1.0e6);
    let v = v.clamp(-1.0e6, 1.0e6);
    let x = u * lvl.width as f32;
    let y = v * lvl.height as f32;
    if !linear {
        return lvl.fetch(x.floor() as i64, y.floor() as i64, ws, wt);
    }
    let xf = x - 0.5;
    let yf = y - 0.5;
    let x0 = xf.floor();
    let y0 = yf.floor();
    let fx = xf - x0;
    let fy = yf - y0;
    let (x0, y0) = (x0 as i64, y0 as i64);
    let a = lvl.fetch(x0, y0, ws, wt);
    let b = lvl.fetch(x0 + 1, y0, ws, wt);
    let c = lvl.fetch(x0, y0 + 1, ws, wt);
    let d = lvl.fetch(x0 + 1, y0 + 1, ws, wt);
    lerp4(lerp4(a, b, fx), lerp4(c, d, fx), fy)
}

// ---------------------------------------------------------------------
// Cache.
// ---------------------------------------------------------------------

#[derive(Hash, PartialEq, Eq)]
enum CacheKey {
    /// `Arc` data pointer of an asset source (the `Arc` itself is
    /// retained in `held`, so the address cannot be recycled while
    /// cached).
    Source(usize),
    External(String),
}

/// Memoising texture loader: resolves [`Texture`]s to
/// [`TextureData`] through a [`TextureResolver`], caching results
/// (including failures) by asset identity so repeated frames of the
/// same scene decode each image once.
pub struct TextureCache {
    resolver: Arc<dyn TextureResolver>,
    entries: HashMap<CacheKey, Option<Arc<TextureData>>>,
    held: Vec<Arc<dyn AssetSource>>,
}

impl std::fmt::Debug for TextureCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextureCache")
            .field("entries", &self.entries.len())
            .finish()
    }
}

impl Default for TextureCache {
    fn default() -> Self {
        Self::new(Arc::new(NoTextureResolver))
    }
}

impl TextureCache {
    /// Cache decoding through `resolver`.
    pub fn new(resolver: Arc<dyn TextureResolver>) -> Self {
        Self {
            resolver,
            entries: HashMap::new(),
            held: Vec::new(),
        }
    }

    /// Replace the resolver (clears the cache).
    pub fn set_resolver(&mut self, resolver: Arc<dyn TextureResolver>) {
        self.resolver = resolver;
        self.clear();
    }

    /// Drop every cached image.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.held.clear();
    }

    /// Number of cached entries (successful or failed).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when nothing is cached.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Resolve `tex` into a sampleable [`PreparedTexture`]. `None`
    /// when the image cannot be obtained or decoded.
    pub fn prepare(&mut self, tex: &Texture) -> Option<PreparedTexture> {
        let data = self.resolve_image(&tex.image)?;
        Some(PreparedTexture {
            data,
            sampler: tex.sampler,
        })
    }

    /// Resolve an [`ImageData`] to decoded, cached texel data.
    pub fn resolve_image(&mut self, image: &ImageData) -> Option<Arc<TextureData>> {
        match image {
            ImageData::Source(src) => {
                let key = CacheKey::Source(Arc::as_ptr(src) as *const () as usize);
                if let Some(e) = self.entries.get(&key) {
                    return e.clone();
                }
                let decoded = load_source(src.as_ref(), self.resolver.as_ref());
                let entry = decoded.map(|d| Arc::new(TextureData::new(d)));
                self.held.push(Arc::clone(src));
                self.entries.insert(key, entry.clone());
                entry
            }
            ImageData::External { uri, mime } => {
                let key = CacheKey::External(uri.clone());
                if let Some(e) = self.entries.get(&key) {
                    return e.clone();
                }
                let decoded = self.resolver.fetch_external(uri).and_then(|bytes| {
                    let mime = mime.as_deref().or_else(|| mime_from_uri(uri));
                    decode_bytes(&bytes, mime, self.resolver.as_ref())
                });
                let entry = decoded.map(|d| Arc::new(TextureData::new(d)));
                self.entries.insert(key, entry.clone());
                entry
            }
            #[allow(unreachable_patterns)]
            other => embedded_image(other).map(|d| Arc::new(TextureData::new(d))),
        }
    }
}

fn load_source(src: &dyn AssetSource, resolver: &dyn TextureResolver) -> Option<DecodedImage> {
    let mut bytes = Vec::new();
    if let Some(raw) = src.raw_storage() {
        if raw.scheme.is_empty() || raw.scheme == "stored" {
            bytes.extend_from_slice(raw.bytes);
        }
    }
    if bytes.is_empty() {
        src.open().ok()?.read_to_end(&mut bytes).ok()?;
    }
    decode_bytes(&bytes, src.mime(), resolver)
}

fn decode_bytes(
    bytes: &[u8],
    mime: Option<&str>,
    resolver: &dyn TextureResolver,
) -> Option<DecodedImage> {
    if mime == Some(RAW_RGBA8_MIME) || bytes.starts_with(RAW_MAGIC) {
        return decode_raw_rgba8(bytes);
    }
    resolver.decode(bytes, mime)
}

/// `ImageData::Embedded` carries a `VideoFrame`, which records no
/// width / pixel format. Best effort: a single packed plane is read
/// as RGBA8 with `width = stride / 4`.
#[cfg(feature = "registry")]
fn embedded_image(image: &ImageData) -> Option<DecodedImage> {
    let ImageData::Embedded(frame) = image else {
        return None;
    };
    let plane = frame.planes.first()?;
    if plane.stride == 0 || plane.stride % 4 != 0 {
        return None;
    }
    let w = plane.stride / 4;
    let h = plane.data.len() / plane.stride;
    DecodedImage::from_rgba8(w as u32, h as u32, plane.data[..w * h * 4].to_vec())
}

#[cfg(not(feature = "registry"))]
fn embedded_image(_image: &ImageData) -> Option<DecodedImage> {
    None
}

// ---------------------------------------------------------------------
// Gateway-backed resolver (oxideav-image).
// ---------------------------------------------------------------------

/// Probe extension hint for a MIME type (`image/png` → `png`).
///
/// Known glTF / web image types map explicitly; any other `image/*`
/// subtype is passed through (`image/avif` → `avif`, `image/heic` →
/// `heic`, `image/jxl` → `jxl`), with vendor / experimental prefixes
/// (`x-`, `vnd.`) dropped. `None` for non-image types.
#[cfg(feature = "registry")]
pub fn ext_hint_from_mime(mime: &str) -> Option<String> {
    let m = mime.trim().to_ascii_lowercase();
    let m = m.split(';').next().unwrap_or("").trim();
    Some(
        match m {
            "image/png" => "png",
            "image/jpeg" | "image/jpg" | "image/pjpeg" => "jpg",
            "image/webp" => "webp",
            "image/bmp" | "image/x-bmp" | "image/x-ms-bmp" => "bmp",
            "image/gif" => "gif",
            "image/tiff" | "image/tiff-fx" => "tiff",
            "image/ktx2" => "ktx2",
            "image/x-exr" | "image/exr" | "image/aces" => "exr",
            "image/vnd.radiance" | "image/x-hdr" => "hdr",
            "image/x-tga" | "image/x-targa" | "image/targa" => "tga",
            "image/heic" | "image/heif" | "image/heic-sequence" | "image/heif-sequence" => "heic",
            "image/avif" | "image/avif-sequence" => "avif",
            "image/jxl" => "jxl",
            "image/jp2" | "image/jpx" | "image/jpm" => "jp2",
            "image/vnd.ms-photo" | "image/jxr" => "jxr",
            "image/vnd-ms.dds" | "image/vnd.ms-dds" | "image/x-dds" => "dds",
            "image/x-icon" | "image/vnd.microsoft.icon" => "ico",
            "image/x-portable-pixmap" | "image/x-portable-anymap" => "ppm",
            "image/x-pcx" | "image/vnd.zbrush.pcx" => "pcx",
            "image/qoi" | "image/x-qoi" => "qoi",
            "image/svg+xml" => "svg",
            "image/x-pict" | "image/pict" => "pict",
            "image/vnd.wap.wbmp" => "wbmp",
            other => {
                let sub = other.strip_prefix("image/")?;
                let sub = sub.strip_prefix("x-").unwrap_or(sub);
                let sub = sub.strip_prefix("vnd.").unwrap_or(sub);
                let sub: String = sub
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric())
                    .collect();
                if sub.is_empty() {
                    return None;
                }
                return Some(sub);
            }
        }
        .to_string(),
    )
}

/// Turn a decoded gateway picture into renderer texels.
///
/// * Float layouts (`is_float()`: EXR, Radiance, float TIFF) become
///   [`DecodedPixels::RgbaF32`] — scene-linear, never sRGB-decoded;
///   missing alpha is opaque.
/// * Everything else becomes [`DecodedPixels::Rgba8`] through the
///   gateway's `to_rgba8` (YCbCr → RGB with the stream's colour
///   signal, palettes expanded, alpha opaque when absent). Deeper
///   integer layouts (`Rgb48Le`, `Rgba64Le`, `Gray16Le`, 10 / 12-bit
///   YUV) are rounded to 8 bits — the sampler's colour-space rule
///   (sRGB for colour slots, linear for data slots) needs an encoded
///   integer source, and 8 bits is what the glTF texture slots are
///   specified against.
///
/// `None` when the gateway cannot reach either layout.
#[cfg(feature = "registry")]
pub fn decoded_from_image(img: &oxideav_image::Image) -> Option<DecodedImage> {
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return None;
    }
    if img.format().is_float() {
        let bytes = img.to_packed(oxideav_core::PixelFormat::RgbaF32Le).ok()?;
        let n = (w as usize).checked_mul(h as usize)?.checked_mul(4)?;
        if bytes.len() != n * 4 {
            return None;
        }
        let px: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        return DecodedImage::from_rgba_f32(w, h, px);
    }
    DecodedImage::from_rgba8(w, h, img.to_rgba8().ok()?)
}

impl DecodedImage {
    /// The texels as a scene-linear [`HdrImage`] — the input of
    /// [`crate::EnvironmentMap::new`]. `RgbaF32` texels copy straight
    /// across; `Rgba8` texels are sRGB-decoded (an 8-bit environment
    /// map is a colour image), alpha linear.
    pub fn to_hdr(&self) -> HdrImage {
        let pixels = match &self.pixels {
            DecodedPixels::RgbaF32(f) => f.clone(),
            DecodedPixels::Rgba8(b) => {
                let lut = srgb_u8_lut();
                let mut out = Vec::with_capacity(b.len());
                for p in b.chunks_exact(4) {
                    out.push(lut[p[0] as usize]);
                    out.push(lut[p[1] as usize]);
                    out.push(lut[p[2] as usize]);
                    out.push(p[3] as f32 / 255.0);
                }
                out
            }
        };
        HdrImage {
            width: self.width,
            height: self.height,
            pixels,
        }
    }
}

/// [`TextureResolver`] riding the [`oxideav-image`](oxideav_image)
/// gateway: `oxideav_image::decode_bytes_with` probes the bytes
/// through the caller's `oxideav_core::RuntimeContext` (extension hint
/// from the MIME type), decodes the primary picture, and
/// [`decoded_from_image`] converts it — RGBA8 for integer sources,
/// linear RGBA `f32` for float sources (EXR / Radiance environment
/// maps, float TIFF). Every image format registered in the context
/// (PNG, JPEG, WebP, HEIF / AVIF, TIFF, OpenEXR, … — all of
/// `oxideav_meta::register_all`) resolves; the crate itself links no
/// codec.
///
/// `External` URIs are read from the filesystem relative to
/// [`ImageTextureResolver::with_base_dir`] (absolute paths as-is);
/// `data:` / network URIs are refused. [`with_max_pixels`](Self::with_max_pixels)
/// bounds the decoded size of one texture (the gateway refuses before
/// decoding).
///
/// ```no_run
/// # use std::sync::Arc;
/// # use oxideav_render::{make_renderer, ImageTextureResolver, RenderBackend};
/// let mut ctx = oxideav_core::RuntimeContext::new();
/// // oxideav_meta::register_all(&mut ctx);  // or the few image crates you need
/// let resolver = Arc::new(ImageTextureResolver::new(Arc::new(ctx)).with_base_dir("assets/"));
/// let mut renderer = make_renderer(RenderBackend::PathTrace)?;
/// renderer.set_texture_resolver(resolver);
/// # Ok::<(), oxideav_render::Error>(())
/// ```
#[cfg(feature = "registry")]
#[derive(Clone)]
pub struct ImageTextureResolver {
    ctx: Arc<oxideav_core::RuntimeContext>,
    base_dir: Option<std::path::PathBuf>,
    max_pixels: Option<u64>,
}

#[cfg(feature = "registry")]
impl std::fmt::Debug for ImageTextureResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageTextureResolver")
            .field("base_dir", &self.base_dir)
            .field("max_pixels", &self.max_pixels)
            .finish()
    }
}

#[cfg(feature = "registry")]
impl ImageTextureResolver {
    /// Decode through the formats registered in `ctx`.
    pub fn new(ctx: Arc<oxideav_core::RuntimeContext>) -> Self {
        Self {
            ctx,
            base_dir: None,
            max_pixels: None,
        }
    }

    /// Resolve relative `External` URIs against `dir`.
    pub fn with_base_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.base_dir = Some(dir.into());
        self
    }

    /// Refuse textures whose decoded pixel count exceeds `n`
    /// (`oxideav_image::OpenOptions::max_pixels`; checked from the
    /// stream geometry before any pixel is decoded).
    pub fn with_max_pixels(mut self, n: u64) -> Self {
        self.max_pixels = Some(n);
        self
    }

    /// The runtime context textures decode through.
    pub fn context(&self) -> &Arc<oxideav_core::RuntimeContext> {
        &self.ctx
    }

    /// The directory relative `External` URIs resolve against.
    pub fn base_dir(&self) -> Option<&std::path::Path> {
        self.base_dir.as_deref()
    }

    /// Decode `bytes` and keep the gateway's error (the
    /// [`TextureResolver::decode`] impl discards it): `UnknownFormat`
    /// when no registered container probes the bytes, `NoImage` when
    /// the container holds no decodable picture, `LimitExceeded` over
    /// the pixel budget, `Unsupported` when no RGBA layout is
    /// reachable, `Core` for codec failures.
    pub fn decode_image(
        &self,
        bytes: &[u8],
        mime: Option<&str>,
    ) -> Result<DecodedImage, oxideav_image::ImageError> {
        let mut opts = oxideav_image::OpenOptions::new().with_max_frames(1);
        if let Some(ext) = mime.and_then(ext_hint_from_mime) {
            opts = opts.with_ext_hint(ext);
        }
        if let Some(n) = self.max_pixels {
            opts = opts.with_max_pixels(n);
        }
        let file = oxideav_image::decode_bytes_with(&self.ctx, bytes, &opts)?;
        decoded_from_image(file.primary()).ok_or_else(|| {
            oxideav_image::ImageError::Unsupported(format!(
                "{}x{} {:?} picture reaches neither RGBA8 nor RGBA f32",
                file.primary().width(),
                file.primary().height(),
                file.primary().format()
            ))
        })
    }

    /// Open an image file from disk and decode it like a texture (the
    /// extension is the probe hint). For environment maps:
    /// `resolver.open_image("sky.exr")?.to_hdr()`.
    pub fn open_image(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<DecodedImage, oxideav_image::ImageError> {
        let path = path.as_ref();
        let full = match (&self.base_dir, path.is_absolute()) {
            (Some(base), false) => base.join(path),
            _ => path.to_path_buf(),
        };
        let mut opts = oxideav_image::OpenOptions::new().with_max_frames(1);
        if let Some(n) = self.max_pixels {
            opts = opts.with_max_pixels(n);
        }
        let file = oxideav_image::open_with(&self.ctx, &full, &opts)?;
        decoded_from_image(file.primary()).ok_or_else(|| {
            oxideav_image::ImageError::Unsupported(format!(
                "{}: {:?} picture reaches neither RGBA8 nor RGBA f32",
                full.display(),
                file.primary().format()
            ))
        })
    }

    fn resolve_uri(&self, uri: &str) -> Option<std::path::PathBuf> {
        if uri.contains("://") || uri.starts_with("data:") {
            return None;
        }
        let path = std::path::Path::new(uri);
        Some(match (&self.base_dir, path.is_absolute()) {
            (Some(base), false) => base.join(path),
            _ => path.to_path_buf(),
        })
    }
}

#[cfg(feature = "registry")]
impl TextureResolver for ImageTextureResolver {
    fn decode(&self, bytes: &[u8], mime: Option<&str>) -> Option<DecodedImage> {
        self.decode_image(bytes, mime).ok()
    }

    fn fetch_external(&self, uri: &str) -> Option<Vec<u8>> {
        std::fs::read(self.resolve_uri(uri)?).ok()
    }
}

// ---------------------------------------------------------------------
// Registry-backed resolver (deprecated thin wrapper).
// ---------------------------------------------------------------------

/// Previous name of [`ImageTextureResolver`]; decodes through the same
/// gateway path. Kept for one release as a thin wrapper.
#[cfg(feature = "registry")]
#[deprecated(
    since = "0.0.6",
    note = "renamed to ImageTextureResolver (decodes through the oxideav-image gateway)"
)]
#[derive(Clone)]
pub struct RegistryTextureResolver {
    inner: ImageTextureResolver,
}

#[cfg(feature = "registry")]
#[allow(deprecated)]
impl std::fmt::Debug for RegistryTextureResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryTextureResolver")
            .field("base_dir", &self.inner.base_dir)
            .finish()
    }
}

#[cfg(feature = "registry")]
#[allow(deprecated)]
impl RegistryTextureResolver {
    /// Decode through `ctx`.
    pub fn new(ctx: Arc<oxideav_core::RuntimeContext>) -> Self {
        Self {
            inner: ImageTextureResolver::new(ctx),
        }
    }

    /// Resolve relative `External` URIs against `dir`.
    pub fn with_base_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.inner = self.inner.with_base_dir(dir);
        self
    }

    /// The gateway resolver this wraps.
    pub fn into_inner(self) -> ImageTextureResolver {
        self.inner
    }
}

#[cfg(feature = "registry")]
#[allow(deprecated)]
impl TextureResolver for RegistryTextureResolver {
    fn decode(&self, bytes: &[u8], mime: Option<&str>) -> Option<DecodedImage> {
        self.inner.decode(bytes, mime)
    }

    fn fetch_external(&self, uri: &str) -> Option<Vec<u8>> {
        self.inner.fetch_external(uri)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_mesh3d::InMemoryAsset;

    fn tex_from(w: u32, h: u32, rgba: Vec<u8>, sampler: Sampler) -> PreparedTexture {
        PreparedTexture {
            data: Arc::new(TextureData::new(
                DecodedImage::from_rgba8(w, h, rgba).unwrap(),
            )),
            sampler,
        }
    }

    fn two_by_one() -> Vec<u8> {
        vec![0, 0, 0, 255, 255, 255, 255, 255]
    }

    #[test]
    fn raw_container_round_trips() {
        let bytes = encode_raw_rgba8(2, 1, &two_by_one());
        let img = decode_raw_rgba8(&bytes).unwrap();
        assert_eq!((img.width, img.height), (2, 1));
        assert!(decode_raw_rgba8(&bytes[..10]).is_none());
    }

    #[test]
    fn cache_resolves_raw_source_once() {
        let tex = Texture::from_source(Arc::new(InMemoryAsset::new(
            Some(RAW_RGBA8_MIME.to_string()),
            encode_raw_rgba8(2, 1, &two_by_one()),
        )));
        let mut cache = TextureCache::default();
        let a = cache.prepare(&tex).unwrap();
        let b = cache.prepare(&tex).unwrap();
        assert!(Arc::ptr_eq(&a.data, &b.data));
        assert_eq!(cache.len(), 1);
        // Unknown format with the no-op resolver: absent.
        let png = Texture::from_encoded("image/png", vec![1, 2, 3]);
        assert!(cache.prepare(&png).is_none());
        assert!(cache.prepare(&Texture::from_uri("x.png")).is_none());
    }

    #[test]
    fn nearest_and_wrap_modes() {
        let nearest = Sampler::default_sampler()
            .with_mag_filter(MagFilter::Nearest)
            .with_min_filter(MinFilter::Nearest);
        let t = tex_from(2, 1, two_by_one(), nearest);
        let s = |u: f32| t.sample_lod([u, 0.5], 0.0, ColorSpace::Linear)[0];
        assert_eq!(s(0.25), 0.0);
        assert_eq!(s(0.75), 1.0);
        assert_eq!(s(1.25), 0.0, "repeat");
        let clamp = tex_from(
            2,
            1,
            two_by_one(),
            nearest.with_wrap(WrapMode::ClampToEdge, WrapMode::ClampToEdge),
        );
        assert_eq!(
            clamp.sample_lod([1.25, 0.5], 0.0, ColorSpace::Linear)[0],
            1.0
        );
        assert_eq!(
            clamp.sample_lod([-3.0, 0.5], 0.0, ColorSpace::Linear)[0],
            0.0
        );
        let mirror = tex_from(
            2,
            1,
            two_by_one(),
            nearest.with_wrap(WrapMode::MirroredRepeat, WrapMode::Repeat),
        );
        assert_eq!(
            mirror.sample_lod([1.25, 0.5], 0.0, ColorSpace::Linear)[0],
            1.0
        );
        assert_eq!(
            mirror.sample_lod([1.75, 0.5], 0.0, ColorSpace::Linear)[0],
            0.0
        );
    }

    #[test]
    fn bilinear_midpoint_and_srgb_decode() {
        let t = tex_from(
            2,
            1,
            two_by_one(),
            Sampler::default_sampler().with_wrap(WrapMode::ClampToEdge, WrapMode::ClampToEdge),
        );
        let mid = t.sample_lod([0.5, 0.5], 0.0, ColorSpace::Linear)[0];
        assert!((mid - 0.5).abs() < 1e-6);
        // sRGB 128 → linear ≈ 0.2158.
        let g = tex_from(1, 1, vec![128, 128, 128, 255], Sampler::default_sampler());
        let v = g.sample_lod([0.3, 0.3], 0.0, ColorSpace::Srgb)[0];
        assert!((v - 0.2158).abs() < 1e-3, "{v}");
        let l = g.sample_lod([0.3, 0.3], 0.0, ColorSpace::Linear)[0];
        assert!((l - 128.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn mip_chain_box_filters_to_mean() {
        // 4×4 black/white checker → the 1×1 top level is mid-grey.
        let mut px = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                let v = if (x + y) % 2 == 0 { 255 } else { 0 };
                px.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let t = tex_from(4, 4, px, Sampler::default_sampler());
        let mips = t.data.mips(ColorSpace::Linear);
        assert_eq!(mips.len(), 3);
        assert!((mips[2].texels[0][0] - 0.5).abs() < 1e-6);
        // Huge derivatives select the top level (trilinear default).
        let v = t.sample_grad([0.1, 0.1], [10.0, 0.0], [0.0, 10.0], ColorSpace::Linear);
        assert!((v[0] - 0.5).abs() < 1e-5);
        // Zero derivatives: magnification on level 0 (bilinear between
        // texel centres at an exact centre gives that texel).
        let v0 = t.sample_grad([0.125, 0.125], [0.0; 2], [0.0; 2], ColorSpace::Linear);
        assert!((v0[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn non_power_of_two_mips_terminate() {
        let t = tex_from(5, 3, vec![255; 5 * 3 * 4], Sampler::default_sampler());
        let mips = t.data.mips(ColorSpace::Srgb);
        let last = mips.last().unwrap();
        assert_eq!((last.width, last.height), (1, 1));
        assert!((last.texels[0][0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mime_guess() {
        assert_eq!(mime_from_uri("a/b/c.PNG"), Some("image/png"));
        assert_eq!(mime_from_uri("tex.jpeg"), Some("image/jpeg"));
        assert_eq!(mime_from_uri("noext"), None);
    }

    #[cfg(feature = "registry")]
    #[test]
    fn ext_hint_covers_known_and_generic_image_types() {
        let h = |m: &str| ext_hint_from_mime(m);
        assert_eq!(h("image/png").as_deref(), Some("png"));
        assert_eq!(h("IMAGE/JPEG; charset=binary").as_deref(), Some("jpg"));
        assert_eq!(h("image/x-exr").as_deref(), Some("exr"));
        assert_eq!(h("image/vnd.radiance").as_deref(), Some("hdr"));
        assert_eq!(h("image/heic").as_deref(), Some("heic"));
        assert_eq!(h("image/avif").as_deref(), Some("avif"));
        assert_eq!(h("image/jxl").as_deref(), Some("jxl"));
        // Unknown subtypes pass through with vendor prefixes dropped.
        assert_eq!(h("image/x-foo").as_deref(), Some("foo"));
        assert_eq!(h("image/vnd.bar+xml").as_deref(), Some("bar"));
        assert_eq!(h("application/octet-stream"), None);
        assert_eq!(h("image/"), None);
    }

    #[test]
    fn decoded_to_hdr_decodes_srgb_bytes_and_copies_floats() {
        let b = DecodedImage::from_rgba8(1, 1, vec![255, 0, 128, 51]).unwrap();
        let h = b.to_hdr();
        let p = h.pixel(0, 0).unwrap();
        assert_eq!(p[0], 1.0);
        assert_eq!(p[1], 0.0);
        assert!((p[2] - 0.2158).abs() < 1e-3);
        assert!((p[3] - 0.2).abs() < 1e-6);
        let f = DecodedImage::from_rgba_f32(1, 1, vec![9.0, 0.5, -1.0, 1.0]).unwrap();
        assert_eq!(f.to_hdr().pixel(0, 0), Some([9.0, 0.5, -1.0, 1.0]));
    }
}
