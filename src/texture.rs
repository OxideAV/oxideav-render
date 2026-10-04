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
//! `registry` feature [`RegistryTextureResolver`] decodes through an
//! `oxideav_core::RuntimeContext` (whatever image containers/codecs
//! the caller registered, e.g. via `oxideav-meta`). One format is
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

use crate::hdr::srgb_u8_lut;

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
// Registry-backed resolver.
// ---------------------------------------------------------------------

/// [`TextureResolver`] that decodes through an
/// `oxideav_core::RuntimeContext`: the container registry probes the
/// bytes (with an extension hint derived from the MIME type), the
/// first video stream's codec decodes one frame, and `oxideav-pixfmt`
/// converts it to RGBA8. Supports whatever image formats the caller
/// registered into the context (e.g. `oxideav_meta::register_all`).
///
/// `External` URIs are read from the filesystem relative to
/// [`RegistryTextureResolver::with_base_dir`] (absolute paths as-is);
/// `data:` / network URIs are refused.
#[cfg(feature = "registry")]
pub struct RegistryTextureResolver {
    ctx: Arc<oxideav_core::RuntimeContext>,
    base_dir: Option<std::path::PathBuf>,
}

#[cfg(feature = "registry")]
impl std::fmt::Debug for RegistryTextureResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryTextureResolver")
            .field("base_dir", &self.base_dir)
            .finish()
    }
}

#[cfg(feature = "registry")]
impl RegistryTextureResolver {
    /// Decode through `ctx`.
    pub fn new(ctx: Arc<oxideav_core::RuntimeContext>) -> Self {
        Self {
            ctx,
            base_dir: None,
        }
    }

    /// Resolve relative `External` URIs against `dir`.
    pub fn with_base_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.base_dir = Some(dir.into());
        self
    }

    fn decode_inner(&self, bytes: &[u8], mime: Option<&str>) -> oxideav_core::Result<DecodedImage> {
        use oxideav_core::{Error as CoreError, Frame};
        let ext_hint = mime.and_then(|m| {
            Some(match m {
                "image/png" => "png",
                "image/jpeg" => "jpg",
                "image/webp" => "webp",
                "image/bmp" => "bmp",
                "image/gif" => "gif",
                "image/tiff" => "tiff",
                "image/ktx2" => "ktx2",
                "image/x-exr" => "exr",
                "image/vnd.radiance" => "hdr",
                "image/x-tga" => "tga",
                _ => return None,
            })
        });
        let mut probe_input = std::io::Cursor::new(bytes.to_vec());
        let name = self
            .ctx
            .containers
            .probe_input(&mut probe_input, ext_hint)?;
        let input: Box<dyn oxideav_core::ReadSeek> = Box::new(std::io::Cursor::new(bytes.to_vec()));
        let mut demux = self
            .ctx
            .containers
            .open_demuxer(&name, input, &self.ctx.codecs)?;
        let (stream_index, params) = demux
            .streams()
            .iter()
            .find(|s| s.params.width.is_some())
            .map(|s| (s.index, s.params.clone()))
            .ok_or_else(|| CoreError::invalid("no video stream in image container"))?;
        let mut dec = self.ctx.codecs.first_decoder(&params)?;
        let frame = loop {
            match dec.receive_frame() {
                Ok(Frame::Video(f)) => break f,
                Ok(_) => continue,
                Err(_) => {}
            }
            let pkt = match demux.next_packet() {
                Ok(p) => p,
                Err(CoreError::Eof) => {
                    dec.flush()?;
                    match dec.receive_frame()? {
                        Frame::Video(f) => break f,
                        _ => return Err(CoreError::invalid("image decoder produced no picture")),
                    }
                }
                Err(e) => return Err(e),
            };
            if pkt.stream_index != stream_index {
                continue;
            }
            dec.send_packet(&pkt)?;
        };
        let out_params = params;
        let width = out_params
            .width
            .ok_or_else(|| CoreError::invalid("decoder reported no width"))?;
        let height = out_params
            .height
            .ok_or_else(|| CoreError::invalid("decoder reported no height"))?;
        let fmt = out_params
            .pixel_format
            .ok_or_else(|| CoreError::invalid("decoder reported no pixel format"))?;
        let info = oxideav_pixfmt::FrameInfo::new(fmt, width, height);
        let rgba = oxideav_pixfmt::convert(
            &frame,
            info,
            oxideav_core::PixelFormat::Rgba,
            &oxideav_pixfmt::ConvertOptions::default(),
        )?;
        let plane = rgba
            .planes
            .first()
            .ok_or_else(|| CoreError::invalid("converted frame has no plane"))?;
        let row = width as usize * 4;
        let mut px = Vec::with_capacity(row * height as usize);
        for y in 0..height as usize {
            let start = y * plane.stride;
            px.extend_from_slice(
                plane
                    .data
                    .get(start..start + row)
                    .ok_or_else(|| CoreError::invalid("short RGBA plane"))?,
            );
        }
        DecodedImage::from_rgba8(width, height, px)
            .ok_or_else(|| CoreError::invalid("bad decoded image dimensions"))
    }
}

#[cfg(feature = "registry")]
impl TextureResolver for RegistryTextureResolver {
    fn decode(&self, bytes: &[u8], mime: Option<&str>) -> Option<DecodedImage> {
        self.decode_inner(bytes, mime).ok()
    }

    fn fetch_external(&self, uri: &str) -> Option<Vec<u8>> {
        if uri.contains("://") || uri.starts_with("data:") {
            return None;
        }
        let path = std::path::Path::new(uri);
        let full = match (&self.base_dir, path.is_absolute()) {
            (Some(base), false) => base.join(path),
            _ => path.to_path_buf(),
        };
        std::fs::read(full).ok()
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
}
