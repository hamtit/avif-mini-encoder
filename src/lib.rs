//! A pure-Rust AVIF encoder for still images, modelled on libaom's all-intra
//! mode as configured by libheif / libvips for AVIF (tune IQ).

mod avail;

/// Whether AVX2 can be used (checked once).
#[inline]
pub(crate) fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static F: OnceLock<bool> = OnceLock::new();
        *F.get_or_init(|| std::arch::is_x86_feature_detected!("avx2") && !debug_flag("AV1_NO_AVX2"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// Developer switches read once from AV1_* environment variables.
pub(crate) fn debug_flag(name: &str) -> bool {
    use std::sync::OnceLock;
    static FLAGS: OnceLock<Vec<String>> = OnceLock::new();
    FLAGS.get_or_init(|| std::env::vars().map(|(k, _)| k).filter(|k| k.starts_with("AV1_")).collect()).iter().any(|k| k == name)
}
mod cdef;
mod cdf;
mod coef;
mod consts;
mod ec;
pub mod enc;
mod frame;
mod lf;
mod obu;
mod parallel;
mod predict;
mod quant;
mod tables;
mod txfm;

pub use obu::{ColorDesc, SequenceHeader};

use enc::FrameConfig;

/// Chroma format of the coded planes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chroma {
    Yuv420,
    Yuv444,
    /// One plane (used for alpha).
    Mono,
}

/// What the encoder optimizes for (libaom's tuning).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tune {
    /// libaom's tune IQ: quantization matrices, variance-boost delta q,
    /// chroma delta q, sharpness 7, QM-weighted distortion.
    Iq,
    /// libaom's default PSNR tuning (used for alpha planes).
    Psnr,
}

/// libaom's aom_get_qmlevel_allintra.
fn qm_level_allintra(q: i32) -> u8 {
    let l = match q {
        ..=40 => 10,
        41..=100 => 9,
        101..=160 => 8,
        161..=200 => 7,
        201..=220 => 6,
        221..=240 => 5,
        _ => 4,
    };
    l.clamp(2, 10)
}

/// libaom's aom_get_qmlevel_444_chroma.
fn qm_level_444_chroma(q: i32) -> u8 {
    let l = match q {
        ..=12 => 10,
        13..=24 => 9,
        25..=32 => 8,
        33..=36 => 7,
        37..=44 => 6,
        45..=48 => 5,
        49..=56 => 4,
        57..=88 => 3,
        _ => 2,
    };
    l.clamp(2, 10)
}

/// Frame settings libaom derives for all-intra coding with the given tuning.
pub(crate) fn frame_config(qindex: u8, chroma: Chroma, tune: Tune, speed: u8) -> FrameConfig {
    let q = qindex as i32;
    let (num_planes, ssx, ssy) = match chroma {
        Chroma::Yuv420 => (3, 1, 1),
        Chroma::Yuv444 => (3, 0, 0),
        Chroma::Mono => (1, 1, 1),
    };
    let iq = tune == Tune::Iq;
    let uv_delta_q = if iq && q > 0 {
        match chroma {
            Chroma::Yuv420 => {
                let d = -((q / 2) - 14).clamp(0, 16);
                (d, d)
            }
            Chroma::Yuv444 => (0, (q / 2).clamp(0, 24)),
            Chroma::Mono => (0, 0),
        }
    } else {
        (0, 0)
    };
    let qm = if iq && q > 0 && !debug_flag("AV1_NO_QM") {
        let y = qm_level_allintra(q);
        let uv = match chroma {
            Chroma::Yuv444 => qm_level_444_chroma(q + uv_delta_q.1),
            _ => qm_level_allintra(q + uv_delta_q.1),
        };
        Some((y, uv))
    } else {
        None
    };
    FrameConfig {
        base_q: qindex,
        num_planes,
        ssx,
        ssy,
        uv_delta_q,
        qm,
        qm_dist: iq && !debug_flag("AV1_NO_QMDIST"),
        deltaq_variance_boost: iq && !debug_flag("AV1_NO_DELTAQ"),
        tune_iq: iq && !debug_flag("AV1_NO_TUNE"),
        sharpness: if iq { 7 } else { 0 },
        enable_cdef: true,
        speed,
        threads: 1,
    }
}

/// Encoding error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// Encode planar 8-bit samples to AV1 OBUs (sequence header + frame).
/// Returns the OBUs and the reconstructed planes.
#[allow(clippy::too_many_arguments)]
pub fn encode_av1(
    planes: &[&[u8]],
    width: usize,
    height: usize,
    chroma: Chroma,
    color: Option<ColorDesc>,
    full_range: bool,
    qindex: u8,
    tune: Tune,
    speed: u8,
) -> Result<(Vec<u8>, Vec<Vec<u8>>), Error> {
    encode_av1_threads(planes, width, height, chroma, color, full_range, qindex, tune, speed, 1)
}

/// Like [`encode_av1`], using up to `threads` threads for the stages that can
/// run in parallel (currently the CDEF strength search). The output is
/// byte-identical for every thread count.
#[allow(clippy::too_many_arguments)]
pub fn encode_av1_threads(
    planes: &[&[u8]],
    width: usize,
    height: usize,
    chroma: Chroma,
    color: Option<ColorDesc>,
    full_range: bool,
    qindex: u8,
    tune: Tune,
    speed: u8,
    threads: usize,
) -> Result<(Vec<u8>, Vec<Vec<u8>>), Error> {
    if width == 0 || height == 0 || width > 65536 || height > 65536 {
        return Err(Error(format!("unsupported size {width}x{height}")));
    }
    // Only one tile is written: the frame must fit AV1's single-tile limits
    // (4096 wide, 4096 * 2304 samples).
    let mi_cols = 2 * width.div_ceil(8);
    let mi_rows = 2 * height.div_ceil(8);
    if obu::tile_limits(mi_cols, mi_rows).min_tiles_log2 > 0 {
        return Err(Error(format!("{width}x{height} needs more than one tile, which is not implemented")));
    }
    if chroma != Chroma::Yuv444
        && chroma != Chroma::Mono
        && matches!(color, Some(ColorDesc { primaries: 1, transfer: 13, matrix: 0 }))
    {
        return Err(Error("identity matrix (sRGB) requires 4:4:4".into()));
    }
    let mut cfg = frame_config(qindex, chroma, tune, speed);
    cfg.threads = threads.max(1);
    let (cw, ch) = ((width + cfg.ssx) >> cfg.ssx, (height + cfg.ssy) >> cfg.ssy);
    let mut p: Vec<(&[u8], usize, usize, usize)> = vec![(planes[0], width, width, height)];
    if cfg.num_planes == 3 {
        p.push((planes[1], cw, cw, ch));
        p.push((planes[2], cw, cw, ch));
    }
    let seq = SequenceHeader {
        profile: if chroma == Chroma::Yuv444 { 1 } else { 0 },
        level: 31,
        width,
        height,
        mono: chroma == Chroma::Mono,
        subsampled: chroma != Chroma::Yuv444,
        color,
        full_range,
        enable_cdef: cfg.enable_cdef,
    };
    let out = raw::encode(cfg, &p, &seq);
    Ok((out.obus, out.recon))
}

/// Arguments of one [`encode_av1`] call, for [`encode_av1_pair`].
#[derive(Clone, Copy)]
pub struct EncodeJob<'a> {
    pub planes: &'a [&'a [u8]],
    pub width: usize,
    pub height: usize,
    pub chroma: Chroma,
    pub color: Option<ColorDesc>,
    pub full_range: bool,
    pub qindex: u8,
    pub tune: Tune,
    pub speed: u8,
}

/// Result of one encode: the OBUs and the reconstructed planes.
pub type EncodeResult = Result<(Vec<u8>, Vec<Vec<u8>>), Error>;

/// Encode a colour image and its alpha plane (two independent AV1 streams).
/// With `threads >= 2` the alpha stream is encoded on its own thread while
/// the colour stream uses the remaining threads for its parallel stages.
/// The outputs are byte-identical to two sequential [`encode_av1`] calls.
pub fn encode_av1_pair(color: EncodeJob, alpha: EncodeJob, threads: usize) -> (EncodeResult, EncodeResult) {
    let run = |j: EncodeJob, t: usize| {
        encode_av1_threads(j.planes, j.width, j.height, j.chroma, j.color, j.full_range, j.qindex, j.tune, j.speed, t)
    };
    if threads < 2 {
        return (run(color, 1), run(alpha, 1));
    }
    std::thread::scope(|s| {
        let a = s.spawn(|| run(alpha, 1));
        let c = run(color, threads - 1);
        (c, a.join().unwrap_or_else(|_| Err(Error("alpha encoder panicked".into()))))
    })
}

/// Benchmark helper: BT.601 full range sRGB YUV at a given qindex, tune IQ.
#[doc(hidden)]
pub fn encode_yuv_for_bench(w: usize, h: usize, yuv: &[Vec<u8>], ss: usize, qindex: u8) -> Result<(Vec<u8>, Vec<Vec<u8>>), Error> {
    let planes: Vec<&[u8]> = yuv.iter().map(|p| p.as_slice()).collect();
    encode_av1(
        &planes,
        w,
        h,
        if ss == 1 { Chroma::Yuv420 } else { Chroma::Yuv444 },
        Some(ColorDesc { primaries: 1, transfer: 13, matrix: 6 }),
        true,
        qindex,
        Tune::Iq,
        5,
    )
}

/// Raw AV1 encoding of planar 8-bit YUV (or a single plane), for testing.
pub mod raw {
    use crate::enc::{Encoder, FrameConfig};
    use crate::obu::SequenceHeader;

    pub struct Output {
        pub obus: Vec<u8>,
        /// Reconstructed planes (visible size), as the decoder will see them.
        pub recon: Vec<Vec<u8>>,
    }

    /// `planes`: (data, stride, width, height) per plane.
    pub fn encode(cfg: FrameConfig, planes: &[(&[u8], usize, usize, usize)], seq: &SequenceHeader) -> Output {
        let mut e = Encoder::new(cfg, planes);
        let obus = e.encode(seq);
        let recon = e
            .rec
            .iter()
            .map(|p| {
                let mut v = Vec::with_capacity(p.width * p.height);
                for y in 0..p.height {
                    v.extend_from_slice(&p.data[y * p.stride..y * p.stride + p.width]);
                }
                v
            })
            .collect();
        Output { obus, recon }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(w: usize, h: usize) -> Vec<u8> {
        vec![128; w * h]
    }

    #[test]
    fn rejects_frames_needing_several_tiles() {
        let (w, h) = (4104, 8);
        let y = flat(w, h);
        let r = encode_av1(&[&y], w, h, Chroma::Mono, None, true, 100, Tune::Psnr, 5);
        assert!(r.is_err());
        let (w, h) = (4096, 8);
        let y = flat(w, h);
        assert!(encode_av1(&[&y], w, h, Chroma::Mono, None, true, 100, Tune::Psnr, 5).is_ok());
    }

    #[test]
    fn pair_matches_sequential_encodes() {
        let (w, h) = (70, 50);
        let y: Vec<u8> = (0..w * h).map(|i| ((i * 7 + (i / w) * 13) % 251) as u8).collect();
        let c: Vec<u8> = (0..(w / 2) * (h / 2)).map(|i| (i * 5 % 200) as u8 + 20).collect();
        let a: Vec<u8> = (0..w * h).map(|i| if (i % w) < 30 { 255 } else { (i % 97) as u8 }).collect();
        let col = [&y[..], &c[..], &c[..]];
        let alp = [&a[..]];
        let cj = EncodeJob { planes: &col, width: w, height: h, chroma: Chroma::Yuv420, color: None, full_range: true, qindex: 120, tune: Tune::Iq, speed: 5 };
        let aj = EncodeJob { planes: &alp, width: w, height: h, chroma: Chroma::Mono, color: None, full_range: true, qindex: 90, tune: Tune::Psnr, speed: 5 };
        let seq_c = encode_av1(cj.planes, w, h, cj.chroma, None, true, 120, Tune::Iq, 5).unwrap();
        let seq_a = encode_av1(aj.planes, w, h, aj.chroma, None, true, 90, Tune::Psnr, 5).unwrap();
        for threads in [1, 2, 4] {
            let (pc, pa) = encode_av1_pair(cj, aj, threads);
            assert_eq!(pc.unwrap(), seq_c);
            assert_eq!(pa.unwrap(), seq_a);
        }
    }

    #[test]
    fn rejects_identity_matrix_with_subsampling() {
        let y = flat(16, 16);
        let c = flat(8, 8);
        let srgb = Some(ColorDesc { primaries: 1, transfer: 13, matrix: 0 });
        assert!(encode_av1(&[&y, &c, &c], 16, 16, Chroma::Yuv420, srgb, true, 100, Tune::Iq, 5).is_err());
    }
}
