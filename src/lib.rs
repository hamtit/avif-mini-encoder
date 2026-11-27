//! A pure-Rust AVIF encoder for still images, modelled on libaom's all-intra
//! mode as configured by libheif / libvips for AVIF (tune IQ).

// Index loops mirror the layout of the C reference code (libaom) the kernels are ported from.
#![allow(clippy::needless_range_loop)]

mod avail;
mod error;

/// Whether AVX2 can be used (checked once).
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn has_avx2() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::arch::is_x86_feature_detected!("avx2") && !debug_flag("AV1_NO_AVX2"))
}

/// Whether the NEON kernels can be used (checked once). NEON is part of the
/// aarch64 baseline, so only the `AV1_NO_NEON` developer switch can turn it off.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn has_neon() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| !debug_flag("AV1_NO_NEON"))
}

/// Developer switches read once from AV1_* environment variables.
pub(crate) fn debug_flag(name: &str) -> bool {
    use std::sync::OnceLock;
    static FLAGS: OnceLock<Vec<String>> = OnceLock::new();
    FLAGS
        .get_or_init(|| std::env::vars().map(|(k, _)| k).filter(|k| k.starts_with("AV1_")).collect())
        .iter()
        .any(|k| k == name)
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

pub use error::Error;
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

/// Version of this crate (`CARGO_PKG_VERSION`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

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

/// Validate encode arguments and construct `(FrameConfig, plane descriptors, SequenceHeader)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn prepare_encode<'a>(
    planes: &'a [&'a [u8]],
    width: usize,
    height: usize,
    chroma: Chroma,
    color: Option<ColorDesc>,
    full_range: bool,
    qindex: u8,
    tune: Tune,
    speed: u8,
    threads: usize,
) -> Result<(FrameConfig, Vec<(&'a [u8], usize, usize, usize)>, SequenceHeader), Error> {
    if width == 0 || height == 0 || width > 65536 || height > 65536 {
        return Err(Error::InvalidDimensions { width, height });
    }
    // Frames over AV1's single-tile limits get mandatory tiles (see `choose_tile_grid`).
    // Validate the planes so a short buffer is an error, not a panic. This runs
    // after the size limits, so the products below cannot overflow.
    let (ssx, ssy) = if chroma == Chroma::Yuv444 { (0, 0) } else { (1, 1) };
    let (cw, ch) = ((width + ssx) >> ssx, (height + ssy) >> ssy);
    let expected: &[usize] =
        if chroma == Chroma::Mono { &[width * height] } else { &[width * height, cw * ch, cw * ch] };
    if planes.len() != expected.len() {
        return Err(Error::PlaneCount { expected: expected.len(), actual: planes.len() });
    }
    for (i, (plane, &len)) in planes.iter().zip(expected).enumerate() {
        if plane.len() != len {
            return Err(Error::PlaneSize { plane: i, expected: len, actual: plane.len() });
        }
    }
    if chroma != Chroma::Yuv444
        && chroma != Chroma::Mono
        && matches!(color, Some(ColorDesc { primaries: 1, transfer: 13, matrix: 0 }))
    {
        return Err(Error::IdentityRequires444);
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
    Ok((cfg, p, seq))
}

/// Like [`encode_av1`], using up to `threads` threads for parallel tile
/// encoding and CDEF strength search.
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
    let (cfg, p, seq) = prepare_encode(planes, width, height, chroma, color, full_range, qindex, tune, speed, threads)?;
    let out = raw::encode(cfg, &p, &seq);
    Ok((out.obus, out.recon))
}

/// Like [`encode_av1`], returning only the AV1 OBUs without allocating or
/// computing the post-CDEF reconstructed planes.
#[allow(clippy::too_many_arguments)]
pub fn encode_av1_obu(
    planes: &[&[u8]],
    width: usize,
    height: usize,
    chroma: Chroma,
    color: Option<ColorDesc>,
    full_range: bool,
    qindex: u8,
    tune: Tune,
    speed: u8,
) -> Result<Vec<u8>, Error> {
    encode_av1_obu_threads(planes, width, height, chroma, color, full_range, qindex, tune, speed, 1)
}

/// Like [`encode_av1_threads`], returning only the AV1 OBUs without allocating
/// or computing the post-CDEF reconstructed planes.
#[allow(clippy::too_many_arguments)]
pub fn encode_av1_obu_threads(
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
) -> Result<Vec<u8>, Error> {
    let (cfg, p, seq) = prepare_encode(planes, width, height, chroma, color, full_range, qindex, tune, speed, threads)?;
    Ok(raw::encode_obu(cfg, &p, &seq))
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
        (c, a.join().unwrap_or(Err(Error::AlphaEncoderPanicked)))
    })
}

/// Benchmark helper: BT.601 full range sRGB YUV at a given qindex, tune IQ.
#[doc(hidden)]
pub fn encode_yuv_for_bench(
    w: usize,
    h: usize,
    yuv: &[Vec<u8>],
    ss: usize,
    qindex: u8,
) -> Result<(Vec<u8>, Vec<Vec<u8>>), Error> {
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

    /// Encode to AV1 OBUs only, skipping post-CDEF reconstruction.
    pub fn encode_obu(cfg: FrameConfig, planes: &[(&[u8], usize, usize, usize)], seq: &SequenceHeader) -> Vec<u8> {
        let mut e = Encoder::new(cfg, planes);
        e.encode_obu(seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(w: usize, h: usize) -> Vec<u8> {
        vec![128; w * h]
    }

    #[test]
    fn encodes_frames_needing_mandatory_tiles() {
        // 4104 wide needs two tile columns even with one thread.
        let (w, h) = (4104, 16);
        let y: Vec<u8> = (0..w * h).map(|i| ((i % w) * 3 / 50 + (i / w) * 5) as u8).collect();
        for threads in [1, 2] {
            let (obus, recon) =
                encode_av1_threads(&[&y], w, h, Chroma::Mono, None, true, 60, Tune::Psnr, 6, threads).unwrap();
            assert!(!obus.is_empty());
            let mse =
                y.iter().zip(&recon[0]).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum::<f64>() / (w * h) as f64;
            assert!(mse < 4.0, "threads {threads}: mse {mse}");
        }
    }

    #[test]
    fn pair_matches_sequential_encodes() {
        let (w, h) = (70, 50);
        let y: Vec<u8> = (0..w * h).map(|i| ((i * 7 + (i / w) * 13) % 251) as u8).collect();
        let c: Vec<u8> = (0..(w / 2) * (h / 2)).map(|i| (i * 5 % 200) as u8 + 20).collect();
        let a: Vec<u8> = (0..w * h).map(|i| if (i % w) < 30 { 255 } else { (i % 97) as u8 }).collect();
        let col = [&y[..], &c[..], &c[..]];
        let alp = [&a[..]];
        let cj = EncodeJob {
            planes: &col,
            width: w,
            height: h,
            chroma: Chroma::Yuv420,
            color: None,
            full_range: true,
            qindex: 120,
            tune: Tune::Iq,
            speed: 5,
        };
        let aj = EncodeJob {
            planes: &alp,
            width: w,
            height: h,
            chroma: Chroma::Mono,
            color: None,
            full_range: true,
            qindex: 90,
            tune: Tune::Psnr,
            speed: 5,
        };
        let seq_c = encode_av1(cj.planes, w, h, cj.chroma, None, true, 120, Tune::Iq, 5).unwrap();
        let seq_a = encode_av1(aj.planes, w, h, aj.chroma, None, true, 90, Tune::Psnr, 5).unwrap();
        for threads in [1, 2, 4] {
            let (pc, pa) = encode_av1_pair(cj, aj, threads);
            assert_eq!(pc.unwrap(), seq_c);
            assert_eq!(pa.unwrap(), seq_a);
        }
    }

    #[test]
    fn rejects_malformed_planes() {
        let y = flat(16, 16);
        let c = flat(8, 8);
        let run = |planes: &[&[u8]], chroma| encode_av1(planes, 16, 16, chroma, None, true, 100, Tune::Iq, 5);
        assert!(run(&[&y, &c, &c], Chroma::Yuv420).is_ok());
        let err = |planes: &[&[u8]], chroma| run(planes, chroma).unwrap_err();
        assert_eq!(
            err(&[&y[..255], &c, &c], Chroma::Yuv420),
            Error::PlaneSize { plane: 0, expected: 256, actual: 255 }
        );
        assert_eq!(err(&[&y, &c[..63], &c], Chroma::Yuv420), Error::PlaneSize { plane: 1, expected: 64, actual: 63 });
        assert_eq!(err(&[&y, &c, &c], Chroma::Yuv444), Error::PlaneSize { plane: 1, expected: 256, actual: 64 });
        assert_eq!(err(&[&y], Chroma::Yuv420), Error::PlaneCount { expected: 3, actual: 1 });
        assert_eq!(err(&[&y, &y], Chroma::Mono), Error::PlaneCount { expected: 1, actual: 2 });
        assert!(run(&[&y], Chroma::Mono).is_ok());
    }

    #[test]
    fn rejects_unsupported_sizes() {
        let r = |w, h| encode_av1(&[], w, h, Chroma::Mono, None, true, 100, Tune::Iq, 5).unwrap_err();
        assert_eq!(r(0, 16), Error::InvalidDimensions { width: 0, height: 16 });
        assert_eq!(r(16, 65537), Error::InvalidDimensions { width: 16, height: 65537 });
    }

    #[test]
    fn error_messages_are_stable() {
        assert_eq!(Error::InvalidDimensions { width: 0, height: 4 }.to_string(), "unsupported size 0x4");
        assert_eq!(Error::PlaneCount { expected: 3, actual: 1 }.to_string(), "expected 3 plane(s), got 1");
        assert_eq!(
            Error::PlaneSize { plane: 1, expected: 64, actual: 63 }.to_string(),
            "plane 1 has 63 bytes, expected 64"
        );
    }

    #[test]
    fn exposes_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn rejects_identity_matrix_with_subsampling() {
        let y = flat(16, 16);
        let c = flat(8, 8);
        let srgb = Some(ColorDesc { primaries: 1, transfer: 13, matrix: 0 });
        let r = encode_av1(&[&y, &c, &c], 16, 16, Chroma::Yuv420, srgb, true, 100, Tune::Iq, 5);
        assert_eq!(r.unwrap_err(), Error::IdentityRequires444);
    }

    #[test]
    fn obu_api_and_multi_tile_sanity() {
        let (w, h) = (512, 512);
        let mut y = vec![0u8; w * h];
        let mut seed = 0x1357_9BDFu32;
        for r in 0..h {
            for c in 0..w {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let grad = ((r * 160 / h) + (c * 55 / w)) as i32;
                let tex = ((((r * 3 + c * 5) % 29) as i32 - 14) * 2) + ((seed >> 27) as i32 - 16);
                y[r * w + c] = (grad + tex).clamp(0, 255) as u8;
            }
        }
        let u = vec![120u8; (w / 2) * (h / 2)];
        let v = vec![136u8; (w / 2) * (h / 2)];
        let planes = [&y[..], &u[..], &v[..]];

        let (obu1, rec1) = encode_av1_threads(&planes, w, h, Chroma::Yuv420, None, true, 120, Tune::Iq, 5, 1).unwrap();
        let obu1_only = encode_av1_obu_threads(&planes, w, h, Chroma::Yuv420, None, true, 120, Tune::Iq, 5, 1).unwrap();
        assert_eq!(obu1, obu1_only);

        let (obu2, rec2) = encode_av1_threads(&planes, w, h, Chroma::Yuv420, None, true, 120, Tune::Iq, 5, 2).unwrap();
        let (obu4, rec4) = encode_av1_threads(&planes, w, h, Chroma::Yuv420, None, true, 120, Tune::Iq, 5, 4).unwrap();
        let obu4_only = encode_av1_obu_threads(&planes, w, h, Chroma::Yuv420, None, true, 120, Tune::Iq, 5, 4).unwrap();
        assert_eq!(obu4, obu4_only);

        let psnr = |rec: &[u8]| -> f64 {
            let mse: f64 = y
                .iter()
                .zip(rec)
                .map(|(&a, &b)| {
                    let d = a as f64 - b as f64;
                    d * d
                })
                .sum::<f64>()
                / (w * h) as f64;
            10.0 * (255.0 * 255.0 / mse.max(1e-9)).log10()
        };
        let p1 = psnr(&rec1[0]);
        let p2 = psnr(&rec2[0]);
        let p4 = psnr(&rec4[0]);
        eprintln!(
            "512x512 q120: 1t={}B ({:.2}dB), 2t={}B ({:.2}dB), 4t={}B ({:.2}dB)",
            obu1.len(),
            p1,
            obu2.len(),
            p2,
            obu4.len(),
            p4
        );
        assert!((p1 - p4).abs() < 1.0, "PSNR drift too large: 1t={p1:.2} 4t={p4:.2}");
        assert!(obu4.len() < obu1.len() * 103 / 100, "4-tile overhead too large: 1t={} 4t={}", obu1.len(), obu4.len());
    }
}
