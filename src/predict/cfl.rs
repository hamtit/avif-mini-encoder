//! CfL uses Q3 luma and Q3 alpha; signed rounding is away from zero.

pub(crate) fn ac(luma: &mut [i16]) {
    let sum: i32 = luma.iter().map(|&v| v as i32).sum();
    let shift = luma.len().trailing_zeros();
    let avg = (sum + (1 << (shift - 1))) >> shift;
    for v in luma {
        *v -= avg as i16;
    }
}

#[inline(always)]
fn pixel(dc: u8, ac: i16, alpha: i32) -> u8 {
    let v = alpha * ac as i32;
    let v = if v < 0 { -((-v + 32) >> 6) } else { (v + 32) >> 6 };
    (dc as i32 + v).clamp(0, 255) as u8
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sse(src: &[u8], ss: usize, pred: &[u8], ps: usize, w: usize, h: usize, ac: &[i16], alpha: i32) -> u64 {
    assert!(w > 0 && h > 0 && w <= 32 && h <= 32);
    assert!(src.len() >= (h - 1) * ss + w && pred.len() >= (h - 1) * ps + w && ac.len() >= w * h);
    #[cfg(target_arch = "x86_64")]
    if crate::has_avx2() {
        // SAFETY: detected AVX2; all row bounds checked above.
        return unsafe { sse_avx2(src, ss, pred, ps, w, h, ac, alpha) };
    }
    #[cfg(target_arch = "aarch64")]
    if crate::has_neon() {
        // SAFETY: NEON is part of the aarch64 baseline; all row bounds checked above.
        return unsafe { sse_neon(src, ss, pred, ps, w, h, ac, alpha) };
    }
    sse_scalar(src, ss, pred, ps, w, h, ac, alpha)
}

/// SSE of every alpha in -16..=16 (index `alpha + 16`); same values as `sse` per alpha.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sse_all(src: &[u8], ss: usize, pred: &[u8], ps: usize, w: usize, h: usize, ac: &[i16]) -> [u64; 33] {
    assert!(w > 0 && h > 0 && w <= 32 && h <= 32 && w.is_multiple_of(2));
    assert!(src.len() >= (h - 1) * ss + w && pred.len() >= (h - 1) * ps + w && ac.len() >= w * h);
    #[cfg(target_arch = "x86_64")]
    if crate::has_avx2() {
        // SAFETY: detected AVX2; all row bounds checked above.
        let mut out = unsafe { sse_all_avx2(src, ss, pred, ps, w, h, ac) };
        out[32] = sse(src, ss, pred, ps, w, h, ac, 16);
        return out;
    }
    #[cfg(target_arch = "aarch64")]
    if crate::has_neon() {
        // SAFETY: NEON is part of the aarch64 baseline; all row bounds checked above.
        let mut out = unsafe { sse_all_neon(src, ss, pred, ps, w, h, ac) };
        out[32] = sse(src, ss, pred, ps, w, h, ac, 16);
        return out;
    }
    core::array::from_fn(|i| sse_scalar(src, ss, pred, ps, w, h, ac, i as i32 - 16))
}

/// Alphas -16..=15 for all pixels in one pass: 16-bit lanes (|alpha * ac| <= 32640), two
/// pixels interleaved so that `madd` sums their squared differences into 32-bit lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn sse_all_avx2(src: &[u8], ss: usize, pred: &[u8], ps: usize, w: usize, h: usize, ac: &[i16]) -> [u64; 33] {
    use std::arch::x86_64::*;
    // SAFETY: AVX2 is enabled for this function. `sse_all` asserted that `w` is even (so the
    // pixel pair `c`, `c + 1` stays inside the row) and that `src`, `pred` and `ac` cover `h`
    // rows of `w` pixels, which bounds every unchecked read; the stores go to local arrays.
    unsafe {
        let a0 = _mm256_setr_epi16(-16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1);
        let a1 = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let c32 = _mm256_set1_epi16(32);
        let zero = _mm256_setzero_si256();
        let c255 = _mm256_set1_epi16(255);
        // Squared difference (as the pair-summed input to madd) for one pixel and one alpha vector.
        let diff = |a: __m256i, acv: __m256i, pv: __m256i, sv: __m256i| {
            let v = _mm256_mullo_epi16(a, acv);
            let r = _mm256_srli_epi16::<6>(_mm256_add_epi16(_mm256_abs_epi16(v), c32));
            let scaled = _mm256_sign_epi16(r, v);
            let p = _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(pv, scaled), zero), c255);
            _mm256_sub_epi16(sv, p)
        };
        let mut acc = [_mm256_setzero_si256(); 4];
        for r in 0..h {
            let mut c = 0;
            while c < w {
                let mut acv = [zero; 2];
                let mut pv = [zero; 2];
                let mut sv = [zero; 2];
                for k in 0..2 {
                    acv[k] = _mm256_set1_epi16(*ac.get_unchecked(r * w + c + k));
                    pv[k] = _mm256_set1_epi16(*pred.get_unchecked(r * ps + c + k) as i16);
                    sv[k] = _mm256_set1_epi16(*src.get_unchecked(r * ss + c + k) as i16);
                }
                for (i, a) in [a0, a1].into_iter().enumerate() {
                    let d0 = diff(a, acv[0], pv[0], sv[0]);
                    let d1 = diff(a, acv[1], pv[1], sv[1]);
                    let lo = _mm256_unpacklo_epi16(d0, d1);
                    let hi = _mm256_unpackhi_epi16(d0, d1);
                    acc[2 * i] = _mm256_add_epi32(acc[2 * i], _mm256_madd_epi16(lo, lo));
                    acc[2 * i + 1] = _mm256_add_epi32(acc[2 * i + 1], _mm256_madd_epi16(hi, hi));
                }
                c += 2;
            }
        }
        let mut out = [0u64; 33];
        for i in 0..2 {
            let mut lo = [0i32; 8];
            let mut hi = [0i32; 8];
            _mm256_storeu_si256(lo.as_mut_ptr().cast(), acc[2 * i]);
            _mm256_storeu_si256(hi.as_mut_ptr().cast(), acc[2 * i + 1]);
            // unpacklo covers alphas 0-3 and 8-11 of the vector, unpackhi 4-7 and 12-15.
            for j in 0..4 {
                out[16 * i + j] = lo[j] as u64;
                out[16 * i + 4 + j] = hi[j] as u64;
                out[16 * i + 8 + j] = lo[4 + j] as u64;
                out[16 * i + 12 + j] = hi[4 + j] as u64;
            }
        }
        out
    }
}

/// NEON twin of `sse_all_avx2`: one 16-bit lane per alpha (-16..=15, four vectors of
/// eight), 32-bit accumulators indexed directly by `alpha + 16`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn sse_all_neon(src: &[u8], ss: usize, pred: &[u8], ps: usize, w: usize, h: usize, ac: &[i16]) -> [u64; 33] {
    use std::arch::aarch64::*;
    const A: [[i16; 8]; 4] = [
        [-16, -15, -14, -13, -12, -11, -10, -9],
        [-8, -7, -6, -5, -4, -3, -2, -1],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15],
    ];
    // SAFETY: NEON is enabled for this function. `sse_all` asserted that `src`, `pred` and `ac`
    // cover `h` rows of `w` pixels, which bounds every unchecked read; `A` rows hold eight lanes.
    unsafe {
        let a =
            [vld1q_s16(A[0].as_ptr()), vld1q_s16(A[1].as_ptr()), vld1q_s16(A[2].as_ptr()), vld1q_s16(A[3].as_ptr())];
        let c32 = vdupq_n_u16(32);
        let zero = vdupq_n_s16(0);
        let c255 = vdupq_n_s16(255);
        // Reconstructed-minus-source difference for one pixel and eight alphas.
        let diff = |a: int16x8_t, acv: int16x8_t, pv: int16x8_t, sv: int16x8_t| {
            let v = vmulq_s16(a, acv);
            let r = vshrq_n_u16::<6>(vaddq_u16(vreinterpretq_u16_s16(vabsq_s16(v)), c32));
            let r = vreinterpretq_s16_u16(r);
            let scaled = vbslq_s16(vcltq_s16(v, zero), vnegq_s16(r), r);
            let p = vminq_s16(vmaxq_s16(vaddq_s16(pv, scaled), zero), c255);
            vsubq_s16(sv, p)
        };
        let mut acc = [vdupq_n_s32(0); 8];
        for r in 0..h {
            let mut c = 0;
            while c < w {
                let mut acv = [zero; 2];
                let mut pv = [zero; 2];
                let mut sv = [zero; 2];
                for k in 0..2 {
                    acv[k] = vdupq_n_s16(*ac.get_unchecked(r * w + c + k));
                    pv[k] = vdupq_n_s16(*pred.get_unchecked(r * ps + c + k) as i16);
                    sv[k] = vdupq_n_s16(*src.get_unchecked(r * ss + c + k) as i16);
                }
                for (i, &av) in a.iter().enumerate() {
                    let d0 = diff(av, acv[0], pv[0], sv[0]);
                    let d1 = diff(av, acv[1], pv[1], sv[1]);
                    let lo =
                        vmlal_s16(vmull_s16(vget_low_s16(d0), vget_low_s16(d0)), vget_low_s16(d1), vget_low_s16(d1));
                    let hi = vmlal_high_s16(vmull_high_s16(d0, d0), d1, d1);
                    acc[2 * i] = vaddq_s32(acc[2 * i], lo);
                    acc[2 * i + 1] = vaddq_s32(acc[2 * i + 1], hi);
                }
                c += 2;
            }
        }
        let mut out = [0u64; 33];
        for (i, v) in acc.iter().enumerate() {
            let mut lanes = [0i32; 4];
            vst1q_s32(lanes.as_mut_ptr(), *v);
            for j in 0..4 {
                out[4 * i + j] = lanes[j] as u64;
            }
        }
        out
    }
}

/// NEON twin of `sse_avx2`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn sse_neon(src: &[u8], ss: usize, pred: &[u8], ps: usize, w: usize, h: usize, ac: &[i16], alpha: i32) -> u64 {
    use std::arch::aarch64::*;
    // A CfL block is at most 32x32: even a single lane accumulating every
    // squared 8-bit difference fits i32 (1024 * 65025 < 2^31).
    let alpha_v = vdupq_n_s32(alpha);
    let (c32, c255, zero) = (vdupq_n_u32(32), vdupq_n_s32(255), vdupq_n_s32(0));
    let mut sums = vdupq_n_u32(0);
    let mut tail = 0u64;
    for r in 0..h {
        let mut c = 0;
        while c + 8 <= w {
            // SAFETY: NEON is enabled for this function. `c + 8 <= w`, so the 8 lanes of `ac` and
            // the 8 bytes of `pred` and `src` read here lie inside the rows the caller asserted.
            unsafe {
                let l16 = vld1q_s16(ac.as_ptr().add(r * w + c));
                let p16 = vmovl_u8(vld1_u8(pred.as_ptr().add(r * ps + c)));
                let s16 = vmovl_u8(vld1_u8(src.as_ptr().add(r * ss + c)));
                for half in 0..2 {
                    let (l, dc, s) = if half == 0 {
                        (vmovl_s16(vget_low_s16(l16)), vmovl_u16(vget_low_u16(p16)), vmovl_u16(vget_low_u16(s16)))
                    } else {
                        (vmovl_high_s16(l16), vmovl_high_u16(p16), vmovl_high_u16(s16))
                    };
                    let v = vmulq_s32(l, alpha_v);
                    let rounded =
                        vreinterpretq_s32_u32(vshrq_n_u32::<6>(vaddq_u32(vreinterpretq_u32_s32(vabsq_s32(v)), c32)));
                    let scaled = vbslq_s32(vcltq_s32(v, zero), vnegq_s32(rounded), rounded);
                    let p = vminq_s32(c255, vmaxq_s32(zero, vaddq_s32(vreinterpretq_s32_u32(dc), scaled)));
                    let d = vsubq_s32(vreinterpretq_s32_u32(s), p);
                    sums = vaddq_u32(sums, vreinterpretq_u32_s32(vmulq_s32(d, d)));
                }
            }
            c += 8;
        }
        for c in c..w {
            let d = src[r * ss + c] as i32 - pixel(pred[r * ps + c], ac[r * w + c], alpha) as i32;
            tail += (d * d) as u64;
        }
    }
    tail + vaddlvq_u32(sums)
}

#[allow(clippy::too_many_arguments)]
fn sse_scalar(src: &[u8], ss: usize, pred: &[u8], ps: usize, w: usize, h: usize, ac: &[i16], alpha: i32) -> u64 {
    let mut sum = 0;
    for r in 0..h {
        for c in 0..w {
            let d = src[r * ss + c] as i32 - pixel(pred[r * ps + c], ac[r * w + c], alpha) as i32;
            sum += (d * d) as u64;
        }
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn sse_avx2(src: &[u8], ss: usize, pred: &[u8], ps: usize, w: usize, h: usize, ac: &[i16], alpha: i32) -> u64 {
    use std::arch::x86_64::*;
    // A CfL block is at most 32x32: even a single lane accumulating every
    // squared 8-bit difference fits i32 (1024 * 65025 < 2^31).
    let mut sums = _mm256_setzero_si256();
    let mut tail = 0u64;
    for r in 0..h {
        let mut c = 0;
        while c + 8 <= w {
            // SAFETY: AVX2 is enabled for this function. `c + 8 <= w`, so the 8 lanes of `ac` and
            // the 8 bytes of `pred` and `src` read here lie inside the rows the caller asserted.
            unsafe {
                let l = _mm256_cvtepi16_epi32(_mm_loadu_si128(ac.as_ptr().add(r * w + c).cast()));
                let v = _mm256_mullo_epi32(l, _mm256_set1_epi32(alpha));
                let rounded = _mm256_srli_epi32::<6>(_mm256_add_epi32(_mm256_abs_epi32(v), _mm256_set1_epi32(32)));
                let scaled = _mm256_sign_epi32(rounded, v);
                let dc = _mm256_cvtepu8_epi32(_mm_loadl_epi64(pred.as_ptr().add(r * ps + c).cast()));
                let p = _mm256_min_epi32(
                    _mm256_set1_epi32(255),
                    _mm256_max_epi32(_mm256_setzero_si256(), _mm256_add_epi32(dc, scaled)),
                );
                let s = _mm256_cvtepu8_epi32(_mm_loadl_epi64(src.as_ptr().add(r * ss + c).cast()));
                let d = _mm256_sub_epi32(s, p);
                sums = _mm256_add_epi32(sums, _mm256_mullo_epi32(d, d));
            }
            c += 8;
        }
        for c in c..w {
            let d = src[r * ss + c] as i32 - pixel(pred[r * ps + c], ac[r * w + c], alpha) as i32;
            tail += (d * d) as u64;
        }
    }
    let mut lanes = [0i32; 8];
    // SAFETY: AVX2 is enabled for this function; `lanes` holds 8 i32, the width of the store.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sums) };
    tail + lanes.iter().map(|&v| v as u64).sum::<u64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_matches_reference() {
        for w in [4usize, 8, 16, 32] {
            for h in [4, 8, 16, 32] {
                let ss = w + 3;
                let ps = w + 7;
                // No padding after the final row. Vary DC, including clipping.
                let src: Vec<_> = (0..(h - 1) * ss + w).map(|i| (i * 137 + 19) as u8).collect();
                let pred: Vec<_> = (0..(h - 1) * ps + w).map(|i| (i * 71) as u8).collect();
                for pattern in 0..4 {
                    let mut luma: Vec<i16> = (0..w * h)
                        .map(|i| match pattern {
                            0 => 0,
                            1 => 2040,
                            2 => {
                                if i % 2 == 0 {
                                    0
                                } else {
                                    2040
                                }
                            }
                            _ => ((i * 997 + 31) % 2041) as i16,
                        })
                        .collect();
                    let original = luma.clone();
                    ac(&mut luma);
                    let shift = (w * h).trailing_zeros();
                    let avg = (original.iter().map(|&v| v as i32).sum::<i32>() + (1 << (shift - 1))) >> shift;
                    for alpha in -16..=16 {
                        let mut expected = 0u64;
                        for r in 0..h {
                            for c in 0..w {
                                let v = alpha * (original[r * w + c] as i32 - avg);
                                let v = if v < 0 { -((-v + 32) >> 6) } else { (v + 32) >> 6 };
                                let p = (pred[r * ps + c] as i32 + v).clamp(0, 255);
                                let d = src[r * ss + c] as i32 - p;
                                expected += (d * d) as u64;
                            }
                        }
                        assert_eq!(sse_scalar(&src, ss, &pred, ps, w, h, &luma, alpha), expected);
                        assert_eq!(sse(&src, ss, &pred, ps, w, h, &luma, alpha), expected);
                        assert_eq!(sse_all(&src, ss, &pred, ps, w, h, &luma)[(alpha + 16) as usize], expected);
                    }
                }
            }
        }
    }
}
