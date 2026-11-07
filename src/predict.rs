//! Intra prediction, bit-exact with libaom's `av1_predict_intra_block`.

mod cfl;
#[cfg(test)]
mod direction_tests;
#[cfg(test)]
mod tests;
pub(crate) use cfl::{ac as cfl_ac, sse_all as cfl_sse_all};

// Keep one arithmetic implementation; inline it into an AVX2 context so LLVM
// can vectorize min/max/abs without raising the baseline CPU requirement.
macro_rules! dispatch_prediction {
    ($name:ident, $simd:ident, $scalar:ident, ($($arg:ident: $ty:ty),*)) => {
        fn $name($($arg: $ty),*) {
            #[cfg(target_arch = "x86_64")]
            if crate::has_avx2() {
                // SAFETY: runtime detection respects AV1_NO_AVX2.
                return unsafe { $simd($($arg),*) };
            }
            $scalar($($arg),*)
        }
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        unsafe fn $simd($($arg: $ty),*) {
            $scalar($($arg),*)
        }
    };
}

dispatch_prediction!(smooth, smooth_avx2, smooth_scalar, (dst: &mut [u8], stride: usize, w: usize, h: usize, above: &[u8], left: &[u8]));
dispatch_prediction!(smooth_v, smooth_v_avx2, smooth_v_scalar, (dst: &mut [u8], stride: usize, w: usize, h: usize, above: &[u8], left: &[u8]));
dispatch_prediction!(smooth_h, smooth_h_avx2, smooth_h_scalar, (dst: &mut [u8], stride: usize, w: usize, h: usize, above: &[u8], left: &[u8]));
dispatch_prediction!(paeth, paeth_avx2, paeth_scalar, (dst: &mut [u8], stride: usize, w: usize, h: usize, above: &[u8], left: &[u8], top_left: u8));
dispatch_prediction!(cfl_apply, cfl_apply_avx2, cfl_apply_scalar, (dst: &mut [u8], stride: usize, w: usize, h: usize, luma_q3: &[i16], alpha_q3: i32));

use crate::consts::*;
use crate::tables::misc::{DR_INTRA_DERIVATIVE, FILTER_INTRA_TAPS, SMOOTH_WEIGHTS};

/// Which neighbouring samples exist for one transform block (libaom's
/// n_top_px / n_topright_px / n_left_px / n_bottomleft_px).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeAvail {
    pub n_top: usize,
    /// Number of available top-right samples, or 0 when not available.
    pub n_topright: usize,
    pub n_left: usize,
    pub n_bottomleft: usize,
}

const NEED_LEFT: u8 = 1 << 1;
const NEED_ABOVE: u8 = 1 << 2;
const NEED_ABOVELEFT: u8 = 1 << 5;
const EXTEND_MODES: [u8; 13] = [
    NEED_ABOVE | NEED_LEFT,
    NEED_ABOVE,
    NEED_LEFT,
    NEED_ABOVE,
    NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,
    NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,
    NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,
    NEED_LEFT,
    NEED_ABOVE,
    NEED_LEFT | NEED_ABOVE,
    NEED_LEFT | NEED_ABOVE,
    NEED_LEFT | NEED_ABOVE,
    NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,
];

const EDGE_OFF: usize = 16;
const EDGE_LEN: usize = 64 * 2 + 32 + 32;

#[inline]
fn fill(dst: &mut [u8], stride: usize, w: usize, h: usize, v: u8) {
    for r in 0..h {
        dst[r * stride..r * stride + w].fill(v);
    }
}

pub struct IntraParams {
    pub mode: u8,
    pub angle_delta: i32,
    pub filter_intra: Option<u8>,
    pub edge_filter: bool,
    /// Above or left neighbour uses a smooth mode (intra edge filter type).
    pub smooth_neighbor: bool,
}

/// Predict a `tx` sized block at `(x, y)` of `plane` (`frame` / `stride`)
/// into `dst`.
pub fn predict(
    frame: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    tx: TxSize,
    p: &IntraParams,
    av: EdgeAvail,
    dst: &mut [u8],
    dst_stride: usize,
) {
    let w = tx.width();
    let h = tx.height();
    let at = |xx: usize, yy: usize| frame[yy * stride + xx];
    let is_dr = is_directional(p.mode);
    let use_fi = p.filter_intra.is_some();
    let mut need_left = EXTEND_MODES[p.mode as usize] & NEED_LEFT != 0;
    let mut need_above = EXTEND_MODES[p.mode as usize] & NEED_ABOVE != 0;
    let mut need_above_left = EXTEND_MODES[p.mode as usize] & NEED_ABOVELEFT != 0;
    let p_angle = MODE_TO_ANGLE[p.mode as usize] + p.angle_delta * 3;
    if is_dr {
        (need_above, need_left, need_above_left) = if p_angle <= 90 {
            (true, false, true)
        } else if p_angle < 180 {
            (true, true, true)
        } else {
            (false, true, true)
        };
    }
    if use_fi {
        need_left = true;
        need_above = true;
        need_above_left = true;
    }
    let (n_top, n_left) = (av.n_top, av.n_left);
    if (!need_above && n_left == 0) || (!need_left && n_top == 0) {
        let v = if need_left {
            if n_top > 0 { at(x, y - 1) } else { 129 }
        } else if n_left > 0 {
            at(x - 1, y)
        } else {
            127
        };
        fill(dst, dst_stride, w, h, v);
        return;
    }

    let mut above_data = [127u8; EDGE_LEN];
    let mut left_data = [129u8; EDGE_LEN];
    let need_right = is_dr && p_angle < 90;
    let need_bottom = is_dr && p_angle > 180;

    if need_left {
        let needed = h + if need_bottom { w } else { 0 };
        let left = &mut left_data[EDGE_OFF..];
        if n_left > 0 {
            let mut i = 0;
            while i < n_left {
                left[i] = at(x - 1, y + i);
                i += 1;
            }
            if need_bottom && av.n_bottomleft > 0 {
                while i < h + av.n_bottomleft {
                    left[i] = at(x - 1, y + i);
                    i += 1;
                }
            }
            if i < needed {
                let v = left[i - 1];
                left[i..needed].fill(v);
            }
        } else if n_top > 0 {
            left[..needed].fill(at(x, y - 1));
        }
    }
    if need_above {
        let needed = w + if need_right { h } else { 0 };
        let above = &mut above_data[EDGE_OFF..];
        if n_top > 0 {
            above[..n_top].copy_from_slice(&frame[(y - 1) * stride + x..(y - 1) * stride + x + n_top]);
            let mut i = n_top;
            if need_right && av.n_topright > 0 {
                above[w..w + av.n_topright]
                    .copy_from_slice(&frame[(y - 1) * stride + x + w..(y - 1) * stride + x + w + av.n_topright]);
                i += av.n_topright;
            }
            if i < needed {
                let v = above[i - 1];
                above[i..needed].fill(v);
            }
        } else if n_left > 0 {
            above[..needed].fill(at(x - 1, y));
        }
    }
    if need_above_left {
        let v = if n_top > 0 && n_left > 0 {
            at(x - 1, y - 1)
        } else if n_top > 0 {
            at(x, y - 1)
        } else if n_left > 0 {
            at(x - 1, y)
        } else {
            128
        };
        above_data[EDGE_OFF - 1] = v;
        left_data[EDGE_OFF - 1] = v;
    }

    if let Some(fm) = p.filter_intra {
        filter_intra(dst, dst_stride, w, h, &above_data, &left_data, fm as usize);
        return;
    }
    if !is_dr {
        let above = &above_data[EDGE_OFF..EDGE_OFF + w];
        let left = &left_data[EDGE_OFF..EDGE_OFF + h];
        match p.mode {
            DC_PRED => dc(dst, dst_stride, w, h, above, left, n_top > 0, n_left > 0),
            SMOOTH_PRED => smooth(dst, dst_stride, w, h, above, left),
            SMOOTH_V_PRED => smooth_v(dst, dst_stride, w, h, above, left),
            SMOOTH_H_PRED => smooth_h(dst, dst_stride, w, h, above, left),
            _ => paeth(dst, dst_stride, w, h, above, left, above_data[EDGE_OFF - 1]),
        }
        return;
    }

    let mut upsample_above = false;
    let mut upsample_left = false;
    if p.edge_filter {
        let ty = p.smooth_neighbor;
        if p_angle != 90 && p_angle != 180 {
            if need_above && need_left && w + h >= 24 {
                // filter_intra_edge_corner
                let s = left_data[EDGE_OFF] as i32 * 5
                    + above_data[EDGE_OFF - 1] as i32 * 6
                    + above_data[EDGE_OFF] as i32 * 5;
                let s = ((s + 8) >> 4) as u8;
                above_data[EDGE_OFF - 1] = s;
                left_data[EDGE_OFF - 1] = s;
            }
            if need_above && n_top > 0 {
                let strength = edge_filter_strength(w, h, p_angle - 90, ty);
                let n_px = n_top + 1 + if need_right { h } else { 0 };
                filter_edge(&mut above_data[EDGE_OFF - 1..], n_px, strength);
            }
            if need_left && n_left > 0 {
                let strength = edge_filter_strength(h, w, p_angle - 180, ty);
                let n_px = n_left + 1 + if need_bottom { w } else { 0 };
                filter_edge(&mut left_data[EDGE_OFF - 1..], n_px, strength);
            }
        }
        upsample_above = use_upsample(w, h, p_angle - 90, ty);
        if need_above && upsample_above {
            let n_px = w + if need_right { h } else { 0 };
            upsample_edge(&mut above_data, n_px);
        }
        upsample_left = use_upsample(h, w, p_angle - 180, ty);
        if need_left && upsample_left {
            let n_px = h + if need_bottom { w } else { 0 };
            upsample_edge(&mut left_data, n_px);
        }
    }
    let ua = upsample_above as usize;
    let ul = upsample_left as usize;
    directional(dst, dst_stride, w, h, &above_data, &left_data, p_angle, ua, ul);
}

#[allow(clippy::too_many_arguments)]
fn directional(
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
    above_data: &[u8],
    left_data: &[u8],
    p_angle: i32,
    ua: usize,
    ul: usize,
) {
    // Indexing helpers: index -2.. relative to EDGE_OFF.
    let above = |i: isize| above_data[(EDGE_OFF as isize + i) as usize] as i32;
    let left = |i: isize| left_data[(EDGE_OFF as isize + i) as usize] as i32;
    if p_angle < 90 {
        let dx = DR_INTRA_DERIVATIVE[p_angle as usize] as i32;
        let max_base_x = ((w + h - 1) << ua) as i32;
        let frac_bits = 6 - ua as i32;
        let base_inc = 1 << ua;
        let mut xx = dx;
        for r in 0..h {
            let row = &mut dst[r * dst_stride..r * dst_stride + w];
            let base = xx >> frac_bits;
            let shift = ((xx << ua) & 0x3F) >> 1;
            if base >= max_base_x {
                let v = above(max_base_x as isize) as u8;
                for rr in r..h {
                    dst[rr * dst_stride..rr * dst_stride + w].fill(v);
                }
                return;
            }
            let n = w.min(((max_base_x - base + base_inc - 1) / base_inc) as usize);
            let edge = &above_data[EDGE_OFF + base as usize..];
            if ua == 0 {
                interpolate::<1>(&mut row[..n], edge, shift);
            } else {
                interpolate::<2>(&mut row[..n], edge, shift);
            }
            row[n..].fill(above(max_base_x as isize) as u8);
            xx += dx;
        }
    } else if p_angle > 90 && p_angle < 180 {
        let dx = DR_INTRA_DERIVATIVE[(180 - p_angle) as usize] as i32;
        let dy = DR_INTRA_DERIVATIVE[(p_angle - 90) as usize] as i32;
        let fbx = 6 - ua as i32;
        let fby = 6 - ul as i32;
        for r in 0..h {
            let row = &mut dst[r * dst_stride..r * dst_stride + w];
            // The reference switches from left to above exactly once per row.
            // base_x >= -(1 << ua) iff c * 64 >= (r + 1) * dx - 64.
            let start = ((((r as i32 + 1) * dx - 1) >> 6) as usize).min(w);
            for (c, v) in row[..start].iter_mut().enumerate() {
                let yy = ((r as i32) << 6) - (c as i32 + 1) * dy;
                let base_y = yy >> fby;
                let shift = ((yy << ul) & 0x3F) >> 1;
                let val = left(base_y as isize) * (32 - shift) + left(base_y as isize + 1) * shift;
                *v = ((val + 16) >> 5) as u8;
            }
            if start < w {
                let xx = ((start as i32) << 6) - (r as i32 + 1) * dx;
                let base = (EDGE_OFF as i32 + (xx >> fbx)) as usize;
                let shift = ((xx << ua) & 0x3F) >> 1;
                if ua == 0 {
                    interpolate::<1>(&mut row[start..], &above_data[base..], shift);
                } else {
                    interpolate::<2>(&mut row[start..], &above_data[base..], shift);
                }
            }
        }
    } else if p_angle > 180 {
        let dy = DR_INTRA_DERIVATIVE[(270 - p_angle) as usize] as i32;
        let max_base_y = ((w + h - 1) << ul) as i32;
        let frac_bits = 6 - ul as i32;
        let base_inc = 1 << ul;
        let mut yy = dy;
        for c in 0..w {
            let mut base = yy >> frac_bits;
            let shift = ((yy << ul) & 0x3F) >> 1;
            let mut r = 0;
            while r < h {
                if base < max_base_y {
                    let val = left(base as isize) * (32 - shift) + left(base as isize + 1) * shift;
                    dst[r * dst_stride + c] = ((val + 16) >> 5) as u8;
                } else {
                    let v = left(max_base_y as isize) as u8;
                    while r < h {
                        dst[r * dst_stride + c] = v;
                        r += 1;
                    }
                    break;
                }
                base += base_inc;
                r += 1;
            }
            yy += dy;
        }
    } else if p_angle == 90 {
        for r in 0..h {
            dst[r * dst_stride..r * dst_stride + w].copy_from_slice(&above_data[EDGE_OFF..EDGE_OFF + w]);
        }
    } else {
        for r in 0..h {
            dst[r * dst_stride..r * dst_stride + w].fill(left_data[EDGE_OFF + r]);
        }
    }
}

#[inline(always)]
fn interpolate<const STEP: usize>(out: &mut [u8], edge: &[u8], shift: i32) {
    debug_assert!(STEP == 1 || STEP == 2);
    if out.is_empty() {
        return;
    }
    let edge = &edge[..(out.len() - 1) * STEP + 2];
    let mut i = 0;
    #[cfg(target_arch = "x86_64")]
    if crate::has_avx2() {
        // SAFETY: AVX2 was detected; the helper only accesses complete chunks.
        i = unsafe { interpolate_avx2::<STEP>(out, edge, shift) };
    }
    for j in i..out.len() {
        let a = edge[j * STEP] as i32;
        let b = edge[j * STEP + 1] as i32;
        out[j] = ((a * (32 - shift) + b * shift + 16) >> 5) as u8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn interpolate_avx2<const STEP: usize>(out: &mut [u8], edge: &[u8], shift: i32) -> usize {
    use std::arch::x86_64::*;
    let mut i = 0;
    let wa = _mm_set1_epi16((32 - shift) as i16);
    let wb = _mm_set1_epi16(shift as i16);
    // Eight adjacent or interleaved pairs; no read beyond the edge slice.
    while i + 8 <= out.len() {
        let pos = i * STEP;
        let (a, b) = if STEP == 1 {
            let a = unsafe { _mm_loadl_epi64(edge.as_ptr().add(pos).cast()) };
            let b = unsafe { _mm_loadl_epi64(edge.as_ptr().add(pos + 1).cast()) };
            (_mm_cvtepu8_epi16(a), _mm_cvtepu8_epi16(b))
        } else {
            let pairs = unsafe { _mm_loadu_si128(edge.as_ptr().add(pos).cast()) };
            (_mm_and_si128(pairs, _mm_set1_epi16(255)), _mm_srli_epi16::<8>(pairs))
        };
        let sum = _mm_add_epi16(_mm_mullo_epi16(a, wa), _mm_mullo_epi16(b, wb));
        let rounded = _mm_srli_epi16::<5>(_mm_add_epi16(sum, _mm_set1_epi16(16)));
        let pixels = _mm_packus_epi16(rounded, rounded);
        unsafe { _mm_storel_epi64(out.as_mut_ptr().add(i).cast(), pixels) };
        i += 8;
    }
    i
}

fn edge_filter_strength(bs0: usize, bs1: usize, delta: i32, smooth: bool) -> usize {
    let d = delta.abs();
    let blk_wh = bs0 + bs1;
    let mut s = 0;
    if !smooth {
        if blk_wh <= 8 {
            if d >= 56 {
                s = 1;
            }
        } else if blk_wh <= 16 {
            if d >= 40 {
                s = 1;
            }
        } else if blk_wh <= 24 {
            if d >= 8 {
                s = 1;
            }
            if d >= 16 {
                s = 2;
            }
            if d >= 32 {
                s = 3;
            }
        } else if blk_wh <= 32 {
            if d >= 1 {
                s = 1;
            }
            if d >= 4 {
                s = 2;
            }
            if d >= 32 {
                s = 3;
            }
        } else if d >= 1 {
            s = 3;
        }
    } else if blk_wh <= 8 {
        if d >= 40 {
            s = 1;
        }
        if d >= 64 {
            s = 2;
        }
    } else if blk_wh <= 16 {
        if d >= 20 {
            s = 1;
        }
        if d >= 48 {
            s = 2;
        }
    } else if blk_wh <= 24 {
        if d >= 4 {
            s = 3;
        }
    } else if d >= 1 {
        s = 3;
    }
    s
}

fn use_upsample(bs0: usize, bs1: usize, delta: i32, smooth: bool) -> bool {
    let d = delta.abs();
    let blk_wh = bs0 + bs1;
    if d == 0 || d >= 40 {
        return false;
    }
    if smooth { blk_wh <= 8 } else { blk_wh <= 16 }
}

fn filter_edge(p: &mut [u8], sz: usize, strength: usize) {
    if strength == 0 {
        return;
    }
    const KERNEL: [[i32; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
    let k = &KERNEL[strength - 1];
    // Replicate the end samples so that every tap reads in range without clamping.
    let mut edge = [0u8; 160 + 4];
    edge[0] = p[0];
    edge[1] = p[0];
    edge[2..sz + 2].copy_from_slice(&p[..sz]);
    edge[sz + 2] = p[sz - 1];
    edge[sz + 3] = p[sz - 1];
    let (e0, e1, e2, e3, e4) = (&edge[0..], &edge[1..], &edge[2..], &edge[3..], &edge[4..]);
    for i in 1..sz {
        let s =
            e0[i] as i32 * k[0] + e1[i] as i32 * k[1] + e2[i] as i32 * k[2] + e3[i] as i32 * k[3] + e4[i] as i32 * k[4];
        p[i] = ((s + 8) >> 4) as u8;
    }
}

/// Upsample the edge starting at data[EDGE_OFF - 1] (p[-1]) for `sz` samples.
fn upsample_edge(data: &mut [u8], sz: usize) {
    let p0 = EDGE_OFF as isize;
    let get = |d: &[u8], i: isize| d[(p0 + i) as usize] as i32;
    let mut inb = [0i32; 64 + 3];
    inb[0] = get(data, -1);
    inb[1] = get(data, -1);
    for i in 0..sz {
        inb[i + 2] = get(data, i as isize);
    }
    inb[sz + 2] = get(data, sz as isize - 1);
    data[(p0 - 2) as usize] = inb[0] as u8;
    for i in 0..sz {
        let s = -inb[i] + 9 * inb[i + 1] + 9 * inb[i + 2] - inb[i + 3];
        let s = ((s + 8) >> 4).clamp(0, 255);
        data[(p0 + 2 * i as isize - 1) as usize] = s as u8;
        data[(p0 + 2 * i as isize) as usize] = inb[i + 2] as u8;
    }
}

dispatch_prediction!(filter_intra, filter_intra_avx2, filter_intra_scalar, (dst: &mut [u8], stride: usize, w: usize, h: usize, above: &[u8], left: &[u8], mode: usize));

// Each lane is one of the eight outputs in a dependent 4x2 tile. Transpose
// at compile time, rather than gathering seven strided taps per output.
const FILTER_TAPS_LANES: [[[i32; 8]; 7]; 5] = {
    let mut taps = [[[0; 8]; 7]; 5];
    let mut m = 0;
    while m < 5 {
        let mut i = 0;
        while i < 7 {
            let mut k = 0;
            while k < 8 {
                taps[m][i][k] = FILTER_INTRA_TAPS[m * 56 + k * 7 + i] as i32;
                k += 1;
            }
            i += 1;
        }
        m += 1;
    }
    taps
};

#[inline(always)]
fn filter_intra_scalar(dst: &mut [u8], stride: usize, w: usize, h: usize, above: &[u8], left: &[u8], mode: usize) {
    let mut buf = [[0u8; 33]; 33];
    for r in 0..h {
        buf[r + 1][0] = left[EDGE_OFF + r];
    }
    buf[0][..w + 1].copy_from_slice(&above[EDGE_OFF - 1..EDGE_OFF + w]);
    let taps = &FILTER_TAPS_LANES[mode];
    for r in (1..h + 1).step_by(2) {
        for c in (1..w + 1).step_by(4) {
            let p = [
                buf[r - 1][c - 1] as i32,
                buf[r - 1][c] as i32,
                buf[r - 1][c + 1] as i32,
                buf[r - 1][c + 2] as i32,
                buf[r - 1][c + 3] as i32,
                buf[r][c - 1] as i32,
                buf[r + 1][c - 1] as i32,
            ];
            let mut values = [8i32; 8];
            for i in 0..7 {
                for k in 0..8 {
                    values[k] += taps[i][k] * p[i];
                }
            }
            for k in 0..8 {
                buf[r + (k >> 2)][c + (k & 3)] = (values[k] >> 4).clamp(0, 255) as u8;
            }
        }
    }
    for r in 0..h {
        dst[r * stride..r * stride + w].copy_from_slice(&buf[r + 1][1..w + 1]);
    }
}

#[allow(clippy::too_many_arguments)]
fn dc(dst: &mut [u8], stride: usize, w: usize, h: usize, above: &[u8], left: &[u8], has_top: bool, has_left: bool) {
    let v = match (has_top, has_left) {
        (true, true) => {
            let sum: u32 = above.iter().map(|&v| v as u32).sum::<u32>() + left.iter().map(|&v| v as u32).sum::<u32>();
            ((sum + ((w + h) as u32 >> 1)) / (w + h) as u32) as u8
        }
        (true, false) => {
            let sum: u32 = above.iter().map(|&v| v as u32).sum();
            ((sum + (w as u32 >> 1)) >> w.trailing_zeros()) as u8
        }
        (false, true) => {
            let sum: u32 = left.iter().map(|&v| v as u32).sum();
            ((sum + (h as u32 >> 1)) >> h.trailing_zeros()) as u8
        }
        (false, false) => 128,
    };
    fill(dst, stride, w, h, v);
}

#[inline]
fn weights(n: usize) -> &'static [u8] {
    &SMOOTH_WEIGHTS[n - 4..n - 4 + n]
}

#[inline(always)]
fn smooth_scalar(dst: &mut [u8], stride: usize, w: usize, h: usize, above: &[u8], left: &[u8]) {
    let below = left[h - 1] as u32;
    let right = above[w - 1] as u32;
    let ww = weights(w);
    let wh = weights(h);
    let above = &above[..w];
    for r in 0..h {
        let row = &mut dst[r * stride..r * stride + w];
        for c in 0..w {
            let p = wh[r] as u32 * above[c] as u32
                + (256 - wh[r] as u32) * below
                + ww[c] as u32 * left[r] as u32
                + (256 - ww[c] as u32) * right;
            row[c] = ((p + 256) >> 9) as u8;
        }
    }
}

#[inline(always)]
fn smooth_v_scalar(dst: &mut [u8], stride: usize, w: usize, h: usize, above: &[u8], left: &[u8]) {
    let below = left[h - 1] as u32;
    let wh = weights(h);
    let above = &above[..w];
    for r in 0..h {
        let row = &mut dst[r * stride..r * stride + w];
        for c in 0..w {
            let p = wh[r] as u32 * above[c] as u32 + (256 - wh[r] as u32) * below;
            row[c] = ((p + 128) >> 8) as u8;
        }
    }
}

#[inline(always)]
fn smooth_h_scalar(dst: &mut [u8], stride: usize, w: usize, h: usize, above: &[u8], left: &[u8]) {
    let right = above[w - 1] as u32;
    let ww = weights(w);
    for r in 0..h {
        let row = &mut dst[r * stride..r * stride + w];
        for c in 0..w {
            let p = ww[c] as u32 * left[r] as u32 + (256 - ww[c] as u32) * right;
            row[c] = ((p + 128) >> 8) as u8;
        }
    }
}

#[inline(always)]
fn paeth_scalar(dst: &mut [u8], stride: usize, w: usize, h: usize, above: &[u8], left: &[u8], top_left: u8) {
    let tl = top_left as i32;
    let above = &above[..w];
    for r in 0..h {
        let row = &mut dst[r * stride..r * stride + w];
        for c in 0..w {
            let t = above[c] as i32;
            let l = left[r] as i32;
            let base = t + l - tl;
            let pl = (base - l).abs();
            let pt = (base - t).abs();
            let ptl = (base - tl).abs();
            row[c] = if pl <= pt && pl <= ptl {
                l as u8
            } else if pt <= ptl {
                t as u8
            } else {
                tl as u8
            };
        }
    }
}

/// Chroma-from-luma: `dst` holds the DC prediction; `luma_q3` the
/// subsampled luma (already padded to the transform size, in Q3).
pub fn cfl_predict(dst: &mut [u8], stride: usize, w: usize, h: usize, luma_q3: &[i16], alpha_q3: i32) {
    cfl_apply(dst, stride, w, h, luma_q3, alpha_q3);
}

#[inline(always)]
fn cfl_apply_scalar(dst: &mut [u8], stride: usize, w: usize, h: usize, luma_q3: &[i16], alpha_q3: i32) {
    let n = w * h;
    let sum: i32 = luma_q3[..n].iter().map(|&v| v as i32).sum();
    let shift = (w * h).trailing_zeros();
    let avg = (sum + (1 << (shift - 1))) >> shift;
    for r in 0..h {
        let row = &mut dst[r * stride..r * stride + w];
        let luma = &luma_q3[r * w..(r + 1) * w];
        for c in 0..w {
            let ac = luma[c] as i32 - avg;
            let scaled = alpha_q3 * ac;
            let scaled = if scaled < 0 { -((-scaled + 32) >> 6) } else { (scaled + 32) >> 6 };
            let p = &mut row[c];
            *p = (*p as i32 + scaled).clamp(0, 255) as u8;
        }
    }
}
