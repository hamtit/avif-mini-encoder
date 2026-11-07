//! Deblocking loop filter (libaom av1_loopfilter.c / aom_dsp/loopfilter.c)
//! and the filter level search (picklpf.c, LPF_PICK_FROM_FULL_IMAGE_NON_DUAL).

use crate::consts::*;
use crate::enc::Encoder;
use crate::enc::block::uv_tx_size;
use crate::frame::Plane;
use crate::obu::LoopFilterParams;

const MAX_LOOP_FILTER: i32 = 63;

#[derive(Clone, Copy)]
struct Thresh {
    mblim: i32,
    lim: i32,
    hev: i32,
}

fn thresholds(level: i32, sharpness: i32) -> Thresh {
    let mut lim = level >> ((sharpness > 0) as i32 + (sharpness > 4) as i32);
    if sharpness > 0 && lim > 9 - sharpness {
        lim = 9 - sharpness;
    }
    lim = lim.max(1);
    Thresh { mblim: 2 * (level + 2) + lim, lim, hev: level >> 4 }
}

#[inline(always)]
fn sclamp(t: i32) -> i32 {
    t.clamp(-128, 127)
}

#[inline(always)]
fn filter4(mask: bool, hev_thr: i32, p: &mut [i32; 8]) {
    // p[0..4] = p3 p2 p1 p0 ; p[4..8] = q0 q1 q2 q3 (only p1 p0 q0 q1 used)
    let (op1, op0, oq0, oq1) = (p[2], p[3], p[4], p[5]);
    let ps1 = op1 - 128;
    let ps0 = op0 - 128;
    let qs0 = oq0 - 128;
    let qs1 = oq1 - 128;
    let hev = (op1 - op0).abs() > hev_thr || (oq1 - oq0).abs() > hev_thr;
    let mut filter = if hev { sclamp(ps1 - qs1) } else { 0 };
    filter = if mask { sclamp(filter + 3 * (qs0 - ps0)) } else { 0 };
    let filter1 = sclamp(filter + 4) >> 3;
    let filter2 = sclamp(filter + 3) >> 3;
    p[4] = sclamp(qs0 - filter1) + 128;
    p[3] = sclamp(ps0 + filter2) + 128;
    let f = if hev { 0 } else { (filter1 + 1) >> 1 };
    p[5] = sclamp(qs1 - f) + 128;
    p[2] = sclamp(ps1 + f) + 128;
}

#[inline(always)]
fn r3(v: i32) -> i32 {
    (v + 4) >> 3
}
#[inline(always)]
fn r4(v: i32) -> i32 {
    (v + 8) >> 4
}

/// Filter one line across an edge. `get(i)` reads sample at signed offset i
/// from the edge (q0 = 0, p0 = -1); `set` writes.
#[inline(always)]
fn filter_line(len: usize, t: Thresh, s: &mut [u8], base: isize, step: isize) {
    let at = |s: &[u8], i: isize| s[(base + i * step) as usize] as i32;
    let ad = |a: i32, b: i32| (a - b).abs();
    match len {
        4 => {
            let (p1, p0, q0, q1) = (at(s, -2), at(s, -1), at(s, 0), at(s, 1));
            let mask = ad(p1, p0) <= t.lim && ad(q1, q0) <= t.lim && ad(p0, q0) * 2 + ad(p1, q1) / 2 <= t.mblim;
            let mut v = [0, 0, p1, p0, q0, q1, 0, 0];
            filter4(mask, t.hev, &mut v);
            for (k, i) in (-2..2).enumerate() {
                s[(base + i * step) as usize] = v[k + 2] as u8;
            }
        }
        6 => {
            let (p2, p1, p0, q0, q1, q2) = (at(s, -3), at(s, -2), at(s, -1), at(s, 0), at(s, 1), at(s, 2));
            let mask = ad(p2, p1) <= t.lim
                && ad(p1, p0) <= t.lim
                && ad(q1, q0) <= t.lim
                && ad(q2, q1) <= t.lim
                && ad(p0, q0) * 2 + ad(p1, q1) / 2 <= t.mblim;
            let flat = ad(p1, p0) <= 1 && ad(q1, q0) <= 1 && ad(p2, p0) <= 1 && ad(q2, q0) <= 1;
            if flat && mask {
                let o = [
                    r3(p2 * 3 + p1 * 2 + p0 * 2 + q0),
                    r3(p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1),
                    r3(p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2),
                    r3(p0 + q0 * 2 + q1 * 2 + q2 * 3),
                ];
                for (k, i) in (-2..2).enumerate() {
                    s[(base + i * step) as usize] = o[k] as u8;
                }
            } else {
                let mut v = [0, 0, p1, p0, q0, q1, 0, 0];
                filter4(mask, t.hev, &mut v);
                for (k, i) in (-2..2).enumerate() {
                    s[(base + i * step) as usize] = v[k + 2] as u8;
                }
            }
        }
        _ => {
            let (p3, p2, p1, p0) = (at(s, -4), at(s, -3), at(s, -2), at(s, -1));
            let (q0, q1, q2, q3) = (at(s, 0), at(s, 1), at(s, 2), at(s, 3));
            let mask = ad(p3, p2) <= t.lim
                && ad(p2, p1) <= t.lim
                && ad(p1, p0) <= t.lim
                && ad(q1, q0) <= t.lim
                && ad(q2, q1) <= t.lim
                && ad(q3, q2) <= t.lim
                && ad(p0, q0) * 2 + ad(p1, q1) / 2 <= t.mblim;
            let flat = ad(p1, p0) <= 1
                && ad(q1, q0) <= 1
                && ad(p2, p0) <= 1
                && ad(q2, q0) <= 1
                && ad(p3, p0) <= 1
                && ad(q3, q0) <= 1;
            if len == 14 {
                let (p6, p5, p4) = (at(s, -7), at(s, -6), at(s, -5));
                let (q4, q5, q6) = (at(s, 4), at(s, 5), at(s, 6));
                let flat2 = ad(p5, p0) <= 1
                    && ad(q5, q0) <= 1
                    && ad(p4, p0) <= 1
                    && ad(q4, q0) <= 1
                    && ad(p6, p0) <= 1
                    && ad(q6, q0) <= 1;
                if flat2 && flat && mask {
                    let o = [
                        r4(p6 * 7 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0),
                        r4(p6 * 5 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1),
                        r4(p6 * 4 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2),
                        r4(p6 * 3 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3),
                        r4(p6 * 2 + p5 + p4 + p3 + p2 * 2 + p1 * 2 + p0 * 2 + q0 + q1 + q2 + q3 + q4),
                        r4(p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + q2 + q3 + q4 + q5),
                        r4(p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + q3 + q4 + q5 + q6),
                        r4(p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 * 2 + q2 * 2 + q3 + q4 + q5 + q6 * 2),
                        r4(p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 * 2 + q3 * 2 + q4 + q5 + q6 * 3),
                        r4(p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 * 2 + q4 * 2 + q5 + q6 * 4),
                        r4(p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 * 2 + q5 * 2 + q6 * 5),
                        r4(p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 * 2 + q6 * 7),
                    ];
                    for (k, i) in (-6..6).enumerate() {
                        s[(base + i * step) as usize] = o[k] as u8;
                    }
                    return;
                }
            }
            if flat && mask {
                let o = [
                    r3(p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0),
                    r3(p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1),
                    r3(p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2),
                    r3(p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3),
                    r3(p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3),
                    r3(p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3),
                ];
                for (k, i) in (-3..3).enumerate() {
                    s[(base + i * step) as usize] = o[k] as u8;
                }
            } else {
                let mut v = [p3, p2, p1, p0, q0, q1, q2, q3];
                filter4(mask, t.hev, &mut v);
                for (k, i) in (-2..2).enumerate() {
                    s[(base + i * step) as usize] = v[k + 2] as u8;
                }
            }
        }
    }
}

/// Transform size of a plane at a mi position.
fn plane_tx(e: &Encoder, plane: usize, mi_row: usize, mi_col: usize) -> TxSize {
    if e.lossless {
        return TxSize::T4x4;
    }
    let m = e.mi(mi_row, mi_col);
    if plane == 0 { m.tx_size } else { uv_tx_size(m.bsize, e.ssx, e.ssy) }
}

/// Filter one plane in place with the given vertical / horizontal levels.
pub fn filter_plane(e: &Encoder, plane: usize, levels: [i32; 2], sharpness: i32, p: &mut Plane) {
    if levels == [0, 0] {
        return;
    }
    let (sx, sy) = if plane == 0 { (0, 0) } else { (e.ssx, e.ssy) };
    let width = p.width;
    let height = p.height;
    let stride = p.stride;
    let th = [thresholds(levels[0], sharpness), thresholds(levels[1], sharpness)];
    for dir in 0..2 {
        let lvl = levels[dir];
        if lvl == 0 {
            continue;
        }
        let t = th[dir];
        let mut y = 0;
        while y < height {
            let mut x = 0;
            while x < width {
                let coord = if dir == 0 { x } else { y };
                if coord > 0 {
                    let mi_row = sy | ((y << sy) >> 2);
                    let mi_col = sx | ((x << sx) >> 2);
                    let ts = plane_tx(e, plane, mi_row, mi_col);
                    let tsize = if dir == 0 { ts.width() } else { ts.height() };
                    if coord & (tsize - 1) == 0 {
                        let (pr, pc) =
                            if dir == 0 { (mi_row, mi_col - (1 << sx)) } else { (mi_row - (1 << sy), mi_col) };
                        let pts = plane_tx(e, plane, pr, pc);
                        let dim = if dir == 0 {
                            ts.w4().trailing_zeros().min(pts.w4().trailing_zeros())
                        } else {
                            ts.h4().trailing_zeros().min(pts.h4().trailing_zeros())
                        };
                        let len =
                            if plane > 0 { if dim == 0 { 4 } else { 6 } } else { [4, 8, 14, 14, 14][dim as usize] };
                        for i in 0..4 {
                            if dir == 0 {
                                filter_line(len, t, &mut p.data, ((y + i) * stride + x) as isize, 1);
                            } else {
                                filter_line(len, t, &mut p.data, (y * stride + x + i) as isize, stride as isize);
                            }
                        }
                    }
                }
                x += 4;
            }
            y += 4;
        }
    }
}

fn sse(a: &Plane, b: &Plane) -> u64 {
    let mut s = 0u64;
    for y in 0..a.height {
        let ra = &a.data[y * a.stride..y * a.stride + a.width];
        let rb = &b.data[y * b.stride..y * b.stride + a.width];
        s += ra.iter().zip(rb).map(|(&x, &y)| ((x as i32 - y as i32) * (x as i32 - y as i32)) as u64).sum::<u64>();
    }
    s
}

/// libaom's search_filter_level for one plane (dir 2 = both luma directions).
fn search_level(e: &Encoder, plane: usize, sharpness: i32, unfiltered: &Plane) -> i32 {
    let mut cache = [-1i64; 64];
    let mut work = unfiltered.clone();
    let mut try_level = |lvl: i32| -> i64 {
        if cache[lvl as usize] >= 0 {
            return cache[lvl as usize];
        }
        work.data.copy_from_slice(&unfiltered.data);
        filter_plane(e, plane, [lvl, lvl], sharpness, &mut work);
        let err = sse(&e.src[plane], &work) as i64;
        cache[lvl as usize] = err;
        err
    };
    let mut filt_mid = 0;
    let mut filter_step = if filt_mid < 16 { 4 } else { filt_mid / 4 };
    let mut best_err = try_level(filt_mid);
    let mut filt_best = filt_mid;
    let mut filt_direction = 0;
    while filter_step > 0 {
        let filt_high = (filt_mid + filter_step).min(MAX_LOOP_FILTER);
        let filt_low = (filt_mid - filter_step).max(0);
        let mut bias = (best_err >> (15 - (filt_mid / 8))) * filter_step as i64;
        bias >>= 1; // tx_mode != ONLY_4X4
        if filt_direction <= 0 && filt_low != filt_mid {
            let e_low = try_level(filt_low);
            if e_low < best_err + bias {
                if e_low < best_err {
                    best_err = e_low;
                }
                filt_best = filt_low;
            }
        }
        if filt_direction >= 0 && filt_high != filt_mid {
            let e_high = try_level(filt_high);
            if e_high < best_err - bias {
                best_err = e_high;
                filt_best = filt_high;
            }
        }
        if filt_best == filt_mid {
            filter_step /= 2;
            filt_direction = 0;
        } else {
            filt_direction = if filt_best < filt_mid { -1 } else { 1 };
            filt_mid = filt_best;
        }
    }
    filt_best
}

/// Pick the filter levels, filter the reconstruction and return the frame
/// header parameters.
pub fn pick_and_apply(e: &mut Encoder) -> LoopFilterParams {
    let mut sharpness = e.sharpness;
    if e.cfg.tune_iq {
        // Adaptive sharpness (tune IQ).
        let q = e.cfg.base_q;
        let max = if q <= 112 {
            7
        } else if q <= 160 {
            1
        } else {
            0
        };
        sharpness = sharpness.min(max);
    }
    let mut level = [0i32; 4];
    let y = search_level(e, 0, sharpness, &e.rec[0]);
    level[0] = y;
    level[1] = y;
    if e.num_planes > 1 {
        level[2] = search_level(e, 1, sharpness, &e.rec[1]);
        level[3] = search_level(e, 2, sharpness, &e.rec[2]);
    }
    if level[0] == 0 && level[1] == 0 {
        // Chroma levels are not coded when luma is off.
        level[2] = 0;
        level[3] = 0;
    }
    let mut rec = std::mem::take(&mut e.rec);
    filter_plane(e, 0, [level[0], level[1]], sharpness, &mut rec[0]);
    if e.num_planes > 1 {
        filter_plane(e, 1, [level[2], level[2]], sharpness, &mut rec[1]);
        filter_plane(e, 2, [level[3], level[3]], sharpness, &mut rec[2]);
    }
    e.rec = rec;
    LoopFilterParams {
        level: [level[0] as u8, level[1] as u8, level[2] as u8, level[3] as u8],
        sharpness: sharpness as u8,
    }
}
