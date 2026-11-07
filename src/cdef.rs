//! CDEF (libaom cdef.c / cdef_block.c) and its strength search (pickcdef.c,
//! CDEF_FAST_SEARCH_LVL3 with tune IQ's adaptive strength reduction).

use crate::cdf::COST_SHIFT;
use crate::coef::rdcost;
use crate::enc::{Decisions, Encoder};
use crate::frame::Plane;
use crate::obu::CdefParams;

const VERY_LARGE: i32 = 30000;
const SEC_STRENGTHS: usize = 4;

/// (dy, dx) of the two taps along each direction.
const DIRS: [[(isize, isize); 2]; 8] = [
    [(-1, 1), (-2, 2)],
    [(0, 1), (-1, 2)],
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2)],
    [(1, 1), (2, 2)],
    [(1, 0), (2, 1)],
    [(1, 0), (2, 0)],
    [(1, 0), (2, -1)],
];

/// Pre-CDEF samples of one plane with the frame boundary (mi-aligned) treated
/// as unavailable.
struct Src<'a> {
    p: &'a Plane,
    w: isize,
    h: isize,
}

const PB: usize = 3;
const PSTRIDE: usize = 64 + 2 * PB;

/// Padded copy of a 64x64 (or subsampled) filter block region with
/// unavailable samples set to VERY_LARGE.
struct Pad {
    buf: Vec<i16>,
    x0: isize,
    y0: isize,
}

impl Pad {
    fn new() -> Pad {
        Pad { buf: vec![0; PSTRIDE * PSTRIDE], x0: 0, y0: 0 }
    }
    fn load(&mut self, src: &Src, x0: usize, y0: usize, w: usize, h: usize) {
        self.x0 = x0 as isize - PB as isize;
        self.y0 = y0 as isize - PB as isize;
        for r in 0..h + 2 * PB {
            let y = self.y0 + r as isize;
            let row = &mut self.buf[r * PSTRIDE..r * PSTRIDE + w + 2 * PB];
            if y < 0 || y >= src.h {
                row.fill(VERY_LARGE as i16);
                continue;
            }
            for (c, v) in row.iter_mut().enumerate() {
                let x = self.x0 + c as isize;
                *v = if x < 0 || x >= src.w {
                    VERY_LARGE as i16
                } else {
                    src.p.data[y as usize * src.p.stride + x as usize] as i16
                };
            }
        }
    }
}

fn find_dir(p: &Plane, x0: usize, y0: usize) -> (usize, i32) {
    let mut cost = [0i32; 8];
    let mut partial = [[0i32; 15]; 8];
    const DIV: [i32; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];
    for i in 0..8 {
        for j in 0..8 {
            let x = p.data[(y0 + i) * p.stride + x0 + j] as i32 - 128;
            partial[0][i + j] += x;
            partial[1][i + j / 2] += x;
            partial[2][i] += x;
            partial[3][3 + i - j / 2] += x;
            partial[4][7 + i - j] += x;
            partial[5][3 - i / 2 + j] += x;
            partial[6][j] += x;
            partial[7][i / 2 + j] += x;
        }
    }
    for i in 0..8 {
        cost[2] += partial[2][i] * partial[2][i];
        cost[6] += partial[6][i] * partial[6][i];
    }
    cost[2] *= DIV[8];
    cost[6] *= DIV[8];
    for i in 0..7 {
        cost[0] += (partial[0][i] * partial[0][i] + partial[0][14 - i] * partial[0][14 - i]) * DIV[i + 1];
        cost[4] += (partial[4][i] * partial[4][i] + partial[4][14 - i] * partial[4][14 - i]) * DIV[i + 1];
    }
    cost[0] += partial[0][7] * partial[0][7] * DIV[8];
    cost[4] += partial[4][7] * partial[4][7] * DIV[8];
    for i in (1..8).step_by(2) {
        for j in 0..5 {
            cost[i] += partial[i][3 + j] * partial[i][3 + j];
        }
        cost[i] *= DIV[8];
        for j in 0..3 {
            cost[i] += (partial[i][j] * partial[i][j] + partial[i][10 - j] * partial[i][10 - j]) * DIV[2 * j + 2];
        }
    }
    let mut best_cost = 0;
    let mut best_dir = 0;
    for (i, &c) in cost.iter().enumerate() {
        if c > best_cost {
            best_cost = c;
            best_dir = i;
        }
    }
    let var = (best_cost - cost[(best_dir + 4) & 7]) >> 10;
    (best_dir, var)
}

fn adjust_strength(strength: i32, var: i32) -> i32 {
    let i = if var >> 6 != 0 { (31 - ((var >> 6) as u32).leading_zeros() as i32).min(12) } else { 0 };
    if var != 0 { (strength * (4 + i) + 8) >> 4 } else { 0 }
}

/// Filter one 8x8 (or chroma 4x4 / 4x8 / 8x4) block into `out` (stride `os`).
#[allow(clippy::too_many_arguments)]
fn filter_block(
    src: &Pad,
    out: &mut [u8],
    os: usize,
    x0: usize,
    y0: usize,
    bw: usize,
    bh: usize,
    pri: i32,
    sec: i32,
    dir: usize,
    damping: i32,
) {
    #[cfg(target_arch = "x86_64")]
    if crate::has_avx2() {
        // SAFETY: AVX2 support was checked at runtime.
        unsafe { filter_block_avx2(src, out, os, x0, y0, bw, bh, pri, sec, dir, damping) };
        return;
    }
    if bw == 8 {
        filter_rows::<8>(src, out, os, x0, y0, bh, pri, sec, dir, damping)
    } else {
        filter_rows::<4>(src, out, os, x0, y0, bh, pri, sec, dir, damping)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn filter_block_avx2(
    src: &Pad,
    out: &mut [u8],
    os: usize,
    x0: usize,
    y0: usize,
    bw: usize,
    bh: usize,
    pri: i32,
    sec: i32,
    dir: usize,
    damping: i32,
) {
    if bw == 8 {
        filter_rows::<8>(src, out, os, x0, y0, bh, pri, sec, dir, damping)
    } else {
        filter_rows::<4>(src, out, os, x0, y0, bh, pri, sec, dir, damping)
    }
}

#[inline(always)]
fn cons(diff: i32, t: i32, shift: i32) -> i32 {
    let a = diff.abs();
    let v = (t - (a >> shift)).clamp(0, a);
    if diff < 0 { -v } else { v }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn filter_rows<const W: usize>(
    src: &Pad,
    out: &mut [u8],
    os: usize,
    x0: usize,
    y0: usize,
    bh: usize,
    pri: i32,
    sec: i32,
    dir: usize,
    damping: i32,
) {
    let clip = pri != 0 && sec != 0;
    let pri_taps: [i32; 2] = if pri & 1 == 0 { [4, 2] } else { [3, 3] };
    let sec_taps = [2, 1];
    let msb = |t: i32| if t == 0 { 0 } else { 31 - (t as u32).leading_zeros() as i32 };
    let pshift = (damping - msb(pri)).max(0);
    let sshift = (damping - msb(sec)).max(0);
    let off = |d: (isize, isize)| d.0 * PSTRIDE as isize + d.1;
    // (offset, tap weight, strength, shift) for every tap pair
    let mut taps: [(isize, i32, i32, i32); 12] = [(0, 0, 0, 0); 12];
    let mut n = 0;
    for k in 0..2 {
        if pri != 0 {
            let o = off(DIRS[dir][k]);
            taps[n] = (o, pri_taps[k], pri, pshift);
            taps[n + 1] = (-o, pri_taps[k], pri, pshift);
            n += 2;
        }
        if sec != 0 {
            for d in [DIRS[(dir + 2) & 7][k], DIRS[(dir + 6) & 7][k]] {
                let o = off(d);
                taps[n] = (o, sec_taps[k], sec, sshift);
                taps[n + 1] = (-o, sec_taps[k], sec, sshift);
                n += 2;
            }
        }
    }
    let buf = &src.buf;
    for i in 0..bh {
        let pos = ((y0 as isize - src.y0) as usize + i) * PSTRIDE + (x0 as isize - src.x0) as usize;
        let x: [i32; W] = core::array::from_fn(|j| buf[pos + j] as i32);
        let mut sum = [0i32; W];
        let mut max = x;
        let mut min = x;
        for &(o, w, t, sh) in &taps[..n] {
            let p = (pos as isize + o) as usize;
            let v: [i32; W] = core::array::from_fn(|j| buf[p + j] as i32);
            for j in 0..W {
                sum[j] += w * cons(v[j] - x[j], t, sh);
                if clip {
                    max[j] = max[j].max(if v[j] == VERY_LARGE { x[j] } else { v[j] });
                    min[j] = min[j].min(v[j]);
                }
            }
        }
        let row = &mut out[i * os..i * os + W];
        for j in 0..W {
            let s = sum[j] as i16 as i32;
            let mut v = x[j] + ((8 + s - (s < 0) as i32) >> 4);
            if clip {
                v = v.clamp(min[j], max[j]);
            }
            row[j] = v as u8;
        }
    }
}

/// Search-time evaluation of all `TOTAL` (pri, sec) strength pairs for one
/// block, returning the SSE against `srcpix` for each. The result for every
/// pair is bit-identical to `filter_rows` followed by an SSE, but the
/// strength-independent parts (clipping bounds, secondary taps) are computed
/// once and shared.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn search_rows<const W: usize>(
    src: &Pad,
    srcpix: &[u8],
    sstride: usize,
    x0: usize,
    y0: usize,
    bh: usize,
    ts: &[i32; 5],
    dir: usize,
    damping: i32,
    err: &mut [u64; TOTAL],
) {
    let msb = |t: i32| 31 - (t as u32).leading_zeros() as i32;
    let sshift = (damping - msb(2)).max(0);
    let off = |d: (isize, isize)| d.0 * PSTRIDE as isize + d.1;
    let poff = [off(DIRS[dir][0]), off(DIRS[dir][1])];
    let soff = [
        [off(DIRS[(dir + 2) & 7][0]), off(DIRS[(dir + 6) & 7][0])],
        [off(DIRS[(dir + 2) & 7][1]), off(DIRS[(dir + 6) & 7][1])],
    ];
    let soff0 = [[off(DIRS[2][0]), off(DIRS[6][0])], [off(DIRS[2][1]), off(DIRS[6][1])]];
    let pshift: [i32; 5] = core::array::from_fn(|i| if ts[i] == 0 { 0 } else { (damping - msb(ts[i])).max(0) });
    let sec_taps = [2i32, 1];
    let buf = &src.buf;
    let mut acc = [0i64; TOTAL];
    let rnd = |s: i32| (8 + s - (s < 0) as i32) >> 4;
    for i in 0..bh {
        let pos = ((y0 as isize - src.y0) as usize + i) * PSTRIDE + (x0 as isize - src.x0) as usize;
        let x: [i32; W] = core::array::from_fn(|j| buf[pos + j] as i32);
        let sp: [i32; W] = core::array::from_fn(|j| srcpix[i * sstride + j] as i32);
        let load = |o: isize| -> [i32; W] {
            let p = (pos as isize + o) as usize;
            core::array::from_fn(|j| buf[p + j] as i32)
        };
        let mut lo = x;
        let mut hi = x;
        let upd = |v: &[i32; W], lo: &mut [i32; W], hi: &mut [i32; W]| {
            for j in 0..W {
                hi[j] = hi[j].max(if v[j] == VERY_LARGE { x[j] } else { v[j] });
                lo[j] = lo[j].min(v[j]);
            }
        };
        // secondary sums along the block direction and along direction 0
        let mut ssum = [0i32; W];
        let mut ssum0 = [0i32; W];
        for k in 0..2 {
            for sgn in [1isize, -1] {
                for d in 0..2 {
                    let v = load(sgn * soff[k][d]);
                    for j in 0..W {
                        ssum[j] += sec_taps[k] * cons(v[j] - x[j], 2, sshift);
                    }
                    upd(&v, &mut lo, &mut hi);
                    let v0 = load(sgn * soff0[k][d]);
                    for j in 0..W {
                        ssum0[j] += sec_taps[k] * cons(v0[j] - x[j], 2, sshift);
                    }
                }
            }
        }
        let mut pv = [[0i32; W]; 4];
        for k in 0..2 {
            for sgn in [1isize, -1] {
                let v = load(sgn * poff[k]);
                upd(&v, &mut lo, &mut hi);
                pv[k * 2 + (sgn < 0) as usize] = v;
            }
        }
        let sse = |o: &[i32; W]| -> i64 {
            let mut t = 0i32;
            for j in 0..W {
                let d = sp[j] - (o[j] as u8 as i32);
                t += d * d;
            }
            t as i64
        };
        // pri 0
        acc[0] += sse(&x);
        let o: [i32; W] = core::array::from_fn(|j| x[j] + rnd(ssum0[j] as i16 as i32));
        acc[1] += sse(&o);
        for pi in 1..5 {
            let t = ts[pi];
            if t == 0 {
                acc[pi * 2] += sse(&x);
                let o: [i32; W] = core::array::from_fn(|j| x[j] + rnd(ssum[j] as i16 as i32));
                acc[pi * 2 + 1] += sse(&o);
                continue;
            }
            let w = if t & 1 == 0 { [4i32, 2] } else { [3, 3] };
            let mut psum = [0i32; W];
            for q in 0..4 {
                for j in 0..W {
                    psum[j] += w[q >> 1] * cons(pv[q][j] - x[j], t, pshift[pi]);
                }
            }
            let o: [i32; W] = core::array::from_fn(|j| x[j] + rnd(psum[j] as i16 as i32));
            acc[pi * 2] += sse(&o);
            let o: [i32; W] = core::array::from_fn(|j| {
                let s = (psum[j] + ssum[j]) as i16 as i32;
                (x[j] + rnd(s)).clamp(lo[j], hi[j])
            });
            acc[pi * 2 + 1] += sse(&o);
        }
    }
    for g in 0..TOTAL {
        err[g] += acc[g] as u64;
    }
}

#[allow(clippy::too_many_arguments)]
fn search_block(
    src: &Pad,
    srcpix: &[u8],
    sstride: usize,
    x0: usize,
    y0: usize,
    bw: usize,
    bh: usize,
    ts: &[i32; 5],
    dir: usize,
    damping: i32,
    err: &mut [u64; TOTAL],
) {
    #[cfg(target_arch = "x86_64")]
    if crate::has_avx2() {
        // SAFETY: AVX2 support was checked at runtime.
        unsafe { search_block_avx2(src, srcpix, sstride, x0, y0, bw, bh, ts, dir, damping, err) };
        return;
    }
    if bw == 8 {
        search_rows::<8>(src, srcpix, sstride, x0, y0, bh, ts, dir, damping, err)
    } else {
        search_rows::<4>(src, srcpix, sstride, x0, y0, bh, ts, dir, damping, err)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn search_block_avx2(
    src: &Pad,
    srcpix: &[u8],
    sstride: usize,
    x0: usize,
    y0: usize,
    bw: usize,
    bh: usize,
    ts: &[i32; 5],
    dir: usize,
    damping: i32,
    err: &mut [u64; TOTAL],
) {
    if bw == 8 {
        search_rows::<8>(src, srcpix, sstride, x0, y0, bh, ts, dir, damping, err)
    } else {
        search_rows::<4>(src, srcpix, sstride, x0, y0, bh, ts, dir, damping, err)
    }
}

/// Non-skip 8x8 blocks (in luma 8x8 units) of a 64x64 filter block.
fn block_list(e: &Encoder, fbr: usize, fbc: usize, list: &mut Vec<(usize, usize)>) {
    list.clear();
    let maxr = (e.mi_rows - fbr * 16).min(16);
    let maxc = (e.mi_cols - fbc * 16).min(16);
    for r in (0..maxr).step_by(2) {
        for c in (0..maxc).step_by(2) {
            let (mr, mc) = (fbr * 16 + r, fbc * 16 + c);
            let skip = e.mi(mr, mc).skip && e.mi(mr, mc + 1).skip && e.mi(mr + 1, mc).skip && e.mi(mr + 1, mc + 1).skip;
            if !skip {
                list.push((r >> 1, c >> 1));
            }
        }
    }
}

/// Filter the blocks of one 64x64 unit of one plane with (pri, sec) into `out`
/// (a buffer of the 64x64 region, stride 64). Returns nothing; `dirs` holds
/// (dir, var) per listed block.
#[allow(clippy::too_many_arguments)]
fn filter_fb(
    e: &Encoder,
    plane: usize,
    src: &Pad,
    fbr: usize,
    fbc: usize,
    list: &[(usize, usize)],
    dirs: &[(usize, i32)],
    pri: i32,
    sec: i32,
    damping: i32,
    out: &mut [u8],
) {
    let (sx, sy) = if plane == 0 { (0, 0) } else { (e.ssx, e.ssy) };
    let bw = 8 >> sx;
    let bh = 8 >> sy;
    let ox = (fbc * 64) >> sx;
    let oy = (fbr * 64) >> sy;
    let damping = damping - (plane > 0) as i32;
    for (k, &(by, bx)) in list.iter().enumerate() {
        let (dir, var) = dirs[k];
        let t = if plane == 0 { adjust_strength(pri, var) } else { pri };
        let x0 = ox + bx * bw;
        let y0 = oy + by * bh;
        let o = (by * bh) * 64 + bx * bw;
        if t == 0 && sec == 0 {
            for i in 0..bh {
                for j in 0..bw {
                    let pos = ((y0 as isize - src.y0) as usize + i) * PSTRIDE + (x0 as isize - src.x0) as usize + j;
                    out[o + i * 64 + j] = src.buf[pos] as u8;
                }
            }
            continue;
        }
        filter_block(src, &mut out[o..], 64, x0, y0, bw, bh, t, sec, if pri != 0 { dir } else { 0 }, damping);
    }
}

const PRICONV: [i32; 5] = [0, 2, 4, 8, 14];
const SECCONV: [i32; 2] = [0, 2];
const TOTAL: usize = 10;

fn sec_value(code: i32) -> i32 {
    code + (code == 3) as i32
}

pub fn pick_and_apply(e: &mut Encoder, _sbs: &[Decisions]) -> Option<CdefParams> {
    let q = e.cfg.base_q as i32;
    let adaptive = e.cfg.tune_iq;
    if adaptive && q <= 32 {
        e.cdef_bits = 0;
        e.cdef_idx.clear();
        return None;
    }
    let nvfb = e.mi_rows.div_ceil(16);
    let nhfb = e.mi_cols.div_ceil(16);
    let damping = 3 + (q >> 6);
    let np = e.num_planes;
    let pre: Vec<Plane> = e.rec.clone();
    let srcs: Vec<Src> = (0..np)
        .map(|p| {
            let (sx, sy) = if p == 0 { (0, 0) } else { (e.ssx, e.ssy) };
            Src { p: &pre[p], w: ((e.mi_cols * 4) >> sx) as isize, h: ((e.mi_rows * 4) >> sy) as isize }
        })
        .collect();
    // Per-filter-block statistics are independent, so rows of filter blocks
    // are searched in parallel; results are gathered in raster order.
    let thr = e.cfg.threads;
    let ee: &Encoder = e;
    let rows: Vec<Vec<(usize, usize, [[u64; TOTAL]; 2], Vec<(usize, i32)>)>> =
        crate::parallel::par_map(nvfb, thr, |fbr| {
            let mut list = Vec::new();
            let mut pad = Pad::new();
            let mut res = Vec::new();
            for fbc in 0..nhfb {
                block_list(ee, fbr, fbc, &mut list);
                if list.is_empty() {
                    continue;
                }
                let dirs: Vec<(usize, i32)> =
                    list.iter().map(|&(by, bx)| find_dir(&pre[0], fbc * 64 + bx * 8, fbr * 64 + by * 8)).collect();
                let mut m = [[0u64; TOTAL]; 2];
                for p in 0..np {
                    let (sx, sy) = if p == 0 { (0, 0) } else { (ee.ssx, ee.ssy) };
                    let bw = 8 >> sx;
                    let bh = 8 >> sy;
                    pad.load(&srcs[p], (fbc * 64) >> sx, (fbr * 64) >> sy, 64 >> sx, 64 >> sy);
                    let damp = damping - (p > 0) as i32;
                    let s = &ee.src[p];
                    let ox = (fbc * 64) >> sx;
                    let oy = (fbr * 64) >> sy;
                    let mut err = [0u64; TOTAL];
                    for (k, &(by, bx)) in list.iter().enumerate() {
                        let (dir, var) = dirs[k];
                        let ts: [i32; 5] =
                            core::array::from_fn(
                                |i| if p == 0 { adjust_strength(PRICONV[i], var) } else { PRICONV[i] },
                            );
                        let (x0, y0) = (ox + bx * bw, oy + by * bh);
                        search_block(
                            &pad,
                            &s.data[y0 * s.stride + x0..],
                            s.stride,
                            x0,
                            y0,
                            bw,
                            bh,
                            &ts,
                            dir,
                            damp,
                            &mut err,
                        );
                    }
                    for gi in 0..TOTAL {
                        m[(p > 0) as usize][gi] += err[gi];
                    }
                }
                res.push((fbr, fbc, m, dirs));
            }
            res
        });
    let mut list = Vec::new();
    let mut out = vec![0u8; 64 * 64];
    let mut pad = Pad::new();
    // mse[plane type][sb][strength]
    let mut mse: [Vec<[u64; TOTAL]>; 2] = [Vec::new(), Vec::new()];
    let mut sb_index = Vec::new();
    let mut all_dirs = Vec::new();
    for (fbr, fbc, m, dirs) in rows.into_iter().flatten() {
        mse[0].push(m[0]);
        mse[1].push(m[1]);
        sb_index.push((fbr, fbc));
        all_dirs.push(dirs);
    }
    let sb_count = sb_index.len();
    // Number of signalling bits and the strength sets.
    let rdmult = e.rd_mult(e.cfg.base_q);
    let total = TOTAL;
    let joint = if np > 1 { total * total } else { total };
    let max_bits = if joint == 1 { 0 } else { (usize::BITS - (joint - 1).leading_zeros()) as usize };
    let reduce = adaptive && q <= 220;
    let zero_low = reduce && q <= 140;
    let min_bits = if zero_low && max_bits > 0 { 1 } else { 0 };
    let mut best_rd = u64::MAX;
    let mut best_bits = 0;
    let mut best0 = [0usize; 8];
    let mut best1 = [0usize; 8];
    for i in min_bits..=3 {
        if i > max_bits {
            break;
        }
        let n = 1 << i;
        let mut l0 = [0usize; 8];
        let mut l1 = [0usize; 8];
        let tot = if np > 1 {
            joint_search_dual(&mut l0, &mut l1, n, &mse, sb_count)
        } else {
            let mut t = 0;
            for k in 0..n {
                t = search_one(&mut l0, k, &mse[0], sb_count);
            }
            t
        };
        let bits = sb_count * i + n * 6 * if np > 1 { 2 } else { 1 };
        let rd = rdcost(rdmult, (bits << COST_SHIFT) as i64, (tot * 16) as i64) as u64;
        if rd < best_rd {
            best_rd = rd;
            best_bits = i;
            best0 = l0;
            best1 = l1;
        }
    }
    let n = 1 << best_bits;
    // Per-SB choice.
    let mut idx = vec![0i8; nvfb * nhfb];
    let mut choice = vec![0usize; sb_count];
    for s in 0..sb_count {
        let mut best = u64::MAX;
        for gi in 0..n {
            let mut c = mse[0][s][best0[gi]];
            if np > 1 {
                c += mse[1][s][best1[gi]];
            }
            if c < best {
                best = c;
                choice[s] = gi;
            }
        }
        let (r, c) = sb_index[s];
        idx[r * nhfb + c] = choice[s] as i8;
    }
    // Convert search indices to coded strengths (pri * 4 + sec code), then
    // the adaptive reduction / zeroing.
    let mut strengths = Vec::with_capacity(n);
    for gi in 0..n {
        let to_code = |s: usize| (PRICONV[s / 2] as usize) * SEC_STRENGTHS + SECCONV[s % 2] as usize;
        let mut ys = to_code(best0[gi]);
        let mut uvs = if np > 1 { to_code(best1[gi]) } else { 0 };
        if reduce {
            let (yp, ysec) = ((ys / SEC_STRENGTHS) >> 1, (ys % SEC_STRENGTHS) >> 1);
            ys = yp * SEC_STRENGTHS + ysec;
            let (up, usec) = ((uvs / SEC_STRENGTHS) >> 1, (uvs % SEC_STRENGTHS) >> 1);
            uvs = up * SEC_STRENGTHS + usec;
            if zero_low {
                let low_y = yp <= 4 && ysec <= 1;
                if low_y {
                    ys = 0;
                }
                if np > 1 && (low_y || (up <= 4 && usec <= 1)) {
                    uvs = 0;
                }
            }
        }
        strengths.push(((ys / 4) as u8, (ys % 4) as u8, (uvs / 4) as u8, (uvs % 4) as u8));
    }
    // Apply.
    for (s, &(fbr, fbc)) in sb_index.iter().enumerate() {
        block_list(e, fbr, fbc, &mut list);
        let st = strengths[choice[s]];
        let dirs = &all_dirs[s];
        for p in 0..np {
            let (pri, sec) =
                if p == 0 { (st.0 as i32, sec_value(st.1 as i32)) } else { (st.2 as i32, sec_value(st.3 as i32)) };
            if pri == 0 && sec == 0 {
                continue;
            }
            let (sx, sy) = if p == 0 { (0, 0) } else { (e.ssx, e.ssy) };
            pad.load(&srcs[p], (fbc * 64) >> sx, (fbr * 64) >> sy, 64 >> sx, 64 >> sy);
            filter_fb(e, p, &pad, fbr, fbc, &list, dirs, pri, sec, damping, &mut out);
            let bw = 8 >> sx;
            let bh = 8 >> sy;
            let ox = (fbc * 64) >> sx;
            let oy = (fbr * 64) >> sy;
            let rec = &mut e.rec[p];
            for &(by, bx) in &list {
                for i in 0..bh {
                    let y = oy + by * bh + i;
                    let x = ox + bx * bw;
                    let o = (by * bh + i) * 64 + bx * bw;
                    rec.data[y * rec.stride + x..y * rec.stride + x + bw].copy_from_slice(&out[o..o + bw]);
                }
            }
        }
    }
    e.cdef_bits = best_bits as u8;
    e.cdef_idx = idx;
    Some(CdefParams { damping: damping as u8, bits: best_bits as u8, strengths })
}

fn search_one(lev: &mut [usize; 8], nb: usize, mse: &[[u64; TOTAL]], sb_count: usize) -> u64 {
    let mut tot = [0u64; TOTAL];
    for i in 0..sb_count {
        let mut best = u64::MAX >> 1;
        for &l in &lev[..nb] {
            best = best.min(mse[i][l]);
        }
        for j in 0..TOTAL {
            tot[j] += best.min(mse[i][j]);
        }
    }
    let (mut best_id, mut best_tot) = (0, u64::MAX);
    for (j, &t) in tot.iter().enumerate() {
        if t < best_tot {
            best_tot = t;
            best_id = j;
        }
    }
    lev[nb] = best_id;
    best_tot
}

fn search_one_dual(
    l0: &mut [usize; 8],
    l1: &mut [usize; 8],
    nb: usize,
    mse: &[Vec<[u64; TOTAL]>; 2],
    sb_count: usize,
) -> u64 {
    let mut tot = [[0u64; TOTAL]; TOTAL];
    for i in 0..sb_count {
        let mut best = u64::MAX >> 1;
        for g in 0..nb {
            best = best.min(mse[0][i][l0[g]] + mse[1][i][l1[g]]);
        }
        for j in 0..TOTAL {
            for k in 0..TOTAL {
                tot[j][k] += best.min(mse[0][i][j] + mse[1][i][k]);
            }
        }
    }
    let (mut b0, mut b1, mut best_tot) = (0, 0, u64::MAX);
    for j in 0..TOTAL {
        for k in 0..TOTAL {
            if tot[j][k] < best_tot {
                best_tot = tot[j][k];
                b0 = j;
                b1 = k;
            }
        }
    }
    l0[nb] = b0;
    l1[nb] = b1;
    best_tot
}

fn joint_search_dual(
    l0: &mut [usize; 8],
    l1: &mut [usize; 8],
    nb: usize,
    mse: &[Vec<[u64; TOTAL]>; 2],
    sb_count: usize,
) -> u64 {
    let mut t = 0;
    for i in 0..nb {
        t = search_one_dual(l0, l1, i, mse, sb_count);
    }
    for _ in 0..4 * nb {
        for j in 0..nb - 1 {
            l0[j] = l0[j + 1];
            l1[j] = l1[j + 1];
        }
        t = search_one_dual(l0, l1, nb - 1, mse, sb_count);
    }
    t
}
