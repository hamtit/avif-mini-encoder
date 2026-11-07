//! Coefficient coding: contexts, bitstream writing, rate estimation and the
//! trellis optimization (ports of libaom's encodetxb.c / txb_rdopt.c).

use crate::cdf::{CoefCdfs, costs_from_cdf};
use crate::consts::*;
use crate::ec::Writer;
use crate::quant::{PlaneQuant, dequant_coeff, tx_scale};
use crate::tables::nz_map_ctx_offset;

pub const COEFF_BASE_RANGE: usize = 12;
pub const NUM_BASE_LEVELS: i32 = 2;
const TX_PAD_HOR: usize = 4;
const LEVELS_REGION: usize = (32 + TX_PAD_HOR) * (32 + 4) + 16 + 64;
/// Levels buffer: the padded levels followed by their coefficient-base context map.
pub const LEVELS_BUF: usize = 2 * LEVELS_REGION;

const EOB_GROUP_START: [u16; 12] = [0, 1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513];
const EOB_OFFSET_BITS: [u32; 12] = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
const EOB_TO_POS_SMALL: [u8; 33] =
    [0, 1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6];
const EOB_TO_POS_LARGE: [u8; 17] = [6, 7, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11];

/// (eob token, extra bits value)
#[inline]
pub fn eob_pos_token(eob: usize) -> (usize, u32) {
    let t = if eob < 33 { EOB_TO_POS_SMALL[eob] } else { EOB_TO_POS_LARGE[((eob - 1) >> 5).min(16)] } as usize;
    (t, (eob - EOB_GROUP_START[t] as usize) as u32)
}

const NZ_MAP_CTX_OFFSET_1D: [usize; 32] = {
    let mut a = [26 + 10; 32];
    a[0] = 26;
    a[1] = 26 + 5;
    a
};

/// Per transform block coding geometry.
#[derive(Clone, Copy)]
pub struct TxbInfo {
    pub tx: TxSize,
    /// Coded (≤32) height log2 (libaom's bhl) and width.
    pub bhl: usize,
    pub width: usize,
    pub height: usize,
    pub tx_class: u8,
    pub txs_ctx: usize,
}

impl TxbInfo {
    pub fn new(tx: TxSize, tx_type: TxType) -> Self {
        let c = tx.coded();
        TxbInfo {
            tx,
            bhl: c.h_log2(),
            width: c.width(),
            height: c.height(),
            tx_class: TX_TYPE_CLASS[tx_type as usize],
            txs_ctx: tx.ctx(),
        }
    }
    #[inline]
    fn stride(&self) -> usize {
        self.height + TX_PAD_HOR
    }
    #[inline]
    pub fn padded(&self, ci: usize) -> usize {
        ci + ((ci >> self.bhl) << 2)
    }
}

/// Fill the padded levels buffer (|q| clamped to 127). Every coefficient past
/// `eob` in scan order is zero, so only the first `eob` scan positions are set.
pub fn init_levels(q: &[i32], scan: &[u16], eob: usize, info: &TxbInfo, levels: &mut [u8]) -> usize {
    let stride = info.stride();
    let n = stride * (info.width + 4) + 16;
    levels[..n].fill(0);
    let mut max_pos = 0;
    for &rc in &scan[..eob] {
        let rc = rc as usize;
        let pos = info.padded(rc);
        levels[pos] = q[rc].unsigned_abs().min(127) as u8;
        max_pos = max_pos.max(pos);
    }
    max_pos
}

#[inline]
fn nz_mag(levels: &[u8], pos: usize, info: &TxbInfo) -> usize {
    let s = info.stride();
    let l = |i: usize| (levels[pos + i] as usize).min(3);
    let mut mag = l(s) + l(1);
    match info.tx_class {
        TX_CLASS_2D => mag += l(s + 1) + l(2 * s) + l(2),
        TX_CLASS_VERT => mag += l(2) + l(3) + l(4),
        _ => mag += l(2 * s) + l(3 * s) + l(4 * s),
    }
    mag
}

/// Context for coeff_base at raster index `ci` (not the last coefficient).
#[inline]
pub fn lower_levels_ctx(levels: &[u8], ci: usize, info: &TxbInfo) -> usize {
    let stats = nz_mag(levels, info.padded(ci), info);
    if info.tx_class == TX_CLASS_2D && ci == 0 {
        return 0;
    }
    let ctx = ((stats + 1) >> 1).min(4);
    match info.tx_class {
        TX_CLASS_2D => ctx + nz_map_ctx_offset(info.tx)[ci] as usize,
        TX_CLASS_HORIZ => ctx + NZ_MAP_CTX_OFFSET_1D[ci >> info.bhl],
        _ => ctx + NZ_MAP_CTX_OFFSET_1D[ci & (info.height - 1)],
    }
}

/// Context for coeff_base_eob at scan index `si`.
#[inline]
pub fn eob_ctx(si: usize, info: &TxbInfo) -> usize {
    let n = info.width << info.bhl;
    if si == 0 {
        0
    } else if si <= n / 8 {
        1
    } else if si <= n / 4 {
        2
    } else {
        3
    }
}

#[inline]
pub fn br_ctx(levels: &[u8], ci: usize, info: &TxbInfo) -> usize {
    let col = ci >> info.bhl;
    let row = ci & (info.height - 1);
    let s = info.stride();
    let pos = col * s + row;
    let mut mag = levels[pos + 1] as usize + levels[pos + s] as usize;
    match info.tx_class {
        TX_CLASS_2D => {
            mag += levels[pos + s + 1] as usize;
            mag = ((mag + 1) >> 1).min(6);
            if ci == 0 {
                return mag;
            }
            if row < 2 && col < 2 {
                return mag + 7;
            }
        }
        TX_CLASS_HORIZ => {
            mag += levels[pos + 2 * s] as usize;
            mag = ((mag + 1) >> 1).min(6);
            if ci == 0 {
                return mag;
            }
            if col == 0 {
                return mag + 7;
            }
        }
        _ => {
            mag += levels[pos + 2] as usize;
            mag = ((mag + 1) >> 1).min(6);
            if ci == 0 {
                return mag;
            }
            if row == 0 {
                return mag + 7;
            }
        }
    }
    mag + 14
}

#[inline]
fn br_ctx_eob(ci: usize, info: &TxbInfo) -> usize {
    if ci == 0 {
        return 0;
    }
    let col = ci >> info.bhl;
    let row = ci & (info.height - 1);
    if (info.tx_class == TX_CLASS_2D && row < 2 && col < 2)
        || (info.tx_class == TX_CLASS_HORIZ && col == 0)
        || (info.tx_class == TX_CLASS_VERT && row == 0)
    {
        7
    } else {
        14
    }
}

/// libaom's av1_get_txb_entropy_context (level sum with the dc sign category).
pub fn txb_entropy_ctx(q: &[i32], scan: &[u16], eob: usize) -> u8 {
    if eob == 0 {
        return 0;
    }
    let mut cul = 0u32;
    for &rc in &scan[..eob] {
        cul += q[rc as usize].unsigned_abs();
        if cul > 63 {
            break;
        }
    }
    let mut cul = cul.min(63) as u8;
    let dc = q[0];
    if dc < 0 {
        cul |= 1 << 6;
    } else if dc > 0 {
        cul += 2 << 6;
    }
    cul
}

const SKIP_CONTEXTS: [[u8; 5]; 5] =
    [[1, 2, 2, 2, 3], [2, 4, 4, 4, 5], [2, 4, 4, 4, 5], [2, 4, 4, 4, 5], [3, 5, 5, 5, 6]];

/// (txb_skip_ctx, dc_sign_ctx) from the above/left entropy contexts.
pub fn txb_ctx(plane: usize, plane_bsize: BlockSize, tx: TxSize, a: &[u8], l: &[u8]) -> (usize, usize) {
    let (w4, h4) = (tx.w4(), tx.h4());
    let mut dc_sign = 0i32;
    for &v in a[..w4].iter().chain(l[..h4].iter()) {
        dc_sign += [0, -1, 1][(v >> 6) as usize];
    }
    let dc_ctx = match dc_sign.signum() {
        0 => 0,
        -1 => 1,
        _ => 2,
    };
    let skip_ctx = if plane == 0 {
        if plane_bsize == tx.block() {
            0
        } else {
            let top = (a[..w4].iter().fold(0, |acc, &v| acc | v) & 63).min(4);
            let left = (l[..h4].iter().fold(0, |acc, &v| acc | v) & 63).min(4);
            SKIP_CONTEXTS[top as usize][left as usize] as usize
        }
    } else {
        let above = a[..w4].iter().any(|&v| v != 0) as usize;
        let left = l[..h4].iter().any(|&v| v != 0) as usize;
        let base = above + left;
        let off = if plane_bsize.pels_log2() > tx.block().pels_log2() { 10 } else { 7 };
        base + off
    };
    (skip_ctx, dc_ctx)
}

/// Rate tables derived from the coefficient CDFs (libaom LV_MAP_COEFF_COST).
#[derive(Clone)]
pub struct CoefCosts {
    pub txb_skip: [[[u32; 2]; 13]; 5],
    pub base_eob: [[[[u32; 3]; 4]; 2]; 5],
    pub base: [[[[i32; 8]; 42]; 2]; 5],
    pub eob_extra: [[[[u32; 2]; 9]; 2]; 5],
    pub dc_sign: [[[u32; 2]; 3]; 2],
    pub lps: [[[[i32; 2 * COEFF_BASE_RANGE + 2]; 21]; 2]; 5],
    /// [eob multi size][plane type][ctx][token - 1]
    pub eob: [[[[u32; 11]; 2]; 2]; 7],
}

impl CoefCosts {
    pub fn new(c: &CoefCdfs) -> Box<CoefCosts> {
        let mut k = Box::new(CoefCosts {
            txb_skip: [[[0; 2]; 13]; 5],
            base_eob: [[[[0; 3]; 4]; 2]; 5],
            base: [[[[0; 8]; 42]; 2]; 5],
            eob_extra: [[[[0; 2]; 9]; 2]; 5],
            dc_sign: [[[0; 2]; 3]; 2],
            lps: [[[[0; 2 * COEFF_BASE_RANGE + 2]; 21]; 2]; 5],
            eob: [[[[0; 11]; 2]; 2]; 7],
        });
        k.update(c);
        k
    }

    pub fn update(&mut self, c: &CoefCdfs) {
        let mut tmp = [0u32; 16];
        for pt in 0..2 {
            for ctx in 0..2 {
                costs_from_cdf(&c.eob16[pt][ctx], 5, &mut self.eob[0][pt][ctx]);
                costs_from_cdf(&c.eob32[pt][ctx], 6, &mut self.eob[1][pt][ctx]);
                costs_from_cdf(&c.eob64[pt][ctx], 7, &mut self.eob[2][pt][ctx]);
                costs_from_cdf(&c.eob128[pt][ctx], 8, &mut self.eob[3][pt][ctx]);
                costs_from_cdf(&c.eob256[pt][ctx], 9, &mut self.eob[4][pt][ctx]);
                costs_from_cdf(&c.eob512[pt][ctx], 10, &mut self.eob[5][pt][ctx]);
                costs_from_cdf(&c.eob1024[pt][ctx], 11, &mut self.eob[6][pt][ctx]);
            }
            for ctx in 0..3 {
                costs_from_cdf(&c.dc_sign[pt][ctx], 2, &mut self.dc_sign[pt][ctx]);
            }
        }
        for ts in 0..5 {
            for ctx in 0..13 {
                costs_from_cdf(&c.txb_skip[ts][ctx], 2, &mut self.txb_skip[ts][ctx]);
            }
            for pt in 0..2 {
                for ctx in 0..4 {
                    costs_from_cdf(&c.base_eob[ts][pt][ctx], 3, &mut self.base_eob[ts][pt][ctx]);
                }
                for ctx in 0..42 {
                    costs_from_cdf(&c.base[ts][pt][ctx], 4, &mut tmp);
                    let b = &mut self.base[ts][pt][ctx];
                    for i in 0..4 {
                        b[i] = tmp[i] as i32;
                    }
                    b[4] = 0;
                    b[5] = b[1] + 512 - b[0];
                    b[6] = b[2] - b[1];
                    b[7] = b[3] - b[2];
                }
                for ctx in 0..9 {
                    costs_from_cdf(&c.eob_extra[ts][pt][ctx], 2, &mut self.eob_extra[ts][pt][ctx]);
                }
                for ctx in 0..21 {
                    costs_from_cdf(&c.br[ts.min(3)][pt][ctx], 4, &mut tmp);
                    let lps = &mut self.lps[ts][pt][ctx];
                    let mut prev = 0i32;
                    let mut i = 0;
                    while i < COEFF_BASE_RANGE {
                        for j in 0..3 {
                            lps[i + j] = prev + tmp[j] as i32;
                        }
                        prev += tmp[3] as i32;
                        i += 3;
                    }
                    lps[i] = prev;
                    lps[COEFF_BASE_RANGE + 1] = lps[0];
                    for i in 1..=COEFF_BASE_RANGE {
                        lps[i + COEFF_BASE_RANGE + 1] = lps[i] - lps[i - 1];
                    }
                }
            }
        }
    }
}

#[inline]
fn golomb_cost(abs: i32) -> i32 {
    if abs >= 15 {
        let r = (abs - 14) as u32;
        let len = 32 - r.leading_zeros();
        ((2 * len - 1) << 9) as i32
    } else {
        0
    }
}

#[inline]
fn br_cost(level: i32, lps: &[i32]) -> i32 {
    let base_range = ((level - 3) as usize).min(COEFF_BASE_RANGE);
    lps[base_range] + golomb_cost(level)
}

/// Costs view for one transform block.
pub struct TxbCosts<'a> {
    pub k: &'a CoefCosts,
    pub ts: usize,
    pub pt: usize,
}

impl TxbCosts<'_> {
    #[inline]
    fn base(&self, ctx: usize) -> &[i32; 8] {
        &self.k.base[self.ts][self.pt][ctx]
    }
    #[inline]
    fn lps(&self, ctx: usize) -> &[i32] {
        &self.k.lps[self.ts][self.pt][ctx]
    }

    pub fn eob_cost(&self, eob: usize, info: &TxbInfo) -> i32 {
        let (pt_tok, extra) = eob_pos_token(eob);
        let multi = info.tx.coded().w_log2() + info.tx.coded().h_log2() - 4;
        let ctx = (info.tx_class != TX_CLASS_2D) as usize;
        let mut c = self.k.eob[multi][self.pt][ctx][pt_tok - 1] as i32;
        let bits = EOB_OFFSET_BITS[pt_tok];
        if bits > 0 {
            let bit = (extra >> (bits - 1)) & 1;
            c += self.k.eob_extra[self.ts][self.pt][pt_tok - 3][bit as usize] as i32;
            c += ((bits - 1) << 9) as i32;
        }
        c
    }

    #[inline]
    fn coeff_cost_eob(&self, ci: usize, abs: i32, sign: usize, ctx: usize, dc_ctx: usize, info: &TxbInfo) -> i32 {
        let mut c = self.k.base_eob[self.ts][self.pt][ctx][(abs.min(3) - 1) as usize] as i32;
        if ci == 0 {
            c += self.k.dc_sign[self.pt][dc_ctx][sign] as i32;
        } else {
            c += 512;
        }
        if abs > NUM_BASE_LEVELS {
            c += br_cost(abs, self.lps(br_ctx_eob(ci, info)));
        }
        c
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn coeff_cost_general(
        &self,
        is_last: bool,
        ci: usize,
        abs: i32,
        sign: usize,
        ctx: usize,
        dc_ctx: usize,
        info: &TxbInfo,
        levels: &[u8],
    ) -> i32 {
        let mut c = if is_last {
            self.k.base_eob[self.ts][self.pt][ctx][(abs.min(3) - 1) as usize] as i32
        } else {
            self.base(ctx)[abs.min(3) as usize]
        };
        if abs != 0 {
            if ci == 0 {
                c += self.k.dc_sign[self.pt][dc_ctx][sign] as i32;
            } else {
                c += 512;
            }
            if abs > NUM_BASE_LEVELS {
                let b = if is_last { br_ctx_eob(ci, info) } else { br_ctx(levels, ci, info) };
                c += br_cost(abs, self.lps(b));
            }
        }
        c
    }
}

/// Rate of a transform block's coefficients (excluding the tx type), like
/// libaom's av1_cost_coeffs_txb. `q` is column-major (coded size).
#[allow(clippy::too_many_arguments)]
pub fn cost_coeffs(
    k: &CoefCosts,
    q: &[i32],
    eob: usize,
    scan: &[u16],
    info: &TxbInfo,
    plane_type: usize,
    skip_ctx: usize,
    dc_ctx: usize,
    levels: &mut [u8],
) -> i32 {
    let ts = info.txs_ctx;
    if eob == 0 {
        return k.txb_skip[ts][skip_ctx][1] as i32;
    }
    let c = TxbCosts { k, ts, pt: plane_type };
    let mut cost = k.txb_skip[ts][skip_ctx][0] as i32 + c.eob_cost(eob, info);
    let max_pos = init_levels(q, scan, eob, info, levels);
    // Last coefficient (coded with the eob context).
    {
        let si = eob - 1;
        let ci = scan[si] as usize;
        let v = q[ci];
        cost += c.coeff_cost_general(true, ci, v.abs(), (v < 0) as usize, eob_ctx(si, info), dc_ctx, info, levels);
    }
    let n = info.stride() * (info.width + 4) + 16;
    assert!(levels.len() >= LEVELS_BUF && scan.len() >= eob && q.len() >= info.width * info.height);
    assert!(n <= LEVELS_REGION - 64 && max_pos < n);
    #[cfg(target_arch = "x86_64")]
    if crate::has_avx2() {
        // SAFETY: AVX2 support was checked at runtime.
        return cost
            + unsafe {
                match info.tx_class {
                    TX_CLASS_2D => cost_rest_avx2::<TX_CLASS_2D>(&c, q, eob, scan, info, dc_ctx, levels, max_pos),
                    TX_CLASS_HORIZ => cost_rest_avx2::<TX_CLASS_HORIZ>(&c, q, eob, scan, info, dc_ctx, levels, max_pos),
                    _ => cost_rest_avx2::<TX_CLASS_VERT>(&c, q, eob, scan, info, dc_ctx, levels, max_pos),
                }
            };
    }
    #[cfg(target_arch = "aarch64")]
    if crate::has_neon() {
        // SAFETY: NEON is part of the aarch64 baseline; the sizes were asserted above.
        return cost
            + unsafe {
                match info.tx_class {
                    TX_CLASS_2D => cost_rest_neon::<TX_CLASS_2D>(&c, q, eob, scan, info, dc_ctx, levels, max_pos),
                    TX_CLASS_HORIZ => cost_rest_neon::<TX_CLASS_HORIZ>(&c, q, eob, scan, info, dc_ctx, levels, max_pos),
                    _ => cost_rest_neon::<TX_CLASS_VERT>(&c, q, eob, scan, info, dc_ctx, levels, max_pos),
                }
            };
    }
    match info.tx_class {
        TX_CLASS_2D => cost + cost_rest::<TX_CLASS_2D, false>(&c, q, eob, scan, info, dc_ctx, levels),
        TX_CLASS_HORIZ => cost + cost_rest::<TX_CLASS_HORIZ, false>(&c, q, eob, scan, info, dc_ctx, levels),
        _ => cost + cost_rest::<TX_CLASS_VERT, false>(&c, q, eob, scan, info, dc_ctx, levels),
    }
}

/// Padded coefficient-base context offsets per (transform size, class), indexed
/// like the levels buffer.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn ctx_offsets(tx: TxSize, class: u8) -> &'static [u8] {
    use std::sync::OnceLock;
    static T: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    let t = T.get_or_init(|| {
        let mut all = Vec::new();
        for &tx in &TxSize::ALL {
            for class in 0..3u8 {
                let c = tx.coded();
                let (w, h, bhl) = (c.width(), c.height(), c.h_log2());
                let s = h + TX_PAD_HOR;
                let mut v = vec![0u8; LEVELS_REGION];
                for col in 0..w {
                    for row in 0..h {
                        let ci = (col << bhl) | row;
                        v[col * s + row] = match class {
                            TX_CLASS_2D => nz_map_ctx_offset(tx)[ci] as u8,
                            TX_CLASS_HORIZ => NZ_MAP_CTX_OFFSET_1D[col] as u8,
                            _ => NZ_MAP_CTX_OFFSET_1D[row] as u8,
                        };
                    }
                }
                all.push(v);
            }
        }
        all
    });
    &t[tx as usize * 3 + class as usize]
}

/// Coefficient-base contexts of every position up to `max_pos` at once.
///
/// # Safety
/// AVX2 must be available; `levels` holds the padded levels in its first
/// region (sized for `info`, `max_pos` within it) and has room for the map in the second.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn ctx_map_avx2<const CLASS: u8>(levels: &mut [u8], info: &TxbInfo, max_pos: usize) {
    use core::arch::x86_64::*;
    let s = info.stride();
    let offs = ctx_offsets(info.tx, CLASS);
    let (lv, map) = levels.split_at_mut(LEVELS_REGION);
    let lp = lv.as_ptr();
    let mp = map.as_mut_ptr();
    let op = offs.as_ptr();
    let three = _mm256_set1_epi8(3);
    let four = _mm256_set1_epi8(4);
    let one = _mm256_set1_epi8(1);
    let mask = _mm256_set1_epi8(0x7f);
    let mut p = 0usize;
    // Reads reach at most max_pos + 31 + 4 * stride < LEVELS_REGION.
    // SAFETY: AVX2 is guaranteed by the caller; every load/store stays inside the
    // padded regions per the bounds argument above.
    unsafe {
        while p <= max_pos {
            let m = |o: usize| _mm256_min_epu8(_mm256_loadu_si256(lp.add(p + o) as *const __m256i), three);
            let mut mag = _mm256_add_epi8(m(s), m(1));
            mag = match CLASS {
                TX_CLASS_2D => _mm256_add_epi8(mag, _mm256_add_epi8(_mm256_add_epi8(m(s + 1), m(2 * s)), m(2))),
                TX_CLASS_VERT => _mm256_add_epi8(mag, _mm256_add_epi8(_mm256_add_epi8(m(2), m(3)), m(4))),
                _ => _mm256_add_epi8(mag, _mm256_add_epi8(_mm256_add_epi8(m(2 * s), m(3 * s)), m(4 * s))),
            };
            let half = _mm256_and_si256(_mm256_srli_epi16(_mm256_add_epi8(mag, one), 1), mask);
            let ctx = _mm256_add_epi8(_mm256_min_epu8(half, four), _mm256_loadu_si256(op.add(p) as *const __m256i));
            _mm256_storeu_si256(mp.add(p) as *mut __m256i, ctx);
            p += 32;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn cost_rest_avx2<const CLASS: u8>(
    c: &TxbCosts,
    q: &[i32],
    eob: usize,
    scan: &[u16],
    info: &TxbInfo,
    dc_ctx: usize,
    levels: &mut [u8],
    max_pos: usize,
) -> i32 {
    // SAFETY: forwarded from the caller (AVX2 present, sizes asserted).
    unsafe { ctx_map_avx2::<CLASS>(levels, info, max_pos) };
    cost_rest::<CLASS, true>(c, q, eob, scan, info, dc_ctx, levels)
}

/// NEON twin of `ctx_map_avx2` (sixteen positions per step).
///
/// # Safety
/// `levels` holds the padded levels in its first region (sized for `info`,
/// `max_pos` within it) and has room for the map in the second.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn ctx_map_neon<const CLASS: u8>(levels: &mut [u8], info: &TxbInfo, max_pos: usize) {
    use core::arch::aarch64::*;
    let s = info.stride();
    let offs = ctx_offsets(info.tx, CLASS);
    let (lv, map) = levels.split_at_mut(LEVELS_REGION);
    let lp = lv.as_ptr();
    let mp = map.as_mut_ptr();
    let op = offs.as_ptr();
    let three = vdupq_n_u8(3);
    let four = vdupq_n_u8(4);
    let one = vdupq_n_u8(1);
    let mut p = 0usize;
    // Reads reach at most max_pos + 15 + 4 * stride < LEVELS_REGION.
    // SAFETY: every load/store stays inside the padded regions per the bounds argument above.
    unsafe {
        while p <= max_pos {
            let m = |o: usize| vminq_u8(vld1q_u8(lp.add(p + o)), three);
            let mut mag = vaddq_u8(m(s), m(1));
            mag = match CLASS {
                TX_CLASS_2D => vaddq_u8(mag, vaddq_u8(vaddq_u8(m(s + 1), m(2 * s)), m(2))),
                TX_CLASS_VERT => vaddq_u8(mag, vaddq_u8(vaddq_u8(m(2), m(3)), m(4))),
                _ => vaddq_u8(mag, vaddq_u8(vaddq_u8(m(2 * s), m(3 * s)), m(4 * s))),
            };
            let half = vshrq_n_u8::<1>(vaddq_u8(mag, one));
            let ctx = vaddq_u8(vminq_u8(half, four), vld1q_u8(op.add(p)));
            vst1q_u8(mp.add(p), ctx);
            p += 16;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn cost_rest_neon<const CLASS: u8>(
    c: &TxbCosts,
    q: &[i32],
    eob: usize,
    scan: &[u16],
    info: &TxbInfo,
    dc_ctx: usize,
    levels: &mut [u8],
    max_pos: usize,
) -> i32 {
    // SAFETY: forwarded from the caller (sizes asserted).
    unsafe { ctx_map_neon::<CLASS>(levels, info, max_pos) };
    cost_rest::<CLASS, true>(c, q, eob, scan, info, dc_ctx, levels)
}

/// Cost of the coefficients before the last one in scan order. The caller has
/// checked that `levels`, `q` and `scan` are large enough for `info`.
#[inline(always)]
fn cost_rest<const CLASS: u8, const MAP: bool>(
    c: &TxbCosts,
    q: &[i32],
    eob: usize,
    scan: &[u16],
    info: &TxbInfo,
    dc_ctx: usize,
    levels: &[u8],
) -> i32 {
    let bhl = info.bhl;
    let s = info.stride();
    let hmask = info.height - 1;
    let offs = nz_map_ctx_offset(info.tx);
    let base = &c.k.base[c.ts][c.pt];
    let mut cost = 0;
    // Scan position 0 is always the DC coefficient (raster index 0); every other
    // position has ci != 0, so its sign costs a flat bit.
    for si in (0..eob - 1).rev() {
        // SAFETY: `scan[si]` < width * height <= q.len() and offs.len(); the padded
        // position plus the largest neighbour offset is < stride * (width + 4) + 16 <= levels.len().
        let (ci, abs, v, pos) = unsafe {
            let ci = *scan.get_unchecked(si) as usize;
            let v = *q.get_unchecked(ci);
            (ci, v.abs(), v, ci + ((ci >> bhl) << 2))
        };
        let ctx = if CLASS == TX_CLASS_2D && ci == 0 {
            0
        } else if MAP {
            // SAFETY: `pos` < LEVELS_REGION (see above) and the context map follows the levels
            // region in `levels` (LEVELS_BUF), so `LEVELS_REGION + pos` is in bounds.
            unsafe { *levels.get_unchecked(LEVELS_REGION + pos) as usize }
        } else {
            // SAFETY: `pos` plus the largest neighbour offset used here (4 * s) is in bounds
            // of `levels` (see above).
            let l = |i: usize| unsafe { (*levels.get_unchecked(pos + i) as usize).min(3) };
            let mut mag = l(s) + l(1);
            match CLASS {
                TX_CLASS_2D => mag += l(s + 1) + l(2 * s) + l(2),
                TX_CLASS_VERT => mag += l(2) + l(3) + l(4),
                _ => mag += l(2 * s) + l(3 * s) + l(4 * s),
            }
            let ctx = ((mag + 1) >> 1).min(4);
            match CLASS {
                // SAFETY: `ci` < width * height <= offs.len() (see above).
                TX_CLASS_2D => ctx + unsafe { *offs.get_unchecked(ci) } as usize,
                TX_CLASS_HORIZ => ctx + NZ_MAP_CTX_OFFSET_1D[ci >> bhl],
                _ => ctx + NZ_MAP_CTX_OFFSET_1D[ci & hmask],
            }
        };
        // SAFETY: `ctx` is at most 4 + 36 (min(mag, 4) plus the largest context offset), which is
        // below the 42 base contexts in `base`.
        let mut cc = unsafe { base.get_unchecked(ctx)[abs.min(3) as usize] };
        if si == 0 {
            if abs != 0 {
                cc += c.k.dc_sign[c.pt][dc_ctx][(v < 0) as usize] as i32;
            }
        } else {
            cc += (abs != 0) as i32 * 512;
        }
        if abs > NUM_BASE_LEVELS {
            cc += br_cost(abs, c.lps(br_ctx(levels, ci, info)));
        }
        cost += cc;
    }
    cost
}

/// Write the coefficients of one transform block (libaom av1_write_coeffs_txb),
/// including the luma transform type.
#[allow(clippy::too_many_arguments)]
pub fn write_coeffs<W: Writer>(
    w: &mut W,
    cdf: &mut CoefCdfs,
    q: &[i32],
    eob: usize,
    scan: &[u16],
    info: &TxbInfo,
    plane_type: usize,
    skip_ctx: usize,
    dc_ctx: usize,
    write_tx_type: impl FnOnce(&mut W),
    levels: &mut [u8],
) {
    let ts = info.txs_ctx;
    w.symbol((eob == 0) as usize, &mut cdf.txb_skip[ts][skip_ctx], 2);
    if eob == 0 {
        return;
    }
    write_tx_type(w);
    let (pt_tok, extra) = eob_pos_token(eob);
    let multi = info.tx.coded().w_log2() + info.tx.coded().h_log2() - 4;
    let ectx = (info.tx_class != TX_CLASS_2D) as usize;
    let s = pt_tok - 1;
    match multi {
        0 => w.symbol(s, &mut cdf.eob16[plane_type][ectx], 5),
        1 => w.symbol(s, &mut cdf.eob32[plane_type][ectx], 6),
        2 => w.symbol(s, &mut cdf.eob64[plane_type][ectx], 7),
        3 => w.symbol(s, &mut cdf.eob128[plane_type][ectx], 8),
        4 => w.symbol(s, &mut cdf.eob256[plane_type][ectx], 9),
        5 => w.symbol(s, &mut cdf.eob512[plane_type][ectx], 10),
        _ => w.symbol(s, &mut cdf.eob1024[plane_type][ectx], 11),
    }
    let bits = EOB_OFFSET_BITS[pt_tok];
    if bits > 0 {
        let bit = ((extra >> (bits - 1)) & 1) as usize;
        w.symbol(bit, &mut cdf.eob_extra[ts][plane_type][pt_tok - 3], 2);
        for i in 1..bits {
            w.bit((extra >> (bits - 1 - i)) & 1 != 0);
        }
    }
    init_levels(q, scan, eob, info, levels);
    for si in (0..eob).rev() {
        let ci = scan[si] as usize;
        let level = q[ci].unsigned_abs() as usize;
        if si == eob - 1 {
            let ctx = eob_ctx(si, info);
            w.symbol(level.min(3) - 1, &mut cdf.base_eob[ts][plane_type][ctx], 3);
        } else {
            let ctx = lower_levels_ctx(levels, ci, info);
            w.symbol(level.min(3), &mut cdf.base[ts][plane_type][ctx], 4);
        }
        if level > 2 {
            let base_range = level - 3;
            let b = br_ctx(levels, ci, info);
            let brcdf = &mut cdf.br[ts.min(3)][plane_type][b];
            let mut idx = 0;
            while idx < COEFF_BASE_RANGE {
                let k = (base_range - idx).min(3);
                w.symbol(k, brcdf, 4);
                if k < 3 {
                    break;
                }
                idx += 3;
            }
        }
    }
    for si in 0..eob {
        let v = q[scan[si] as usize];
        if v != 0 {
            let sign = v < 0;
            if si == 0 {
                w.symbol(sign as usize, &mut cdf.dc_sign[plane_type][dc_ctx], 2);
            } else {
                w.bit(sign);
            }
            let level = v.unsigned_abs();
            if level > 14 {
                w.golomb(level - 15);
            }
        }
    }
}

#[inline]
fn coeff_dist(t: i32, dq: i32, shift: u32, wt: Option<&[u8]>, ci: usize) -> i64 {
    let diff = ((t - dq) as i64) << shift;
    match wt {
        None => diff * diff,
        Some(m) => {
            let d = diff * m[ci] as i64;
            (d * d + (1 << 9)) >> 10
        }
    }
}

#[inline]
pub fn rdcost(rdmult: i64, rate: i64, dist: i64) -> i64 {
    ((rate * rdmult + 256) >> 9) + (dist << 7)
}

/// libaom's av1_optimize_txb with sharpness > 0 (tune IQ uses 7).
/// Modifies `q`/`dq` in place and returns (eob, rate) where the rate
/// includes the skip flag and the eob but not the transform type.
#[allow(clippy::too_many_arguments)]
pub fn optimize(
    k: &CoefCosts,
    tcoeff: &[i32],
    q: &mut [i32],
    dq: &mut [i32],
    mut eob: usize,
    scan: &[u16],
    info: &TxbInfo,
    plane_type: usize,
    skip_ctx: usize,
    dc_ctx: usize,
    quant: &PlaneQuant,
    tx_type: TxType,
    rdmult: i64,
    sharpness: i32,
    levels: &mut [u8],
) -> (usize, i32) {
    let ts = info.txs_ctx;
    if eob == 0 {
        return (0, k.txb_skip[ts][skip_ctx][1] as i32);
    }
    let c = TxbCosts { k, ts, pt: plane_type };
    let shift = tx_scale(info.tx);
    let mats = quant.matrices(info.tx, tx_type);
    let iqm = mats.map(|m| m.1);
    let qm = mats.map(|m| m.0);
    // Trellis rdmult: rdmult * (8 - sharpness) * plane_rd_mult_chroma[intra][pt] >> 7 (tune IQ rshift)
    let plane_mult = if plane_type == 0 { 17 } else { 13 };
    let rdmult = (rdmult * (8 - sharpness) as i64 * plane_mult + 64) >> 7;

    init_levels(q, scan, eob, info, levels);
    let non_skip = k.txb_skip[ts][skip_ctx][0] as i32;
    let mut accu_rate = c.eob_cost(eob, info);
    let mut accu_dist: i64 = 0;
    let mut si = eob as isize - 1;
    let ci = scan[si as usize] as usize;
    let abs_qc = q[ci].abs();
    let mut nz_num = 1;
    let mut nz_ci = [ci, 0, 0];
    let dqv = |ci: usize| quant.dqv(ci, iqm);

    // update_coeff_general
    let update_general = |si: usize,
                          eob: usize,
                          q: &mut [i32],
                          dq: &mut [i32],
                          levels: &mut [u8],
                          accu_rate: &mut i32,
                          accu_dist: &mut i64| {
        let ci = scan[si] as usize;
        let qc = q[ci];
        let is_last = si == eob - 1;
        let ctx = if is_last { eob_ctx(si, info) } else { lower_levels_ctx(levels, ci, info) };
        if qc == 0 {
            *accu_rate += c.base(ctx)[0];
            return;
        }
        let sign = (qc < 0) as usize;
        let abs = qc.abs();
        let t = tcoeff[ci];
        let d = dq[ci];
        let rate = c.coeff_cost_general(is_last, ci, abs, sign, ctx, dc_ctx, info, levels);
        let (q_low, dq_low, rate_low) = if abs == 1 {
            (0, 0, c.base(ctx)[0])
        } else {
            let al = abs - 1;
            let adl = (al * dqv(ci)) >> shift;
            let rl = c.coeff_cost_general(is_last, ci, al, sign, ctx, dc_ctx, info, levels);
            (if sign == 1 { -al } else { al }, if sign == 1 { -adl } else { adl }, rl)
        };
        let d0 = coeff_dist(t, 0, shift, qm, ci);
        let dd = coeff_dist(t, d, shift, qm, ci) - d0;
        let dd_low = if abs == 1 { 0 } else { coeff_dist(t, dq_low, shift, qm, ci) - d0 };
        if rdcost(rdmult, rate_low as i64, dd_low) < rdcost(rdmult, rate as i64, dd) {
            q[ci] = q_low;
            dq[ci] = dq_low;
            levels[info.padded(ci)] = q_low.unsigned_abs().min(127) as u8;
            *accu_rate += rate_low;
            *accu_dist += dd_low;
        } else {
            *accu_rate += rate;
            *accu_dist += dd;
        }
    };

    if abs_qc >= 2 {
        update_general(si as usize, eob, q, dq, levels, &mut accu_rate, &mut accu_dist);
        si -= 1;
    } else {
        let ctx = eob_ctx(si as usize, info);
        accu_rate += c.coeff_cost_eob(ci, abs_qc, (q[ci] < 0) as usize, ctx, dc_ctx, info);
        let t = tcoeff[ci];
        accu_dist += coeff_dist(t, dq[ci], shift, qm, ci) - coeff_dist(t, 0, shift, qm, ci);
        si -= 1;
    }

    // update_coeff_eob for up to max_nz_num (2) nonzeros
    while si >= 0 && nz_num <= 2 {
        let s = si as usize;
        let ci = scan[s] as usize;
        let qc = q[ci];
        let ctx = lower_levels_ctx(levels, ci, info);
        if qc == 0 {
            accu_rate += c.base(ctx)[0];
        } else {
            let mut lower_level = false;
            let abs = qc.abs();
            let t = tcoeff[ci];
            let d = dq[ci];
            let sign = (qc < 0) as usize;
            let (q_low, dq_low, abs_low, rate_low) = if abs == 1 {
                (0, 0, 0, c.base(ctx)[0])
            } else {
                let al = abs - 1;
                let adl = (al * dqv(ci)) >> shift;
                let rl = c.coeff_cost_general(false, ci, al, sign, ctx, dc_ctx, info, levels);
                (if sign == 1 { -al } else { al }, if sign == 1 { -adl } else { adl }, al, rl)
            };
            let d0 = coeff_dist(t, 0, shift, qm, ci);
            let mut dist = coeff_dist(t, d, shift, qm, ci) - d0;
            let dist_low = if abs == 1 { 0 } else { coeff_dist(t, dq_low, shift, qm, ci) - d0 };
            let mut rate = c.coeff_cost_general(false, ci, abs, sign, ctx, dc_ctx, info, levels);
            let mut rd = rdcost(rdmult, (accu_rate + rate) as i64, accu_dist + dist);
            let rd_low = rdcost(rdmult, (accu_rate + rate_low) as i64, accu_dist + dist_low);
            let mut lower_level_new_eob = false;
            let new_eob = s + 1;
            let ctx_new_eob = eob_ctx(s, info);
            let new_eob_cost = c.eob_cost(new_eob, info);
            let mut rate_coeff_eob = new_eob_cost + c.coeff_cost_eob(ci, abs, sign, ctx_new_eob, dc_ctx, info);
            let mut dist_new_eob = dist;
            let mut rd_new_eob = rdcost(rdmult, rate_coeff_eob as i64, dist_new_eob);
            if abs_low > 0 {
                let r = new_eob_cost + c.coeff_cost_eob(ci, abs_low, sign, ctx_new_eob, dc_ctx, info);
                let rdl = rdcost(rdmult, r as i64, dist_low);
                if rdl < rd_new_eob {
                    lower_level_new_eob = true;
                    rd_new_eob = rdl;
                    rate_coeff_eob = r;
                    dist_new_eob = dist_low;
                }
            }
            let thr = if s <= 5 { 2 } else { 1 };
            let allow_lower = if sharpness != 0 { abs > thr } else { true };
            if allow_lower && rd_low < rd {
                lower_level = true;
                rd = rd_low;
                rate = rate_low;
                dist = dist_low;
            }
            if (sharpness == 0 || new_eob >= 5) && rd_new_eob < rd {
                for &lc in &nz_ci[..nz_num] {
                    levels[info.padded(lc)] = 0;
                    q[lc] = 0;
                    dq[lc] = 0;
                }
                eob = new_eob;
                nz_num = 0;
                accu_rate = rate_coeff_eob;
                accu_dist = dist_new_eob;
                lower_level = lower_level_new_eob;
            } else {
                accu_rate += rate;
                accu_dist += dist;
            }
            if lower_level {
                q[ci] = q_low;
                dq[ci] = dq_low;
                levels[info.padded(ci)] = q_low.unsigned_abs().min(127) as u8;
            }
            if q[ci] != 0 {
                nz_ci[nz_num] = ci;
                nz_num += 1;
            }
        }
        si -= 1;
    }
    // (update_skip only runs when sharpness == 0)
    if si == -1 && nz_num <= 2 && sharpness == 0 {
        let skip = k.txb_skip[ts][skip_ctx][1] as i32;
        if rdcost(rdmult, skip as i64, 0) < rdcost(rdmult, (accu_rate + non_skip) as i64, accu_dist) {
            for &lc in &nz_ci[..nz_num] {
                q[lc] = 0;
                dq[lc] = 0;
            }
            return (0, skip);
        }
    }

    // update_coeff_simple for si >= 1
    while si >= 1 {
        let s = si as usize;
        let ci = scan[s] as usize;
        let qc = q[ci];
        let ctx = lower_levels_ctx(levels, ci, info);
        if qc == 0 {
            accu_rate += c.base(ctx)[0];
        } else {
            let abs = qc.abs();
            let abs_t = tcoeff[ci].abs();
            let abs_dq = dq[ci].abs();
            let base = c.base(ctx);
            if abs == 1 {
                let rate = base[1] + 512;
                if abs_dq < abs_t {
                    accu_rate += rate;
                } else if sharpness == 0 {
                    let dd = coeff_dist(abs_t, abs_dq, shift, qm, ci) - coeff_dist(abs_t, 0, shift, qm, ci);
                    let rate_low = rate - base[5];
                    if rdcost(rdmult, rate_low as i64, 0) < rdcost(rdmult, rate as i64, dd) {
                        q[ci] = 0;
                        dq[ci] = 0;
                        levels[info.padded(ci)] = 0;
                        accu_rate += rate_low;
                    } else {
                        accu_rate += rate;
                    }
                } else {
                    accu_rate += rate;
                }
            } else {
                // get_two_coeff_cost_simple
                let mut cost = base[abs.min(3) as usize];
                let mut diff = if abs <= 3 { base[abs as usize + 4] } else { 0 };
                cost += 512;
                if abs > NUM_BASE_LEVELS {
                    let lps = c.lps(br_ctx(levels, ci, info));
                    let base_range = ((abs - 3) as usize).min(COEFF_BASE_RANGE);
                    let mut golomb = 0;
                    if abs <= (COEFF_BASE_RANGE as i32) + 1 + NUM_BASE_LEVELS {
                        diff += lps[base_range + COEFF_BASE_RANGE + 1];
                    }
                    if abs >= (COEFF_BASE_RANGE as i32) + 1 + NUM_BASE_LEVELS {
                        let r = abs - COEFF_BASE_RANGE as i32 - NUM_BASE_LEVELS;
                        golomb = golomb_cost(abs);
                        // golomb_cost_diff: extra cost when r is a power of two
                        if r > 1 && (r & (r - 1)) == 0 {
                            diff += 1024;
                        } else if r == 1 {
                            diff += 512;
                        }
                    }
                    cost += lps[base_range] + golomb;
                }
                let rate = cost;
                let rate_low = cost - diff;
                if abs_dq < abs_t {
                    accu_rate += rate;
                } else {
                    let allow = if sharpness != 0 { abs > 1 } else { true };
                    if allow {
                        let al = abs - 1;
                        let adl = (al * dqv(ci)) >> shift;
                        let d0 = coeff_dist(abs_t, 0, shift, qm, ci);
                        let dd = coeff_dist(abs_t, abs_dq, shift, qm, ci) - d0;
                        let dd_low = coeff_dist(abs_t, adl, shift, qm, ci) - d0;
                        if rdcost(rdmult, rate_low as i64, dd_low) < rdcost(rdmult, rate as i64, dd) {
                            let neg = qc < 0;
                            q[ci] = if neg { -al } else { al };
                            dq[ci] = if neg { -adl } else { adl };
                            levels[info.padded(ci)] = al.min(127) as u8;
                            accu_rate += rate_low;
                        } else {
                            accu_rate += rate;
                        }
                    } else {
                        accu_rate += rate;
                    }
                }
            }
        }
        si -= 1;
    }
    if si == 0 {
        let mut dummy = 0;
        update_general(0, eob, q, dq, levels, &mut accu_rate, &mut dummy);
    }
    // eob may have shrunk if trailing coefficients became zero in the general update of the last one.
    while eob > 0 && q[scan[eob - 1] as usize] == 0 {
        eob -= 1;
    }
    if eob == 0 {
        return (0, k.txb_skip[ts][skip_ctx][1] as i32);
    }
    (eob, accu_rate + non_skip)
}

/// Decoder-exact dequantization of a block's coefficients.
pub fn dequantize(
    q: &[i32],
    scan: &[u16],
    eob: usize,
    quant: &PlaneQuant,
    tx: TxSize,
    tx_type: TxType,
    dq: &mut [i32],
) {
    let n = tx.coded().width() * tx.coded().height();
    dq[..n].fill(0);
    let shift = tx_scale(tx);
    let iqm = quant.matrices(tx, tx_type).map(|m| m.1);
    for &rc in &scan[..eob] {
        let rc = rc as usize;
        if q[rc] != 0 {
            dq[rc] = dequant_coeff(q[rc], quant.dqv(rc, iqm), shift);
        }
    }
}

#[cfg(all(test, any(target_arch = "x86_64", target_arch = "aarch64")))]
mod simd_tests {
    use super::*;

    /// The vector context map must equal the scalar neighbourhood formula for
    /// every position up to `max_pos`, in every transform size and class.
    #[test]
    fn context_map_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        if !crate::has_avx2() {
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if !crate::has_neon() {
            return;
        }
        let mut seed = 0x2468_ace1u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for &tx in &TxSize::ALL {
            for class in [TX_CLASS_2D, TX_CLASS_HORIZ, TX_CLASS_VERT] {
                let mut info = TxbInfo::new(tx, 0);
                info.tx_class = class;
                let s = info.stride();
                let n = s * (info.width + 4) + 16;
                let max_pos = n - 1;
                let mut levels = vec![0u8; LEVELS_BUF];
                for v in &mut levels[..LEVELS_REGION] {
                    *v = match rnd() % 5 {
                        0 => 0,
                        1 => (rnd() % 4) as u8,
                        2 => 127,
                        _ => (rnd() % 8) as u8,
                    };
                }
                let offs = ctx_offsets(tx, class).to_vec();
                // SAFETY: support was checked above; `levels` has the sizes the kernel expects and
                // `max_pos` is below the bound asserted by `cost_coeffs`.
                unsafe {
                    #[cfg(target_arch = "x86_64")]
                    match class {
                        TX_CLASS_2D => ctx_map_avx2::<TX_CLASS_2D>(&mut levels, &info, max_pos),
                        TX_CLASS_HORIZ => ctx_map_avx2::<TX_CLASS_HORIZ>(&mut levels, &info, max_pos),
                        _ => ctx_map_avx2::<TX_CLASS_VERT>(&mut levels, &info, max_pos),
                    }
                    #[cfg(target_arch = "aarch64")]
                    match class {
                        TX_CLASS_2D => ctx_map_neon::<TX_CLASS_2D>(&mut levels, &info, max_pos),
                        TX_CLASS_HORIZ => ctx_map_neon::<TX_CLASS_HORIZ>(&mut levels, &info, max_pos),
                        _ => ctx_map_neon::<TX_CLASS_VERT>(&mut levels, &info, max_pos),
                    }
                }
                for pos in 0..=max_pos {
                    let l = |o: usize| (levels[pos + o] as usize).min(3);
                    let mag = match class {
                        TX_CLASS_2D => l(s) + l(1) + l(s + 1) + l(2 * s) + l(2),
                        TX_CLASS_VERT => l(s) + l(1) + l(2) + l(3) + l(4),
                        _ => l(s) + l(1) + l(2 * s) + l(3 * s) + l(4 * s),
                    };
                    let expected = ((mag + 1) >> 1).min(4) + offs[pos] as usize;
                    assert_eq!(levels[LEVELS_REGION + pos] as usize, expected, "{tx:?} class {class} pos {pos}");
                }
            }
        }
    }
}
