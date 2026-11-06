//! Coding of one block: geometry, prediction availability, transform type
//! search for each transform block, reconstruction and entropy contexts.

use super::{Encoder, MiInfo, TxbDec};
use crate::avail::{has_bottom_left, has_top_right, scale_chroma_bsize};
use crate::coef::{self, LEVELS_BUF, TxbInfo};
use crate::consts::*;
use crate::predict::{self, EdgeAvail, IntraParams};
use crate::quant::{PlaneQuant, quantize_fp, tx_scale};
use crate::tables::scan;
use crate::txfm;

/// Transform types tried for an intra luma block (libaom's reduced intra set).
pub fn tx_type_candidates(tx: TxSize, intra_dir: u8, out: &mut [TxType; 16]) -> usize {
    let set = intra_tx_set(tx, false);
    if set == EXT_TX_SET_DCTONLY {
        out[0] = DCT_DCT;
        return 1;
    }
    let mask: u16 = if set == EXT_TX_SET_DTT4_IDTX_1DDCT {
        const REDUCED: [u16; 13] =
            [0x080F, 0x040F, 0x080F, 0x020F, 0x080F, 0x040F, 0x080F, 0x080F, 0x040F, 0x080F, 0x040F, 0x080F, 0x0C0E];
        REDUCED[intra_dir as usize]
    } else {
        0x020F
    };
    let mut n = 0;
    for t in 0..16u8 {
        if mask & (1 << t) != 0 && tx_in_set(set, t) {
            out[n] = t;
            n += 1;
        }
    }
    n
}

/// Transform type of a chroma transform block (derived from the uv mode).
pub fn uv_tx_type(uv_mode: u8, tx: TxSize) -> TxType {
    let t = MODE_TO_TXFM[uv_mode as usize];
    let set = intra_tx_set(tx, false);
    if tx_in_set(set, t) { t } else { DCT_DCT }
}

/// Chroma transform size of a block.
pub fn uv_tx_size(bsize: BlockSize, ssx: usize, ssy: usize) -> TxSize {
    bsize.subsampled(ssx, ssy).max_tx().coded()
}

/// Mode parameters of a block.
#[derive(Clone, Copy, Debug)]
pub struct Modes {
    pub y_mode: u8,
    pub angle_y: i8,
    pub filter_intra: Option<u8>,
    pub uv_mode: u8,
    pub angle_uv: i8,
    pub cfl_alpha: [i8; 2],
}

impl Default for Modes {
    fn default() -> Self {
        Modes { y_mode: DC_PRED, angle_y: 0, filter_intra: None, uv_mode: DC_PRED, angle_uv: 0, cfl_alpha: [0, 0] }
    }
}

/// How hard to search transform types for one transform block.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TxSearch {
    /// Use the given type.
    Fixed(TxType),
    /// Try DCT_DCT and the given type.
    Pair(TxType),
    /// Try the candidate set; `trellis` enables coefficient optimization.
    Search { trellis: bool },
}

/// Result of coding one plane of a block.
#[derive(Default, Clone, Copy, Debug)]
pub struct PlaneRd {
    pub rate: i64,
    pub dist: i64,
    /// Any non-zero coefficient.
    pub nonzero: bool,
}

/// Exclusive bound for a ranking candidate, before any final skip-rate adjustment.
/// Rate and distortion already accumulated outside the plane are combined before
/// rdcost rounding. A rejected plane leaves partial state; the caller must reset
/// candidate contexts and reconstruct the eventual winner.
#[derive(Clone, Copy)]
pub(super) struct PlaneBound {
    pub rd: i64,
    pub rate: i64,
    pub dist: i64,
}

/// Block position info shared by the planes.
#[derive(Clone, Copy)]
pub struct BlockPos {
    pub bsize: BlockSize,
    pub mi_row: usize,
    pub mi_col: usize,
    pub partition: Partition,
    /// Luma transform size (needed by chroma-from-luma).
    pub luma_tx: TxSize,
}

pub struct Scratch {
    pub pred: Vec<u8>,
    pub resid: Vec<i16>,
    pub coeff: Vec<i32>,
    pub q: Vec<i32>,
    pub dq: Vec<i32>,
    pub best_q: Vec<i32>,
    pub best_dq: Vec<i32>,
    pub levels: Vec<u8>,
    pub cfl_luma: Vec<i16>,
    pub recon: Vec<u8>,
    /// Set around ranking-only calls: a single-transform-block plane skips
    /// the reconstruction, which the final coding pass rebuilds.
    pub rank_only: bool,
    /// Plane whose per-transform-type search results the next single-transform-block call
    /// records in `rec` (a ranking pass whose winner is searched again later).
    pub rec_plane: Option<usize>,
    /// Plane whose results in `best_rec` replace the matching transform type evaluations.
    pub replay_plane: Option<usize>,
    pub rec: [TxbRecords; 3],
    pub best_rec: [TxbRecords; 3],
}

/// Search result of one transform type for a plane that is a single transform
/// block: the quantized coefficients and the RD terms.
pub struct TxbRecord {
    pub tt: TxType,
    pub eob: usize,
    pub rate: i64,
    pub dist: i64,
    pub q: Vec<i32>,
}

/// Up to two recorded transform types (the ranking passes evaluate at most two).
pub struct TxbRecords {
    pub n: usize,
    pub ncoef: usize,
    pub items: [TxbRecord; 2],
}

impl TxbRecords {
    fn new() -> TxbRecords {
        let item = || TxbRecord { tt: DCT_DCT, eob: 0, rate: 0, dist: 0, q: vec![0; 32 * 32] };
        TxbRecords { n: 0, ncoef: 0, items: [item(), item()] }
    }
    pub fn copy_from(&mut self, o: &TxbRecords) {
        self.n = o.n;
        self.ncoef = o.ncoef;
        for (d, s) in self.items.iter_mut().zip(&o.items).take(o.n) {
            (d.tt, d.eob, d.rate, d.dist) = (s.tt, s.eob, s.rate, s.dist);
            d.q[..o.ncoef].copy_from_slice(&s.q[..o.ncoef]);
        }
    }
    fn find(&self, ncoef: usize, tt: TxType) -> Option<&TxbRecord> {
        if self.ncoef != ncoef {
            return None;
        }
        self.items[..self.n].iter().find(|r| r.tt == tt)
    }
}

impl Scratch {
    pub fn new() -> Scratch {
        Scratch {
            pred: vec![0; 64 * 64],
            resid: vec![0; 64 * 64],
            coeff: vec![0; 64 * 64],
            q: vec![0; 32 * 32],
            dq: vec![0; 32 * 32],
            best_q: vec![0; 32 * 32],
            best_dq: vec![0; 32 * 32],
            levels: vec![0; LEVELS_BUF],
            cfl_luma: vec![0; 32 * 32],
            recon: vec![0; 64 * 64],
            rank_only: false,
            rec_plane: None,
            replay_plane: None,
            rec: [TxbRecords::new(), TxbRecords::new(), TxbRecords::new()],
            best_rec: [TxbRecords::new(), TxbRecords::new(), TxbRecords::new()],
        }
    }
}

fn residual(src: &[u8], ss: usize, pred: &[u8], w: usize, h: usize, out: &mut [i16]) {
    #[cfg(target_arch = "x86_64")]
    if crate::has_avx2() {
        // SAFETY: runtime detection respects AV1_NO_AVX2.
        return unsafe { residual_avx2(src, ss, pred, w, h, out) };
    }
    residual_scalar(src, ss, pred, w, h, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn residual_avx2(src: &[u8], ss: usize, pred: &[u8], w: usize, h: usize, out: &mut [i16]) {
    residual_scalar(src, ss, pred, w, h, out)
}

#[inline(always)]
fn residual_scalar(src: &[u8], ss: usize, pred: &[u8], w: usize, h: usize, out: &mut [i16]) {
    for r in 0..h {
        let src = &src[r * ss..r * ss + w];
        let pred = &pred[r * 64..r * 64 + w];
        let out = &mut out[r * w..(r + 1) * w];
        for c in 0..w {
            out[c] = src[c] as i16 - pred[c] as i16;
        }
    }
}

/// Transform-domain distortion, scaled like libaom (≈16 x pixel SSE),
/// optionally weighted by a quantization matrix (QM-PSNR).
pub fn tx_dist(t: &[i32], dq: &[i32], n: usize, scale: u32, wt: Option<&[u8]>) -> i64 {
    #[cfg(target_arch = "x86_64")]
    if crate::has_avx2() {
        // SAFETY: runtime detection also respects AV1_NO_AVX2.
        return unsafe { tx_dist_avx2(t, dq, n, scale, wt) };
    }
    tx_dist_scalar(t, dq, n, scale, wt)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn tx_dist_avx2(t: &[i32], dq: &[i32], n: usize, scale: u32, wt: Option<&[u8]>) -> i64 {
    tx_dist_scalar(t, dq, n, scale, wt)
}

// Inline into the target-feature wrapper so LLVM can vectorize the reduction
// for AVX2, while retaining the portable implementation and i64 arithmetic.
#[inline(always)]
fn tx_dist_scalar(t: &[i32], dq: &[i32], n: usize, scale: u32, wt: Option<&[u8]>) -> i64 {
    let t = &t[..n];
    let dq = &dq[..n];
    let mut e: i64 = 0;
    match wt {
        None => {
            for (&t, &dq) in t.iter().zip(dq) {
                let d = (t - dq) as i64;
                e += d * d;
            }
        }
        Some(m) => {
            for ((&t, &dq), &w) in t.iter().zip(dq).zip(&m[..n]) {
                let d = (t - dq) as i64 * w as i64;
                e += (d * d + 512) >> 10;
            }
        }
    }
    let shift = (1 - scale as i32) * 2;
    if shift >= 0 { e >> shift } else { e << (-shift) }
}

#[cfg(test)]
mod tests;

impl Encoder {
    #[inline]
    pub fn mi(&self, r: usize, c: usize) -> &MiInfo {
        &self.mi[r * self.mi_cols + c]
    }

    pub fn has_chroma(&self, bsize: BlockSize, mi_row: usize, mi_col: usize) -> bool {
        if self.num_planes == 1 {
            return false;
        }
        let (bw, bh) = (bsize.w4(), bsize.h4());
        ((mi_row & 1) == 1 || (bh & 1) == 0 || self.ssy == 0) && ((mi_col & 1) == 1 || (bw & 1) == 0 || self.ssx == 0)
    }

    /// Neighbour availability (inside the current tile).
    #[inline]
    pub fn avail_up(&self, mi_row: usize) -> bool {
        mi_row > self.tile.mi_row_start
    }
    #[inline]
    pub fn avail_left(&self, mi_col: usize) -> bool {
        mi_col > self.tile.mi_col_start
    }

    /// Whether the above or left neighbour (of the plane's reference block)
    /// uses a smooth mode: selects the intra edge filter strength table.
    pub fn smooth_neighbor(&self, plane: usize, bp: &BlockPos) -> bool {
        let is_smooth = |m: u8| (SMOOTH_PRED..=SMOOTH_H_PRED).contains(&m);
        let (r, c) = (bp.mi_row, bp.mi_col);
        if plane == 0 {
            let a = self.avail_up(r) && is_smooth(self.mi(r - 1, c).y_mode);
            let l = self.avail_left(c) && is_smooth(self.mi(r, c - 1).y_mode);
            a || l
        } else {
            let (bw, bh) = (bp.bsize.w4(), bp.bsize.h4());
            let up = if self.ssy == 1 && bh < 2 { r >= 1 && r - 1 > self.tile.mi_row_start } else { self.avail_up(r) };
            let left = if self.ssx == 1 && bw < 2 { c >= 1 && c - 1 > self.tile.mi_col_start } else { self.avail_left(c) };
            let br = r - (r & self.ssy);
            let bc = c - (c & self.ssx);
            let a = up && is_smooth(self.mi(br - 1, bc + self.ssx).uv_mode);
            let l = left && is_smooth(self.mi(br + self.ssy, bc - 1).uv_mode);
            a || l
        }
    }

    /// Code (predict, transform, quantize, reconstruct) one plane of a block
    /// with uniform transform size `tx`. Appends the transform blocks to
    /// `txbs` / `coefs` when given and updates the entropy contexts.
    #[allow(clippy::too_many_arguments)]
    pub fn code_plane(
        &mut self,
        plane: usize,
        bp: &BlockPos,
        modes: &Modes,
        tx: TxSize,
        search: TxSearch,
        quant: &PlaneQuant,
        rdmult: i64,
        s: &mut Scratch,
        out: Option<(&mut Vec<TxbDec>, &mut Vec<i32>)>,
    ) -> PlaneRd {
        self.try_code_plane(plane, bp, modes, tx, search, quant, rdmult, s, out, None).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_code_plane(
        &mut self,
        plane: usize,
        bp: &BlockPos,
        modes: &Modes,
        tx: TxSize,
        search: TxSearch,
        quant: &PlaneQuant,
        rdmult: i64,
        s: &mut Scratch,
        mut out: Option<(&mut Vec<TxbDec>, &mut Vec<i32>)>,
        bound: Option<PlaneBound>,
    ) -> Option<PlaneRd> {
        let (ssx, ssy) = if plane == 0 { (0, 0) } else { (self.ssx, self.ssy) };
        let bsize = bp.bsize;
        let pbs = bsize.subsampled(ssx, ssy);
        let pw = pbs.width();
        let ph = pbs.height();
        // Plane block origin in plane pixels.
        let x0 = (bp.mi_col >> ssx) * 4;
        let y0 = (bp.mi_row >> ssy) * 4;
        let plane_mi_w = (self.mi_cols * 4) >> ssx;
        let plane_mi_h = (self.mi_rows * 4) >> ssy;
        let (txw, txh) = (tx.width(), tx.height());
        let mode = if plane == 0 { modes.y_mode } else if modes.uv_mode == UV_CFL_PRED { DC_PRED } else { modes.uv_mode };
        let angle = if plane == 0 { modes.angle_y } else { modes.angle_uv } as i32;
        let fi = if plane == 0 { modes.filter_intra } else { None };
        let ip = IntraParams {
            mode,
            angle_delta: angle,
            filter_intra: fi,
            edge_filter: true,
            smooth_neighbor: if is_directional(mode) { self.smooth_neighbor(plane, bp) } else { false },
        };
        let (bw4, bh4) = (bsize.w4() as isize, bsize.h4() as isize);
        let mb_to_right = ((self.mi_cols as isize - bw4 - bp.mi_col as isize) * 4) >> ssx;
        let mb_to_bottom = ((self.mi_rows as isize - bh4 - bp.mi_row as isize) * 4) >> ssy;
        let wpx = pw.max(4) as isize;
        let hpx = ph.max(4) as isize;
        let (up_av, left_av) = if plane == 0 {
            (self.avail_up(bp.mi_row), self.avail_left(bp.mi_col))
        } else {
            let up = if ssy == 1 && bsize.h4() < 2 {
                bp.mi_row >= 1 && bp.mi_row - 1 > self.tile.mi_row_start
            } else {
                self.avail_up(bp.mi_row)
            };
            let left = if ssx == 1 && bsize.w4() < 2 {
                bp.mi_col >= 1 && bp.mi_col - 1 > self.tile.mi_col_start
            } else {
                self.avail_left(bp.mi_col)
            };
            (up, left)
        };
        let tr_bsize = if plane > 0 { scale_chroma_bsize(bsize, ssx, ssy) } else { bsize };
        let is_dr = is_directional(mode);
        let p_angle = MODE_TO_ANGLE[mode as usize] + angle * 3;
        let need_tr = fi.is_none() && is_dr && p_angle < 90;
        let need_bl = fi.is_none() && is_dr && p_angle > 180;
        let plane_type = (plane > 0) as usize;
        let intra_dir = match fi {
            Some(f) => FILTER_INTRA_MODE_TO_DIR[f as usize],
            None => modes.y_mode,
        };
        let mut cands = [0u8; 16];
        let ncand = if self.lossless {
            // Lossless blocks use the WHT and an implicit DCT_DCT type.
            cands[0] = DCT_DCT;
            1
        } else {
            match search {
            TxSearch::Fixed(t) => {
                cands[0] = t;
                1
            }
            TxSearch::Pair(t) => {
                cands[0] = DCT_DCT;
                cands[1] = t;
                if t == DCT_DCT || plane > 0 { 1 } else { 2 }
            }
            TxSearch::Search { .. } => {
                if crate::debug_flag("AV1_DCT_ONLY") {
                    cands[0] = DCT_DCT;
                    1
                } else if plane == 0 {
                    tx_type_candidates(tx, intra_dir, &mut cands)
                } else {
                    cands[0] = uv_tx_type(modes.uv_mode, tx);
                    1
                }
            }
            }
        };
        // Trellis quantization is off by default: on Kodak (24 images, qindex 30-220) it changed BD-rate
        // by less than 0.05% (PSNR and SSIMULACRA2) while costing 5-9% of the encode time.
        // `AV1_TRELLIS=1` turns it back on.
        let trellis = matches!(search, TxSearch::Search { trellis: true }) && crate::debug_flag("AV1_TRELLIS");
        let scale = tx_scale(tx);
        let coded = tx.coded();
        let ncoef = coded.width() * coded.height();
        let is64 = tx.width() == 64 || tx.height() == 64;

        let mut total = PlaneRd::default();
        let mut row_off = 0;
        while row_off < ph {
            let y = y0 + row_off;
            if y >= plane_mi_h {
                break;
            }
            let mut col_off = 0;
            while col_off < pw {
                let x = x0 + col_off;
                if x >= plane_mi_w {
                    break;
                }
                // --- prediction ---
                let xr = mb_to_right + wpx - col_off as isize - txw as isize;
                let yd = mb_to_bottom + hpx - row_off as isize - txh as isize;
                let have_top = row_off > 0 || up_av;
                let have_left = col_off > 0 || left_av;
                let n_top = if have_top { (txw as isize).min(xr + txw as isize).max(0) as usize } else { 0 };
                let n_left = if have_left { (txh as isize).min(yd + txh as isize).max(0) as usize } else { 0 };
                let mut av = EdgeAvail { n_top, n_topright: 0, n_left, n_bottomleft: 0 };
                if need_tr {
                    let right_available =
                        bp.mi_col + ((col_off / 4 + tx.w4()) << ssx) < self.tile.mi_col_end;
                    if has_top_right(
                        tr_bsize,
                        bp.mi_row,
                        bp.mi_col,
                        have_top,
                        right_available,
                        bp.partition,
                        tx,
                        row_off / 4,
                        col_off / 4,
                        ssx,
                        ssy,
                    ) {
                        av.n_topright = (txw as isize).min(xr).max(0) as usize;
                    }
                }
                if need_bl {
                    let bottom_available =
                        yd > 0 && bp.mi_row + ((row_off / 4 + tx.h4()) << ssy) < self.tile.mi_row_end;
                    if has_bottom_left(
                        tr_bsize,
                        bp.mi_row,
                        bp.mi_col,
                        bottom_available,
                        have_left,
                        bp.partition,
                        tx,
                        row_off / 4,
                        col_off / 4,
                        ssx,
                        ssy,
                    ) {
                        av.n_bottomleft = (txh as isize).min(yd).max(0) as usize;
                    }
                }
                {
                    let rec = &self.rec[plane];
                    predict::predict(&rec.data, rec.stride, x, y, tx, &ip, av, &mut s.pred, 64);
                }
                if plane > 0 && modes.uv_mode == UV_CFL_PRED {
                    self.cfl_luma(bp, plane, x, y, txw, txh, &mut s.cfl_luma);
                    predict::cfl_predict(&mut s.pred, 64, txw, txh, &s.cfl_luma, modes.cfl_alpha[plane - 1] as i32);
                }
                let x4 = x >> 2;
                let y4 = y >> 2;
                let single = row_off == 0 && col_off == 0 && txw >= pw && txh >= ph;
                let replay_src = if single { s.replay_plane } else { None };
                let rec_dst = if single { s.rec_plane } else { None };
                if let Some(i) = rec_dst {
                    (s.rec[i].n, s.rec[i].ncoef) = (0, ncoef);
                }
                let mut resid_ready = false;
                let (skip_ctx, dc_ctx) = {
                    let a = &self.ctx.above_ent[plane][x4..x4 + tx.w4()];
                    let l = &self.ctx.left_ent[plane][y4..y4 + tx.h4()];
                    coef::txb_ctx(plane, pbs, tx, a, l)
                };
                // --- transform type search ---
                let mut best = (i64::MAX, 0i64, 0i64, DCT_DCT, 0usize);
                for &tt in &cands[..ncand] {
                    // A ranking pass over this same block (same contexts and quantizer) may have
                    // evaluated this transform type already.
                    let replayed = replay_src.and_then(|i| s.best_rec[i].find(ncoef, tt));
                    let (eob, rate_all, dist) = if let Some(r) = replayed {
                        s.q[..ncoef].copy_from_slice(&r.q[..ncoef]);
                        (r.eob, r.rate, r.dist)
                    } else {
                        if !resid_ready {
                            let src = &self.src[plane];
                            residual(&src.data[y * src.stride + x..], src.stride, &s.pred, txw, txh, &mut s.resid);
                            resid_ready = true;
                        }
                        if self.lossless {
                            txfm::forward_wht4(&s.resid, txw, &mut s.coeff);
                        } else {
                            txfm::forward(&s.resid, txw, tx, tt, &mut s.coeff);
                        }
                        let info = TxbInfo::new(tx, tt);
                        let sc = scan(coded, info.tx_class);
                        let mut eob = quantize_fp(&s.coeff, quant, tx, tt, sc, &mut s.q, &mut s.dq);
                        let rate;
                        if trellis && eob > 0 && !self.lossless {
                            let (e, r) = coef::optimize(
                                &self.coef_costs,
                                &s.coeff,
                                &mut s.q,
                                &mut s.dq,
                                eob,
                                sc,
                                &info,
                                plane_type,
                                skip_ctx,
                                dc_ctx,
                                quant,
                                tt,
                                rdmult,
                                self.sharpness,
                                &mut s.levels,
                            );
                            eob = e;
                            rate = r as i64;
                        } else {
                            rate = coef::cost_coeffs(
                                &self.coef_costs,
                                &s.q,
                                eob,
                                sc,
                                &info,
                                plane_type,
                                skip_ctx,
                                dc_ctx,
                                &mut s.levels,
                            ) as i64;
                        }
                        let tt_rate = if plane == 0 && eob > 0 && !self.lossless {
                            self.mode_costs.tx_type_cost(tx, intra_dir, tt) as i64
                        } else {
                            0
                        };
                        let dist = if is64 {
                            // 64-point transforms drop half of the coefficients:
                            // measure the distortion in the pixel domain.
                            for r in 0..txh {
                                s.recon[r * 64..r * 64 + txw].copy_from_slice(&s.pred[r * 64..r * 64 + txw]);
                            }
                            if eob > 0 {
                                coef::dequantize(&s.q, sc, eob, quant, tx, tt, &mut s.dq);
                                txfm::inverse_add(&s.dq, tx, tt, eob, &mut s.recon, 64);
                            }
                            let src = &self.src[plane];
                            let mut sse = 0i64;
                            for r in 0..txh {
                                let so = (y + r) * src.stride + x;
                                for c in 0..txw {
                                    let d = src.data[so + c] as i32 - s.recon[r * 64 + c] as i32;
                                    sse += (d * d) as i64;
                                }
                            }
                            sse * 16
                        } else {
                            let wt = if self.qm_dist { quant.matrices(tx, tt).map(|m| m.0) } else { None };
                            tx_dist(&s.coeff, &s.dq, ncoef, scale, wt)
                        };
                        (eob, rate + tt_rate, dist)
                    };
                    if let Some(i) = rec_dst {
                        let rr = &mut s.rec[i];
                        if rr.n < 2 {
                            let it = &mut rr.items[rr.n];
                            (it.tt, it.eob, it.rate, it.dist) = (tt, eob, rate_all, dist);
                            it.q[..ncoef].copy_from_slice(&s.q[..ncoef]);
                            rr.n += 1;
                        }
                    }
                    let rd = coef::rdcost(rdmult, rate_all, dist);
                    // A DCT block with no coefficients is not worth a type search.
                    let first_empty = eob == 0 && best.0 == i64::MAX && tt == DCT_DCT;
                    if rd < best.0 {
                        best = (rd, rate_all, dist, tt, eob);
                        if ncand > 1 {
                            s.best_q[..ncoef].copy_from_slice(&s.q[..ncoef]);
                        }
                    }
                    if first_empty {
                        break;
                    }
                }
                let (_, rate, dist, tt, eob) = best;
                if ncand > 1 {
                    s.q[..ncoef].copy_from_slice(&s.best_q[..ncoef]);
                }
                total.rate += rate;
                total.dist += dist;
                total.nonzero |= eob > 0;
                if let Some(b) = bound {
                    // Later transform blocks add nonnegative costs. Only ranking
                    // uses this bound: final skip signalling can subtract rate.
                    if coef::rdcost(rdmult, b.rate + total.rate, b.dist + total.dist) >= b.rd {
                        return None;
                    }
                }
                let info = TxbInfo::new(tx, tt);
                let sc = scan(coded, info.tx_class);
                // --- reconstruction (decoder-exact) ---
                // A ranking-only call (no decisions kept) whose plane block is a
                // single transform block never reads its reconstruction: every
                // later stage predicts and reconstructs the block again.
                let recon_unused = s.rank_only && out.is_none() && row_off == 0 && col_off == 0 && txw >= pw && txh >= ph;
                if !recon_unused {
                    let rec = &mut self.rec[plane];
                    let stride = rec.stride;
                    let base = y * stride + x;
                    for r in 0..txh {
                        rec.data[base + r * stride..base + r * stride + txw]
                            .copy_from_slice(&s.pred[r * 64..r * 64 + txw]);
                    }
                    if eob > 0 {
                        coef::dequantize(&s.q, sc, eob, quant, tx, tt, &mut s.dq);
                        if self.lossless {
                            txfm::inverse_wht4_add(&s.dq, &mut rec.data[base..], stride);
                        } else {
                            txfm::inverse_add(&s.dq, tx, tt, eob, &mut rec.data[base..], stride);
                        }
                    }
                }
                // --- contexts ---
                let cul = coef::txb_entropy_ctx(&s.q, sc, eob);
                let max_x4 = plane_mi_w >> 2;
                let max_y4 = plane_mi_h >> 2;
                for i in 0..tx.w4() {
                    self.ctx.above_ent[plane][x4 + i] = if x4 + i < max_x4 { cul } else { 0 };
                }
                for i in 0..tx.h4() {
                    self.ctx.left_ent[plane][y4 + i] = if y4 + i < max_y4 { cul } else { 0 };
                }
                if let Some((txbs, coefs)) = out.as_mut() {
                    let start = coefs.len() as u32;
                    coefs.extend(sc[..eob].iter().map(|&rc| s.q[rc as usize]));
                    txbs.push(TxbDec { tx_type: tt, eob: eob as u16, coef_start: start });
                }
                col_off += txw;
            }
            row_off += txh;
        }
        Some(total)
    }

    /// Subsampled luma (Q3) for chroma-from-luma, padded to the transform size.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn cfl_luma(&self, bp: &BlockPos, _plane: usize, cx: usize, cy: usize, w: usize, h: usize, out: &mut [i16]) {
        let (ssx, ssy) = (self.ssx, self.ssy);
        let lx = cx << ssx;
        let ly = cy << ssy;
        // Luma extent that was reconstructed: luma transform blocks start
        // inside the mi-aligned frame.
        let ltx = bp.luma_tx;
        let lw = (w << ssx).min((self.mi_cols * 4).saturating_sub(lx).div_ceil(ltx.width()) * ltx.width());
        let lh = (h << ssy).min((self.mi_rows * 4).saturating_sub(ly).div_ceil(ltx.height()) * ltx.height());
        let sub_w = lw >> ssx;
        let sub_h = lh >> ssy;
        let rec = &self.rec[0];
        for r in 0..sub_h {
            for c in 0..sub_w {
                let v = if ssx == 1 && ssy == 1 {
                    let o = (ly + 2 * r) * rec.stride + lx + 2 * c;
                    (rec.data[o] as i16 + rec.data[o + 1] as i16 + rec.data[o + rec.stride] as i16
                        + rec.data[o + rec.stride + 1] as i16)
                        << 1
                } else {
                    (rec.data[(ly + r) * rec.stride + lx + c] as i16) << 3
                };
                out[r * w + c] = v;
            }
            for c in sub_w..w {
                out[r * w + c] = out[r * w + sub_w - 1];
            }
        }
        for r in sub_h..h {
            for c in 0..w {
                out[r * w + c] = out[(sub_h - 1) * w + c];
            }
        }
    }
}

