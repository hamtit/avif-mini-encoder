//! Bitstream syntax of superblocks (partition, mode info, coefficients).

use super::block::uv_tx_size;
use super::search::{cfl_allowed, filter_intra_allowed, gathered_prob, max_tx_depth, tx_depth, tx_size_cat};
use super::{BlockDec, Contexts, Decisions, Encoder};
use crate::cdf::CdfContext;
use crate::coef::{self, TxbInfo};
use crate::consts::*;
use crate::ec::Writer;
use crate::tables::scan;
use crate::tables::sizes::EXT_TX_IND;

struct WriteState<'a> {
    first: bool,
    cdef_written: bool,
    dq: Option<&'a mut u8>,
    cdef_idx: Option<i8>,
    part: usize,
    block: usize,
    levels: Vec<u8>,
    q: Vec<i32>,
}

impl Encoder {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_sb<W: Writer>(
        &self,
        w: &mut W,
        cdf: &mut CdfContext,
        ctx: &mut Contexts,
        dec: &Decisions,
        mi_row: usize,
        mi_col: usize,
        dq: Option<&mut u8>,
        cdef_idx: Option<i8>,
    ) {
        let mut st = WriteState {
            first: true,
            cdef_written: false,
            dq,
            cdef_idx,
            part: 0,
            block: 0,
            levels: vec![0; coef::LEVELS_BUF],
            q: vec![0; 32 * 32],
        };
        self.write_part(w, cdf, ctx, dec, BlockSize::B64x64, mi_row, mi_col, &mut st);
        debug_assert_eq!(st.part, dec.parts.len());
        debug_assert_eq!(st.block, dec.blocks.len());
    }

    #[allow(clippy::too_many_arguments)]
    fn write_part<W: Writer>(
        &self,
        w: &mut W,
        cdf: &mut CdfContext,
        ctx: &mut Contexts,
        dec: &Decisions,
        bsize: BlockSize,
        mi_row: usize,
        mi_col: usize,
        st: &mut WriteState,
    ) {
        if mi_row >= self.mi_rows || mi_col >= self.mi_cols {
            return;
        }
        let p = dec.parts[st.part];
        st.part += 1;
        let hbs = bsize.w4() / 2;
        let has_rows = mi_row + hbs < self.mi_rows;
        let has_cols = mi_col + hbs < self.mi_cols;
        let pctx = self.partition_ctx(bsize, mi_row, mi_col);
        let n = if bsize == BlockSize::B8x8 { 4 } else { 10 };
        if has_rows && has_cols {
            w.symbol(p as usize, &mut cdf.partition[pctx], n);
        } else if has_rows || has_cols {
            let psum = gathered_prob(&cdf.partition[pctx], !has_rows);
            let bin = [psum as u16, 0, 0];
            w.symbol_noadapt((p == Partition::Split) as usize, &bin, 2);
        }
        let sub = bsize.subsize(p).unwrap();
        match p {
            Partition::None => self.write_block(w, cdf, ctx, dec, st),
            Partition::Horz => {
                self.write_block(w, cdf, ctx, dec, st);
                if mi_row + hbs < self.mi_rows {
                    self.write_block(w, cdf, ctx, dec, st);
                }
            }
            Partition::Vert => {
                self.write_block(w, cdf, ctx, dec, st);
                if mi_col + hbs < self.mi_cols {
                    self.write_block(w, cdf, ctx, dec, st);
                }
            }
            _ => {
                for i in 0..4 {
                    let r = mi_row + (i >> 1) * hbs;
                    let c = mi_col + (i & 1) * hbs;
                    if r >= self.mi_rows || c >= self.mi_cols {
                        continue;
                    }
                    if sub == BlockSize::B4x4 {
                        self.write_block(w, cdf, ctx, dec, st);
                    } else {
                        self.write_part(w, cdf, ctx, dec, sub, r, c, st);
                    }
                }
            }
        }
    }

    fn write_block<W: Writer>(&self, w: &mut W, cdf: &mut CdfContext, ctx: &mut Contexts, dec: &Decisions, st: &mut WriteState) {
        let b: &BlockDec = &dec.blocks[st.block];
        st.block += 1;
        let (bsize, mi_row, mi_col) = (b.bsize, b.mi_row as usize, b.mi_col as usize);
        let m = &b.modes;
        // skip
        let sctx = {
            let a = self.avail_up(mi_row) && self.mi(mi_row - 1, mi_col).skip;
            let l = self.avail_left(mi_col) && self.mi(mi_row, mi_col - 1).skip;
            a as usize + l as usize
        };
        w.symbol(b.skip as usize, &mut cdf.skip[sctx], 2);
        // cdef index
        if !b.skip
            && !st.cdef_written
            && let Some(idx) = st.cdef_idx
        {
            w.literal(idx as u32, self.cdef_bits as u32);
            st.cdef_written = true;
        }
        // delta q
        if st.first {
            st.first = false;
            if let Some(cur) = st.dq.as_deref_mut()
                && !(bsize == BlockSize::B64x64 && b.skip)
            {
                let res = self.delta_q_res as i32;
                let delta = (dec.qindex as i32 - *cur as i32) / res;
                debug_assert_eq!(delta * res, dec.qindex as i32 - *cur as i32);
                let abs = delta.unsigned_abs();
                w.symbol(abs.min(3) as usize, &mut cdf.delta_q, 4);
                if abs >= 3 {
                    let rem_bits = 31 - (abs - 1).leading_zeros();
                    let thr = (1 << rem_bits) + 1;
                    w.literal(rem_bits - 1, 3);
                    w.literal(abs - thr, rem_bits);
                }
                if abs > 0 {
                    w.bit(delta < 0);
                }
                *cur = dec.qindex;
            }
        }
        // y mode
        let actx = if self.avail_up(mi_row) { INTRA_MODE_CONTEXT[self.mi(mi_row - 1, mi_col).y_mode as usize] } else { 0 };
        let lctx = if self.avail_left(mi_col) { INTRA_MODE_CONTEXT[self.mi(mi_row, mi_col - 1).y_mode as usize] } else { 0 };
        w.symbol(m.y_mode as usize, &mut cdf.kf_y[actx][lctx], 13);
        let use_angle = bsize >= BlockSize::B8x8;
        if use_angle && is_directional(m.y_mode) {
            w.symbol((m.angle_y + 3) as usize, &mut cdf.angle_delta[(m.y_mode - V_PRED) as usize], 7);
        }
        let has_chroma = self.has_chroma(bsize, mi_row, mi_col);
        if has_chroma {
            let cfl = cfl_allowed(bsize, self.lossless, self.ssx, self.ssy);
            w.symbol(m.uv_mode as usize, &mut cdf.uv_mode[cfl as usize][m.y_mode as usize], 13 + cfl as usize);
            if m.uv_mode == UV_CFL_PRED {
                let si = |a: i8| -> usize {
                    match a.signum() {
                        0 => 0,
                        -1 => 1,
                        _ => 2,
                    }
                };
                let (su, sv) = (si(m.cfl_alpha[0]), si(m.cfl_alpha[1]));
                let js = su * 3 + sv - 1;
                w.symbol(js, &mut cdf.cfl_sign, 8);
                if su != 0 {
                    w.symbol((m.cfl_alpha[0].unsigned_abs() - 1) as usize, &mut cdf.cfl_alpha[js + 1 - 3], 16);
                }
                if sv != 0 {
                    w.symbol((m.cfl_alpha[1].unsigned_abs() - 1) as usize, &mut cdf.cfl_alpha[sv * 3 + su - 3], 16);
                }
            } else if use_angle && is_directional(m.uv_mode) {
                w.symbol((m.angle_uv + 3) as usize, &mut cdf.angle_delta[(m.uv_mode - V_PRED) as usize], 7);
            }
        }
        if m.y_mode == DC_PRED && filter_intra_allowed(bsize) {
            w.symbol(m.filter_intra.is_some() as usize, &mut cdf.filter_intra[bsize as usize], 2);
            if let Some(f) = m.filter_intra {
                w.symbol(f as usize, &mut cdf.filter_intra_mode, 5);
            }
        }
        // transform size
        if bsize != BlockSize::B4x4 && !self.lossless {
            let max = bsize.max_tx();
            let depth = tx_depth(max, b.tx_size);
            let cat = tx_size_cat(bsize);
            let tctx = self.tx_size_ctx(bsize, mi_row, mi_col);
            w.symbol(depth, &mut cdf.tx_size[cat][tctx], max_tx_depth(bsize) + 1);
        }
        let nplanes = if has_chroma { 3 } else { 1 };
        if b.skip {
            for p in 0..nplanes {
                let (sx, sy) = if p == 0 { (0, 0) } else { (self.ssx, self.ssy) };
                let pbs = bsize.subsampled(sx, sy);
                let x4 = mi_col >> sx;
                let y4 = mi_row >> sy;
                ctx.above_ent[p][x4..x4 + pbs.w4()].fill(0);
                ctx.left_ent[p][y4..y4 + pbs.h4()].fill(0);
            }
            return;
        }
        let mut ti = b.txb_start as usize;
        let intra_dir = match m.filter_intra {
            Some(f) => FILTER_INTRA_MODE_TO_DIR[f as usize],
            None => m.y_mode,
        };
        for p in 0..nplanes {
            let (sx, sy) = if p == 0 { (0, 0) } else { (self.ssx, self.ssy) };
            let pbs = bsize.subsampled(sx, sy);
            let tx = if p == 0 { b.tx_size } else { uv_tx_size(bsize, self.ssx, self.ssy) };
            let tx = if self.lossless { TxSize::T4x4 } else { tx };
            let x0 = (mi_col >> sx) * 4;
            let y0 = (mi_row >> sy) * 4;
            let pw = (self.mi_cols * 4) >> sx;
            let ph = (self.mi_rows * 4) >> sy;
            let mut ro = 0;
            while ro < pbs.height() && y0 + ro < ph {
                let mut co = 0;
                while co < pbs.width() && x0 + co < pw {
                    let t = dec.txbs[ti];
                    ti += 1;
                    let x4 = (x0 + co) >> 2;
                    let y4 = (y0 + ro) >> 2;
                    let (skip_ctx, dc_ctx) = coef::txb_ctx(
                        p,
                        pbs,
                        tx,
                        &ctx.above_ent[p][x4..x4 + tx.w4()],
                        &ctx.left_ent[p][y4..y4 + tx.h4()],
                    );
                    let info = TxbInfo::new(tx, t.tx_type);
                    let sc = scan(tx.coded(), info.tx_class);
                    let n = info.width * info.height;
                    st.q[..n].fill(0);
                    let eob = t.eob as usize;
                    for (i, &rc) in sc[..eob].iter().enumerate() {
                        st.q[rc as usize] = dec.coefs[t.coef_start as usize + i];
                    }
                    let lossless = self.lossless;
                    let tt = t.tx_type;
                    let (coef_cdf, tx_cdf) = split_cdf(cdf);
                    coef::write_coeffs(
                        w,
                        coef_cdf,
                        &st.q,
                        eob,
                        sc,
                        &info,
                        (p > 0) as usize,
                        skip_ctx,
                        dc_ctx,
                        |w: &mut W| {
                            if p == 0 && !lossless {
                                let set = intra_tx_set(tx, false);
                                let ns = NUM_EXT_TX_SET[set];
                                if ns > 1 {
                                    w.symbol(
                                        EXT_TX_IND[set][tt as usize] as usize,
                                        &mut tx_cdf[intra_tx_set_index(set)][tx.sqr() as usize][intra_dir as usize],
                                        ns,
                                    );
                                }
                            }
                        },
                        &mut st.levels,
                    );
                    let cul = coef::txb_entropy_ctx(&st.q, sc, eob);
                    let max_x4 = pw >> 2;
                    let max_y4 = ph >> 2;
                    for i in 0..tx.w4() {
                        ctx.above_ent[p][x4 + i] = if x4 + i < max_x4 { cul } else { 0 };
                    }
                    for i in 0..tx.h4() {
                        ctx.left_ent[p][y4 + i] = if y4 + i < max_y4 { cul } else { 0 };
                    }
                    co += tx.width();
                }
                ro += tx.height();
            }
        }
        debug_assert_eq!(ti, (b.txb_start + b.txb_count) as usize);
    }
}

/// Borrow the coefficient CDFs and the transform type CDFs separately.
fn split_cdf(cdf: &mut CdfContext) -> (&mut crate::cdf::CoefCdfs, &mut [[[[u16; 17]; 13]; 4]; 3]) {
    (&mut cdf.coef, &mut cdf.intra_ext_tx)
}
