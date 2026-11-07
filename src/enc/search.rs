//! Rate-distortion search: partitions, luma / chroma modes, transform sizes.

use super::block::{BlockPos, Modes, PlaneBound, PlaneRd, Scratch, TxSearch, uv_tx_size};
use super::{BlockDec, Decisions, Encoder, MiInfo};
use crate::cdf::cost_symbol;
use crate::coef::rdcost;
use crate::consts::*;
use crate::predict::{self, EdgeAvail, IntraParams};
use crate::quant::PlaneQuant;

pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static T: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
    pub fn add(i: usize, t: std::time::Instant) {
        T[i].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    pub fn report() {
        if std::env::var("AV1_PROF").is_ok() {
            let names = ["model", "luma_rd", "txsize", "luma_final", "chroma", "calls", "lf", "cdef"];
            for (i, n) in names.iter().enumerate() {
                eprintln!("{n}: {:.3}", T[i].load(Ordering::Relaxed) as f64 / 1e9);
            }
        }
    }
}

const ANGLE_SEARCH_MODES: usize = 2;
/// Rectangles are tried only when SPLIT is within this factor of NONE.
/// Largest-side sizes (log2) whose transform is never split in the transform
/// size search. Splitting 32-point and 8-point transforms was chosen too often:
/// measured on Kodak (24 images), skipping both gives smaller files at equal
/// PSNR / SSIMULACRA2 and saves one candidate evaluation. `AV1_TXSPLIT_ALL=1`
/// restores the old behaviour; `AV1_NOSPLIT_MASK=<bits>` overrides the sizes.
const NO_TX_SPLIT_DEFAULT: u32 = (1 << 3) | (1 << 5);

fn tx_split_allowed(max_tx: TxSize) -> bool {
    static MASK: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let mask = *MASK.get_or_init(|| {
        if dbg("AV1_TXSPLIT_ALL") {
            0
        } else {
            std::env::var("AV1_NOSPLIT_MASK").ok().and_then(|v| v.parse().ok()).unwrap_or(NO_TX_SPLIT_DEFAULT)
        }
    });
    (mask >> max_tx.width().max(max_tx.height()).trailing_zeros()) & 1 == 0
}

static RECT_PRUNE_ENV: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
#[allow(non_snake_case)]
fn RECT_PRUNE_F() -> f64 {
    *RECT_PRUNE_ENV.get_or_init(|| std::env::var("AV1_RECT_PRUNE").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0))
}

static GATE_ENV: std::sync::OnceLock<(Option<f64>, Option<u8>)> = std::sync::OnceLock::new();
/// Flat-block gate: (K, largest block width in pixels that is gated).
/// Square blocks whose source is flat (sum of squared deviations per 8x8 area
/// below K * rdmult / 128) are coded as one block without trying SPLIT or
/// rectangles. The defaults follow `speed` (5 is the libaom-equivalent
/// setting; 6 and 7 trade a little compression for speed); the developer
/// variables `AV1_GATE_K` (0 disables) and `AV1_GATE_MAX` override them.
fn gate_params(speed: u8) -> (f64, u8) {
    let (k, m) = match speed {
        ..=5 => (16.0, 8),
        6 => (32.0, 8),
        _ => (32.0, 16),
    };
    let (ek, em) = *GATE_ENV.get_or_init(|| {
        (
            std::env::var("AV1_GATE_K").ok().and_then(|v| v.parse().ok()),
            std::env::var("AV1_GATE_MAX").ok().and_then(|v| v.parse().ok()),
        )
    });
    (ek.unwrap_or(k), em.unwrap_or(m))
}

fn dbg(name: &str) -> bool {
    crate::debug_flag(name)
}

/// Mode-independent inputs of the model SATD of one plane of a block.
struct ModelEnv {
    plane: usize,
    x: usize,
    y: usize,
    smooth_neighbor: bool,
    n_top: usize,
    n_left: usize,
    n_topright: usize,
    n_bottomleft: usize,
}

pub struct SearchState {
    pub sb_q: u8,
    pub last_sb_q: u8,
    pub quant: [PlaneQuant; 3],
    pub rdmult: i64,
    pub scratch: Scratch,
    pool: Vec<Decisions>,
    // Only snapshots no longer held by a recursion frame are available here.
    snapshots: Vec<Snapshot>,
    cfl_ac: Vec<i16>,
    pred: Vec<u8>,
    luma: LumaSnapshot,
    use_rd_budget: bool,
    use_tx_cache: bool,
}

impl SearchState {
    pub fn new() -> SearchState {
        SearchState {
            sb_q: 0,
            last_sb_q: 0,
            quant: [PlaneQuant::new(0, 0, 0, 15, false); 3],
            rdmult: 1,
            scratch: Scratch::new(),
            pool: Vec::new(),
            snapshots: Vec::new(),
            cfl_ac: vec![0; 32 * 32],
            pred: vec![0; 64 * 64],
            luma: LumaSnapshot { rec: vec![0; 64 * 64], above: [0; 16], left: [0; 16] },
            use_rd_budget: !dbg("AV1_NO_RD_BUDGET"),
            use_tx_cache: !dbg("AV1_NO_TX_CACHE"),
        }
    }
    fn take(&mut self) -> Decisions {
        let mut d = self.pool.pop().unwrap_or_default();
        d.clear();
        d
    }
    fn give(&mut self, d: Decisions) {
        self.pool.push(d);
    }
}

/// Only luma reconstruction and coefficient contexts change during tx-size
/// search. Reuse this storage across blocks; mode info and chroma need no copy.
struct LumaSnapshot {
    rec: Vec<u8>,
    above: [u8; 16],
    left: [u8; 16],
}

impl LumaSnapshot {
    fn save(&mut self, e: &Encoder, bp: &BlockPos) {
        let (x, y, w, h) = plane_region(e, 0, bp.bsize, bp.mi_row, bp.mi_col);
        let rec = &e.rec[0];
        for r in 0..h {
            let o = (y + r) * rec.stride + x;
            self.rec[r * w..(r + 1) * w].copy_from_slice(&rec.data[o..o + w]);
        }
        self.above[..w / 4].copy_from_slice(&e.ctx.above_ent[0][x / 4..(x + w) / 4]);
        self.left[..h / 4].copy_from_slice(&e.ctx.left_ent[0][y / 4..(y + h) / 4]);
    }

    fn restore(&self, e: &mut Encoder, bp: &BlockPos) {
        let (x, y, w, h) = plane_region(e, 0, bp.bsize, bp.mi_row, bp.mi_col);
        let rec = &mut e.rec[0];
        for r in 0..h {
            let o = (y + r) * rec.stride + x;
            rec.data[o..o + w].copy_from_slice(&self.rec[r * w..(r + 1) * w]);
        }
        e.ctx.above_ent[0][x / 4..(x + w) / 4].copy_from_slice(&self.above[..w / 4]);
        e.ctx.left_ent[0][y / 4..(y + h) / 4].copy_from_slice(&self.left[..h / 4]);
    }
}

/// Saved encoder state of a block region (for trying alternatives).
struct Snapshot {
    rec: [Vec<u8>; 3],
    above: [Vec<u8>; 3],
    left: [Vec<u8>; 3],
    mi: Vec<MiInfo>,
}

/// Plane regions covered by a block: (x, y, w, h) in pixels, and the entropy
/// context ranges (x4, w4, y4, h4).
fn plane_region(
    e: &Encoder,
    plane: usize,
    bsize: BlockSize,
    mi_row: usize,
    mi_col: usize,
) -> (usize, usize, usize, usize) {
    let (sx, sy) = if plane == 0 { (0, 0) } else { (e.ssx, e.ssy) };
    let x = (mi_col >> sx) * 4;
    let y = (mi_row >> sy) * 4;
    let w = (bsize.width() >> sx).max(4);
    let h = (bsize.height() >> sy).max(4);
    (x, y, w, h)
}

impl Encoder {
    fn snapshot(&self, bsize: BlockSize, mi_row: usize, mi_col: usize, snap: Option<Snapshot>) -> Snapshot {
        let mut sn = snap.unwrap_or(Snapshot {
            rec: [Vec::new(), Vec::new(), Vec::new()],
            above: [Vec::new(), Vec::new(), Vec::new()],
            left: [Vec::new(), Vec::new(), Vec::new()],
            mi: Vec::new(),
        });
        for p in 0..self.num_planes {
            let (x, y, w, h) = plane_region(self, p, bsize, mi_row, mi_col);
            let rec = &self.rec[p];
            sn.rec[p].clear();
            for r in 0..h {
                let o = (y + r) * rec.stride + x;
                sn.rec[p].extend_from_slice(&rec.data[o..o + w]);
            }
            sn.above[p].clear();
            sn.above[p].extend_from_slice(&self.ctx.above_ent[p][x / 4..x / 4 + w / 4]);
            sn.left[p].clear();
            sn.left[p].extend_from_slice(&self.ctx.left_ent[p][y / 4..y / 4 + h / 4]);
        }
        sn.mi.clear();
        let rows = bsize.h4().min(self.mi_rows - mi_row);
        let cols = bsize.w4().min(self.mi_cols - mi_col);
        for r in 0..rows {
            let o = (mi_row + r) * self.mi_cols + mi_col;
            sn.mi.extend_from_slice(&self.mi[o..o + cols]);
        }
        sn
    }

    fn restore(&mut self, bsize: BlockSize, mi_row: usize, mi_col: usize, sn: &Snapshot) {
        for p in 0..self.num_planes {
            let (x, y, w, h) = plane_region(self, p, bsize, mi_row, mi_col);
            let rec = &mut self.rec[p];
            for r in 0..h {
                let o = (y + r) * rec.stride + x;
                rec.data[o..o + w].copy_from_slice(&sn.rec[p][r * w..r * w + w]);
            }
            self.ctx.above_ent[p][x / 4..x / 4 + w / 4].copy_from_slice(&sn.above[p]);
            self.ctx.left_ent[p][y / 4..y / 4 + h / 4].copy_from_slice(&sn.left[p]);
        }
        let rows = bsize.h4().min(self.mi_rows - mi_row);
        let cols = bsize.w4().min(self.mi_cols - mi_col);
        for r in 0..rows {
            let o = (mi_row + r) * self.mi_cols + mi_col;
            self.mi[o..o + cols].copy_from_slice(&sn.mi[r * cols..r * cols + cols]);
        }
    }

    pub fn partition_ctx(&self, bsize: BlockSize, mi_row: usize, mi_col: usize) -> usize {
        let bsl = bsize.w4_log2();
        let above = self.avail_up(mi_row) && self.mi(mi_row - 1, mi_col).bsize.w4_log2() < bsl;
        let left = self.avail_left(mi_col) && self.mi(mi_row, mi_col - 1).bsize.h4_log2() < bsl;
        (left as usize * 2 + above as usize) + (bsl - 1) * 4
    }

    /// Rate of signalling partition `p` (0 when implied).
    pub fn partition_rate(&self, bsize: BlockSize, mi_row: usize, mi_col: usize, p: Partition) -> u32 {
        let hbs = bsize.w4() / 2;
        let has_rows = mi_row + hbs < self.mi_rows;
        let has_cols = mi_col + hbs < self.mi_cols;
        let ctx = self.partition_ctx(bsize, mi_row, mi_col);
        if has_rows && has_cols {
            return self.mode_costs.partition[ctx][p as usize];
        }
        if !has_rows && !has_cols {
            return 0;
        }
        let psum = gathered_prob(&self.cdf.partition[ctx], !has_rows);
        if p == Partition::Split { cost_symbol(psum) } else { cost_symbol(32768 - psum) }
    }

    pub(super) fn rd_pick_partition(
        &mut self,
        s: &mut SearchState,
        bsize: BlockSize,
        mi_row: usize,
        mi_col: usize,
        out: &mut Decisions,
        budget: i64,
    ) -> i64 {
        // A finite budget is an exclusive upper bound. If no candidate beats
        // it, return MAX with no decisions and restore the entry state.
        let budget = if s.use_rd_budget { budget } else { i64::MAX };
        let hbs = bsize.w4() / 2;
        let has_rows = mi_row + hbs < self.mi_rows;
        let has_cols = mi_col + hbs < self.mi_cols;
        // Order: NONE, SPLIT, then the rectangles (which may be pruned).
        let mut cands: [Option<Partition>; 4] = [None; 4];
        if has_rows && has_cols {
            cands[0] = Some(Partition::None);
        }
        cands[1] = Some(Partition::Split);
        // libaom's use_square_partition_only_threshold (speed >= 2): above it,
        // rectangles are only used where the frame edge forces them.
        let sq_only = if self.mi_rows.min(self.mi_cols) * 4 >= 720 { BlockSize::B64x64 } else { BlockSize::B32x32 };
        let rect_ok = !dbg("AV1_NO_RECT") && (bsize <= sq_only || !(has_rows && has_cols));
        if rect_ok {
            if has_cols {
                cands[2] = Some(Partition::Horz);
            }
            if has_rows {
                cands[3] = Some(Partition::Vert);
            }
        }
        let (gk, gmax) = gate_params(self.cfg.speed);
        if gk > 0.0 && has_rows && has_cols && bsize.w4() * 4 <= gmax as usize && bsize.w4() == bsize.h4() {
            let n = bsize.w4() * 4;
            let dev = crate::enc::block_variance(&self.src[0], mi_col * 4, mi_row * 4, n);
            // Normalise to an 8x8 area so one K works for every gated size.
            let per8 = (dev as f64) * 64.0 / (n * n) as f64;
            if per8 * 128.0 < gk * (s.rdmult as f64) {
                cands = [Some(Partition::None), None, None, None];
            }
        }
        if !has_rows && !has_cols {
            cands = [None, Some(Partition::Split), None, None];
        } else if !has_rows {
            cands[3] = None;
        } else if !has_cols {
            cands[2] = None;
        }
        let mut rd_none = i64::MAX;
        let mut rd_split = i64::MAX;
        let start = self.snapshot(bsize, mi_row, mi_col, s.snapshots.pop());
        let mut best_after: Option<Snapshot> = None;
        let mut best_rd = budget;
        let mut best_dec = s.take();
        let has_rect = cands[2].is_some() || cands[3].is_some();
        let mut first = true;
        for p in cands.into_iter().flatten() {
            if matches!(p, Partition::Horz | Partition::Vert)
                && rd_none != i64::MAX
                && rd_split != i64::MAX
                && (rd_split as f64) > (rd_none as f64) * RECT_PRUNE_F()
            {
                continue;
            }
            if !first {
                self.restore(bsize, mi_row, mi_col, &start);
            }
            first = false;
            let mut cand = s.take();
            cand.parts.push(p);
            let prate = self.partition_rate(bsize, mi_row, mi_col, p);
            let mut rd = rdcost(s.rdmult, prate as i64, 0);
            let mut complete = true;
            // SPLIT also controls the rectangle heuristic. Until its lower
            // bound proves that rectangles are pruned, keep computing its
            // exact RD rather than changing which rectangles get searched.
            let needs_split_rd = p == Partition::Split && has_rect && rd_none != i64::MAX;
            // Remaining block/partition costs are non-negative. Equality
            // cannot win either: candidates replace the incumbent only on <.
            let limit = if s.use_rd_budget { best_rd } else { i64::MAX };
            match p {
                Partition::None => {
                    rd += self.pick_block(s, bsize, mi_row, mi_col, p, &mut cand);
                }
                Partition::Horz => {
                    let sub = bsize.subsize(p).unwrap();
                    rd += self.pick_block(s, sub, mi_row, mi_col, p, &mut cand);
                    if mi_row + hbs < self.mi_rows {
                        if rd >= limit {
                            complete = false;
                        } else {
                            rd += self.pick_block(s, sub, mi_row + hbs, mi_col, p, &mut cand);
                        }
                    }
                }
                Partition::Vert => {
                    let sub = bsize.subsize(p).unwrap();
                    rd += self.pick_block(s, sub, mi_row, mi_col, p, &mut cand);
                    if mi_col + hbs < self.mi_cols {
                        if rd >= limit {
                            complete = false;
                        } else {
                            rd += self.pick_block(s, sub, mi_row, mi_col + hbs, p, &mut cand);
                        }
                    }
                }
                _ => {
                    let sub = bsize.subsize(p).unwrap();
                    for i in 0..4 {
                        let r = mi_row + (i >> 1) * hbs;
                        let c = mi_col + (i & 1) * hbs;
                        if r >= self.mi_rows || c >= self.mi_cols {
                            continue;
                        }
                        if rd >= limit && (!needs_split_rd || (rd as f64) > (rd_none as f64) * RECT_PRUNE_F()) {
                            complete = false;
                            break;
                        }
                        if sub == BlockSize::B4x4 {
                            rd += self.pick_block(s, sub, r, c, p, &mut cand);
                        } else {
                            let mut child = s.take();
                            // A failed child supplies no exact RD, so only
                            // pass a finite budget when no rectangle gate
                            // depends on this SPLIT's exact cost.
                            let child_budget = if needs_split_rd { i64::MAX } else { limit - rd };
                            let child_rd = self.rd_pick_partition(s, sub, r, c, &mut child, child_budget);
                            if child_rd == i64::MAX {
                                // Never add the failure sentinel to an RD cost.
                                complete = false;
                                s.give(child);
                                break;
                            }
                            rd += child_rd;
                            cand.append(&child);
                            s.give(child);
                        }
                    }
                }
            }
            match p {
                Partition::None => rd_none = rd,
                Partition::Split => rd_split = rd,
                _ => {}
            }
            if complete && rd < best_rd {
                best_rd = rd;
                best_after =
                    Some(self.snapshot(bsize, mi_row, mi_col, best_after.take().or_else(|| s.snapshots.pop())));
                std::mem::swap(&mut best_dec, &mut cand);
            }
            s.give(cand);
        }
        let result = if let Some(sn) = best_after {
            self.restore(bsize, mi_row, mi_col, &sn);
            out.append(&best_dec);
            s.snapshots.push(sn);
            best_rd
        } else {
            self.restore(bsize, mi_row, mi_col, &start);
            i64::MAX
        };
        s.snapshots.push(start);
        s.give(best_dec);
        result
    }

    fn y_mode_ctx(&self, mi_row: usize, mi_col: usize) -> (usize, usize) {
        let a = if self.avail_up(mi_row) { INTRA_MODE_CONTEXT[self.mi(mi_row - 1, mi_col).y_mode as usize] } else { 0 };
        let l =
            if self.avail_left(mi_col) { INTRA_MODE_CONTEXT[self.mi(mi_row, mi_col - 1).y_mode as usize] } else { 0 };
        (a, l)
    }

    fn skip_ctx(&self, mi_row: usize, mi_col: usize) -> usize {
        let a = self.avail_up(mi_row) && self.mi(mi_row - 1, mi_col).skip;
        let l = self.avail_left(mi_col) && self.mi(mi_row, mi_col - 1).skip;
        a as usize + l as usize
    }

    pub fn tx_size_ctx(&self, bsize: BlockSize, mi_row: usize, mi_col: usize) -> usize {
        let max = bsize.max_tx();
        let has_above = self.avail_up(mi_row);
        let has_left = self.avail_left(mi_col);
        let above = has_above && self.mi(mi_row - 1, mi_col).tx_size.width() >= max.width();
        let left = has_left && self.mi(mi_row, mi_col - 1).tx_size.height() >= max.height();
        match (has_above, has_left) {
            (true, true) => above as usize + left as usize,
            (true, false) => above as usize,
            (false, true) => left as usize,
            _ => 0,
        }
    }

    /// Rate of the transform size (depth from the largest size).
    pub fn tx_size_rate(&self, bsize: BlockSize, mi_row: usize, mi_col: usize, tx: TxSize) -> u32 {
        if bsize == BlockSize::B4x4 || self.lossless {
            return 0;
        }
        let max = bsize.max_tx();
        let cat = tx_size_cat(bsize);
        let depth = tx_depth(max, tx);
        self.mode_costs.tx_size[cat][self.tx_size_ctx(bsize, mi_row, mi_col)][depth]
    }

    fn save_ent(&self, plane: usize, bsize: BlockSize, mi_row: usize, mi_col: usize, out: &mut (Vec<u8>, Vec<u8>)) {
        let (x, y, w, h) = plane_region(self, plane, bsize, mi_row, mi_col);
        out.0.clear();
        out.0.extend_from_slice(&self.ctx.above_ent[plane][x / 4..x / 4 + w / 4]);
        out.1.clear();
        out.1.extend_from_slice(&self.ctx.left_ent[plane][y / 4..y / 4 + h / 4]);
    }

    fn load_ent(&mut self, plane: usize, bsize: BlockSize, mi_row: usize, mi_col: usize, saved: &(Vec<u8>, Vec<u8>)) {
        let (x, y, w, h) = plane_region(self, plane, bsize, mi_row, mi_col);
        self.ctx.above_ent[plane][x / 4..x / 4 + w / 4].copy_from_slice(&saved.0);
        self.ctx.left_ent[plane][y / 4..y / 4 + h / 4].copy_from_slice(&saved.1);
    }

    /// Predict the whole block (one transform of size `tx` at the block
    /// origin) and return the SATD of the residual — the model cost used
    /// to rank intra modes.
    #[allow(clippy::too_many_arguments)]
    /// Block-level inputs of `model_satd` that do not depend on the mode.
    fn model_env(&self, plane: usize, bp: &BlockPos, tx: TxSize) -> ModelEnv {
        let (sx, sy) = if plane == 0 { (0, 0) } else { (self.ssx, self.ssy) };
        // Any directional mode with the matching angle range sees the same availability.
        let probe = |mode: u8| self.block_edge_avail(plane, bp, tx, mode, 0, false);
        let base = probe(DC_PRED);
        ModelEnv {
            plane,
            x: (bp.mi_col >> sx) * 4,
            y: (bp.mi_row >> sy) * 4,
            smooth_neighbor: self.smooth_neighbor(plane, bp),
            n_top: base.n_top,
            n_left: base.n_left,
            n_topright: probe(D45_PRED).n_topright,
            n_bottomleft: probe(D203_PRED).n_bottomleft,
        }
    }

    fn model_satd(&self, env: &ModelEnv, bp: &BlockPos, modes: &Modes, tx: TxSize, pred: &mut [u8]) -> u64 {
        let plane = env.plane;
        let (x, y) = (env.x, env.y);
        let mode = if plane == 0 { modes.y_mode } else { modes.uv_mode };
        let angle = if plane == 0 { modes.angle_y } else { modes.angle_uv } as i32;
        let dr = is_directional(mode);
        let ip = IntraParams {
            mode,
            angle_delta: angle,
            filter_intra: if plane == 0 { modes.filter_intra } else { None },
            edge_filter: true,
            smooth_neighbor: dr && env.smooth_neighbor,
        };
        let p_angle = MODE_TO_ANGLE[mode as usize] + angle * 3;
        let plain = ip.filter_intra.is_none() && dr;
        let av = EdgeAvail {
            n_top: env.n_top,
            n_left: env.n_left,
            n_topright: if plain && p_angle < 90 { env.n_topright } else { 0 },
            n_bottomleft: if plain && p_angle > 180 { env.n_bottomleft } else { 0 },
        };
        debug_assert!(av == self.block_edge_avail(plane, bp, tx, mode, angle, ip.filter_intra.is_some()));
        let rec = &self.rec[plane];
        predict::predict(&rec.data, rec.stride, x, y, tx, &ip, av, pred, 64);
        let src = &self.src[plane];
        satd(&src.data[y * src.stride + x..], src.stride, pred, 64, tx.width(), tx.height())
    }

    /// Edge availability for a transform at the block origin.
    fn block_edge_avail(&self, plane: usize, bp: &BlockPos, tx: TxSize, mode: u8, angle: i32, fi: bool) -> EdgeAvail {
        let (sx, sy) = if plane == 0 { (0, 0) } else { (self.ssx, self.ssy) };
        let bsize = bp.bsize;
        let pbs = bsize.subsampled(sx, sy);
        let (bw4, bh4) = (bsize.w4() as isize, bsize.h4() as isize);
        let mb_to_right = ((self.mi_cols as isize - bw4 - bp.mi_col as isize) * 4) >> sx;
        let mb_to_bottom = ((self.mi_rows as isize - bh4 - bp.mi_row as isize) * 4) >> sy;
        let (txw, txh) = (tx.width() as isize, tx.height() as isize);
        let xr = mb_to_right + (pbs.width().max(4) as isize) - txw;
        let yd = mb_to_bottom + (pbs.height().max(4) as isize) - txh;
        let (up, left) = if plane == 0 {
            (self.avail_up(bp.mi_row), self.avail_left(bp.mi_col))
        } else {
            let up = if sy == 1 && bsize.h4() < 2 {
                bp.mi_row >= 1 && bp.mi_row - 1 > self.tile.mi_row_start
            } else {
                self.avail_up(bp.mi_row)
            };
            let left = if sx == 1 && bsize.w4() < 2 {
                bp.mi_col >= 1 && bp.mi_col - 1 > self.tile.mi_col_start
            } else {
                self.avail_left(bp.mi_col)
            };
            (up, left)
        };
        let mut av = EdgeAvail {
            n_top: if up { txw.min(xr + txw).max(0) as usize } else { 0 },
            n_topright: 0,
            n_left: if left { txh.min(yd + txh).max(0) as usize } else { 0 },
            n_bottomleft: 0,
        };
        let p_angle = MODE_TO_ANGLE[mode as usize] + angle * 3;
        let tr_bsize = if plane > 0 { crate::avail::scale_chroma_bsize(bsize, sx, sy) } else { bsize };
        if !fi && is_directional(mode) && p_angle < 90 {
            let right_available = bp.mi_col + (tx.w4() << sx) < self.tile.mi_col_end;
            if crate::avail::has_top_right(
                tr_bsize,
                bp.mi_row,
                bp.mi_col,
                up,
                right_available,
                bp.partition,
                tx,
                0,
                0,
                sx,
                sy,
            ) {
                av.n_topright = txw.min(xr).max(0) as usize;
            }
        }
        if !fi && is_directional(mode) && p_angle > 180 {
            let bottom_available = yd > 0 && bp.mi_row + (tx.h4() << sy) < self.tile.mi_row_end;
            if crate::avail::has_bottom_left(
                tr_bsize,
                bp.mi_row,
                bp.mi_col,
                bottom_available,
                left,
                bp.partition,
                tx,
                0,
                0,
                sx,
                sy,
            ) {
                av.n_bottomleft = txh.min(yd).max(0) as usize;
            }
        }
        av
    }

    /// Mode rate of the luma part.
    fn luma_mode_rate(&self, bsize: BlockSize, ctx: (usize, usize), m: &Modes) -> u32 {
        let mut r = self.mode_costs.kf_y[ctx.0][ctx.1][m.y_mode as usize];
        if bsize >= BlockSize::B8x8 && is_directional(m.y_mode) {
            r += self.mode_costs.angle[(m.y_mode - V_PRED) as usize][(m.angle_y + 3) as usize];
        }
        if m.y_mode == DC_PRED && filter_intra_allowed(bsize) {
            let b = &self.mode_costs.fi_use[bsize as usize];
            match m.filter_intra {
                Some(f) => r += b[1] + self.mode_costs.fi_mode[f as usize],
                None => r += b[0],
            }
        }
        r
    }

    fn uv_mode_rate(&self, bsize: BlockSize, m: &Modes) -> u32 {
        let cfl_allowed = cfl_allowed(bsize, self.lossless, self.ssx, self.ssy);
        let mut r = self.mode_costs.uv[cfl_allowed as usize][m.y_mode as usize][m.uv_mode as usize];
        if m.uv_mode == UV_CFL_PRED {
            let su = sign_idx(m.cfl_alpha[0]);
            let sv = sign_idx(m.cfl_alpha[1]);
            let js = su * 3 + sv - 1;
            r += self.mode_costs.cfl_sign[js];
            if su != 0 {
                r += self.mode_costs.cfl_alpha[js + 1 - 3][(m.cfl_alpha[0].unsigned_abs() - 1) as usize];
            }
            if sv != 0 {
                r += self.mode_costs.cfl_alpha[sv * 3 + su - 3][(m.cfl_alpha[1].unsigned_abs() - 1) as usize];
            }
        } else if bsize >= BlockSize::B8x8 && is_directional(m.uv_mode) {
            r += self.mode_costs.angle[(m.uv_mode - V_PRED) as usize][(m.angle_uv + 3) as usize];
        }
        r
    }

    /// Choose and code one block. Returns its RD cost and appends it to `out`.
    fn pick_block(
        &mut self,
        s: &mut SearchState,
        bsize: BlockSize,
        mi_row: usize,
        mi_col: usize,
        partition: Partition,
        out: &mut Decisions,
    ) -> i64 {
        let rdmult = self.block_rdmult(s.rdmult, bsize, mi_row, mi_col);
        let quant = s.quant;
        let max_tx = if self.lossless { TxSize::T4x4 } else { bsize.max_tx() };
        let mut bp = BlockPos { bsize, mi_row, mi_col, partition, luma_tx: max_tx };
        let has_chroma = self.has_chroma(bsize, mi_row, mi_col);
        let ctx = self.y_mode_ctx(mi_row, mi_col);
        let mut saved = [(Vec::new(), Vec::new()), (Vec::new(), Vec::new()), (Vec::new(), Vec::new())];
        self.save_ent(0, bsize, mi_row, mi_col, &mut saved[0]);
        if has_chroma {
            self.save_ent(1, bsize, mi_row, mi_col, &mut saved[1]);
            self.save_ent(2, bsize, mi_row, mi_col, &mut saved[2]);
        }
        let lambda_sad = ((rdmult as f64) / (1u64 << 20) as f64).sqrt();
        let use_angle = bsize >= BlockSize::B8x8;

        // ---- luma: model ranking
        prof::T[5].fetch_add(1_000_000_000, std::sync::atomic::Ordering::Relaxed);
        let t0 = std::time::Instant::now();
        let mut ranked: Vec<(f64, Modes)> = Vec::with_capacity(80);
        let model_tx = if self.lossless { TxSize::T4x4 } else { max_tx };
        let env = self.model_env(0, &bp, model_tx);
        let push = |e: &Encoder, m: Modes, pred: &mut [u8]| {
            let satd = e.model_satd(&env, &bp, &m, model_tx, pred) as f64;
            let rate = e.luma_mode_rate(bsize, ctx, &m) as f64 / 512.0;
            (satd + lambda_sad * rate * 4.0, m)
        };
        for mode in 0..INTRA_MODES as u8 {
            if dbg("AV1_DC_ONLY") && mode != DC_PRED {
                continue;
            }
            let m = Modes { y_mode: mode, ..Modes::default() };
            ranked.push(push(self, m, &mut s.pred));
        }
        if use_angle && !dbg("AV1_NO_ANGLE") {
            // Angle deltas only around the two best directional modes.
            let mut dirs: Vec<(f64, u8)> =
                ranked.iter().filter(|r| is_directional(r.1.y_mode)).map(|r| (r.0, r.1.y_mode)).collect();
            sort_front(&mut dirs, ANGLE_SEARCH_MODES);
            for &(_, mode) in dirs.iter().take(ANGLE_SEARCH_MODES) {
                for a in [-3i8, -2, -1, 1, 2, 3] {
                    let m = Modes { y_mode: mode, angle_y: a, ..Modes::default() };
                    ranked.push(push(self, m, &mut s.pred));
                }
            }
        }
        if filter_intra_allowed(bsize) && !self.lossless && !dbg("AV1_NO_FI") {
            for f in 0..5u8 {
                let m = Modes { y_mode: DC_PRED, filter_intra: Some(f), ..Modes::default() };
                ranked.push(push(self, m, &mut s.pred));
            }
        }
        let top_k = 3.min(ranked.len());
        sort_front(&mut ranked, top_k);
        prof::add(0, t0);
        let t0 = std::time::Instant::now();

        // ---- luma: RD of the best candidates at the largest transform
        let mut best_y = (i64::MAX, Modes::default());
        // The winner's transform search at the largest size evaluates the two transform types
        // tried here again (trellis is off by default): record them and reuse the results.
        let luma_replay = !self.lossless && !dbg("AV1_TRELLIS") && !dbg("AV1_NO_REPLAY");
        s.scratch.best_rec[0].n = 0;
        for &(_, m) in &ranked[..top_k] {
            self.load_ent(0, bsize, mi_row, mi_col, &saved[0]);
            let ts = if dbg("AV1_FULL_TX_EVAL") {
                TxSearch::Search { trellis: false }
            } else {
                // Fast: the transform type implied by the mode (full type search
                // is done for the winner).
                let dir = m.filter_intra.map(|f| FILTER_INTRA_MODE_TO_DIR[f as usize]).unwrap_or(m.y_mode);
                let t = MODE_TO_TXFM[dir as usize];
                TxSearch::Pair(if tx_in_set(intra_tx_set(max_tx, false), t) { t } else { DCT_DCT })
            };
            let mode_rate = self.luma_mode_rate(bsize, ctx, &m) as i64;
            let bound = s.use_rd_budget.then_some(PlaneBound { rd: best_y.0, rate: mode_rate, dist: 0 });
            s.scratch.rank_only = true;
            s.scratch.rec_plane = luma_replay.then_some(0);
            let r = self.try_code_plane(0, &bp, &m, max_tx, ts, &quant[0], rdmult, &mut s.scratch, None, bound);
            s.scratch.rank_only = false;
            s.scratch.rec_plane = None;
            let Some(r) = r else {
                continue;
            };
            let rate = r.rate + mode_rate;
            let rd = rdcost(rdmult, rate, r.dist);
            if rd < best_y.0 {
                best_y = (rd, m);
                if luma_replay {
                    s.scratch.best_rec[0].copy_from(&s.scratch.rec[0]);
                }
            }
        }
        let ym = best_y.1;
        prof::add(1, t0);
        let t0 = std::time::Instant::now();

        // ---- luma: transform size of the winner (largest and one split)
        let mut tx_cands = [max_tx, max_tx];
        let mut ntx = 1;
        // A 64x64 transform is not split into 32x32: measured on several images,
        // always keeping it gives 3-4% smaller files at equal or better
        // PSNR/SSIM (the split was chosen too often; cause not yet identified)
        // and saves one full candidate evaluation per 64x64 block.
        if !self.lossless
            && bsize != BlockSize::B4x4
            && max_tx != TxSize::T4x4
            && max_tx != TxSize::T64x64
            && !dbg("AV1_NO_TXSPLIT")
            && tx_split_allowed(max_tx)
        {
            tx_cands[1] = max_tx.sub();
            ntx = 2;
        }
        let mut best_tx = (i64::MAX, max_tx, PlaneRd::default(), 0usize);
        let mut dec = s.take();
        let mut cand = s.take();
        for (i, &tx) in tx_cands[..ntx].iter().enumerate() {
            self.load_ent(0, bsize, mi_row, mi_col, &saved[0]);
            cand.clear();
            bp.luma_tx = tx;
            let tx_rate = self.tx_size_rate(bsize, mi_row, mi_col, tx) as i64;
            let bound = s.use_rd_budget.then_some(PlaneBound { rd: best_tx.0, rate: tx_rate, dist: 0 });
            s.scratch.replay_plane = (luma_replay && tx == max_tx).then_some(0);
            let r = self.try_code_plane(
                0,
                &bp,
                &ym,
                tx,
                TxSearch::Search { trellis: true },
                &quant[0],
                rdmult,
                &mut s.scratch,
                Some((&mut cand.txbs, &mut cand.coefs)),
                bound,
            );
            s.scratch.replay_plane = None;
            let Some(r) = r else {
                continue;
            };
            let rate = r.rate + tx_rate;
            let rd = rdcost(rdmult, rate, r.dist);
            if rd < best_tx.0 {
                best_tx = (rd, tx, r, i);
                // Only a winner followed by another candidate needs saving.
                if s.use_tx_cache && i + 1 < ntx {
                    s.luma.save(self, &bp);
                }
                std::mem::swap(&mut dec, &mut cand);
            }
        }
        s.give(cand);
        let tx = best_tx.1;
        prof::add(2, t0);
        let t0 = std::time::Instant::now();
        bp.luma_tx = tx;
        let yr = if best_tx.3 + 1 == ntx {
            best_tx.2
        } else if s.use_tx_cache {
            s.luma.restore(self, &bp);
            best_tx.2
        } else {
            // The winner is not the last one coded: code it again so the
            // reconstruction and contexts match.
            dec.clear();
            self.load_ent(0, bsize, mi_row, mi_col, &saved[0]);
            self.code_plane(
                0,
                &bp,
                &ym,
                tx,
                TxSearch::Search { trellis: true },
                &quant[0],
                rdmult,
                &mut s.scratch,
                Some((&mut dec.txbs, &mut dec.coefs)),
            )
        };
        let mut modes = ym;
        let mut rate =
            yr.rate + self.luma_mode_rate(bsize, ctx, &ym) as i64 + self.tx_size_rate(bsize, mi_row, mi_col, tx) as i64;
        let mut dist = yr.dist;
        let mut nonzero = yr.nonzero;
        let mut coef_rate = yr.rate;

        prof::add(3, t0);
        let t0 = std::time::Instant::now();
        // ---- chroma
        if has_chroma {
            let uvtx = if self.lossless { TxSize::T4x4 } else { uv_tx_size(bsize, self.ssx, self.ssy) };
            let mut ranked: Vec<(f64, Modes)> = Vec::with_capacity(16);
            let (env1, env2) = (self.model_env(1, &bp, uvtx), self.model_env(2, &bp, uvtx));
            for mode in 0..INTRA_MODES as u8 {
                if dbg("AV1_DC_ONLY") && mode != DC_PRED {
                    continue;
                }
                let m = Modes { uv_mode: mode, ..ym };
                let su = self.model_satd(&env1, &bp, &m, uvtx, &mut s.pred)
                    + self.model_satd(&env2, &bp, &m, uvtx, &mut s.pred);
                let rate = self.uv_mode_rate(bsize, &m) as f64 / 512.0;
                ranked.push((su as f64 + lambda_sad * rate * 4.0, m));
            }
            sort_front(&mut ranked, 3);
            let mut cands: Vec<Modes> = ranked.iter().take(3).map(|r| r.1).collect();
            if cfl_allowed(bsize, self.lossless, self.ssx, self.ssy)
                && bsize.w4() >= 2
                && bsize.h4() >= 2
                && !dbg("AV1_NO_CFL")
                && let Some(m) = self.search_cfl_alpha(s, &bp, &ym, uvtx)
            {
                cands.push(m);
            }
            let mut best_uv = (i64::MAX, ym);
            // The final pass repeats the ranking pass of the chosen candidate (trellis is off by
            // default), so its transform search is recorded and replayed instead of redone.
            let replay_ok = !self.lossless && !dbg("AV1_TRELLIS") && !dbg("AV1_NO_REPLAY");
            for r in s.scratch.best_rec.iter_mut() {
                r.n = 0;
            }
            'candidate: for m in cands {
                self.load_ent(1, bsize, mi_row, mi_col, &saved[1]);
                self.load_ent(2, bsize, mi_row, mi_col, &saved[2]);
                let mut r = self.uv_mode_rate(bsize, &m) as i64;
                let mut d = 0;
                for p in 1..3 {
                    let bound = s.use_rd_budget.then_some(PlaneBound { rd: best_uv.0, rate: r, dist: d });
                    s.scratch.rank_only = true;
                    s.scratch.rec_plane = replay_ok.then_some(p);
                    let pr = self.try_code_plane(
                        p,
                        &bp,
                        &m,
                        uvtx,
                        TxSearch::Search { trellis: false },
                        &quant[p],
                        rdmult,
                        &mut s.scratch,
                        None,
                        bound,
                    );
                    s.scratch.rank_only = false;
                    s.scratch.rec_plane = None;
                    let Some(pr) = pr else {
                        continue 'candidate;
                    };
                    r += pr.rate;
                    d += pr.dist;
                }
                let rd = rdcost(rdmult, r, d);
                if rd < best_uv.0 {
                    best_uv = (rd, m);
                    if replay_ok {
                        for p in 1..3 {
                            s.scratch.best_rec[p].copy_from(&s.scratch.rec[p]);
                        }
                    }
                }
            }
            modes = best_uv.1;
            self.load_ent(1, bsize, mi_row, mi_col, &saved[1]);
            self.load_ent(2, bsize, mi_row, mi_col, &saved[2]);
            rate += self.uv_mode_rate(bsize, &modes) as i64;
            for p in 1..3 {
                s.scratch.replay_plane = replay_ok.then_some(p);
                let pr = self.code_plane(
                    p,
                    &bp,
                    &modes,
                    uvtx,
                    TxSearch::Search { trellis: true },
                    &quant[p],
                    rdmult,
                    &mut s.scratch,
                    Some((&mut dec.txbs, &mut dec.coefs)),
                );
                s.scratch.replay_plane = None;
                rate += pr.rate;
                coef_rate += pr.rate;
                dist += pr.dist;
                nonzero |= pr.nonzero;
            }
        }

        prof::add(4, t0);
        // ---- skip flag
        let sctx = self.skip_ctx(mi_row, mi_col);
        let skip = !nonzero;
        if skip {
            // No coefficient flags are coded for a skipped block.
            rate -= coef_rate;
            rate += self.mode_costs.skip[sctx][1] as i64;
            dec.txbs.clear();
            dec.coefs.clear();
            // Entropy contexts of a skipped block are zero (already, as eob = 0).
        } else {
            rate += self.mode_costs.skip[sctx][0] as i64;
        }

        // ---- commit
        let info = MiInfo { bsize, y_mode: modes.y_mode, uv_mode: modes.uv_mode, skip, tx_size: tx };
        let rows = bsize.h4().min(self.mi_rows - mi_row);
        let cols = bsize.w4().min(self.mi_cols - mi_col);
        for r in 0..rows {
            let o = (mi_row + r) * self.mi_cols + mi_col;
            self.mi[o..o + cols].fill(info);
        }
        let txb_start = out.txbs.len() as u32;
        out.blocks.push(BlockDec {
            bsize,
            mi_row: mi_row as u16,
            mi_col: mi_col as u16,
            modes,
            tx_size: tx,
            skip,
            txb_start: 0,
            txb_count: dec.txbs.len() as u32,
        });
        let cb = out.coefs.len() as u32;
        out.txbs.extend(dec.txbs.iter().map(|t| super::TxbDec { coef_start: t.coef_start + cb, ..*t }));
        out.coefs.extend_from_slice(&dec.coefs);
        out.blocks.last_mut().unwrap().txb_start = txb_start;
        s.give(dec);
        rdcost(rdmult, rate, dist)
    }

    /// Search CfL alphas for U and V independently (by SSE of the prediction).
    fn search_cfl_alpha(&mut self, s: &mut SearchState, bp: &BlockPos, ym: &Modes, uvtx: TxSize) -> Option<Modes> {
        let mut alphas = [0i8; 2];
        let x = (bp.mi_col >> self.ssx) * 4;
        let y = (bp.mi_row >> self.ssy) * 4;
        let (w, h) = (uvtx.width(), uvtx.height());
        // U and V use the same reconstructed luma and subsampling geometry.
        let ac = &mut s.cfl_ac[..w * h];
        self.cfl_luma(bp, 1, x, y, w, h, ac);
        predict::cfl_ac(ac);
        for p in 1..3 {
            let av = self.block_edge_avail(p, bp, uvtx, DC_PRED, 0, false);
            let ip = IntraParams {
                mode: DC_PRED,
                angle_delta: 0,
                filter_intra: None,
                edge_filter: true,
                smooth_neighbor: false,
            };
            {
                let rec = &self.rec[p];
                predict::predict(&rec.data, rec.stride, x, y, uvtx, &ip, av, &mut s.pred, 64);
            }
            let src = &self.src[p];
            let mut best = (u64::MAX, 0i8);
            let all = predict::cfl_sse_all(&src.data[y * src.stride + x..], src.stride, &s.pred, 64, w, h, ac);
            for (i, &sse) in all.iter().enumerate() {
                if sse < best.0 {
                    best = (sse, i as i8 - 16);
                }
            }
            alphas[p - 1] = best.1;
        }
        if alphas == [0, 0] {
            return None;
        }
        Some(Modes { uv_mode: UV_CFL_PRED, angle_uv: 0, cfl_alpha: alphas, ..*ym })
    }
}

/// Order the first `k` entries as a stable sort by the `f64` key would, leaving the rest in
/// unspecified order (only those `k` are used).
fn sort_front<T>(v: &mut [(f64, T)], k: usize) {
    for i in 0..k.min(v.len()) {
        let mut best = i;
        for j in i + 1..v.len() {
            if v[j].0 < v[best].0 {
                best = j;
            }
        }
        // Rotating keeps the remaining entries in their original order, so ties
        // still resolve to the earlier entry.
        v[i..=best].rotate_right(1);
    }
}

fn sign_idx(a: i8) -> usize {
    match a.signum() {
        0 => 0,
        -1 => 1,
        _ => 2,
    }
}

pub fn filter_intra_allowed(bsize: BlockSize) -> bool {
    bsize.width() <= 32 && bsize.height() <= 32
}

pub fn cfl_allowed(bsize: BlockSize, lossless: bool, ssx: usize, ssy: usize) -> bool {
    if lossless {
        return bsize.subsampled(ssx, ssy) == BlockSize::B4x4;
    }
    bsize.width() <= 32 && bsize.height() <= 32
}

pub fn tx_size_cat(bsize: BlockSize) -> usize {
    let mut t = bsize.max_tx();
    let mut depth = 0;
    while t != TxSize::T4x4 {
        depth += 1;
        t = t.sub();
    }
    depth - 1
}

pub fn max_tx_depth(bsize: BlockSize) -> usize {
    let mut t = bsize.max_tx();
    let mut depth = 0;
    while depth < 2 && t != TxSize::T4x4 {
        depth += 1;
        t = t.sub();
    }
    depth
}

pub fn tx_depth(max: TxSize, tx: TxSize) -> usize {
    let mut t = max;
    let mut d = 0;
    while t != tx {
        t = t.sub();
        d += 1;
    }
    d
}

/// Probability (Q15) of the partitions "alike" a split along the missing
/// dimension, for blocks crossing the frame edge.
pub fn gathered_prob(cdf: &[u16], missing_rows: bool) -> u32 {
    let p = |e: usize| -> u32 { (if e > 0 { cdf[e - 1] as u32 } else { 32768 }) - cdf[e] as u32 };
    use Partition::*;
    if missing_rows {
        // partition_gather_vert_alike
        p(Vert as usize)
            + p(Split as usize)
            + p(HorzA as usize)
            + p(VertA as usize)
            + p(VertB as usize)
            + p(Vert4 as usize)
    } else {
        // partition_gather_horz_alike
        p(Horz as usize)
            + p(Split as usize)
            + p(HorzA as usize)
            + p(HorzB as usize)
            + p(VertA as usize)
            + p(Horz4 as usize)
    }
}

/// Sum of absolute 4x4 Hadamard transformed differences.
pub fn satd(src: &[u8], sstride: usize, pred: &[u8], pstride: usize, w: usize, h: usize) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if w != 0 && h != 0 && w.is_multiple_of(4) && h.is_multiple_of(4) && crate::has_avx2() {
        // Validate the full footprint before entering the pointer-based kernel.
        for (data, stride) in [(src, sstride), (pred, pstride)] {
            let end = (h - 1).checked_mul(stride).and_then(|n| n.checked_add(w)).expect("SATD size overflow");
            assert!(data.len() >= end, "SATD plane is too short");
        }
        // SAFETY: AVX2 is detected above; each load stays within the footprint.
        return unsafe { satd_avx2(src, sstride, pred, pstride, w, h) };
    }
    #[cfg(target_arch = "aarch64")]
    if w != 0 && h != 0 && w.is_multiple_of(4) && h.is_multiple_of(4) && crate::has_neon() {
        // Validate the full footprint before entering the pointer-based kernel.
        for (data, stride) in [(src, sstride), (pred, pstride)] {
            let end = (h - 1).checked_mul(stride).and_then(|n| n.checked_add(w)).expect("SATD size overflow");
            assert!(data.len() >= end, "SATD plane is too short");
        }
        // SAFETY: NEON is part of the aarch64 baseline; each load stays within the footprint.
        return unsafe { satd_neon(src, sstride, pred, pstride, w, h) };
    }
    satd_portable(src, sstride, pred, pstride, w, h)
}

fn satd_portable(src: &[u8], sstride: usize, pred: &[u8], pstride: usize, w: usize, h: usize) -> u64 {
    let mut total = 0u64;
    for by in (0..h).step_by(4) {
        for bx in (0..w).step_by(4) {
            let mut d = [0i32; 16];
            for r in 0..4 {
                for c in 0..4 {
                    d[r * 4 + c] = src[(by + r) * sstride + bx + c] as i32 - pred[(by + r) * pstride + bx + c] as i32;
                }
            }
            for r in 0..4 {
                let (a, b, c, e) = (d[r * 4], d[r * 4 + 1], d[r * 4 + 2], d[r * 4 + 3]);
                let (s0, s1, d0, d1) = (a + b, c + e, a - b, c - e);
                d[r * 4] = s0 + s1;
                d[r * 4 + 1] = s0 - s1;
                d[r * 4 + 2] = d0 + d1;
                d[r * 4 + 3] = d0 - d1;
            }
            for c in 0..4 {
                let (a, b, cc, e) = (d[c], d[4 + c], d[8 + c], d[12 + c]);
                let (s0, s1, d0, d1) = (a + b, cc + e, a - b, cc - e);
                total += ((s0 + s1).abs() + (s0 - s1).abs() + (d0 + d1).abs() + (d0 - d1).abs()) as u64;
            }
        }
    }
    total / 2
}

/// Four independent 4x4 transforms per iteration. The vertical transform
/// precedes the horizontal one; integer Hadamard transforms commute exactly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn satd_avx2(src: &[u8], sstride: usize, pred: &[u8], pstride: usize, w: usize, h: usize) -> u64 {
    use std::arch::x86_64::*;
    // SAFETY: the caller checks CPU support and both input footprints. Loads
    // use exactly 4, 8 or 16 bytes, including at the right/bottom image edges.
    unsafe {
        let mut total = 0u64;
        let ones = _mm256_set1_epi16(1);
        for by in (0..h).step_by(4) {
            let mut bx = 0;
            while bx < w {
                let n = (w - bx).min(16);
                let mut rows = [_mm256_setzero_si256(); 4];
                for r in 0..4 {
                    let s = src.as_ptr().add((by + r) * sstride + bx);
                    let p = pred.as_ptr().add((by + r) * pstride + bx);
                    let load = |ptr: *const u8| {
                        if n >= 16 {
                            _mm_loadu_si128(ptr.cast())
                        } else if n >= 8 {
                            _mm_loadl_epi64(ptr.cast())
                        } else {
                            _mm_cvtsi32_si128(std::ptr::read_unaligned(ptr.cast()))
                        }
                    };
                    rows[r] = _mm256_sub_epi16(_mm256_cvtepu8_epi16(load(s)), _mm256_cvtepu8_epi16(load(p)));
                }
                let s0 = _mm256_add_epi16(rows[0], rows[1]);
                let s1 = _mm256_add_epi16(rows[2], rows[3]);
                let d0 = _mm256_sub_epi16(rows[0], rows[1]);
                let d1 = _mm256_sub_epi16(rows[2], rows[3]);
                let vertical = [
                    _mm256_add_epi16(s0, s1),
                    _mm256_sub_epi16(s0, s1),
                    _mm256_add_epi16(d0, d1),
                    _mm256_sub_epi16(d0, d1),
                ];
                let mut sums = _mm256_setzero_si256();
                for v in vertical {
                    let s = _mm256_hadd_epi16(v, v);
                    let d = _mm256_hsub_epi16(v, v);
                    let a = _mm256_abs_epi16(_mm256_hadd_epi16(s, d));
                    let b = _mm256_abs_epi16(_mm256_hsub_epi16(s, d));
                    // Max coefficient = 16 * 255 = 4080. The pairwise
                    // absolute sums fit i16; accumulate the totals in i32.
                    sums = _mm256_add_epi32(sums, _mm256_madd_epi16(_mm256_add_epi16(a, b), ones));
                }
                let mut lanes = [0i32; 8];
                _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sums);
                total += lanes.iter().map(|&v| v as u64).sum::<u64>();
                bx += if n >= 16 {
                    16
                } else if n >= 8 {
                    8
                } else {
                    4
                };
            }
        }
        // Horizontal adds/subtracts duplicate each coefficient twice. Keep
        // the original single rounding step after summing all 4x4 blocks.
        total / 4
    }
}

/// Two 4x4 transforms per iteration (one 4-column block when only four columns
/// remain, the upper lanes then stay zero). Vertical transform first; the
/// horizontal one only needs `|x + y| + |x - y| = 2 * max(|x|, |y|)` on the
/// pair sums and differences, so each 4-point row contributes
/// `max(|s0|, |s1|) + max(|d0|, |d1|)`, which is half its absolute sum.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn satd_neon(src: &[u8], sstride: usize, pred: &[u8], pstride: usize, w: usize, h: usize) -> u64 {
    use std::arch::aarch64::*;
    // SAFETY: the caller checks both input footprints. Loads use exactly 4 or
    // 8 bytes, including at the right/bottom image edges.
    unsafe {
        let mut acc = vdupq_n_u32(0);
        for by in (0..h).step_by(4) {
            let mut bx = 0;
            while bx < w {
                let wide = w - bx >= 8;
                let mut rows = [vdupq_n_s16(0); 4];
                for r in 0..4 {
                    let s = src.as_ptr().add((by + r) * sstride + bx);
                    let p = pred.as_ptr().add((by + r) * pstride + bx);
                    let load = |ptr: *const u8| {
                        if wide {
                            vld1_u8(ptr)
                        } else {
                            vcreate_u8(u64::from(std::ptr::read_unaligned(ptr.cast::<u32>())))
                        }
                    };
                    rows[r] = vreinterpretq_s16_u16(vsubl_u8(load(s), load(p)));
                }
                let s0 = vaddq_s16(rows[0], rows[1]);
                let s1 = vaddq_s16(rows[2], rows[3]);
                let d0 = vsubq_s16(rows[0], rows[1]);
                let d1 = vsubq_s16(rows[2], rows[3]);
                let vertical = [vaddq_s16(s0, s1), vsubq_s16(s0, s1), vaddq_s16(d0, d1), vsubq_s16(d0, d1)];
                for v in vertical {
                    // Pair sums and differences of each 4-point row: lanes 0, 1 are block 0 and
                    // lanes 2, 3 block 1 (the upper half repeats them).
                    let (e, o) = (vuzp1q_s16(v, v), vuzp2q_s16(v, v));
                    let s = vabs_s16(vget_low_s16(vaddq_s16(e, o)));
                    let d = vabs_s16(vget_low_s16(vsubq_s16(e, o)));
                    let m = vadd_s16(vpmax_s16(s, s), vpmax_s16(d, d));
                    acc = vaddw_u16(acc, vreinterpret_u16_s16(m));
                }
                bx += if wide { 8 } else { 4 };
            }
        }
        // Every pair maximum was added twice (lanes 2 and 3 duplicate lanes 0 and 1).
        u64::from(vaddvq_u32(acc)) / 2
    }
}

#[cfg(test)]
mod search_tests {
    use super::*;
    use crate::{Chroma, Tune, frame_config};

    fn encoder(w: usize, h: usize, q: u8, chroma: Chroma, pattern: usize) -> Encoder {
        let cfg = frame_config(q, chroma, Tune::Iq, 5);
        let mut seed = 0x12345678u32;
        let data: Vec<Vec<u8>> = (0..cfg.num_planes)
            .map(|p| {
                let (sx, sy) = if p == 0 { (0, 0) } else { (cfg.ssx, cfg.ssy) };
                let (pw, ph) = ((w + sx) >> sx, (h + sy) >> sy);
                (0..pw * ph)
                    .map(|i| {
                        seed ^= seed << 13;
                        seed ^= seed >> 17;
                        seed ^= seed << 5;
                        match pattern {
                            0 => 128,
                            1 => ((i % pw * 3 + i / pw * 5 + p * 31) % 256) as u8,
                            _ => seed as u8,
                        }
                    })
                    .collect()
            })
            .collect();
        let planes: Vec<_> = data
            .iter()
            .enumerate()
            .map(|(p, v)| {
                let (sx, sy) = if p == 0 { (0, 0) } else { (cfg.ssx, cfg.ssy) };
                let (pw, ph) = ((w + sx) >> sx, (h + sy) >> sy);
                (v.as_slice(), pw, pw, ph)
            })
            .collect();
        let mut e = Encoder::new(cfg, &planes);
        if q != 0 {
            e.compute_ssim_scaling();
        }
        e
    }

    fn state(e: &Encoder) -> String {
        // Include padding and state outside the searched region as well.
        format!(
            "{:?}{:?}{:?}{:?}",
            e.rec.iter().map(|p| &p.data).collect::<Vec<_>>(),
            e.ctx.above_ent,
            e.ctx.left_ent,
            e.mi
        )
    }

    fn decisions(d: &Decisions) -> String {
        format!("{:?}{:?}{:?}{:?}", d.parts, d.blocks, d.txbs, d.coefs)
    }

    fn search(e: &mut Encoder, rd_budget: bool, tx_cache: bool, budget: i64) -> (i64, Decisions) {
        let mut s = SearchState::new();
        s.use_rd_budget = rd_budget;
        s.use_tx_cache = tx_cache;
        s.quant = e.quant(e.cfg.base_q);
        s.rdmult = e.rd_mult(e.cfg.base_q);
        let mut d = Decisions::default();
        let rd = e.rd_pick_partition(&mut s, BlockSize::B64x64, 0, 0, &mut d, budget);
        (rd, d)
    }

    #[test]
    fn optimized_search_matches_exhaustive() {
        for (w, h) in [(37, 29), (64, 64), (73, 65)] {
            for q in [0, 1, 128, 255] {
                for chroma in [Chroma::Yuv420, Chroma::Yuv444, Chroma::Mono] {
                    for pattern in 0..3 {
                        let mut reference = encoder(w, h, q, chroma, pattern);
                        let (rd, d) = search(&mut reference, false, false, i64::MAX);
                        let expected_state = state(&reference);
                        let expected_decisions = decisions(&d);
                        for (rd_budget, tx_cache) in [(true, false), (false, true), (true, true)] {
                            let mut e = encoder(w, h, q, chroma, pattern);
                            let (actual_rd, actual_d) = search(&mut e, rd_budget, tx_cache, i64::MAX);
                            let case = (w, h, q, chroma, pattern, rd_budget, tx_cache);
                            assert_eq!(actual_rd, rd, "RD: {case:?}");
                            assert_eq!(decisions(&actual_d), expected_decisions, "decisions: {case:?}");
                            assert_eq!(state(&e), expected_state, "state: {case:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn plane_bound_is_exclusive_and_combines_rate_before_rounding() {
        for q in [0, 128, 255] {
            for pattern in 0..3 {
                let bp = BlockPos {
                    bsize: BlockSize::B32x32,
                    mi_row: 0,
                    mi_col: 0,
                    partition: Partition::None,
                    luma_tx: TxSize::T4x4,
                };
                let modes = Modes::default();
                let mut reference = encoder(64, 64, q, Chroma::Mono, pattern);
                let quant = reference.quant(q)[0];
                let rdmult = reference.rd_mult(q);
                let pr = reference.code_plane(
                    0,
                    &bp,
                    &modes,
                    bp.luma_tx,
                    TxSearch::Search { trellis: true },
                    &quant,
                    rdmult,
                    &mut Scratch::new(),
                    None,
                );
                // Keep rates unrounded until the full candidate is compared.
                let rate = 511;
                let dist = 3;
                let rd = rdcost(rdmult, rate + pr.rate, dist + pr.dist);
                for budget in [0, rd / 2, rd, rd + 1] {
                    let mut e = encoder(64, 64, q, Chroma::Mono, pattern);
                    let initial = state(&e);
                    let mut txbs = Vec::new();
                    let mut coefs = Vec::new();
                    let result = e.try_code_plane(
                        0,
                        &bp,
                        &modes,
                        bp.luma_tx,
                        TxSearch::Search { trellis: true },
                        &quant,
                        rdmult,
                        &mut Scratch::new(),
                        Some((&mut txbs, &mut coefs)),
                        Some(PlaneBound { rd: budget, rate, dist }),
                    );
                    assert_eq!(result.is_some(), budget > rd, "q={q} pattern={pattern} budget={budget}");
                    if let Some(actual) = result {
                        assert_eq!(actual.rate, pr.rate);
                        assert_eq!(actual.dist, pr.dist);
                        assert_eq!(actual.nonzero, pr.nonzero);
                        assert_eq!(state(&e), state(&reference));
                    } else if budget == 0 {
                        assert!(txbs.is_empty());
                        assert!(coefs.is_empty());
                        assert_eq!(state(&e), initial);
                    }
                }
            }
        }
    }

    #[test]
    fn finite_budget_is_exclusive_and_restores_failed_search() {
        for (w, h) in [(37, 29), (64, 64)] {
            for chroma in [Chroma::Yuv420, Chroma::Yuv444, Chroma::Mono] {
                let mut reference = encoder(w, h, 128, chroma, 1);
                let (rd, d) = search(&mut reference, false, false, i64::MAX);
                let mut e = encoder(w, h, 128, chroma, 1);
                let (actual_rd, actual_d) = search(&mut e, true, true, rd + 1);
                assert_eq!(actual_rd, rd);
                assert_eq!(decisions(&actual_d), decisions(&d));
                assert_eq!(state(&e), state(&reference));
                for budget in [0, rd / 2, rd] {
                    let mut e = encoder(w, h, 128, chroma, 1);
                    let before = state(&e);
                    let (actual_rd, actual_d) = search(&mut e, true, true, budget);
                    assert_eq!(actual_rd, i64::MAX);
                    assert!(actual_d.parts.is_empty());
                    assert!(actual_d.blocks.is_empty());
                    assert!(actual_d.txbs.is_empty());
                    assert!(actual_d.coefs.is_empty());
                    assert_eq!(state(&e), before);
                }
            }
        }
    }
}

#[cfg(all(test, any(target_arch = "x86_64", target_arch = "aarch64")))]
mod satd_tests {
    use super::*;

    #[test]
    fn simd_satd_matches_portable() {
        #[cfg(target_arch = "x86_64")]
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut seed = 0x12345678u32;
        for w in [4, 8, 12, 16, 20, 32, 64] {
            for h in [4, 8, 16, 32, 64] {
                // Different, unaligned strides; no padding after the last row.
                let (ss, ps) = (w + 3, w + 7);
                let mut src = vec![0u8; (h - 1) * ss + w];
                let mut pred = vec![0u8; (h - 1) * ps + w];
                for pattern in 0..6 {
                    for data in [&mut src, &mut pred] {
                        for (i, v) in data.iter_mut().enumerate() {
                            seed ^= seed << 13;
                            seed ^= seed >> 17;
                            seed ^= seed << 5;
                            *v = match pattern {
                                0 => 0,
                                1 => {
                                    if i % 2 == 0 {
                                        255
                                    } else {
                                        0
                                    }
                                }
                                2 => 255,
                                _ => seed as u8,
                            };
                        }
                    }
                    if pattern == 2 {
                        pred.fill(0);
                    }
                    let expected = satd_portable(&src, ss, &pred, ps, w, h);
                    // SAFETY: the test returns early unless AVX2 was detected; NEON is baseline.
                    #[cfg(target_arch = "x86_64")]
                    let actual = unsafe { satd_avx2(&src, ss, &pred, ps, w, h) };
                    // SAFETY: NEON is part of the aarch64 baseline.
                    #[cfg(target_arch = "aarch64")]
                    let actual = unsafe { satd_neon(&src, ss, &pred, ps, w, h) };
                    assert_eq!(actual, expected, "{w}x{h}, pattern {pattern}");
                    assert_eq!(satd(&src, ss, &pred, ps, w, h), expected);
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn satd_rejects_short_plane() {
        satd(&[0; 15], 4, &[0; 16], 4, 4, 4);
    }
}
