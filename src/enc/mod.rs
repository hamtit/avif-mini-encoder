//! AV1 key-frame encoder (all-intra, one frame).

pub mod block;
pub mod costs;
mod search;
mod write;

use crate::cdf::CdfContext;
use crate::coef::CoefCosts;
use crate::consts::*;
use crate::ec::{BitWriter, CdfOnly, SymbolWriter};
use crate::frame::Plane;
use crate::obu::{self, FrameHeader};
use crate::quant::PlaneQuant;
use costs::ModeCosts;

/// Per 4x4 (mi) mode information needed for contexts and filtering.
#[derive(Clone, Copy, Debug)]
pub struct MiInfo {
    pub bsize: BlockSize,
    pub y_mode: u8,
    pub uv_mode: u8,
    pub skip: bool,
    pub tx_size: TxSize,
}

impl Default for MiInfo {
    fn default() -> Self {
        MiInfo { bsize: BlockSize::B4x4, y_mode: DC_PRED, uv_mode: DC_PRED, skip: false, tx_size: TxSize::T4x4 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TxbDec {
    pub tx_type: TxType,
    pub eob: u16,
    pub coef_start: u32,
}

#[derive(Clone, Debug)]
pub struct BlockDec {
    pub bsize: BlockSize,
    pub mi_row: u16,
    pub mi_col: u16,
    pub modes: block::Modes,
    pub tx_size: TxSize,
    pub skip: bool,
    pub txb_start: u32,
    pub txb_count: u32,
}

/// Coding decisions of a superblock in bitstream order.
#[derive(Default, Clone)]
pub struct Decisions {
    /// Superblock qindex (delta q).
    pub qindex: u8,
    pub parts: Vec<Partition>,
    pub blocks: Vec<BlockDec>,
    pub txbs: Vec<TxbDec>,
    pub coefs: Vec<i16>,
}

impl Decisions {
    pub fn clear(&mut self) {
        self.parts.clear();
        self.blocks.clear();
        self.txbs.clear();
        self.coefs.clear();
    }

    pub fn append(&mut self, o: &Decisions) {
        let tb = self.txbs.len() as u32;
        let cb = self.coefs.len() as u32;
        self.parts.extend_from_slice(&o.parts);
        self.blocks.extend(o.blocks.iter().map(|b| BlockDec { txb_start: b.txb_start + tb, ..b.clone() }));
        self.txbs.extend(o.txbs.iter().map(|t| TxbDec { coef_start: t.coef_start + cb, ..*t }));
        self.coefs.extend_from_slice(&o.coefs);
    }

    fn as_view(&self) -> write::SbView<'_> {
        write::SbView {
            qindex: self.qindex,
            parts: &self.parts,
            blocks: &self.blocks,
            txbs: &self.txbs,
            coefs: &self.coefs,
        }
    }
}

/// Slice range of one superblock inside a [`FrameDecisions`] arena.
#[derive(Clone, Copy, Debug)]
pub struct SbRange {
    pub qindex: u8,
    pub part_start: u32,
    pub part_end: u32,
    pub block_start: u32,
    pub block_end: u32,
}

/// Flat arena holding all superblock coding decisions for a tile or frame.
#[derive(Default)]
pub struct FrameDecisions {
    pub sbs: Vec<SbRange>,
    pub parts: Vec<Partition>,
    pub blocks: Vec<BlockDec>,
    pub txbs: Vec<TxbDec>,
    pub coefs: Vec<i16>,
}

impl FrameDecisions {
    pub fn with_sb_capacity(num_sbs: usize) -> Self {
        FrameDecisions {
            sbs: Vec::with_capacity(num_sbs),
            parts: Vec::with_capacity(num_sbs * 4),
            blocks: Vec::with_capacity(num_sbs * 4),
            txbs: Vec::with_capacity(num_sbs * 8),
            coefs: Vec::with_capacity(num_sbs * 64),
        }
    }

    pub fn push_sb(&mut self, dec: &Decisions, mi_row_off: u16, mi_col_off: u16) {
        let part_start = self.parts.len() as u32;
        let block_start = self.blocks.len() as u32;
        let tb = self.txbs.len() as u32;
        let cb = self.coefs.len() as u32;
        self.parts.extend_from_slice(&dec.parts);
        self.blocks.extend(dec.blocks.iter().map(|b| BlockDec {
            mi_row: b.mi_row + mi_row_off,
            mi_col: b.mi_col + mi_col_off,
            txb_start: b.txb_start + tb,
            ..b.clone()
        }));
        self.txbs.extend(dec.txbs.iter().map(|t| TxbDec { coef_start: t.coef_start + cb, ..*t }));
        self.coefs.extend_from_slice(&dec.coefs);
        self.sbs.push(SbRange {
            qindex: dec.qindex,
            part_start,
            part_end: self.parts.len() as u32,
            block_start,
            block_end: self.blocks.len() as u32,
        });
    }

    fn sb_view(&self, idx: usize) -> write::SbView<'_> {
        let r = self.sbs[idx];
        write::SbView {
            qindex: r.qindex,
            parts: &self.parts[r.part_start as usize..r.part_end as usize],
            blocks: &self.blocks[r.block_start as usize..r.block_end as usize],
            txbs: &self.txbs,
            coefs: &self.coefs,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TileInfo {
    pub mi_row_start: usize,
    pub mi_row_end: usize,
    pub mi_col_start: usize,
    pub mi_col_end: usize,
}

#[derive(Clone, Copy, Debug)]
struct TileRect {
    sb_row_start: usize,
    sb_row_end: usize,
    sb_col_start: usize,
    sb_col_end: usize,
    info: TileInfo,
}

/// Choose a uniform tile grid `(tile_cols_log2, tile_rows_log2)`: the mandatory
/// minimum for frames over AV1's single-tile limits, raised for multi-threaded
/// encoding when `threads >= 2` and the frame is large enough that tile overhead
/// stays small.
fn choose_tile_grid(mi_cols: usize, mi_rows: usize, threads: usize) -> (u32, u32) {
    let sb_cols = mi_cols.div_ceil(16);
    let sb_rows = mi_rows.div_ceil(16);
    let lim = obu::tile_limits(mi_cols, mi_rows);
    let mut cols_log2 = 0u32;
    let mut rows_log2 = 0u32;
    if threads >= 2 {
        if sb_cols >= 4 && sb_rows >= 4 {
            cols_log2 = 1.min(lim.max_cols_log2);
            if threads >= 4 && sb_cols >= 6 && sb_rows >= 6 {
                rows_log2 = 1.min(lim.max_rows_log2);
            }
            if threads >= 8 && sb_cols >= 16 && sb_rows >= 16 {
                cols_log2 = 2.min(lim.max_cols_log2);
            }
        } else if sb_cols >= 8 {
            cols_log2 = 1.min(lim.max_cols_log2);
        } else if sb_rows >= 8 {
            rows_log2 = 1.min(lim.max_rows_log2);
        }
    }
    // Frames over AV1's single-tile limits (4096 wide, 4096 * 2304 samples) need
    // mandatory tiles regardless of the thread count.
    cols_log2 = cols_log2.max(lim.min_cols_log2);
    while cols_log2 + rows_log2 < lim.min_tiles_log2 && rows_log2 < lim.max_rows_log2 {
        rows_log2 += 1;
    }
    while cols_log2 + rows_log2 < lim.min_tiles_log2 && cols_log2 < lim.max_cols_log2 {
        cols_log2 += 1;
    }
    (cols_log2, rows_log2)
}

/// Compute the AV1 uniform tile rectangles in raster order.
fn compute_tiles(mi_cols: usize, mi_rows: usize, tile_cols_log2: u32, tile_rows_log2: u32) -> Vec<TileRect> {
    let sb_cols = mi_cols.div_ceil(16);
    let sb_rows = mi_rows.div_ceil(16);
    let tile_width_sb = sb_cols.div_ceil(1 << tile_cols_log2);
    let tile_height_sb = sb_rows.div_ceil(1 << tile_rows_log2);
    let mut col_ranges = Vec::new();
    let mut sc = 0;
    while sc < sb_cols {
        col_ranges.push((sc, (sc + tile_width_sb).min(sb_cols)));
        sc += tile_width_sb;
    }
    let mut row_ranges = Vec::new();
    let mut sr = 0;
    while sr < sb_rows {
        row_ranges.push((sr, (sr + tile_height_sb).min(sb_rows)));
        sr += tile_height_sb;
    }
    let mut tiles = Vec::with_capacity(row_ranges.len() * col_ranges.len());
    for &(sr0, sr1) in &row_ranges {
        for &(sc0, sc1) in &col_ranges {
            tiles.push(TileRect {
                sb_row_start: sr0,
                sb_row_end: sr1,
                sb_col_start: sc0,
                sb_col_end: sc1,
                info: TileInfo {
                    mi_row_start: sr0 * 16,
                    mi_row_end: (sr1 * 16).min(mi_rows),
                    mi_col_start: sc0 * 16,
                    mi_col_end: (sc1 * 16).min(mi_cols),
                },
            });
        }
    }
    tiles
}

/// Schedule for updating cost tables from adapted CDFs: frequent at the start
/// of a tile while CDFs adapt rapidly, then periodic every 16 superblocks.
#[inline]
fn should_update_costs(sb_idx: usize) -> bool {
    sb_idx < 4 || sb_idx == 7 || sb_idx == 15 || (sb_idx + 1).is_multiple_of(16)
}

/// Above / left entropy contexts (level + dc sign per 4 samples).
#[derive(Clone)]
pub struct Contexts {
    pub above_ent: [Vec<u8>; 3],
    pub left_ent: [Vec<u8>; 3],
}

impl Contexts {
    fn new(mi_cols: usize, mi_rows: usize) -> Contexts {
        let w = mi_cols + 32;
        let h = mi_rows + 32;
        Contexts { above_ent: [vec![0; w], vec![0; w], vec![0; w]], left_ent: [vec![0; h], vec![0; h], vec![0; h]] }
    }
}

/// Settings for one AV1 frame.
#[derive(Clone, Debug)]
pub struct FrameConfig {
    pub base_q: u8,
    pub num_planes: usize,
    pub ssx: usize,
    pub ssy: usize,
    /// Chroma dc/ac delta q (frame level).
    pub uv_delta_q: (i32, i32),
    /// QM levels (y, uv); None disables quantization matrices.
    pub qm: Option<(u8, u8)>,
    pub qm_dist: bool,
    pub deltaq_variance_boost: bool,
    pub tune_iq: bool,
    pub sharpness: i32,
    pub enable_cdef: bool,
    pub speed: u8,
    /// Worker threads for parallel tile encoding and CDEF strength search.
    pub threads: usize,
}

pub struct Encoder {
    pub cfg: FrameConfig,
    pub mi_rows: usize,
    pub mi_cols: usize,
    pub num_planes: usize,
    pub ssx: usize,
    pub ssy: usize,
    pub src: Vec<Plane>,
    pub rec: Vec<Plane>,
    pub mi: Vec<MiInfo>,
    pub tile: TileInfo,
    pub ctx: Contexts,
    pub cdf: Option<Box<CdfContext>>,
    pub coef_costs: Box<CoefCosts>,
    pub mode_costs: Box<ModeCosts>,
    pub lossless: bool,
    pub sharpness: i32,
    pub qm_dist: bool,
    /// Quantizers per qindex (lazily filled): [y, u, v].
    quants: Vec<Option<[PlaneQuant; 3]>>,
    /// Variance boost qindex per superblock.
    sb_qindex: Vec<u8>,
    /// SSIM rdmult scaling per 16x16.
    ssim_scale: Vec<f64>,
    ssim_cols: usize,
    pub delta_q_res: u8,
    pub cdef_idx: Vec<i8>,
    pub cdef_bits: u8,
}

pub fn round_up(v: usize, a: usize) -> usize {
    v.div_ceil(a) * a
}

impl Encoder {
    /// `planes`: Y (and U, V) source samples with their visible sizes.
    pub fn new(cfg: FrameConfig, planes: &[(&[u8], usize, usize, usize)]) -> Encoder {
        let (_, _, width, height) = planes[0];
        let mi_cols = 2 * width.div_ceil(8);
        let mi_rows = 2 * height.div_ceil(8);
        // Allocate whole superblocks plus a margin so transform blocks that
        // cross the frame edge stay inside the buffer.
        let aw = round_up(mi_cols * 4, 64) + 64;
        let ah = round_up(mi_rows * 4, 64) + 64;
        let mut src = Vec::new();
        let mut rec = Vec::new();
        for (i, &(data, stride, w, h)) in planes.iter().enumerate() {
            let (sx, sy) = if i == 0 { (0, 0) } else { (cfg.ssx, cfg.ssy) };
            src.push(Plane::from_slice(data, stride, w, h, aw >> sx, ah >> sy));
            rec.push(Plane::new(w, h, aw >> sx, ah >> sy));
        }
        let cdf = CdfContext::new(cfg.base_q);
        let coef_costs = CoefCosts::new(&cdf.coef);
        let mode_costs = ModeCosts::new(&cdf);
        let lossless = cfg.base_q == 0 && cfg.uv_delta_q == (0, 0);
        Encoder {
            mi_rows,
            mi_cols,
            num_planes: cfg.num_planes,
            ssx: cfg.ssx,
            ssy: cfg.ssy,
            src,
            rec,
            mi: vec![MiInfo::default(); mi_rows * mi_cols],
            tile: TileInfo { mi_row_start: 0, mi_row_end: mi_rows, mi_col_start: 0, mi_col_end: mi_cols },
            ctx: Contexts::new(mi_cols, mi_rows),
            cdf: Some(cdf),
            coef_costs,
            mode_costs,
            lossless,
            sharpness: cfg.sharpness,
            qm_dist: cfg.qm_dist,
            quants: vec![None; 256],
            sb_qindex: Vec::new(),
            ssim_scale: Vec::new(),
            ssim_cols: 0,
            delta_q_res: 0,
            cdef_idx: Vec::new(),
            cdef_bits: 0,
            cfg,
        }
    }

    pub fn quant(&mut self, qindex: u8) -> [PlaneQuant; 3] {
        if let Some(q) = self.quants[qindex as usize] {
            return q;
        }
        let (qy, quv) = self.cfg.qm.unwrap_or((15, 15));
        let (dc, ac) = self.cfg.uv_delta_q;
        let rounding_fp = if self.cfg.tune_iq { 64 } else { 48 };
        let q = [
            PlaneQuant::new_with_rounding(qindex as i32, 0, 0, qy, false, rounding_fp),
            PlaneQuant::new_with_rounding(qindex as i32, dc, ac, quv, true, rounding_fp),
            PlaneQuant::new_with_rounding(qindex as i32, dc, ac, quv, true, rounding_fp),
        ];
        self.quants[qindex as usize] = Some(q);
        q
    }

    /// libaom's av1_compute_rd_mult_based_on_qindex for a key frame.
    pub fn rd_mult(&self, qindex: u8) -> i64 {
        let q = crate::quant::dc_q(qindex as i32, 0) as f64;
        let mut rdmult = (q * q * (3.3 + 0.0015 * q)) as i64;
        if self.cfg.tune_iq {
            let weight = ((255 - qindex as i64) * 3 / 4).clamp(0, 72) + 128;
            rdmult = (rdmult as f64 * weight as f64 / 128.0) as i64;
        }
        // Development switch: scale lambda (read once).
        // Default: 0.6 for tune IQ. This search ends up at a lower rate than
        // libaom at the same qindex (about 6% smaller, SSIMULACRA2 1-3 points
        // lower, on the same RD curve); 0.6 matches libaom's quality at the
        // same quality setting (size -2%, SSIMULACRA2 +-0.5 on Kodak).
        static RD_SCALE: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
        let env = *RD_SCALE.get_or_init(|| std::env::var("AV1_RDSCALE").ok().and_then(|v| v.parse().ok()));
        let scale = env.unwrap_or(if self.cfg.tune_iq { 0.6 } else { 1.0 });
        if scale != 1.0 {
            rdmult = (rdmult as f64 * scale) as i64;
        }
        rdmult.max(1)
    }

    /// Block rdmult scaled by the SSIM factors of the 16x16 units it covers.
    pub fn block_rdmult(&self, base: i64, bsize: BlockSize, mi_row: usize, mi_col: usize) -> i64 {
        if self.ssim_scale.is_empty() {
            return base;
        }
        let rows = self.mi_rows.div_ceil(4);
        let r0 = mi_row / 4;
        let c0 = mi_col / 4;
        let nr = bsize.h4().div_ceil(4);
        let nc = bsize.w4().div_ceil(4);
        let r1 = (r0 + nr).min(rows);
        let c1 = (c0 + nc).min(self.ssim_cols);
        let mut prod = 1.0f64;
        let mut n = 0usize;
        for r in r0..r1 {
            let row = r * self.ssim_cols;
            for c in c0..c1 {
                prod *= self.ssim_scale[row + c];
                n += 1;
            }
        }
        let g = match n {
            1 => prod,
            2 => prod.sqrt(),
            4 => prod.sqrt().sqrt(),
            _ => prod.powf(1.0 / n as f64),
        };
        ((base as f64 * g + 0.5) as i64).max(1)
    }

    fn compute_ssim_scaling(&mut self) {
        let cols = self.mi_cols.div_ceil(4);
        let rows = self.mi_rows.div_ceil(4);
        let src = &self.src[0];
        let mut f = vec![0.0f64; rows * cols];
        let mut log_sum = 0.0;
        for r in 0..rows {
            for c in 0..cols {
                let mut var = 0.0;
                let mut n = 0.0;
                let mut mr = r * 4;
                while mr < self.mi_rows && mr < (r + 1) * 4 {
                    let mut mc = c * 4;
                    while mc < self.mi_cols && mc < (c + 1) * 4 {
                        var += block_variance(src, mc * 4, mr * 4, 8) as f64 / 64.0;
                        n += 1.0;
                        mc += 2;
                    }
                    mr += 2;
                }
                let v = var / n;
                let v = 67.035434 * (1.0 - (-0.0021489 * v).exp()) + 17.492222;
                f[r * cols + c] = v;
                log_sum += v.ln();
            }
        }
        let g = (log_sum / (rows * cols) as f64).exp();
        for v in &mut f {
            *v /= g;
        }
        self.ssim_scale = f;
        self.ssim_cols = cols;
    }

    fn compute_variance_boost(&mut self) {
        let sb_cols = self.mi_cols.div_ceil(16);
        let sb_rows = self.mi_rows.div_ceil(16);
        let base = self.cfg.base_q as i32;
        let mut out = vec![self.cfg.base_q; sb_cols * sb_rows];
        let src = &self.src[0];
        for sr in 0..sb_rows {
            for sc in 0..sb_cols {
                let mut vars = [0u32; 64];
                for i in 0..8 {
                    for j in 0..8 {
                        vars[i * 8 + j] = (block_variance(src, sc * 64 + j * 8, sr * 64 + i * 8, 8) / 64) as u32;
                    }
                }
                vars.sort_unstable();
                let octile = 5;
                let per = 8;
                let mid = octile * per - 1;
                let lo = (per - 1).max(mid - per);
                let hi = 63.min(mid + per);
                let mut variance = (vars[lo] + vars[mid] * 2 + vars[hi] + 2) / 4;
                if variance == 0 {
                    variance = 1;
                }
                let strength = 3.0f64;
                let ratio = (0.15 * strength * (-(variance as f64).log2() + 10.0) + 1.0).clamp(1.0, 8.0);
                let base_q = crate::quant::ac_q(base, 0) as f64 / 4.0;
                let target_q = base_q / ratio;
                let mut tq = 0;
                while tq < 255 && (crate::quant::ac_q(tq, 0) as f64 / 4.0) < target_q {
                    tq += 1;
                }
                let boost = (((base as f64 + 544.0) * (base - tq) as f64) / 1279.0).round() as i32;
                let boost = boost.min(80);
                out[sr * sb_cols + sc] = (base - boost).max(1) as u8;
            }
        }
        self.sb_qindex = out;
    }

    /// Encode one tile's superblocks in raster order and return its flat
    /// [`FrameDecisions`] arena (with `BlockDec` coordinates shifted by
    /// `(mi_row_off, mi_col_off)`).
    fn encode_tile_sbs(
        &mut self,
        sb_rows: usize,
        sb_cols: usize,
        use_deltaq: bool,
        mi_row_off: u16,
        mi_col_off: u16,
    ) -> FrameDecisions {
        let mut out = FrameDecisions::with_sb_capacity(sb_rows * sb_cols);
        let mut wctx = Contexts::new(self.mi_cols, self.mi_rows);
        let mut wscratch = write::WriteScratch::new();
        let mut cur_q = self.cfg.base_q;
        let mut replay_q = self.cfg.base_q;
        let mut search = search::SearchState::new();
        let mut dec = Decisions::default();
        let mut sb_idx = 0usize;
        for sr in 0..sb_rows {
            for p in 0..3 {
                self.ctx.left_ent[p].fill(0);
                wctx.left_ent[p].fill(0);
            }
            for sc in 0..sb_cols {
                let (mi_row, mi_col) = (sr * 16, sc * 16);
                let mut sb_q = self.cfg.base_q;
                if use_deltaq {
                    let target = self.sb_qindex[sr * sb_cols + sc] as i32;
                    let res = self.delta_q_res as i32;
                    let t = target.clamp(res, 256 - res);
                    let prev = cur_q as i32;
                    let sign = if t - prev >= 0 { 1 } else { -1 };
                    let dz = res / 4;
                    let abs = ((t - prev).abs() + dz) & !(res - 1);
                    sb_q = (prev + sign * abs).max(1) as u8;
                }
                dec.clear();
                self.encode_sb(&mut search, mi_row, mi_col, sb_q, &mut dec);
                let first = &dec.blocks[0];
                if use_deltaq && !(first.bsize == BlockSize::B64x64 && first.skip) {
                    cur_q = sb_q;
                }
                dec.qindex = sb_q;
                // Keep the CDFs in step with the bitstream without reallocating CdfContext.
                let mut cdf = self.cdf.take().unwrap();
                let dq = if use_deltaq { Some(&mut replay_q) } else { None };
                let view = dec.as_view();
                self.write_sb(&mut CdfOnly, &mut cdf, &mut wctx, &view, mi_row, mi_col, dq, None, &mut wscratch);
                if should_update_costs(sb_idx) {
                    self.coef_costs.update(&cdf.coef);
                    self.mode_costs.update(&cdf);
                }
                self.cdf = Some(cdf);
                out.push_sb(&dec, mi_row_off, mi_col_off);
                sb_idx += 1;
            }
        }
        out
    }

    /// Build a tile-local [`Encoder`] for parallel multi-tile encoding.
    fn make_tile_encoder(&self, tr: TileRect, use_deltaq: bool) -> Encoder {
        let t_mi_rows = tr.info.mi_row_end - tr.info.mi_row_start;
        let t_mi_cols = tr.info.mi_col_end - tr.info.mi_col_start;
        let t_sb_rows = tr.sb_row_end - tr.sb_row_start;
        let t_sb_cols = tr.sb_col_end - tr.sb_col_start;
        let aw = round_up(t_mi_cols * 4, 64) + 64;
        let ah = round_up(t_mi_rows * 4, 64) + 64;
        let mut src = Vec::with_capacity(self.num_planes);
        let mut rec = Vec::with_capacity(self.num_planes);
        for p in 0..self.num_planes {
            let (sx, sy) = if p == 0 { (0, 0) } else { (self.ssx, self.ssy) };
            let x0 = (tr.info.mi_col_start * 4) >> sx;
            let y0 = (tr.info.mi_row_start * 4) >> sy;
            let tw = ((t_mi_cols * 4) >> sx).min(self.src[p].width.saturating_sub(x0)).max(1);
            let th = ((t_mi_rows * 4) >> sy).min(self.src[p].height.saturating_sub(y0)).max(1);
            let full_w = (t_mi_cols * 4) >> sx;
            let full_h = (t_mi_rows * 4) >> sy;
            // Copy from padded `self.src[p]` (which already has edge replication
            // into its padding) into tile-local `src`.
            let sp = &self.src[p];
            let mut tp = Plane::new(tw, th, aw >> sx, ah >> sy);
            for y in 0..tp.alloc_h {
                let sy_idx = (y0 + y.min(full_h.saturating_sub(1))).min(sp.alloc_h - 1);
                let src_row = &sp.data[sy_idx * sp.stride + x0..sy_idx * sp.stride + x0 + full_w];
                let dst_row = &mut tp.data[y * tp.stride..(y + 1) * tp.stride];
                dst_row[..full_w].copy_from_slice(src_row);
                let last = src_row[full_w - 1];
                dst_row[full_w..].fill(last);
            }
            src.push(tp);
            rec.push(Plane::new(tw, th, aw >> sx, ah >> sy));
        }
        let mut sb_qindex = Vec::new();
        if use_deltaq {
            let frame_sb_cols = self.mi_cols.div_ceil(16);
            sb_qindex.reserve(t_sb_rows * t_sb_cols);
            for sr in tr.sb_row_start..tr.sb_row_end {
                let o = sr * frame_sb_cols + tr.sb_col_start;
                sb_qindex.extend_from_slice(&self.sb_qindex[o..o + t_sb_cols]);
            }
        }
        let mut ssim_scale = Vec::new();
        let mut ssim_cols = 0;
        if !self.ssim_scale.is_empty() {
            let r0 = tr.info.mi_row_start / 4;
            let r1 = tr.info.mi_row_end.div_ceil(4);
            let c0 = tr.info.mi_col_start / 4;
            let c1 = tr.info.mi_col_end.div_ceil(4);
            ssim_cols = c1 - c0;
            ssim_scale.reserve((r1 - r0) * ssim_cols);
            for r in r0..r1 {
                let o = r * self.ssim_cols + c0;
                ssim_scale.extend_from_slice(&self.ssim_scale[o..o + ssim_cols]);
            }
        }
        let cdf = CdfContext::new(self.cfg.base_q);
        let coef_costs = CoefCosts::new(&cdf.coef);
        let mode_costs = ModeCosts::new(&cdf);
        let mut cfg = self.cfg.clone();
        cfg.threads = 1;
        Encoder {
            mi_rows: t_mi_rows,
            mi_cols: t_mi_cols,
            num_planes: self.num_planes,
            ssx: self.ssx,
            ssy: self.ssy,
            src,
            rec,
            mi: vec![MiInfo::default(); t_mi_rows * t_mi_cols],
            tile: TileInfo { mi_row_start: 0, mi_row_end: t_mi_rows, mi_col_start: 0, mi_col_end: t_mi_cols },
            ctx: Contexts::new(t_mi_cols, t_mi_rows),
            cdf: Some(cdf),
            coef_costs,
            mode_costs,
            lossless: self.lossless,
            sharpness: self.sharpness,
            qm_dist: self.qm_dist,
            quants: self.quants.clone(),
            sb_qindex,
            ssim_scale,
            ssim_cols,
            delta_q_res: self.delta_q_res,
            cdef_idx: Vec::new(),
            cdef_bits: 0,
            cfg,
        }
    }

    /// Encode the frame and return the OBUs (sequence header + frame), keeping
    /// the final reconstructed planes in `self.rec`.
    pub fn encode(&mut self, seq: &obu::SequenceHeader) -> Vec<u8> {
        self.encode_internal(seq, true)
    }

    /// Encode the frame and return only the OBUs, skipping the CDEF apply pass
    /// and dropping internal pixel buffers as early as possible.
    pub fn encode_obu(&mut self, seq: &obu::SequenceHeader) -> Vec<u8> {
        self.encode_internal(seq, false)
    }

    fn encode_internal(&mut self, seq: &obu::SequenceHeader, need_recon: bool) -> Vec<u8> {
        search::prof::refresh();
        let tune = self.cfg.tune_iq;
        if tune && !self.lossless {
            self.compute_ssim_scaling();
        }
        let use_deltaq = self.cfg.deltaq_variance_boost && self.cfg.base_q > 0 && !self.lossless;
        if use_deltaq {
            self.compute_variance_boost();
            let q = self.cfg.base_q;
            self.delta_q_res = if q >= 160 {
                8
            } else if q >= 120 {
                4
            } else if q >= 80 {
                2
            } else {
                1
            };
        }
        let sb_cols = self.mi_cols.div_ceil(16);
        let sb_rows = self.mi_rows.div_ceil(16);
        let (tile_cols_log2, tile_rows_log2) = choose_tile_grid(self.mi_cols, self.mi_rows, self.cfg.threads);
        let tiles = compute_tiles(self.mi_cols, self.mi_rows, tile_cols_log2, tile_rows_log2);

        let tile_decs: Vec<FrameDecisions> = if tiles.len() == 1 {
            vec![self.encode_tile_sbs(sb_rows, sb_cols, use_deltaq, 0, 0)]
        } else {
            let thr = self.cfg.threads;
            let results: Vec<(FrameDecisions, Vec<Plane>, Vec<MiInfo>)> = {
                let ee: &Encoder = self;
                crate::parallel::par_map(tiles.len(), thr, |ti| {
                    let tr = tiles[ti];
                    let mut te = ee.make_tile_encoder(tr, use_deltaq);
                    let decs = te.encode_tile_sbs(
                        tr.sb_row_end - tr.sb_row_start,
                        tr.sb_col_end - tr.sb_col_start,
                        use_deltaq,
                        tr.info.mi_row_start as u16,
                        tr.info.mi_col_start as u16,
                    );
                    (decs, te.rec, te.mi)
                })
            };
            let mut decs_all = Vec::with_capacity(tiles.len());
            for (ti, (decs, trec, tmi)) in results.into_iter().enumerate() {
                let tr = tiles[ti];
                let t_mi_rows = tr.info.mi_row_end - tr.info.mi_row_start;
                let t_mi_cols = tr.info.mi_col_end - tr.info.mi_col_start;
                for r in 0..t_mi_rows {
                    let dst_o = (tr.info.mi_row_start + r) * self.mi_cols + tr.info.mi_col_start;
                    let src_o = r * t_mi_cols;
                    self.mi[dst_o..dst_o + t_mi_cols].copy_from_slice(&tmi[src_o..src_o + t_mi_cols]);
                }
                for p in 0..self.num_planes {
                    let (sx, sy) = if p == 0 { (0, 0) } else { (self.ssx, self.ssy) };
                    let x0 = (tr.info.mi_col_start * 4) >> sx;
                    let y0 = (tr.info.mi_row_start * 4) >> sy;
                    let w = (t_mi_cols * 4) >> sx;
                    let h = (t_mi_rows * 4) >> sy;
                    let dst = &mut self.rec[p];
                    let src = &trec[p];
                    for y in 0..h {
                        dst.data[(y0 + y) * dst.stride + x0..(y0 + y) * dst.stride + x0 + w]
                            .copy_from_slice(&src.data[y * src.stride..y * src.stride + w]);
                    }
                }
                decs_all.push(decs);
            }
            decs_all
        };
        // Drop search-only tables as soon as SB search finishes.
        self.sb_qindex = Vec::new();
        self.ssim_scale = Vec::new();

        // Loop filtering (across full frame, including tile boundaries)
        let t0 = search::prof::start();
        let lf = if self.lossless {
            obu::LoopFilterParams { level: [0; 4], sharpness: 0 }
        } else {
            crate::lf::pick_and_apply(self)
        };
        search::prof::add(6, t0);
        let t0 = search::prof::start();
        let cdef = if self.cfg.enable_cdef && !self.lossless {
            Some(crate::cdef::pick_and_apply(self, need_recon).unwrap_or(obu::CdefParams {
                damping: 3,
                bits: 0,
                strengths: vec![(0, 0, 0, 0)],
            }))
        } else {
            None
        };
        // Source pixels are no longer needed after CDEF.
        self.src = Vec::new();
        if !need_recon {
            self.rec = Vec::new();
        }

        search::prof::add(7, t0);
        search::prof::report();
        // Bitstream
        let mut out = Vec::new();
        obu::obu(obu::OBU_SEQUENCE_HEADER, &seq.write(), &mut out);
        let fh = FrameHeader {
            mi_cols: self.mi_cols,
            mi_rows: self.mi_rows,
            num_planes: self.num_planes,
            base_q_idx: self.cfg.base_q,
            delta_q_y_dc: 0,
            delta_q_u_dc: self.cfg.uv_delta_q.0,
            delta_q_u_ac: self.cfg.uv_delta_q.1,
            qm: self.cfg.qm,
            delta_q_res_log2: if use_deltaq { Some(self.delta_q_res.trailing_zeros() as u8) } else { None },
            lossless: self.lossless,
            lf,
            cdef,
            tile_cols_log2,
            tile_rows_log2,
            screen_content: false,
        };
        let mut bw = BitWriter::default();
        fh.write(&mut bw);
        bw.byte_align();
        let mut payload = bw.buf;
        let num_tiles = tiles.len();
        if num_tiles > 1 {
            // tile_group_obu: tile_start_and_end_present_flag = 0, then byte_alignment().
            payload.push(0);
        }
        let mut wscratch = write::WriteScratch::new();
        let mut wctx = Contexts::new(self.mi_cols, self.mi_rows);
        let trace = crate::debug_flag("AV1_TRACE");
        for (ti, tr) in tiles.iter().enumerate() {
            self.tile = tr.info;
            let mut cdf = CdfContext::new(self.cfg.base_q);
            for p in 0..3 {
                wctx.above_ent[p].fill(0);
            }
            let mut sw = SymbolWriter::new();
            let mut cur_q = self.cfg.base_q;
            let decs = &tile_decs[ti];
            let mut i = 0usize;
            for sr in tr.sb_row_start..tr.sb_row_end {
                for p in 0..3 {
                    wctx.left_ent[p].fill(0);
                }
                for sc in tr.sb_col_start..tr.sb_col_end {
                    let dq = if use_deltaq { Some(&mut cur_q) } else { None };
                    let cdef_idx = if fh.cdef.is_some() {
                        Some(self.cdef_idx.get(sr * sb_cols + sc).copied().unwrap_or(0))
                    } else {
                        None
                    };
                    let view = decs.sb_view(i);
                    if trace {
                        let mut t = crate::ec::Tracing(&mut sw);
                        self.write_sb(
                            &mut t,
                            &mut cdf,
                            &mut wctx,
                            &view,
                            sr * 16,
                            sc * 16,
                            dq,
                            cdef_idx,
                            &mut wscratch,
                        );
                    } else {
                        self.write_sb(
                            &mut sw,
                            &mut cdf,
                            &mut wctx,
                            &view,
                            sr * 16,
                            sc * 16,
                            dq,
                            cdef_idx,
                            &mut wscratch,
                        );
                    }
                    i += 1;
                }
            }
            let tile_bytes = sw.finish();
            if ti + 1 < num_tiles {
                // tile_size_minus_1 (4 bytes little-endian; tile_size_bytes_minus_1 == 3).
                assert!(
                    !tile_bytes.is_empty() && tile_bytes.len() - 1 <= u32::MAX as usize,
                    "tile size must fit the 4-byte tile_size_minus_1"
                );
                let sz_minus_1 = (tile_bytes.len() - 1) as u32;
                payload.extend_from_slice(&sz_minus_1.to_le_bytes());
            }
            payload.extend_from_slice(&tile_bytes);
        }
        // Drop all coding decisions before assembling the final OBU.
        drop(tile_decs);
        obu::obu(obu::OBU_FRAME, &payload, &mut out);
        out
    }

    /// libaom's all-intra `intra_sb_rdmult_modifier` (/128): lowers lambda in
    /// superblocks that mix flat and detailed 4x4 areas.
    fn sb_rdmult_modifier(&self, mi_row: usize, mi_col: usize) -> i64 {
        if crate::debug_flag("AV1_NO_SBMOD") {
            return 128;
        }
        let src = &self.src[0];
        let bh = (self.mi_rows - mi_row).min(16);
        let bw = (self.mi_cols - mi_col).min(16);
        let (mut vmin, mut vmax) = (u64::MAX, 0u64);
        for r in 0..bh {
            for c in 0..bw {
                let v = block_variance(src, (mi_col + c) * 4, (mi_row + r) * 4, 4);
                vmin = vmin.min(v);
                vmax = vmax.max(v);
            }
        }
        let lmin = (vmin as f64 / 16.0).ln_1p();
        let lmax = (vmax as f64 / 16.0).ln_1p();
        let mut m = 128;
        if lmin < 2.0 && lmax > 4.0 {
            m -= if lmax - lmin > 8.0 { 48 } else { ((lmax - lmin) * 6.0) as i64 };
        }
        m
    }

    fn encode_sb(
        &mut self,
        s: &mut search::SearchState,
        mi_row: usize,
        mi_col: usize,
        qindex: u8,
        dec: &mut Decisions,
    ) {
        s.sb_q = qindex;
        s.quant = self.quant(qindex);
        s.rdmult = self.rd_mult(qindex) * self.sb_rdmult_modifier(mi_row, mi_col) / 128;
        self.rd_pick_partition(s, BlockSize::B64x64, mi_row, mi_col, dec, i64::MAX);
        // The SB qindex of each block is the same; stored for the writer.
        s.last_sb_q = qindex;
    }
}

/// Sum of squared deviations of an n x n block (libaom variance with a zero reference).
pub fn block_variance(p: &Plane, x: usize, y: usize, n: usize) -> u64 {
    let mut sum: i64 = 0;
    let mut sse: i64 = 0;
    for r in 0..n {
        let row = &p.data[(y + r) * p.stride + x..(y + r) * p.stride + x + n];
        for &v in row {
            sum += v as i64;
            sse += (v as i64) * (v as i64);
        }
    }
    (sse - (sum * sum) / (n * n) as i64) as u64
}
