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
    pub coefs: Vec<i32>,
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
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TileInfo {
    pub mi_row_start: usize,
    pub mi_row_end: usize,
    pub mi_col_start: usize,
    pub mi_col_end: usize,
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
        Contexts {
            above_ent: [vec![0; w], vec![0; w], vec![0; w]],
            left_ent: [vec![0; h], vec![0; h], vec![0; h]],
        }
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
    /// Worker threads for the stages that can run in parallel (CDEF strength
    /// search). The output does not depend on this value.
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
    pub cdf: Box<CdfContext>,
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
            cdf,
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
        let q = [
            PlaneQuant::new(qindex as i32, 0, 0, qy, false),
            PlaneQuant::new(qindex as i32, dc, ac, quv, true),
            PlaneQuant::new(qindex as i32, dc, ac, quv, true),
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
        let mut prod = 1.0f64;
        let mut n = 0.0;
        for r in r0..(r0 + nr).min(rows) {
            for c in c0..(c0 + nc).min(self.ssim_cols) {
                prod *= self.ssim_scale[r * self.ssim_cols + c];
                n += 1.0;
            }
        }
        let g = prod.powf(1.0 / n);
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

    /// Encode the frame and return the OBUs (sequence header + frame).
    pub fn encode(&mut self, seq: &obu::SequenceHeader) -> Vec<u8> {
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
        let mut sbs: Vec<Decisions> = Vec::with_capacity(sb_cols * sb_rows);
        let mut wctx = Contexts::new(self.mi_cols, self.mi_rows);
        let mut cur_q = self.cfg.base_q;
        let mut replay_q = self.cfg.base_q;
        let mut search = search::SearchState::new();
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
                let mut dec = Decisions::default();
                self.encode_sb(&mut search, mi_row, mi_col, sb_q, &mut dec);
                let first = &dec.blocks[0];
                if use_deltaq && !(first.bsize == BlockSize::B64x64 && first.skip) {
                    cur_q = sb_q;
                }
                dec.qindex = sb_q;
                // Keep the CDFs in step with the bitstream.
                let mut cdf = std::mem::replace(&mut self.cdf, CdfContext::new(0));
                let dq = if use_deltaq { Some(&mut replay_q) } else { None };
                self.write_sb(&mut CdfOnly, &mut cdf, &mut wctx, &dec, mi_row, mi_col, dq, None);
                self.cdf = cdf;
                self.coef_costs.update(&self.cdf.coef);
                self.mode_costs.update(&self.cdf);
                sbs.push(dec);
            }
        }

        // Loop filtering
        let t0 = std::time::Instant::now();
        let lf = if self.lossless {
            obu::LoopFilterParams { level: [0; 4], sharpness: 0 }
        } else {
            crate::lf::pick_and_apply(self)
        };
        search::prof::add(6, t0);
        let t0 = std::time::Instant::now();
        let cdef = if self.cfg.enable_cdef && !self.lossless {
            Some(crate::cdef::pick_and_apply(self, &sbs).unwrap_or(obu::CdefParams {
                damping: 3,
                bits: 0,
                strengths: vec![(0, 0, 0, 0)],
            }))
        } else {
            None
        };

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
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            screen_content: false,
        };
        let mut bw = BitWriter::default();
        fh.write(&mut bw);
        bw.byte_align();
        let mut payload = bw.buf;
        // Tile data (single tile)
        let mut cdf = CdfContext::new(self.cfg.base_q);
        let mut wctx = Contexts::new(self.mi_cols, self.mi_rows);
        let mut sw = SymbolWriter::new();
        let mut cur_q = self.cfg.base_q;
        let mut i = 0;
        for sr in 0..sb_rows {
            for p in 0..3 {
                wctx.left_ent[p].fill(0);
            }
            for sc in 0..sb_cols {
                let dq = if use_deltaq { Some(&mut cur_q) } else { None };
                let cdef_idx = if fh.cdef.is_some() { Some(self.cdef_idx.get(sr * sb_cols + sc).copied().unwrap_or(0)) } else { None };
                if crate::debug_flag("AV1_TRACE") {
                    let mut t = crate::ec::Tracing(&mut sw);
                    self.write_sb(&mut t, &mut cdf, &mut wctx, &sbs[i], sr * 16, sc * 16, dq, cdef_idx);
                } else {
                    self.write_sb(&mut sw, &mut cdf, &mut wctx, &sbs[i], sr * 16, sc * 16, dq, cdef_idx);
                }
                i += 1;
            }
        }
        payload.extend_from_slice(&sw.finish());
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

    fn encode_sb(&mut self, s: &mut search::SearchState, mi_row: usize, mi_col: usize, qindex: u8, dec: &mut Decisions) {
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
