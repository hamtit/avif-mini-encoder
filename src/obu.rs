//! Sequence header and frame header OBUs for a reduced still picture.

use crate::ec::{BitWriter, leb128};

pub const OBU_SEQUENCE_HEADER: u8 = 1;
pub const OBU_FRAME: u8 = 6;

pub fn obu(kind: u8, payload: &[u8], out: &mut Vec<u8>) {
    out.push((kind << 3) | 0x02);
    leb128(payload.len(), out);
    out.extend_from_slice(payload);
}

#[derive(Clone, Copy, Debug)]
pub struct ColorDesc {
    pub primaries: u8,
    pub transfer: u8,
    pub matrix: u8,
}

pub struct SequenceHeader {
    pub profile: u8,
    pub level: u8,
    pub width: usize,
    pub height: usize,
    pub mono: bool,
    /// 4:2:0 when true (profile 0), otherwise 4:4:4 (profile 1) or mono.
    pub subsampled: bool,
    pub color: Option<ColorDesc>,
    pub full_range: bool,
    pub enable_cdef: bool,
}

fn bits_for(v: usize) -> u32 {
    (usize::BITS - v.leading_zeros()).max(1)
}

impl SequenceHeader {
    pub fn write(&self) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.put(self.profile as u64, 3);
        w.flag(true); // still_picture
        w.flag(true); // reduced_still_picture_header
        w.put(self.level as u64, 5);
        let wb = bits_for(self.width - 1);
        let hb = bits_for(self.height - 1);
        w.put((wb - 1) as u64, 4);
        w.put((hb - 1) as u64, 4);
        w.put((self.width - 1) as u64, wb);
        w.put((self.height - 1) as u64, hb);
        w.flag(false); // use_128x128_superblock
        w.flag(true); // enable_filter_intra
        w.flag(true); // enable_intra_edge_filter
        w.flag(false); // enable_superres
        w.flag(self.enable_cdef);
        w.flag(false); // enable_restoration
        // color_config
        w.flag(false); // high_bitdepth
        if self.profile != 1 {
            w.flag(self.mono);
        }
        w.flag(self.color.is_some());
        if let Some(c) = self.color {
            w.put(c.primaries as u64, 8);
            w.put(c.transfer as u64, 8);
            w.put(c.matrix as u64, 8);
        }
        let srgb_identity =
            !self.mono && matches!(self.color, Some(ColorDesc { primaries: 1, transfer: 13, matrix: 0 }));
        if srgb_identity {
            // color_range is implied (full) and subsampling is 4:4:4.
            debug_assert!(!self.subsampled && self.full_range);
        } else {
            w.flag(self.full_range);
            if !self.mono && self.subsampled {
                w.put(0, 2); // chroma_sample_position: unknown
            }
        }
        if !self.mono {
            w.flag(false); // separate_uv_delta_q
        }
        w.flag(false); // film_grain_params_present
        w.trailing_bits();
        w.buf
    }
}

pub struct LoopFilterParams {
    /// [y vertical, y horizontal, u, v]
    pub level: [u8; 4],
    pub sharpness: u8,
}

pub struct CdefParams {
    pub damping: u8,
    pub bits: u8,
    /// (y_pri, y_sec, uv_pri, uv_sec) per index; sec in 0..=3 coded form.
    pub strengths: Vec<(u8, u8, u8, u8)>,
}

pub struct FrameHeader {
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub num_planes: usize,
    pub base_q_idx: u8,
    pub delta_q_y_dc: i32,
    pub delta_q_u_dc: i32,
    pub delta_q_u_ac: i32,
    /// (qm_y, qm_u) or None for no quantization matrices.
    pub qm: Option<(u8, u8)>,
    /// log2 of delta_q_res when delta q is used.
    pub delta_q_res_log2: Option<u8>,
    pub lossless: bool,
    pub lf: LoopFilterParams,
    pub cdef: Option<CdefParams>,
    pub tile_cols_log2: u32,
    pub tile_rows_log2: u32,
    pub screen_content: bool,
}

pub fn tile_log2(blk: usize, target: usize) -> u32 {
    let mut k = 0;
    while (blk << k) < target {
        k += 1;
    }
    k
}

pub struct TileLimits {
    pub min_cols_log2: u32,
    pub max_cols_log2: u32,
    pub max_rows_log2: u32,
    pub min_tiles_log2: u32,
}

pub fn tile_limits(mi_cols: usize, mi_rows: usize) -> TileLimits {
    let sb_cols = mi_cols.div_ceil(16);
    let sb_rows = mi_rows.div_ceil(16);
    let max_tile_width_sb = 4096 >> 6;
    let max_tile_area_sb = (4096 * 2304) >> 12;
    let min_cols_log2 = tile_log2(max_tile_width_sb, sb_cols);
    TileLimits {
        min_cols_log2,
        max_cols_log2: tile_log2(1, sb_cols.min(64)),
        max_rows_log2: tile_log2(1, sb_rows.min(64)),
        min_tiles_log2: min_cols_log2.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols)),
    }
}

fn write_delta_q(w: &mut BitWriter, v: i32) {
    w.flag(v != 0);
    if v != 0 {
        w.su(v, 6);
    }
}

impl FrameHeader {
    /// uncompressed_header() for a shown key frame of a reduced still picture.
    pub fn write(&self, w: &mut BitWriter) {
        w.flag(false); // disable_cdf_update
        w.flag(self.screen_content); // allow_screen_content_tools
        if self.screen_content {
            w.flag(false); // force_integer_mv (forced to 1 for intra frames anyway)
        }
        w.flag(false); // render_and_frame_size_different
        if self.screen_content {
            w.flag(false); // allow_intrabc
        }
        // tile_info
        let lim = tile_limits(self.mi_cols, self.mi_rows);
        w.flag(true); // uniform_tile_spacing_flag
        let mut log2 = lim.min_cols_log2;
        while log2 < lim.max_cols_log2 {
            let inc = log2 < self.tile_cols_log2;
            w.flag(inc);
            if !inc {
                break;
            }
            log2 += 1;
        }
        let min_rows_log2 = lim.min_tiles_log2.saturating_sub(self.tile_cols_log2);
        let mut log2 = min_rows_log2;
        while log2 < lim.max_rows_log2 {
            let inc = log2 < self.tile_rows_log2;
            w.flag(inc);
            if !inc {
                break;
            }
            log2 += 1;
        }
        if self.tile_cols_log2 > 0 || self.tile_rows_log2 > 0 {
            w.put(0, self.tile_cols_log2 + self.tile_rows_log2); // context_update_tile_id
            w.put(3, 2); // tile_size_bytes_minus_1 (4 bytes)
        }
        // quantization_params
        w.put(self.base_q_idx as u64, 8);
        write_delta_q(w, self.delta_q_y_dc);
        if self.num_planes > 1 {
            write_delta_q(w, self.delta_q_u_dc);
            write_delta_q(w, self.delta_q_u_ac);
        }
        w.flag(self.qm.is_some());
        if let Some((y, u)) = self.qm {
            w.put(y as u64, 4);
            w.put(u as u64, 4);
        }
        w.flag(false); // segmentation_enabled
        if self.base_q_idx > 0 {
            w.flag(self.delta_q_res_log2.is_some());
            if let Some(r) = self.delta_q_res_log2 {
                w.put(r as u64, 2);
                w.flag(false); // delta_lf_present
            }
        }
        if !self.lossless {
            // loop_filter_params
            w.put(self.lf.level[0] as u64, 6);
            w.put(self.lf.level[1] as u64, 6);
            if self.num_planes > 1 && (self.lf.level[0] != 0 || self.lf.level[1] != 0) {
                w.put(self.lf.level[2] as u64, 6);
                w.put(self.lf.level[3] as u64, 6);
            }
            w.put(self.lf.sharpness as u64, 3);
            w.flag(false); // loop_filter_delta_enabled
            if let Some(c) = &self.cdef {
                w.put((c.damping - 3) as u64, 2);
                w.put(c.bits as u64, 2);
                for s in &c.strengths {
                    w.put(s.0 as u64, 4);
                    w.put(s.1 as u64, 2);
                    if self.num_planes > 1 {
                        w.put(s.2 as u64, 4);
                        w.put(s.3 as u64, 2);
                    }
                }
            }
            w.flag(true); // tx_mode_select
        }
        w.flag(false); // reduced_tx_set
    }
}
