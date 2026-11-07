//! Constant tables taken from libaom 3.15.1 (BSD-2-Clause, see LICENSE).
//! `cdf_default.rs`, `misc.rs`, `sizes.rs`, `qm_offsets.rs` and `qm.bin` are
//! extracted from the libaom 3.15.1 sources (BSD-2-Clause).

// Generated/extracted tables stay in their source layout (one row per line).
#[rustfmt::skip]
#[allow(non_upper_case_globals, dead_code)]
pub mod cdf_default;
#[rustfmt::skip]
#[allow(dead_code)]
pub mod misc;
#[rustfmt::skip]
#[allow(dead_code)]
pub mod sizes;
#[rustfmt::skip]
mod qm_offsets;

/// Quantization matrices: `[level 0..15][plane type][3344]` weights followed
/// by the same layout of inverse weights.
static QM_BIN: &[u8; 200640] = include_bytes!("qm.bin");
const QM_LEVEL_SIZE: usize = 3344;

/// Forward (quantizer) weights for a QM level < 15, plane type and transform size.
pub fn qm(level: usize, chroma: bool, tx_size: usize) -> &'static [u8] {
    let base = (level * 2 + chroma as usize) * QM_LEVEL_SIZE + qm_offsets::QM_TX_OFFSET[tx_size] as usize;
    &QM_BIN[base..]
}

/// Inverse (dequantizer) weights.
pub fn iqm(level: usize, chroma: bool, tx_size: usize) -> &'static [u8] {
    let base = 15 * 2 * QM_LEVEL_SIZE
        + (level * 2 + chroma as usize) * QM_LEVEL_SIZE
        + qm_offsets::QM_TX_OFFSET[tx_size] as usize;
    &QM_BIN[base..]
}

use crate::consts::{TX_CLASS_HORIZ, TX_CLASS_VERT, TxSize};

/// Scan order of a transform size (coded size for 64-point) and class.
pub fn scan(tx: TxSize, tx_class: u8) -> &'static [u16] {
    let k = match tx_class {
        TX_CLASS_VERT => 1,
        TX_CLASS_HORIZ => 2,
        _ => 0,
    };
    SCANS[tx as usize][k]
}

/// libaom's av1_nz_map_ctx_offset for a transform size.
pub fn nz_map_ctx_offset(tx: TxSize) -> &'static [i8] {
    NZ_MAP[tx as usize]
}

static SCANS: [[&[u16]; 3]; 19] = [
    [&misc::SCAN_0_0, &misc::SCAN_0_1, &misc::SCAN_0_2],
    [&misc::SCAN_1_0, &misc::SCAN_1_1, &misc::SCAN_1_2],
    [&misc::SCAN_2_0, &misc::SCAN_2_1, &misc::SCAN_2_2],
    [&misc::SCAN_3_0, &misc::SCAN_3_1, &misc::SCAN_3_2],
    [&misc::SCAN_4_0, &misc::SCAN_4_1, &misc::SCAN_4_2],
    [&misc::SCAN_5_0, &misc::SCAN_5_1, &misc::SCAN_5_2],
    [&misc::SCAN_6_0, &misc::SCAN_6_1, &misc::SCAN_6_2],
    [&misc::SCAN_7_0, &misc::SCAN_7_1, &misc::SCAN_7_2],
    [&misc::SCAN_8_0, &misc::SCAN_8_1, &misc::SCAN_8_2],
    [&misc::SCAN_9_0, &misc::SCAN_9_1, &misc::SCAN_9_2],
    [&misc::SCAN_10_0, &misc::SCAN_10_1, &misc::SCAN_10_2],
    [&misc::SCAN_11_0, &misc::SCAN_11_1, &misc::SCAN_11_2],
    [&misc::SCAN_12_0, &misc::SCAN_12_1, &misc::SCAN_12_2],
    [&misc::SCAN_13_0, &misc::SCAN_13_1, &misc::SCAN_13_2],
    [&misc::SCAN_14_0, &misc::SCAN_14_1, &misc::SCAN_14_2],
    [&misc::SCAN_15_0, &misc::SCAN_15_1, &misc::SCAN_15_2],
    [&misc::SCAN_16_0, &misc::SCAN_16_1, &misc::SCAN_16_2],
    [&misc::SCAN_17_0, &misc::SCAN_17_1, &misc::SCAN_17_2],
    [&misc::SCAN_18_0, &misc::SCAN_18_1, &misc::SCAN_18_2],
];
static NZ_MAP: [&[i8]; 19] = [
    &misc::NZ_MAP_CTX_OFFSET_0,
    &misc::NZ_MAP_CTX_OFFSET_1,
    &misc::NZ_MAP_CTX_OFFSET_2,
    &misc::NZ_MAP_CTX_OFFSET_3,
    &misc::NZ_MAP_CTX_OFFSET_4,
    &misc::NZ_MAP_CTX_OFFSET_5,
    &misc::NZ_MAP_CTX_OFFSET_6,
    &misc::NZ_MAP_CTX_OFFSET_7,
    &misc::NZ_MAP_CTX_OFFSET_8,
    &misc::NZ_MAP_CTX_OFFSET_9,
    &misc::NZ_MAP_CTX_OFFSET_10,
    &misc::NZ_MAP_CTX_OFFSET_11,
    &misc::NZ_MAP_CTX_OFFSET_12,
    &misc::NZ_MAP_CTX_OFFSET_13,
    &misc::NZ_MAP_CTX_OFFSET_14,
    &misc::NZ_MAP_CTX_OFFSET_15,
    &misc::NZ_MAP_CTX_OFFSET_16,
    &misc::NZ_MAP_CTX_OFFSET_17,
    &misc::NZ_MAP_CTX_OFFSET_18,
];
#[allow(dead_code)]
pub mod edge;
