//! Availability of top-right / bottom-left neighbours for intra prediction
//! (libaom's has_top_right / has_bottom_left), for 64x64 superblocks.

use crate::consts::{BlockSize, Partition, TxSize};
use crate::tables::edge::{HAS_BL_TABLES, HAS_BL_VERT_TABLES, HAS_TR_TABLES, HAS_TR_VERT_TABLES};

const SB_MI: usize = 16;
const MAX_MIB_SIZE_LOG2: usize = 5;

/// libaom's scale_chroma_bsize for 4:2:0 / 4:4:4.
pub fn scale_chroma_bsize(b: BlockSize, ssx: usize, ssy: usize) -> BlockSize {
    use BlockSize::*;
    match (b, ssx, ssy) {
        (B4x4, 1, 1) => B8x8,
        (B4x4, 1, 0) => B8x4,
        (B4x4, 0, 1) => B4x8,
        (B4x8, 1, _) => B8x8,
        (B8x4, _, 1) => B8x8,
        (B4x16, 1, 1) => B8x16,
        (B4x16, 1, 0) => B8x16,
        (B16x4, 1, 1) => B16x8,
        (B16x4, 0, 1) => B16x8,
        _ => b,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn has_top_right(
    bsize: BlockSize,
    mi_row: usize,
    mi_col: usize,
    top_available: bool,
    right_available: bool,
    partition: Partition,
    tx: TxSize,
    row_off: usize,
    col_off: usize,
    ssx: usize,
    _ssy: usize,
) -> bool {
    if !top_available || !right_available {
        return false;
    }
    let bw_unit = bsize.w4();
    let plane_bw_unit = (bw_unit >> ssx).max(1);
    let tr_count = tx.w4();
    if row_off > 0 {
        return col_off + tr_count < plane_bw_unit;
    }
    if col_off + tr_count < plane_bw_unit {
        return true;
    }
    let bw_log2 = bsize.w4_log2();
    let bh_log2 = bsize.h4_log2();
    let blk_row_in_sb = (mi_row & (SB_MI - 1)) >> bh_log2;
    let blk_col_in_sb = (mi_col & (SB_MI - 1)) >> bw_log2;
    if blk_row_in_sb == 0 {
        return true;
    }
    if ((blk_col_in_sb + 1) << bw_log2) >= SB_MI {
        return false;
    }
    let idx = (blk_row_in_sb << (MAX_MIB_SIZE_LOG2 - bw_log2)) + blk_col_in_sb;
    let table = if matches!(partition, Partition::VertA | Partition::VertB) {
        HAS_TR_VERT_TABLES[bsize as usize]
    } else {
        HAS_TR_TABLES[bsize as usize]
    }
    .expect("has_tr table");
    (table[idx / 8] >> (idx % 8)) & 1 != 0
}

#[allow(clippy::too_many_arguments)]
pub fn has_bottom_left(
    bsize: BlockSize,
    mi_row: usize,
    mi_col: usize,
    bottom_available: bool,
    left_available: bool,
    partition: Partition,
    tx: TxSize,
    row_off: usize,
    col_off: usize,
    _ssx: usize,
    ssy: usize,
) -> bool {
    if !bottom_available || !left_available {
        return false;
    }
    if col_off > 0 {
        return false;
    }
    let bh_unit = bsize.h4();
    let plane_bh_unit = (bh_unit >> ssy).max(1);
    let bl_count = tx.h4();
    if row_off + bl_count < plane_bh_unit {
        return true;
    }
    let bw_log2 = bsize.w4_log2();
    let bh_log2 = bsize.h4_log2();
    let blk_row_in_sb = (mi_row & (SB_MI - 1)) >> bh_log2;
    let blk_col_in_sb = (mi_col & (SB_MI - 1)) >> bw_log2;
    if blk_col_in_sb == 0 {
        let blk_start_row_off = (blk_row_in_sb << bh_log2) >> ssy;
        let row_off_in_sb = blk_start_row_off + row_off;
        let sb_height_unit = SB_MI >> ssy;
        return row_off_in_sb + bl_count < sb_height_unit;
    }
    if ((blk_row_in_sb + 1) << bh_log2) >= SB_MI {
        return false;
    }
    let idx = (blk_row_in_sb << (MAX_MIB_SIZE_LOG2 - bw_log2)) + blk_col_in_sb;
    let table = if matches!(partition, Partition::VertA | Partition::VertB) {
        HAS_BL_VERT_TABLES[bsize as usize]
    } else {
        HAS_BL_TABLES[bsize as usize]
    }
    .expect("has_bl table");
    (table[idx / 8] >> (idx % 8)) & 1 != 0
}
