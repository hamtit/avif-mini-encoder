//! Block / transform sizes and prediction modes (numbering as in libaom).

use crate::tables::sizes as s;

#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum BlockSize {
    B4x4,
    B4x8,
    B8x4,
    B8x8,
    B8x16,
    B16x8,
    B16x16,
    B16x32,
    B32x16,
    B32x32,
    B32x64,
    B64x32,
    B64x64,
    B64x128,
    B128x64,
    B128x128,
    B4x16,
    B16x4,
    B8x32,
    B32x8,
    B16x64,
    B64x16,
}

impl BlockSize {
    pub const ALL: [BlockSize; 22] = {
        use BlockSize::*;
        [
            B4x4, B4x8, B8x4, B8x8, B8x16, B16x8, B16x16, B16x32, B32x16, B32x32, B32x64, B64x32, B64x64, B64x128,
            B128x64, B128x128, B4x16, B16x4, B8x32, B32x8, B16x64, B64x16,
        ]
    };

    #[inline]
    pub fn from_u8(v: u8) -> BlockSize {
        Self::ALL[v as usize]
    }
    /// Width in 4x4 units, log2.
    #[inline]
    pub fn w4_log2(self) -> usize {
        s::BW_LOG2[self as usize] as usize
    }
    #[inline]
    pub fn h4_log2(self) -> usize {
        s::BH_LOG2[self as usize] as usize
    }
    #[inline]
    pub fn w4(self) -> usize {
        1 << self.w4_log2()
    }
    #[inline]
    pub fn h4(self) -> usize {
        1 << self.h4_log2()
    }
    #[inline]
    pub fn width(self) -> usize {
        4 << self.w4_log2()
    }
    #[inline]
    pub fn height(self) -> usize {
        4 << self.h4_log2()
    }
    #[inline]
    pub fn pels_log2(self) -> usize {
        s::NUM_PELS_LOG2[self as usize] as usize
    }
    /// Largest rectangular transform that fits.
    #[inline]
    pub fn max_tx(self) -> TxSize {
        TxSize::from_u8(s::MAX_TXSIZE_RECT[self as usize])
    }
    /// Size of a sub-block after `partition`, for square sizes 8x8..64x64.
    #[inline]
    pub fn subsize(self, p: Partition) -> Option<BlockSize> {
        let sq = match self {
            BlockSize::B4x4 => 0,
            BlockSize::B8x8 => 1,
            BlockSize::B16x16 => 2,
            BlockSize::B32x32 => 3,
            BlockSize::B64x64 => 4,
            BlockSize::B128x128 => 5,
            _ => return None,
        };
        let v = s::SUBSIZE[p as usize][sq];
        (v != 255).then(|| BlockSize::from_u8(v))
    }
    /// Block size of a chroma plane with the given subsampling.
    #[inline]
    pub fn subsampled(self, ssx: usize, ssy: usize) -> BlockSize {
        BlockSize::from_u8(s::SS_SIZE[self as usize][ssx][ssy])
    }
    pub fn from_dims(w: usize, h: usize) -> BlockSize {
        *Self::ALL.iter().find(|b| b.width() == w && b.height() == h).expect("valid block size")
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum TxSize {
    T4x4,
    T8x8,
    T16x16,
    T32x32,
    T64x64,
    T4x8,
    T8x4,
    T8x16,
    T16x8,
    T16x32,
    T32x16,
    T32x64,
    T64x32,
    T4x16,
    T16x4,
    T8x32,
    T32x8,
    T16x64,
    T64x16,
}

impl TxSize {
    pub const ALL: [TxSize; 19] = {
        use TxSize::*;
        [
            T4x4, T8x8, T16x16, T32x32, T64x64, T4x8, T8x4, T8x16, T16x8, T16x32, T32x16, T32x64, T64x32, T4x16,
            T16x4, T8x32, T32x8, T16x64, T64x16,
        ]
    };
    #[inline]
    pub fn from_u8(v: u8) -> TxSize {
        Self::ALL[v as usize]
    }
    #[inline]
    pub fn w_log2(self) -> usize {
        s::TX_W_LOG2[self as usize] as usize
    }
    #[inline]
    pub fn h_log2(self) -> usize {
        s::TX_H_LOG2[self as usize] as usize
    }
    #[inline]
    pub fn width(self) -> usize {
        1 << self.w_log2()
    }
    #[inline]
    pub fn height(self) -> usize {
        1 << self.h_log2()
    }
    /// Width / height in 4-sample units.
    #[inline]
    pub fn w4(self) -> usize {
        self.width() >> 2
    }
    #[inline]
    pub fn h4(self) -> usize {
        self.height() >> 2
    }
    #[inline]
    pub fn sub(self) -> TxSize {
        TxSize::from_u8(s::SUB_TX_SIZE[self as usize])
    }
    #[inline]
    pub fn sqr(self) -> TxSize {
        TxSize::from_u8(s::TXSIZE_SQR[self as usize])
    }
    #[inline]
    pub fn sqr_up(self) -> TxSize {
        TxSize::from_u8(s::TXSIZE_SQR_UP[self as usize])
    }
    #[inline]
    pub fn block(self) -> BlockSize {
        BlockSize::from_u8(s::TXSIZE_TO_BSIZE[self as usize])
    }
    /// Size whose coefficients are coded (64-point dimensions become 32).
    #[inline]
    pub fn coded(self) -> TxSize {
        match self {
            TxSize::T64x64 | TxSize::T32x64 | TxSize::T64x32 => TxSize::T32x32,
            TxSize::T16x64 => TxSize::T16x32,
            TxSize::T64x16 => TxSize::T32x16,
            t => t,
        }
    }
    /// Context index for coefficient CDFs: (sqr + sqr_up + 1) / 2.
    #[inline]
    pub fn ctx(self) -> usize {
        (self.sqr() as usize + self.sqr_up() as usize).div_ceil(2)
    }
    pub fn from_dims(w: usize, h: usize) -> TxSize {
        *Self::ALL.iter().find(|t| t.width() == w && t.height() == h).expect("valid tx size")
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Partition {
    None,
    Horz,
    Vert,
    Split,
    HorzA,
    HorzB,
    VertA,
    VertB,
    Horz4,
    Vert4,
}

pub type TxType = u8;
pub const DCT_DCT: TxType = 0;
pub const ADST_DCT: TxType = 1;
pub const DCT_ADST: TxType = 2;
pub const ADST_ADST: TxType = 3;
pub const IDTX: TxType = 9;
#[allow(dead_code)]
pub const V_DCT: TxType = 10;
#[allow(dead_code)]
pub const H_DCT: TxType = 11;

/// Scan class of a transform type: 0 = 2-D, 1 = vertical-only (V_*),
/// 2 = horizontal-only (H_*). libaom's TX_CLASS ordering is 2D, HORIZ, VERT;
/// here we keep libaom's enum values.
pub const TX_CLASS_2D: u8 = 0;
pub const TX_CLASS_HORIZ: u8 = 1;
pub const TX_CLASS_VERT: u8 = 2;
pub const TX_TYPE_CLASS: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 2, 1];

/// Intra prediction modes.
pub const DC_PRED: u8 = 0;
pub const V_PRED: u8 = 1;
pub const H_PRED: u8 = 2;
pub const D45_PRED: u8 = 3;
#[allow(dead_code)]
pub const D135_PRED: u8 = 4;
#[allow(dead_code)]
pub const D113_PRED: u8 = 5;
pub const D157_PRED: u8 = 6;
pub const D203_PRED: u8 = 7;
pub const D67_PRED: u8 = 8;
pub const SMOOTH_PRED: u8 = 9;
pub const SMOOTH_V_PRED: u8 = 10;
pub const SMOOTH_H_PRED: u8 = 11;
#[allow(dead_code)]
pub const PAETH_PRED: u8 = 12;
pub const UV_CFL_PRED: u8 = 13;
pub const INTRA_MODES: usize = 13;

#[inline]
pub fn is_directional(mode: u8) -> bool {
    (V_PRED..=D67_PRED).contains(&mode)
}

pub const MODE_TO_ANGLE: [i32; 13] = [0, 90, 180, 45, 135, 113, 157, 203, 67, 0, 0, 0, 0];

/// Default transform type implied by an intra mode (libaom's intra_mode_to_tx_type).
pub const MODE_TO_TXFM: [TxType; 14] = [
    DCT_DCT,  // DC
    ADST_DCT, // V
    DCT_ADST, // H
    DCT_DCT,  // D45
    ADST_ADST, // D135
    ADST_DCT, // D113
    DCT_ADST, // D157
    DCT_ADST, // D203
    ADST_DCT, // D67
    ADST_ADST, // SMOOTH
    ADST_DCT, // SMOOTH_V
    DCT_ADST, // SMOOTH_H
    ADST_ADST, // PAETH
    DCT_DCT,  // UV_CFL
];

/// Context of the above / left mode for the key-frame y-mode CDF.
pub const INTRA_MODE_CONTEXT: [usize; 13] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];

/// Filter intra mode -> equivalent intra direction for tx type CDFs.
pub const FILTER_INTRA_MODE_TO_DIR: [u8; 5] = [DC_PRED, V_PRED, H_PRED, D157_PRED, DC_PRED];

/// Transform set types (libaom TxSetType).
pub const EXT_TX_SET_DCTONLY: usize = 0;
pub const EXT_TX_SET_DTT4_IDTX: usize = 2;
pub const EXT_TX_SET_DTT4_IDTX_1DDCT: usize = 3;
pub const NUM_EXT_TX_SET: [usize; 6] = [1, 2, 5, 7, 12, 16];

/// Intra transform set type for a transform size.
pub fn intra_tx_set(tx: TxSize, reduced: bool) -> usize {
    let sqr_up = tx.sqr_up();
    if sqr_up > TxSize::T32x32 || sqr_up == TxSize::T32x32 {
        return EXT_TX_SET_DCTONLY;
    }
    if reduced {
        return EXT_TX_SET_DTT4_IDTX;
    }
    if tx.sqr() == TxSize::T16x16 { EXT_TX_SET_DTT4_IDTX } else { EXT_TX_SET_DTT4_IDTX_1DDCT }
}

/// Index of a set type in the intra_ext_tx CDF array.
pub fn intra_tx_set_index(set: usize) -> usize {
    match set {
        EXT_TX_SET_DTT4_IDTX_1DDCT => 1,
        EXT_TX_SET_DTT4_IDTX => 2,
        _ => 0,
    }
}

#[inline]
pub fn tx_in_set(set: usize, t: TxType) -> bool {
    s::EXT_TX_USED[set][t as usize] != 0
}
