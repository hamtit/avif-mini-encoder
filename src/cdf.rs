//! Adaptive CDFs (libaom's inverted representation: `cdf[i] = 32768 - P(x <= i)`,
//! with the adaptation counter stored after the last symbol) and the symbol
//! costs derived from them.

use crate::tables::cdf_default as d;

/// Number of q contexts for the default coefficient CDFs.
const TOKEN_CDF_Q_CTXS: usize = 4;

macro_rules! reshape {
    ($src:expr) => {{
        // The target type comes from the field being initialised, and `transmute` fails to
        // compile unless both sizes match, so spelling the types out would add nothing.
        #[allow(clippy::missing_transmute_annotations)]
        // SAFETY: nested arrays of u16 have the same layout as the flat array
        // of the same total length (checked by the compiler through the sizes).
        let reshaped = unsafe { core::mem::transmute($src) };
        reshaped
    }};
}

#[derive(Clone)]
pub struct CdfContext {
    pub partition: [[u16; 11]; 20],
    pub kf_y: [[[u16; 14]; 5]; 5],
    pub uv_mode: [[[u16; 15]; 13]; 2],
    pub angle_delta: [[u16; 8]; 8],
    pub filter_intra: [[u16; 3]; 22],
    pub filter_intra_mode: [u16; 6],
    pub cfl_sign: [u16; 9],
    pub cfl_alpha: [[u16; 17]; 6],
    pub skip: [[u16; 3]; 3],
    pub intra_ext_tx: [[[[u16; 17]; 13]; 4]; 3],
    pub tx_size: [[[u16; 4]; 3]; 4],
    pub delta_q: [u16; 5],
    pub palette_y_mode: [[[u16; 3]; 3]; 7],
    pub palette_uv_mode: [[u16; 3]; 2],
    pub palette_y_size: [[u16; 8]; 7],
    pub palette_uv_size: [[u16; 8]; 7],
    pub palette_y_color: [[[u16; 9]; 5]; 7],
    pub palette_uv_color: [[[u16; 9]; 5]; 7],
    pub coef: CoefCdfs,
}

#[derive(Clone)]
pub struct CoefCdfs {
    pub txb_skip: [[[u16; 3]; 13]; 5],
    pub eob_extra: [[[[u16; 3]; 9]; 2]; 5],
    pub dc_sign: [[[u16; 3]; 3]; 2],
    pub eob16: [[[u16; 6]; 2]; 2],
    pub eob32: [[[u16; 7]; 2]; 2],
    pub eob64: [[[u16; 8]; 2]; 2],
    pub eob128: [[[u16; 9]; 2]; 2],
    pub eob256: [[[u16; 10]; 2]; 2],
    pub eob512: [[[u16; 11]; 2]; 2],
    pub eob1024: [[[u16; 12]; 2]; 2],
    pub base_eob: [[[[u16; 4]; 4]; 2]; 5],
    pub base: [[[[u16; 5]; 42]; 2]; 5],
    pub br: [[[[u16; 5]; 21]; 2]; 5],
}

macro_rules! coef_defaults {
    ($m:ident) => {
        CoefCdfs {
            txb_skip: reshape!(d::$m::txb_skip_cdf),
            eob_extra: reshape!(d::$m::eob_extra_cdf),
            dc_sign: reshape!(d::$m::dc_sign_cdf),
            eob16: reshape!(d::$m::eob_flag_cdf16),
            eob32: reshape!(d::$m::eob_flag_cdf32),
            eob64: reshape!(d::$m::eob_flag_cdf64),
            eob128: reshape!(d::$m::eob_flag_cdf128),
            eob256: reshape!(d::$m::eob_flag_cdf256),
            eob512: reshape!(d::$m::eob_flag_cdf512),
            eob1024: reshape!(d::$m::eob_flag_cdf1024),
            base_eob: reshape!(d::$m::coeff_base_eob_cdf),
            base: reshape!(d::$m::coeff_base_cdf),
            br: reshape!(d::$m::coeff_br_cdf),
        }
    };
}

impl CdfContext {
    /// Default CDFs for a frame with the given base qindex.
    pub fn new(base_qindex: u8) -> Box<Self> {
        let q_ctx = match base_qindex {
            0..=20 => 0,
            21..=60 => 1,
            61..=120 => 2,
            _ => 3,
        };
        debug_assert!(q_ctx < TOKEN_CDF_Q_CTXS);
        let coef = match q_ctx {
            0 => coef_defaults!(coef0),
            1 => coef_defaults!(coef1),
            2 => coef_defaults!(coef2),
            _ => coef_defaults!(coef3),
        };
        Box::new(CdfContext {
            partition: reshape!(d::partition_cdf),
            kf_y: reshape!(d::kf_y_cdf),
            uv_mode: reshape!(d::uv_mode_cdf),
            angle_delta: reshape!(d::angle_delta_cdf),
            filter_intra: reshape!(d::filter_intra_cdfs),
            filter_intra_mode: d::filter_intra_mode_cdf,
            cfl_sign: d::cfl_sign_cdf,
            cfl_alpha: reshape!(d::cfl_alpha_cdf),
            skip: reshape!(d::skip_txfm_cdfs),
            intra_ext_tx: reshape!(d::intra_ext_tx_cdf),
            tx_size: reshape!(d::tx_size_cdf),
            delta_q: d::delta_q_cdf,
            palette_y_mode: reshape!(d::palette_y_mode_cdf),
            palette_uv_mode: reshape!(d::palette_uv_mode_cdf),
            palette_y_size: reshape!(d::palette_y_size_cdf),
            palette_uv_size: reshape!(d::palette_uv_size_cdf),
            palette_y_color: reshape!(d::palette_y_color_index_cdf),
            palette_uv_color: reshape!(d::palette_uv_color_index_cdf),
            coef,
        })
    }
}

/// libaom's `update_cdf`.
#[inline]
pub fn update_cdf(cdf: &mut [u16], val: usize, nsymbs: usize) {
    let count = cdf[nsymbs];
    let rate = 4 + (count >> 4) as u32 + (nsymbs > 3) as u32;
    for (i, c) in cdf[..nsymbs - 1].iter_mut().enumerate() {
        if i < val {
            *c += (32768 - *c) >> rate;
        } else {
            *c -= *c >> rate;
        }
    }
    cdf[nsymbs] += (count < 32) as u16;
}

/// Cost unit: 1/512 bit (libaom's AV1_PROB_COST_SHIFT = 9).
pub const COST_SHIFT: u32 = 9;

const PROB_COST: [u16; 128] = [
    512, 506, 501, 495, 489, 484, 478, 473, 467, 462, 456, 451, 446, 441, 435, 430, 425, 420, 415, 410, 405, 400, 395,
    390, 385, 380, 375, 371, 366, 361, 356, 352, 347, 343, 338, 333, 329, 324, 320, 316, 311, 307, 302, 298, 294, 289,
    285, 281, 277, 273, 268, 264, 260, 256, 252, 248, 244, 240, 236, 232, 228, 224, 220, 216, 212, 209, 205, 201, 197,
    194, 190, 186, 182, 179, 175, 171, 168, 164, 161, 157, 153, 150, 146, 143, 139, 136, 132, 129, 125, 122, 119, 115,
    112, 109, 105, 102, 99, 95, 92, 89, 86, 82, 79, 76, 73, 70, 66, 63, 60, 57, 54, 51, 48, 45, 42, 38, 35, 32, 29, 26,
    23, 20, 18, 15, 12, 9, 6, 3,
];

/// libaom's `av1_cost_symbol`: cost of a symbol with probability `p15 / 32768`.
#[inline]
pub fn cost_symbol(p15: u32) -> u32 {
    let p15 = p15.clamp(1, 32767);
    let shift = 15 - 1 - (31 - p15.leading_zeros());
    let prob = ((p15 << shift) + 64) >> 7; // get_prob(p15 << shift, 32768)
    let prob = prob.clamp(128, 255);
    PROB_COST[(prob - 128) as usize] as u32 + (shift << COST_SHIFT)
}

/// Costs of every symbol of a CDF (libaom's `av1_cost_tokens_from_cdf`).
pub fn costs_from_cdf(cdf: &[u16], nsymbs: usize, out: &mut [u32]) {
    let mut prev = 0u32;
    for i in 0..nsymbs {
        let cur = 32768 - cdf[i] as u32;
        let p = (cur - prev).max(4);
        prev = cur;
        out[i] = cost_symbol(p);
    }
}
