//! Quantization (libaom's FP quantizer with quantization matrices) and the
//! decoder's dequantization.

use crate::consts::{IDTX, TX_CLASS_HORIZ, TX_CLASS_VERT, TxSize, TxType};
use crate::tables::misc::{AC_Q, DC_Q};
use crate::tables::{iqm, qm};

pub const QM_BITS: u32 = 5;

pub fn dc_q(qindex: i32, delta: i32) -> i32 {
    DC_Q[(qindex + delta).clamp(0, 255) as usize] as i32
}

pub fn ac_q(qindex: i32, delta: i32) -> i32 {
    AC_Q[(qindex + delta).clamp(0, 255) as usize] as i32
}

/// libaom's `av1_get_tx_scale`.
#[inline]
pub fn tx_scale(tx: TxSize) -> u32 {
    let pels = tx.width() * tx.height();
    (pels > 256) as u32 + (pels > 1024) as u32
}

/// Quantizer for one plane at one qindex.
#[derive(Clone, Copy, Debug)]
pub struct PlaneQuant {
    /// [dc, ac]
    pub dequant: [i32; 2],
    pub quant_fp: [i32; 2],
    pub round_fp: [i32; 2],
    /// QM level (15 = flat).
    pub qm_level: u8,
    pub chroma: bool,
}

impl PlaneQuant {
    pub fn new(qindex: i32, dc_delta: i32, ac_delta: i32, qm_level: u8, chroma: bool) -> Self {
        let dq = [dc_q(qindex, dc_delta), ac_q(qindex, ac_delta)];
        // Sharpness 7 (tune IQ) makes the FP rounding factor 64 (round to nearest).
        let rounding_fp = 64;
        PlaneQuant {
            dequant: dq,
            quant_fp: [(1 << 16) / dq[0], (1 << 16) / dq[1]],
            round_fp: [(rounding_fp * dq[0]) >> 7, (rounding_fp * dq[1]) >> 7],
            qm_level,
            chroma,
        }
    }

    /// Forward and inverse weights for a transform, or None when flat.
    #[inline]
    pub fn matrices(&self, tx: TxSize, tx_type: TxType) -> Option<(&'static [u8], &'static [u8])> {
        if self.qm_level >= 15 || tx_type >= IDTX {
            return None;
        }
        let t = tx as usize;
        Some((qm(self.qm_level as usize, self.chroma, t), iqm(self.qm_level as usize, self.chroma, t)))
    }

    /// Effective dequantizer of the coefficient at raster index `rc`
    /// (decoder's get_dqv).
    #[inline]
    pub fn dqv(&self, rc: usize, iwt: Option<&[u8]>) -> i32 {
        let d = self.dequant[(rc != 0) as usize];
        match iwt {
            Some(m) => (d * m[rc] as i32 + (1 << (QM_BITS - 1))) >> QM_BITS,
            None => d,
        }
    }
}

/// libaom's quantize_fp_helper_c with optional QM. Returns eob.
#[allow(clippy::too_many_arguments)]
pub fn quantize_fp(
    coeff: &[i32],
    q: &PlaneQuant,
    tx: TxSize,
    tx_type: TxType,
    scan: &[u16],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> usize {
    #[cfg(target_arch = "x86_64")]
    if crate::has_avx2() {
        // SAFETY: runtime detection also respects AV1_NO_AVX2.
        return unsafe { quantize_fp_avx2(coeff, q, tx, tx_type, scan, qcoeff, dqcoeff) };
    }
    quantize_fp_scalar(coeff, q, tx, tx_type, scan, qcoeff, dqcoeff)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn quantize_fp_avx2(coeff: &[i32], q: &PlaneQuant, tx: TxSize, tx_type: TxType, scan: &[u16], qcoeff: &mut [i32], dqcoeff: &mut [i32]) -> usize {
    quantize_fp_scalar(coeff, q, tx, tx_type, scan, qcoeff, dqcoeff)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn quantize_fp_scalar(coeff: &[i32], q: &PlaneQuant, tx: TxSize, tx_type: TxType, scan: &[u16], qcoeff: &mut [i32], dqcoeff: &mut [i32]) -> usize {
    let n = scan.len();
    let log_scale = tx_scale(tx);
    let rounding = [
        (q.round_fp[0] + ((1 << log_scale) >> 1)) >> log_scale,
        (q.round_fp[1] + ((1 << log_scale) >> 1)) >> log_scale,
    ];
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);
    // Quantization is coefficient-local. Walk raster order for contiguous
    // loads/stores, then locate EOB by walking backward in scan order.
    match q.matrices(tx, tx_type) {
        None => {
            for rc in 0..n {
                let k = (rc != 0) as usize;
                let c = coeff[rc];
                let abs = c.unsigned_abs() as i64;
                if (abs << (1 + log_scale)) >= q.dequant[k] as i64 {
                    let a = (abs + rounding[k] as i64).clamp(-32768, 32767);
                    let tmp = ((a * q.quant_fp[k] as i64) >> (16 - log_scale)) as i32;
                    if tmp != 0 {
                        qcoeff[rc] = if c < 0 { -tmp } else { tmp };
                        let adq = (tmp * q.dequant[k]) >> log_scale;
                        dqcoeff[rc] = if c < 0 { -adq } else { adq };
                    }
                }
            }
        }
        Some((wt, iwt)) => {
            for rc in 0..n {
                let k = (rc != 0) as usize;
                let c = coeff[rc];
                let w = wt[rc] as i64;
                let dequant = (q.dequant[k] * iwt[rc] as i32 + (1 << (QM_BITS - 1))) >> QM_BITS;
                let abs = c.unsigned_abs() as i64;
                if abs * w >= (q.dequant[k] as i64) << (QM_BITS - (1 + log_scale)) {
                    let a = (abs + rounding[k] as i64).clamp(-32768, 32767);
                    let tmp = ((a * w * q.quant_fp[k] as i64) >> (16 - log_scale + QM_BITS)) as i32;
                    qcoeff[rc] = if c < 0 { -tmp } else { tmp };
                    let adq = (tmp * dequant) >> log_scale;
                    dqcoeff[rc] = if c < 0 { -adq } else { adq };
                }
            }
        }
    }
    // EOB: one past the largest scan index among the nonzero levels (branch-free, vectorizable).
    let Some(iscan) = iscan(tx.coded(), scan) else {
        return scan.iter().rposition(|&rc| qcoeff[rc as usize] != 0).map_or(0, |i| i + 1);
    };
    let mut eob = 0u32;
    for (&v, &i) in qcoeff[..n].iter().zip(&iscan[..n]) {
        eob = eob.max(i as u32 & ((v != 0) as u32).wrapping_neg());
    }
    eob as usize
}

/// One plus the position of each raster index in `scan`, which must be one of the scan orders of
/// the (coded) transform size.
fn iscan(tx: TxSize, scan: &[u16]) -> Option<&'static [u16]> {
    use std::sync::OnceLock;
    const CLASSES: [u8; 3] = [0, TX_CLASS_VERT, TX_CLASS_HORIZ];
    static T: OnceLock<Vec<Vec<u16>>> = OnceLock::new();
    let t = T.get_or_init(|| {
        let mut all = Vec::new();
        for &tx in &TxSize::ALL {
            for class in CLASSES {
                let sc = crate::tables::scan(tx, class);
                let mut inv = vec![0u16; sc.len()];
                for (i, &rc) in sc.iter().enumerate() {
                    inv[rc as usize] = i as u16 + 1;
                }
                all.push(inv);
            }
        }
        all
    });
    CLASSES
        .iter()
        .position(|&class| core::ptr::eq(crate::tables::scan(tx, class), scan))
        .map(|k| &t[tx as usize * 3 + k][..])
}

/// The decoder's dequantization of one coefficient.
#[inline]
pub fn dequant_coeff(level: i32, dqv: i32, log_scale: u32) -> i32 {
    let a = ((level.unsigned_abs() as i64 * dqv as i64) & 0xFF_FFFF) as i32 >> log_scale;
    let v = if level < 0 { -a } else { a };
    v.clamp(-(1 << 15), (1 << 15) - 1)
}

#[cfg(test)]
mod tests;
