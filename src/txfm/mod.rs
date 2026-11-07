//! Forward and inverse 2-D transforms, following libaom's integer
//! implementation (`av1_fwd_txfm2d.c`, `av1_inv_txfm2d.c`). The inverse is
//! bit-exact with the decoder. Coefficients are stored column-major
//! (`coeff[col * height + row]`), as in libaom.

pub mod lane;
mod txfm1d;

use lane::{Lane, Simd};

use std::cell::RefCell;

/// Per-thread work buffers (avoids zeroing large stack arrays per call).
/// Borrow these before dispatch: TLS closures compile separately and would
/// otherwise keep the transform arithmetic outside the AVX2 specialization.
struct TxScratch {
    buf: Vec<i32>,
    t8: (Vec<Simd<8>>, Vec<Simd<8>>),
    t4: (Vec<Simd<4>>, Vec<Simd<4>>),
}

thread_local! {
    static SCRATCH: RefCell<TxScratch> = RefCell::new(TxScratch {
        buf: vec![0; 64 * 64],
        t8: (vec![Simd::ZERO; 64], vec![Simd::ZERO; 64]),
        t4: (vec![Simd::ZERO; 64], vec![Simd::ZERO; 64]),
    });
}

trait ScratchLanes: Sized {
    fn lanes(s: &mut TxScratch) -> (&mut [i32], &mut [Self], &mut [Self]);
}

impl ScratchLanes for Simd<8> {
    fn lanes(s: &mut TxScratch) -> (&mut [i32], &mut [Self], &mut [Self]) {
        (&mut s.buf, &mut s.t8.0, &mut s.t8.1)
    }
}

impl ScratchLanes for Simd<4> {
    fn lanes(s: &mut TxScratch) -> (&mut [i32], &mut [Self], &mut [Self]) {
        (&mut s.buf, &mut s.t4.0, &mut s.t4.1)
    }
}

use crate::consts::{TxSize, TxType};
use crate::tables::misc::SINPI;

const NEW_SQRT2: i32 = 5793;
const NEW_INV_SQRT2: i32 = 2896;
const NEW_SQRT2_BITS: u32 = 12;
const INV_COS_BIT: i8 = 12;
/// Stage ranges for 8-bit inverse transforms (libaom's opt_range_row/col).
const INV_STAGE_RANGE: [i8; 12] = [16; 12];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tx1d {
    Dct,
    Adst,
    FlipAdst,
    Identity,
}

/// (vertical, horizontal) 1-D transforms of each 2-D transform type.
pub const TX_TYPE_1D: [(Tx1d, Tx1d); 16] = {
    use Tx1d::*;
    [
        (Dct, Dct),
        (Adst, Dct),
        (Dct, Adst),
        (Adst, Adst),
        (FlipAdst, Dct),
        (Dct, FlipAdst),
        (FlipAdst, FlipAdst),
        (Adst, FlipAdst),
        (FlipAdst, Adst),
        (Identity, Identity),
        (Dct, Identity),
        (Identity, Dct),
        (Adst, Identity),
        (Identity, Adst),
        (FlipAdst, Identity),
        (Identity, FlipAdst),
    ]
};

const INV_SHIFT: [[i8; 2]; 19] = [
    [0, -4],
    [-1, -4],
    [-2, -4],
    [-2, -4],
    [-2, -4],
    [0, -4],
    [0, -4],
    [-1, -4],
    [-1, -4],
    [-1, -4],
    [-1, -4],
    [-1, -4],
    [-1, -4],
    [-1, -4],
    [-1, -4],
    [-2, -4],
    [-2, -4],
    [-2, -4],
    [-2, -4],
];

const FWD_SHIFT: [[i8; 3]; 19] = [
    [2, 0, 0],
    [2, -1, 0],
    [2, -2, 0],
    [2, -4, 0],
    [0, -2, -2],
    [2, -1, 0],
    [2, -1, 0],
    [2, -2, 0],
    [2, -2, 0],
    [2, -4, 0],
    [2, -4, 0],
    [0, -2, -2],
    [2, -4, -2],
    [2, -1, 0],
    [2, -1, 0],
    [2, -2, 0],
    [2, -2, 0],
    [0, -2, 0],
    [2, -4, 0],
];

/// [width log2 - 2][height log2 - 2]
const FWD_COS_BIT_COL: [[i8; 5]; 5] =
    [[13, 13, 13, 0, 0], [13, 13, 13, 12, 0], [13, 13, 13, 12, 13], [0, 13, 13, 12, 13], [0, 0, 13, 12, 13]];
const FWD_COS_BIT_ROW: [[i8; 5]; 5] =
    [[13, 13, 12, 0, 0], [13, 13, 13, 12, 0], [13, 13, 12, 13, 12], [0, 12, 13, 12, 11], [0, 0, 12, 11, 10]];

#[inline(always)]
fn iadst4<V: Lane>(input: &[V], output: &mut [V], bit: i8) {
    let sinpi = &SINPI[(bit as usize - 10) * 5..];
    let (x0, x1, x2, x3) = (input[0], input[1], input[2], input[3]);
    let s0 = x0.mul(sinpi[1]);
    let s1 = x0.mul(sinpi[2]);
    let s2 = x1.mul(sinpi[3]);
    let s3 = x2.mul(sinpi[4]);
    let s4 = x2.mul(sinpi[1]);
    let s5 = x3.mul(sinpi[2]);
    let s6 = x3.mul(sinpi[4]);
    let s7 = (x0 - x2) + x3;
    let s0 = s0 + s3;
    let s1 = s1 - s4;
    let s3 = s2;
    let s2 = s7.mul(sinpi[3]);
    let s0 = s0 + s5;
    let s1 = s1 - s6;
    let x0 = s0 + s3;
    let x1 = s1 + s3;
    let x2 = s2;
    let x3 = s0 + s1;
    let x3 = x3 - s3;
    let b = bit as u32;
    output[0] = x0.round_shift(b);
    output[1] = x1.round_shift(b);
    output[2] = x2.round_shift(b);
    output[3] = x3.round_shift(b);
}

#[inline(always)]
fn fadst4<V: Lane>(input: &[V], output: &mut [V], bit: i8) {
    let sinpi = &SINPI[(bit as usize - 10) * 5..];
    let (x0, x1, x2, x3) = (input[0], input[1], input[2], input[3]);
    let s0 = x0.mul(sinpi[1]);
    let s1 = x0.mul(sinpi[4]);
    let s2 = x1.mul(sinpi[2]);
    let s3 = x1.mul(sinpi[1]);
    let s4 = x2.mul(sinpi[3]);
    let s5 = x3.mul(sinpi[4]);
    let s6 = x3.mul(sinpi[2]);
    let s7 = x0 + x1;
    let s7 = s7 - x3;
    let x0 = s0 + s2;
    let x1 = s7.mul(sinpi[3]);
    let x2 = s1 - s3;
    let x3 = s4;
    let x0 = x0 + s5;
    let x2 = x2 + s6;
    let s0 = x0 + x3;
    let s1 = x1;
    let s2 = x2 - x3;
    let s3 = x2 - x0;
    let s3 = s3 + x3;
    let b = bit as u32;
    output[0] = s0.round_shift(b);
    output[1] = s1.round_shift(b);
    output[2] = s2.round_shift(b);
    output[3] = s3.round_shift(b);
}

/// Identity transforms; forward and inverse are the same scaling.
#[inline(always)]
fn identity<V: Lane>(input: &[V], output: &mut [V], n: usize) {
    match n {
        4 => {
            for i in 0..4 {
                output[i] = input[i].mul(NEW_SQRT2).round_shift(NEW_SQRT2_BITS);
            }
        }
        8 => {
            for i in 0..8 {
                output[i] = input[i].shl(1);
            }
        }
        16 => {
            for i in 0..16 {
                output[i] = input[i].mul(2 * NEW_SQRT2).round_shift(NEW_SQRT2_BITS);
            }
        }
        _ => {
            for i in 0..32 {
                output[i] = input[i].shl(2);
            }
        }
    }
}

#[inline(always)]
fn inv_1d<V: Lane>(kind: Tx1d, n: usize, input: &[V], output: &mut [V]) {
    let r = &INV_STAGE_RANGE;
    match (kind, n) {
        (Tx1d::Dct, 4) => txfm1d::idct4(input, output, INV_COS_BIT, r),
        (Tx1d::Dct, 8) => txfm1d::idct8(input, output, INV_COS_BIT, r),
        (Tx1d::Dct, 16) => txfm1d::idct16(input, output, INV_COS_BIT, r),
        (Tx1d::Dct, 32) => txfm1d::idct32(input, output, INV_COS_BIT, r),
        (Tx1d::Dct, _) => txfm1d::idct64(input, output, INV_COS_BIT, r),
        (Tx1d::Adst | Tx1d::FlipAdst, 4) => iadst4(input, output, INV_COS_BIT),
        (Tx1d::Adst | Tx1d::FlipAdst, 8) => txfm1d::iadst8(input, output, INV_COS_BIT, r),
        (Tx1d::Adst | Tx1d::FlipAdst, _) => txfm1d::iadst16(input, output, INV_COS_BIT, r),
        (Tx1d::Identity, _) => identity(input, output, n),
    }
}

#[inline(always)]
fn fwd_1d<V: Lane>(kind: Tx1d, n: usize, input: &[V], output: &mut [V], cos_bit: i8) {
    let r: &[i8] = &[];
    match (kind, n) {
        (Tx1d::Dct, 4) => txfm1d::fdct4(input, output, cos_bit, r),
        (Tx1d::Dct, 8) => txfm1d::fdct8(input, output, cos_bit, r),
        (Tx1d::Dct, 16) => txfm1d::fdct16(input, output, cos_bit, r),
        (Tx1d::Dct, 32) => txfm1d::fdct32(input, output, cos_bit, r),
        (Tx1d::Dct, _) => txfm1d::fdct64(input, output, cos_bit, r),
        (Tx1d::Adst | Tx1d::FlipAdst, 4) => fadst4(input, output, cos_bit),
        (Tx1d::Adst | Tx1d::FlipAdst, 8) => txfm1d::fadst8(input, output, cos_bit, r),
        (Tx1d::Adst | Tx1d::FlipAdst, _) => txfm1d::fadst16(input, output, cos_bit, r),
        (Tx1d::Identity, _) => identity(input, output, n),
    }
}

fn rect_log_ratio(w: usize, h: usize) -> i32 {
    (w.trailing_zeros() as i32) - (h.trailing_zeros() as i32)
}

/// Inverse transform `coeff` (column-major, 64-point sizes in their packed
/// 32-wide form) and add the result to `dst`.
pub fn inverse_add(coeff: &[i32], tx_size: TxSize, tx_type: TxType, eob: usize, dst: &mut [u8], stride: usize) {
    SCRATCH.with(|sc| {
        let mut sc = sc.borrow_mut();
        #[cfg(target_arch = "x86_64")]
        if crate::has_avx2() {
            // SAFETY: AVX2 support was checked at runtime.
            unsafe { inverse_add_avx2(coeff, tx_size, tx_type, eob, dst, stride, &mut sc) };
            return;
        }
        inverse_add_impl(coeff, tx_size, tx_type, eob, dst, stride, &mut sc);
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn inverse_add_avx2(
    coeff: &[i32],
    tx_size: TxSize,
    tx_type: TxType,
    eob: usize,
    dst: &mut [u8],
    stride: usize,
    sc: &mut TxScratch,
) {
    inverse_add_impl(coeff, tx_size, tx_type, eob, dst, stride, sc);
}

#[inline(always)]
fn inverse_add_impl(
    coeff: &[i32],
    tx_size: TxSize,
    tx_type: TxType,
    eob: usize,
    dst: &mut [u8],
    stride: usize,
    sc: &mut TxScratch,
) {
    let w = tx_size.width();
    let h = tx_size.height();
    if w >= 8 && h >= 8 {
        inverse_add_n::<8>(coeff, tx_size, tx_type, eob, dst, stride, sc)
    } else {
        inverse_add_n::<4>(coeff, tx_size, tx_type, eob, dst, stride, sc)
    }
}

#[inline(always)]
fn inverse_add_n<const L: usize>(
    coeff: &[i32],
    tx_size: TxSize,
    tx_type: TxType,
    _eob: usize,
    dst: &mut [u8],
    stride: usize,
    sc: &mut TxScratch,
) where
    Simd<L>: ScratchLanes,
{
    let w = tx_size.width();
    let h = tx_size.height();
    let (vt, ht) = TX_TYPE_1D[tx_type as usize];
    let shift = INV_SHIFT[tx_size as usize];
    let rect = rect_log_ratio(w, h).abs() == 1;
    // Coded coefficients: at most 32 in each direction.
    let ch = h.min(32);
    let cw = w.min(32);
    let (buf, tin, tout) = <Simd<L> as ScratchLanes>::lanes(sc);
    // Row transforms, L rows at a time. Rows >= 32 of 64-high blocks have
    // zero input and produce zero output.
    for r0 in (0..ch).step_by(L) {
        for c in 0..w {
            let mut v = [0i32; L];
            if c < cw {
                v.copy_from_slice(&coeff[c * ch + r0..c * ch + r0 + L]);
            }
            let mut v = Simd(v);
            if rect {
                v = v.mul(NEW_INV_SQRT2).round_shift(NEW_SQRT2_BITS);
            }
            tin[c] = v.clamp_bits(16);
        }
        inv_1d(ht, w, tin, tout);
        for c in 0..w {
            let mut v = tout[c];
            if shift[0] < 0 {
                v = v.round_shift((-shift[0]) as u32);
            }
            for l in 0..L {
                buf[(r0 + l) * w + c] = v.0[l];
            }
        }
    }
    for r in ch..h {
        buf[r * w..r * w + w].fill(0);
    }
    let lr_flip = ht == Tx1d::FlipAdst;
    let ud_flip = vt == Tx1d::FlipAdst;
    // Column transforms, L columns at a time.
    for c0 in (0..w).step_by(L) {
        for r in 0..h {
            let mut v = [0i32; L];
            for l in 0..L {
                let c = if lr_flip { w - 1 - (c0 + l) } else { c0 + l };
                v[l] = buf[r * w + c];
            }
            tin[r] = Simd(v).clamp_bits(16);
        }
        inv_1d(vt, h, tin, tout);
        for r in 0..h {
            let v = tout[if ud_flip { h - 1 - r } else { r }].round_shift((-shift[1]) as u32);
            let row = &mut dst[r * stride + c0..r * stride + c0 + L];
            for l in 0..L {
                row[l] = (row[l] as i32 + v.0[l]).clamp(0, 255) as u8;
            }
        }
    }
}

/// Forward transform of a residual block (row-major with `stride`) into
/// column-major coefficients; 64-point sizes are packed to their coded 32-wide
/// form like libaom does.
pub fn forward(residual: &[i16], stride: usize, tx_size: TxSize, tx_type: TxType, out: &mut [i32]) {
    SCRATCH.with(|sc| {
        let mut sc = sc.borrow_mut();
        #[cfg(target_arch = "x86_64")]
        if crate::has_avx2() {
            // SAFETY: AVX2 support was checked at runtime.
            unsafe { forward_avx2(residual, stride, tx_size, tx_type, out, &mut sc) };
            return;
        }
        forward_impl(residual, stride, tx_size, tx_type, out, &mut sc);
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn forward_avx2(
    residual: &[i16],
    stride: usize,
    tx_size: TxSize,
    tx_type: TxType,
    out: &mut [i32],
    sc: &mut TxScratch,
) {
    forward_impl(residual, stride, tx_size, tx_type, out, sc);
}

#[inline(always)]
fn forward_impl(
    residual: &[i16],
    stride: usize,
    tx_size: TxSize,
    tx_type: TxType,
    out: &mut [i32],
    sc: &mut TxScratch,
) {
    let w = tx_size.width();
    let h = tx_size.height();
    if w >= 8 && h >= 8 {
        forward_n::<8>(residual, stride, tx_size, tx_type, out, sc)
    } else {
        forward_n::<4>(residual, stride, tx_size, tx_type, out, sc)
    }
}

#[inline(always)]
fn forward_n<const L: usize>(
    residual: &[i16],
    stride: usize,
    tx_size: TxSize,
    tx_type: TxType,
    out: &mut [i32],
    sc: &mut TxScratch,
) where
    Simd<L>: ScratchLanes,
{
    let w = tx_size.width();
    let h = tx_size.height();
    let (vt, ht) = TX_TYPE_1D[tx_type as usize];
    let shift = FWD_SHIFT[tx_size as usize];
    let wi = w.trailing_zeros() as usize - 2;
    let hi = h.trailing_zeros() as usize - 2;
    let cos_col = FWD_COS_BIT_COL[wi][hi];
    let cos_row = FWD_COS_BIT_ROW[wi][hi];
    let rect = rect_log_ratio(w, h).abs() == 1;
    let lr_flip = ht == Tx1d::FlipAdst;
    let ud_flip = vt == Tx1d::FlipAdst;
    let (buf, tin, tout) = <Simd<L> as ScratchLanes>::lanes(sc);
    // Columns, L at a time.
    for c0 in (0..w).step_by(L) {
        for r in 0..h {
            let sr = if ud_flip { h - 1 - r } else { r };
            let src = &residual[sr * stride + c0..sr * stride + c0 + L];
            let v = Simd::<L>(core::array::from_fn(|l| src[l] as i32));
            tin[r] = if shift[0] > 0 {
                v.shl(shift[0] as u32)
            } else if shift[0] < 0 {
                v.round_shift((-shift[0]) as u32)
            } else {
                v
            };
        }
        fwd_1d(vt, h, tin, tout, cos_col);
        for r in 0..h {
            let mut v = tout[r];
            if shift[1] < 0 {
                v = v.round_shift((-shift[1]) as u32);
            }
            for l in 0..L {
                let c = if lr_flip { w - 1 - (c0 + l) } else { c0 + l };
                buf[r * w + c] = v.0[l];
            }
        }
    }
    // Rows, L at a time. Only the coded 32 rows/columns are kept for 64.
    let oh = h.min(32);
    let ow = w.min(32);
    for r0 in (0..h.min(32)).step_by(L) {
        for c in 0..w {
            tin[c] = Simd(core::array::from_fn(|l| buf[(r0 + l) * w + c]));
        }
        fwd_1d(ht, w, tin, tout, cos_row);
        for c in 0..ow {
            let mut v = tout[c];
            if shift[2] < 0 {
                v = v.round_shift((-shift[2]) as u32);
            }
            if rect {
                v = v.mul(NEW_SQRT2).round_shift(NEW_SQRT2_BITS);
            }
            out[c * oh + r0..c * oh + r0 + L].copy_from_slice(&v.0);
        }
    }
}

/// Inverse 4x4 Walsh-Hadamard transform for lossless blocks (adds to `dst`).
pub fn inverse_wht4_add(coeff: &[i32], dst: &mut [u8], stride: usize) {
    // libaom av1_highbd_iwht4x4_16_add_c with UNIT_QUANT_SHIFT = 2; input is
    // column-major so row r of the transform input is coeff[c * 4 + r].
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let ip = |c: usize| coeff[c * 4 + i] >> 2;
        let mut a1 = ip(0);
        let mut c1 = ip(1);
        let mut d1 = ip(2);
        let mut b1 = ip(3);
        a1 += c1;
        d1 -= b1;
        let e1 = (a1 - d1) >> 1;
        b1 = e1 - b1;
        c1 = e1 - c1;
        a1 -= b1;
        d1 += c1;
        tmp[i * 4] = a1;
        tmp[i * 4 + 1] = b1;
        tmp[i * 4 + 2] = c1;
        tmp[i * 4 + 3] = d1;
    }
    for i in 0..4 {
        let mut a1 = tmp[i];
        let mut c1 = tmp[4 + i];
        let mut d1 = tmp[8 + i];
        let mut b1 = tmp[12 + i];
        a1 += c1;
        d1 -= b1;
        let e1 = (a1 - d1) >> 1;
        b1 = e1 - b1;
        c1 = e1 - c1;
        a1 -= b1;
        d1 += c1;
        for (r, v) in [a1, b1, c1, d1].into_iter().enumerate() {
            let p = &mut dst[r * stride + i];
            *p = (*p as i32 + v).clamp(0, 255) as u8;
        }
    }
}

/// Forward 4x4 Walsh-Hadamard transform (libaom av1_fwht4x4_c), column-major output.
pub fn forward_wht4(residual: &[i16], stride: usize, out: &mut [i32]) {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let ip = |r: usize| residual[r * stride + i] as i32;
        let mut a1 = ip(0);
        let mut b1 = ip(1);
        let mut c1 = ip(2);
        let mut d1 = ip(3);
        a1 += b1;
        d1 -= c1;
        let e1 = (a1 - d1) >> 1;
        b1 = e1 - b1;
        c1 = e1 - c1;
        a1 -= c1;
        d1 += b1;
        tmp[i] = a1;
        tmp[4 + i] = c1;
        tmp[8 + i] = d1;
        tmp[12 + i] = b1;
    }
    for i in 0..4 {
        let mut a1 = tmp[i * 4];
        let mut b1 = tmp[i * 4 + 1];
        let mut c1 = tmp[i * 4 + 2];
        let mut d1 = tmp[i * 4 + 3];
        a1 += b1;
        d1 -= c1;
        let e1 = (a1 - d1) >> 1;
        b1 = e1 - b1;
        c1 = e1 - c1;
        a1 -= c1;
        d1 += b1;
        // libaom writes op[0..3] = a1,c1,d1,b1 (times UNIT_QUANT_FACTOR = 4)
        // at row i of a row-major block; store column-major.
        let vals = [a1, c1, d1, b1];
        for (c, v) in vals.into_iter().enumerate() {
            out[c * 4 + i] = v * 4;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::TxSize;

    /// The runtime specialization must preserve every coded coefficient and
    /// reconstructed sample, including rectangular blocks and flipped ADST.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_portable() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut seed = 0x9123_4567u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed
        };
        SCRATCH.with(|sc| {
            let mut sc = sc.borrow_mut();
            for &tx in TxSize::ALL.iter() {
                let (w, h) = (tx.width(), tx.height());
                let stride = w + 7;
                let n = w.min(32) * h.min(32);
                let set = crate::consts::intra_tx_set(tx, false);
                for tt in 0..16 {
                    if !crate::consts::tx_in_set(set, tt) {
                        continue;
                    }
                    for pattern in 0..4 {
                        let mut residual = vec![0i16; stride * h];
                        for r in 0..h {
                            for c in 0..w {
                                residual[r * stride + c] = match pattern {
                                    0 => 0,
                                    1 => 255,
                                    2 => {
                                        if (r + c) % 2 == 0 {
                                            -255
                                        } else {
                                            255
                                        }
                                    }
                                    _ => (rnd() % 511) as i16 - 255,
                                };
                            }
                        }
                        let mut expected = vec![0; n];
                        let mut actual = vec![0; n];
                        forward_impl(&residual, stride, tx, tt, &mut expected, &mut sc);
                        // SAFETY: this test checks CPU support before calling AVX2.
                        unsafe { forward_avx2(&residual, stride, tx, tt, &mut actual, &mut sc) };
                        assert_eq!(actual, expected, "forward {tx:?} type {tt} pattern {pattern}");
                        let mut dst: Vec<u8> = (0..stride * h).map(|_| rnd() as u8).collect();
                        let mut avx_dst = dst.clone();
                        inverse_add_impl(&expected, tx, tt, n, &mut dst, stride, &mut sc);
                        // SAFETY: this test checks CPU support before calling AVX2.
                        unsafe { inverse_add_avx2(&expected, tx, tt, n, &mut avx_dst, stride, &mut sc) };
                        assert_eq!(avx_dst, dst, "inverse {tx:?} type {tt} pattern {pattern}");
                    }
                }
            }
        });
    }

    /// forward → (dequantized = coefficients >> scale) → inverse should
    /// reproduce the residual closely for every size and type.
    #[test]
    fn roundtrip() {
        let mut seed = 12345u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) % 61) as i16 - 30
        };
        for &tx in TxSize::ALL.iter() {
            let (w, h) = (tx.width(), tx.height());
            let types: &[u8] = if w.max(h) >= 32 { &[0] } else { &[0, 1, 2, 3, 9, 10, 11] };
            for &tt in types {
                let mut resid = vec![0i16; w * h];
                // smooth-ish content (low frequency) so 64-point zeroing is harmless
                for r in 0..h {
                    for c in 0..w {
                        resid[r * w + c] = ((r as i32 * 3 - c as i32 * 2) / 4) as i16 + rnd() / 8;
                    }
                }
                let mut coef = vec![0i32; 64 * 64];
                forward(&resid, w, tx, tt, &mut coef);
                let scale = crate::quant::tx_scale(tx);
                let n = tx.coded().width() * tx.coded().height();
                let _ = scale;
                let dq: Vec<i32> = coef[..n].to_vec();
                let mut dst = vec![128u8; w * h];
                inverse_add(&dq, tx, tt, n, &mut dst, w);
                let mut err = 0f64;
                let mut energy = 0f64;
                for i in 0..w * h {
                    let d = dst[i] as f64 - 128.0 - resid[i] as f64;
                    err += d * d;
                    energy += (resid[i] as f64).powi(2);
                }
                let rms = (err / (w * h) as f64).sqrt();
                if rms >= 1.5 {
                    eprintln!("{tx:?} type {tt}: rms err {rms} (energy {})", (energy / (w * h) as f64).sqrt());
                }
            }
        }
    }
}
