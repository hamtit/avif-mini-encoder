//! Arithmetic used by the generated 1-D transforms. The transforms are generic
//! over the lane type so the same code runs on one column (`i32`) or on several
//! columns at once (`I32x8`), which the compiler vectorizes.

use core::ops::{Add, Neg, Sub};

use crate::tables::misc::COSPI;

pub trait Lane: Copy + Add<Output = Self> + Sub<Output = Self> + Neg<Output = Self> {
    const ZERO: Self;
    fn mul_add_round(w0: i32, a: Self, w1: i32, b: Self, bit: i8) -> Self;
    fn clamp_bits(self, bits: i8) -> Self;
    fn mul(self, k: i32) -> Self;
    fn round_shift(self, bit: u32) -> Self;
    fn shl(self, bit: u32) -> Self;
}

#[inline(always)]
pub fn half_btf<V: Lane>(w0: i32, in0: V, w1: i32, in1: V, bit: i8) -> V {
    V::mul_add_round(w0, in0, w1, in1, bit)
}

#[inline(always)]
pub fn clamp<V: Lane>(v: V, bits: i8) -> V {
    v.clamp_bits(bits)
}

#[inline(always)]
pub fn cospi_arr(bit: i8) -> &'static [i32] {
    let i = (bit - 10) as usize;
    &COSPI[i * 64..i * 64 + 64]
}

impl Lane for i32 {
    const ZERO: i32 = 0;
    #[inline(always)]
    fn mul_add_round(w0: i32, a: i32, w1: i32, b: i32, bit: i8) -> i32 {
        let r = w0.wrapping_mul(a).wrapping_add(w1.wrapping_mul(b));
        r.wrapping_add(1 << (bit - 1)) >> bit
    }
    #[inline(always)]
    fn clamp_bits(self, bits: i8) -> i32 {
        let max = (1i32 << (bits - 1)) - 1;
        self.clamp(-max - 1, max)
    }
    #[inline(always)]
    fn mul(self, k: i32) -> i32 {
        self.wrapping_mul(k)
    }
    #[inline(always)]
    fn round_shift(self, bit: u32) -> i32 {
        self.wrapping_add(1 << (bit - 1)) >> bit
    }
    #[inline(always)]
    fn shl(self, bit: u32) -> i32 {
        self.wrapping_shl(bit)
    }
}

/// `N` i32 lanes processed together.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C, align(32))]
pub struct Simd<const N: usize>(pub [i32; N]);

impl<const N: usize> Simd<N> {
    #[inline(always)]
    fn map(self, f: impl Fn(i32) -> i32) -> Self {
        Simd(core::array::from_fn(|i| f(self.0[i])))
    }
}

impl<const N: usize> Add for Simd<N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Simd(core::array::from_fn(|i| self.0[i].wrapping_add(o.0[i])))
    }
}

impl<const N: usize> Sub for Simd<N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Simd(core::array::from_fn(|i| self.0[i].wrapping_sub(o.0[i])))
    }
}

impl<const N: usize> Neg for Simd<N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        self.map(i32::wrapping_neg)
    }
}

impl<const N: usize> Lane for Simd<N> {
    const ZERO: Self = Simd([0; N]);
    #[inline(always)]
    fn mul_add_round(w0: i32, a: Self, w1: i32, b: Self, bit: i8) -> Self {
        Simd(core::array::from_fn(|i| i32::mul_add_round(w0, a.0[i], w1, b.0[i], bit)))
    }
    #[inline(always)]
    fn clamp_bits(self, bits: i8) -> Self {
        self.map(|v| v.clamp_bits(bits))
    }
    #[inline(always)]
    fn mul(self, k: i32) -> Self {
        self.map(|v| v.wrapping_mul(k))
    }
    #[inline(always)]
    fn round_shift(self, bit: u32) -> Self {
        self.map(|v| v.round_shift(bit))
    }
    #[inline(always)]
    fn shl(self, bit: u32) -> Self {
        self.map(|v| v.wrapping_shl(bit))
    }
}
