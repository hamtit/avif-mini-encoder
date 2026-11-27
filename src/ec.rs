//! The AV1 multi-symbol arithmetic encoder (libaom's `od_ec_enc`) and a plain
//! MSB-first bit writer for headers.

use crate::cdf::update_cdf;

const EC_PROB_SHIFT: u32 = 6;
const EC_MIN_PROB: u32 = 4;

pub struct SymbolWriter {
    /// Finalized output bytes (all carries already propagated).
    buf: Vec<u8>,
    /// Most recent non-0xFF byte waiting for potential carry propagation, or None before first byte.
    cached: Option<u8>,
    /// Number of consecutive 0xFF bytes following `cached`.
    ff_count: usize,
    low: u32,
    rng: u32,
    cnt: i32,
}

impl Default for SymbolWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolWriter {
    pub fn new() -> Self {
        SymbolWriter { buf: Vec::with_capacity(1 << 12), cached: None, ff_count: 0, low: 0, rng: 0x8000, cnt: -9 }
    }

    /// Emit one pre-carry byte `v` (0..=511), propagating any carry online.
    #[inline]
    fn emit(&mut self, v: u32) {
        let carry = v >> 8;
        let byte = (v & 0xFF) as u8;
        if carry != 0 {
            if let Some(b) = self.cached {
                self.buf.push(b.wrapping_add(carry as u8));
                if self.ff_count != 0 {
                    self.buf.resize(self.buf.len() + self.ff_count, 0x00);
                    self.ff_count = 0;
                }
            }
            if byte == 0xFF && !self.buf.is_empty() {
                self.cached = self.buf.pop();
                self.ff_count = 1;
            } else {
                self.cached = Some(byte);
            }
        } else if byte == 0xFF && self.cached.is_some() {
            self.ff_count += 1;
        } else {
            if let Some(b) = self.cached {
                self.buf.push(b);
                if self.ff_count != 0 {
                    self.buf.resize(self.buf.len() + self.ff_count, 0xFF);
                    self.ff_count = 0;
                }
            }
            self.cached = Some(byte);
        }
    }

    #[inline]
    fn normalize(&mut self, mut low: u32, rng: u32) {
        let d = rng.leading_zeros() as i32 - 16;
        let mut c = self.cnt;
        let mut s = c + d;
        if s >= 0 {
            c += 16;
            let mut m = (1u32 << c) - 1;
            if s >= 8 {
                self.emit(low >> c);
                low &= m;
                c -= 8;
                m >>= 8;
            }
            self.emit(low >> c);
            s = c + d - 24;
            low &= m;
        }
        self.low = low << d;
        self.rng = rng << d;
        self.cnt = s;
    }

    /// Encode with `fl`/`fh` as in libaom (inverted cumulative frequencies).
    #[inline]
    fn encode_q15(&mut self, fl: u32, fh: u32, s: u32, nsyms: u32) {
        let mut l = self.low;
        let mut r = self.rng;
        let n = nsyms - 1;
        if fl < 32768 {
            let u = (((r >> 8) * (fl >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (n - (s - 1));
            let v = (((r >> 8) * (fh >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (n - s);
            l += r - u;
            r = u - v;
        } else {
            r -= (((r >> 8) * (fh >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (n - s);
        }
        self.normalize(l, r);
    }

    /// Write symbol `s` with a CDF of `nsyms` symbols and adapt the CDF.
    #[inline]
    pub fn symbol(&mut self, s: usize, cdf: &mut [u16], nsyms: usize) {
        self.symbol_noadapt(s, cdf, nsyms);
        update_cdf(cdf, s, nsyms);
    }

    /// Write symbol `s` without adapting the CDF.
    #[inline]
    pub fn symbol_noadapt(&mut self, s: usize, cdf: &[u16], nsyms: usize) {
        let fl = if s > 0 { cdf[s - 1] as u32 } else { 32768 };
        self.encode_q15(fl, cdf[s] as u32, s as u32, nsyms as u32);
    }

    /// A raw bit with probability 1/2.
    #[inline]
    pub fn bit(&mut self, bit: bool) {
        let f = 16384u32;
        let l = self.low;
        let r = self.rng;
        let mut v = ((r >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
        v += EC_MIN_PROB;
        let (l, r) = if bit { (l + r - v, v) } else { (l, r - v) };
        self.normalize(l, r);
    }

    /// Flush and return the coded bytes.
    pub fn finish(mut self) -> Vec<u8> {
        let l = self.low;
        let mut c = self.cnt;
        let mut s = 10;
        let m = 0x3FFFu32;
        let mut e = ((l + m) & !m) | (m + 1);
        s += c;
        if s > 0 {
            let mut n = (1u32 << (c + 16)) - 1;
            loop {
                self.emit(e >> (c + 16));
                e &= n;
                s -= 8;
                c -= 8;
                n >>= 8;
                if s <= 0 {
                    break;
                }
            }
        }
        if let Some(b) = self.cached {
            self.buf.push(b);
            if self.ff_count != 0 {
                self.buf.resize(self.buf.len() + self.ff_count, 0xFF);
            }
        }
        self.buf
    }
}

/// Symbol sink: the real arithmetic coder, or one that only adapts the CDFs
/// (used to keep the encoder's CDFs in step with the bitstream while searching).
pub trait Writer {
    fn symbol(&mut self, s: usize, cdf: &mut [u16], nsyms: usize);
    fn symbol_noadapt(&mut self, s: usize, cdf: &[u16], nsyms: usize);
    fn bit(&mut self, bit: bool);
    fn literal(&mut self, v: u32, bits: u32) {
        for i in (0..bits).rev() {
            self.bit((v >> i) & 1 != 0);
        }
    }
    fn golomb(&mut self, level: u32) {
        let x = level + 1;
        let length = 32 - x.leading_zeros();
        for _ in 1..length {
            self.bit(false);
        }
        for i in (0..length).rev() {
            self.bit((x >> i) & 1 != 0);
        }
    }
}

impl Writer for SymbolWriter {
    #[inline]
    fn symbol(&mut self, s: usize, cdf: &mut [u16], nsyms: usize) {
        SymbolWriter::symbol(self, s, cdf, nsyms)
    }
    #[inline]
    fn symbol_noadapt(&mut self, s: usize, cdf: &[u16], nsyms: usize) {
        SymbolWriter::symbol_noadapt(self, s, cdf, nsyms)
    }
    #[inline]
    fn bit(&mut self, bit: bool) {
        SymbolWriter::bit(self, bit)
    }
}

/// Debug wrapper that logs every symbol to stderr (same format as a
/// trace-instrumented libaom decoder).
pub struct Tracing<'a, W: Writer>(pub &'a mut W);

impl<W: Writer> Writer for Tracing<'_, W> {
    fn symbol(&mut self, s: usize, cdf: &mut [u16], nsyms: usize) {
        eprintln!("S {} {} {}", s, nsyms, cdf[0]);
        self.0.symbol(s, cdf, nsyms)
    }
    fn symbol_noadapt(&mut self, s: usize, cdf: &[u16], nsyms: usize) {
        eprintln!("S {} {} {}", s, nsyms, cdf[0]);
        self.0.symbol_noadapt(s, cdf, nsyms)
    }
    fn bit(&mut self, bit: bool) {
        eprintln!("B {} 16384", bit as u8);
        self.0.bit(bit)
    }
}

/// Adapts CDFs without producing output.
pub struct CdfOnly;

impl Writer for CdfOnly {
    #[inline]
    fn symbol(&mut self, s: usize, cdf: &mut [u16], nsyms: usize) {
        update_cdf(cdf, s, nsyms);
    }
    #[inline]
    fn symbol_noadapt(&mut self, _s: usize, _cdf: &[u16], _nsyms: usize) {}
    #[inline]
    fn bit(&mut self, _bit: bool) {}
    fn literal(&mut self, _v: u32, _bits: u32) {}
    fn golomb(&mut self, _level: u32) {}
}

/// MSB-first bit writer for OBU headers.
#[derive(Default)]
pub struct BitWriter {
    pub buf: Vec<u8>,
    bits: u32,
}

impl BitWriter {
    pub fn put(&mut self, v: u64, n: u32) {
        for i in (0..n).rev() {
            let bit = ((v >> i) & 1) as u8;
            if self.bits.is_multiple_of(8) {
                self.buf.push(0);
            }
            let last = self.buf.last_mut().unwrap();
            *last |= bit << (7 - self.bits % 8);
            self.bits += 1;
        }
    }

    pub fn flag(&mut self, b: bool) {
        self.put(b as u64, 1);
    }

    /// Signed value in `n + 1` bits (su(1+n)).
    pub fn su(&mut self, v: i32, n: u32) {
        let mask = (1u64 << (n + 1)) - 1;
        self.put((v as i64 as u64) & mask, n + 1);
    }

    /// Pad to a byte boundary with zero bits.
    pub fn byte_align(&mut self) {
        self.bits = self.bits.div_ceil(8) * 8;
    }

    /// trailing_bits(): a one bit followed by zeros up to the byte boundary.
    pub fn trailing_bits(&mut self) {
        self.put(1, 1);
        self.byte_align();
    }
}

pub fn leb128(mut v: usize, out: &mut Vec<u8>) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference carry-propagation using the original `precarry: Vec<u16>` logic.
    fn reference_carry(vals: &[u32]) -> Vec<u8> {
        let mut out = vec![0u8; vals.len()];
        let mut carry = 0u32;
        for i in (0..vals.len()).rev() {
            carry += vals[i];
            out[i] = carry as u8;
            carry >>= 8;
        }
        out
    }

    #[test]
    fn online_carry_matches_precarry_reference() {
        // Test long 0xFF chains, carries, and random pre-carry sequences.
        let patterns: &[&[u32]] = &[
            &[],
            &[0x00],
            &[0xFF],
            &[0x12, 0xFF, 0xFF, 0xFF, 0x100],
            &[0x12, 0xFF, 0xFF, 0x34, 0xFF, 0x100, 0xFF, 0x101],
            &[0x00, 0xFF, 0x1FF, 0x100, 0xFF, 0xFF, 0x100],
        ];
        for &pat in patterns {
            let mut sw = SymbolWriter::new();
            for &v in pat {
                sw.emit(v);
            }
            if let Some(b) = sw.cached.take() {
                sw.buf.push(b);
                if sw.ff_count != 0 {
                    sw.buf.resize(sw.buf.len() + sw.ff_count, 0xFF);
                }
            }
            assert_eq!(sw.buf, reference_carry(pat), "pattern {pat:x?}");
        }
        // Also test a long stream of biased bits that trigger real range-coder carries.
        let mut sw = SymbolWriter::new();
        let mut seed = 0x1234_5678u32;
        for i in 0..4096 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let b = if i % 64 < 48 { true } else { (seed >> 31) != 0 };
            sw.bit(b);
        }
        let bytes = sw.finish();
        assert!(!bytes.is_empty());
    }
}
