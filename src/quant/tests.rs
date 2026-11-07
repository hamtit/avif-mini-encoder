use super::*;
use crate::tables::scan;

// Scan-order reference with the same FP rounding, thresholds and QM rules.
fn reference(coeff: &[i32], q: &PlaneQuant, tx: TxSize, tt: TxType, sc: &[u16]) -> (Vec<i32>, Vec<i32>, usize) {
    let mut qc = vec![0; sc.len()];
    let mut dq = vec![0; sc.len()];
    let mut eob = 0;
    let scale = tx_scale(tx);
    let matrices = q.matrices(tx, tt);
    for (i, &rc) in sc.iter().enumerate() {
        let rc = rc as usize;
        let k = (rc != 0) as usize;
        let abs = coeff[rc].unsigned_abs() as i64;
        let round = (q.round_fp[k] + ((1 << scale) >> 1)) >> scale;
        let (weight, dequant) = match matrices {
            Some((w, iw)) => (w[rc] as i64, (q.dequant[k] * iw[rc] as i32 + 16) >> 5),
            None => (32, q.dequant[k]),
        };
        if abs * weight < (q.dequant[k] as i64) << (QM_BITS - (1 + scale)) {
            continue;
        }
        let a = (abs + round as i64).clamp(-32768, 32767);
        let level = ((a * weight * q.quant_fp[k] as i64) >> (16 - scale + QM_BITS)) as i32;
        let sign = if coeff[rc] < 0 { -1 } else { 1 };
        qc[rc] = level * sign;
        dq[rc] = ((level * dequant) >> scale) * sign;
        if level != 0 {
            eob = i + 1;
        }
    }
    (qc, dq, eob)
}

#[test]
fn raster_quantization_matches_scan_order() {
    let mut seed = 17u32;
    for tx in TxSize::ALL {
        for class in 0..3 {
            let sc = scan(tx.coded(), class);
            let n = sc.len();
            for qindex in [0, 1, 80, 128, 255] {
                for level in [0, 8, 15] {
                    for chroma in [false, true] {
                        let quant = PlaneQuant::new(qindex, 0, 0, level, chroma);
                        for tt in [0, IDTX] {
                            for pattern in 0..6 {
                                let mut coeff = vec![0; n];
                                for (i, c) in coeff.iter_mut().enumerate() {
                                    seed ^= seed << 13;
                                    seed ^= seed >> 17;
                                    seed ^= seed << 5;
                                    *c = match pattern {
                                        0 => 0,
                                        1 => {
                                            if i == 0 {
                                                -32768
                                            } else {
                                                0
                                            }
                                        }
                                        2 => {
                                            if i == sc[n - 1] as usize {
                                                32767
                                            } else {
                                                0
                                            }
                                        }
                                        3 => {
                                            if i % 2 == 0 {
                                                -32768
                                            } else {
                                                32767
                                            }
                                        }
                                        4 => (seed % 131071) as i32 - 65535,
                                        _ => (seed % 65) as i32 - 32,
                                    };
                                }
                                let (expected_q, expected_dq, expected_eob) = reference(&coeff, &quant, tx, tt, sc);
                                // Sentinel tails must remain untouched.
                                let mut qc = vec![123; n + 7];
                                let mut dq = vec![123; n + 7];
                                let eob = quantize_fp(&coeff, &quant, tx, tt, sc, &mut qc, &mut dq);
                                assert_eq!(eob, expected_eob, "{tx:?} class={class} q={qindex} qm={level}");
                                assert_eq!(&qc[..n], expected_q);
                                assert_eq!(&dq[..n], expected_dq);
                                assert_eq!(&qc[n..], &[123; 7]);
                                assert_eq!(&dq[n..], &[123; 7]);
                            }
                        }
                    }
                }
            }
        }
    }
}
