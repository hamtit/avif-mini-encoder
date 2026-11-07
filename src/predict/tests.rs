use super::*;

#[test]
fn dispatched_predictions_match_scalar() {
    for w in [4, 8, 16, 32, 64] {
        for h in [4, 8, 16, 32, 64] {
            for pattern in 0..5 {
                let edge = |i: usize| match pattern {
                    0 => 0,
                    1 => 255,
                    2 => {
                        if i.is_multiple_of(2) {
                            0
                        } else {
                            255
                        }
                    }
                    _ => ((i * 131 + pattern * 71) % 256) as u8,
                };
                let above: Vec<_> = (0..w).map(edge).collect();
                let left: Vec<_> = (1..=h).map(edge).collect();
                let stride = w + 5;
                let mut expected = vec![0xA5; (h - 1) * stride + w];
                let mut actual = expected.clone();
                type Predictor = fn(&mut [u8], usize, usize, usize, &[u8], &[u8]);
                for (scalar, dispatched) in [
                    (smooth_scalar as Predictor, smooth as Predictor),
                    (smooth_v_scalar as Predictor, smooth_v as Predictor),
                    (smooth_h_scalar as Predictor, smooth_h as Predictor),
                ] {
                    scalar(&mut expected, stride, w, h, &above, &left);
                    dispatched(&mut actual, stride, w, h, &above, &left);
                    assert_eq!(actual, expected, "{w}x{h} pattern {pattern}");
                }
                for tl in [0, 1, 127, 128, 254, 255] {
                    paeth_scalar(&mut expected, stride, w, h, &above, &left, tl);
                    paeth(&mut actual, stride, w, h, &above, &left, tl);
                    assert_eq!(actual, expected, "paeth {w}x{h} tl {tl}");
                }
            }
        }
    }
}

#[test]
fn dispatched_cfl_matches_scalar() {
    for w in [4, 8, 16, 32] {
        for h in [4, 8, 16, 32] {
            let stride = w + 5;
            for pattern in 0..4 {
                let luma: Vec<i16> = (0..w * h)
                    .map(|i| match pattern {
                        0 => 0,
                        1 => 2040,
                        2 => {
                            if i % 2 == 0 {
                                0
                            } else {
                                2040
                            }
                        }
                        _ => ((i * 997 + 19) % 2041) as i16,
                    })
                    .collect();
                for dc in [0, 1, 127, 128, 254, 255] {
                    for alpha in -16..=16 {
                        let mut expected = vec![dc; (h - 1) * stride + w];
                        let mut actual = expected.clone();
                        cfl_apply_scalar(&mut expected, stride, w, h, &luma, alpha);
                        cfl_predict(&mut actual, stride, w, h, &luma, alpha);
                        assert_eq!(actual, expected, "CfL {w}x{h} dc {dc} alpha {alpha}");
                    }
                }
            }
        }
    }
}

// Keep the original output-major arithmetic as an independent reference.
fn filter_intra_reference(dst: &mut [u8], stride: usize, w: usize, h: usize, above: &[u8], left: &[u8], mode: usize) {
    let mut buf = [[0u8; 33]; 33];
    for r in 0..h {
        buf[r + 1][0] = left[EDGE_OFF + r];
    }
    buf[0][..w + 1].copy_from_slice(&above[EDGE_OFF - 1..EDGE_OFF + w]);
    for r in (1..=h).step_by(2) {
        for c in (1..=w).step_by(4) {
            let p = [
                buf[r - 1][c - 1],
                buf[r - 1][c],
                buf[r - 1][c + 1],
                buf[r - 1][c + 2],
                buf[r - 1][c + 3],
                buf[r][c - 1],
                buf[r + 1][c - 1],
            ];
            for k in 0..8 {
                let taps = &FILTER_INTRA_TAPS[mode * 56 + k * 7..][..7];
                let sum: i32 = (0..7).map(|i| taps[i] as i32 * p[i] as i32).sum();
                buf[r + (k >> 2)][c + (k & 3)] = ((sum + 8) >> 4).clamp(0, 255) as u8;
            }
        }
    }
    for r in 0..h {
        dst[r * stride..r * stride + w].copy_from_slice(&buf[r + 1][1..=w]);
    }
}

#[test]
fn filter_intra_matches_reference() {
    let mut seed = 0x12345678u32;
    for w in [4, 8, 16, 32] {
        for h in [4, 8, 16, 32] {
            for pattern in 0..12 {
                let mut above = [0u8; EDGE_LEN];
                let mut left = [0u8; EDGE_LEN];
                for (i, v) in above.iter_mut().chain(left.iter_mut()).enumerate() {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    *v = match pattern {
                        0 => 0,
                        1 => 255,
                        2 => {
                            if i % 2 == 0 {
                                255
                            } else {
                                0
                            }
                        }
                        3 => {
                            if i % 2 == 0 {
                                0
                            } else {
                                255
                            }
                        }
                        _ => seed as u8,
                    };
                }
                for mode in 0..5 {
                    let stride = w + 5;
                    // Sentinel padding, and no padding after the final row.
                    let mut expected = vec![0xA5; (h - 1) * stride + w];
                    filter_intra_reference(&mut expected, stride, w, h, &above, &left, mode);
                    let mut actual = vec![0xA5; expected.len()];
                    filter_intra_scalar(&mut actual, stride, w, h, &above, &left, mode);
                    assert_eq!(actual, expected, "scalar {w}x{h}, mode {mode}, pattern {pattern}");
                    actual.fill(0xA5);
                    filter_intra(&mut actual, stride, w, h, &above, &left, mode);
                    assert_eq!(actual, expected, "dispatch {w}x{h}, mode {mode}, pattern {pattern}");
                }
            }
        }
    }
}

/// The vectorised edge interpolation must equal the scalar formula for every
/// shift, both steps and lengths around the vector width.
#[test]
fn interpolate_matches_scalar() {
    let mut seed = 0x1357_9bdfu32;
    for step in [1usize, 2] {
        for len in 1..40 {
            let edge: Vec<u8> = (0..(len - 1) * step + 2)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    seed as u8
                })
                .collect();
            for shift in 0..32 {
                let expected: Vec<u8> = (0..len)
                    .map(|j| {
                        let (a, b) = (edge[j * step] as i32, edge[j * step + 1] as i32);
                        ((a * (32 - shift) + b * shift + 16) >> 5) as u8
                    })
                    .collect();
                let mut out = vec![0u8; len];
                if step == 1 {
                    interpolate::<1>(&mut out, &edge, shift);
                } else {
                    interpolate::<2>(&mut out, &edge, shift);
                }
                assert_eq!(out, expected, "step {step} len {len} shift {shift}");
            }
        }
    }
}
