use super::*;

#[test]
fn residual_matches_reference() {
    for w in [4, 8, 16, 32, 64] {
        for h in [4, 8, 16, 32, 64] {
            let ss = w + 3;
            let src: Vec<_> = (0..(h - 1) * ss + w).map(|i| (i * 137) as u8).collect();
            let pred: Vec<_> = (0..(h - 1) * 64 + w).map(|i| (i * 71 + 255) as u8).collect();
            let mut expected = vec![1234; w * h + 7];
            for r in 0..h {
                for c in 0..w {
                    expected[r * w + c] = src[r * ss + c] as i16 - pred[r * 64 + c] as i16;
                }
            }
            let mut actual = vec![1234; w * h + 7];
            residual(&src, ss, &pred, w, h, &mut actual);
            assert_eq!(actual, expected);
            residual_scalar(&src, ss, &pred, w, h, &mut actual);
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn distortion_matches_reference() {
    let mut seed = 123u32;
    for n in [0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 63, 64, 256, 1024] {
        for pattern in 0..4 {
            let mut t = vec![0; n];
            let mut dq = vec![0; n];
            let mut wt = vec![0; n];
            for i in 0..n {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                t[i] = match pattern {
                    0 => 0,
                    1 => {
                        if i % 2 == 0 {
                            -65535
                        } else {
                            65535
                        }
                    }
                    _ => (seed % 131071) as i32 - 65535,
                };
                dq[i] = if pattern == 0 { 0 } else { (seed.rotate_left(13) % 65536) as i32 - 32768 };
                wt[i] = match pattern {
                    0 => 0,
                    1 => 255,
                    2 => 32,
                    _ => seed as u8,
                };
            }
            for scale in 0..=2 {
                for weights in [None, Some(wt.as_slice())] {
                    let mut expected = 0i64;
                    for i in 0..n {
                        let d = (t[i] - dq[i]) as i64;
                        expected += match weights {
                            None => d * d,
                            Some(m) => {
                                let d = d * m[i] as i64;
                                (d * d + 512) >> 10
                            }
                        };
                    }
                    expected = match scale {
                        0 => expected >> 2,
                        1 => expected,
                        _ => expected << 2,
                    };
                    assert_eq!(tx_dist_scalar(&t, &dq, n, scale, weights), expected);
                    assert_eq!(tx_dist(&t, &dq, n, scale, weights), expected);
                    #[cfg(target_arch = "x86_64")]
                    if std::arch::is_x86_feature_detected!("avx2") {
                        // SAFETY: CPU support was checked above.
                        assert_eq!(unsafe { tx_dist_avx2(&t, &dq, n, scale, weights) }, expected);
                    }
                }
            }
        }
    }
}
