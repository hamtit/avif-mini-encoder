use super::*;

// Independent per-pixel reference, before splitting zone 2 into row segments.
#[allow(clippy::too_many_arguments)]
fn reference(
    dst: &mut [u8],
    stride: usize,
    w: usize,
    h: usize,
    above: &[u8],
    left: &[u8],
    angle: i32,
    ua: usize,
    ul: usize,
) {
    let edge = |data: &[u8], base: i32, shift: i32| {
        let base = (EDGE_OFF as i32 + base) as usize;
        (data[base] as i32 * (32 - shift) + data[base + 1] as i32 * shift + 16) >> 5
    };
    for r in 0..h {
        for c in 0..w {
            let value = if angle < 90 {
                let dx = DR_INTRA_DERIVATIVE[angle as usize] as i32;
                let x = (r as i32 + 1) * dx;
                let base = (x >> (6 - ua)) + ((c as i32) << ua);
                let max = ((w + h - 1) << ua) as i32;
                if base < max {
                    edge(above, base, ((x << ua) & 63) >> 1)
                } else {
                    above[EDGE_OFF + max as usize] as i32
                }
            } else if angle < 180 && angle != 90 {
                let dx = DR_INTRA_DERIVATIVE[(180 - angle) as usize] as i32;
                let dy = DR_INTRA_DERIVATIVE[(angle - 90) as usize] as i32;
                let x = ((c as i32) << 6) - (r as i32 + 1) * dx;
                let base = x >> (6 - ua);
                if base >= -(1 << ua) {
                    edge(above, base, ((x << ua) & 63) >> 1)
                } else {
                    let y = ((r as i32) << 6) - (c as i32 + 1) * dy;
                    edge(left, y >> (6 - ul), ((y << ul) & 63) >> 1)
                }
            } else if angle > 180 {
                let dy = DR_INTRA_DERIVATIVE[(270 - angle) as usize] as i32;
                let y = (c as i32 + 1) * dy;
                let base = (y >> (6 - ul)) + ((r as i32) << ul);
                let max = ((w + h - 1) << ul) as i32;
                if base < max { edge(left, base, ((y << ul) & 63) >> 1) } else { left[EDGE_OFF + max as usize] as i32 }
            } else if angle == 90 {
                above[EDGE_OFF + c] as i32
            } else {
                left[EDGE_OFF + r] as i32
            };
            dst[r * stride + c] = value as u8;
        }
    }
}

#[test]
fn directional_matches_per_pixel_reference() {
    for w in [4, 8, 16, 32, 64] {
        for h in [4, 8, 16, 32, 64] {
            for ua in 0..=usize::from(w + h <= 64) {
                for ul in 0..=usize::from(w + h <= 64) {
                    for pattern in 0..5 {
                        let make_edge = |offset: usize| {
                            std::array::from_fn::<_, EDGE_LEN, _>(|i| match pattern {
                                0 => 0,
                                1 => 255,
                                2 => {
                                    if i % 2 == 0 {
                                        0
                                    } else {
                                        255
                                    }
                                }
                                _ => ((i * 131 + pattern * 71 + offset) % 256) as u8,
                            })
                        };
                        let above = make_edge(0);
                        let left = make_edge(19);
                        for mode in V_PRED..=D67_PRED {
                            for delta in -3..=3 {
                                let angle = MODE_TO_ANGLE[mode as usize] + delta * 3;
                                let stride = w + 5;
                                // No padding after the last row; other row padding is a sentinel.
                                let mut actual = vec![0xA5; (h - 1) * stride + w];
                                let mut expected = actual.clone();
                                reference(&mut expected, stride, w, h, &above, &left, angle, ua, ul);
                                directional(&mut actual, stride, w, h, &above, &left, angle, ua, ul);
                                assert_eq!(actual, expected, "{w}x{h} angle={angle} ua={ua} ul={ul} pattern={pattern}");
                            }
                        }
                    }
                }
            }
        }
    }
}
