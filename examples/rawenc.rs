//! Test tool: encode a PPM to a raw AV1 OBU stream and dump the encoder's
//! reconstruction, for comparison with a reference decoder.
//! usage: rawenc in.ppm qindex out_prefix [420|444|mono] [iq]
//! `iq` benchmarks the public encode_av1 API with all tune-IQ adjustments.
use avif_mini_encoder::enc::FrameConfig;
use avif_mini_encoder::{ColorDesc, SequenceHeader, raw};

fn read_ppm(path: &str) -> (usize, usize, Vec<u8>) {
    let d = std::fs::read(path).unwrap();
    let mut pos = 0;
    let mut fields = Vec::new();
    while fields.len() < 4 {
        while d[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if d[pos] == b'#' {
            while d[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        let s = pos;
        while !d[pos].is_ascii_whitespace() {
            pos += 1;
        }
        fields.push(String::from_utf8_lossy(&d[s..pos]).to_string());
    }
    pos += 1;
    let w: usize = fields[1].parse().unwrap();
    let h: usize = fields[2].parse().unwrap();
    (w, h, d[pos..pos + w * h * 3].to_vec())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (w, h, rgb) = read_ppm(&a[1]);
    let q: u8 = a[2].parse().unwrap();
    let mode = a.get(4).map(|s| s.as_str()).unwrap_or("420");
    let (ssx, ssy, np) = match mode {
        "444" => (0, 0, 3),
        "mono" => (1, 1, 1),
        _ => (1, 1, 3),
    };
    let (kr, kb) = (0.299f32, 0.114f32);
    let kg = 1.0 - kr - kb;
    let px = |x: usize, y: usize| {
        let o = (y * w + x) * 3;
        (rgb[o] as f32, rgb[o + 1] as f32, rgb[o + 2] as f32)
    };
    let mut yp = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let (r, g, b) = px(x, y);
            yp[y * w + x] = (kr * r + kg * g + kb * b).round().clamp(0.0, 255.0) as u8;
        }
    }
    let (cw, ch) = ((w + ssx) >> ssx, (h + ssy) >> ssy);
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for cy in 0..ch {
        for cx in 0..cw {
            let (mut r, mut g, mut b, mut n) = (0.0, 0.0, 0.0, 0.0);
            for yy in (cy << ssy)..((cy << ssy) + (1 << ssy)).min(h) {
                for xx in (cx << ssx)..((cx << ssx) + (1 << ssx)).min(w) {
                    let p = px(xx, yy);
                    r += p.0;
                    g += p.1;
                    b += p.2;
                    n += 1.0;
                }
            }
            let (r, g, b) = (r / n, g / n, b / n);
            let l = kr * r + kg * g + kb * b;
            u[cy * cw + cx] = ((b - l) / (2.0 * (1.0 - kb)) + 128.0).round().clamp(0.0, 255.0) as u8;
            v[cy * cw + cx] = ((r - l) / (2.0 * (1.0 - kr)) + 128.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    let mut planes: Vec<(&[u8], usize, usize, usize)> = vec![(&yp, w, w, h)];
    if np == 3 {
        planes.push((&u, cw, cw, ch));
        planes.push((&v, cw, cw, ch));
    }
    let speed: u8 = std::env::var("AV1_SPEED").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let tune = std::env::var("TUNE").map(|v| v != "0").unwrap_or(true);
    let cfg = FrameConfig {
        base_q: q,
        num_planes: np,
        ssx,
        ssy,
        uv_delta_q: (0, 0),
        qm: None,
        qm_dist: false,
        deltaq_variance_boost: std::env::var("DELTAQ").map(|v| v != "0").unwrap_or(false),
        tune_iq: tune,
        sharpness: 7,
        enable_cdef: true,
        speed,
        threads: std::env::var("AV1_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1),
    };
    let seq = SequenceHeader {
        profile: if np == 3 && ssx == 0 { 1 } else { 0 },
        level: 31,
        width: w,
        height: h,
        mono: np == 1,
        subsampled: ssx == 1,
        color: if np == 3 {
            Some(ColorDesc {
                primaries: 1,
                transfer: 13,
                matrix: std::env::var("MATRIX").map(|v| v.parse().unwrap()).unwrap_or(6),
            })
        } else {
            None
        },
        full_range: true,
        enable_cdef: true,
    };
    let t = std::time::Instant::now();
    let out = if a.get(5).map(String::as_str) == Some("iq") {
        let refs: Vec<&[u8]> = planes.iter().map(|p| p.0).collect();
        let chroma = match mode {
            "444" => avif_mini_encoder::Chroma::Yuv444,
            "mono" => avif_mini_encoder::Chroma::Mono,
            _ => avif_mini_encoder::Chroma::Yuv420,
        };
        let threads = std::env::var("AV1_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let (obus, recon) = avif_mini_encoder::encode_av1_threads(
            &refs,
            w,
            h,
            chroma,
            seq.color,
            true,
            q,
            avif_mini_encoder::Tune::Iq,
            speed,
            threads,
        )
        .unwrap();
        raw::Output { obus, recon }
    } else {
        raw::encode(cfg, &planes, &seq)
    };
    eprintln!("{} bytes in {:?}", out.obus.len(), t.elapsed());
    let mut f = vec![0x12, 0x00]; // temporal delimiter
    f.extend_from_slice(&out.obus);
    std::fs::write(format!("{}.obu", a[3]), f).unwrap();
    let mut r = Vec::new();
    for p in &out.recon {
        r.extend_from_slice(p);
    }
    std::fs::write(format!("{}.rec.yuv", a[3]), r).unwrap();
    let mut s = Vec::new();
    s.extend_from_slice(&yp);
    if np == 3 {
        s.extend_from_slice(&u);
        s.extend_from_slice(&v);
    }
    std::fs::write(format!("{}.src.yuv", a[3]), s).unwrap();
}
