//! Benchmark: colour + alpha encoding, sequential vs `encode_av1_pair`.
//! usage: pairbench in.ppm qindex threads
//! The alpha plane is a synthetic mask (soft circle) of the same size.
use avif_mini_encoder::{Chroma, ColorDesc, EncodeJob, Tune, encode_av1, encode_av1_pair};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d = std::fs::read(&a[1]).unwrap();
    let mut it = d.split(|b| b.is_ascii_whitespace()).filter(|t| !t.is_empty());
    it.next();
    let w: usize = String::from_utf8_lossy(it.next().unwrap()).parse().unwrap();
    let h: usize = String::from_utf8_lossy(it.next().unwrap()).parse().unwrap();
    let rgb = &d[d.len() - w * h * 3..];
    let q: u8 = a[2].parse().unwrap();
    let threads: usize = a[3].parse().unwrap();
    let y: Vec<u8> = (0..w * h)
        .map(|i| ((rgb[i * 3] as u32 * 299 + rgb[i * 3 + 1] as u32 * 587 + rgb[i * 3 + 2] as u32 * 114) / 1000) as u8)
        .collect();
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let u = vec![128u8; cw * ch];
    let alpha: Vec<u8> = (0..w * h)
        .map(|i| {
            let (x, yy) = ((i % w) as f32 - w as f32 / 2.0, (i / w) as f32 - h as f32 / 2.0);
            let r = (x * x + yy * yy).sqrt() / (w.min(h) as f32 / 2.0);
            ((1.2 - r).clamp(0.0, 1.0) * 255.0) as u8
        })
        .collect();
    let col = [&y[..], &u[..], &u[..]];
    let alp = [&alpha[..]];
    let color = Some(ColorDesc { primaries: 1, transfer: 13, matrix: 6 });
    let cj = EncodeJob {
        planes: &col,
        width: w,
        height: h,
        chroma: Chroma::Yuv420,
        color,
        full_range: true,
        qindex: q,
        tune: Tune::Iq,
        speed: 5,
    };
    let aj = EncodeJob {
        planes: &alp,
        width: w,
        height: h,
        chroma: Chroma::Mono,
        color: None,
        full_range: true,
        qindex: q,
        tune: Tune::Psnr,
        speed: 5,
    };
    let t = std::time::Instant::now();
    let (c, al) = if threads == 0 {
        let c = encode_av1(cj.planes, w, h, cj.chroma, color, true, q, Tune::Iq, 5);
        let al = encode_av1(aj.planes, w, h, aj.chroma, None, true, q, Tune::Psnr, 5);
        (c, al)
    } else {
        encode_av1_pair(cj, aj, threads)
    };
    let (c, al) = (c.unwrap(), al.unwrap());
    eprintln!("{} bytes in {:?} (color {} + alpha {})", c.0.len() + al.0.len(), t.elapsed(), c.0.len(), al.0.len());
}
