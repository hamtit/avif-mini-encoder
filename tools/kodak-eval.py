#!/usr/bin/env python3
"""Quality / size / time evaluation on Kodak images, with BD-rate comparison.

  python3 tools/kodak-eval.py run  RAWENC LABEL --kodak DIR [--env K=V ...] [--imgs 01,03] [--qs 40,80,..] [-j 4]
  python3 tools/kodak-eval.py bd   BASE_LABEL NEW_LABEL

`run` encodes DIR/kodimNN.png (https://r0k.us/graphics/kodak/) with the public
encode_av1() path (`rawenc ... 420 iq`) and records size, encode time, luma PSNR
and SSIMULACRA2 per (image, qindex) in res_LABEL.json. The encoder's own
reconstruction is used (it matches aomdec bit for bit).
`bd` prints per-image and mean BD-rate (negative = NEW is smaller at equal
quality) for PSNR and SSIMULACRA2.

Needs numpy, pillow, scipy, ssimulacra2 (wheels are in the repository root) and ffmpeg.
Encode times measured with -j > 1 are noisy (about +-5% here); use them only roughly.
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
from multiprocessing import Pool


def one(a):
    import numpy as np
    from PIL import Image
    import ssimulacra2 as s2

    rawenc, env, kodak, img, q = a
    d = tempfile.mkdtemp(prefix="kodak-eval-")
    e = dict(os.environ)
    e.update(env)
    ppm = f"{kodak}/kodim{img}.ppm"
    png = f"{kodak}/kodim{img}.png"
    r = subprocess.run([rawenc, ppm, str(q), d + "/o", "420", "iq"], capture_output=True, text=True, env=e, check=True)
    m = re.search(r"(\d+) bytes in ([\d.]+)(ms|s)", r.stderr)
    size = int(m.group(1))
    t = float(m.group(2)) / (1000 if m.group(3) == "ms" else 1)
    src = np.array(Image.open(png).convert("RGB"))
    h, w = src.shape[:2]
    y = np.frombuffer(open(d + "/o.rec.yuv", "rb").read()[: w * h], np.uint8).reshape(h, w)
    sy = np.frombuffer(open(d + "/o.src.yuv", "rb").read()[: w * h], np.uint8).reshape(h, w)
    mse = ((y.astype(float) - sy) ** 2).mean()
    psnr = 10 * np.log10(255**2 / max(mse, 1e-9))
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuvj420p", "-s", f"{w}x{h}",
         "-i", d + "/o.rec.yuv", "-vf", "scale=in_range=pc:in_color_matrix=bt601:out_range=pc",
         "-pix_fmt", "rgb24", d + "/o.png"],
        check=True,
    )
    sc = s2.compute_ssimulacra2(png, d + "/o.png")
    subprocess.run(["rm", "-rf", d])
    return dict(img=img, q=q, size=size, time=t, psnr=float(psnr), s2=float(sc))


def bd(r1, r2):
    import numpy as np
    from scipy.interpolate import PchipInterpolator

    def prep(r):
        r = sorted(r, key=lambda x: x[1])
        return np.array([x[1] for x in r]), np.log([x[0] for x in r])

    m1, l1 = prep(r1)
    m2, l2 = prep(r2)
    lo, hi = max(m1.min(), m2.min()), min(m1.max(), m2.max())
    if hi <= lo:
        return float("nan")
    xs = np.linspace(lo, hi, 200)
    return (np.exp((PchipInterpolator(m2, l2)(xs) - PchipInterpolator(m1, l1)(xs)).mean()) - 1) * 100


def main():
    import numpy as np

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("rawenc")
    r.add_argument("label")
    r.add_argument("--kodak", required=True)
    r.add_argument("--env", nargs="*", default=[])
    r.add_argument("--imgs", default="01,03,05,08,13,19,22,23")
    r.add_argument("--qs", default="40,80,120,160,200")
    r.add_argument("-j", type=int, default=4)
    b = sub.add_parser("bd")
    b.add_argument("base")
    b.add_argument("new")
    a = p.parse_args()
    if a.cmd == "run":
        env = dict(x.split("=", 1) for x in a.env)
        # Convert before starting workers: they share the PPM of an image.
        for i in a.imgs.split(","):
            ppm, png = f"{a.kodak}/kodim{i}.ppm", f"{a.kodak}/kodim{i}.png"
            if not os.path.exists(ppm):
                tmp = ppm + ".tmp.ppm"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", png, tmp], check=True)
                os.replace(tmp, ppm)
        jobs = [(os.path.abspath(a.rawenc), env, os.path.abspath(a.kodak), i, int(q))
                for i in a.imgs.split(",") for q in a.qs.split(",")]
        with Pool(a.j) as pl:
            res = pl.map(one, jobs, chunksize=1)
        json.dump(res, open(f"res_{a.label}.json", "w"))
        print(a.label, "total size", sum(x["size"] for x in res), "time", round(sum(x["time"] for x in res), 2))
    else:
        A = json.load(open(f"res_{a.base}.json"))
        B = json.load(open(f"res_{a.new}.json"))
        imgs = sorted({x["img"] for x in A})
        print("img   ", " ".join(imgs))
        for k in ("psnr", "s2"):
            v = [bd([(x["size"], x[k]) for x in A if x["img"] == i], [(x["size"], x[k]) for x in B if x["img"] == i]) for i in imgs]
            print(k.ljust(5), " ".join(f"{x:+.1f}" for x in v), " mean %+.2f%%" % np.nanmean(v))
        ta, tb = sum(x["time"] for x in A), sum(x["time"] for x in B)
        print("time %.2f -> %.2f (%+.1f%%)  size %d -> %d" % (ta, tb, (tb / ta - 1) * 100, sum(x["size"] for x in A), sum(x["size"] for x in B)))


if __name__ == "__main__":
    main()
