#!/usr/bin/env python3
"""Compare two release rawenc binaries; require identical OBU/reconstruction.

Usage: python3 tools/bench-compare.py BEFORE AFTER IMAGE.ppm [IMAGE.ppm ...]
Times are encode-only (reported by rawenc), with alternating execution order.
"""

import argparse
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--memory", action="store_true", help="also measure process peak RSS with GNU time (KiB)")
    parser.add_argument("--iq", action="store_true", help="use rawenc's public API tune-IQ mode (both binaries must support it)")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    binaries = {"before": args.before.resolve(), "after": args.after.resolve()}
    # Developer switches can change both speed and the encoded result.
    env = {k: v for k, v in os.environ.items() if not k.startswith("AV1_")}
    header = "image\tqindex\tchroma\tbefore(s)\tafter(s)\treduction(%)"
    if args.memory:
        header += "\tbefore_RSS(KiB)\tafter_RSS(KiB)"
    print(header, flush=True)
    with tempfile.TemporaryDirectory(prefix="avif-bench-") as directory:
        for image in args.images:
            for q, mode in [(80, "420"), (128, "420"), (128, "444"), (128, "mono")]:
                times = {label: [] for label in binaries}
                rss = {label: [] for label in binaries}
                for run in range(args.runs):
                    order = ["before", "after"] if run % 2 == 0 else ["after", "before"]
                    for label in order:
                        prefix = Path(directory) / label
                        command = [str(binaries[label]), str(image.resolve()), str(q), str(prefix), mode]
                        if args.iq:
                            command.append("iq")
                        if args.memory:
                            command = ["/usr/bin/time", "-f", "PEAK_RSS_KIB=%M", *command]
                        result = subprocess.run(
                            command,
                            env=env, capture_output=True, text=True, check=True,
                        )
                        match = re.search(r"in ([0-9.]+)(ms|s|µs|ns)", result.stderr)
                        if match is None:
                            raise RuntimeError(f"Cannot parse rawenc timing: {result.stderr}")
                        seconds = float(match[1]) * {"s": 1, "ms": 1e-3, "µs": 1e-6, "ns": 1e-9}[match[2]]
                        times[label].append(seconds)
                        if args.memory:
                            peak = re.search(r"PEAK_RSS_KIB=(\d+)", result.stderr)
                            if peak is None:
                                raise RuntimeError(f"Cannot parse peak RSS: {result.stderr}")
                            rss[label].append(int(peak[1]))
                    for suffix in [".obu", ".rec.yuv"]:
                        before = (Path(directory) / ("before" + suffix)).read_bytes()
                        after = (Path(directory) / ("after" + suffix)).read_bytes()
                        if before != after:
                            raise RuntimeError(f"Output differs: {image}, q={q}, {mode}, {suffix}")
                old, new = (statistics.median(times[label]) for label in ["before", "after"])
                row = f"{image.name}\t{q}\t{mode}\t{old:.6f}\t{new:.6f}\t{100 * (1 - new / old):.1f}"
                if args.memory:
                    row += f"\t{max(rss['before'])}\t{max(rss['after'])}"
                print(row, flush=True)


if __name__ == "__main__":
    main()
