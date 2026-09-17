#!/usr/bin/env python3
"""Welch t statistic per metric between validation/orig_<mode>_*.txt and validation/new_<mode>_*.txt.

Runs of all images are pooled per metric (PSNR, SSIM, BER after each attack)."""
import glob
import math
import os
import re
import sys
from collections import defaultdict


def read(pattern):
    samples = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        for line in open(path):
            run = re.match(r"run seed=\d+ .*psnr=(\S+) ssim=(\S+)", line)
            if run:
                samples["psnr"].append(float(run.group(1)))
                samples["ssim"].append(float(run.group(2)))
            attack = re.match(r"  (?!MEAN)(.+?)\s+ber=(\S+) ties=", line)
            if attack:
                samples["ber " + attack.group(1)].append(float(attack.group(2)))
    return samples


def welch(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    spread = math.sqrt(va / len(a) + vb / len(b))
    return ma, mb, (mb - ma) / spread if spread > 0 else 0.0


def main():
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation")
    largest = 0.0
    for mode in ("base", "quad"):
        orig, new = read(f"{here}/orig_{mode}_*.txt"), read(f"{here}/new_{mode}_*.txt")
        print(f"{mode}: {len(orig['psnr'])} original runs, {len(new['psnr'])} new runs")
        for metric in orig:
            ma, mb, t = welch(orig[metric], new[metric])
            largest = max(largest, abs(t))
            print(f"  {metric:32s} orig={ma:.6f} new={mb:.6f} t={t:+.2f}")
    print(f"largest |t| = {largest:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
