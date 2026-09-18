#!/usr/bin/env python3
"""Writes a few valid inputs per fuzz target, so the fuzzers start from accepted images
instead of having to discover the input layout (see fuzz_common.h, takeMat)."""
import os
import random
import struct
import sys

CV_8UC1_INDEX = 0   # index into kAnyType / kGrayOrColor


def mat(rows, cols, rng, roi=0, flat=None):
    pixels = bytes([flat] * (rows * cols)) if flat is not None else bytes(rng.randrange(256) for _ in range(rows * cols))
    return struct.pack("<HHBB", rows, cols, CV_8UC1_INDEX, roi) + pixels


def write(directory, name, payload):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, name), "wb") as f:
        f.write(payload)


def main(out):
    rng = random.Random(1)
    # takeMat draws sides as value % (max_side + 1), so sides below max_side encode as themselves.
    for i, (rows, cols) in enumerate([(8, 8), (32, 32), (64, 64), (33, 47), (96, 96)]):
        head = struct.pack("<BBBH", i, i % 4, i & 1, 0xFFFF)
        write(f"{out}/extract", f"seed{i}", head + mat(rows, cols, rng, roi=i & 1))
    for i, (rows, cols, quadrants) in enumerate([(8, 8, 0), (16, 24, 0), (40, 40, 0), (32, 32, 1), (19, 21, 0)]):
        head = struct.pack("<BQBB", i, 1000 + i, i, quadrants)
        write(f"{out}/embed", f"seed{i}", head + mat(32, 32, rng) + mat(rows, cols, rng, roi=i & 1))
    for i, flat in enumerate([None, 0, 255, 128, None, None, None, None]):
        head = struct.pack("<BQBB", i, 7 + i, i & 1, i % 4)
        block = bytes([flat] * 64) if flat is not None else bytes(rng.randrange(256) for _ in range(64))
        write(f"{out}/block", f"seed{i}", head + block)
    for i, quality in enumerate([1, 50, 70, 80, 100]):
        head = struct.pack("<HdB", quality + 20, 1.1, 0)
        write(f"{out}/kernels", f"seed{i}", head + bytes(rng.randrange(256) for _ in range(64 + 128)))
    for which in range(13):
        head = struct.pack("<BidB", which, 3, 1.1, 1)
        write(f"{out}/metrics_attacks", f"seed{which}", head + mat(16, 16, rng) + bytes(rng.randrange(256) for _ in range(256)))

    # fuzz_schemes_json takes the file contents as they are.
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "embedding_schemes.json"), "rb") as f:
        write(f"{out}/schemes_json", "repo_file", f.read())
    small = [
        b'{"a":{"REG0":[[0,1]],"REG1":[[1,0]],"ZONE0":[[0,1],[1,0]]}}',
        b'\xef\xbb\xbf{ "s\\u00e9\\ud83d\\ude00" : { "name" : "n \\"q\\" \\/", "note\\n" : "\\t", "extra" : [1.5e-3, true, null, {"k": []}],\n'
        b'  "REG0" : [ [7, 7] ], "REG1" : [ [0, 0] ], "ZONE0" : [ [0, 0] ] } }\n',
        b'{"a":{"REG0":[[0,1]],"REG1":[[1,0]],"ZONE0":[[0,1]]},"b":{"description":"","REG0":[[2,2]],"REG1":[[3,3]],"ZONE0":[[3,3],[2,2]]}}',
    ]
    for i, text in enumerate(small):
        write(f"{out}/schemes_json", f"seed{i}", text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fuzz/seeds")
