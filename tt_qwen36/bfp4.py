"""Decode tt-metal bfloat4_b tiles from raw bytes (for the Lottolabs GPTQ-BFP4 pool), validated against ttnn.

A 32x32 tile = 576 bytes: 64 shared exponents (one per 16-value face row: faces 0..3 = top-left, top-right,
bottom-left, bottom-right; 16 rows each), then 512 bytes of 4-bit sign|mantissa codes in the same order, low nibble first.
value = (-1)^sign * (mantissa / 4) * 2^(exponent - 127)        (layout checked by `python -m tt_qwen36.bfp4 --selftest`)
"""
from __future__ import annotations

import numpy as np


def decode_tiles(raw: np.ndarray, rows: int, cols: int, mant_div: float = 4.0, nib_hi_first: bool = False) -> np.ndarray:
    """raw uint8 [rows/32 * cols/32 * 576] (tiles row-major) -> float32 [rows, cols]."""
    nt_r, nt_c = rows // 32, cols // 32
    t = raw.reshape(nt_r * nt_c, 576)
    exp = t[:, :64].astype(np.int32)                                   # [tiles, 64] = [tiles, face(4), row(16)]
    data = t[:, 64:]                                                   # [tiles, 512]
    lo, hi = data & 0x0F, data >> 4
    nib = np.stack([hi, lo] if nib_hi_first else [lo, hi], -1).reshape(-1, 1024)   # [tiles, 1024] = face,row,col16
    sign = np.where(nib & 0x8, -1.0, 1.0)
    mant = (nib & 0x7).astype(np.float32) / mant_div
    scale = np.ldexp(1.0, exp - 127).astype(np.float32)               # [tiles, 64]
    v = (sign * mant).reshape(-1, 64, 16) * scale[:, :, None]          # [tiles, face*16+row, 16]
    v = v.reshape(-1, 4, 16, 16)                                       # face, row, col
    top = np.concatenate([v[:, 0], v[:, 1]], axis=2)                   # rows 0..15: faces 0,1
    bot = np.concatenate([v[:, 2], v[:, 3]], axis=2)
    tile = np.concatenate([top, bot], axis=1)                          # [tiles, 32, 32]
    return tile.reshape(nt_r, nt_c, 32, 32).transpose(0, 2, 1, 3).reshape(rows, cols).astype(np.float32)


if __name__ == "__main__":
    import itertools, os, tempfile
    import torch, ttnn
    torch.manual_seed(0)
    w = torch.randn(1, 1, 64, 96) * torch.logspace(-3, 1, 96)
    t = ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
    want = ttnn.to_torch(t).float()[0, 0].numpy()
    f = os.path.join(tempfile.mkdtemp(), "t.tensorbin")
    ttnn.dump_tensor(f, t)
    b = np.fromfile(f, np.uint8)
    n = (64 // 32) * (96 // 32) * 576
    print("file bytes", b.size, "tile bytes", n)
    for off, md, hf in itertools.product(range(0, b.size - n + 1, 1), (2.0, 4.0, 8.0), (False, True)):
        try:
            got = decode_tiles(b[off:off + n], 64, 96, md, hf)
        except Exception:
            continue
        if np.array_equal(got, want):
            print("MATCH at offset", off, "from end", b.size - off - n, "mant_div", md, "nib_hi_first", hf)
            break
    else:
        print("no exact match")
