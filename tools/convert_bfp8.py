"""Convert the BFP8 upgrade layers from the original BF16 weights (CPU only; no card needed).

    python tools/convert_bfp8.py --out DIR [--layers "dn:30-39,mtp;gu:36-39,mtp"] [--bf16 DIR] [--only N] [--check]

Reads Qwen/Qwen3.6-35B-A3B at the revision Lottolabs quantized (tt_qwen36/weights.py QWEN_REVISION; --bf16 or Q36_BF16
for a local copy, otherwise ~67 GB are downloaded into the Hugging Face cache) and writes, per expert matrix and
layer, 17 ttnn tensorbins in bfloat8_b (16 groups of 16 routed experts + the shared expert), round-to-nearest, exactly
as the server loads them, plus DIR/SHA256SUMS. The conversion is deterministic: --check compares every file with
tt_qwen36/bfp8_sha256.txt (the published files). Host RAM ~0.5 GB; ~10 s per file.

The output directory is what Q36_BFP8_SOURCE=dir Q36_BFP8_DIR=DIR serves, and what the model repo ships as bfp8/.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def _mock_cluster():
    """ttnn's host-side BFP8 packing wants a cluster description; without a card use tt-metal's P100 descriptor"""
    if os.path.exists("/dev/tenstorrent") or os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):
        return
    rel = "tt_metal/third_party/tt-cluster-descriptors/blackhole/p100_cluster_desc/p100_cluster_desc.yaml"
    for root in (os.environ.get("TT_METAL_HOME", ""), "/tt-metal", "/opt/tt-metal"):
        if root and os.path.exists(os.path.join(root, rel)):
            os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = os.path.join(root, rel)
            return


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    from tt_qwen36.weights import DEFAULT_BFP8_LAYERS, checksums, files_for
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default=DEFAULT_BFP8_LAYERS)
    ap.add_argument("--bf16", default="", help="local copy of the BF16 checkpoint (else Q36_BF16 or a download)")
    ap.add_argument("--only", type=int, default=0, help="convert only the first N files (a quick reproducibility check)")
    ap.add_argument("--check", action="store_true", help="compare with tt_qwen36/bfp8_sha256.txt")
    a = ap.parse_args()
    if a.bf16:
        os.environ["Q36_BF16"] = a.bf16
    _mock_cluster()
    import ttnn
    from tt_qwen36.q36.model import Q36Weights, convert_group
    from tt_qwen36.ref.q36_weights import Q36Weights as HostW

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = files_for(Q36Weights.parse_promote(a.layers))
    if a.only:
        names = names[:a.only]
    hw = HostW("bf16")
    want = checksums() if a.check else {}
    sums, bad, t0 = [], 0, time.time()
    for i, name in enumerate(names):
        prefix, kind, g = name.rsplit(".", 3)[0], name.rsplit(".", 3)[1], int(name.rsplit(".", 3)[2][1:])
        dst = out / name
        if not dst.exists():
            t = convert_group(hw, prefix, kind, g)
            tmp = dst.with_name(dst.name.replace(".tensorbin", ".part.tensorbin"))
            ttnn.dump_tensor(str(tmp), t)
            os.replace(tmp, dst)
        s = sha256(dst)
        sums.append(f"{s}  {name}")
        status = ""
        if a.check:
            status = "ok" if want.get(name) == s else f"MISMATCH (expected {want.get(name)})"
            bad += status != "ok"
        print(f"[{i + 1}/{len(names)}] {name} {dst.stat().st_size / 1e6:.1f} MB {status} {time.time() - t0:.0f} s",
              flush=True)
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    if a.check:
        print(f"check: {len(names) - bad}/{len(names)} identical to tt_qwen36/bfp8_sha256.txt")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
