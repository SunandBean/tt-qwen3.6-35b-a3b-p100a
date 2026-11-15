"""Where the weights come from.

Base weights: the Lottolabs TT-native checkpoint of Qwen3.6-35B-A3B (Apache-2.0), pinned to one revision. Experts in
GPTQ BFP4, projections and LM head in BFP8, norms / router / conv / embedding in BF16. Only `checkpoint/` is used.

BFP8 upgrade layers (default on): the routed + shared experts of a few layers in BFP8 instead of GPTQ BFP4, converted
round-to-nearest from the original BF16 weights of Qwen/Qwen3.6-35B-A3B (pinned revision) by `tools/convert_bfp8.py`.
Default set "dn:30-39,mtp;gu:36-39,mtp" (down projections of layers 30-39 and the MTP layer, gate/up of 36-39 and
the MTP layer): 272 tensorbins, 6.0 GB on disk, +2.84 GB on the card, KL vs BF16 0.0173 -> 0.0130.

Q36_BFP8_SOURCE picks where those files come from:
  hf     (default) download `bfp8/` of the model repo (BFP8_REPO) at Q36_BFP8_REVISION into the Hugging Face cache;
         files already in Q36_CACHE (see bf16; e.g. copied there for an offline machine) are used instead
  dir    read them from Q36_BFP8_DIR (e.g. the output of tools/convert_bfp8.py)
  bf16   convert them here from the BF16 original (Q36_BF16, or a ~67 GB download of the pinned Qwen revision),
         cached in Q36_CACHE (default: $TT_DIT_CACHE_DIR/bfp8 inside a tt-model container, else ~/.cache/tt-qwen36/bfp8)
  off    no upgrade layers: every expert is Lottolabs' GPTQ BFP4 (KL vs BF16 0.0173; needed for --ctx 262144)
Every file is checked against tt_qwen36/bfp8_sha256.txt before it is used (once per file; the result is remembered).

Overrides: Q36_LOTTO (a local copy of the checkpoint/ directory), Q36_BFP8_LAYERS (another layer set; only the
default set has published files and checksums), TT_MODEL_WEIGHTS_REVISION (set by tt-model from tt-model.yaml).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib

LOTTO_REPO = "Lottolabs/Qwen3.6-35B-A3B-TT-Mixed-BFP4-BFP8-P150"
LOTTO_REVISION = "c355c41d4fd079c6df3c4ba9bab1c40f37c17f09"
QWEN_REPO = "Qwen/Qwen3.6-35B-A3B"
QWEN_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"     # the revision Lottolabs quantized
BFP8_REPO = "sunandbean/qwen3.6-35b-a3b-p100a"
DEFAULT_BFP8_LAYERS = "dn:30-39,mtp;gu:36-39,mtp"
SHA_FILE = pathlib.Path(__file__).resolve().parent / "bfp8_sha256.txt"
NL = 40


class LazyDir(os.PathLike):
    """A directory resolved on first use (so importing the package never downloads anything)."""

    def __init__(self, resolve):
        self._resolve, self._p = resolve, None

    def path(self) -> pathlib.Path:
        if self._p is None:
            self._p = pathlib.Path(self._resolve())
        return self._p

    def __truediv__(self, other):
        return self.path() / other

    def __fspath__(self):
        return str(self.path())

    def __str__(self):
        return str(self.path())


def _snapshot(repo, revision, patterns):
    from huggingface_hub import snapshot_download
    return pathlib.Path(snapshot_download(repo, revision=revision, allow_patterns=patterns))


def lotto_dir() -> pathlib.Path:
    if os.environ.get("Q36_LOTTO"):
        return pathlib.Path(os.environ["Q36_LOTTO"])
    rev = os.environ.get("TT_MODEL_WEIGHTS_REVISION") or LOTTO_REVISION
    print(f"weights: {LOTTO_REPO}@{rev[:8]} checkpoint/ (Hugging Face cache)", flush=True)
    return _snapshot(LOTTO_REPO, rev, ["checkpoint/*", "LICENSE*", "README.md"]) / "checkpoint"


def bf16_dir() -> pathlib.Path:
    if os.environ.get("Q36_BF16"):
        return pathlib.Path(os.environ["Q36_BF16"])
    print(f"weights: {QWEN_REPO}@{QWEN_REVISION[:8]} BF16 safetensors (~67 GB, Hugging Face cache)", flush=True)
    return _snapshot(QWEN_REPO, QWEN_REVISION, ["*.safetensors", "*.json", "LICENSE*"])


def bfp8_source() -> str:
    s = os.environ.get("Q36_BFP8_SOURCE", "hf").strip().lower()
    if s not in ("hf", "dir", "bf16", "off"):
        raise ValueError(f"Q36_BFP8_SOURCE={s!r}: expected hf, dir, bf16 or off")
    return s


def bfp8_layers() -> str:
    """the layer set, '' when the upgrade is off"""
    if bfp8_source() == "off":
        return ""
    spec = os.environ.get("Q36_BFP8_LAYERS", os.environ.get("P100A_Q36_BFP8", DEFAULT_BFP8_LAYERS))
    return "" if spec.strip() in ("", "none", "off") else spec


def file_name(prefix: str, kind: str, g: int) -> str:
    """prefix 'layers.L' or 'mtp.layers.0', kind gate_up_proj | down_proj, group g 0..15 (16 routed experts each) or
    16 (the shared expert)"""
    return f"{prefix}.{kind}.g{g:02d}.tensorbin"


def files_for(promote: set) -> list:
    out = []
    for key, L in sorted(promote, key=lambda x: (x[0], str(x[1]))):
        prefix = "mtp.layers.0" if L == "mtp" else f"layers.{L}"
        kind = "gate_up_proj" if key == "gu" else "down_proj"
        out += [file_name(prefix, kind, g) for g in range(17)]
    return out


def checksums() -> dict:
    out = {}
    for line in SHA_FILE.read_text().splitlines():
        if line.strip():
            sha, name = line.split()
            out[name] = sha
    return out


def cache_dir() -> pathlib.Path:
    if os.environ.get("Q36_CACHE"):
        return pathlib.Path(os.environ["Q36_CACHE"])
    if os.environ.get("TT_DIT_CACHE_DIR"):                      # tt-model's per-model writable cache mount
        return pathlib.Path(os.environ["TT_DIT_CACHE_DIR"]) / "bfp8"
    return pathlib.Path.home() / ".cache" / "tt-qwen36" / "bfp8"


def _verified_db() -> pathlib.Path:
    return cache_dir() / "verified.json"


def verify(path: pathlib.Path, sums: dict) -> None:
    """sha256 against the shipped list, once per (file, size, mtime)"""
    name = path.name
    if name not in sums:
        raise RuntimeError(f"{name}: not in {SHA_FILE.name}; only the default layer set has published checksums "
                           "(Q36_BFP8_VERIFY=0 skips the check for a custom set)")
    st = path.stat()
    key = f"{path.resolve()}:{st.st_size}:{int(st.st_mtime)}"
    db = _verified_db()
    try:
        seen = json.loads(db.read_text())
    except (OSError, ValueError):
        seen = {}
    if seen.get(key) == sums[name]:
        return
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    if h.hexdigest() != sums[name]:
        raise RuntimeError(f"{path}: sha256 {h.hexdigest()} != expected {sums[name]} (corrupt or a different "
                           "conversion); delete it and start again")
    seen[key] = sums[name]
    try:
        db.parent.mkdir(parents=True, exist_ok=True)
        db.write_text(json.dumps(seen))
    except OSError:
        pass


class Bfp8Files:
    """Resolves the BFP8 upgrade tensorbins for a layer set: path(name) -> a verified local file, or None when the
    file has to be converted here (source bf16; convert() then writes it to cache_dir())."""

    def __init__(self, promote: set):
        self.source = bfp8_source()
        self.names = files_for(promote)
        self.check = os.environ.get("Q36_BFP8_VERIFY", "1") != "0"
        self.sums = checksums() if self.check else {}
        self.dir = None
        if not self.names:
            return
        if self.source == "dir":
            self.dir = pathlib.Path(os.environ["Q36_BFP8_DIR"])
        elif self.source == "hf" and all((cache_dir() / n).exists() for n in self.names):
            self.dir = cache_dir()                  # provisioned beforehand (offline copy of bfp8/): no download
            print(f"weights: BFP8 upgrade layers from {self.dir} ({len(self.names)} files)", flush=True)
        elif self.source == "hf":
            rev = os.environ.get("Q36_BFP8_REVISION", "main")
            print(f"weights: BFP8 upgrade layers {BFP8_REPO}@{rev[:8]} bfp8/ ({len(self.names)} files, "
                  "6.0 GB for the default set; Hugging Face cache)", flush=True)
            self.dir = _snapshot(BFP8_REPO, rev, [f"bfp8/{n}" for n in self.names]) / "bfp8"
        else:
            self.dir = cache_dir()
            self.dir.mkdir(parents=True, exist_ok=True)
        missing = [n for n in self.names if not (self.dir / n).exists()]
        if missing and self.source != "bf16":
            raise FileNotFoundError(f"BFP8 upgrade layers: {len(missing)} of {len(self.names)} files missing in "
                                    f"{self.dir} (first: {missing[0]}). Q36_BFP8_SOURCE=off serves without them.")

    def path(self, name: str):
        p = self.dir / name
        if not p.exists():
            return None
        if self.check:
            verify(p, self.sums)
        return p

    def out(self, name: str) -> pathlib.Path:
        return self.dir / name
