"""Qwen3.6-35B-A3B weights under one naming (HF text names without the `model.language_model.` prefix:
`layers.L.linear_attn.in_proj_qkv.weight`, `embed_tokens.weight`, `lm_head.weight`, `norm.weight`, `mtp.*`).

Two sources (where they come from: tt_qwen36/weights.py):
  lotto  the Lottolabs TT-native checkpoint (Apache-2.0): BF16 safetensors for norms, router, conv, A_log, dt_bias,
         embedding; BFP8 / BFP4 ttnn tile tensors (`ttnn.load_tensor`) for every projection, the LM head and the
         experts (16 experts per grouped file). BFP4/BFP8 values are exact in bf16, so a dequantized tensor is
         returned as float32 without loss. Needs ttnn.
  bf16   the original Qwen safetensors (the revision Lottolabs quantized); only used to convert the BFP8 upgrade layers.
Experts are served per expert (`expert(L, e)` -> gate [512,2048], up [512,2048], down [2048,512], float32).
"""
from __future__ import annotations

import json
import os
import pathlib
from collections import OrderedDict

import torch

from ..weights import LazyDir, bf16_dir, lotto_dir

LOTTO = LazyDir(lotto_dir)
BF16 = LazyDir(bf16_dir)
GROUP = 16

# canonical name -> Lotto manifest name (only where they differ)
_LOTTO_ALIAS = {"embed_tokens.weight": "tok_embeddings.weight", "lm_head.weight": "output.weight"}


def _lotto_name(n: str) -> str:
    if n in _LOTTO_ALIAS:
        return _LOTTO_ALIAS[n]
    return n.replace("linear_attn.in_proj_qkv.", "linear_attn.qkv_proj.")


class Q36Weights:
    def __init__(self, source: str = "lotto", group_cache: int = 48):
        self.source = source
        self.cfg = json.load(open((LOTTO if source == "lotto" else BF16) / "config.json"))["text_config"]
        self._st = {}
        self._groups: OrderedDict = OrderedDict()
        self.group_cache = group_cache
        self._mm_files = {}
        self.fast = True               # per-expert raw tile decode (False: whole group files through ttnn)
        if source == "lotto":
            self.man = json.load(open(LOTTO / "native_manifest.json"))
            self.T = self.man["tensors"]
        else:
            idx = json.load(open(BF16 / "model.safetensors.index.json"))["weight_map"]
            self.T = {}
            for k, f in idx.items():
                if k.startswith("model.visual"):
                    continue
                c = k.replace("model.language_model.", "") if k.startswith("model.language_model.") else k
                self.T[c] = f

    # ---- raw access
    def _safe(self, path: pathlib.Path):
        from safetensors import safe_open
        p = str(path)
        if p not in self._st:
            self._st[p] = safe_open(p, framework="pt", device="cpu")
        return self._st[p]

    def _ttnn(self, path: pathlib.Path) -> torch.Tensor:
        import ttnn
        return ttnn.to_torch(ttnn.load_tensor(str(path))).float()

    def has(self, n: str) -> bool:
        if self.source == "lotto":
            if n.endswith("linear_attn.conv1d.weight"):
                return n.replace("conv1d", "q_conv") in self.T
            return _lotto_name(n) in self.T
        return n in self.T

    def get(self, n: str) -> torch.Tensor:
        """float32 tensor in the HF source orientation ([out, in] for linears)."""
        if self.source == "bf16":
            f = self.T[n]
            key = n if n.startswith(("mtp.", "lm_head")) else "model.language_model." + n
            return self._safe(BF16 / f).get_tensor(key).float()
        if n.endswith("linear_attn.conv1d.weight"):          # Lotto splits the conv into q / k / v channels
            return torch.cat([self.get(n.replace("conv1d", x)) for x in ("q_conv", "k_conv", "v_conv")], 0)
        ln = _lotto_name(n)
        e = self.T[ln]
        path = LOTTO / e["file"]
        if e["storage"] == "safetensors-lossless":
            return self._safe(path).get_tensor(ln).float()
        assert "expert" not in e["orientation"], "experts: use expert()"
        t = self._ttnn(path)
        assert e["orientation"] == "input,output", e
        return t.reshape(t.shape[-2], t.shape[-1]).T.contiguous()

    def rows(self, n: str, ids) -> torch.Tensor:
        """embedding rows (float32)"""
        if self.source == "bf16":
            t = self._safe(BF16 / self.T[n]).get_slice("model.language_model." + n)
        else:
            ln = _lotto_name(n)
            t = self._safe(LOTTO / self.T[ln]["file"]).get_slice(ln)
        return torch.stack([t[int(i):int(i) + 1][0] for i in ids]).float()

    # ---- experts
    def _group(self, prefix: str, kind: str, g: int) -> torch.Tensor:
        key = (prefix, kind, g)
        if key in self._groups:
            self._groups.move_to_end(key)
            return self._groups[key]
        name = f"{prefix}.mlp.experts.{kind}"
        stem = pathlib.Path(self.T[name]["file"]).stem
        t = self._ttnn(LOTTO / "expert_groups" / stem / f"{g:02d}.tensorbin")      # [1,16,in,out]
        t = t.reshape(t.shape[-3], t.shape[-2], t.shape[-1]).to(torch.bfloat16)   # exact (BFP4 in bf16)
        self._groups[key] = t
        while len(self._groups) > self.group_cache:
            self._groups.popitem(last=False)
        return t

    def _expert_raw(self, prefix: str, kind: str, e: int) -> torch.Tensor:
        """one expert's [in, out] matrix decoded straight from its BFP4 tiles (tt_qwen36/bfp4.py; the tile data is the
        tail of the tensorbin, experts in order, tiles row-major): ~10 ms instead of a whole 16-expert file"""
        import numpy as np
        from tt_qwen36.bfp4 import decode_tiles
        name = f"{prefix}.mlp.experts.{kind}"
        stem = pathlib.Path(self.T[name]["file"]).stem
        g, i = divmod(e, GROUP)
        path = LOTTO / "expert_groups" / stem / f"{g:02d}.tensorbin"
        rows, cols = (2048, 1024) if kind == "gate_up_proj" else (512, 2048)
        per = rows * cols // 1024 * 576
        size = path.stat().st_size
        with open(path, "rb") as fh:                      # (a memmap per file ran out of file descriptors)
            fh.seek(size - GROUP * per + i * per)
            raw = np.frombuffer(fh.read(per), np.uint8)
        return torch.from_numpy(decode_tiles(raw, rows, cols))

    def expert(self, prefix: str, e: int):
        """prefix 'layers.L' or 'mtp.layers.0'. Returns gate [F,H], up [F,H], down [H,F] float32."""
        F = self.cfg["moe_intermediate_size"]
        if self.source == "bf16":
            st = self._safe(BF16 / self.T[f"{prefix}.mlp.experts.gate_up_proj"])
            pre = "" if prefix.startswith("mtp.") else "model.language_model."
            gu = st.get_slice(f"{pre}{prefix}.mlp.experts.gate_up_proj")[e].float()    # [2F, H]
            dn = self._safe(BF16 / self.T[f"{prefix}.mlp.experts.down_proj"]).get_slice(
                f"{pre}{prefix}.mlp.experts.down_proj")[e].float()                    # [H, F]
            return gu[:F], gu[F:], dn
        g, i = divmod(e, GROUP)
        if (prefix, "gate_up_proj", g) in self._groups or not self.fast:
            gu = self._group(prefix, "gate_up_proj", g)[i].float().T                   # native [H, 2F] -> [2F, H]
            dn = self._group(prefix, "down_proj", g)[i].float().T                      # native [F, H] -> [H, F]
        else:
            gu, dn = self._expert_raw(prefix, "gate_up_proj", e).T, self._expert_raw(prefix, "down_proj", e).T
        return gu[:F], gu[F:], dn
