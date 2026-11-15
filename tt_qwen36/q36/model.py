"""Qwen3.6-35B-A3B on one P100a, everything on the card (no CPU experts, no host step inside a pass).

A pass reads T consecutive tokens at positions pos..pos+T-1 (T = drafts+1 for decode/verify, 28 for prompts) through
all 40 layers and the LM head; it does not move the recurrent state. commit(n) keeps the first n rows (GDN state,
conv history, position), exactly as tt_qwen36/verify.py does for Qwen3.8 (same chunk form of the gated delta rule, see
its docstring). Every vector is a [1,1,32,N] TILE tensor whose rows 0..T-1 are the tokens. Rows >= T start at zero
(embedding) and pick up finite garbage from the GDN conv / chunk math; nothing reads them for a valid row (matmuls
are row-local, the solve's upper triangle is exactly 0 and the values stay finite, routing and SDPA use rows < T,
commit and MTP select valid rows), so they are not re-zeroed per layer (P100A_Q36_MASK=1 does).

Per layer: x = rms1p(X) -> GDN (30 layers) or gated attention (10) -> X += y; h = rms1p(X) -> router (fp32) ->
route kernel (top 8, weights) -> MoE kernels on the active experts + shared expert -> X += y.
Weights: the Lottolabs checkpoint as stored (BFP8 projections and LM head, BFP4 experts / shared expert, BF16
norms / router / conv / embedding); nothing re-quantized. Formulas: the float32 reference of the port (not shipped here).
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import time

import numpy as np
import torch
import ttnn

from tt_qwen36.ref.q36_weights import LOTTO

from ..kkv import KvWrite
from ..weights import file_name
from .kmoe import Q36Moe

RM, TILE = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
H, NL, NE, F = 2048, 40, 256, 512
NK, NV, DK = 16, 32, 128            # GDN: key heads, value heads, head dim
NQ, NKV, HD, RD = 16, 2, 256, 64    # attention: q heads, kv heads, head dim, rotary dims
QKV = 2 * NK * DK + NV * DK         # 8192
VOC = 248320
EPS = 1e-6
NB = 32                              # KV page size


def is_attn(L):
    return L % 4 == 3


def convert_group(hw, prefix: str, kind: str, g: int):
    """One BFP8 upgrade tensorbin (host ttnn tensor, bfloat8_b TILE): group g < 16 = routed experts 16g..16g+15 as
    [1,16,in,out]; g = 16 = the shared expert as [1,1,in,out]. Round-to-nearest from the BF16 original `hw`
    (tt_qwen36.ref.q36_weights.Q36Weights("bf16")); deterministic, so the files match bfp8_sha256.txt."""
    if g < 16:
        mats = []
        for e in range(16 * g, 16 * g + 16):
            gt, ut, dt = hw.expert(prefix, e)                  # [F,H], [F,H], [H,F] fp32
            mats.append(torch.cat([gt, ut], 0).T if kind == "gate_up_proj" else dt.T)
        x = torch.stack(mats)[None]                             # [1,16,in,out]
    else:
        sh = prefix + ".mlp.shared_expert."
        if kind == "gate_up_proj":
            x = torch.cat([hw.get(sh + "gate_proj.weight"), hw.get(sh + "up_proj.weight")], 0).T[None, None]
        else:
            x = hw.get(sh + "down_proj.weight").T[None, None]
    return ttnn.from_torch(x.contiguous(), dtype=ttnn.bfloat8_b, layout=TILE)


class Q36Weights:
    """All weights on the card, from the Lottolabs checkpoint (~20.6 GB)."""

    def __init__(self, dev, mtp: bool = True, verbose: bool = True):
        self.dev = dev
        self.man = json.load(open(LOTTO / "native_manifest.json"))["tensors"]
        self.groups = json.load(open(LOTTO / "native_manifest.json"))["expert_groups"]
        self.w = {}
        self.bytes = 0
        self._st = {}
        # BFP8 upgrade layers (tt_qwen36/weights.py): by default the down projections of layers 30-39 + MTP and the
        # gate/up of 36-39 + MTP in BFP8 from the BF16 original (+2.84 GB on the card; KL vs BF16 0.0173 -> 0.0130 on
        # 1,532 positions). Q36_BFP8_SOURCE=off: every expert in Lottolabs' GPTQ BFP4.
        from ..weights import Bfp8Files, bfp8_layers
        spec = bfp8_layers()
        self.promote = self.parse_promote(spec) if spec else set()
        if not mtp:
            self.promote = {x for x in self.promote if x[1] != "mtp"}
        self.bfp8 = Bfp8Files(self.promote)
        self._bf16 = None
        if self.promote and verbose:
            print(f"  BFP8 upgrade layers: {spec} ({self.bfp8.source})", flush=True)
        t0 = time.time()
        for L in range(NL):
            self._layer(L, f"layers.{L}", is_attn(L))
            if verbose and L % 8 == 7:
                print(f"  weights: layer {L} {self.bytes / 1e9:.1f} GB {time.time() - t0:.0f} s", flush=True)
        self.w["norm"] = self._row1p("norm.weight")
        self.w["lm_head"] = self._tb("output.weight")
        if mtp:
            self._layer("mtp", "mtp.layers.0", True)
            self.w["mtp.fc"] = self._tb("mtp.fc.weight")                                    # [4096, 2048]
            for n in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
                self.w[f"mtp.{n}"] = self._row1p(f"mtp.{n}.weight")
            self._draft_head()
        ttnn.synchronize_device(dev)
        if verbose:
            print(f"  weights on the card: {self.bytes / 1e9:.2f} GB in {time.time() - t0:.0f} s", flush=True)

    def _draft_head(self):
        """compact LM head for the MTP drafts (Lottolabs' QWEN36_MTP_DRAFT_VOCAB idea): vocabulary rows [0, V) (BPE ids
        are roughly frequency ordered) + the tile-aligned special-token tail [248032, 248320). The verify pass uses the
        full head, so outputs do not change; only which drafts get proposed can. P100A_Q36_DRAFT_VOCAB=0: full head.
        The BFP8 values are copied exactly (whole tiles, so every shared-exponent block keeps its members)."""
        V = int(os.environ.get("P100A_Q36_DRAFT_VOCAB", "32768"))
        self.draft_v, self.draft_off = V, 0
        if not V:
            return
        tail = (248044 // 32) * 32
        e = self.man["output.weight"]
        full = ttnn.to_torch(ttnn.load_tensor(str(LOTTO / e["file"]))).float()          # host [2048, 248320]
        rows = torch.cat([full[:, :V], full[:, tail:]], 1).contiguous()
        del full
        t = ttnn.from_torch(rows.reshape(1, 1, H, -1), dtype=ttnn.bfloat8_b, layout=TILE, device=self.dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.bytes += rows.numel() * 1.0625
        self.w["draft_head"] = t
        self.draft_off = tail - V

    # ---- loaders
    def _safe(self, name):
        from safetensors import safe_open
        f = str(LOTTO / self.man[name]["file"])
        if f not in self._st:
            self._st[f] = safe_open(f, framework="pt", device="cpu")
        return self._st[f].get_tensor(name).float()

    def _put(self, t: torch.Tensor, dtype=ttnn.float32):
        x = ttnn.from_torch(t.contiguous(), dtype=dtype, layout=TILE, device=self.dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.bytes += x.volume() * {ttnn.float32: 4, ttnn.bfloat16: 2}.get(dtype, 1)
        return x

    def _tb(self, name):
        """BFP8/BFP4 tile tensor [in, out] as stored -> card, [1,1,in,out]"""
        e = self.man[name]
        t = ttnn.load_tensor(str(LOTTO / e["file"]))
        k, n = t.shape[-2], t.shape[-1]
        self.bytes += k * n * (1.0625 if e["precision"] == "bfp8" else 0.5625)
        t = ttnn.to_device(t, self.dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(t, [1, 1, k, n])

    def _row1p(self, name):
        return self._put((1.0 + self._safe(name)).reshape(1, 1, 1, -1))

    @staticmethod
    def parse_promote(spec: str) -> set:
        """Q36_BFP8_LAYERS, e.g. "dn:0-39,mtp;gu:0,1,39": which expert matrices (gu = gate|up, dn = down; all 257 incl.
        the shared expert) of which layers are served in BFP8 (round-to-nearest from the BF16 original) instead of
        Lottolabs' GPTQ BFP4."""
        import re
        out = set()
        for part in re.split(r"[;\s]+", spec.strip()):
            if not part:
                continue
            key, _, layers = part.partition(":")
            for item in layers.split(","):
                if item == "mtp":
                    out.add((key, "mtp"))
                elif item == "all":
                    out.update((key, L) for L in list(range(NL)) + ["mtp"])
                elif "-" in item:
                    a, b = item.split("-")
                    out.update((key, L) for L in range(int(a), int(b) + 1))
                elif item:
                    out.add((key, int(item)))
        return out

    def _experts_bfp8(self, prefix, kind):
        """[1,257,in,out] BFP8: routed experts + shared expert (index 256), 17 tensorbins of 16 experts (the last one:
        the shared expert) from tt_qwen36/weights.Bfp8Files; with Q36_BFP8_SOURCE=bf16 a missing file is converted
        here from the BF16 safetensors (host RAM ~0.3 GB at a time) and written to the cache."""
        parts = []
        for g in range(17):                                     # 16 groups of 16 routed experts + the shared expert
            name = file_name(prefix, kind, g)
            f = self.bfp8.path(name)
            if f is not None:
                t = ttnn.load_tensor(str(f))
            else:
                if self._bf16 is None:
                    from tt_qwen36.ref.q36_weights import Q36Weights as HostW
                    self._bf16 = HostW("bf16")
                t = convert_group(self._bf16, prefix, kind, g)
                dst = self.bfp8.out(name)
                tmp = dst.with_name(dst.name.replace(".tensorbin", ".part.tensorbin"))
                ttnn.dump_tensor(str(tmp), t)
                os.replace(tmp, dst)
                self.bfp8.path(name)                            # checksum of what was just written
            parts.append(ttnn.to_device(t, self.dev, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            self.bytes += t.volume() * 1.0625
        ttnn.synchronize_device(self.dev)
        out = ttnn.concat(parts, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for q in parts:
            ttnn.deallocate(q)
        return out

    def _experts(self, name, shared=None):
        files = self.groups[name]["files"]
        parts = []
        for f in files:
            g = ttnn.load_tensor(str(LOTTO / f["file"]))
            parts.append(ttnn.to_device(g, self.dev, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            self.bytes += f["bytes"]
        if shared is not None:                                 # the shared expert as expert 256 (same shapes)
            parts.append(shared)
        ttnn.synchronize_device(self.dev)
        out = ttnn.concat(parts, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for p in parts:
            ttnn.deallocate(p)
        return out

    def _layer(self, L, p, attn):
        w, k = self.w, f"{L}."
        w[k + "ln1"] = self._row1p(p + ".input_layernorm.weight")
        w[k + "ln2"] = self._row1p(p + ".post_attention_layernorm.weight")
        if attn:
            a = p + ".self_attn."
            # one input projection [2048, 9216] = q+gate (8192) | k (512) | v (512)
            w[k + "qkv_a"] = ttnn.concat([self._tb(a + "q_proj.weight"), self._tb(a + "k_proj.weight"),
                                          self._tb(a + "v_proj.weight")], dim=-1)
            w[k + "o"] = self._tb(a + "o_proj.weight")                                         # [4096, 2048]
            w[k + "q_norm"] = self._row1p(a + "q_norm.weight")
            w[k + "k_norm"] = self._row1p(a + "k_norm.weight")
        else:
            g = p + ".linear_attn."
            # one input projection [2048, 12352] = qkv (8192) | z (4096) | a (32) | b (32): tile columns 0..255 qkv,
            # 256..383 z, 384..385 ab (the fused conv / core kernels read them in place)
            w[k + "in"] = ttnn.concat([self._tb(g + "qkv_proj.weight"), self._tb(g + "in_proj_z.weight"),
                                       self._tb(g + "in_proj_a.weight"), self._tb(g + "in_proj_b.weight")], dim=-1)
            w[k + "out"] = self._tb(g + "out_proj.weight")                                     # [4096, 2048]
            conv = torch.cat([self._safe(g + f"{x}_conv.weight")[:, 0, :] for x in "qkv"], 0)   # [8192, 4]
            for t in range(4):
                w[k + f"conv{t}"] = self._put(conv[:, t].reshape(1, 1, 1, QKV))
            w[k + "neg_a"] = self._put((-self._safe(g + "A_log").exp()).reshape(1, 1, 1, NV))
            w[k + "dt"] = self._put(self._safe(g + "dt_bias").reshape(1, 1, 1, NV))
            w[k + "gnorm"] = self._put(self._safe(g + "norm.weight").reshape(1, 1, 1, DK))
        m = p + ".mlp."
        # router [2048, 288] fp32: columns 0..255 the experts, column 256 the shared-expert gate (the route kernel
        # gives the shared expert, stored as expert 256, the weight sigmoid(gate) in every row)
        r = torch.zeros(H, NE + 32)
        r[:, :NE] = self._safe(m + "gate.weight").T
        r[:, NE] = self._safe(m + "shared_expert_gate.weight").reshape(-1)
        w[k + "router"] = self._put(r.reshape(1, 1, H, NE + 32))
        sh_gu = ttnn.concat([self._tb(m + "shared_expert.gate_proj.weight"),
                             self._tb(m + "shared_expert.up_proj.weight")], dim=-1)                # [1,1,2048,1024]
        sh_dn = self._tb(m + "shared_expert.down_proj.weight")                                   # [1,1,512,2048]
        for key, kind, sh in (("gu", "gate_up_proj", sh_gu), ("dn", "down_proj", sh_dn)):
            if (key, L) in self.promote:          # this layer's experts in BFP8 from the BF16 original
                w[k + key] = self._experts_bfp8(p, kind)
            else:
                w[k + key] = self._experts(m + f"experts.{kind}", sh)                    # [1,257,2048,1024] / [1,257,512,2048]

    def __getitem__(self, k):
        return self.w[k]


class Q36Pass:
    """T-row pass over the persistent model state. sibling(T) shares the state and weights (prompt reading)."""

    def __init__(self, dev, W: Q36Weights, T: int = 4, s_max: int = 32768, mtp: bool = True):
        self.dev, self.W, self.T, self.s_max = dev, W, T, s_max
        self.mtp_on = mtp
        self.ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                   fp32_dest_acc_en=True, packer_l1_acc=True)
        self.ckf = self.ck
        self.moe = Q36Moe(dev)
        self.pos, self.tokens = 0, []
        self.io = {"h2d_bytes": 0, "d2h_bytes": 0, "d2h_n": 0, "d2h_s": 0.0}
        self.phase = {}
        self.stats = {"passes": 0, "rows": 0}
        self._alloc_state()
        self._alloc_pass()
        self.traces = None
        self.gdn_keep = {}
        from tt_qwen36.ref.q36_weights import Q36Weights as HostW
        self.hw = HostW("lotto")                    # host: embedding rows (BF16 safetensors)
        self.reset()

    # ---- helpers
    def _dev(self, t, dtype=ttnn.float32, layout=TILE):
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=self.dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # explicit 1D (mcast_in0) program configs for decode shapes where ttnn's default is slow, cores from
    # tt_qwen36/q36/mm_bench.py; P100A_Q36_MM1D=0 = default configs everywhere
    MM1D = {(4096, 2048): 88, (2048, 288): 22}   # 0.078 -> 0.030 ms and 0.031 -> 0.015 ms (bench 2026-11-13 06:20)

    def _mm_cfg(self, a, b):
        if os.environ.get("P100A_Q36_MM1D", "1") == "0" or a.shape[-2] != 32 or a.shape[0] * a.shape[1] != 1:
            return None
        k, n = b.shape[-2], b.shape[-1]
        cores = self.MM1D.get((k, n))
        if cores is None:
            return None
        cache = self.__dict__.setdefault("_mm1d", {})
        if (k, n) not in cache:
            g = self.dev.compute_with_storage_grid_size()
            cols = min(g.x, cores)
            rows = math.ceil(cores / cols)
            kt, nt = k // 32, math.ceil(n / 32)
            pn = math.ceil(nt / (cols * rows))
            cache[(k, n)] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(cols, rows), in0_block_w=max(d for d in range(1, 9) if kt % d == 0),
                out_subblock_h=1, out_subblock_w=max(i for i in range(1, 5) if pn % i == 0), per_core_M=1,
                per_core_N=pn, fuse_batch=True, fused_activation=None, mcast_in0=True)
        return cache[(k, n)]

    def _mm(self, a, b, dtype=ttnn.float32):
        if a.dtype != ttnn.bfloat16:
            a = ttnn.typecast(a, ttnn.bfloat16)
        pc = self._mm_cfg(a, b)
        if pc is not None:
            return ttnn.matmul(a, b, dtype=dtype, compute_kernel_config=self.ck, program_config=pc)
        return ttnn.matmul(a, b, dtype=dtype, compute_kernel_config=self.ck)

    def _mmf(self, a, b, **kw):
        return ttnn.matmul(a, b, dtype=ttnn.float32, compute_kernel_config=self.ck, **kw)

    def _mm_sel(self, sel, b):
        hi = ttnn.typecast(ttnn.typecast(b, ttnn.bfloat16), ttnn.float32)
        return ttnn.add(self._mmf(sel, hi), self._mmf(sel, ttnn.subtract(b, hi)))

    def _pad32(self, rm):
        r = rm.shape[2]
        if r < 32:
            rm = ttnn.pad(rm, [(0, 0), (0, 0), (0, 32 - r), (0, 0)], 0.0)
        return ttnn.to_layout(rm, TILE)

    def _rms1p(self, x, w):
        """zero-centred RMSNorm in fp32 (w = 1 + weight, a broadcast row)"""
        ms = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True)
        return ttnn.multiply(ttnn.multiply(x, ttnn.rsqrt(ttnn.add(ms, EPS))), w)

    def _rope(self, t, cos, sin):
        return ttnn.add(ttnn.multiply(t, cos), ttnn.multiply(self._mmf(t, self.rope_P), sin))

    def _kv(self, S, dt):
        """zeroed KV cache (zeros, not uninitialized: a NaN pattern in a masked row could still meet a zero softmax
        weight inside the last attended chunk). [2,1,S,256] (heads as separate block ranges), or with
        P100A_Q36_KVBLK=1 the standard paged layout [S/32, 2, 32, 256] (block-major, both KV heads per block)"""
        if self.kvblk:
            return self._dev(torch.zeros(S // NB, NKV, NB, HD), dtype=dt)
        return self._dev(torch.zeros(NKV, 1, S, HD), dtype=dt)

    # ---- buffers
    def _alloc_state(self):
        z = torch.zeros
        S = self.s_max
        self.X = self._dev(z(1, 1, 32, H))
        self.S, self.hist3, self.kc, self.vc = {}, {}, {}, {}
        # KV cache dtype (bf16, or bfp8 = half the bytes, as Lottolabs serve it)
        self.kv_bfp8 = os.environ.get("P100A_Q36_KV", "bf16") == "bfp8"
        # block KV layout; decode = GQA paged SDPA (one user per row), prompt passes = chunked flash SDPA
        self.kvblk = os.environ.get("P100A_Q36_KVBLK", "0") == "1"
        kvdt = ttnn.bfloat8_b if self.kv_bfp8 else ttnn.bfloat16
        for L in range(NL):
            if is_attn(L):
                self.kc[L] = self._kv(S, kvdt)
                self.vc[L] = self._kv(S, kvdt)
            else:
                self.S[L] = self._dev(z(1, NV, DK, DK))
                self.hist3[L] = self._dev(z(1, 1, 32, QKV))
        if self.mtp_on:
            self.kc["mtp"] = self._kv(S, kvdt)
            self.vc["mtp"] = self._kv(S, kvdt)
        self._zS = self._dev(z(1, NV, DK, DK))
        self._zh = self._dev(z(1, 1, 32, QKV))
        self._zX = self._dev(z(1, 1, 32, H))
        self.hid = self._dev(z(1, 1, 32, H))           # final-normed hidden of the last main pass (MTP input)
        self.hm = self._dev(z(1, 1, 32, H))            # MTP-normed output of the last MTP pass
        P = z(HD, HD)
        r = RD // 2
        for i in range(r):
            P[i + r, i] = -1.0
            P[i, i + r] = 1.0
        self.rope_P = self._dev(P.reshape(1, 1, HD, HD))
        e = torch.eye(32)
        sh3 = z(32, 32)
        for rr in range(29):
            sh3[rr + 3, rr] = 1
        self.c_sh3 = self._dev(sh3.reshape(1, 1, 32, 32))
        ups = []
        for k in range(4):
            u = z(32, 32)
            for rr in range(32 - k):
                u[rr, rr + k] = 1
            ups.append(self._dev(u.reshape(1, 1, 32, 32)))
        self.c_up = ups
        tril = torch.tril(torch.ones(32, 32))
        self.c_cumsum = self._dev(tril.reshape(1, 1, 32, 32))
        self.c_tril = self._dev(tril.expand(1, NV, 32, 32).contiguous())
        self.c_stril = self._dev((tril - e).expand(1, NV, 32, 32).contiguous())
        self.c_eye = self._dev(e.expand(1, NV, 32, 32).contiguous())
        self.esel = self._dev(z(1, 1, 32, 32), layout=RM)
        self.msel = self._dev(z(1, 1, 32, NV), layout=RM)
        self.csel = self._dev(z(1, 1, 32, 32), layout=RM)
        self.sdpa_cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), exp_approx_mode=False,
                                               q_chunk_size=0, k_chunk_size=128)
        # step-4 fused GDN programs at Qwen3.6 shapes (tt_qwen36/q36/kgdn.py); P100A_Q36_FUSED=0 = the ttnn op path
        self.fused = os.environ.get("P100A_Q36_FUSED", "1") != "0"
        self.mask_rows = os.environ.get("P100A_Q36_MASK", "0") == "1"
        self.xl1 = os.environ.get("P100A_Q36_XL1", "1") == "1"      # MoE input row in L1 (every A core reads it)
        self.route_stats = torch.zeros(NL, NE + 1, dtype=torch.int64) if os.environ.get("P100A_Q36_ROUTESTATS") else None
        if self.fused:
            from .kgdn import GdnCommit36, GdnConv36
            self.kconv = GdnConv36(self.dev)
            self.kcommit = GdnCommit36(self.dev)
            self.kg_y = self._dev(z(1, 1, 32, NV * DK))
            self.kg_kn = {L: self._dev(z(1, NV, 32, DK)) for L in self.S}
            self.kg_U = {L: self._dev(z(1, NV, 32, DK)) for L in self.S}
            self.kg_g = {L: self._dev(z(1, NV, 32, 32)) for L in self.S}
            self.kg_conv = self._dev(z(1, 1, 32, QKV))
            self.kg_E = {L: self._dev(z(1, 1, 32, QKV)) for L in self.S}
            rd = lambda k: ttnn.to_torch(self.W[k]).float().flatten()[:NV].tolist()
            self.kg_dt = {L: rd(f"{L}.dt") for L in self.S}
            self.kg_a = {L: rd(f"{L}.neg_a") for L in self.S}

    def _alloc_pass(self):
        T, z = self.T, torch.zeros
        assert 1 <= T <= 29
        self.emb = self._dev(z(1, 1, T, H), layout=RM)
        self.cos = self._dev(torch.ones(1, 1, T, HD), layout=RM)
        self.sin = self._dev(z(1, 1, T, HD), layout=RM)
        self.cur = self._dev(torch.zeros(2 * T, dtype=torch.int32), dtype=ttnn.int32, layout=RM)
        nblk = self.s_max // NB
        pt = torch.stack([torch.arange(nblk, dtype=torch.int32) + (u % 2) * nblk for u in range(2 * T)])
        self.ptab = self._dev(pt, dtype=ttnn.int32, layout=RM)
        if self.kvblk:
            self.cur1 = self._dev(torch.zeros(T, dtype=torch.int32), dtype=ttnn.int32, layout=RM)   # pos per row
            self.ptab1 = self._dev(torch.arange(nblk, dtype=torch.int32).repeat(T, 1), dtype=ttnn.int32, layout=RM)
            self.flash = T > 4                          # prompt passes: chunked flash SDPA over a 64-row q window
            if self.flash:
                self.cstart = self._dev(torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, layout=RM)
                # +8 entries: the 64-row window may run past the end; and a page-table row must be a multiple of 32 B
                pf = torch.cat([torch.arange(nblk, dtype=torch.int32), torch.full((8,), nblk - 1, dtype=torch.int32)])
                self.ptabf = self._dev(pf.reshape(1, nblk + 8), dtype=ttnn.int32, layout=RM)
                self.place = self._dev(z(1, NQ, 64, 32), dtype=ttnn.bfloat16)     # q rows -> rows o..o+T-1
                self.placeT = self._dev(z(1, NQ, 32, 64), dtype=ttnn.bfloat16)
                self.flash_cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), exp_approx_mode=False,
                                                        q_chunk_size=32, k_chunk_size=32)
            if self.mtp_on:
                self.cur1_m = self._dev(torch.zeros(T, dtype=torch.int32), dtype=ttnn.int32, layout=RM)
        m = z(1, 1, 32, H)
        m[:, :, :T] = 1
        self.mask = self._dev(m)
        # expert-kernel shape per row count (tt_qwen36/q36/test_moe.py sweep 23:45): 1 row: 64 A cores, 2 output tiles per
        # B core; more rows: 4 output tiles per B core (H re-reads / 2), 32-tile A chunks
        pk = T if os.environ.get("P100A_Q36_PICKS", "1") == "1" else 0      # routing one row per core
        self.moe = Q36Moe(self.dev, picks_T=pk) if T == 1 else \
            Q36Moe(self.dev, nc_a=64, nq=4, chunk=32, kc=16, picks_T=pk)
        self.ctl = {L: self.moe.new_ctl() for L in list(range(NL)) + (["mtp"] if self.mtp_on else [])}
        if pk:                                # valid-row counts: main pass, MTP pass (padding rows route nowhere)
            self.nv_main, self.nv_mtp = self.moe.new_nvalid(), self.moe.new_nvalid()
        self.y_moe = self._dev(z(1, 1, 32, H))
        if self.kv_bfp8:
            from .kmoe import KvWrite8
            self.kvw = KvWrite8(self.dev, T, NKV, HD, self.s_max, blk=self.kvblk)
        elif self.kvblk:
            from .kmoe import KvWriteBlk
            self.kvw = KvWriteBlk(self.dev, T, NKV, HD, self.s_max)
        else:
            self.kvw = KvWrite(self.dev, T, NKV, HD, self.s_max)
        if self.fused:
            from .kgdn import GdnCore36
            self.kg = GdnCore36(self.dev, T, eps=EPS)
        if self.mtp_on:                                 # MTP pass inputs
            self.selA = self._dev(z(1, 1, 32, 32), layout=RM)
            self.selB = self._dev(z(1, 1, 32, 32), layout=RM)
            self.emb_m = self._dev(z(1, 1, T, H), layout=RM)
            self.cur_m = self._dev(torch.zeros(2 * T, dtype=torch.int32), dtype=ttnn.int32, layout=RM)
            self.cos_m = self._dev(torch.ones(1, 1, T, HD), layout=RM)
            self.sin_m = self._dev(z(1, 1, T, HD), layout=RM)

    def sibling(self, T: int) -> "Q36Pass":
        sib = Q36Pass.__new__(Q36Pass)
        sib.__dict__.update(self.__dict__)
        sib.T = T
        sib.traces, sib.gdn_keep = None, {}
        sib._alloc_pass()
        return sib

    def reset(self):
        for L in self.S:
            ttnn.copy(self._zS, self.S[L])
            ttnn.copy(self._zh, self.hist3[L])
        ttnn.copy(self._zX, self.X)
        self.pos, self.tokens = 0, []

    def reset_host(self):
        self.pos, self.tokens = 0, []

    # ---- layer pieces
    def gdn(self, L, x):
        k_ = f"{L}."
        w, T = self.W, self.T
        x16 = ttnn.typecast(x, ttnn.bfloat16)
        if self.fused:
            pin = self._mm(x16, w[k_ + "in"])                                              # [1,1,32,12352] fp32
            E = self.kg_E[L]
            self.kconv(pin, self.hist3[L], [w[k_ + f"conv{t}"] for t in range(4)], self.kg_conv, E)
            self.kg(self.kg_conv, self.S[L], pin, w[k_ + "gnorm"], self.kg_y, self.kg_kn[L], self.kg_U[L], pin,
                    self.kg_dt[L], self.kg_a[L], self.kg_g[L], z_off=QKV // 32, ab_off=(QKV + NV * DK) // 32)
            self.gdn_keep[L] = (E, self.kg_kn[L], self.kg_U[L], self.kg_g[L])
            return self._mm(self.kg_y, w[k_ + "out"])
        E = ttnn.add(self.hist3[L], self._mmf(self.c_sh3, qkv))                             # rows 0..2 history
        conv = None
        for t in range(4):
            c = ttnn.multiply(self._mmf(self.c_up[t], E), w[k_ + f"conv{t}"])
            conv = c if conv is None else ttnn.add(conv, c)
        conv = ttnn.silu(conv)
        hd = ttnn.permute(ttnn.reshape(conv, [1, 32, QKV // DK, DK]), (0, 2, 1, 3))         # [1,64,32,128]
        q = ttnn.repeat_interleave(ttnn.slice(hd, [0, 0, 0, 0], [1, NK, 32, DK]), NV // NK, dim=1)
        k = ttnn.repeat_interleave(ttnn.slice(hd, [0, NK, 0, 0], [1, 2 * NK, 32, DK]), NV // NK, dim=1)
        v = ttnn.slice(hd, [0, 2 * NK, 0, 0], [1, 2 * NK + NV, 32, DK])
        q = ttnn.multiply(ttnn.rms_norm(q, epsilon=1e-6 / DK, compute_kernel_config=self.ck), 1.0 / DK)
        k = ttnn.multiply(ttnn.rms_norm(k, epsilon=1e-6 / DK, compute_kernel_config=self.ck), DK ** -0.5)
        ab = ttnn.to_layout(ttnn.slice(pin, [0, 0, 0, QKV + NV * DK], [1, 1, 32, QKV + NV * DK + 2 * NV]), RM)
        a = ttnn.to_layout(ttnn.slice(ab, [0, 0, 0, 0], [1, 1, 32, NV]), TILE)
        beta = ttnn.sigmoid(ttnn.to_layout(ttnn.slice(ab, [0, 0, 0, NV], [1, 1, 32, 2 * NV]), TILE))
        g = ttnn.multiply(ttnn.softplus(ttnn.add(a, w[k_ + "dt"])), w[k_ + "neg_a"])        # [1,1,32,32]
        gam = self._mm_sel(self.c_cumsum, g)
        gcol = ttnn.permute(gam, (0, 3, 2, 1))                                             # [1,32h,32r,1]
        grow = ttnn.permute(gam, (0, 3, 1, 2))                                             # [1,32h,1,32r]
        dlog = ttnn.subtract(ttnn.repeat(gcol, ttnn.Shape([1, 1, 1, 32])), grow)
        D = ttnn.multiply(ttnn.exp(ttnn.minimum(dlog, 0.0)), self.c_tril)
        bcol = ttnn.permute(beta, (0, 3, 2, 1))
        Gcol = ttnn.exp(gcol)
        S0 = self.S[L]
        A = ttnn.multiply(ttnn.multiply(self._mmf(k, k, transpose_b=True), D), self.c_stril)
        A = ttnn.multiply(A, bcol)
        VB = ttnn.multiply(ttnn.subtract(v, ttnn.multiply(self._mmf(k, S0), Gcol)), bcol)
        if T <= 4:
            inv = self._mmf(ttnn.subtract(self.c_eye, A), ttnn.add(self.c_eye, self._mmf(A, A)))
            U = self._mmf(inv, VB)
        else:
            U = VB
            for _ in range(T - 1):
                U = ttnn.subtract(VB, self._mmf(A, U))
        O = ttnn.add(ttnn.multiply(self._mmf(q, S0), Gcol),
                     self._mmf(ttnn.multiply(self._mmf(q, k, transpose_b=True), D), U))
        O = ttnn.multiply(ttnn.rms_norm(O, epsilon=EPS, compute_kernel_config=self.ck), w[k_ + "gnorm"])
        zt = ttnn.permute(ttnn.reshape(ttnn.slice(pin, [0, 0, 0, QKV], [1, 1, 32, QKV + NV * DK]), [1, 32, NV, DK]),
                          (0, 2, 1, 3))
        y = ttnn.multiply(O, ttnn.silu(zt))
        y = ttnn.reshape(ttnn.permute(y, (0, 2, 1, 3)), [1, 1, 32, NV * DK])
        self.gdn_keep[L] = (E, k, U, gam)
        return self._mm(y, w[k_ + "out"])

    def _hnorm(self, t, wkey):
        """per-head RMSNorm with (1 + w): two ops, or one ttnn.rms_norm with a row-major weight (P100A_Q36_ATTN_RMS=1)"""
        if os.environ.get("P100A_Q36_ATTN_RMS", "0") == "1":
            rw = self.__dict__.setdefault("_hn_rm", {})
            if wkey not in rw:
                rw[wkey] = ttnn.to_layout(ttnn.reshape(self.W[wkey], [1, 1, 8, 32]), RM) if False else ttnn.from_torch(
                    ttnn.to_torch(self.W[wkey]).float().reshape(1, 1, 8, 32), dtype=ttnn.float32, layout=RM,
                    device=self.dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.rms_norm(t, epsilon=EPS, weight=rw[wkey], compute_kernel_config=self.ck)
        return ttnn.multiply(ttnn.rms_norm(t, epsilon=EPS, compute_kernel_config=self.ck), self.W[wkey])

    def attn(self, key, x, cos, sin, cosq, sinq, cur, caches=None, kvcur=None, ptab=None, cur1=None):
        """caches / kvcur / ptab: batched decode (tt_qwen36/q36/batch.py) passes its own caches, the physical write rows
        and the per-user page table; cur is then each user's logical position"""
        k_ = f"{key}."
        w, T = self.W, self.T
        prm = ttnn.to_layout(self._mm(x, w[k_ + "qkv_a"]), RM)                              # [1,1,32,9216]
        qg = ttnn.slice(prm, [0, 0, 0, 0], [1, 1, T, 2 * NQ * HD])
        rm = ttnn.reshape(qg, [1, 1, NQ * T, 2 * HD])                                       # rows (r, head)
        q = ttnn.to_layout(ttnn.slice(rm, [0, 0, 0, 0], [1, 1, NQ * T, HD]), TILE)
        gate = ttnn.to_layout(ttnn.slice(rm, [0, 0, 0, HD], [1, 1, NQ * T, 2 * HD]), TILE)
        q = self._rope(self._hnorm(q, k_ + "q_norm"), cosq, sinq)
        kv = ttnn.slice(prm, [0, 0, 0, 2 * NQ * HD], [1, 1, T, 2 * NQ * HD + 2 * NKV * HD])
        kv = ttnn.reshape(kv, [1, T, 2 * NKV, HD])                                         # k0 k1 v0 v1
        k = ttnn.to_layout(ttnn.slice(kv, [0, 0, 0, 0], [1, T, NKV, HD]), TILE)
        v = ttnn.slice(kv, [0, 0, NKV, 0], [1, T, 2 * NKV, HD])
        k = self._rope(self._hnorm(k, k_ + "k_norm"), cos, sin)
        k = ttnn.slice(ttnn.to_layout(ttnn.typecast(k, ttnn.bfloat16), RM), [0, 0, 0, 0], [1, T, NKV, HD])
        v = ttnn.typecast(v, ttnn.bfloat16) if v.dtype != ttnn.bfloat16 else v
        kc, vc = (self.kc[key], self.vc[key]) if caches is None else caches
        self.kvw([kc, vc], [k, v], cur if kvcur is None else kvcur)
        qr = ttnn.slice(ttnn.to_layout(ttnn.typecast(q, ttnn.bfloat16), RM), [0, 0, 0, 0], [1, 1, NQ * T, HD])
        if self.kvblk and self.flash and caches is None:      # prompt pass: chunked flash SDPA
            qt = ttnn.permute(ttnn.reshape(qr, [1, T, NQ, HD]), (0, 2, 1, 3))                   # [1,16,T,256]
            qt = ttnn.to_layout(ttnn.pad(qt, [(0, 0), (0, 0), (0, 32 - T), (0, 0)], 0.0), TILE)
            qf = ttnn.matmul(self.place, qt, dtype=ttnn.bfloat16)                           # [1,16,64,256]
            of = ttnn.transformer.chunked_scaled_dot_product_attention(
                qf, kc, vc, self.ptabf, chunk_start_idx_tensor=self.cstart, scale=1.0 / math.sqrt(HD),
                program_config=self.flash_cfg)
            ob = ttnn.matmul(self.placeT, of, dtype=ttnn.bfloat16)                          # [1,16,32,256]
            ob = ttnn.slice(ttnn.to_layout(ob, RM), [0, 0, 0, 0], [1, NQ, T, HD])
            o = ttnn.reshape(ttnn.permute(ob, (0, 2, 1, 3)), [1, 1, NQ * T, HD])
        elif self.kvblk:                                     # decode / verify: GQA paged SDPA, one user per row
            qh = ttnn.to_layout(ttnn.reshape(qr, [1, T, NQ, HD]), TILE)
            o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                qh, kc, vc, self.ptab1 if ptab is None else ptab, cur_pos_tensor=cur1 if cur1 is not None else self.cur1,
                scale=1.0 / math.sqrt(HD), program_config=self.sdpa_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            o = ttnn.reshape(ttnn.slice(ttnn.to_layout(o, RM), [0, 0, 0, 0], [1, T, NQ, HD]), [1, 1, NQ * T, HD])
        else:
            qh = ttnn.to_layout(ttnn.reshape(qr, [1, NKV * T, NQ // NKV, HD]), TILE)        # user = 2r + kv head
            nblk = kc.shape[2] // NB
            kp = ttnn.reshape(kc, [NKV * nblk, 1, NB, HD])
            vp = ttnn.reshape(vc, [NKV * nblk, 1, NB, HD])
            o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                qh, kp, vp, self.ptab if ptab is None else ptab, cur_pos_tensor=cur, scale=1.0 / math.sqrt(HD),
                program_config=self.sdpa_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)          # [1,2T,8,256]
            o = ttnn.reshape(ttnn.slice(ttnn.to_layout(o, RM), [0, 0, 0, 0], [1, NKV * T, NQ // NKV, HD]),
                             [1, 1, NQ * T, HD])
        o = ttnn.multiply(ttnn.to_layout(o, TILE), ttnn.sigmoid(gate))
        o = ttnn.slice(ttnn.to_layout(o, RM), [0, 0, 0, 0], [1, 1, NQ * T, HD])
        return self._mm(self._pad32(ttnn.reshape(o, [1, 1, T, NQ * HD])), w[k_ + "o"])

    def moe_block(self, key, h):
        """h: fp32 [1,1,32,2048] (normed); returns routed + gated shared expert output"""
        k_ = f"{key}."
        w = self.W
        pc = self._mm_cfg(h, w[k_ + "router"])
        lg = self._mmf(h, w[k_ + "router"], **({"program_config": pc} if pc is not None else {}))   # fp32 [1,1,32,288]
        ctl = self.ctl[key]
        if self.moe.picks_T:                               # 8 routed experts + the shared one (entry 256) per row
            self.moe.route_rows(lg, ctl, self.nv_mtp if key == "mtp" else self.nv_main)
        else:
            self.moe.route(lg, ctl, self.T)
        h16 = ttnn.typecast(h, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG if self.xl1 else None)
        self.moe.experts(h16, w[k_ + "gu"], w[k_ + "dn"], ctl, self.y_moe)
        return self.y_moe

    def _rope_rows(self, cos_rm, sin_rm):
        T = self.T
        cos, sin = ttnn.to_layout(cos_rm, TILE), ttnn.to_layout(sin_rm, TILE)
        cosk, sink = ttnn.reshape(cos, [1, T, 1, HD]), ttnn.reshape(sin, [1, T, 1, HD])
        cosq = ttnn.to_layout(ttnn.repeat_interleave(cos_rm, NQ, dim=2), TILE)
        sinq = ttnn.to_layout(ttnn.repeat_interleave(sin_rm, NQ, dim=2), TILE)
        return cosk, sink, cosq, sinq

    def layer(self, L, X, rope, key=None, cur=None):
        key = L if key is None else key
        w = self.W
        h = self._rms1p(X, w[f"{key}.ln1"])
        if key == "mtp" or is_attn(L):
            y = self.attn(key, h, *rope, cur if cur is not None else self.cur,
                          cur1=(self.cur1_m if key == "mtp" else None) if self.kvblk else None)
        else:
            y = self.gdn(L, h)
        X = ttnn.add(X, y)
        h = self._rms1p(X, w[f"{key}.ln2"])
        X = ttnn.add(X, self.moe_block(key, h))
        if self.mask_rows:                    # rows >= T: finite garbage that never reaches a valid row (see docstring)
            X = ttnn.where(self.mask, X, self._zX)
        return X

    # ---- the pass (device ops only; traced as a whole)
    def seg_pass(self):
        X = self._pad32(self.emb)
        rope = self._rope_rows(self.cos, self.sin)
        for L in range(NL):
            X = self.layer(L, X, rope)
        ttnn.copy(X, self.X)
        hid = self._rms1p(X, self.W["norm"])
        ttnn.copy(hid, self.hid)
        lg = self._mm(hid, self.W["lm_head"], dtype=ttnn.bfloat16)
        rm = ttnn.slice(ttnn.to_layout(lg, RM), [0, 0, 0, 0], [1, 1, self.T, VOC])
        return {"logits": rm, "argmax": ttnn.argmax(rm, dim=-1)}

    def seg_commit(self):
        esel, msel, csel = (ttnn.to_layout(t, TILE) for t in (self.esel, self.msel, self.csel))
        for L in self.S:
            E, k, U, gam = self.gdn_keep[L]
            if self.fused:                    # gam = the gamma column tiles of the fused core
                self.kcommit(self.S[L], k, U, gam)
                ttnn.copy(self._mmf(csel, E), self.hist3[L])
                continue
            gr = ttnn.slice(self._mm_sel(esel, gam), [0, 0, 0, 0], [1, 1, 1, NV])         # gamma_r
            dcol = ttnn.multiply(ttnn.exp(ttnn.minimum(ttnn.subtract(gr, gam), 0.0)), msel)
            dh = ttnn.permute(dcol, (0, 3, 2, 1))                                        # [1,32,32,1]
            Sn = self._mmf(k, ttnn.multiply(U, dh), transpose_a=True)                    # [1,32,128,128]
            Gr = ttnn.permute(ttnn.exp(gr), (0, 3, 1, 2))
            ttnn.copy(ttnn.add(Sn, ttnn.multiply(self.S[L], Gr)), self.S[L])
            ttnn.copy(self._mmf(csel, E), self.hist3[L])
        return {}

    def seg_mtp(self):
        """MTP rows: hidden row r <- selA @ hid (main pass rows) + selB @ hm (previous MTP output rows)"""
        w = self.W
        src = ttnn.add(self._mmf(ttnn.to_layout(self.selA, TILE), self.hid),
                       self._mmf(ttnn.to_layout(self.selB, TILE), self.hm))
        e = self._rms1p(self._pad32(self.emb_m), w["mtp.pre_fc_norm_embedding"])
        hn = self._rms1p(src, w["mtp.pre_fc_norm_hidden"])
        x = self._mm(ttnn.concat([e, hn], dim=-1), w["mtp.fc"])
        x = ttnn.where(self.mask, x, self._zX)
        rope = self._rope_rows(self.cos_m, self.sin_m)
        x = self.layer(None, x, rope, key="mtp", cur=self.cur_m)
        hm = self._rms1p(x, w["mtp.norm"])
        ttnn.copy(hm, self.hm)
        head = w.w["draft_head"] if "draft_head" in w.w else w["lm_head"]
        lg = self._mm(hm, head, dtype=ttnn.bfloat16)
        rm = ttnn.slice(ttnn.to_layout(lg, RM), [0, 0, 0, 0], [1, 1, self.T, head.shape[-1]])
        return {"argmax": ttnn.argmax(rm, dim=-1)}

    # ---- host side
    def _write(self, dst, t: torch.Tensor, dtype=ttnn.float32):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(t.contiguous(), dtype=dtype, layout=RM), dst)
        self.io["h2d_bytes"] += t.numel() * t.element_size()

    def _read(self, t) -> torch.Tensor:
        t0 = time.perf_counter()
        x = ttnn.to_torch(t)
        self.io["d2h_s"] += time.perf_counter() - t0
        self.io["d2h_n"] += 1
        return x

    def _flash_inputs(self, pos0):
        """chunked flash SDPA window: start = pos0 rounded down to 32; the T rows sit at offset o = pos0 - start"""
        T = self.T
        start = (pos0 // NB) * NB
        o = pos0 - start
        pl, plT = torch.zeros(1, NQ, 64, 32), torch.zeros(1, NQ, 32, 64)
        for r in range(T):
            pl[0, :, o + r, r] = 1.0
            plT[0, :, r, o + r] = 1.0
        self._write(self.cstart, torch.tensor([start], dtype=torch.int32), ttnn.int32)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(pl, dtype=ttnn.bfloat16, layout=TILE), self.place)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(plT, dtype=ttnn.bfloat16, layout=TILE), self.placeT)

    def _write_nv(self, t, n):
        c = self.__dict__.setdefault("_nv_last", {})
        if c.get(id(t)) == n:
            return
        self._write(t, torch.full((1, 1, 1, 8), int(n), dtype=torch.int32), ttnn.uint32)
        c[id(t)] = n

    def _rope_tab(self, pos0):
        T = self.T
        inv = 1.0 / (1e7 ** (torch.arange(0, RD, 2, dtype=torch.int64).float() / RD))
        ang = torch.arange(pos0, pos0 + T, dtype=torch.float32)[:, None] * inv[None]
        cos, sin = torch.ones(T, HD), torch.zeros(T, HD)
        cos[:, :RD] = torch.cat([ang.cos(), ang.cos()], -1)
        sin[:, :RD] = torch.cat([ang.sin(), ang.sin()], -1)
        return cos.reshape(1, 1, T, HD), sin.reshape(1, 1, T, HD)

    def _inputs(self, tokens):
        T = self.T
        self._write(self.emb, self.hw.rows("embed_tokens.weight", tokens).reshape(1, 1, T, H))
        cos, sin = self._rope_tab(self.pos)
        self._write(self.cos, cos)
        self._write(self.sin, sin)
        lim = self.s_max - 1
        self._write(self.cur, torch.tensor([min(self.pos + u // 2, lim) for u in range(2 * T)], dtype=torch.int32),
                    ttnn.int32)
        if self.kvblk:
            self._write(self.cur1, torch.tensor([min(self.pos + r, lim) for r in range(T)], dtype=torch.int32),
                        ttnn.int32)
            if self.flash:
                self._flash_inputs(self.pos)

    def _run(self, name, fn):
        if self.traces is None or name not in self.traces:
            return fn()
        tid, out = self.traces[name]
        ttnn.execute_trace(self.dev, tid, cq_id=0, blocking=False)
        return out

    def verify(self, tokens: list, logits: bool = False, n_valid: int | None = None):
        """T tokens at pos.. (no state change until commit). Returns per-row argmax [T] (and bf16->fp32 logits)."""
        T = self.T
        assert len(tokens) == T
        if self.pos + T > self.s_max:
            raise RuntimeError("context limit")
        t0 = time.perf_counter()
        self._inputs(tokens)
        if self.moe.picks_T:
            self._write_nv(self.nv_main, T if n_valid is None else n_valid)
        f = self._run("pass", self.seg_pass)
        am = self._read(f["argmax"]).reshape(-1)[:T].long().tolist()
        lg = self._read(f["logits"]).float().reshape(T, -1) if logits else None
        if self.route_stats is not None and self.moe.picks_T:   # expert hit counts of real traffic
            nv = T if n_valid is None else n_valid
            for L in range(NL):
                pk = ttnn.to_torch(self.ctl[L]).reshape(32, 32)[:nv, 0:18:2].to(torch.int64) & 0xFFFFFFFF
                self.route_stats[L] += torch.bincount(pk[pk <= NE].reshape(-1), minlength=NE + 1)
        self._v_tokens = list(tokens)
        self.stats["passes"] += 1
        self.stats["rows"] += T
        self.phase["verify"] = self.phase.get("verify", 0) + time.perf_counter() - t0
        return (am, lg) if logits else am

    def commit(self, n: int):
        assert 1 <= n <= self.T
        es = torch.zeros(1, 1, 32, 32)
        es[0, 0, 0, n - 1] = 1
        ms = torch.zeros(1, 1, 32, NV)
        ms[0, 0, :n] = 1
        cs = torch.zeros(1, 1, 32, 32)
        for j in range(3):
            cs[0, 0, j, n + j] = 1
        self._write(self.esel, es)
        self._write(self.msel, ms)
        self._write(self.csel, cs)
        if self.fused:
            self.kcommit.write_sel(n)
        self._run("commit", self.seg_commit)
        self.pos += n
        self.tokens.extend(self._v_tokens[:n])

    def mtp_pass(self, src: str, tokens: list, pos0: int, head_rows=(), src_row: int = 0, src_off: int = 0):
        """src 'main': MTP row r <- main pass row src_off + r; 'mtp': row 0 <- previous MTP output row src_row.
        tokens: the next token per row (padded with the last). Returns the argmax of head_rows."""
        T = self.T
        t0 = time.perf_counter()
        sa, sb = torch.zeros(1, 1, 32, 32), torch.zeros(1, 1, 32, 32)
        if src == "main":
            for r in range(min(T, 32 - src_off)):
                sa[0, 0, r, src_off + r] = 1
        else:
            sb[0, 0, 0, src_row] = 1
        self._write(self.selA, sa)
        self._write(self.selB, sb)
        toks = list(tokens) + [tokens[-1]] * (T - len(tokens))
        self._write(self.emb_m, self.hw.rows("embed_tokens.weight", toks).reshape(1, 1, T, H))
        cos, sin = self._rope_tab(pos0)
        self._write(self.cos_m, cos)
        self._write(self.sin_m, sin)
        lim = self.s_max - 1
        self._write(self.cur_m, torch.tensor([min(pos0 + u // 2, lim) for u in range(2 * T)], dtype=torch.int32),
                    ttnn.int32)
        if self.kvblk:
            self._write(self.cur1_m, torch.tensor([min(pos0 + r, lim) for r in range(T)], dtype=torch.int32),
                        ttnn.int32)
            if self.flash:
                self._flash_inputs(pos0)
        if self.moe.picks_T:                  # only rows up to the head row need the MoE (the MTP K/V come from the
            self._write_nv(self.nv_mtp, max(head_rows) + 1 if head_rows else 0)     # attention before it)
        out = self._run("mtp", self.seg_mtp)
        self.phase["mtp"] = self.phase.get("mtp", 0) + time.perf_counter() - t0
        if not head_rows:
            return []
        am = self._read(out["argmax"]).reshape(-1)[:T].long().tolist()
        if "draft_head" in self.W.w:                       # compact draft head: map tail columns back to token ids
            V, off = self.W.draft_v, self.W.draft_off
            am = [a if a < V else a + off for a in am]
        self.last_probs = [1.0] * len(head_rows)
        return [am[r] for r in head_rows]

    # ---- warm-up and capture (one trace set per process; no allocation after the first capture)
    def warm(self):
        assert self.traces is None
        self.reset()
        self.verify([198] * self.T)
        self.commit(self.T)
        if self.mtp_on:
            self.mtp_pass("main", [198] * self.T, 0, head_rows=(0,))
        self.reset()

    def capture(self):
        tr = {}
        for name, fn in (("pass", self.seg_pass), ("commit", self.seg_commit)) + \
                ((("mtp", self.seg_mtp),) if self.mtp_on else ()):
            tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
            out = fn()
            ttnn.end_trace_capture(self.dev, tid, cq_id=0)
            tr[name] = (tid, out)
        self.traces = tr
        self.reset()
