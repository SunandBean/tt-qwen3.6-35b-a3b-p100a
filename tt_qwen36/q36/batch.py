"""True batched decode. B users, one token each per pass (row b = user b), no MTP drafts.

A Q36Batch is a sibling of the single-sequence model (same weights, MoE / attention code) with its own per-user state:
  GDN state S_b [B,32,128,128] per layer, updated in place by the batched core (tt_qwen36/q36/kernels/gdn_bat);
  conv history H [1,1,32,8192] per layer: rows 3b..3b+2 = user b's last three qkv rows; the fused conv kernel runs
    with batch selection constants (output row b = sum_k w_k E[3b+k] + w_3 E[3B+b], E = H + place(x)) and the new
    history is a selection of E;
  KV caches [2,1,B*s_user,256] per attention layer: user b owns rows [b*s_user, (b+1)*s_user); the paged SDPA reads
    them through a per-user page table, so attention is per user with no copying.
Prompts are read on the single-sequence path (28-row passes, prefix cache), then `admit(b)` copies the live state into
slot b (page copies for S and KV; a captured trace places the conv history rows).
"""
from __future__ import annotations

import math
import time

import torch
import ttnn

from .kmoe import PageCopy
from .model import DK, EPS, H, HD, NB, NKV, NL, NV, QKV, RM, TILE, VOC, Q36Pass, is_attn


class Q36Batch:
    def __init__(self, single: Q36Pass, B: int = 4, s_user: int = 16384):
        assert 1 <= B <= 8 and s_user % NB == 0
        self.m = m = single.sibling(B)                      # T = B rows: MoE, rope, emb buffers of its own
        self.B, self.s_user, self.dev = B, s_user, single.dev
        dev = self.dev
        z = torch.zeros
        mk = lambda t, dt=ttnn.float32, lay=TILE: ttnn.from_torch(t.contiguous(), dtype=dt, layout=lay, device=dev,
                                                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.mk = mk
        from .kgdn import GdnBatch, GdnConv36
        self.core = GdnBatch(dev, B, eps=EPS)
        self.conv = GdnConv36(dev)
        c = z(1, 1, 32, 160)                                # batch selection constants for the conv kernel
        for b in range(B):
            c[0, 0, 3 * B + b, b] = 1.0                     # SH: E row 3B+b <- x row b
            for k in range(3):
                c[0, 0, b, 32 * (k + 1) + 3 * b + k] = 1.0  # UP_k: out row b <- E row 3b+k
            c[0, 0, b, 32 * 4 + 3 * B + b] = 1.0            # UP_3: out row b <- E row 3B+b
        self.conv.consts = mk(c)
        hs = z(32, 32)                                      # new history: rows 3b..3b+2 <- E rows 3b+1, 3b+2, 3B+b
        for b in range(B):
            hs[3 * b, 3 * b + 1] = hs[3 * b + 1, 3 * b + 2] = hs[3 * b + 2, 3 * B + b] = 1.0
        self.hsel = mk(hs.reshape(1, 1, 32, 32))
        self.S = {L: mk(z(B, NV, DK, DK)) for L in m.S}
        self.Hh = {L: mk(z(1, 1, 32, QKV)) for L in m.S}
        self.E = mk(z(1, 1, 32, QKV))
        self.cv = mk(z(1, 1, 32, QKV))
        self.y = mk(z(1, 1, 32, NV * DK))
        st = B * s_user
        kvdt = ttnn.bfloat8_b if m.kv_bfp8 else ttnn.bfloat16
        self.blk = m.kvblk
        kvshape = (st // NB, NKV, NB, HD) if self.blk else (NKV, 1, st, HD)
        self.kc = {L: mk(z(*kvshape), kvdt) for L in m.kc if L != "mtp"}
        self.vc = {L: mk(z(*kvshape), kvdt) for L in m.kc if L != "mtp"}
        if m.kv_bfp8:
            from .kmoe import KvWrite8
            m.kvw = KvWrite8(dev, B, NKV, HD, st, blk=self.blk)
        elif self.blk:
            from .kmoe import KvWriteBlk
            m.kvw = KvWriteBlk(dev, B, NKV, HD, st)
        else:
            from ..kkv import KvWrite
            m.kvw = KvWrite(dev, B, NKV, HD, st)
        nbu, nbt = s_user // NB, st // NB                   # pages per user, per head
        pt = torch.stack([torch.arange(nbu, dtype=torch.int32) + (u % 2) * nbt + (u // 2) * nbu for u in range(2 * B)])
        self.ptab = mk(pt, ttnn.int32, RM)
        if self.blk:                                        # block layout: user b owns blocks [b*nbu, (b+1)*nbu)
            self.ptab = mk(torch.stack([torch.arange(nbu, dtype=torch.int32) + b * nbu for b in range(B)]),
                           ttnn.int32, RM)
            self.cur1 = mk(torch.zeros(B, dtype=torch.int32), ttnn.int32, RM)
        self.kvcur = mk(torch.zeros(2 * B, dtype=torch.int32), ttnn.int32, RM)
        self.pcopy = PageCopy(dev)
        # per-user placement constants for admit(): history rows 3b..3b+2 <- live hist3 rows 0..2
        self.place = []
        for b in range(B):
            P, M = z(32, 32), torch.ones(1, 1, 32, QKV)
            for k in range(3):
                P[3 * b + k, k] = 1.0
                M[0, 0, 3 * b + k] = 0.0
            self.place.append((mk(P.reshape(1, 1, 32, 32)), mk(M)))
        self.back = []                                      # slot b -> live hist3 rows 0..2 (rows 3.. zero)
        for b in range(B):
            Q = z(32, 32)
            for k in range(3):
                Q[k, 3 * b + k] = 1.0
            self.back.append(mk(Q.reshape(1, 1, 32, 32)))
        self.pos = [0] * B
        self.active = [False] * B
        self.traces = None
        self.stats = {"passes": 0, "tokens": 0}

    # ---- the batched pass
    def gdn(self, L, x):
        m, w = self.m, self.m.W
        k_ = f"{L}."
        pin = m._mm(ttnn.typecast(x, ttnn.bfloat16), w[k_ + "in"])                     # [1,1,32,12352]
        self.conv(pin, self.Hh[L], [w[k_ + f"conv{t}"] for t in range(4)], self.cv, self.E)
        ttnn.copy(m._mmf(self.hsel, self.E), self.Hh[L])
        self.core(self.cv, self.S[L], pin, w[k_ + "gnorm"], self.y, pin, m.kg_dt[L], m.kg_a[L],
                  z_off=QKV // 32, ab_off=(QKV + NV * DK) // 32)
        return m._mm(self.y, w[k_ + "out"])

    def seg_pass(self):
        m, w = self.m, self.m.W
        X = m._pad32(m.emb)
        cosk, sink, cosq, sinq = m._rope_rows(m.cos, m.sin)
        for L in range(NL):
            h = m._rms1p(X, w[f"{L}.ln1"])
            if is_attn(L):
                y = m.attn(L, h, cosk, sink, cosq, sinq, m.cur, caches=(self.kc[L], self.vc[L]), kvcur=self.kvcur,
                           ptab=self.ptab, cur1=self.cur1 if self.blk else None)
            else:
                y = self.gdn(L, h)
            X = ttnn.add(X, y)
            h = m._rms1p(X, w[f"{L}.ln2"])
            X = ttnn.add(X, m.moe_block(L, h))
        hid = m._rms1p(X, w["norm"])
        lg = m._mm(hid, w["lm_head"], dtype=ttnn.bfloat16)
        rm = ttnn.slice(ttnn.to_layout(lg, RM), [0, 0, 0, 0], [1, 1, self.B, VOC])
        return {"logits": rm, "argmax": ttnn.argmax(rm, dim=-1)}

    def seg_admit(self, b):
        """slot b <- the single-sequence model's conv history (the S / KV page copies run outside the trace)"""
        m = self.m
        P, M = self.place[b]
        for L in m.S:
            ttnn.copy(ttnn.add(ttnn.multiply(self.Hh[L], M), m._mm_sel(P, m.hist3[L])), self.Hh[L])
        return {}

    # ---- host side
    def step(self, tokens: list, logits: bool = False):
        """one token for every slot (inactive slots: any token, ignored). Returns argmax per slot (and logits)."""
        m, B = self.m, self.B
        rows = [t if self.active[b] else 0 for b, t in enumerate(tokens)]
        m._write(m.emb, m.hw.rows("embed_tokens.weight", rows).reshape(1, 1, B, H))
        inv = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.int64).float() / 64))
        ang = torch.tensor(self.pos, dtype=torch.float32)[:, None] * inv[None]
        cos, sin = torch.ones(B, HD), torch.zeros(B, HD)
        cos[:, :64] = torch.cat([ang.cos(), ang.cos()], -1)
        sin[:, :64] = torch.cat([ang.sin(), ang.sin()], -1)
        m._write(m.cos, cos.reshape(1, 1, B, HD))
        m._write(m.sin, sin.reshape(1, 1, B, HD))
        m._write(m.cur, torch.tensor([self.pos[u // 2] for u in range(2 * B)], dtype=torch.int32), ttnn.int32)
        lim = self.s_user - 1                               # a full slot writes only inside its own region
        m._write(self.kvcur, torch.tensor([(u // 2) * self.s_user + min(self.pos[u // 2], lim) for u in range(2 * B)],
                                          dtype=torch.int32), ttnn.int32)
        if self.blk:
            m._write(self.cur1, torch.tensor(self.pos, dtype=torch.int32), ttnn.int32)
        if m.moe.picks_T:
            m._write_nv(m.nv_main, max((b + 1 for b in range(B) if self.active[b]), default=1))
        if self.traces is None:
            f = self.seg_pass()
        else:
            tid, f = self.traces["pass"]
            ttnn.execute_trace(self.dev, tid, cq_id=0, blocking=False)
        am = m._read(f["argmax"]).reshape(-1)[:B].long().tolist()
        lg = m._read(f["logits"]).float().reshape(B, -1) if logits else None
        for b in range(B):
            if self.active[b] and self.pos[b] + 2 < self.s_user:   # full slots are frozen (scheduler moves them out)
                self.pos[b] += 1
        self.stats["passes"] += 1
        self.stats["tokens"] += sum(self.active)
        return (am, lg) if logits else am

    def admit(self, b: int, single: Q36Pass):
        """slot b <- the live state of the single-sequence model after it read a prompt (pos = single.pos)"""
        n = single.pos
        assert n + 1 <= self.s_user
        for L in single.S:                                  # S: 512 pages of [1,32,128,128] -> pages of user b
            self._pages(single.S[L], self.S[L], 0, b * NV * 16, NV * 16)
        if self.blk:                                        # block layout: one contiguous page range per cache
            npg = -(-n // NB) * NKV * (HD // 32)
            for L in self.kc:
                for src, dst in ((single.kc[L], self.kc[L]), (single.vc[L], self.vc[L])):
                    self._pages(src, dst, 0, b * (self.s_user // NB) * NKV * (HD // 32), npg)
        for L in ([] if self.blk else self.kc):             # KV rows < n -> user b's rows
            st, su = single.kc[L].shape[2], self.s_user
            wt = HD // 32
            npg = -(-n // 32) * wt
            for hd in range(NKV):
                for src, dst in ((single.kc[L], self.kc[L]), (single.vc[L], self.vc[L])):
                    self._pages(src, dst, hd * (st // 32) * wt, (hd * self.B * su + b * su) // 32 * wt, npg)
        if self.traces is None:
            self.seg_admit(b)
        else:
            ttnn.execute_trace(self.dev, self.traces[f"admit{b}"][0], cq_id=0, blocking=False)
        self.pos[b] = n
        self.active[b] = True

    def _pages(self, src, dst, s0, d0, n):
        pc = self.pcopy
        per = -(-n // len(pc.cores))
        page = {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088}[src.dtype]
        rt = ttnn.RuntimeArgs()
        for i, c in enumerate(pc.cores):
            rt[c.x][c.y] = [src.buffer_address(), dst.buffer_address(), s0, d0, min(n, i * per), min(n, (i + 1) * per)]
        ct = [page] + ttnn.TensorAccessorArgs(src).get_compile_time_args() + ttnn.TensorAccessorArgs(dst).get_compile_time_args()
        cbs = [ttnn.CBDescriptor(total_size=8 * page, core_ranges=pc.grid, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=src.dtype, page_size=page)])]
        from .kmoe import KDIR
        k = ttnn.KernelDescriptor(kernel_source=f"{KDIR}/pagecopy.cpp", core_ranges=pc.grid, compile_time_args=ct,
                                  runtime_args=rt, config=ttnn.ReaderConfigDescriptor())
        ttnn.generic_op([src, dst], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=cbs))

    def seg_back(self, b):
        m = self.m
        for L in m.S:
            ttnn.copy(m._mm_sel(self.back[b], self.Hh[L]), m.hist3[L])
        return {}

    def to_single(self, b: int, single):
        """the request in slot b continues on the single-sequence model (MTP path): S, KV rows < pos, conv history"""
        n = self.pos[b]
        for L in single.S:
            self._pages(self.S[L], single.S[L], b * NV * 16, 0, NV * 16)
        if self.blk:
            npg = -(-n // NB) * NKV * (HD // 32)
            for L in self.kc:
                for src, dst in ((self.kc[L], single.kc[L]), (self.vc[L], single.vc[L])):
                    self._pages(src, dst, b * (self.s_user // NB) * NKV * (HD // 32), 0, npg)
        for L in ([] if self.blk else self.kc):
            st, su, wt = single.kc[L].shape[2], self.s_user, HD // 32
            npg = -(-n // 32) * wt
            for hd in range(NKV):
                for src, dst in ((self.kc[L], single.kc[L]), (self.vc[L], single.vc[L])):
                    self._pages(src, dst, (hd * self.B * su + b * su) // 32 * wt, hd * (st // 32) * wt, npg)
        if self.traces is None:
            self.seg_back(b)
        else:
            ttnn.execute_trace(self.dev, self.traces[f"back{b}"][0], cq_id=0, blocking=False)
        single.pos = n
        if single.pre is not None:
            single.pre.pos = n
        if single.pcache is not None:
            single.pcache.live = []                         # live KV rows no longer match any prompt
        self.release(b)

    def release(self, b: int):
        self.active[b] = False

    # ---- warm-up / capture (after the single model's own capture is NOT allowed: call before it, see warm_all)
    def warm(self, single):
        single.reset()
        single.verify([198] * single.T)
        single.commit(single.T)
        for b in range(self.B):
            self.admit(b, single)
        self.step([198] * self.B)
        for b in range(self.B):
            self.seg_back(b)
            self.release(b)
        single.reset()

    def capture(self):
        tr = {}
        tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
        out = self.seg_pass()
        ttnn.end_trace_capture(self.dev, tid, cq_id=0)
        tr["pass"] = (tid, out)
        for b in range(self.B):
            tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
            out = self.seg_admit(b)
            ttnn.end_trace_capture(self.dev, tid, cq_id=0)
            tr[f"admit{b}"] = (tid, out)
            tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
            out = self.seg_back(b)
            ttnn.end_trace_capture(self.dev, tid, cq_id=0)
            tr[f"back{b}"] = (tid, out)
        self.traces = tr
