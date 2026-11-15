"""Generation driver for Qwen3.6 on the P100a: MTP speculative decoding (the model's own MTP layer, all on
the card), batched prompt reading (28-row passes), prefix cache (tt_qwen36/prefix.py), time slicing (tt_qwen36/sessions.py).
The round logic is tt_qwen36/spec.py's (steps 5-6): exact greedy, and exact sampling with deterministic (argmax) drafts.

MTP cell c = (final-normed main hidden at c, token c+1) at rope position / KV slot c -> predicts token c+2.
Prompt: each 28-row chunk is followed by one 28-row MTP pass (K/V for the chunk's cells; the last prompt cell's
row gives the first draft). Decode round: window [x, d0..d_{K-1}] -> verify -> a = leading matches -> commit(a+1)
-> one MTP catch-up pass (rows 0..a, head row a) -> chain the remaining drafts one MTP row at a time.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .model import Q36Pass


class Q36Spec(Q36Pass):
    def __init__(self, dev, W, drafts: int = 3, prefill_rows: int = 28, s_max: int = 32768, prefix_slots: int = 0,
                 **kw):
        super().__init__(dev, W, T=drafts + 1, s_max=s_max, mtp=drafts > 0)
        self.K = drafts
        # past 32k context the prefix-cache / time-slicing slots keep only the first slot_rows KV rows
        import os
        self.slot_rows = 0 if s_max <= 32768 else min(s_max, int(os.environ.get("P100A_Q36_SLOT_ROWS", "32768")))
        if self.slot_rows:
            from .kmoe import PageCopy
            self.page_copy = PageCopy(dev)
        self.min_p = 0.0
        self.ple_state = None
        self.pre = self.sibling(prefill_rows) if prefill_rows > self.T else None
        if self.pre is not None:
            self.pre.ple_state = None
        self.use_prefill = True
        self.debug_top2 = False
        self.last_probs = []
        self.spec_stats = {"rounds": 0, "accepted": 0, "drafted": 0, "emitted": 0}
        # adaptive draft count. Per draft position j: decayed counts of "proposed" and "accepted given the
        # earlier ones were"; each round picks k maximizing expected tokens / expected ms (P100A_Q36_ADAPT=0: always K)
        import os
        self.adapt = os.environ.get("P100A_Q36_ADAPT", "1") == "1" and drafts > 0
        self.cost = [float(x) for x in os.environ.get("P100A_Q36_COST", "26,1.2,1.4").split(",")]  # base, row, chain
        self.acc_n = [0.0] * (drafts + 1)
        self.acc_a = [0.0] * (drafts + 1)
        self.pcache = None
        if prefix_slots and self.pre is not None:
            from ..prefix import PrefixCache
            self.pcache = PrefixCache(self, prefix_slots, prefill_rows)

    def cached_prefix(self, ids) -> int:
        if self.pcache is None or not self.use_prefill:
            return 0
        return self.pcache.lookup(ids)[1]

    def capture_all(self):
        """all eager warm-ups first, then every trace back to back (no device allocation after the first capture)"""
        self.warm()
        if self.pre is not None:
            self.pre.warm()
        if self.slot_rows and self.pcache is not None and self.pcache.slots:   # compile the page copy before capture
            k = next(iter(self.kc))
            self.page_copy(self.kc[k], self.pcache.slots[0].bufs[f"k{k}"], 32)
        self.capture()
        if self.pre is not None:
            self.pre.capture()
        self.reset()

    def generate(self, ids: list, max_new: int, eos=(), spec: bool = True, on_token=None, temperature: float = 0.0,
                 top_p: float = 1.0, top_k: int = 0, seed=None):
        res = {}
        for _ in self.steps(ids, max_new, eos, spec, on_token, temperature, top_p, top_k, seed, res):
            pass
        return res["out"]

    def steps(self, ids: list, max_new: int, eos=(), spec: bool = True, on_token=None, temperature: float = 0.0,
              top_p: float = 1.0, top_k: int = 0, seed=None, result=None):
        T, K = self.T, self.K
        spec = spec and K > 0
        samp = temperature > 0
        rng = np.random.default_rng(seed)

        def dist(lg):
            z = lg.double() / temperature
            if top_k and top_k < z.numel():
                kth = torch.topk(z, top_k).values[-1]
                z = torch.where(z < kth, torch.full_like(z, -float("inf")), z)
            pr = torch.softmax(z, -1)
            if top_p < 1.0:
                sp, si = torch.sort(pr, descending=True)
                keep = torch.cumsum(sp, 0) - sp < top_p
                mask = torch.zeros_like(pr, dtype=torch.bool)
                mask[si[keep]] = True
                pr = torch.where(mask, pr, torch.zeros_like(pr))
                pr = pr / pr.sum()
            return pr.numpy()

        def draw(pr):
            return int(rng.choice(pr.size, p=pr))

        stop = {"flag": False}

        def emit(t):
            out.append(t)
            if on_token and on_token(t):
                stop["flag"] = True
        self.reset()
        n = len(ids)
        out = []
        t0 = time.perf_counter()
        pm = self.pre if (self.pre is not None and self.use_prefill) else self
        if pm is not self:
            pm.reset_host()
        P = pm.T
        i = 0
        pc = self.pcache if (self.pcache is not None and pm is not self) else None
        saves = set()
        if pc is not None:
            pc.stats["lookups"] += 1
            slot, b = pc.lookup(ids)
            if slot is not None:
                pc.restore(slot)
                pm.pos, pm.tokens = b, list(ids[:b])
                i = b
                pc.stats["hits"] += 1
                pc.stats["hit_tokens"] += b
            saves = pc.plan(ids)
            pc.live = []
        self.last_cached = cached0 = i
        d, head_row = [], 0
        while i < n:
            chunk = ids[i:i + P]
            nv = len(chunk)
            last = i + nv >= n
            res = pm.verify(chunk + [chunk[-1]] * (P - nv), n_valid=nv, logits=last and (self.debug_top2 or samp))
            outv = res[0] if isinstance(res, tuple) else res
            if isinstance(res, tuple):
                v, ix = res[1][nv - 1].float().topk(2)
                self.last_top2 = [(int(ix[0]), round(float(v[0]), 3)), (int(ix[1]), round(float(v[1]), 3))]
                if samp:
                    outv = list(outv)
                    outv[nv - 1] = draw(dist(res[1][nv - 1]))
            pm.commit(nv)
            if spec:
                toks = [ids[c + 1] if c + 1 < n else outv[nv - 1] for c in range(i, i + nv)]
                head = (nv - 1,) if last else ()
                d = pm.mtp_pass("main", toks, i, head_rows=head)
                if head:
                    head_row = nv - 1
            i += nv
            if pc is not None and i in saves and i < n:
                pc.save(ids, i, None)
            if i < n:
                yield "prompt"
        if pc is not None:
            pc.live = list(ids)
        if pm is not self:
            self.pos, self.tokens = pm.pos, list(pm.tokens)
        t1 = time.perf_counter()
        x = outv[nv - 1]
        p = n
        drafts = []
        if spec:
            drafts = self._chain(d, p - 1, head_row, self.pick_k())
        acc_hist, prop_hist = [], []
        while True:
            emit(x)
            if stop["flag"] or x in eos or len(out) >= max_new or p + T > self.s_max:
                break
            nprop = len(drafts) if spec else 0
            window = [x] + drafts[:nprop] + [x] * (K - nprop)
            if samp:
                _, lg = self.verify(window, n_valid=1 + nprop, logits=True)
                a, final = 0, None
                while a < nprop:
                    pr = dist(lg[a])
                    dd = window[a + 1]
                    if rng.random() < pr[dd]:
                        a += 1
                        continue
                    pr[dd] = 0.0
                    final = draw(pr / pr.sum())
                    break
                if final is None:
                    final = draw(dist(lg[a]))
                emitted = window[1:a + 1] + [final]
            else:
                outv = self.verify(window, n_valid=1 + nprop)
                a = 0
                while a < nprop and window[a + 1] == outv[a]:
                    a += 1
                emitted = outv[: a + 1]
            self.commit(a + 1)
            acc_hist.append(a)
            prop_hist.append(nprop)
            if spec:
                self.note_acceptance(nprop, a)
            for tkn in emitted[:-1]:
                emit(tkn)
                if stop["flag"] or tkn in eos or len(out) >= max_new:
                    break
            else:
                x = emitted[-1]
                if spec:
                    k = self.pick_k()
                    d0 = self.mtp_pass("main", emitted, p, head_rows=(a,) if k else ())
                    drafts = self._chain(d0, p + a, a, k)
                p += a + 1
                yield "round"
                continue
            break
        t2 = time.perf_counter()
        self.spec_stats["rounds"] += len(acc_hist)
        self.spec_stats["accepted"] += sum(acc_hist)
        self.spec_stats["drafted"] += sum(prop_hist)
        result["out"] = (out[:max_new], {"prompt_s": t1 - t0, "decode_s": t2 - t1, "rounds": len(acc_hist),
                                         "accepted": acc_hist, "proposed": prop_hist, "cached": cached0})

    def pick_k(self) -> int:
        """drafts to propose this round: argmax_k (1 + sum_{j<=k} prod_{i<=j} a_i) / (base + k*row + (k-1)*chain)"""
        if not self.adapt:
            return self.K
        base, row, chain = self.cost
        best, best_v, p, tok = 0, 1.0 / base, 1.0, 1.0
        for k in range(1, self.K + 1):
            a = (self.acc_a[k] + 1.0) / (self.acc_n[k] + 2.0)          # prior 0.5
            p *= a
            tok += p
            v = tok / (base + k * row + (k - 1) * chain)
            if v > best_v:
                best, best_v = k, v
        return best

    def note_acceptance(self, nprop: int, a: int):
        for j in range(1, self.K + 1):
            self.acc_n[j] *= 0.9
            self.acc_a[j] *= 0.9
        for j in range(1, nprop + 1):
            if a >= j - 1:                                             # draft j was reached
                self.acc_n[j] += 1.0
                self.acc_a[j] += float(a >= j)

    def resume_steps(self, x: int, max_new: int, eos=(), spec: bool = True, on_token=None, temperature: float = 0.0,
                     top_p: float = 1.0, top_k: int = 0, seed=None, result=None):
        """continue a request whose state (positions < self.pos) was put back on this model (tt_qwen36/q36/batch.py
        to_single); x = next input token (already emitted). The first round proposes no drafts (no MTP cell for the
        last position yet); the MTP K/V cells of the batched stretch are stale, which only costs acceptance."""
        T, K = self.T, self.K
        spec = spec and K > 0
        samp = temperature > 0
        rng = np.random.default_rng(seed)
        out, acc_hist, prop_hist = [], [], []
        stop = {"flag": False}
        p = self.pos
        drafts = []
        t1 = time.perf_counter()
        while True:
            if stop["flag"] or x in eos or len(out) >= max_new or p + T > self.s_max:
                break
            nprop = len(drafts) if spec else 0
            window = [x] + drafts[:nprop] + [x] * (K - nprop)
            res = self.verify(window, n_valid=1 + nprop, logits=samp)
            if samp:
                outv, lg = res
                a, final = 0, None
                while a < nprop:
                    pr = self._dist(lg[a], temperature, top_k, top_p)
                    dd = window[a + 1]
                    if rng.random() < pr[dd]:
                        a += 1
                        continue
                    pr[dd] = 0.0
                    final = int(rng.choice(pr.size, p=pr / pr.sum()))
                    break
                if final is None:
                    pr = self._dist(lg[a], temperature, top_k, top_p)
                    final = int(rng.choice(pr.size, p=pr))
                emitted = window[1:a + 1] + [final]
            else:
                outv = res
                a = 0
                while a < nprop and window[a + 1] == outv[a]:
                    a += 1
                emitted = outv[: a + 1]
            self.commit(a + 1)
            acc_hist.append(a)
            prop_hist.append(nprop)
            if spec:
                self.note_acceptance(nprop, a)
            done = False
            for tkn in emitted:
                out.append(tkn)
                if (on_token and on_token(tkn)) or tkn in eos or len(out) >= max_new:
                    done = True
                    break
            if done:
                break
            x = emitted[-1]
            if spec:
                k = self.pick_k()
                d0 = self.mtp_pass("main", emitted, p, head_rows=(a,) if k else ())
                drafts = self._chain(d0, p + a, a, k)
            p += a + 1
            yield "round"
        result["out"] = (out[:max_new], {"prompt_s": 0.0, "decode_s": time.perf_counter() - t1, "rounds": len(acc_hist),
                                         "accepted": acc_hist, "proposed": prop_hist, "cached": 0})

    @staticmethod
    def _dist(lg, temperature, top_k, top_p):
        z = lg.double() / temperature
        if top_k and top_k < z.numel():
            kth = torch.topk(z, top_k).values[-1]
            z = torch.where(z < kth, torch.full_like(z, -float("inf")), z)
        pr = torch.softmax(z, -1)
        if top_p < 1.0:
            sp, si = torch.sort(pr, descending=True)
            keep = torch.cumsum(sp, 0) - sp < top_p
            mask = torch.zeros_like(pr, dtype=torch.bool)
            mask[si[keep]] = True
            pr = torch.where(mask, pr, torch.zeros_like(pr))
            pr = pr / pr.sum()
        return pr.numpy()

    def _chain(self, d0: list, cell0: int, src_row: int, k: int | None = None) -> list:
        k = self.K if k is None else k
        drafts = list(d0)[:k]
        while 0 < len(drafts) < k:
            j = len(drafts)
            drafts += self.mtp_pass("mtp", [drafts[-1]], cell0 + j, head_rows=(0,), src_row=src_row if j == 1 else 0)
        return drafts[:k]
