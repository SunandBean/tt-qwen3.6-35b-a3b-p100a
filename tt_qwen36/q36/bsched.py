"""Hybrid scheduler: time slicing with MTP for 1-2 requests, true batched decode for 3+
(tt_qwen36/q36/batch.py). Same interface as tt_qwen36/sessions.Scheduler (submit / busy; Session objects).

Measured over HTTP (bench/concurrency.py, 200 tokens each, 2026-11-13 08:55): time slicing 98 / 91 / 78 tok/s total
for 1 / 2 / 4 users, batched 98 (solo) / 64.5 / 109. So:
  slice mode (<= 2 requests): one request runs Q36Spec.steps / resume_steps on the single model (MTP drafts); after a
    quantum it is parked in a batch slot (bt.admit: its state is copied there) and the next one resumes from its slot
    (bt.to_single) or starts reading its prompt. Parking happens at decode-round boundaries.
  batch mode (>= 3 requests): every request sits in a slot; one batched pass advances all by one token.
Prompts are always read on the single-sequence path (28-row passes, prefix cache).
"""
from __future__ import annotations

import collections
import threading
import time

import numpy as np
import torch


class BatchScheduler:
    def __init__(self, model, batch, eos=(), quantum_s: float = 0.5, batch_from: int = 3):
        self.m, self.bt, self.eos = model, batch, set(eos)
        self.B, self.quantum, self.batch_from = batch.B, quantum_s, batch_from
        self.wait = collections.deque()
        self.slots = [None] * self.B          # (Session, st) parked or batched
        self.solo = None                      # (Session, generator, st) running on the single model
        self.rr = 0
        self.cv = threading.Condition()
        self.stats = {"batched_passes": 0, "batched_tokens": 0, "parks": 0, "resumes": 0, "admitted": 0,
                      "max_active": 0}
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, sess):
        with self.cv:
            self.wait.append(sess)
            self.cv.notify()

    def busy(self) -> bool:
        with self.cv:
            return bool(self.wait or self.solo or any(self.slots))

    def _fits(self, sess, st=None) -> bool:
        """a request may enter a batch slot while its current context leaves room in the slot (s_user rows); one that
        later outgrows it is moved back to the single-sequence path (_batched_pass), never cut off"""
        return len(sess.ids) + (len(st["out"]) if st else 0) + 256 <= self.bt.s_user

    def _full(self, b) -> bool:
        return self.bt.pos[b] + 2 >= self.bt.s_user

    def _n_active(self):
        return (self.solo is not None) + sum(s is not None for s in self.slots)

    # ---- per-request host state
    def _st(self, sess):
        kw = sess.kwargs
        return {"t": float(kw.get("temperature") or 0.0), "p": float(kw.get("top_p") or 1.0),
                "k": int(kw.get("top_k") or 0), "rng": np.random.default_rng(kw.get("seed")), "out": [],
                "t0": time.perf_counter(), "t1": None, "x": None, "cached": 0}

    def _pick(self, st, am, lg):
        if st["t"] <= 0:
            return am
        z = lg.double() / st["t"]
        if st["k"] and st["k"] < z.numel():
            kth = torch.topk(z, st["k"]).values[-1]
            z = torch.where(z < kth, torch.full_like(z, -float("inf")), z)
        pr = torch.softmax(z, -1)
        if st["p"] < 1.0:
            sp, si = torch.sort(pr, descending=True)
            keep = torch.cumsum(sp, 0) - sp < st["p"]
            mask = torch.zeros_like(pr, dtype=torch.bool)
            mask[si[keep]] = True
            pr = torch.where(mask, pr, torch.zeros_like(pr))
            pr = pr / pr.sum()
        p = pr.numpy()
        return int(st["rng"].choice(p.size, p=p))

    def _on_token(self, sess, st):
        def f(t):
            if st["t1"] is None:
                st["t1"] = time.perf_counter()
            st["out"].append(t)
            return sess.on_token(t) if sess.on_token else False
        return f

    def _finish(self, sess, st, err=None):
        t2 = time.perf_counter()
        t1 = st["t1"] or t2
        info = {"prompt_s": t1 - st["t0"], "decode_s": t2 - t1, "rounds": 0, "accepted": [], "proposed": [],
                "cached": st["cached"]}
        res = st.get("res", {}).get("out")
        if res is not None:                          # MTP statistics of the solo stretch
            for k in ("rounds", "accepted", "proposed"):
                info[k] = res[1].get(k, info[k])
            if not st.get("resumed_any"):
                info["cached"] = res[1].get("cached", 0)
                info["prompt_s"] = res[1].get("prompt_s", info["prompt_s"])
        sess.result["out"] = (st["out"][:sess.max_new], info)
        sess.error = err
        sess.done.set()

    def _ended(self, sess, st) -> bool:
        return bool(sess.cancelled or (st["out"] and st["out"][-1] in self.eos) or len(st["out"]) >= sess.max_new)

    # ---- the card loop
    def _loop(self):
        while True:
            with self.cv:
                while not (self.wait or self.solo or any(self.slots)):
                    self.cv.wait()
                n_total = self._n_active() + len(self.wait)
                fit_all = all(self._fits(w) for w in self.wait) and (self.solo is None or
                                                                    self._fits(self.solo[0], self.solo[2])) and \
                    not any(s is not None and self._full(b) for b, s in enumerate(self.slots))
            try:
                if n_total >= self.batch_from and fit_all and (None in self.slots or not self.wait):
                    self._batch_mode()
                else:
                    self._slice_mode()
            except Exception as e:  # noqa: BLE001  (surfaced to the requests)
                for b, s in enumerate(self.slots):
                    if s is not None:
                        self._finish(s[0], s[1], e)
                        self.slots[b] = None
                        self.bt.release(b)
                if self.solo is not None:
                    self._finish(self.solo[0], self.solo[2], e)
                    self.solo = None

    # ---- slice mode
    def _start_solo(self, sess):
        st = self._st(sess)
        kw = sess.kwargs
        st["res"] = {}
        gen = self.m.steps(sess.ids, sess.max_new, eos=kw.get("eos", self.eos), spec=kw.get("spec", True),
                           on_token=self._on_token(sess, st), temperature=kw.get("temperature", 0.0),
                           top_p=kw.get("top_p", 1.0), top_k=kw.get("top_k", 0), seed=kw.get("seed"),
                           result=st["res"])
        self.solo = (sess, gen, st)
        self.stats["admitted"] += 1

    def _resume_solo(self, b):
        sess, st = self.slots[b]
        self.bt.to_single(b, self.m)
        self.slots[b] = None
        kw = sess.kwargs
        st["res"] = {}
        gen = self.m.resume_steps(st["x"], sess.max_new - len(st["out"]), eos=kw.get("eos", self.eos),
                                  spec=kw.get("spec", True), on_token=self._on_token(sess, st),
                                  temperature=kw.get("temperature", 0.0), top_p=kw.get("top_p", 1.0),
                                  top_k=kw.get("top_k", 0), seed=kw.get("seed"), result=st["res"])
        st["resumed_any"] = True
        self.solo = (sess, gen, st)
        self.stats["resumes"] += 1

    def _park_solo(self) -> bool:
        """solo -> a free slot at its current round boundary; False if it finished meanwhile"""
        sess, gen, st = self.solo
        while st.get("last") != "round":
            if not self._step_solo():
                return False
            sess, gen, st = self.solo
        gen.close()
        b = self.slots.index(None)
        self.bt.admit(b, self.m)
        st["x"] = st["out"][-1]
        st["last"] = None
        self.slots[b] = (sess, st)
        self.solo = None
        self.stats["parks"] += 1
        return True

    def _step_solo(self) -> bool:
        """one generator step of the solo request; False when it finished"""
        sess, gen, st = self.solo
        try:
            st["last"] = next(gen)
            if sess.cancelled:
                raise StopIteration
            return True
        except StopIteration:
            gen.close()
            self._finish(sess, st)
            self.solo = None
            return False

    def _slice_mode(self):
        if self.solo is None:
            with self.cv:
                new = self.wait.popleft() if self.wait else None
            parked = [b for b, s in enumerate(self.slots) if s is not None]
            full = [b for b in parked if self._full(b)]
            if full:
                self._resume_solo(full[0])
                self.rr += 1
                parked, new = [], new                       # run it now; a new request waits for the next turn
                if new is not None:
                    with self.cv:
                        self.wait.appendleft(new)
                    new = None
            if new is not None and (not parked or self.rr % 2 == 0):
                self._start_solo(new)
            elif parked:
                self._resume_solo(parked[self.rr % len(parked)])
                if new is not None:
                    with self.cv:
                        self.wait.appendleft(new)
            elif new is not None:
                self._start_solo(new)
            else:
                return
            self.rr += 1
        t0 = time.perf_counter()
        while self.solo is not None:
            if not self._step_solo():
                return
            with self.cv:
                others = bool(self.wait) or any(self.slots)
            if others and time.perf_counter() - t0 >= self.quantum and self.solo[2].get("last") == "round":
                if None in self.slots and self._fits(self.solo[0], self.solo[2]):
                    self._park_solo()
                    return
                # too long for a slot: it runs to the end on its own; the others wait

    # ---- batch mode
    def _admit_new(self, sess):
        st = self._st(sess)
        kw = sess.kwargs
        out, info = self.m.generate(sess.ids, 1, eos=(), spec=False, temperature=kw.get("temperature", 0.0),
                                    top_p=kw.get("top_p", 1.0), top_k=kw.get("top_k", 0), seed=kw.get("seed"))
        st["cached"] = info.get("cached", 0)
        st["t1"] = time.perf_counter()
        self.stats["admitted"] += 1
        st["out"].append(out[0])
        if (sess.on_token and sess.on_token(out[0])) or self._ended(sess, st):
            self._finish(sess, st)
            return
        b = self.slots.index(None)
        self.bt.admit(b, self.m)
        st["x"] = out[0]
        self.slots[b] = (sess, st)

    def _batch_mode(self):
        if self.solo is not None:
            self._park_solo()
        with self.cv:
            while self.wait and None in self.slots:
                new = self.wait.popleft()
                self.cv.release()
                try:
                    self._admit_new(new)
                finally:
                    self.cv.acquire()
        self.stats["max_active"] = max(self.stats["max_active"], self._n_active())
        while sum(s is not None and not self._full(b) for b, s in enumerate(self.slots)) >= self.batch_from:
            self._batched_pass()
            with self.cv:
                if self.wait and None in self.slots:
                    return                                   # admit newcomers first
        # fewer than batch_from left: back to slicing (they stay parked in their slots)

    def _batched_pass(self):
        toks = [s[1]["x"] if s is not None else 0 for s in self.slots]
        samp = any(s is not None and s[1]["t"] > 0 for s in self.slots)
        res = self.bt.step(toks, logits=samp)
        am, lg = (res[0], res[1]) if samp else (res, None)
        self.stats["batched_passes"] += 1
        for b, s in enumerate(self.slots):
            if s is None or self._full(b):
                continue
            sess, st = s
            t = self._pick(st, am[b], lg[b] if lg is not None else None)
            self.stats["batched_tokens"] += 1
            st["out"].append(t)
            stop = sess.on_token(t) if sess.on_token else False
            if stop or self._ended(sess, st):
                self.bt.release(b)
                self.slots[b] = None
                self._finish(sess, st)
            else:
                st["x"] = t
        for b, s in enumerate(self.slots):                   # a full slot: continue on the single model
            if s is not None and self._full(b) and self.solo is None:
                self._resume_solo(b)
                self.stats["outgrown"] = self.stats.get("outgrown", 0) + 1
