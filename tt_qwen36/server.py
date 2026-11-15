"""OpenAI- and Anthropic-compatible server for Qwen3.6-35B-A3B on one Tenstorrent Blackhole p100a.

    python -m tt_qwen36.server [--host 127.0.0.1] [--port 8000] [--ctx 32768] [--thinking request|off]

Endpoints (tt_qwen36/api.py): POST /v1/chat/completions (OpenAI, tools, streaming), POST /v1/responses,
POST /v1/messages (Anthropic, tools, thinking, streaming), POST /v1/messages/count_tokens, GET /v1/models, GET /health.
Model id: qwen3.6-35b-a3b-p100a (--model-name).

Every weight is on the card: Lottolabs' GPTQ BFP4 experts + BFP8 projections, plus the BFP8 upgrade layers
(tt_qwen36/weights.py; Q36_BFP8_SOURCE=off serves without them). MTP speculative decoding (up to 3 drafts, adaptive),
28-row prompt passes, prefix cache, and several requests at once: time slicing with MTP for 1-2 requests, true
batched decode for 3-4 (--batch 0: time slicing only). Startup ~35 s with the weights cached (plus the one-time
kernel compile on a fresh kernel cache). `timings` in responses follow llama.cpp's names (prompt_n, prompt_ms,
prompt_per_second, predicted_n, predicted_ms, predicted_per_second, draft_n, draft_n_accepted).
"""
from __future__ import annotations

import tt_qwen36.wallclock  # noqa: F401  (self-exit on a hang; P100A_WALL, 0 = off)

import argparse
import json
import os
import queue
import sys
import threading
import time
import types
from http.server import ThreadingHTTPServer

import ttnn

from .api import make_handler
from .chat import Template

EOS = {248046, 248044}
MODEL_NAME = "qwen3.6-35b-a3b-p100a"


class Engine:
    """The card behind tt_qwen36.api (see its docstring for the generate() event protocol)."""

    def __init__(self, model, sched, name: str, tok, template):
        self.m = model
        self.sched = sched
        self.tok = tok
        self.template = template
        self.name = name
        self.lock = threading.Lock()
        self.ctx_limit = model.s_max - model.T - 1
        self.think_mode = "request"

    def busy(self) -> bool:
        return self.sched.busy()

    def generate(self, ids, max_tokens: int, sampling=None):
        """Speculative generation through the scheduler, tokens streamed; sampling = {temperature, top_p, top_k, seed,
        stop}; temperature 0 = greedy. If the client goes away the request is cancelled."""
        from .sessions import Session
        sampling = sampling or {}
        stops = sampling.get("stop") or []
        stops = [stops] if isinstance(stops, str) else [x for x in stops if x]
        m = self.m
        if len(ids) + max_tokens + m.T > m.s_max:
            max_tokens = max(1, m.s_max - len(ids) - m.T)
        q: queue.Queue = queue.Queue()
        halt = threading.Event()

        def on_token(t):
            q.put(t)
            return halt.is_set()

        kw = dict(eos=EOS, spec=m.K > 0, temperature=float(sampling.get("temperature", 0.0) or 0.0),
                  top_p=float(sampling.get("top_p", 1.0) or 1.0), top_k=int(sampling.get("top_k", 0) or 0),
                  seed=sampling.get("seed"))
        cached = m.cached_prefix(ids)
        yield {"type": "start", "prompt_tokens": len(ids), "cached_tokens": cached}
        sess = Session(ids, max_tokens, kw, on_token)
        th = types.SimpleNamespace(join=sess.done.wait)
        threading.Thread(target=lambda: (sess.done.wait(), q.put(None)), daemon=True).start()
        self.sched.submit(sess)
        try:
            gen, sent, hit, hit_s = [], "", None, None
            while True:
                try:
                    t = q.get(timeout=10)
                except queue.Empty:
                    yield {"type": "ping"}
                    continue
                if t is None:
                    break
                if t in EOS or hit is not None:
                    continue
                gen.append(t)
                text = self.tok.decode(gen)
                for st in stops:
                    j = text.find(st, max(0, len(sent) - len(st)))
                    if j >= 0 and (hit is None or j < hit):
                        hit, hit_s = j, st
                if hit is not None:
                    halt.set()
                    if hit > len(sent):
                        yield text[len(sent):hit]
                    sent = text[:hit]
                    continue
                hold = max((len(st) - 1 for st in stops), default=0)   # don't stream a possible stop prefix yet
                safe = len(text) - hold
                if not text.endswith("�") and safe > len(sent):
                    yield text[len(sent):safe]
                    sent = text[:safe]
        finally:
            halt.set()
            sess.cancelled = True           # no-op when it already finished; else the scheduler drops it
            th.join()
        if sess.error is not None:
            raise sess.error
        toks, info = sess.result["out"]
        text = self.tok.decode(gen)
        if hit is None and len(text) > len(sent):
            yield text[len(sent):]
        finish = "stop" if (hit is not None or (toks and toks[-1] in EOS)) else "length"
        pm, dm, steps = info["prompt_s"] * 1e3, info["decode_s"] * 1e3, max(0, len(toks) - 1)
        acc = info["accepted"]
        cached = info.get("cached", 0)
        yield {"type": "end", "finish_reason": finish, "stop_hit": hit_s, "cached_tokens": cached,
               "usage": {"prompt_tokens": len(ids), "completion_tokens": len(gen), "total_tokens": len(ids) + len(gen),
                         "prompt_tokens_details": {"cached_tokens": cached}},
               "timings": {"prompt_n": len(ids) - cached, "cache_n": cached, "prompt_ms": round(pm, 1),
                           "prompt_per_second": round((len(ids) - cached) / max(pm / 1e3, 1e-9), 3),
                           "predicted_n": steps, "predicted_ms": round(dm, 1),
                           "predicted_per_second": round(steps / (dm / 1e3), 3) if steps else None,
                           "draft_n": sum(info["proposed"]), "draft_n_accepted": sum(acc)}}


def idle_watch(engine, srv, idle_min: float):
    """--idle-exit: stop the server (and free the card) after this many idle minutes"""
    last = time.time()
    while True:
        time.sleep(5)
        if engine.busy():
            last = time.time()
            continue
        if time.time() - last > idle_min * 60:
            print(f"idle for {idle_min:g} min: exiting", flush=True)
            srv.shutdown()
            return


def build(dev, a):
    from .q36.model import Q36Weights
    from .q36.spec import Q36Spec
    from .ref.q36_tok import Q36Tokenizer
    from .ref.q36_weights import LOTTO
    from .sessions import Scheduler
    t0 = time.time()
    W = Q36Weights(dev, mtp=a.spec > 0)
    model = Q36Spec(dev, W, drafts=a.spec, prefix_slots=a.prefix_slots, s_max=a.ctx)
    if a.batch > 1:                                  # true batched decode for 3..8 requests at once
        from .q36.batch import Q36Batch
        from .q36.bsched import BatchScheduler
        bt = Q36Batch(model, B=a.batch, s_user=a.batch_ctx)
        model.warm()
        model.pre.warm()
        bt.warm(model)
        model.capture()
        model.pre.capture()
        bt.capture()
        model.reset()
        sched = BatchScheduler(model, bt, eos=EOS)
    else:
        sched = Scheduler(model, a.max_concurrent, a.quantum)
        model.capture_all()
    print(f"model ready in {time.time() - t0:.0f} s", flush=True)
    engine = Engine(model, sched, a.model_name, Q36Tokenizer(), Template((LOTTO / "chat_template.jinja").read_text()))
    engine.think_mode = a.thinking
    return engine


def writable_cwd():
    """tt-metal writes run files (generated/watcher, generated/fabric) under the current directory and fails on the
    first device op if it is read-only (e.g. a read-only code mount, or an image WORKDIR the --user cannot write):
    move to a writable directory first (Q36_WORKDIR, TT_METAL_CACHE, HOME, or the temp dir)."""
    import tempfile
    if os.access(os.getcwd(), os.W_OK):
        return
    for d in (os.environ.get("Q36_WORKDIR"), os.environ.get("TT_METAL_CACHE"), os.environ.get("HOME"),
              tempfile.gettempdir()):
        if d and os.path.isdir(d) and os.access(d, os.W_OK):
            print(f"working directory {os.getcwd()} is read-only: using {d}", flush=True)
            os.chdir(d)
            return


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 inside a container)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model-name", default=MODEL_NAME, help="id reported by /v1/models")
    ap.add_argument("--ctx", type=int, default=32768,
                    help="context length (KV cache rows, multiple of 32): 32768 (default), 131072, or 262144 "
                         "(262144 only with Q36_BFP8_SOURCE=off, --batch 0 and --prefix-slots 0)")
    ap.add_argument("--spec", type=int, default=3, help="MTP drafts per round (0 = plain decode)")
    ap.add_argument("--thinking", default="request", choices=["request", "off"],
                    help="off: never think, whatever the request asks (agents: much faster turns)")
    ap.add_argument("--prefix-slots", type=int, default=2, help="prefix cache: state snapshots kept on the card")
    ap.add_argument("--max-concurrent", type=int, default=2,
                    help="time-sliced requests at once (state slots on the card); more wait in line")
    ap.add_argument("--quantum", type=float, default=0.5, help="seconds a request runs before the next one's turn")
    ap.add_argument("--batch", type=int, default=4,
                    help="true batched decode for 3..N requests at once (0 = time slicing only)")
    ap.add_argument("--batch-ctx", type=int, default=16384, help="context per batched request")
    ap.add_argument("--idle-exit", type=float, default=0.0, help="exit after this many idle minutes (0 = never)")
    ap.add_argument("--log-requests", default="", help="write every request (headers + body) to this directory")
    a = ap.parse_args(argv)
    if os.environ.get("P100A_WALL") is None:            # an interactive server runs until stopped
        import faulthandler
        faulthandler.cancel_dump_traceback_later()
    import signal

    def _term(*_):                                      # docker stop / tt-model stop: close the device cleanly
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    writable_cwd()
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=600 * 1024 * 1024)
    try:
        engine = build(dev, a)
        srv = ThreadingHTTPServer((a.host, a.port), make_handler(engine, a.log_requests))
        if a.idle_exit:
            threading.Thread(target=idle_watch, args=(engine, srv, a.idle_exit), daemon=True).start()
        print(json.dumps({"event": "ready", "model": a.model_name, "host": a.host, "port": a.port}), flush=True)
        print(f"serving on http://{a.host}:{a.port}/v1", flush=True)
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ttnn.close_device(dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
