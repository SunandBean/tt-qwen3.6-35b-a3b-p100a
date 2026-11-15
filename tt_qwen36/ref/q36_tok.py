"""Qwen3.6 tokenizer: Strata's reference BPE tokenizer (tt_qwen36/ref/third_party, MIT) built from the checkpoint's
tokenizer.json. Its vocabulary and merges are identical to Qwen3.8-Flash-Next's GGUF (checked 2026-11-12: 0 token
and 0 merge differences), so the step-2..6 tokenizer behaviour carries over. Added tokens: `special: true` ->
control (type 3, parsed only with parse_special), the others (`<think>`, `<tool_call>`, ...) -> type 4 (always)."""
from __future__ import annotations

import json
from functools import lru_cache

from .q36_weights import LOTTO
from .third_party.strata_tokenizer import Tokenizer


@lru_cache(maxsize=1)
def Q36Tokenizer() -> Tokenizer:
    tj = json.load(open(LOTTO / "tokenizer.json"))
    voc = dict(tj["model"]["vocab"])
    types = {}
    for a in tj["added_tokens"]:
        voc[a["content"]] = a["id"]
        types[a["id"]] = 3 if a["special"] else 4
    n = max(voc.values()) + 1
    toks = [""] * n
    for t, i in voc.items():
        toks[i] = t
    tt = [types.get(i, 1) for i in range(n)]
    merges = [" ".join(m) if isinstance(m, list) else m for m in tj["model"]["merges"]]
    return Tokenizer(toks, merges, tt, "qwen35", {"tokenizer.ggml.eos_token_id": 248046,
                                                  "tokenizer.ggml.bos_token_id": 248044,
                                                  "tokenizer.ggml.padding_token_id": 248044})
