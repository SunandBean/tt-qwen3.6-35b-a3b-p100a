---
tags:
- blackhole
- p100a
- tenstorrent
- ttnn
- tt-metal
- tt-nn
- text-generation
- moe
- qwen3.6
pipeline_tag: text-generation
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE
base_model:
- Qwen/Qwen3.6-35B-A3B
- Lottolabs/Qwen3.6-35B-A3B-TT-Mixed-BFP4-BFP8-P150
base_model_relation: quantized
---

# qwen3.6-35b-a3b-p100a

Qwen3.6-35B-A3B, served from one Tenstorrent Blackhole p100a with an OpenAI- and Anthropic-compatible API.
Weights: [Lottolabs/Qwen3.6-35B-A3B-TT-Mixed-BFP4-BFP8-P150](https://huggingface.co/Lottolabs/Qwen3.6-35B-A3B-TT-Mixed-BFP4-BFP8-P150) (`c355c41`) + BFP8 upgrade layers in this repo · Port: [SunandBean/tt-qwen3.6-35b-a3b-p100a](https://github.com/SunandBean/tt-qwen3.6-35b-a3b-p100a)

Runs on **p100a** (mesh 1×1). **Every weight is on the card**: all 256 experts of all 40 layers, the routing, the
gated delta net and gated attention layers, the LM head and the MTP draft layer. The host only tokenizes, looks up
the embedding row and runs the HTTP server.

**What this port adds**: a single-card build of the whole model: on-card top-8 routing, fused MoE, gated delta net
and KV-writer kernels, MTP speculative decoding with adaptive draft count, 28-row prompt passes, a prefix cache, true
batched decode for several users, contexts up to 262k tokens, and 6.0 GB of BFP8 upgrade layers that cut the KL to
the BF16 original by a quarter.
**What it builds on**: Lottolabs' Tenstorrent-native GPTQ checkpoint and runtime notes, the ttnn op recipes of
[sjettTT/tt-qwen-3.8-flash-next](https://github.com/sjettTT/tt-qwen-3.8-flash-next), and
[tt-metal](https://github.com/tenstorrent/tt-metal), all Apache-2.0; Strata's reference tokenizer (MIT).

**~82–106 tok/s decode for one user, ~110 tok/s total for four, and 49/50 on GSM8K-50.**

## Quickstart (container)

```bash
tt-model pull  sunandbean/qwen3.6-35b-a3b-p100a --with-weights
tt-model serve sunandbean/qwen3.6-35b-a3b-p100a
```

`pull --with-weights` downloads the image and `checkpoint/` of the Lottolabs repo at `c355c41` (~21 GB, into your HF
cache; nothing is baked into the image). On its first start the server also fetches `bfp8/` from this repo
(6.0 GB, the BFP8 upgrade layers) and compiles its kernels; later starts take about 35 s. The server listens on
port 20000 (or the next free one) and logs `{"event": "ready", ...}` when it is ready.

```bash
curl -s localhost:20000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.6-35b-a3b-p100a", "max_tokens": 300,
  "messages": [{"role": "user", "content": "Write a haiku about a small card running a big model."}]}'

curl -s localhost:20000/v1/messages -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' -d '{
  "model": "qwen3.6-35b-a3b-p100a", "max_tokens": 300,
  "messages": [{"role": "user", "content": "Write a haiku about a small card running a big model."}]}'
```

Profiles (`tt-model serve --profile NAME`):

| profile | context | users at once | BFP8 upgrade layers |
|---|---|---|---|
| `default` | 32,768 tokens | 4 (time slicing with MTP for 1–2, batched decode for 3–4) | on |
| `long-128k` | 131,072 tokens | 1 | on |
| `long-262k` | 262,144 tokens | 1 | **off** (they do not fit beside a 262k KV cache) |

### Use it with Claude Code or other agents

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:20000 ANTHROPIC_AUTH_TOKEN=dummy claude
```

Agents ask for thinking on every turn; start the server with `--thinking off` (append it to the serve arguments) for
much faster agent turns.

## Quickstart (Python)

Prerequisite: a tt-metal / ttnn environment at `58648d4` (ttnn is not on PyPI), e.g. tt-metal's
`tt-metalium-ubuntu-22.04-release-models` container.

```bash
hf download sunandbean/qwen3.6-35b-a3b-p100a --include "code/*" --local-dir qwen3.6-35b-a3b-p100a
cd qwen3.6-35b-a3b-p100a/code && pip install jinja2 safetensors huggingface_hub regex
python -m tt_qwen36.server --port 8000                  # downloads the weights on first start
```

| | |
|---|---|
| **API** | `POST /v1/chat/completions` (OpenAI; tools, `tool_choice`, streaming, `reasoning_content`), `POST /v1/responses`, `POST /v1/messages` (Anthropic; tools, thinking blocks, streaming), `POST /v1/messages/count_tokens`, `GET /v1/models`, `GET /health`. Responses carry llama.cpp-style `timings`. |
| **Sampling** | `temperature` (default 1.0, as in the OpenAI API; 0 = greedy), `top_p`, `top_k`, `seed`, `stop` / `stop_sequences`. Sampling stays exact with speculative decoding. |
| **Context** | `--ctx 32768` (default), `131072`, or `262144` (with `Q36_BFP8_SOURCE=off --batch 0 --prefix-slots 0`). |
| **Thinking** | OpenAI: on unless `chat_template_kwargs.enable_thinking` is false or `reasoning_effort` is `"none"`; Anthropic: on when `thinking` is enabled. `--thinking off` disables it for every request. |
| **Weights** | `Lottolabs/Qwen3.6-35B-A3B-TT-Mixed-BFP4-BFP8-P150` pinned to `c355c41d4fd079c6df3c4ba9bab1c40f37c17f09` (`Q36_LOTTO` = a local `checkpoint/`). |
| **BFP8 upgrade** | `Q36_BFP8_SOURCE=hf` (default: `bfp8/` of this repo), `dir` (`Q36_BFP8_DIR`), `bf16` (convert locally from the Qwen BF16 original), `off`. Every file is checked against `tt_qwen36/bfp8_sha256.txt`. |

## Performance

Measured 2026-11-13 on an idle p100a, through the HTTP server, default configuration (BFP8 upgrade layers on, MTP
up to 3 drafts with an adaptive draft count, greedy).

| | chat | code | doc |
|---|---:|---:|---:|
| **Decode, one user (tok/s)** | **81.9** | **105.9** | **88.7** |
| MTP draft acceptance | 0.51 | 0.73 | 0.57 |
| Prompt reading, 40–60-token prompts (tok/s) | 566 | 500 | 590 |

Several users at once (hybrid scheduler: time slicing with MTP for 1–2 requests, batched decode for 3–4):

| users | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| **total tok/s** | 106.7 | 95.2 | 91.2 | **112.4** |
| per user | 112 | 49–69 | 34–37 | 31–33 |

Long context (needle-in-a-haystack, passkey at half depth, greedy):

| context | prompt tokens | found | prompt read in |
|---|---:|:---:|---:|
| `--ctx 131072` | 14,489 / 59,686 / 110,051 | yes / yes / yes | 38 / 250 / 655 s |
| `--ctx 262144` (BFP8 upgrade off) | 184,161 | yes | 1,497 s (123 tok/s) |

Decode does not slow down with the context size of the server (fib prompt: 105–107 tok/s at 128k, 117.8 tok/s at
262k without the upgrade layers).

| | |
|---|---:|
| Weights on the card | 23.75 GB (20.91 GB without the upgrade layers) |
| Server start, weights and kernel cache warm | ~35 s |
| Claude Code (`/v1/messages`), fix a one-line bug in a small repo | 12 s end to end |

## Accuracy

Teacher-forced on 1,532 positions (chat, code and docs answers, Korean prompts and GSM8K answers), against a float32
CPU run of the **original BF16** Qwen3.6-35B-A3B:

| weights on the card | KL vs BF16 | top-1 agreement |
|---|---:|---:|
| Lottolabs GPTQ BFP4 experts only (`Q36_BFP8_SOURCE=off`) | 0.0173 | 0.963 |
| **+ BFP8 upgrade layers (default)** | **0.0130** | **0.971** |

Port fidelity (same weights, float32 CPU reference of this port vs the card, 554 positions): KL 0.0024, top-1 0.984;
greedy output identical for 16/16 tokens on a fixed prompt.

| | |
|---|---:|
| GSM8K, first 50, max 2,048 tokens | **49/50** (180 s) |
| Korean, 4 prompts | 4/4 clean (no Chinese characters) |

The one GSM8K miss is #12 (the lemon-tree question): the model answers 12, the break-even year.

### Why the upgrade layers

Lottolabs quantized the experts with GPTQ to BFP4 so the model fits one P150. The p100a's DRAM has room for a little
more than that 20.9 GB checkpoint beside the traces and caches, so the experts of the layers where BFP4 hurts most are served in BFP8 instead, converted
round-to-nearest from the BF16 original: the down projections of layers 30–39 and the MTP layer, and the gate/up
projections of layers 36–39 and the MTP layer (257 experts each, incl. the shared expert). That is 272 files,
6.0 GB on disk, +2.84 GB on the card, and a quarter less KL. `tools/convert_bfp8.py` rebuilds them from the BF16
weights byte for byte.

## Caveats

- **Single card only.** Mesh 1×1, no multi-chip scaling.
- **262k context needs the upgrade layers off** (`long-262k` profile). A BFP8 KV cache, which would make room, is
  not supported (it is wrong in 28-row prompt passes).
- **Long prompts read slowly**: 590 tok/s at 50 tokens, ~240 tok/s at 60k, ~170 tok/s at 110k, 123 tok/s at 184k.
- **First start downloads 6.0 GB more** (the upgrade layers) inside the container, without a progress bar in
  `tt-model serve`; `tt-model logs -f` shows it.
- Agents (Claude Code) ask for thinking on every turn; use `--thinking off` for them.
- **Greedy is not bit-reproducible across the prefix cache.** A request that resumes from a cached prefix reads the
  rest of its prompt in smaller passes than a cold one, so a near-tie token can come out differently (seen once in
  testing: the same wording up to the second sentence, then a different verb).

## Licensing

- Model: [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), Apache-2.0.
- Base weights: [Lottolabs/Qwen3.6-35B-A3B-TT-Mixed-BFP4-BFP8-P150](https://huggingface.co/Lottolabs/Qwen3.6-35B-A3B-TT-Mixed-BFP4-BFP8-P150), Apache-2.0, revision `c355c41d4fd079c6df3c4ba9bab1c40f37c17f09`. Not redistributed here.
- `bfp8/`: derived from the BF16 weights of Qwen/Qwen3.6-35B-A3B at revision `995ad96eacd98c81ed38be0c5b274b04031597b0`, Apache-2.0 (see `NOTICE`).
- Port code (`code/`): Apache-2.0, from [SunandBean/tt-qwen3.6-35b-a3b-p100a](https://github.com/SunandBean/tt-qwen3.6-35b-a3b-p100a).
- `code/tt_qwen36/ref/third_party/strata_tokenizer.py`: from Strata, MIT.
- `code/models/common`: from [tt-metal](https://github.com/tenstorrent/tt-metal) `58648d4`, Apache-2.0.
- `NOTICE` and `code/THIRD-PARTY.md` list every third-party source with its licence.
