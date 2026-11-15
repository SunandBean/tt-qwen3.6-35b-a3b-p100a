# Porting notes: Qwen3.6-35B-A3B on one p100a

## The model and the budget

Qwen3.6-35B-A3B: 40 layers in the pattern `[gated delta net (GDN) x3, gated attention] x 10`, hidden size 2048, an
MoE block in every layer (256 routed experts, top 8, plus a gated shared expert; expert FFN width 512), one MTP
layer, vocabulary 248,320. Apache-2.0.

The p100a has 27.76 GiB of usable DRAM. The base weights are Lottolabs' Tenstorrent-native checkpoint
(`Lottolabs/Qwen3.6-35B-A3B-TT-Mixed-BFP4-BFP8-P150`, Apache-2.0): routed experts in GPTQ BFP4 (18.6 GB incl. the
MTP layer), dense projections and LM head in BFP8, router / norms / conv / embedding in BF16. Everything except the
embedding (1.02 GB BF16, looked up on the host) lives on the card: 20.9 GB, plus the KV cache for 32k tokens and the
GDN state (~0.8 GB), the traces, and the prefix-cache / time-slicing slots. All-BFP8 experts (35 GB) or BFP4 gate/up
with BFP8 down (24 GB) would not fit beside the caches; selected layers in BFP8 do (below).

## What runs on the card

- **Routing**: the router matmul in fp32, then a data-movement kernel picks the top 8 of 256 per row and computes
  the softmax weights in fixed point (the RISC-V cores have no FPU; exp table and polynomial after Lottolabs'
  `routing_math.hpp`); the shared expert is stored as expert 256 with weight `sigmoid(gate)`. One row per core.
- **MoE**: two programs per layer (gate/up + SiLU, then down + weighted sum) that stream only the experts the pass
  routes to, straight from the checkpoint's BFP4 tiles; dual-NoC weight streaming, input row in L1.
- **GDN**: one fused program for the causal conv, one for the chunked delta-rule core (one value head per core), one
  for the commit of an accepted prefix of rows (exact fp32 state).
- **Gated attention**: paged KV cache (bf16), written by a dedicated KV-row kernel; GQA SDPA for decode.
- **MTP**: the model's own MTP layer drafts up to 3 tokens per round; a compact draft head (the first 32,768
  vocabulary rows + the special-token tail) proposes them, the full LM head verifies them, so outputs are those of
  plain decoding (greedy: identical; sampling: exact). The draft count adapts per round to the measured acceptance.
- **Prompts** are read in 28-row passes; every pass is a captured trace.
- **Several users**: up to 2 requests by time slicing (state slots on the card, MTP on), 3–4 by true batched decode
  (per-user GDN state and KV pages, one token per user per pass); a request that outgrows its batch slot moves back
  to the single-sequence path and finishes there.
- **Prefix cache**: the state at 28-row chunk ends is kept in slots on the card, so agent loops that resend their
  history skip the shared prefix. The rest of the prompt is then read in smaller passes than a cold request would
  use, so a greedy answer can differ from the cold one at a near-tie token (seen once in the package smoke test).

## BFP8 upgrade layers

The remaining DRAM is spent on quality. Teacher-forced KL against a float32 run of the BF16 original, 1,532
positions, per choice of expert matrices served in BFP8 (round-to-nearest from BF16) instead of GPTQ BFP4:

| experts in BFP8 | weights on card | extra on card | KL vs BF16 | top-1 |
|---|---:|---:|---:|---:|
| none (Lottolabs GPTQ BFP4) | 20.91 GB | 0 | 0.01734 | 0.9634 |
| down 20-39 + MTP | 23.75 | 2.84 | 0.01364 | 0.9680 |
| gate/up 30-39 + MTP | 23.87 | 2.96 | 0.01314 | 0.9654 |
| **down 30-39 + MTP, gate/up 36-39 + MTP** (default) | 23.75 | 2.84 | **0.01327** | **0.9700** |
| down all | 26.45 | 5.55 | 0.01127 | 0.9706 |

The default needs 272 tensorbins, **6.01 GB on disk** (the full BFP8 matrices; the +2.84 GB is only the extra card
DRAM over the BFP4 copies they replace). The final gate of the default build measured KL 0.0130 / top-1
0.971.

## Long context

`--ctx 131072` fits with the upgrade layers (1.75 GiB of DRAM left after capture without prefix slots; the server
ran with one slot that keeps 32k rows). At 262,144 tokens
the bf16 KV cache needs 2.95 GB more, which fits only without the upgrade layers. A BFP8 KV cache would halve it, but
it is wrong in 28-row prompt passes and is off. A block KV layout with chunked flash prefill is available
(`P100A_Q36_KVBLK=1`, +6 % prompt speed at 110k) but is not part of the gated default.

## Where the time goes

Profile of one 4-row decode pass (default build): GDN layer 0.56 ms, attention layer 0.67 ms, MoE block 0.23 ms, LM
head + argmax 1.98 ms, whole pass 24.5 ms, against a DRAM-bandwidth floor of 9.2 ms. Most of the gap is per-program
launch overhead (~25 programs per layer).

## Checked against

- A float32 CPU reference of the same architecture, checked layer by layer against transformers' `Qwen3_5Moe`
  decoder layers (≤ 3.3e-6), then against the card on the same weights: layer PCC ≥ 0.9974, greedy 16/16 identical,
  teacher-forced KL 0.0024 / top-1 0.984 (554 positions).
- GSM8K first 50 at 2,048 tokens: 49/50. Korean: 4/4 clean. Claude Code over `/v1/messages`: a one-line bug fixed
  in 12 s.

## Knobs

Environment variables named `P100A_Q36_*` (in `tt_qwen36/q36/`) switch individual kernels and layouts; their
defaults are the measured configuration. `Q36_*` select the weight sources (`tt_qwen36/weights.py`).
