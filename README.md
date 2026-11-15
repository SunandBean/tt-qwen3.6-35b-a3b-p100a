# tt-qwen3.6-35b-a3b-p100a

[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) on one Tenstorrent Blackhole **p100a**, every weight on
the card, behind an OpenAI- and Anthropic-compatible server. Model card, numbers and the container package:
[sunandbean/qwen3.6-35b-a3b-p100a](https://huggingface.co/sunandbean/qwen3.6-35b-a3b-p100a).

- **~82–106 tok/s** decode for one user (MTP speculative decoding), **~110 tok/s total** for four users.
- KL 0.0130 / top-1 0.971 against the BF16 original; GSM8K-50 49/50.
- Context up to 262,144 tokens; prefix cache; tool calls; streaming; thinking on or off.
- Works as a backend for Claude Code (`/v1/messages`) and OpenAI clients (`/v1/chat/completions`, `/v1/responses`).

## Run it

With [tt-model](https://github.com/tenstorrent/tt-model-manager) and Docker:

```bash
tt-model pull  sunandbean/qwen3.6-35b-a3b-p100a --with-weights
tt-model serve sunandbean/qwen3.6-35b-a3b-p100a
```

Or in a tt-metal `58648d4` environment (e.g. the `tt-metalium-ubuntu-22.04-release-models` container) with a p100a:

```bash
pip install -e .                       # jinja2, safetensors, huggingface_hub, regex
python -m tt_qwen36.server --port 8000
python examples/openai_chat.py http://127.0.0.1:8000
python examples/anthropic_messages.py http://127.0.0.1:8000
```

The first start downloads the Lottolabs checkpoint (`checkpoint/`, ~21 GB) and the BFP8 upgrade layers (6.0 GB) into
the Hugging Face cache. Options: `python -m tt_qwen36.server --help`; weight sources: `tt_qwen36/weights.py`.

| setting | how |
|---|---|
| context | `--ctx 32768` (default), `--ctx 131072 --batch 0 --prefix-slots 1 --max-concurrent 1`, or `--ctx 262144 --batch 0 --prefix-slots 0 --max-concurrent 1` with `Q36_BFP8_SOURCE=off` |
| BFP8 upgrade layers | `Q36_BFP8_SOURCE=hf` (default) / `dir` + `Q36_BFP8_DIR` / `bf16` (convert locally) / `off` |
| no thinking (agents) | `--thinking off` |
| several users | `--batch 4` (default: batched decode for 3–4 requests), `--batch 0` (time slicing only) |
| local checkpoint | `Q36_LOTTO=/path/to/checkpoint` |

## Layout

| path | what |
|---|---|
| `tt_qwen36/server.py` | the server (`python -m tt_qwen36.server`) |
| `tt_qwen36/api.py`, `chat.py` | HTTP layer: OpenAI / Anthropic endpoints, chat template, tool-call parsing (CPU only; `python -m tt_qwen36.api` is a scripted test server without a card) |
| `tt_qwen36/q36/` | the model on the card: weights, passes, MTP speculative decoding, batched decode, MoE / GDN / KV kernels (`kernels/`) |
| `tt_qwen36/kgdn.py`, `kkv.py`, `prefix.py`, `sessions.py` | shared GDN / KV-writer kernels, prefix cache, time slicing |
| `tt_qwen36/ref/` | host-side weight reader and tokenizer |
| `tt_qwen36/weights.py`, `bfp8_sha256.txt` | pinned weight sources and the checksums of the BFP8 upgrade layers |
| `tools/convert_bfp8.py` | rebuilds the BFP8 upgrade layers from the BF16 original (CPU) |
| `tt-model.yaml` | the tt-model container package |
| `PORTING_NOTES.md` | how the port works and what was measured |

## License

Apache-2.0 (`LICENSE`, `NOTICE`). Third-party code and its licences: `THIRD-PARTY.md`. The weights are downloaded
from their publishers (Apache-2.0) and are not part of this repository.
